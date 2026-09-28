// Regional-model worker: builds an experiment, steps it, and streams 3-D cloud / rain volumes.

import { RegionalModel } from '../../regional/core.js';
import { weismanKlemp } from '../../regional/kessler.js';
import { IceMicrophysics, QV, QC, QR, QI, QS, QG } from '../../regional/ice.js';
import { RegionalPhysics } from '../../regional/physics.js';
import { tropicalSounding, insertVortex, tcMetrics, eyewallProfile } from '../../regional/tropical.js';
import { GpuRegional } from '../../gpu/regionalGpu.js';
import type { RegionalPhysicsConfig } from '../../regional/physics.js';
import { nestFromGlobal, nestTargets, sampleSurface, NestSpec } from '../../regional/nest.js';
import { refineInto } from '../../regional/refine.js';
import { packSave, unpackSave, type SaveArrays } from '../saves.js';
import { Pacer } from '../pacer.js';
import { tornadoExperiment, StormTracker } from '../../regional/supercell.js';
import { REFINE_TO, type FromRegionalWorker, type GroundField, type NestPayload, type NestSize, type RegionalExperiment, type ToRegionalWorker } from './protocol.js';

let m: RegionalModel | null = null;
let mp: IceMicrophysics | null = null;
let experiment: RegionalExperiment = 'supercell';
let running = false, stepsPerTick = 1, ground: GroundField = 'rain';
let lastFrame = 0, rateSteps = 0, rateT = performance.now(), rate = 0;
let dpEnv = 0;
let gpu: GpuRegional | null = null;
let gpuDevice: GPUDevice | null = null;
let busy = false;
let gpuBatch = 2;
const DEBUG = false;
let physCfg: RegionalPhysicsConfig | null = null;
let frameVel = { u: 0, v: 0 };
let tracker: StormTracker | null = null, lastTrack = 0;
let nestSpec: NestSpec | null = null;
/** how the current model was built (needed to rebuild it from a save) */
let builtGpu = false;
// pacing and display cadence
const pacer = new Pacer();
let frameMode: { kind: 'wall' | 'model' | 'fast'; every: number } = { kind: 'wall', every: 0 };
let lastFrameModel = 0;
const modelNow = (): number => (gpu ? gpu.time : m ? m.time : 0);
const dtNow = (): number => (gpu ? gpu.dt : m ? m.c.dt : 1);
function frameDue(): boolean {
  const el = performance.now() - lastFrame;
  if (frameMode.kind === 'fast') return el > 5000;
  if (frameMode.kind === 'model') return el > 250 && modelNow() - lastFrameModel >= frameMode.every - 1e-6;
  return el > (gpu ? 500 : 300);
}
let nestSource: { payload: NestPayload; lat0: number; lon0: number; size: NestSize } | null = null;
const EXP_LABEL: Record<RegionalExperiment, string> = {
  supercell: '超大胞 2 km / supercell 2 km', supercell_hr: '超大胞 1 km / supercell 1 km', tc: '熱帶氣旋 15 km / tropical cyclone 15 km',
  tc_hr: '熱帶氣旋 5 km / tropical cyclone 5 km', tornado: '龍捲超大胞 250 m / tornadic supercell 250 m', tornado_c: '龍捲超大胞 1 km / tornadic supercell 1 km', nest: '巢狀區域 / nest',
};

// Adaptive time step (GPU): dt = min(acoustic limit, CFL_TARGET / max(|u|/dx + |v|/dy + |w|/dz)),
// between the configured dt0 and 3 dt0; shrinks at once, grows by at most 10 % per check. The
// acoustic limit keeps the horizontal sound Courant number of the split steps c_s dt / (n_s dx) <= 0.45.
const CFL_TARGET = 0.8, CFL_MAX = 1.1;
let adaptive = true, dt0 = 0;
function dtLimits(): { lo: number; hi: number } {
  const c = m!.c;
  const ac = c.nsound * 0.45 * Math.min(c.dx, c.dy) / 350;
  return { lo: 0.25 * dt0, hi: Math.max(dt0, Math.min(3 * dt0, ac)) };
}
async function adaptDt(): Promise<void> {
  if (!gpu || !adaptive || !m) return;
  const rate = await gpu.maxCourantRate();
  if (!Number.isFinite(rate)) return;
  const { lo, hi } = dtLimits(), cur = gpu.dt;
  let next = Math.min(hi, CFL_TARGET / Math.max(rate, 1e-9));
  if (rate * cur > CFL_MAX) next = Math.min(next, 0.7 / rate);          // overshoot: cut back hard
  else if (next > cur) next = Math.min(next, 1.1 * cur);
  next = Math.max(lo, next);
  if (Math.abs(next - cur) > 0.02 * cur) gpu.setDt(next);
}

async function getGpu(): Promise<GPUDevice | null> {
  if (gpuDevice) return gpuDevice;
  const nav = (self as unknown as { navigator: Navigator }).navigator;
  if (!nav.gpu) return null;
  const ad = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!ad) return null;
  gpuDevice = await ad.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize, maxBufferSize: ad.limits.maxBufferSize },
    requiredFeatures: ad.features.has('timestamp-query') ? ['timestamp-query'] : [],
  });
  const dev = gpuDevice;
  dev.lost.then((info) => {
    if (gpuDevice === dev) gpuDevice = null;
    running = false;
    post({ type: 'error', message: `GPU 裝置遺失（可能是顯示卡逾時被系統重設），請重新整理頁面 / GPU device lost (probably reset by the OS watchdog); please reload the page: ${info.message}` });
  });
  return gpuDevice;
}

const gpuFailNote = (r: string): string => `WebGPU 在這個裝置上失敗，改用 CPU（較粗網格）/ WebGPU failed on this device, using the CPU (coarser grid): ${r.slice(0, 300)}`;

/** Build the GPU model for the current CPU model and trial-run one step; on any GPU error or an
 *  implausible state leave gpu = null and return the reason. */
async function tryGpu(device: GPUDevice): Promise<string> {
  const scopes = ['validation', 'internal', 'out-of-memory'] as const;
  for (const sc of scopes) device.pushErrorScope(sc);
  let reason = '';
  try {
    const g = new GpuRegional(device, m!, { moist: true, physics: physCfg, ice: true });
    g.uploadFrom(m!);
    g.step(1);
    const st = await g.readState(), size = m!.size;
    let bad = false;
    for (let i = 3 * size; i < 4 * size; i += 13) if (!Number.isFinite(st[i]!)) { bad = true; break; }
    if (bad) reason = 'GPU 試跑結果出現非數值 / GPU trial produced non-finite values';
    gpu = g;
  } catch (e) { reason = String(e); }
  for (let i = 0; i < scopes.length; i++) { const err = await device.popErrorScope(); if (err && !reason) reason = err.message; }
  if (reason) { gpu?.destroy(); gpu = null; }
  return reason;
}

const post = (msg: FromRegionalWorker, tr: Transferable[] = []): void => (self as unknown as Worker).postMessage(msg, tr);

function build(exp: RegionalExperiment, gpuOk: boolean): { dt: number; description: string } {
  experiment = exp;
  physCfg = null;
  frameVel = { u: 0, v: 0 };
  tracker = null;
  if (exp === 'tornado' || exp === 'tornado_c') {
    // GPU: 250 m LES over 50 km; CPU: a 500 m preview over 40 km (mesocyclone scale only);
    // tornado_c: 1 km spin-up over 100 km (same environment), refined to the 250 m box later
    const e = exp === 'tornado_c' ? tornadoExperiment(1000, 100000, 40, 400, 6, 6)
      : gpuOk ? tornadoExperiment(250, 50000, 64, 250, 2, 8) : tornadoExperiment(500, 40000, 40, 400, 3, 6);
    m = e.model; mp = new IceMicrophysics(m); physCfg = e.physics; frameVel = e.frame;
    tracker = new StormTracker(); lastTrack = 0;
    new RegionalPhysics(m, physCfg);
    return { dt: m.c.dt, description: e.description };
  }
  if (exp === 'supercell' || exp === 'supercell_hr') {
    const hr = exp === 'supercell_hr';
    const L = 120000, dx = hr ? 1000 : 2000, nx = L / dx, nz = hr ? 60 : 40, dz = hr ? 333.3333 : 500, dtm = hr ? 3 : 6;
    m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt: dtm, nsound: 6, f: 0, beta: 0.2, divDamp: 0.1, dampDepth: 6000, dampRate: 1 / 300, kdiff2: 0 }, weismanKlemp, 6);
    mp = new IceMicrophysics(m);
    m.setBaseWind((z) => ({ u: 30 * Math.tanh(z / 3000) - 15, v: 0 }));
    for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) m.scalars[QV]![m.idx(i, j, k)] = m.qv0[k]!;
    const xc = L * 0.35, yc = L / 2;
    for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
      const r = Math.sqrt((((i + 0.5) * dx - xc) / 10000) ** 2 + (((j + 0.5) * dx - yc) / 10000) ** 2 + ((m.zc[k]! - 1400) / 1400) ** 2);
      if (r < 1) m.th[m.idx(i, j, k)] = m.th[m.idx(i, j, k)]! + 2 * Math.cos(0.5 * Math.PI * r) ** 2;
    }
    return { dt: dtm, description: `Weisman–Klemp (1982) 超大胞 / supercell（Δx ${dx / 1000} km）：暖泡在 30 m/s 低層垂直風切中觸發 / warm bubble in 30 m/s low-level shear` };
  } else {
    const hr = exp === 'tc_hr';
    const L = 1200000, dx = hr ? 5000 : 15000, nx = L / dx, nz = hr ? 50 : 25, dz = hr ? 500 : 1000, f = 5e-5, sst = 301.15, dtm = hr ? 30 : 60;
    m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt: dtm, nsound: 6, f, beta: 0.3, divDamp: 0.1, dampDepth: 6000, dampRate: 1 / 300, kdiff2: 0 }, tropicalSounding(sst), 6);
    mp = new IceMicrophysics(m);
    physCfg = { lh: hr ? 1000 : 0.2 * dx, lv: 100, sst, ck: 1.2e-3, radTau: 12 * 3600, radMax: 2 / 86400 };
    new RegionalPhysics(m, physCfg);
    insertVortex(m, f, 15);
    dpEnv = 1e5 * Math.pow(m.pi0[0]!, 1004.5 / 287.05) / 100;
    return { dt: dtm, description: `熱帶氣旋 / tropical cyclone：28°C 海面上的弱渦旋（f 平面，Δx ${dx / 1000} km）/ weak vortex over a 28 °C sea (f-plane)` };
  }
}

/** One-way nest inside the global model at (lat0, lon0): initial and lateral-boundary fields from the
 *  global snapshot, surface skin temperature and wetness from the global surface model. */
function buildNest(g: NestPayload, lat0: number, lon0: number, size: NestSize, gpuOk: boolean): { dt: number; description: string; land: Uint8Array | null } {
  experiment = 'nest';
  const spec: NestSpec = size === 'cp3'
    ? (gpuOk ? { lat0, lon0, L: 960000, dx: 3000, nz: 40, dz: 450, dt: 15, nsound: 6 } : { lat0, lon0, L: 960000, dx: 12000, nz: 30, dz: 600, dt: 60, nsound: 6 })
    : size === 'storm'
    ? (gpuOk ? { lat0, lon0, L: 480000, dx: 4000, nz: 40, dz: 450, dt: 20, nsound: 6 } : { lat0, lon0, L: 480000, dx: 8000, nz: 30, dz: 600, dt: 40, nsound: 6 })
    : (gpuOk ? { lat0, lon0, L: 1200000, dx: 12000, nz: 30, dz: 600, dt: 60, nsound: 6 } : { lat0, lon0, L: 1200000, dx: 20000, nz: 24, dz: 750, dt: 60, nsound: 6 });
  const nest = nestFromGlobal(g, spec, 6);
  nestSpec = spec;
  m = nest.model;
  mp = new IceMicrophysics(m);
  const surface = g.ts && g.wet ? { tsk: sampleSurface(g, spec, g.ts), wet: sampleSurface(g, spec, g.wet) } : null;
  physCfg = { lh: 0.2 * spec.dx, lv: 100, sst: 0, ck: 1.2e-3, radTau: 0, radMax: 0, surface };
  new RegionalPhysics(m, physCfg);
  let land: Uint8Array | null = null;
  if (g.land) { const lf = sampleSurface(g, spec, g.land); land = Uint8Array.from(lf, (x) => (x > 0.5 ? 1 : 0)); }
  const d = (x: number): string => (x * 180 / Math.PI).toFixed(1);
  const lonE = lon0 * 180 / Math.PI;
  return {
    dt: spec.dt, land,
    description: `全球模式巢狀區域 / Nest in the global model (${g.preset}, day ${g.day.toFixed(1)})：${d(lat0)}°${lat0 >= 0 ? 'N' : 'S'}, ${(lonE > 180 ? 360 - lonE : lonE).toFixed(1)}°${lonE > 180 ? 'W' : 'E'}，${spec.L / 1000} km 見方，Δx ${spec.dx / 1000} km；` +
      `側邊界向全球場鬆弛（單向巢狀）${surface ? '，地表溫度與土壤濕度取自全球模式' : '，無地表通量（全球實驗無地表模式）'} / lateral boundaries relax to the global fields (one-way)${surface ? ', surface temperature and wetness from the global model' : ', no surface fluxes (no surface model in this global experiment)'}`,
  };
}

self.onmessage = async (ev: MessageEvent<ToRegionalWorker>): Promise<void> => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      running = false;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      const device = msg.backend === 'cpu' ? null : await getGpu();
      gpu?.destroy(); gpu = null;
      let info = build(msg.experiment, !!device);
      builtGpu = !!device; nestSource = null;
      let note = '';
      if (device) {
        const reason = await tryGpu(device);
        if (reason) { note = gpuFailNote(reason); info = build(msg.experiment, false); builtGpu = false; }
      } else if (msg.backend !== 'cpu') note = 'WebGPU 不可用，改用 CPU / WebGPU unavailable, using CPU';
      if (!gpu && (msg.experiment === 'tc_hr' || msg.experiment === 'supercell_hr' || msg.experiment === 'tornado')) note += (note ? ' · ' : '') + '此高解析實驗在 CPU 上非常慢 / this high-resolution experiment is very slow on the CPU';
      const c = m!.c;
      dt0 = c.dt;
      post({ type: 'ready', land: null, experiment: msg.experiment, nx: c.nx, ny: c.ny, nz: c.nz, dx: c.dx, dz: c.dz, dt: info.dt, description: info.description, backend: gpu ? 'gpu' : 'cpu', note, refineTo: REFINE_TO[msg.experiment] ?? null });
      await sendFrame();
    }
    else if (msg.type === 'initNest') {
      running = false;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      const device = msg.backend === 'cpu' ? null : await getGpu();
      gpu?.destroy(); gpu = null;
      let info = buildNest(msg.payload, msg.lat0, msg.lon0, msg.size, !!device);
      builtGpu = !!device; nestSource = { payload: msg.payload, lat0: msg.lat0, lon0: msg.lon0, size: msg.size };
      let note = '';
      if (device) {
        const reason = await tryGpu(device);
        if (reason) { note = gpuFailNote(reason); info = buildNest(msg.payload, msg.lat0, msg.lon0, msg.size, false); builtGpu = false; }
      } else note = (msg.backend !== 'cpu' ? 'WebGPU 不可用，改用 CPU / WebGPU unavailable, using CPU · ' : '') + 'CPU 使用較粗的網格 / the CPU uses a coarser grid';
      const c = m!.c;
      dt0 = c.dt;
      post({ type: 'ready', experiment: 'nest', nx: c.nx, ny: c.ny, nz: c.nz, dx: c.dx, dz: c.dz, dt: info.dt, description: info.description, backend: gpu ? 'gpu' : 'cpu', note, land: info.land, refineTo: null });
      await sendFrame();
    }
    else if (msg.type === 'run') { running = msg.running; if (running) pacer.reset(modelNow()); }
    else if (msg.type === 'pace') { pacer.target = Math.max(0, msg.target); pacer.reset(modelNow()); }
    else if (msg.type === 'runUntil') { pacer.until = msg.hours > 0 ? modelNow() + msg.hours * 3600 : null; }
    else if (msg.type === 'frames') { frameMode = { kind: msg.kind, every: msg.every ?? 0 }; }
    else if (msg.type === 'step1') {
      if (running || !m || !mp) return;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      busy = true;
      try {
        if (gpu) { gpu.step(1); await gpu.device.queue.onSubmittedWorkDone(); await adaptDt(); }
        else { m.step(); mp.apply(m.c.dt); }
        busy = false;
        await sendFrame();
      } catch (e) { post({ type: 'error', message: String(e) }); }
      busy = false;
    }
    else if (msg.type === 'speed') stepsPerTick = Math.max(1, msg.stepsPerTick | 0);
    else if (msg.type === 'ground') { ground = msg.field; sendFrame(); }
    else if (msg.type === 'save') {
      if (!m || !mp) return;
      const wasRunning = running; running = false;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      busy = true;
      try {
        await syncFromGpu();
        const c = m.c, f32 = (a: ArrayLike<number>): Float32Array => Float32Array.from(a);
        const arrays: SaveArrays = { u: f32(m.u), v: f32(m.v), w: f32(m.w), th: f32(m.th), pp: f32(m.pp), rain: f32(mp.rainAcc), snow: f32(mp.snowAcc) };
        m.scalars.forEach((a, i) => { arrays[`s${i}`] = f32(a); });
        let nest: Record<string, unknown> | null = null;
        if (experiment === 'nest' && nestSource) {
          const g = nestSource.payload;
          nest = { lat0: nestSource.lat0, lon0: nestSource.lon0, size: nestSource.size, preset: g.preset, day: g.day, nlat: g.nlat, nlon: g.nlon, K: g.K, hasQ: !!g.q, hasTs: !!g.ts, hasWet: !!g.wet, hasLand: !!g.land };
          for (const [k, a] of [['lat', g.lat], ['lon', g.lon], ['sigma', g.sigma], ['sigmaHalf', g.sigmaHalf], ['u', g.u], ['v', g.v], ['T', g.T], ['ps', g.ps], ['phis', g.phis], ['q', g.q], ['ts', g.ts], ['wet', g.wet]] as const) if (a) arrays[`p_${k}`] = f32(a);
          if (g.land) arrays.p_land = Uint8Array.from(g.land);
        }
        const t = m.time;
        const meta = { kind: 'regional' as const, title: `${EXP_LABEL[experiment]} · ${t < 7200 ? `${(t / 60).toFixed(0)} min` : `${(t / 3600).toFixed(1)} h`}`,
          experiment, builtGpu, grid: { nx: c.nx, ny: c.ny, nz: c.nz, dx: c.dx, dz: c.dz, size: m.size, nsc: m.scalars.length },
          time: t, steps: m.steps, dt: gpu ? gpu.dt : c.dt, frameVel, dpEnv, adaptive, nest };
        const buffer = packSave(meta, arrays);
        post({ type: 'saveData', meta, buffer }, [buffer]);
      } catch (e) { post({ type: 'error', message: `存檔失敗 / save failed: ${String(e)}` }); }
      busy = false; running = wasRunning;
    }
    else if (msg.type === 'load') {
      running = false;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      busy = true;
      try {
        const { meta, arrays } = unpackSave(msg.buffer);
        const M = meta as unknown as { experiment: RegionalExperiment; builtGpu: boolean; grid: { nx: number; ny: number; nz: number; size: number; nsc: number }; time: number; steps: number; dt: number; frameVel: { u: number; v: number }; dpEnv: number; nest: Record<string, unknown> | null };
        if (meta.kind !== 'regional') throw new Error('不是區域模式存檔 / not a regional save');
        const device = msg.backend === 'cpu' ? null : await getGpu();
        gpu?.destroy(); gpu = null;
        let info: { dt: number; description: string; land?: Uint8Array | null };
        if (M.nest) {
          const n = M.nest as { lat0: number; lon0: number; size: NestSize; preset: string; day: number; nlat: number; nlon: number; K: number };
          const g = (k: string): Float32Array | null => (arrays[`p_${k}`] as Float32Array | undefined) ?? null;
          const payload: NestPayload = { preset: n.preset, day: n.day, nlat: n.nlat, nlon: n.nlon, K: n.K, lat: Float64Array.from(g('lat')!), lon: Float64Array.from(g('lon')!),
            sigma: Float64Array.from(g('sigma')!), sigmaHalf: Float64Array.from(g('sigmaHalf')!), u: g('u')!, v: g('v')!, T: g('T')!, ps: g('ps')!, q: g('q'), phis: g('phis')!, ts: g('ts'), wet: g('wet'),
            land: (arrays.p_land as Uint8Array | undefined) ?? null };
          info = buildNest(payload, n.lat0, n.lon0, n.size, M.builtGpu);
          nestSource = { payload, lat0: n.lat0, lon0: n.lon0, size: n.size };
        } else { info = build(M.experiment, M.builtGpu); nestSource = null; }
        builtGpu = M.builtGpu;
        const mm = m!, mpp = mp!;
        if (mm.size !== M.grid.size || mm.scalars.length !== M.grid.nsc) throw new Error('存檔網格與目前版本不符 / the saved grid does not match this version');
        if (tracker && (M.frameVel.u !== frameVel.u || M.frameVel.v !== frameVel.v)) {
          mm.shiftFrame(M.frameVel.u - frameVel.u, M.frameVel.v - frameVel.v);
          frameVel = { ...M.frameVel }; if (physCfg) physCfg.frameVel = frameVel;
        }
        for (const [k, a] of [['u', mm.u], ['v', mm.v], ['w', mm.w], ['th', mm.th], ['pp', mm.pp]] as const) a.set(arrays[k] as Float32Array);
        mm.scalars.forEach((a, i) => a.set(arrays[`s${i}`] as Float32Array));
        mpp.rainAcc.set(arrays.rain as Float32Array); mpp.snowAcc.set(arrays.snow as Float32Array);
        mm.time = M.time; mm.steps = M.steps;
        if (M.dpEnv) dpEnv = M.dpEnv;
        let note = `已載入存檔 / save loaded (t = ${(M.time / 3600).toFixed(2)} h)`;
        if (device) {
          const reason = await tryGpu(device);
          if (reason) note += ' · ' + gpuFailNote(reason);
          else if (gpu) (gpu as GpuRegional).setDt(M.dt);
        } else if (M.builtGpu) note += ' · 此存檔是 GPU 網格，在 CPU 上會很慢 / GPU-sized grid: very slow on the CPU';
        dt0 = mm.c.dt;
        const c = mm.c;
        post({ type: 'ready', experiment, nx: c.nx, ny: c.ny, nz: c.nz, dx: c.dx, dz: c.dz, dt: info.dt, description: info.description, backend: gpu ? 'gpu' : 'cpu', note, land: info.land ?? null, refineTo: REFINE_TO[experiment] ?? null });
        busy = false;
        await sendFrame();
      } catch (e) { post({ type: 'error', message: `載入失敗 / load failed: ${String((e as Error).message ?? e)}` }); }
      busy = false;
    }
    else if (msg.type === 'refine') {
      const target = REFINE_TO[experiment];
      if (!target || !m || !mp) return;
      const wasRunning = running; running = false;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      busy = true;
      try {
        await syncFromGpu();
        const mc = m, mpc = mp, fromDx = mc.c.dx, frameC = { ...frameVel }, trackC = tracker?.position ?? null;
        const device = gpu ? gpu.device : null;
        gpu?.destroy(); gpu = null;
        const info = build(target, !!device);
        builtGpu = !!device;
        const mf = m!;
        // moving-frame experiments: bring the fine model into the coarse model's current frame
        if (tracker && (frameC.u !== frameVel.u || frameC.v !== frameVel.v)) {
          mf.shiftFrame(frameC.u - frameVel.u, frameC.v - frameVel.v);
          frameVel = frameC; if (physCfg) physCfg.frameVel = frameVel;
        }
        // sub-box around the storm when the fine domain is smaller
        const Lc = mc.c.nx * mc.c.dx, Lf = mf.c.nx * mf.c.dx;
        const cx = trackC?.x ?? Lc / 2, cy = trackC?.y ?? Lc / 2;
        const x0 = Math.max(0, Math.min(Lc - Lf, cx - Lf / 2)), y0 = Math.max(0, Math.min(Lc - Lf, cy - Lf / 2));
        refineInto(mc, mf, x0, y0, { rain: mpc.rainAcc, snow: mpc.snowAcc }, { rain: mp!.rainAcc, snow: mp!.snowAcc });
        let note = `已從 Δx ${fromDx >= 1000 ? `${fromDx / 1000} km` : `${fromDx} m`} 細化 / refined from Δx ${fromDx >= 1000 ? `${fromDx / 1000} km` : `${fromDx} m`} at t = ${(mc.time / 60).toFixed(0)} min`;
        if (device) {
          const reason = await tryGpu(device);
          if (reason) note += ' · ' + gpuFailNote(reason);
        }
        dt0 = mf.c.dt;
        const c = mf.c;
        post({ type: 'ready', land: null, experiment: target, nx: c.nx, ny: c.ny, nz: c.nz, dx: c.dx, dz: c.dz, dt: info.dt, description: info.description, backend: gpu ? 'gpu' : 'cpu', note, refineTo: REFINE_TO[target] ?? null });
        busy = false;
        await sendFrame();
      } catch (e) { post({ type: 'error', message: `細化失敗 / refinement failed: ${String(e)}` }); }
      busy = false; running = wasRunning;
    }
    else if (msg.type === 'adaptive') {
      adaptive = msg.on;
      if (!adaptive && gpu) { while (busy) await new Promise((r) => setTimeout(r, 5)); gpu.setDt(dt0); }
    }
    else if (msg.type === 'profile') {
      if (!gpu || !m) { post({ type: 'profile', text: '效能分析需要 WebGPU 後端 / profiling needs the WebGPU backend' }); return; }
      const wasRunning = running; running = false;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      busy = true;
      try {
        const r = await gpu.profile(3), c = m.c;
        const cells = c.nx * c.ny * c.nz;
        const lines = [`效能分析 / Profile: ${c.nx}×${c.ny}×${c.nz} (${(cells / 1e6).toFixed(2)} M cells), Δt ${c.dt} s, ${r.method}`,
          `每步 / per step: ${r.total.toFixed(1)} ms → ${(1000 / r.total).toFixed(2)} 步/s steps/s, ${(cells * 1000 / r.total / 1e6).toFixed(1)} M cell-steps/s`,
          ...r.rows.map((x) => `${x.ms.toFixed(2).padStart(8)} ms ${(100 * x.ms / r.total).toFixed(1).padStart(5)}%  ×${x.calls}  ${x.label}`)];
        post({ type: 'profile', text: lines.join('\n') });
        m.time = gpu.time; m.steps = gpu.steps;
      } catch (e) { post({ type: 'profile', text: `效能分析失敗 / profiling failed: ${String(e)}` }); }
      busy = false; running = wasRunning;
    }
    else if (msg.type === 'boundary' && m && nestSpec && experiment === 'nest') {
      while (busy) await new Promise((r) => setTimeout(r, 5));
      m.boundary = nestTargets(msg.payload, nestSpec, m);
      gpu?.setBoundary(m.boundary);
    }
  } catch (e) { post({ type: 'error', message: String(e) }); }
};

async function loop(): Promise<void> {
  for (;;) {
    if (m && mp && running && !busy) {
      busy = true;
      try {
        if (gpu) {
          // adaptive batch: about 60 ms x speed setting of GPU work between checks
          const t0 = performance.now();
          // with adaptive stepping, re-check the Courant number at least every 300 model seconds
          const nb0 = adaptive ? Math.max(1, Math.min(gpuBatch, Math.ceil(300 / gpu.dt))) : gpuBatch;
          const nb = pacer.allow(gpu.time, gpu.dt, nb0);
          if (nb === 0) { busy = false; await new Promise((r) => setTimeout(r, 15)); continue; }
          gpu.step(nb); rateSteps += nb;
          await gpu.device.queue.onSubmittedWorkDone();
          await adaptDt();
          const el = performance.now() - t0, target = 60 * stepsPerTick;
          if (DEBUG) console.log(`DBG batch ${gpuBatch} steps ${el.toFixed(0)} ms`);
          const perStep = Math.max(el, 1) / nb;
          gpuBatch = Math.max(1, Math.min(2000, Math.round(Math.min(2 * gpuBatch, Math.max(0.5 * gpuBatch, target / perStep)))));
        }
        else {
          const ns = pacer.allow(m.time, m.c.dt, stepsPerTick);
          if (ns === 0) { busy = false; await new Promise((r) => setTimeout(r, 15)); continue; }
          for (let s = 0; s < ns; s++) { m.step(); mp.apply(m.c.dt); rateSteps++; }
        }
        if (pacer.reached(modelNow())) {
          running = false;
          await sendFrame();
          post({ type: 'paused', reason: `已到達設定時間，自動暫停 / reached the stop time (t = ${(modelNow() / 3600).toFixed(2)} h)` });
        }
        if (tracker && m.time - lastTrack >= 600) { lastTrack = m.time; await followStorm(); }
        if (running && frameDue()) { const tf = performance.now(); await sendFrame(); if (DEBUG) console.log(`DBG frame ${(performance.now() - tf).toFixed(0)} ms`); }
      } catch (e) { running = false; post({ type: 'error', message: String(e) }); }
      busy = false;
    }
    const now = performance.now();
    if (now - rateT > 1000) { rate = rateSteps * 1000 / (now - rateT); rateSteps = 0; rateT = now; }
    await new Promise((r) => setTimeout(r, running ? 0 : 30));
  }
}
void loop();

/** Keep the storm inside the periodic domain: Galilean frame shift and whole-cell re-centring
 *  (exact symmetries); on the GPU the model is rebuilt with the new frame velocity. */
async function followStorm(): Promise<void> {
  if (!m || !mp || !tracker || !physCfg) return;
  await syncFromGpu();
  const a = tracker.update(m);
  if (!a || (!a.du && !a.dv && !a.di && !a.dj)) return;
  if (a.du || a.dv) { m.shiftFrame(a.du, a.dv); frameVel.u += a.du; frameVel.v += a.dv; physCfg.frameVel = frameVel; }
  if (a.di || a.dj) m.roll(a.di, a.dj, [mp.rainAcc, mp.snowAcc]);
  if (gpu) {
    const device = gpu.device, dtNow = gpu.dt;
    gpu.destroy();
    gpu = new GpuRegional(device, m, { moist: true, physics: physCfg, ice: true });
    gpu.uploadFrom(m, { rain: mp.rainAcc, snow: mp.snowAcc });
    gpu.setDt(dtNow);
  }
}

/** Copy the GPU state into the CPU model arrays (display and diagnostics reuse the CPU code). */
async function syncFromGpu(): Promise<void> {
  if (!gpu || !m || !mp) return;
  const st = await gpu.readState(), size = m.size;
  const arrs = [m.u, m.v, m.w, m.th, m.pp, ...m.scalars];
  arrs.forEach((a, f) => { for (let i = 0; i < size; i++) a[i] = st[f * size + i]!; });
  const rain = await gpu.readRain(), snow = await gpu.readSnow();
  for (let j = 0; j < m.c.ny; j++) for (let i = 0; i < m.c.nx; i++) { mp.rainAcc[j * m.c.nx + i] = rain[m.idx(i, j, 0)]!; mp.snowAcc[j * m.c.nx + i] = snow[m.idx(i, j, 0)]!; }
  m.time = gpu.time; m.steps = gpu.steps;
}

async function sendFrame(): Promise<void> {
  if (!m || !mp) return;
  lastFrame = performance.now(); lastFrameModel = modelNow();
  const { nx, ny, nz, dx, dz } = m.c, n = nx * ny * nz;
  const cloud = new Uint8Array(n), rain = new Uint8Array(n);
  let wmax = 0, wmin = 0, qcmax = 0, qrmax = 0;
  if (gpu) {
    // GPU: packed display bytes, column extremes and the few horizontal planes the diagnostics need
    let k15 = 0; for (let k = 0; k < nz; k++) if (Math.abs(m.zc[k]! - 1500) < Math.abs(m.zc[k15]! - 1500)) k15 = k;
    const d = await gpu.readDisplay(k15 === 0 ? [0] : [0, k15]);
    // keep the GPU busy while this frame is unpacked on the CPU
    const tNow = gpu.time, sNow = gpu.steps;
    if (running) { gpu.step(gpuBatch); rateSteps += gpuBatch; }
    for (let i = 0; i < n; i++) { const v = d.packed[i]!; cloud[i] = v & 255; rain[i] = (v >> 8) & 255; }
    for (let c = 0; c < nx * ny; c++) { wmax = Math.max(wmax, d.col[4 * c]!); wmin = Math.min(wmin, d.col[4 * c + 1]!); qcmax = Math.max(qcmax, d.col[4 * c + 2]!); qrmax = Math.max(qrmax, d.col[4 * c + 3]!); }
    for (const [k, pl] of d.planes) { const o = k * m.plane; m.u.set(pl.u, o); m.v.set(pl.v, o); m.th.set(pl.th, o); m.pp.set(pl.pp, o); }
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const q = m.idx(i, j, 0); mp.rainAcc[j * nx + i] = d.rain[q]!; mp.snowAcc[j * nx + i] = d.snow[q]!; }
    m.time = tNow; m.steps = sNow;
    if (!Number.isFinite(wmax)) { running = false; post({ type: 'error', message: '數值發散 / numerical blow-up' }); }
  } else {
    if (!Number.isFinite(m.w[m.idx(0, 0, 1)]!)) { running = false; post({ type: 'error', message: '數值發散 / numerical blow-up' }); }
    const qc = m.scalars[QC]!, qr = m.scalars[QR]!, qi = m.scalars[QI]!, qs = m.scalars[QS]!, qg = m.scalars[QG]!;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const q = m.idx(i, j, k), o = (k * ny + j) * nx + i;
      // cloud: liquid + ice cloud; precipitation: rain + snow + graupel
      const cl = Math.max(0, qc[q]! + qi[q]!), pr = Math.max(0, qr[q]! + qs[q]! + qg[q]!);
      cloud[o] = Math.min(255, Math.round(Math.sqrt(cl / 3e-3) * 255));
      rain[o] = Math.min(255, Math.round(Math.sqrt(pr / 8e-3) * 255));
      qcmax = Math.max(qcmax, cl); qrmax = Math.max(qrmax, pr);
      const w = m.w[q]!; wmax = Math.max(wmax, w); wmin = Math.min(wmin, w);
    }
  }
  const g = new Float32Array(nx * ny);
  let vmax = 0, rainmax = 0;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const q = m.idx(i, j, 0);
    const sp = Math.hypot(0.5 * (m.u[q]! + m.u[q + 1]!), 0.5 * (m.v[q]! + m.v[q + m.sx]!));
    vmax = Math.max(vmax, sp);
    rainmax = Math.max(rainmax, mp.rainAcc[j * nx + i]!);
    g[j * nx + i] = ground === 'rain' ? mp.rainAcc[j * nx + i]! : ground === 'snow' ? mp.snowAcc[j * nx + i]! : ground === 'wind' ? sp : m.th[q]! - m.th0[0]!;
  }
  let lo = Infinity, hi = -Infinity;
  for (const v of g) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (ground === 'theta') { const a = Math.max(Math.abs(lo), Math.abs(hi), 0.5); lo = -a; hi = a; }
  else { lo = 0; hi = Math.max(hi, ground === 'rain' || ground === 'snow' ? 5 : 10); }
  let dp: number | null = null, rmw: number | null = null, eyewalls: { r: number; v: number }[] | null = null;
  if (experiment === 'tc' || experiment === 'tc_hr') { const r = tcMetrics(m); dp = r.pmin - dpEnv; rmw = r.rmw; eyewalls = eyewallProfile(m).peaks; }
  // lowest-level vertical vorticity (cell corners) and ground-relative wind
  let zetaMax = 0, vGround = 0;
  for (let j = 1; j < ny; j++) for (let i = 1; i < nx; i++) {
    const q = m.idx(i, j, 0);
    const zeta = (m.v[q]! - m.v[q - 1]!) / dx - (m.u[q]! - m.u[q - m.sx]!) / m.c.dy;
    zetaMax = Math.max(zetaMax, zeta);
    vGround = Math.max(vGround, Math.hypot(0.5 * (m.u[q]! + m.u[q + 1]!) + frameVel.u, 0.5 * (m.v[q]! + m.v[q + m.sx]!) + frameVel.v));
  }
  post({ type: 'frame', time: m.time, nx, ny, nz, dx, dz, cloud, rain, ground: g, groundField: ground, groundRange: [lo, hi],
    stats: { wmax, wmin, qcmax, qrmax, rainmax, vmax, dp, rmw, eyewalls, zetaMax, vGround }, stepsPerSecond: rate, dt: gpu ? gpu.dt : m.c.dt }, [cloud.buffer, rain.buffer, g.buffer]);
}
