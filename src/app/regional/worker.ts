// Regional-model worker: builds an experiment, steps it, and streams 3-D cloud / rain volumes.

import { RegionalModel } from '../../regional/core.js';
import { weismanKlemp } from '../../regional/kessler.js';
import { IceMicrophysics, QV, QC, QR, QI, QS, QG } from '../../regional/ice.js';
import { RegionalPhysics } from '../../regional/physics.js';
import { tcSounding, insertVortex, tcMetrics, eyewallProfile } from '../../regional/tropical.js';
import { GpuRegional } from '../../gpu/regionalGpu.js';
import type { RegionalPhysicsConfig } from '../../regional/physics.js';
import { nestFromGlobal, nestTargets, sampleSurface, NestSpec } from '../../regional/nest.js';
import { refineInto } from '../../regional/refine.js';
import { Tracers, type TracerParams } from '../../regional/tracers.js';
import { AxiDriver, AXI_DEFAULTS, LEGACY_TC, tcEnvText, type AxiParams } from './axiDriver.js';
import { packSave, unpackSave, type SaveArrays } from '../saves.js';
import { Pacer } from '../pacer.js';
import { tornadoExperiment, StormTracker, TORNADO_DEFAULT, TORNADO_WK82, type TornadoEnv } from '../../regional/supercell.js';
import { REFINE_TO, type ChartData, type ChartRequest, type FromRegionalWorker, type GroundField, type NestPayload, type NestSize, type RegionalExperiment, type TcRain, type ToRegionalWorker } from './protocol.js';
import { C, COL as NCOL, SECTION_VARS, columnDiagnostics, columnProfiles, sectionFromColumns, sliceFields, modelPlanes, compositeMaps, azimuthalMeans, unpackRZ, type LevelPlanes, type SectionVar } from '../../regional/diagnostics.js';

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
/** the CPU physics of the current model (surface state for painting), null without sub-grid / surface physics */
let phys: RegionalPhysics | null = null;
let frameVel = { u: 0, v: 0 };
let tracker: StormTracker | null = null, lastTrack = 0;
let nestSpec: NestSpec | null = null;
/** how the current model was built (needed to rebuild it from a save) */
let builtGpu = false;
// pacing and display cadence
const pacer = new Pacer();
let frameMode: { kind: 'wall' | 'model' | 'fast'; every: number } = { kind: 'wall', every: 0 };
let lastFrameModel = 0;
// the axisymmetric tropical-cyclone experiment runs through its own driver (CPU, revolved for display)
let axi: AxiDriver | null = null;
/** tropical-cyclone environment chosen on the page (SST, humidity, radiation, gustiness, ...) for the 3-D TC experiments too */
let tcEnv: Partial<AxiParams> | null = null;
/** environment of the tornado experiments chosen on the page */
let tornadoEnv: TornadoEnv = { ...TORNADO_DEFAULT };
/** EF rating from a ground-relative wind speed (m/s; 3-s gust thresholds of the Enhanced Fujita scale) */
const efRating = (v: number): number => (v >= 89 ? 5 : v >= 74 ? 4 : v >= 61 ? 3 : v >= 50 ? 2 : v >= 38 ? 1 : 0);
const modelNow = (): number => (axi ? axi.time : gpu ? gpu.time : m ? m.time : 0);
const dtNow = (): number => (axi ? axi.dt : gpu ? gpu.dt : m ? m.c.dt : 1);
function frameDue(): boolean {
  const el = performance.now() - lastFrame;
  if (frameMode.kind === 'fast') return el > 5000;
  if (frameMode.kind === 'model') return el > 250 && modelNow() - lastFrameModel >= frameMode.every - 1e-6;
  return el > (gpu ? 500 : 300);
}
let nestSource: { payload: NestPayload; lat0: number; lon0: number; size: NestSize } | null = null;
// charts: what the page wants in each frame, the 3-D view's second channel
let chartReq: ChartRequest | null = null, volMode = 0;
// ground-relative position of the domain origin: advances with the frame velocity, jumps with rolls and refinement
let origin = { x: 0, y: 0 }, originT = 0;
function advanceOrigin(): void { const t = modelNow(); origin.x += frameVel.u * (t - originT); origin.y += frameVel.v * (t - originT); originT = t; }
function resetOrigin(x = 0, y = 0): void { origin = { x, y }; originT = modelNow(); prevAcc = null; tcAcc = null; tcRain = null; }
// precipitation rate from the change of the accumulation between frames
let prevAcc: { t: number; rain: Float32Array } | null = null, lastRate: Float32Array | null = null;
const currentGpu = (): GpuRegional | null => gpu;
const isTc = (): boolean => experiment === 'tc' || experiment === 'tc_hr' || experiment === 'tc_3';
// tropical cyclones: mean precipitation rate in the core and the outer region over the last completed hour
let tcAcc: { t: number; rain: Float32Array } | null = null, tcRain: TcRain | null = null;
// tracer particles (3-D view): seeded in the lowest 2 km, half of them near the storm
let tracers: Tracers | null = null, tracerN = 0, trSeed = 1, stormDomain: { x: number; y: number } | null = null;
function tracerParams(): TracerParams {
  const c = m!.c, L = Math.min(c.nx * c.dx, c.ny * c.dy), st = stormDomain;
  return { life: isTc() ? 4 * 3600 : experiment === 'nest' ? 6 * 3600 : 1800, cx: st?.x ?? 0, cy: st?.y ?? 0, rad: st ? Math.min(L / 4, isTc() ? 200000 : 20000) : 0, zSeed: 2000 };
}
/** (Re)create the particles for the current model (after it is built, refined or loaded). */
function setupTracers(keep: Float32Array | null = null): void {
  if (!m || tracerN === 0) { tracers = null; gpu?.initTracers(new Float32Array(0)); return; }
  tracers = new Tracers(tracerN, m, tracerParams());
  if (keep && keep.length === tracers.pos.length) tracers.pos.set(keep);
  gpu?.initTracers(Float32Array.from(tracers.pos));
}
function advectTracers(dt: number, nsteps: number): void {
  if (!tracers) return;
  tracers.params = tracerParams();
  const nsub = Math.max(1, Math.min(64, nsteps));
  if (gpu) { trSeed = (trSeed + 1) >>> 0; gpu.advectTracers(dt, nsub, tracers.params, Math.imul(trSeed, 2654435761) >>> 0); }
  else tracers.advect(dt, nsub);
}
async function refreshFrame(): Promise<void> {
  if (running || (!m && !axi)) return;
  while (busy) await new Promise((r) => setTimeout(r, 5));
  busy = true;
  try { await sendFrame(); } catch (e) { post({ type: 'error', message: String(e) }); }
  busy = false;
}
const EXP_LABEL: Record<RegionalExperiment, string> = {
  supercell: '超大胞 2 km / supercell 2 km', supercell_hr: '超大胞 1 km / supercell 1 km', tc: '熱帶氣旋 15 km / tropical cyclone 15 km',
  tc_hr: '熱帶氣旋 5 km / tropical cyclone 5 km', tc_3: '熱帶氣旋 3 km / tropical cyclone 3 km', tornado: '龍捲超大胞 250 m / tornadic supercell 250 m', tornado_c: '龍捲超大胞 1 km / tornadic supercell 1 km', nest: '巢狀區域 / nest', tc_axi: '軸對稱颱風 / axisymmetric TC',
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
  physCfg = null; phys = null;
  frameVel = { u: 0, v: 0 };
  tracker = null;
  if (exp === 'tornado' || exp === 'tornado_c') {
    // GPU: 250 m LES over 50 km; CPU: a 500 m preview over 40 km (mesocyclone scale only);
    // tornado_c: 1 km spin-up over 100 km (same environment), refined to the 250 m box later
    const e = exp === 'tornado_c' ? tornadoExperiment(1000, 100000, 40, 400, 6, 6, tornadoEnv)
      : gpuOk ? tornadoExperiment(250, 50000, 64, 250, 2, 8, tornadoEnv) : tornadoExperiment(500, 40000, 40, 400, 3, 6, tornadoEnv);
    m = e.model; mp = new IceMicrophysics(m); physCfg = e.physics; frameVel = e.frame;
    tracker = new StormTracker(); lastTrack = 0;
    phys = new RegionalPhysics(m, physCfg);
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
    const hr = exp === 'tc_hr' || exp === 'tc_3', P = { ...AXI_DEFAULTS, ...(tcEnv ?? {}) };
    // 3 km needs a discrete GPU; without one the 3 km experiment runs on the 5 km grid
    const L = 1200000, dx = exp === 'tc_3' && gpuOk ? 3000 : hr ? 5000 : 15000, nx = L / dx, nz = hr ? 50 : 25, dz = hr ? 500 : 1000, f = P.f, sst = P.sst, dtm = dx === 3000 ? 18 : hr ? 30 : 60;
    m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt: dtm, nsound: 6, f, beta: 0.3, divDamp: 0.1, dampDepth: 6000, dampRate: 1 / 300, kdiff2: 0 }, tcSounding(P.snd, sst, P.rhTop), 6);
    mp = new IceMicrophysics(m);
    // the mixing lengths stay those of the 3-D set-up (the panel's lh / lv apply to the axisymmetric version)
    physCfg = { lh: 0.2 * dx, lv: 100, sst, ck: P.ck, radTau: 12 * 3600, radMax: P.radMax / 86400, vmin: P.vmin, radConst: P.radConst / 86400, blNoise: P.blNoise };
    phys = new RegionalPhysics(m, physCfg);
    insertVortex(m, f, P.vmax0);
    dpEnv = 1e5 * Math.pow(m.pi0[0]!, 1004.5 / 287.05) / 100;
    const e = tcEnvText(P);
    return { dt: dtm, description: `熱帶氣旋 / tropical cyclone：海面上的弱渦旋（f 平面，Δx ${dx / 1000} km；${e.zh}）/ weak vortex over the sea (f-plane; ${e.en})` };
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
  phys = new RegionalPhysics(m, physCfg);
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
    if (msg.type === 'init' && msg.experiment === 'tc_axi') {
      running = false;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      busy = true;
      try {
        gpu?.destroy(); gpu = null; m = null; mp = null; tracker = null; nestSource = null;
        experiment = 'tc_axi'; frameVel = { u: 0, v: 0 };
        axi = new AxiDriver({ ...AXI_DEFAULTS, ...(msg.axi ?? {}) });
        resetOrigin(); stormDomain = null; tracers = null;
        post({ type: 'ready', land: null, experiment: 'tc_axi', nx: axi.N, ny: axi.N, nz: axi.ax.a.nz, dx: axi.dxv, dz: axi.ax.a.dz, dt: axi.dt, description: axi.description(), backend: 'cpu',
          note: '軸對稱模式在 CPU 上執行（很快）；3D 畫面是把半徑–高度場繞軸旋轉 / the axisymmetric model runs on the CPU (fast); the 3-D view revolves the radius-height fields', refineTo: null });
        await sendFrame();
      } finally { busy = false; }
    }
    else if (msg.type === 'init') {
      running = false;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      busy = true;
      try {
      axi = null;
      const device = msg.backend === 'cpu' ? null : await getGpu();
      gpu?.destroy(); gpu = null;
      if (msg.experiment === 'tc' || msg.experiment === 'tc_hr' || msg.experiment === 'tc_3') tcEnv = msg.axi ?? tcEnv;
      if ((msg.experiment === 'tornado' || msg.experiment === 'tornado_c') && msg.tornado) tornadoEnv = { ...TORNADO_DEFAULT, ...msg.tornado };
      let info = build(msg.experiment, !!device);
      builtGpu = !!device; nestSource = null;
      let note = '';
      if (device) {
        const reason = await tryGpu(device);
        if (reason) { note = gpuFailNote(reason); info = build(msg.experiment, false); builtGpu = false; }
      } else if (msg.backend !== 'cpu') note = 'WebGPU 不可用，改用 CPU / WebGPU unavailable, using CPU';
      if (!gpu && (msg.experiment === 'tc_hr' || msg.experiment === 'tc_3' || msg.experiment === 'supercell_hr' || msg.experiment === 'tornado')) note += (note ? ' · ' : '') + '此高解析實驗在 CPU 上非常慢 / this high-resolution experiment is very slow on the CPU';
      const c = m!.c;
      dt0 = c.dt; resetOrigin(); stormDomain = null; setupTracers();
      post({ type: 'ready', land: null, experiment: msg.experiment, nx: c.nx, ny: c.ny, nz: c.nz, dx: c.dx, dz: c.dz, dt: info.dt, description: info.description, backend: gpu ? 'gpu' : 'cpu', note, refineTo: REFINE_TO[msg.experiment] ?? null });
      await sendFrame();
      } finally { busy = false; }
    }
    else if (msg.type === 'initNest') {
      running = false;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      busy = true;
      try {
      axi = null;
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
      dt0 = c.dt; resetOrigin(); stormDomain = null; setupTracers();
      post({ type: 'ready', experiment: 'nest', nx: c.nx, ny: c.ny, nz: c.nz, dx: c.dx, dz: c.dz, dt: info.dt, description: info.description, backend: gpu ? 'gpu' : 'cpu', note, land: info.land, refineTo: null });
      await sendFrame();
      } finally { busy = false; }
    }
    else if (msg.type === 'run') { running = msg.running; if (running) pacer.reset(modelNow()); }
    else if (msg.type === 'pace') { pacer.target = Math.max(0, msg.target); pacer.reset(modelNow()); }
    else if (msg.type === 'runUntil') { pacer.until = msg.hours > 0 ? modelNow() + msg.hours * 3600 : null; }
    else if (msg.type === 'frames') { frameMode = { kind: msg.kind, every: msg.every ?? 0 }; }
    else if (msg.type === 'step1' && axi) {
      if (running) return;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      busy = true;
      try { axi.step(1); await sendFrame(); } catch (e) { post({ type: 'error', message: String(e) }); }
      busy = false;
    }
    else if (msg.type === 'step1') {
      if (running || !m || !mp) return;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      busy = true;
      try {
        if (gpu) { const d1 = gpu.dt; gpu.step(1); advectTracers(d1, 1); await gpu.device.queue.onSubmittedWorkDone(); await adaptDt(); }
        else { m.step(); mp.apply(m.c.dt); advectTracers(m.c.dt, 1); }
        busy = false;
        await sendFrame();
      } catch (e) { post({ type: 'error', message: String(e) }); }
      busy = false;
    }
    else if (msg.type === 'speed') stepsPerTick = Math.max(1, msg.stepsPerTick | 0);
    else if (msg.type === 'ground') { ground = msg.field; await refreshFrame(); }
    else if (msg.type === 'charts') { chartReq = msg.req; await refreshFrame(); }
    else if (msg.type === 'volMode') { volMode = msg.mode | 0; await refreshFrame(); }
    else if (msg.type === 'tracers') {
      while (busy) await new Promise((r) => setTimeout(r, 5));
      tracerN = Math.max(0, Math.min(65536, msg.n | 0)); setupTracers(); await refreshFrame();
    }
    else if (msg.type === 'save' && axi) {
      const wasRunning = running; running = false;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      busy = true;
      try {
        const a = axi.ax, t = a.time;
        const meta = { kind: 'regional' as const, title: `${EXP_LABEL.tc_axi} · ${(t / 3600).toFixed(1)} h`, experiment: 'tc_axi', builtGpu: false, axi: axi.p,
          grid: { nx: a.a.nr, ny: 1, nz: a.a.nz, dx: a.a.dr, dz: a.a.dz, size: a.size, nsc: a.scalars.length }, time: t, steps: a.steps, dt: a.a.dt, frameVel, dpEnv: 0, adaptive: false, nest: null, origin };
        const buffer = packSave(meta, axi.arrays());
        post({ type: 'saveData', meta, buffer }, [buffer]);
      } catch (e) { post({ type: 'error', message: `存檔失敗 / save failed: ${String(e)}` }); }
      busy = false; running = wasRunning;
    }
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
          time: t, steps: m.steps, dt: gpu ? gpu.dt : c.dt, frameVel, dpEnv, adaptive, nest, origin, tcEnv, tornadoEnv };
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
        const M = meta as unknown as { experiment: RegionalExperiment; builtGpu: boolean; grid: { nx: number; ny: number; nz: number; size: number; nsc: number }; time: number; steps: number; dt: number; frameVel: { u: number; v: number }; dpEnv: number; nest: Record<string, unknown> | null; origin?: { x: number; y: number } };
        if (meta.kind !== 'regional') throw new Error('不是區域模式存檔 / not a regional save');
        if (M.experiment === 'tc_axi') {
          gpu?.destroy(); gpu = null; m = null; mp = null; tracker = null; nestSource = null; experiment = 'tc_axi'; frameVel = { u: 0, v: 0 };
          axi = new AxiDriver({ ...AXI_DEFAULTS, ...LEGACY_TC, ...((meta as unknown as { axi?: AxiParams }).axi ?? {}) });
          axi.restore(arrays as Record<string, Float32Array>, M.time, M.steps);
          resetOrigin(); stormDomain = null; tracers = null;
          post({ type: 'ready', land: null, experiment: 'tc_axi', nx: axi.N, ny: axi.N, nz: axi.ax.a.nz, dx: axi.dxv, dz: axi.ax.a.dz, dt: axi.dt, description: axi.description(), backend: 'cpu', note: `已載入存檔 / save loaded (t = ${(M.time / 3600).toFixed(2)} h)`, refineTo: null });
          busy = false;
          await sendFrame();
          return;
        }
        axi = null;
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
        } else {
          // saves from before the unstable sounding used the neutral sounding with radiative relaxation
          tcEnv = { ...LEGACY_TC, ...((meta as unknown as { tcEnv?: Partial<AxiParams> | null }).tcEnv ?? {}) };
          tornadoEnv = { ...((meta as unknown as { tornadoEnv?: TornadoEnv }).tornadoEnv ?? TORNADO_WK82) };
          info = build(M.experiment, M.builtGpu); nestSource = null;
        }
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
        resetOrigin(M.origin?.x ?? 0, M.origin?.y ?? 0); stormDomain = null; setupTracers();
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
        advanceOrigin();
        const mc = m, mpc = mp, fromDx = mc.c.dx, frameC = { ...frameVel }, trackC = tracker?.position ?? null, originC = { ...origin };
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
        resetOrigin(originC.x + x0, originC.y + y0); stormDomain = null; setupTracers();
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
    else if (msg.type === 'perturb' || msg.type === 'paint' || msg.type === 'environment') {
      if (!m || !mp) { post({ type: 'error', message: '這個實驗不支援此互動（軸對稱模式請改用參數面板）/ this experiment does not support this interaction (use the panel for the axisymmetric model)' }); return; }
      while (busy) await new Promise((r) => setTimeout(r, 5));
      busy = true;
      try {
        const note = await interact(msg);
        if (note) post({ type: 'log', text: note });
        if (!running) await sendFrame();
      } catch (e) { post({ type: 'error', message: String(e) }); }
      busy = false;
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
    if (axi && running && !busy) {
      busy = true;
      try {
        // steps for about 40 ms x the work setting, within the pacer's allowance
        const t0 = performance.now(), want = pacer.allow(axi.time, axi.dt, 400 * stepsPerTick);
        let n = 0;
        while (n < want && performance.now() - t0 < 40 * stepsPerTick) { axi.step(1); n++; rateSteps++; }
        if (n === 0) { busy = false; await new Promise((r) => setTimeout(r, 15)); continue; }
        if (pacer.reached(modelNow())) {
          running = false;
          await sendFrame();
          post({ type: 'paused', reason: `已到達設定時間，自動暫停 / reached the stop time (t = ${(modelNow() / 3600).toFixed(2)} h)` });
        }
        if (running && frameDue()) await sendFrame();
      } catch (e) { running = false; post({ type: 'error', message: String(e) }); }
      busy = false;
    }
    else if (m && mp && running && !busy) {
      busy = true;
      try {
        if (gpu) {
          // adaptive batch: about 60 ms x speed setting of GPU work between checks
          const t0 = performance.now();
          // with adaptive stepping, re-check the Courant number at least every 300 model seconds
          const nb0 = adaptive ? Math.max(1, Math.min(gpuBatch, Math.ceil(300 / gpu.dt))) : gpuBatch;
          const nb = pacer.allow(gpu.time, gpu.dt, nb0);
          if (nb === 0) { busy = false; await new Promise((r) => setTimeout(r, 15)); continue; }
          const dtb = gpu.dt;
          gpu.step(nb); rateSteps += nb;
          advectTracers(nb * dtb, nb);
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
          advectTracers(ns * m.c.dt, ns);
        }
        advanceOrigin();
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
  advanceOrigin();
  if (a.du || a.dv) { m.shiftFrame(a.du, a.dv); frameVel.u += a.du; frameVel.v += a.dv; physCfg.frameVel = frameVel; }
  if (a.di || a.dj) { m.roll(a.di, a.dj, [mp.rainAcc, mp.snowAcc]); origin.x -= a.di * m.c.dx; origin.y -= a.dj * m.c.dy; prevAcc = null; }
  // particles keep their positions (moved with a roll of the fields)
  let tp = gpu ? (gpu.tracerCount ? await gpu.readTracers() : null) : tracers ? tracers.pos.slice() : null;
  if (tp && (a.di || a.dj)) {
    const Lx = m.c.nx * m.c.dx, Ly = m.c.ny * m.c.dy;
    for (let p = 0; p < tp.length; p += 4) { tp[p] = ((tp[p]! + a.di * m.c.dx) % Lx + Lx) % Lx; tp[p + 1] = ((tp[p + 1]! + a.dj * m.c.dy) % Ly + Ly) % Ly; }
  }
  if (gpu) {
    const device = gpu.device, dtNow = gpu.dt;
    gpu.destroy();
    gpu = new GpuRegional(device, m, { moist: true, physics: physCfg, ice: true });
    gpu.uploadFrom(m, { rain: mp.rainAcc, snow: mp.snowAcc });
    gpu.setDt(dtNow);
  }
  if (tp && tracers) { tracers.pos.set(tp); gpu?.initTracers(tp); }
  else if (tp) setupTracers(tp);
}

/** Interaction: change the conditions (never the outcome). Returns a log line. */
async function interact(msg: Extract<ToRegionalWorker, { type: 'perturb' | 'paint' | 'environment' }>): Promise<string> {
  const mm = m!, { nx, ny, nz, dx, dy } = mm.c, L = Math.min(nx * dx, ny * dy);
  if (msg.type === 'paint') {
    const sf = phys?.surface;
    if (!sf || !physCfg) return '這個實驗沒有地面通量，不能塗海溫或陸地 / this experiment has no surface fluxes to paint';
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      if (Math.hypot((i + 0.5) * dx - msg.x, (j + 0.5) * dy - msg.y) > msg.radius) continue;
      const c = j * nx + i;
      if (msg.kind === 'warmer') sf.tsk[c] = Math.min(310, sf.tsk[c]! + 2);
      else if (msg.kind === 'cooler') sf.tsk[c] = Math.max(271.35, sf.tsk[c]! - 2);
      else if (msg.kind === 'land') sf.wet[c] = 0.3;
      else sf.wet[c] = 1;
    }
    // keep the painting when the GPU model is rebuilt (storm following) and in the configuration
    physCfg.surface = { tsk: sf.tsk, wet: sf.wet };
    gpu?.setSurface(sf.tsk, sf.wet);
    return '';
  }
  await syncFromGpu();
  if (msg.type === 'perturb') {
    // warm bubble (+3 K centred at 1.5 km) or cold pool (-6 K at the ground), horizontal radius 10 km (at least 4 cells, at most L/8)
    const rh = Math.max(4 * dx, Math.min(10000, L / 8)), warm = msg.kind === 'warm', zc = warm ? 1500 : 0, rz = 1500, amp = warm ? 3 : -6;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const r = Math.sqrt((((i + 0.5) * dx - msg.x) / rh) ** 2 + (((j + 0.5) * dy - msg.y) / rh) ** 2 + ((mm.zc[k]! - zc) / rz) ** 2);
      if (r < 1) { const q = mm.idx(i, j, k); mm.th[q] = mm.th[q]! + amp * Math.cos(0.5 * Math.PI * r) ** 2; }
    }
    if (gpu) gpu.uploadFrom(mm, { rain: mp!.rainAcc, snow: mp!.snowAcc });
    return `${warm ? '放暖泡 / warm bubble' : '放冷池 / cold pool'} at (${(msg.x / 1000).toFixed(1)}, ${(msg.y / 1000).toFixed(1)}) km`;
  }
  // environment: wind increment linear to 6 km (constant above) and a humidity factor tapered over 1-8 km
  const du = (z: number): number => msg.du6 * Math.min(1, z / 6000);
  const hf = (z: number): number => 1 + (msg.humidity - 1) * Math.max(0, Math.min(1, (z - 500) / 500, (8500 - z) / 500));
  const qv = mm.scalars[QV]!;
  for (let k = 0; k < nz; k++) {
    const d = du(mm.zc[k]!), f = hf(mm.zc[k]!);
    mm.ub[k] = mm.ub[k]! + d;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const q = mm.idx(i, j, k);
      mm.u[q] = mm.u[q]! + d;
      if (f !== 1) {
        const pi = mm.pi0[k]! + mm.pp[q]!, T = mm.th[q]! * pi, p = 1e5 * Math.pow(pi, 1004.5 / 287.05), es = 611.2 * Math.exp(17.67 * (T - 273.15) / (T - 29.65));
        qv[q] = Math.min(qv[q]! * f, 0.622 * es / Math.max(p - es, 1));
      }
      const b = mm.boundary;
      if (b) { b.u[q] = b.u[q]! + d; if (b.qv) b.qv[q] = b.qv[q]! * f; }
    }
  }
  if (gpu) {
    // the damping-layer wind and the boundary targets live in the GPU model's tables: rebuild it
    const device = gpu.device, dtNow = gpu.dt, tp = gpu.tracerCount ? await gpu.readTracers() : null;
    gpu.destroy();
    gpu = new GpuRegional(device, mm, { moist: true, physics: physCfg, ice: true });
    gpu.uploadFrom(mm, { rain: mp!.rainAcc, snow: mp!.snowAcc });
    gpu.setDt(dtNow);
    if (tp) gpu.initTracers(tp);
  }
  return `改變環境 / environment changed: ${msg.du6 >= 0 ? '+' : ''}${msg.du6} m/s westerly at 6 km, 1-8 km humidity x${msg.humidity}`;
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
  if (axi) {
    lastFrame = performance.now(); lastFrameModel = modelNow();
    const { msg, transfer } = axi.frame(chartReq, ground, volMode, rate, { ...origin });
    post(msg, transfer);
    return;
  }
  if (!m || !mp) return;
  lastFrame = performance.now(); lastFrameModel = modelNow();
  const { nx, ny, nz, dx, dz } = m.c, n = nx * ny * nz;
  const cloud = new Uint8Array(n), rain = new Uint8Array(n);
  const req = chartReq;
  let wmax = 0, wmin = 0, qcmax = 0, qrmax = 0;
  let col: Float32Array | null = null;
  const planeAt = new Map<number, LevelPlanes>();
  const sliceK = req?.slice ? Math.max(0, Math.min(nz - 1, req.slice.k | 0)) : -1;
  let k15 = 0; for (let k = 0; k < nz; k++) if (Math.abs(m.zc[k]! - 1500) < Math.abs(m.zc[k15]! - 1500)) k15 = k;
  let charts: ChartData | null = null, trOut: Float32Array | null = null;
  const transfer = new Set<ArrayBuffer>([cloud.buffer, rain.buffer]);
  // the GPU model of this frame (the module variable can be replaced while a readback is awaited)
  const gpu = currentGpu();
  if (gpu) {
    // GPU: packed display bytes, column composites and the few horizontal planes the diagnostics need
    const levels = [...new Set([0, k15, ...(sliceK >= 0 ? [sliceK] : [])])];
    const d = await gpu.readDisplay(levels, volMode);
    const tNow = gpu.time, sNow = gpu.steps;
    for (let i = 0; i < n; i++) { const v = d.packed[i]!; cloud[i] = v & 255; rain[i] = (v >> 8) & 255; }
    col = d.col;
    for (let c = 0; c < nx * ny; c++) { wmax = Math.max(wmax, col[NCOL * c]!); wmin = Math.min(wmin, col[NCOL * c + 1]!); qcmax = Math.max(qcmax, col[NCOL * c + 2]!); qrmax = Math.max(qrmax, col[NCOL * c + 3]!); }
    for (const [k, pl] of d.planes) {
      const o = k * m.plane; m.u.set(pl.u, o); m.v.set(pl.v, o); m.th.set(pl.th, o); m.pp.set(pl.pp, o);
      planeAt.set(k, pl);
    }
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const q = m.idx(i, j, 0); mp.rainAcc[j * nx + i] = d.rain[q]!; mp.snowAcc[j * nx + i] = d.snow[q]!; }
    m.time = tNow; m.steps = sNow;
    if (!Number.isFinite(wmax)) { running = false; post({ type: 'error', message: '數值發散 / numerical blow-up' }); }
    if (req && Number.isFinite(wmax)) charts = await buildCharts(req, col, planeAt, sliceK, gpu);
    advanceOrigin();
    // keep the GPU busy while this frame is unpacked on the CPU
    if (running) { const dtb = gpu.dt; gpu.step(gpuBatch); rateSteps += gpuBatch; advectTracers(gpuBatch * dtb, gpuBatch); }
    trOut = d.tracers ? d.tracers.slice() : null;
  } else {
    if (!Number.isFinite(m.w[m.idx(0, 0, 1)]!)) { running = false; post({ type: 'error', message: '數值發散 / numerical blow-up' }); }
    const qc = m.scalars[QC]!, qr = m.scalars[QR]!, qi = m.scalars[QI]!, qs = m.scalars[QS]!, qg = m.scalars[QG]!;
    const vm = volMode;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const q = m.idx(i, j, k), o = (k * ny + j) * nx + i;
      // cloud: liquid and ice cloud and snow (anvils are mostly snow); channel 2: precipitation (rain + graupel), updraft or vorticity
      const cl = Math.max(0, qc[q]! + qi[q]!), pr = Math.max(0, qr[q]! + qs[q]! + qg[q]!);
      cloud[o] = Math.min(255, Math.round(Math.sqrt((cl + Math.max(0, qs[q]!)) / 3e-3) * 255));
      let v2 = Math.sqrt(Math.max(0, qr[q]! + qg[q]!) / 8e-3);
      if (vm === 1) v2 = Math.sqrt(Math.max(0.5 * (m.w[q]! + m.w[q + m.plane]!), 0) / 40);
      else if (vm === 2) {
        const zeta = 0.25 * ((m.v[q + 1]! + m.v[q + 1 + m.sx]!) - (m.v[q - 1]! + m.v[q - 1 + m.sx]!)) / dx - 0.25 * ((m.u[q + m.sx]! + m.u[q + m.sx + 1]!) - (m.u[q - m.sx]! + m.u[q - m.sx + 1]!)) / m.c.dy;
        v2 = Math.sqrt(Math.max(zeta, 0) / 0.05);
      }
      rain[o] = Math.min(255, Math.round(v2 * 255));
      qcmax = Math.max(qcmax, cl); qrmax = Math.max(qrmax, pr);
      const w = m.w[q]!; wmax = Math.max(wmax, w); wmin = Math.min(wmin, w);
    }
    if (req) {
      col = columnDiagnostics(m);
      planeAt.set(0, modelPlanes(m, 0));
      if (sliceK >= 0) planeAt.set(sliceK, modelPlanes(m, sliceK));
      charts = await buildCharts(req, col, planeAt, sliceK, null);
    }
    advanceOrigin();
    trOut = tracers ? tracers.pos.slice() : null;
  }
  const g = new Float32Array(nx * ny);
  transfer.add(g.buffer);
  let vmax = 0, rainmax = 0;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const q = m.idx(i, j, 0);
    const sp = Math.hypot(0.5 * (m.u[q]! + m.u[q + 1]!), 0.5 * (m.v[q]! + m.v[q + m.sx]!));
    vmax = Math.max(vmax, sp);
    rainmax = Math.max(rainmax, mp.rainAcc[j * nx + i]!);
    g[j * nx + i] = ground === 'rain' ? mp.rainAcc[j * nx + i]! : ground === 'snow' ? mp.snowAcc[j * nx + i]! : ground === 'wind' ? sp : ground === 'none' ? 0 : m.th[q]! - m.th0[0]!;
  }
  let lo = Infinity, hi = -Infinity;
  for (const v of g) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (ground === 'theta') { const a = Math.max(Math.abs(lo), Math.abs(hi), 0.5); lo = -a; hi = a; }
  else { lo = 0; hi = Math.max(hi, ground === 'rain' || ground === 'snow' ? 5 : 10); }
  let dp: number | null = null, rmw: number | null = null, eyewalls: { r: number; v: number }[] | null = null, vtProfile: { dr: number; vt: number[] } | null = null;
  let storm: { x: number; y: number } | null = null;
  if (isTc()) {
    const r = tcMetrics(m); dp = r.pmin - dpEnv; rmw = r.rmw;
    tcRainUpdate(r.ic, r.jc);
    const ew = eyewallProfile(m); eyewalls = ew.peaks;
    vtProfile = { dr: dx, vt: ew.vt.map((x) => +x.toFixed(2)) };
    storm = { x: origin.x + (r.ic + 0.5) * dx, y: origin.y + (r.jc + 0.5) * m.c.dy };
  }
  // lowest-level vertical vorticity (cell corners) and ground-relative wind
  let zetaMax = 0, vGround = 0, iz = 0, jz = 0;
  for (let j = 1; j < ny; j++) for (let i = 1; i < nx; i++) {
    const q = m.idx(i, j, 0);
    const zeta = (m.v[q]! - m.v[q - 1]!) / dx - (m.u[q]! - m.u[q - m.sx]!) / m.c.dy;
    if (zeta > zetaMax) { zetaMax = zeta; iz = i; jz = j; }
    vGround = Math.max(vGround, Math.hypot(0.5 * (m.u[q]! + m.u[q + 1]!) + frameVel.u, 0.5 * (m.v[q]! + m.v[q + m.sx]!) + frameVel.v));
  }
  // tornado detector (grids of 500 m or finer): strongest near-surface vortex and the ground-relative wind around it
  let tornado: { zeta: number; v: number; ef: number; x: number; y: number } | null = null;
  if (dx <= 500 && zetaMax >= 0.1) {
    const rc = Math.max(1, Math.ceil(1500 / dx));
    let vt = 0;
    for (let j = Math.max(0, jz - rc); j < Math.min(ny, jz + rc); j++) for (let i = Math.max(0, iz - rc); i < Math.min(nx, iz + rc); i++) {
      if (Math.hypot(i - iz, j - jz) * dx > 1500) continue;
      const q = m.idx(i, j, 0);
      vt = Math.max(vt, Math.hypot(0.5 * (m.u[q]! + m.u[q + 1]!) + frameVel.u, 0.5 * (m.v[q]! + m.v[q + m.sx]!) + frameVel.v));
    }
    if (vt >= 29) tornado = { zeta: zetaMax, v: vt, ef: efRating(vt), x: origin.x + iz * dx, y: origin.y + jz * m.c.dy };
  }
  // storm-scale maxima from the column composites; the strongest storm column (max UH, else max w)
  let dbzMax = 0, uhMax = 0, uhMin = 0, capeMax = 0;
  if (col) {
    let cu = -1, cw = -1, wBest = 0;
    for (let c = 0; c < nx * ny; c++) {
      const o = NCOL * c;
      dbzMax = Math.max(dbzMax, col[o + C.dbz]!); capeMax = Math.max(capeMax, col[o + C.cape]!);
      if (col[o + C.uh]! > uhMax) { uhMax = col[o + C.uh]!; cu = c; }
      uhMin = Math.min(uhMin, col[o + C.uh]!);
      if (col[o + C.wmax]! > wBest) { wBest = col[o + C.wmax]!; cw = c; }
    }
    const pick = uhMax > 25 ? cu : wBest > 5 ? cw : -1;
    if (!isTc() && pick >= 0) storm = { x: origin.x + (pick % nx + 0.5) * dx, y: origin.y + (Math.floor(pick / nx) + 0.5) * m.c.dy };
  }
  if (charts) for (const a of chartArrays(charts)) transfer.add(a.buffer as ArrayBuffer);
  if (trOut) transfer.add(trOut.buffer as ArrayBuffer);
  stormDomain = storm ? { x: storm.x - origin.x, y: storm.y - origin.y } : stormDomain;
  post({ type: 'frame', time: m.time, nx, ny, nz, dx, dz, cloud, rain, ground: g, groundField: ground, groundRange: [lo, hi],
    stats: { wmax, wmin, qcmax, qrmax, rainmax, vmax, dp, rmw, eyewalls, zetaMax, vGround, dbzMax, uhMax, uhMin, capeMax, storm, vtProfile, tornado, tcRain: isTc() ? tcRain : null },
    origin: { ...origin }, charts, tracers: trOut, stepsPerSecond: rate, dt: gpu ? gpu.dt : m.c.dt }, [...transfer]);
}

/** Core (< 60 km) and outer (100-300 km) mean precipitation rates around the centre (ic, jc) of the periodic domain, and the
 *  outer area fraction above 1 mm/h, over each completed model hour (as in runTropicalCyclone.js). */
function tcRainUpdate(ic: number, jc: number): void {
  const mm = m!, acc = mp!.rainAcc, t = modelNow(), { nx, ny, dx, dy } = mm.c, Lx = nx * dx, Ly = ny * dy;
  if (!tcAcc || tcAcc.rain.length !== acc.length || t < tcAcc.t) { tcAcc = { t, rain: Float32Array.from(acc) }; tcRain = null; return; }
  if (t - tcAcc.t < 3600 - 1e-6) return;
  const f = 3600 / (t - tcAcc.t);
  let sc = 0, nc = 0, so = 0, no = 0, wet = 0;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    let x = (i - ic) * dx, y = (j - jc) * dy; x -= Math.round(x / Lx) * Lx; y -= Math.round(y / Ly) * Ly;
    const r = Math.hypot(x, y), c = j * nx + i, rate = Math.max(0, acc[c]! - tcAcc.rain[c]!) * f;
    if (r < 60000) { sc += rate; nc++; } else if (r >= 100000 && r < 300000) { so += rate; no++; if (rate > 1) wet++; }
  }
  tcRain = { core: nc ? sc / nc : 0, outer: no ? so / no : 0, wet: no ? wet / no : 0 };
  tcAcc = { t, rain: Float32Array.from(acc) };
}

function chartArrays(c: ChartData): Float32Array[] {
  const out: Float32Array[] = [];
  const add = (r: Record<string, Float32Array | undefined> | undefined | null): void => { if (r) for (const a of Object.values(r)) if (a) out.push(a); };
  add(c.maps); add(c.slice?.vars); add(c.section?.vars); add(c.sounding?.vars); add(c.rz?.vars as Record<string, Float32Array> | undefined);
  return out;
}

/** Chart data for the current state. GPU: planes and column composites from readDisplay, profiles and
 *  azimuthal means read back by small kernels; CPU: the same diagnostics from the model arrays. */
async function buildCharts(req: ChartRequest, col: Float32Array, planes: Map<number, LevelPlanes>, sliceK: number, gpu: GpuRegional | null): Promise<ChartData> {
  const mm = m!, mpp = mp!, { nx, ny, nz, dx, dy } = mm.c, t = modelNow();
  // precipitation rate (mm/h) from the accumulation change since the previous frame
  const acc = Float32Array.from(mpp.rainAcc);
  if (prevAcc && prevAcc.rain.length === acc.length && t > prevAcc.t + 1e-6) {
    const r = new Float32Array(acc.length), f = 3600 / (t - prevAcc.t);
    for (let i = 0; i < acc.length; i++) r[i] = Math.max(0, (acc[i]! - prevAcc.rain[i]!) * f);
    lastRate = r;
  } else if (!prevAcc || prevAcc.rain.length !== acc.length || t < prevAcc.t) lastRate = null;
  if (!prevAcc || t > prevAcc.t + 1e-6 || t < prevAcc.t) prevAcc = { t, rain: acc };
  const charts: ChartData = { maps: {}, slice: null, section: null, sounding: null, rz: null };
  const pl0 = planes.get(0) ?? modelPlanes(mm, 0);
  if (req.maps.length) charts.maps = compositeMaps(mm, col, pl0, { rain: mpp.rainAcc, snow: mpp.snowAcc, rate: lastRate }, req.maps, frameVel);
  if (sliceK >= 0 && req.slice) {
    const pl = planes.get(sliceK) ?? modelPlanes(mm, sliceK);
    charts.slice = { k: sliceK, z: mm.zc[sliceK]!, vars: sliceFields(mm, pl, sliceK, req.slice.vars, frameVel) };
  }
  const cellOf = (x: number, y: number): { i: number; j: number } => ({ i: Math.max(0, Math.min(nx - 1, Math.floor(x / dx))), j: Math.max(0, Math.min(ny - 1, Math.floor(y / dy))) });
  const profiles = async (pts: { i: number; j: number }[]): Promise<{ cols: Float32Array; nf: number }> =>
    gpu ? { cols: await gpu.readColumns(pts), nf: gpu.nf } : { cols: columnProfiles(mm, pts), nf: 5 + mm.scalars.length };
  const split = (all: Float32Array, np: number): Record<SectionVar, Float32Array> => {
    const r = {} as Record<SectionVar, Float32Array>;
    SECTION_VARS.forEach((v, f) => { r[v] = all.slice(f * nz * np, (f + 1) * nz * np); });
    return r;
  };
  if (req.section) {
    const { x0, y0, x1, y1 } = req.section, len = Math.hypot(x1 - x0, y1 - y0);
    const np = Math.max(2, Math.min(600, Math.round(len / dx) + 1));
    const pts = Array.from({ length: np }, (_, p) => cellOf(x0 + (x1 - x0) * p / (np - 1), y0 + (y1 - y0) * p / (np - 1)));
    const { cols, nf } = await profiles(pts);
    charts.section = { x0, y0, x1, y1, np, vars: split(sectionFromColumns(mm, cols, np, nf, frameVel), np) };
  }
  if (req.sounding) {
    const { cols, nf } = await profiles([cellOf(req.sounding.x, req.sounding.y)]);
    charts.sounding = { x: req.sounding.x, y: req.sounding.y, vars: split(sectionFromColumns(mm, cols, 1, nf, frameVel), 1) };
  }
  if (req.rz) {
    let xc: number, yc: number;
    if (req.rz === 'auto') { const r = tcMetrics(mm); xc = (r.ic + 0.5) * dx; yc = (r.jc + 0.5) * dy; }
    else { xc = req.rz.x; yc = req.rz.y; }
    const dr = dx, nr = Math.max(4, Math.min(Math.floor(Math.min(nx, ny) / 2), Math.ceil(Math.max(300000, 20 * dx) / dr)));
    const raw = gpu ? await gpu.readRZ(xc, yc, dr, nr) : azimuthalMeans(mm, xc, yc, dr, nr);
    charts.rz = { xc, yc, dr, nr, vars: unpackRZ(raw, nr, nz) };
  }
  return charts;
}
