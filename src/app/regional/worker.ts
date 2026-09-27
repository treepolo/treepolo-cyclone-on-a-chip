// Regional-model worker: builds an experiment, steps it, and streams 3-D cloud / rain volumes.

import { RegionalModel } from '../../regional/core.js';
import { weismanKlemp } from '../../regional/kessler.js';
import { IceMicrophysics, QV, QC, QR, QI, QS, QG } from '../../regional/ice.js';
import { RegionalPhysics } from '../../regional/physics.js';
import { tropicalSounding, insertVortex, tcMetrics, eyewallProfile } from '../../regional/tropical.js';
import { GpuRegional } from '../../gpu/regionalGpu.js';
import type { RegionalPhysicsConfig } from '../../regional/physics.js';
import { nestFromGlobal, nestTargets, sampleSurface, NestSpec } from '../../regional/nest.js';
import { tornadoExperiment, StormTracker } from '../../regional/supercell.js';
import type { FromRegionalWorker, GroundField, NestPayload, NestSize, RegionalExperiment, ToRegionalWorker } from './protocol.js';

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

async function getGpu(): Promise<GPUDevice | null> {
  if (gpuDevice) return gpuDevice;
  const nav = (self as unknown as { navigator: Navigator }).navigator;
  if (!nav.gpu) return null;
  const ad = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!ad) return null;
  gpuDevice = await ad.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize, maxBufferSize: ad.limits.maxBufferSize } });
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
  if (exp === 'tornado') {
    // GPU: 250 m LES over 50 km; CPU: a 500 m preview over 40 km (mesocyclone scale only)
    const e = gpuOk ? tornadoExperiment(250, 50000, 64, 250, 2, 8) : tornadoExperiment(500, 40000, 40, 400, 3, 6);
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
      let info = build(msg.experiment, !!device);
      gpu = null;
      let note = '';
      if (device) {
        const reason = await tryGpu(device);
        if (reason) { note = gpuFailNote(reason); info = build(msg.experiment, false); }
      } else if (msg.backend !== 'cpu') note = 'WebGPU 不可用，改用 CPU / WebGPU unavailable, using CPU';
      if (!gpu && (msg.experiment === 'tc_hr' || msg.experiment === 'supercell_hr' || msg.experiment === 'tornado')) note += (note ? ' · ' : '') + '此高解析實驗在 CPU 上非常慢 / this high-resolution experiment is very slow on the CPU';
      const c = m!.c;
      post({ type: 'ready', land: null, experiment: msg.experiment, nx: c.nx, ny: c.ny, nz: c.nz, dx: c.dx, dz: c.dz, dt: info.dt, description: info.description, backend: gpu ? 'gpu' : 'cpu', note });
      await sendFrame();
    }
    else if (msg.type === 'initNest') {
      running = false;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      const device = msg.backend === 'cpu' ? null : await getGpu();
      let info = buildNest(msg.payload, msg.lat0, msg.lon0, msg.size, !!device);
      gpu = null;
      let note = '';
      if (device) {
        const reason = await tryGpu(device);
        if (reason) { note = gpuFailNote(reason); info = buildNest(msg.payload, msg.lat0, msg.lon0, msg.size, false); }
      } else note = (msg.backend !== 'cpu' ? 'WebGPU 不可用，改用 CPU / WebGPU unavailable, using CPU · ' : '') + 'CPU 使用較粗的網格 / the CPU uses a coarser grid';
      const c = m!.c;
      post({ type: 'ready', experiment: 'nest', nx: c.nx, ny: c.ny, nz: c.nz, dx: c.dx, dz: c.dz, dt: info.dt, description: info.description, backend: gpu ? 'gpu' : 'cpu', note, land: info.land });
      await sendFrame();
    }
    else if (msg.type === 'run') running = msg.running;
    else if (msg.type === 'speed') stepsPerTick = Math.max(1, msg.stepsPerTick | 0);
    else if (msg.type === 'ground') { ground = msg.field; sendFrame(); }
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
          gpu.step(gpuBatch); rateSteps += gpuBatch;
          await gpu.device.queue.onSubmittedWorkDone();
          const el = performance.now() - t0, target = 60 * stepsPerTick;
          if (DEBUG) console.log(`DBG batch ${gpuBatch} steps ${el.toFixed(0)} ms`);
          gpuBatch = Math.max(1, Math.min(2000, Math.round(gpuBatch * Math.min(2, Math.max(0.5, target / Math.max(el, 1))))));
        }
        else for (let s = 0; s < stepsPerTick; s++) { m.step(); mp.apply(m.c.dt); rateSteps++; }
        if (tracker && m.time - lastTrack >= 600) { lastTrack = m.time; await followStorm(); }
        if (performance.now() - lastFrame > (gpu ? 500 : 300)) { const tf = performance.now(); await sendFrame(); if (DEBUG) console.log(`DBG frame ${(performance.now() - tf).toFixed(0)} ms`); }
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
    const device = gpu.device;
    gpu.destroy();
    gpu = new GpuRegional(device, m, { moist: true, physics: physCfg, ice: true });
    gpu.uploadFrom(m, { rain: mp.rainAcc, snow: mp.snowAcc });
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
  lastFrame = performance.now();
  const { nx, ny, nz, dx, dz } = m.c, n = nx * ny * nz;
  const cloud = new Uint8Array(n), rain = new Uint8Array(n);
  let wmax = 0, wmin = 0, qcmax = 0, qrmax = 0;
  if (gpu) {
    // GPU: packed display bytes, column extremes and the few horizontal planes the diagnostics need
    let k15 = 0; for (let k = 0; k < nz; k++) if (Math.abs(m.zc[k]! - 1500) < Math.abs(m.zc[k15]! - 1500)) k15 = k;
    const d = await gpu.readDisplay(k15 === 0 ? [0] : [0, k15]);
    for (let i = 0; i < n; i++) { const v = d.packed[i]!; cloud[i] = v & 255; rain[i] = (v >> 8) & 255; }
    for (let c = 0; c < nx * ny; c++) { wmax = Math.max(wmax, d.col[4 * c]!); wmin = Math.min(wmin, d.col[4 * c + 1]!); qcmax = Math.max(qcmax, d.col[4 * c + 2]!); qrmax = Math.max(qrmax, d.col[4 * c + 3]!); }
    for (const [k, pl] of d.planes) { const o = k * m.plane; m.u.set(pl.u, o); m.v.set(pl.v, o); m.th.set(pl.th, o); m.pp.set(pl.pp, o); }
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const q = m.idx(i, j, 0); mp.rainAcc[j * nx + i] = d.rain[q]!; mp.snowAcc[j * nx + i] = d.snow[q]!; }
    m.time = gpu.time; m.steps = gpu.steps;
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
    stats: { wmax, wmin, qcmax, qrmax, rainmax, vmax, dp, rmw, eyewalls, zetaMax, vGround }, stepsPerSecond: rate }, [cloud.buffer, rain.buffer, g.buffer]);
}
