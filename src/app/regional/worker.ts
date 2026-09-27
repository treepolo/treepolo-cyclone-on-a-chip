// Regional-model worker: builds an experiment, steps it, and streams 3-D cloud / rain volumes.

import { RegionalModel } from '../../regional/core.js';
import { KesslerMicrophysics, weismanKlemp, QV, QC, QR } from '../../regional/kessler.js';
import { RegionalPhysics } from '../../regional/physics.js';
import { tropicalSounding, insertVortex, tcMetrics } from '../../regional/tropical.js';
import { GpuRegional } from '../../gpu/regionalGpu.js';
import type { RegionalPhysicsConfig } from '../../regional/physics.js';
import type { FromRegionalWorker, GroundField, RegionalExperiment, ToRegionalWorker } from './protocol.js';

let m: RegionalModel | null = null;
let mp: KesslerMicrophysics | null = null;
let experiment: RegionalExperiment = 'supercell';
let running = false, stepsPerTick = 1, ground: GroundField = 'rain';
let lastFrame = 0, rateSteps = 0, rateT = performance.now(), rate = 0;
let dpEnv = 0;
let gpu: GpuRegional | null = null;
let gpuDevice: GPUDevice | null = null;
let busy = false;
let physCfg: RegionalPhysicsConfig | null = null;

async function getGpu(): Promise<GPUDevice | null> {
  if (gpuDevice) return gpuDevice;
  const nav = (self as unknown as { navigator: Navigator }).navigator;
  if (!nav.gpu) return null;
  const ad = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!ad) return null;
  gpuDevice = await ad.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize, maxBufferSize: ad.limits.maxBufferSize } });
  return gpuDevice;
}

const post = (msg: FromRegionalWorker, tr: Transferable[] = []): void => (self as unknown as Worker).postMessage(msg, tr);

function build(exp: RegionalExperiment): { dt: number; description: string } {
  experiment = exp;
  physCfg = null;
  if (exp === 'supercell' || exp === 'supercell_hr') {
    const hr = exp === 'supercell_hr';
    const L = 120000, dx = hr ? 1000 : 2000, nx = L / dx, nz = hr ? 60 : 40, dz = hr ? 333.3333 : 500, dtm = hr ? 3 : 6;
    m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt: dtm, nsound: 6, f: 0, beta: 0.2, divDamp: 0.1, dampDepth: 6000, dampRate: 1 / 300, kdiff2: 0 }, weismanKlemp, 3);
    mp = new KesslerMicrophysics(m);
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
    m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt: dtm, nsound: 6, f, beta: 0.3, divDamp: 0.1, dampDepth: 6000, dampRate: 1 / 300, kdiff2: 0 }, tropicalSounding(sst), 3);
    mp = new KesslerMicrophysics(m);
    physCfg = { lh: hr ? 1000 : 0.2 * dx, lv: 100, sst, ck: 1.2e-3, radTau: 12 * 3600, radMax: 2 / 86400 };
    new RegionalPhysics(m, physCfg);
    insertVortex(m, f, 15);
    dpEnv = 1e5 * Math.pow(m.pi0[0]!, 1004.5 / 287.05) / 100;
    return { dt: dtm, description: `熱帶氣旋 / tropical cyclone：28°C 海面上的弱渦旋（f 平面，Δx ${dx / 1000} km）/ weak vortex over a 28 °C sea (f-plane)` };
  }
}

self.onmessage = async (ev: MessageEvent<ToRegionalWorker>): Promise<void> => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      running = false;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      const info = build(msg.experiment);
      gpu = null;
      let note = '';
      const device = msg.backend === 'cpu' ? null : await getGpu();
      if (device) { gpu = new GpuRegional(device, m!, { moist: true, physics: physCfg }); gpu.uploadFrom(m!); }
      else if (msg.backend !== 'cpu') note = 'WebGPU 不可用，改用 CPU / WebGPU unavailable, using CPU';
      if (!gpu && (msg.experiment === 'tc_hr' || msg.experiment === 'supercell_hr')) note += (note ? ' · ' : '') + '此高解析實驗在 CPU 上非常慢 / this high-resolution experiment is very slow on the CPU';
      const c = m!.c;
      post({ type: 'ready', experiment: msg.experiment, nx: c.nx, ny: c.ny, nz: c.nz, dx: c.dx, dz: c.dz, dt: info.dt, description: info.description, backend: gpu ? 'gpu' : 'cpu', note });
      await sendFrame();
    }
    else if (msg.type === 'run') running = msg.running;
    else if (msg.type === 'speed') stepsPerTick = Math.max(1, msg.stepsPerTick | 0);
    else if (msg.type === 'ground') { ground = msg.field; sendFrame(); }
  } catch (e) { post({ type: 'error', message: String(e) }); }
};

async function loop(): Promise<void> {
  for (;;) {
    if (m && mp && running && !busy) {
      busy = true;
      try {
        if (gpu) { gpu.step(stepsPerTick * 2); rateSteps += stepsPerTick * 2; await gpu.device.queue.onSubmittedWorkDone(); }
        else for (let s = 0; s < stepsPerTick; s++) { m.step(); mp.apply(m.c.dt); rateSteps++; }
        if (performance.now() - lastFrame > 300) await sendFrame();
      } catch (e) { running = false; post({ type: 'error', message: String(e) }); }
      busy = false;
    }
    const now = performance.now();
    if (now - rateT > 1000) { rate = rateSteps * 1000 / (now - rateT); rateSteps = 0; rateT = now; }
    await new Promise((r) => setTimeout(r, running ? 0 : 30));
  }
}
void loop();

/** Copy the GPU state into the CPU model arrays (display and diagnostics reuse the CPU code). */
async function syncFromGpu(): Promise<void> {
  if (!gpu || !m || !mp) return;
  const st = await gpu.readState(), size = m.size;
  const arrs = [m.u, m.v, m.w, m.th, m.pp, m.scalars[0]!, m.scalars[1]!, m.scalars[2]!];
  arrs.forEach((a, f) => { for (let i = 0; i < size; i++) a[i] = st[f * size + i]!; });
  const rain = await gpu.readRain();
  for (let j = 0; j < m.c.ny; j++) for (let i = 0; i < m.c.nx; i++) mp.rainAcc[j * m.c.nx + i] = rain[m.idx(i, j, 0)]!;
  m.time = gpu.time;
}

async function sendFrame(): Promise<void> {
  if (!m || !mp) return;
  lastFrame = performance.now();
  await syncFromGpu();
  if (!Number.isFinite(m.w[m.idx(0, 0, 1)]!)) { running = false; post({ type: 'error', message: '數值發散 / numerical blow-up' }); }
  const { nx, ny, nz, dx, dz } = m.c, n = nx * ny * nz;
  const cloud = new Uint8Array(n), rain = new Uint8Array(n);
  const qc = m.scalars[QC]!, qr = m.scalars[QR]!;
  let wmax = 0, wmin = 0, qcmax = 0, qrmax = 0;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const q = m.idx(i, j, k), o = (k * ny + j) * nx + i;
    cloud[o] = Math.min(255, Math.round(Math.sqrt(Math.max(0, qc[q]!) / 3e-3) * 255));
    rain[o] = Math.min(255, Math.round(Math.sqrt(Math.max(0, qr[q]!) / 8e-3) * 255));
    qcmax = Math.max(qcmax, qc[q]!); qrmax = Math.max(qrmax, qr[q]!);
    const w = m.w[q]!; wmax = Math.max(wmax, w); wmin = Math.min(wmin, w);
  }
  const g = new Float32Array(nx * ny);
  let vmax = 0, rainmax = 0;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const q = m.idx(i, j, 0);
    const sp = Math.hypot(0.5 * (m.u[q]! + m.u[q + 1]!), 0.5 * (m.v[q]! + m.v[q + m.sx]!));
    vmax = Math.max(vmax, sp);
    rainmax = Math.max(rainmax, mp.rainAcc[j * nx + i]!);
    g[j * nx + i] = ground === 'rain' ? mp.rainAcc[j * nx + i]! : ground === 'wind' ? sp : m.th[q]! - m.th0[0]!;
  }
  let lo = Infinity, hi = -Infinity;
  for (const v of g) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (ground === 'theta') { const a = Math.max(Math.abs(lo), Math.abs(hi), 0.5); lo = -a; hi = a; }
  else { lo = 0; hi = Math.max(hi, ground === 'rain' ? 5 : 10); }
  let dp: number | null = null, rmw: number | null = null;
  if (experiment === 'tc') { const r = tcMetrics(m); dp = r.pmin - dpEnv; rmw = r.rmw; }
  post({ type: 'frame', time: m.time, nx, ny, nz, dx, dz, cloud, rain, ground: g, groundField: ground, groundRange: [lo, hi],
    stats: { wmax, wmin, qcmax, qrmax, rainmax, vmax, dp, rmw }, stepsPerSecond: rate }, [cloud.buffer, rain.buffer, g.buffer]);
}
