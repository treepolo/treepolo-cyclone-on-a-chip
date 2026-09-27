// Regional-model worker: builds an experiment, steps it, and streams 3-D cloud / rain volumes.

import { RegionalModel } from '../../regional/core.js';
import { KesslerMicrophysics, weismanKlemp, QV, QC, QR } from '../../regional/kessler.js';
import { RegionalPhysics } from '../../regional/physics.js';
import { tropicalSounding, insertVortex, tcMetrics } from '../../regional/tropical.js';
import type { FromRegionalWorker, GroundField, RegionalExperiment, ToRegionalWorker } from './protocol.js';

let m: RegionalModel | null = null;
let mp: KesslerMicrophysics | null = null;
let experiment: RegionalExperiment = 'supercell';
let running = false, stepsPerTick = 1, ground: GroundField = 'rain';
let lastFrame = 0, rateSteps = 0, rateT = performance.now(), rate = 0;
let dpEnv = 0;

const post = (msg: FromRegionalWorker, tr: Transferable[] = []): void => (self as unknown as Worker).postMessage(msg, tr);

function build(exp: RegionalExperiment): void {
  experiment = exp;
  if (exp === 'supercell') {
    const L = 120000, dx = 2000, nx = L / dx, nz = 40, dz = 500;
    m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt: 6, nsound: 6, f: 0, beta: 0.2, divDamp: 0.1, dampDepth: 6000, dampRate: 1 / 300, kdiff2: 0 }, weismanKlemp, 3);
    mp = new KesslerMicrophysics(m);
    m.setBaseWind((z) => ({ u: 30 * Math.tanh(z / 3000) - 15, v: 0 }));
    for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) m.scalars[QV]![m.idx(i, j, k)] = m.qv0[k]!;
    const xc = L * 0.35, yc = L / 2;
    for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
      const r = Math.sqrt((((i + 0.5) * dx - xc) / 10000) ** 2 + (((j + 0.5) * dx - yc) / 10000) ** 2 + ((m.zc[k]! - 1400) / 1400) ** 2);
      if (r < 1) m.th[m.idx(i, j, k)] = m.th[m.idx(i, j, k)]! + 2 * Math.cos(0.5 * Math.PI * r) ** 2;
    }
    post({ type: 'ready', experiment: exp, nx, ny: nx, nz, dx, dz, dt: 6, description: 'Weisman–Klemp (1982) 超大胞 / supercell：暖泡在 30 m/s 低層垂直風切中觸發 / warm bubble in 30 m/s low-level shear' });
  } else {
    const L = 1200000, dx = 15000, nx = L / dx, nz = 25, dz = 1000, f = 5e-5, sst = 301.15;
    m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt: 60, nsound: 6, f, beta: 0.3, divDamp: 0.1, dampDepth: 6000, dampRate: 1 / 300, kdiff2: 0 }, tropicalSounding(sst), 3);
    mp = new KesslerMicrophysics(m);
    new RegionalPhysics(m, { lh: 0.2 * dx, lv: 100, sst, ck: 1.2e-3, radTau: 12 * 3600, radMax: 2 / 86400 });
    insertVortex(m, f, 15);
    dpEnv = 1e5 * Math.pow(m.pi0[0]!, 1004.5 / 287.05) / 100;
    post({ type: 'ready', experiment: exp, nx, ny: nx, nz, dx, dz, dt: 60, description: '熱帶氣旋 / tropical cyclone：28°C 海面上的弱渦旋（f 平面，15 km 格距）/ weak vortex over a 28 °C sea (f-plane, 15 km grid)' });
  }
}

self.onmessage = (ev: MessageEvent<ToRegionalWorker>): void => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') { running = false; build(msg.experiment); sendFrame(); }
    else if (msg.type === 'run') running = msg.running;
    else if (msg.type === 'speed') stepsPerTick = Math.max(1, msg.stepsPerTick | 0);
    else if (msg.type === 'ground') { ground = msg.field; sendFrame(); }
  } catch (e) { post({ type: 'error', message: String(e) }); }
};

function loop(): void {
  if (m && mp && running) {
    for (let s = 0; s < stepsPerTick; s++) { m.step(); mp.apply(m.c.dt); rateSteps++; }
    if (!Number.isFinite(m.w[m.idx(0, 0, 1)]!)) { running = false; post({ type: 'error', message: '數值發散 / numerical blow-up' }); }
  }
  const now = performance.now();
  if (now - rateT > 1000) { rate = rateSteps * 1000 / (now - rateT); rateSteps = 0; rateT = now; }
  if (m && running && now - lastFrame > 250) sendFrame();
  setTimeout(loop, 0);
}
loop();

function sendFrame(): void {
  if (!m || !mp) return;
  lastFrame = performance.now();
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
