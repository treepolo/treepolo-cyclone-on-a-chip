// Simulation worker: owns the model (CPU Float64 or WebGPU f32 backend), steps it continuously,
// and streams display frames and zonal-mean climatologies to the UI.

import { DAY, EARTH } from '../core/constants.js';
import { Dycore, GridState } from '../model/dycore.js';
import { HS_PRESETS, AQUA_PRESETS, EARTH_PRESETS, createHeldSuarez, createAquaplanet, createEarth, EarthData } from '../model/presets.js';
import type { GrayPhysics } from '../model/moist/aquaplanet.js';
import { ZonalMeanAccumulator } from '../model/diagnostics.js';
import { createJablonowski } from '../model/jablonowski.js';
import { GpuDycore } from '../gpu/dycoreGpu.js';
import { GpuMoist, SFC } from '../gpu/moistGpu.js';
import type { FieldId, FromWorker, ToWorker } from './protocol.js';

/** Everything the UI needs from one model state. Arrays are [k][lat][lon] or [lat][lon]. */
interface Snapshot {
  u: Float32Array; v: Float32Array; T: Float32Array; vor: Float32Array; div: Float32Array; ps: Float32Array;
  q: Float32Array | null; ts: Float32Array | null; olr: Float32Array | null; precipAcc: Float32Array | null; snowAcc: Float32Array | null;
}

interface Backend {
  readonly kind: 'cpu' | 'gpu';
  readonly model: Dycore;              // configuration (and the state itself for the CPU backend)
  time(): number;
  steps(): number;
  advance(n: number): Promise<void>;
  snapshot(): Promise<Snapshot>;
}

class CpuBackend implements Backend {
  readonly kind = 'cpu' as const;
  constructor(readonly model: Dycore, private readonly physics: GrayPhysics | null) {}
  time(): number { return this.model.time; }
  steps(): number { return this.model.steps; }
  async advance(n: number): Promise<void> { for (let i = 0; i < n; i++) this.model.step(); }
  async snapshot(): Promise<Snapshot> {
    const m = this.model, g = m.refreshGrid(), ng = m.ng, K = m.K;
    const vor = new Float32Array(K * ng), div = new Float32Array(K * ng), tmp = new Float64Array(ng);
    for (let k = 0; k < K; k++) {
      m.vorticityGrid(k, tmp); vor.set(tmp, k * ng);
      m.divergenceGrid(k, tmp); div.set(tmp, k * ng);
    }
    const ph = this.physics;
    let precipAcc: Float32Array | null = null, snowAcc: Float32Array | null = null;
    if (ph) { precipAcc = new Float32Array(ng); for (let p = 0; p < ng; p++) precipAcc[p] = ph.f.precipConv[p]! + ph.f.precipLS[p]!; snowAcc = Float32Array.from(ph.f.snowAcc); }
    return {
      u: Float32Array.from(g.u), v: Float32Array.from(g.v), T: Float32Array.from(g.T), vor, div, ps: Float32Array.from(g.ps),
      q: m.moist ? Float32Array.from(m.q) : null, ts: ph ? Float32Array.from(ph.f.sst) : null,
      olr: ph ? Float32Array.from(ph.f.olrNow) : null, precipAcc, snowAcc,
    };
  }
}

class GpuBackend implements Backend {
  readonly kind = 'gpu' as const;
  constructor(readonly model: Dycore, private readonly device: GPUDevice, private readonly gd: GpuDycore, private readonly gm: GpuMoist | null) {}
  time(): number { return this.gd.time; }
  steps(): number { return this.gd.steps; }
  async advance(n: number): Promise<void> {
    this.gd.step(n);
    await this.device.queue.onSubmittedWorkDone();
  }
  async snapshot(): Promise<Snapshot> {
    const m = this.model, K = m.K, ng = m.ng, nlat = m.tr.nlat, nlon = m.tr.nlon;
    const G = await this.gd.readGrid();            // U, V, zeta, D, T (per level), ln ps, ...
    const u = new Float32Array(K * ng), v = new Float32Array(K * ng);
    for (let k = 0; k < K; k++) for (let j = 0; j < nlat; j++) {
      const ic = 1 / m.tr.coslat[j]!;
      for (let i = 0; i < nlon; i++) { const q = k * ng + j * nlon + i; u[q] = G[q]! * ic; v[q] = G[K * ng + q]! * ic; }
    }
    const ps = new Float32Array(ng);
    for (let p = 0; p < ng; p++) ps[p] = Math.exp(G[5 * K * ng + p]!);
    let q: Float32Array | null = null, ts: Float32Array | null = null, olr: Float32Array | null = null, precipAcc: Float32Array | null = null, snowAcc: Float32Array | null = null;
    if (this.gm) {
      q = await this.gm.readQ();
      const s = await this.gm.readSurface();
      ts = s.slice(SFC.ts * ng, (SFC.ts + 1) * ng);
      olr = s.slice(SFC.olrNow * ng, (SFC.olrNow + 1) * ng);
      precipAcc = new Float32Array(ng);
      for (let p = 0; p < ng; p++) precipAcc[p] = s[SFC.precipConv * ng + p]! + s[SFC.precipLS * ng + p]!;
      snowAcc = s.slice(SFC.snowAcc * ng, (SFC.snowAcc + 1) * ng);
    }
    return { u, v, T: G.slice(4 * K * ng, 5 * K * ng), vor: G.slice(2 * K * ng, 3 * K * ng), div: G.slice(3 * K * ng, 4 * K * ng), ps, q, ts, olr, precipAcc, snowAcc };
  }
}

let backend: Backend | null = null;
let physics: GrayPhysics | null = null;
let earthData: EarthData | null = null;
let gpuDevice: GPUDevice | null = null;
let acc: ZonalMeanAccumulator | null = null;
let accFrom = 0;
let running = false;
let stepsPerTick = 4;
let field: FieldId = 'T';
let level = -1;
let ps0 = 1e5;
let busy = false;
let lastFrame = 0, lastZonal = 0, rateSteps = 0, rateT = performance.now(), rate = 0;
let lastSampleStep = 0;
// precipitation display: rate from accumulated precipitation between frames, exponentially smoothed
let precipPrev: Float32Array | null = null, precipPrevT = 0, precipRate: Float32Array | null = null;
let snowPrev: Float32Array | null = null, snowRate: Float32Array | null = null;

const post = (m: FromWorker, transfer: Transferable[] = []): void => (self as unknown as Worker).postMessage(m, transfer);

async function getGpu(): Promise<GPUDevice | null> {
  if (gpuDevice) return gpuDevice;
  const nav = (self as unknown as { navigator: Navigator }).navigator;
  if (!nav.gpu) return null;
  const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  gpuDevice = await adapter.requestDevice({ requiredLimits: {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage),
  } });
  gpuDevice.lost.then((info) => post({ type: 'error', message: `GPU 裝置遺失 / GPU device lost: ${info.message}` }));
  return gpuDevice;
}

self.onmessage = async (ev: MessageEvent<ToWorker>): Promise<void> => {
  const m = ev.data;
  try {
    if (m.type === 'init') {
      running = false;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      let cfg: { trunc: number; dt: number };
      let model: Dycore;
      physics = null;
      if (EARTH_PRESETS[m.preset]) {
        if (!earthData) earthData = await (await fetch(new URL('../../data/earth_t42.json', import.meta.url))).json() as EarthData;
        cfg = EARTH_PRESETS[m.preset]!;
        const built = createEarth(EARTH_PRESETS[m.preset]!, earthData);
        model = built.model; physics = built.physics;
      } else if (AQUA_PRESETS[m.preset]) {
        cfg = AQUA_PRESETS[m.preset]!;
        const built = createAquaplanet(AQUA_PRESETS[m.preset]!);
        model = built.model; physics = built.physics;
      } else if (m.preset === 'JW_T42') {
        cfg = { trunc: 42, dt: 900 };
        model = createJablonowski({ trunc: 42, levels: 26, dt: 900, perturb: true });
      } else {
        const hs = HS_PRESETS[m.preset];
        if (!hs) throw new Error(`unknown preset ${m.preset}`);
        cfg = hs;
        model = createHeldSuarez(hs);
      }
      let note = '';
      const device = m.backend === 'cpu' ? null : await getGpu();
      if (device) {
        const gd = new GpuDycore(device, model, { heldSuarez: !!HS_PRESETS[m.preset] });
        const gm = physics ? new GpuMoist(device, gd, model, physics) : null;
        gd.uploadFrom(model);
        gm?.uploadFrom(model);
        backend = new GpuBackend(model, device, gd, gm);
      } else {
        backend = new CpuBackend(model, physics);
        if (m.backend !== 'cpu') note = 'WebGPU 不可用，改用 CPU / WebGPU unavailable, using CPU';
      }
      acc = new ZonalMeanAccumulator(model.tr.nlat, model.tr.nlon, model.K);
      accFrom = 0; lastSampleStep = 0;
      precipPrev = null; precipRate = null; snowPrev = null; snowRate = null;
      ps0 = model.meanSurfacePressure();
      if (level < 0 || level >= model.K) level = model.K - 1;
      post({ type: 'ready', preset: m.preset, trunc: cfg.trunc, nlat: model.tr.nlat, nlon: model.tr.nlon, K: model.K, dt: cfg.dt,
        moist: physics !== null, lat: model.tr.lat, land: physics && physics.surface.land.some((x) => x === 1) ? physics.surface.land : null,
        backend: backend.kind, note });
      await sendFrame();
    } else if (m.type === 'run') {
      running = m.running;
    } else if (m.type === 'speed') {
      stepsPerTick = Math.max(1, m.stepsPerTick | 0);
    } else if (m.type === 'view') {
      field = m.field;
      level = m.level;
      if (!busy) await sendFrame();
    } else if (m.type === 'resetAverage') {
      acc?.reset();
      accFrom = backend ? backend.time() / DAY : 0;
    }
  } catch (e) {
    post({ type: 'error', message: String(e) });
  }
};

async function loop(): Promise<void> {
  for (;;) {
    if (backend && running && !busy) {
      busy = true;
      try {
        const b = backend, dt = b.model.dt;
        const sampleEvery = Math.max(1, Math.round(6 * 3600 / dt));
        const n = b.kind === 'gpu' ? stepsPerTick * 4 : stepsPerTick;
        await b.advance(n);
        rateSteps += n;
        if (b.steps() - lastSampleStep >= sampleEvery) {
          lastSampleStep = b.steps();
          const s = await b.snapshot();
          if (!Number.isFinite(s.T[0]!) || !Number.isFinite(s.ps[0]!)) {
            running = false;
            post({ type: 'error', message: `數值發散 / Numerical blow-up at day ${(b.time() / DAY).toFixed(2)}` });
          } else acc!.add(toGridState(b.model, s));
        }
        const now = performance.now();
        if (now - lastFrame > 120) await sendFrame();
      } catch (e) {
        running = false;
        post({ type: 'error', message: String(e) });
      }
      busy = false;
    }
    const now = performance.now();
    if (now - rateT > 1000) { rate = rateSteps * 1000 / (now - rateT); rateSteps = 0; rateT = now; }
    if (backend && acc && now - lastZonal > 2000 && acc.samples > 0) {
      lastZonal = now;
      const md = backend.model;
      const c = acc.result(md.tr.lat, md.lev.sigma, md.lev.sigmaHalf, EARTH);
      post({ type: 'zonal', samples: c.samples, fromDay: accFrom, lat: c.lat, sigma: c.sigma, sigmaHalf: c.sigmaHalf, u: c.u, T: c.T, psi: c.psi });
    }
    await new Promise((r) => setTimeout(r, running ? 0 : 30));
  }
}
void loop();

function toGridState(m: Dycore, s: Snapshot): GridState {
  return { nlat: m.tr.nlat, nlon: m.tr.nlon, K: m.K, mu: m.tr.mu, sigma: m.lev.sigma, sigmaHalf: m.lev.sigmaHalf,
    u: Float64Array.from(s.u), v: Float64Array.from(s.v), T: Float64Array.from(s.T), ps: Float64Array.from(s.ps) };
}

async function sendFrame(): Promise<void> {
  const b = backend;
  if (!b) return;
  lastFrame = performance.now();
  const m = b.model, s = await b.snapshot(), ng = m.ng, K = m.K, k = Math.min(level, K - 1), o = k * ng;
  const t = b.time();
  if (s.precipAcc) {
    if (precipPrev && t > precipPrevT) {
      const dtf = t - precipPrevT, a = Math.min(1, dtf / (6 * 3600));
      if (!precipRate) precipRate = new Float32Array(ng);
      for (let p = 0; p < ng; p++) precipRate[p] = precipRate[p]! * (1 - a) + a * (s.precipAcc[p]! - precipPrev[p]!) / dtf;
    }
    if (s.snowAcc && snowPrev && t > precipPrevT) {
      const dtf = t - precipPrevT, a = Math.min(1, dtf / (6 * 3600));
      if (!snowRate) snowRate = new Float32Array(ng);
      for (let p = 0; p < ng; p++) snowRate[p] = snowRate[p]! * (1 - a) + a * (s.snowAcc[p]! - snowPrev[p]!) / dtf;
    }
    if (!precipPrev || t > precipPrevT) { precipPrev = s.precipAcc; snowPrev = s.snowAcc; precipPrevT = t; }
  }
  const scalar = new Float32Array(ng), u = s.u.slice(o, o + ng), v = s.v.slice(o, o + ng);
  let maxWind = 0;
  for (let q = 0; q < s.u.length; q++) maxWind = Math.max(maxWind, Math.hypot(s.u[q]!, s.v[q]!));
  if ((field === 'precip' || field === 'snow' || field === 'sst' || field === 'olr' || field === 'q') && !s.ts) field = 'T';
  if (field === 'precip') { if (precipRate) for (let q = 0; q < ng; q++) scalar[q] = precipRate[q]! * 86400; }
  else if (field === 'snow') { if (snowRate) for (let q = 0; q < ng; q++) scalar[q] = snowRate[q]! * 86400; }
  else if (field === 'sst') scalar.set(s.ts!);
  else if (field === 'olr') scalar.set(s.olr!);
  else if (field === 'q') { for (let q = 0; q < ng; q++) scalar[q] = s.q![o + q]! * 1000; }
  else if (field === 'vor') scalar.set(s.vor.subarray(o, o + ng));
  else if (field === 'div') scalar.set(s.div.subarray(o, o + ng));
  else if (field === 'ps') { for (let q = 0; q < ng; q++) scalar[q] = s.ps[q]! / 100; }
  else {
    for (let q = 0; q < ng; q++) {
      scalar[q] = field === 'T' ? s.T[o + q]! : field === 'u' ? s.u[o + q]! : field === 'v' ? s.v[o + q]! : Math.hypot(s.u[o + q]!, s.v[o + q]!);
    }
  }
  let psMean = 0;
  for (let j = 0; j < m.tr.nlat; j++) { let r = 0; for (let i = 0; i < m.tr.nlon; i++) r += s.ps[j * m.tr.nlon + i]!; psMean += m.tr.weight[j]! * r / m.tr.nlon; }
  psMean /= 2;
  if (physics) physics.setTime(t);
  post({
    type: 'frame', day: t / DAY, steps: b.steps(), stepsPerSecond: rate,
    nlat: m.tr.nlat, nlon: m.tr.nlon, K, lat: m.tr.lat, sigma: m.lev.sigma,
    level: k, field, scalar, u, v, maxWind, psDrift: (psMean - ps0) / ps0,
    declinationDeg: physics && physics.cfg.seasonal ? physics.declination * 180 / Math.PI : null,
  }, [scalar.buffer, u.buffer, v.buffer]);
}
