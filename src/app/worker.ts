// Simulation worker: owns the model (CPU Float64 or WebGPU f32 backend), steps it continuously,
// and streams display frames and zonal-mean climatologies to the UI.

import { DAY, EARTH } from '../core/constants.js';
import { Dycore, GridState } from '../model/dycore.js';
import { applySpinup } from '../model/spinup.js';
import { HS_PRESETS, AQUA_PRESETS, EARTH_PRESETS, OBSERVED_QFLUX_SUFFIX, createHeldSuarez, createAquaplanet, createEarth, EarthData, MonthlyLatLon } from '../model/presets.js';
import type { GrayPhysics } from '../model/moist/aquaplanet.js';
import { ZonalMeanAccumulator } from '../model/diagnostics.js';
import { createJablonowski } from '../model/jablonowski.js';
import { GpuDycore } from '../gpu/dycoreGpu.js';
import { GpuMoist, SFC } from '../gpu/moistGpu.js';
import type { FieldId, FromWorker, ToWorker } from './protocol.js';
import type { NestPayload } from './regional/protocol.js';
import { qsat } from '../model/moist/thermo.js';

/** Everything the UI needs from one model state. Arrays are [k][lat][lon] or [lat][lon]. */
interface Snapshot {
  u: Float32Array; v: Float32Array; T: Float32Array; vor: Float32Array; div: Float32Array; ps: Float32Array;
  q: Float32Array | null; ts: Float32Array | null; olr: Float32Array | null; precipAcc: Float32Array | null; snowAcc: Float32Array | null;
  wet: Float32Array | null;   // surface evaporation efficiency (1 sea, bucket fraction on land)
  ice: Float32Array | null;   // sea-ice thickness, m
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
      ice: ph ? Float32Array.from(ph.f.ice) : null,
      wet: ph ? Float32Array.from({ length: ng }, (_, p) => ph.surface.land[p] ? Math.min(1, ph.f.bucket[p]! / (0.75 * ph.cfg.bucketMax)) : 1) : null,
    };
  }
}

class GpuBackend implements Backend {
  readonly kind = 'gpu' as const;
  constructor(readonly model: Dycore, private readonly device: GPUDevice, private readonly gd: GpuDycore, private readonly gm: GpuMoist | null, private readonly physics: GrayPhysics | null) {}
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
    let q: Float32Array | null = null, ts: Float32Array | null = null, olr: Float32Array | null = null, precipAcc: Float32Array | null = null, snowAcc: Float32Array | null = null, wet: Float32Array | null = null, ice: Float32Array | null = null;
    if (this.gm) {
      q = await this.gm.readQ();
      const s = await this.gm.readSurface();
      ts = s.slice(SFC.ts * ng, (SFC.ts + 1) * ng);
      olr = s.slice(SFC.olrNow * ng, (SFC.olrNow + 1) * ng);
      precipAcc = new Float32Array(ng);
      for (let p = 0; p < ng; p++) precipAcc[p] = s[SFC.precipConv * ng + p]! + s[SFC.precipLS * ng + p]!;
      snowAcc = s.slice(SFC.snowAcc * ng, (SFC.snowAcc + 1) * ng);
      ice = s.slice(SFC.ice * ng, (SFC.ice + 1) * ng);
      const ph = this.physics;
      if (ph) wet = Float32Array.from({ length: ng }, (_, p) => ph.surface.land[p] ? Math.min(1, s[SFC.bucket * ng + p]! / (0.75 * ph.cfg.bucketMax)) : 1);
    }
    return { u, v, T: G.slice(4 * K * ng, 5 * K * ng), vor: G.slice(2 * K * ng, 3 * K * ng), div: G.slice(3 * K * ng, 4 * K * ng), ps, q, ts, olr, precipAcc, snowAcc, wet, ice };
  }
}

let backend: Backend | null = null;
let currentPreset = '';
let physics: GrayPhysics | null = null;
const earthData: Record<string, EarthData> = {};
let spinupData: ArrayBuffer | null = null, spinupNote = '';
/** The spun-up state: the binary file, or its base64 text copy where binary files are not served. */
async function fetchSpinup(): Promise<ArrayBuffer> {
  const bin = await fetch(new URL('../../data/spinup_earth_t42q.bin', import.meta.url)).catch(() => null);
  if (bin && bin.ok && !(bin.headers.get('content-type') ?? '').includes('html')) return bin.arrayBuffer();
  const txt = await fetch(new URL('../../data/spinup_earth_t42q.b64.txt', import.meta.url));
  if (!txt.ok) throw new Error(`spin-up state not found (${txt.status})`);
  const b = atob((await txt.text()).trim()), out = new Uint8Array(b.length);
  for (let i = 0; i < b.length; i++) out[i] = b.charCodeAt(i);
  return out.buffer;
}
let qfluxData: MonthlyLatLon | null = null;
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
      currentPreset = m.preset;
      let cfg: { trunc: number; dt: number };
      let model: Dycore;
      physics = null;
      const earthName = m.preset.endsWith(OBSERVED_QFLUX_SUFFIX) ? m.preset.slice(0, -OBSERVED_QFLUX_SUFFIX.length) : m.preset;
      if (EARTH_PRESETS[earthName]) {
        cfg = EARTH_PRESETS[earthName]!;
        // T85 and finer: 512x256 orography (0.7 deg); coarser: the 128x64 set of the documented runs
        const file = cfg.trunc >= 85 ? 'earth_512.json' : 'earth_t42.json';
        if (!earthData[file]) earthData[file] = await (await fetch(new URL(`../../data/${file}`, import.meta.url))).json() as EarthData;
        let climate: { qflux?: MonthlyLatLon } = {};
        if (earthName !== m.preset) {
          if (!qfluxData) qfluxData = await (await fetch(new URL('../../data/qflux_gray_t21.json', import.meta.url))).json() as MonthlyLatLon;
          climate = { qflux: qfluxData };
        }
        const built = createEarth(EARTH_PRESETS[earthName]!, earthData[cfg.trunc >= 85 ? 'earth_512.json' : 'earth_t42.json']!, climate.qflux ? { qflux: false } : {}, climate);
        model = built.model; physics = built.physics;
        if (m.spinup && climate.qflux) {
          try {
            if (!spinupData) spinupData = await fetchSpinup();
            applySpinup(model, physics, spinupData);
          } catch (e) { spinupNote = `無法載入起轉狀態，從頭開始 / could not load the spun-up state, starting from rest: ${String(e).slice(0, 120)}`; }
        }
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
      let note = spinupNote; spinupNote = '';
      const device = m.backend === 'cpu' ? null : await getGpu();
      let gpuBackend: GpuBackend | null = null;
      if (device) {
        // Build and trial-run the GPU model; any GPU error (shader compilation, limits, out of memory)
        // or an implausible first state falls back to the CPU, with the reason in the log.
        const scopes = ['validation', 'internal', 'out-of-memory'] as const;
        for (const sc of scopes) device.pushErrorScope(sc);
        let reason = '';
        try {
          const gd = new GpuDycore(device, model, { heldSuarez: !!HS_PRESETS[m.preset] });
          const gm = physics ? new GpuMoist(device, gd, model, physics) : null;
          gd.uploadFrom(model);
          gm?.uploadFrom(model);
          const trial = new GpuBackend(model, device, gd, gm, physics);
          await trial.advance(2);
          const snap = await trial.snapshot();
          const bad = (a: Float32Array | null, lo: number, hi: number): boolean => { if (!a) return false; for (let i = 0; i < a.length; i += 7) if (!(a[i]! > lo && a[i]! < hi)) return true; return false; };
          if (bad(snap.T, 150, 350) || bad(snap.ps, 3e4, 1.2e5) || bad(snap.ts, 150, 350)) reason = 'GPU 試跑結果不合理 / implausible GPU trial state';
          gpuBackend = trial;
        } catch (e) { reason = String(e); }
        for (let i = 0; i < scopes.length; i++) { const err = await device.popErrorScope(); if (err && !reason) reason = err.message; }
        if (reason) {
          gpuBackend = null;
          note += (note ? ' · ' : '') + `WebGPU 在這個裝置上失敗，改用 CPU / WebGPU failed on this device, using the CPU: ${reason.slice(0, 300)}`;
        }
      }
      if (gpuBackend) backend = gpuBackend;
      else if (device) backend = new CpuBackend(model, physics);
      else {
        backend = new CpuBackend(model, physics);
        if (m.backend !== 'cpu') note += (note ? ' · ' : '') + 'WebGPU 不可用，改用 CPU / WebGPU unavailable, using CPU';
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
    } else if (m.type === 'snapshot') {
      if (!backend) return;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      const b = backend, md = b.model, s = await b.snapshot(), phisG = new Float64Array(md.ng);
      md.surfaceGeopotentialGrid(phisG);
      const payload: NestPayload = {
        preset: currentPreset, day: b.time() / DAY, nlat: md.tr.nlat, nlon: md.tr.nlon, K: md.K,
        lat: Float64Array.from(md.tr.lat), lon: Float64Array.from(md.tr.lon), sigma: Float64Array.from(md.lev.sigma), sigmaHalf: Float64Array.from(md.lev.sigmaHalf),
        u: s.u, v: s.v, T: s.T, ps: s.ps, q: s.q, phis: Float32Array.from(phisG), ts: s.ts, wet: s.wet,
        land: physics ? Uint8Array.from(physics.surface.land) : null,
      };
      post({ type: 'snapshot', payload });
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
      // 6-hour running mean of the rate; the first value is the instantaneous rate
      const first = !precipRate;
      if (!precipRate) precipRate = new Float32Array(ng);
      for (let p = 0; p < ng; p++) precipRate[p] = first ? (s.precipAcc[p]! - precipPrev[p]!) / dtf : precipRate[p]! * (1 - a) + a * (s.precipAcc[p]! - precipPrev[p]!) / dtf;
    }
    if (s.snowAcc && snowPrev && t > precipPrevT) {
      const dtf = t - precipPrevT, a = Math.min(1, dtf / (6 * 3600));
      const first = !snowRate;
      if (!snowRate) snowRate = new Float32Array(ng);
      for (let p = 0; p < ng; p++) snowRate[p] = first ? (s.snowAcc[p]! - snowPrev[p]!) / dtf : snowRate[p]! * (1 - a) + a * (s.snowAcc[p]! - snowPrev[p]!) / dtf;
    }
    if (!precipPrev || t > precipPrevT) { precipPrev = s.precipAcc; snowPrev = s.snowAcc; precipPrevT = t; }
  }
  const scalar = new Float32Array(ng), u = s.u.slice(o, o + ng), v = s.v.slice(o, o + ng);
  let maxWind = 0;
  for (let q = 0; q < s.u.length; q++) maxWind = Math.max(maxWind, Math.hypot(s.u[q]!, s.v[q]!));
  if ((field === 'precip' || field === 'snow' || field === 'sst' || field === 'olr' || field === 'q' || field === 'ice' || field === 'sat') && !s.ts) field = 'T';
  if (field === 'precip') { if (precipRate) for (let q = 0; q < ng; q++) scalar[q] = precipRate[q]! * 86400; }
  else if (field === 'snow') { if (snowRate) for (let q = 0; q < ng; q++) scalar[q] = snowRate[q]! * 86400; }
  else if (field === 'sst') scalar.set(s.ts!);
  else if (field === 'olr' || field === 'sat') scalar.set(s.olr!);
  else if (field === 'ice') scalar.set(s.ice!);
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
  const cloud3d = s.q ? diagnoseClouds(m, s, precipRate) : null;
  const tr: Transferable[] = [scalar.buffer, u.buffer, v.buffer];
  if (cloud3d) tr.push(cloud3d.buffer);
  post({
    type: 'frame', day: t / DAY, cloud3d, cloudNz: CLOUD_NZ, cloudTop: CLOUD_TOP, steps: b.steps(), stepsPerSecond: rate,
    nlat: m.tr.nlat, nlon: m.tr.nlon, K, lat: m.tr.lat, sigma: m.lev.sigma,
    level: k, field, scalar, u, v, maxWind, psDrift: (psMean - ps0) / ps0,
    declinationDeg: physics && physics.cfg.seasonal ? physics.declination * 180 / Math.PI : null,
  }, tr);
}

/** height levels, top (m) and the reference cloud depth D0 (m) of the renderer (globe.ts uses 2000 m too) */
const CLOUD_NZ = 24, CLOUD_TOP = 16000, CLOUD_D0 = 2000;
let zsCache: { key: Dycore; zs: Float64Array } | null = null;
/**
 * 3-D cloud for display, diagnosed like the radiation's clouds: stratiform fraction
 * ((RH - 0.8) / 0.2)^2 (Slingo 1987) per model level, plus a convective tower of fraction
 * 0.245 + 0.125 ln(P [mm/day]) (capped at 0.8) where it rains more than 2 mm/day, from the lowest
 * level up to the highest moist (RH > 0.6) level; interpolated to height levels above sea level.
 */
function diagnoseClouds(m: Dycore, s: Snapshot, rate: Float32Array | null): Uint8Array {
  const nlat = m.tr.nlat, nlon = m.tr.nlon, ng = m.ng, K = m.K, sig = m.lev.sigma;
  if (!zsCache || zsCache.key !== m) { const ph = new Float64Array(ng); m.surfaceGeopotentialGrid(ph); zsCache = { key: m, zs: ph.map((x) => Math.max(0, x / EARTH.gravity)) }; }
  const zs = zsCache.zs, out = new Uint8Array(2 * CLOUD_NZ * ng), colC = new Float64Array(CLOUD_NZ), cf = new Float64Array(K), zk = new Float64Array(K), dz = CLOUD_TOP / CLOUD_NZ;
  for (let p = 0; p < ng; p++) {
    const ps = s.ps[p]!;
    let topK = K;
    const pr = rate ? rate[p]! * 86400 : 0, cc = pr > 2 ? Math.min(0.8, 0.245 + 0.125 * Math.log(pr)) : 0;
    for (let k = K - 1; k >= 0; k--) {
      const T = s.T[k * ng + p]!, q = s.q![k * ng + p]!, pk = sig[k]! * ps;
      const rh = q / qsat(T, pk);
      // Slingo (1987) with the CCM3 critical humidities: 0.90 below 750 hPa, 0.80 above
      const rhc = pk > 7.5e4 ? 0.9 : 0.8;
      cf[k] = pk > 1e4 && rh > rhc ? Math.min(1, ((rh - rhc) / (1 - rhc)) ** 2) : 0;
      // hydrostatic height: from the surface for the lowest level, layer-mean temperature above
      zk[k] = k === K - 1 ? zs[p]! + 29.27 * T * Math.log(1 / sig[k]!) : zk[k + 1]! + 29.27 * 0.5 * (T + s.T[(k + 1) * ng + p]!) * Math.log(sig[k + 1]! / sig[k]!);
      if (cc > 0 && topK === k + 1 && (rh > 0.6 || pk > 7e4)) topK = k;
    }
    if (cc > 0) for (let k = topK; k < K; k++) cf[k] = Math.max(cf[k]!, cc);
    // total cover with maximum-random overlap (maximum within contiguous cloudy layers, random between)
    let clear = 1, blk = 0;
    for (let kk = 0; kk < K; kk++) { if (cf[kk]! > 0.01) blk = Math.max(blk, cf[kk]!); else { clear *= 1 - blk; blk = 0; } }
    const cover = 1 - clear * (1 - blk);
    // precipitation shafts: from the surface up to the convective top, or to the highest thick
    // stratiform layer (below it the falling rain / snow is visible under the cloud base)
    let rainTop = 0;
    const rp = pr > 2 ? Math.min(1, pr / 50) : 0;
    if (rp > 0) {
      let kr = cc > 0 ? topK : K;
      if (kr === K) for (let kk = 0; kk < K; kk++) if (cf[kk]! > 0.3) { kr = kk; break; }
      rainTop = kr < K ? zk[kr]! : zs[p]! + 2000;
    }
    // height levels (model levels ordered top -> bottom, z decreasing with k)
    let k = K - 1, sum = 0;
    colC.fill(0);
    for (let i = 0; i < CLOUD_NZ; i++) {
      const z = (i + 0.5) * dz;
      if (z < zs[p]!) continue;
      while (k > 0 && zk[k - 1]! < z) k--;
      let c: number;
      if (z <= zk[K - 1]!) c = cf[K - 1]!;
      else if (k === 0 && z >= zk[0]!) c = 0;
      else { const w = (z - zk[k]!) / (zk[k - 1]! - zk[k]!); c = cf[k]! + w * (cf[k - 1]! - cf[k]!); }
      colC[i] = Math.max(0, Math.min(1, c)); sum += colC[i]!;
      if (z < rainTop) out[2 * ((i * nlat + (p / nlon | 0)) * nlon + p % nlon) + 1] = Math.round(rp * 255);
    }
    // display density: keeps the vertical profile, scaled so that looking straight down through the
    // column hides exactly the overlap cover (the renderer's extinction is -ln(1 - 0.98 f) per CLOUD_D0)
    if (sum > 0 && cover > 0.005) {
      const A = -Math.log(1 - 0.98 * cover) * CLOUD_D0 / (dz * sum);
      for (let i = 0; i < CLOUD_NZ; i++) if (colC[i]! > 0) out[2 * ((i * nlat + (p / nlon | 0)) * nlon + p % nlon)] = Math.round(Math.min(1, (1 - Math.exp(-colC[i]! * A)) / 0.98) * 255);
    }
  }
  return out;
}
