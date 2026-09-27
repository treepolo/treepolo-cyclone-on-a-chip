// Simulation worker: owns the model, steps it continuously, streams snapshots to the UI.

import { DAY, EARTH } from '../core/constants.js';
import { Dycore } from '../model/dycore.js';
import { HS_PRESETS, AQUA_PRESETS, createHeldSuarez, createAquaplanet } from '../model/presets.js';
import type { GrayAquaplanet } from '../model/moist/aquaplanet.js';
import { ZonalMeanAccumulator } from '../model/diagnostics.js';
import { createJablonowski } from '../model/jablonowski.js';
import type { FieldId, FromWorker, ToWorker } from './protocol.js';

let model: Dycore | null = null;
let physics: GrayAquaplanet | null = null;
/** precipitation rate smoothed over ~6 model hours (kg m^-2 s^-1) */
let precipSmooth = new Float64Array(0);
let acc: ZonalMeanAccumulator | null = null;
let accFrom = 0;
let running = false;
let stepsPerTick = 4;
let field: FieldId = 'T';
let level = -1;
let ps0 = 1e5;
let lastFrame = 0, lastZonal = 0, rateSteps = 0, rateT = performance.now(), rate = 0;

const post = (m: FromWorker, transfer: Transferable[] = []): void => (self as unknown as Worker).postMessage(m, transfer);

self.onmessage = (ev: MessageEvent<ToWorker>): void => {
  const m = ev.data;
  try {
    if (m.type === 'init') {
      let cfg: { trunc: number; dt: number };
      physics = null;
      if (AQUA_PRESETS[m.preset]) {
        cfg = AQUA_PRESETS[m.preset]!;
        const built = createAquaplanet(AQUA_PRESETS[m.preset]!);
        model = built.model;
        physics = built.physics;
        precipSmooth = new Float64Array(model.ng);
      } else if (m.preset === 'JW_T42') {
        cfg = { trunc: 42, dt: 900 };
        model = createJablonowski({ trunc: 42, levels: 26, dt: 900, perturb: true });
      } else {
        const hs = HS_PRESETS[m.preset];
        if (!hs) throw new Error(`unknown preset ${m.preset}`);
        cfg = hs;
        model = createHeldSuarez(hs);
      }
      acc = new ZonalMeanAccumulator(model.tr.nlat, model.tr.nlon, model.K);
      accFrom = 0;
      ps0 = model.meanSurfacePressure();
      if (level < 0 || level >= model.K) level = model.K - 1;
      post({ type: 'ready', preset: m.preset, trunc: cfg.trunc, nlat: model.tr.nlat, nlon: model.tr.nlon, K: model.K, dt: cfg.dt, moist: physics !== null });
      sendFrame();
    } else if (m.type === 'run') {
      running = m.running;
    } else if (m.type === 'speed') {
      stepsPerTick = Math.max(1, m.stepsPerTick | 0);
    } else if (m.type === 'view') {
      field = m.field;
      level = m.level;
      sendFrame();
    } else if (m.type === 'resetAverage') {
      acc?.reset();
      accFrom = model ? model.time / DAY : 0;
    }
  } catch (e) {
    post({ type: 'error', message: String(e) });
  }
};

function loop(): void {
  if (model && running) {
    const sampleEvery = Math.max(1, Math.round(6 * 3600 / model.dt));
    for (let i = 0; i < stepsPerTick; i++) {
      model.step();
      rateSteps++;
      if (physics) {
        const a = Math.min(1, model.dt / (6 * 3600)), pr = physics.f.precipRate;
        for (let q = 0; q < pr.length; q++) precipSmooth[q] = precipSmooth[q]! * (1 - a) + pr[q]! * a;
      }
      if (model.steps % sampleEvery === 0) acc!.add(model.refreshGrid());
    }
    if (!model.isFinite()) {
      running = false;
      post({ type: 'error', message: `數值發散 / Numerical blow-up at day ${(model.time / DAY).toFixed(2)}` });
    }
  }
  const now = performance.now();
  if (now - rateT > 1000) { rate = rateSteps * 1000 / (now - rateT); rateSteps = 0; rateT = now; }
  if (model && running && now - lastFrame > 100) sendFrame();
  if (model && now - lastZonal > 2000 && acc && acc.samples > 0) {
    lastZonal = now;
    const c = acc.result(model.tr.lat, model.lev.sigma, model.lev.sigmaHalf, EARTH);
    post({ type: 'zonal', samples: c.samples, fromDay: accFrom, lat: c.lat, sigma: c.sigma, sigmaHalf: c.sigmaHalf, u: c.u, T: c.T, psi: c.psi });
  }
  setTimeout(loop, 0);
}
loop();

function sendFrame(): void {
  if (!model) return;
  lastFrame = performance.now();
  const g = model.refreshGrid(), ng = model.ng, k = Math.min(level, model.K - 1);
  const o = k * ng;
  const scalar = new Float32Array(ng), u = new Float32Array(ng), v = new Float32Array(ng);
  let maxWind = 0;
  for (let q = 0; q < g.u.length; q++) maxWind = Math.max(maxWind, Math.hypot(g.u[q]!, g.v[q]!));
  for (let q = 0; q < ng; q++) { u[q] = g.u[o + q]!; v[q] = g.v[o + q]!; }
  if ((field === 'precip' || field === 'sst' || field === 'olr' || field === 'q') && !physics) field = 'T';
  if (field === 'precip' || field === 'sst' || field === 'olr') {
    const src = field === 'precip' ? precipSmooth : field === 'sst' ? physics!.f.sst : physics!.f.olrNow;
    const f = field === 'precip' ? 86400 : 1;
    for (let q = 0; q < ng; q++) scalar[q] = src[q]! * f;
  } else if (field === 'q') {
    for (let q = 0; q < ng; q++) scalar[q] = model.q[o + q]! * 1000;
  } else if (field === 'vor' || field === 'div') {
    const tmp = new Float64Array(ng);
    if (field === 'vor') model.vorticityGrid(k, tmp); else model.divergenceGrid(k, tmp);
    for (let q = 0; q < ng; q++) scalar[q] = tmp[q]!;
  } else {
    for (let q = 0; q < ng; q++) {
      scalar[q] = field === 'T' ? g.T[o + q]! : field === 'u' ? g.u[o + q]! : field === 'v' ? g.v[o + q]!
        : field === 'speed' ? Math.hypot(g.u[o + q]!, g.v[o + q]!) : g.ps[q]! / 100;
    }
  }
  post({
    type: 'frame', day: model.time / DAY, steps: model.steps, stepsPerSecond: rate,
    nlat: model.tr.nlat, nlon: model.tr.nlon, K: model.K, lat: model.tr.lat, sigma: model.lev.sigma,
    level: k, field, scalar, u, v, maxWind, psDrift: (model.meanSurfacePressure() - ps0) / ps0,
  }, [scalar.buffer, u.buffer, v.buffer]);
}
