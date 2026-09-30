// Regional-model worker: builds an experiment, steps it, and streams 3-D cloud / rain volumes.

import { RegionalModel } from '../../regional/core.js';
import { IceMicrophysics, QV } from '../../regional/ice.js';
import { RegionalPhysics } from '../../regional/physics.js';
import { tcMetrics, eyewallProfile } from '../../regional/tropical.js';
import { GpuRegional } from '../../gpu/regionalGpu.js';
import type { RegionalPhysicsConfig } from '../../regional/physics.js';
import { nestFromGlobal, nestTargets, sampleSurface, NestSpec } from '../../regional/nest.js';
import { refineInto, coarsenInto } from '../../regional/refine.js';
import { Tracers, type TracerParams } from '../../regional/tracers.js';
import { AxiDriver, AXI_DEFAULTS, LEGACY_TC, type AxiParams } from './axiDriver.js';
import { buildModel, setupFromLegacy, axiFromSetup, refinedSetup, coarsenedSetup } from './build.js';
import { presetById, setupOf, tcLike, type RegionalSetup } from './setup.js';
import { packSave, unpackSave, type SaveArrays } from '../saves.js';
import { Pacer } from '../pacer.js';
import { StormTracker, type TornadoEnv } from '../../regional/supercell.js';
import { StormCatalog, findStorms, type StormNow } from '../../regional/storms.js';
import { applyWind, MAX_FORCINGS, type WindForcing } from '../../regional/forcing.js';
import { volumeBytes } from '../../regional/display.js';
import { EyeNest, syncModel } from './eyeNest.js';
import { type ChartData, type ChartRequest, type ForcingInfo, type FromRegionalWorker, type GroundField, type NestFrame, type NestInfo, type NestPayload, type NestSize, type RegionalExperiment, type TcRain, type ToRegionalWorker } from './protocol.js';
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
/** every storm of the run with its identity and numbers; the storm the domain follows (id) */
const catalog = new StormCatalog();
let followId: number | null = null;
const followed = (): StormNow | null => (followId === null ? null : catalog.active.find((s) => s.id === followId) ?? null);
/** the main storm: the followed one, else the catalogue's strongest */
const mainStorm = (): StormNow | null => followed() ?? catalog.main();
/** lasting wind forcings (forcing.ts): what the page shows and what the model applies */
let forcings: { info: ForcingInfo; f: WindForcing }[] = [], forcingId = 1;
/** Hand the lasting forcings to the models (the CPU hook reads the list; the GPU keeps a table) and to the page. */
function syncForcings(): void {
  gpu?.setForcings(forcings.map((x) => x.f));
  eye?.setForcings(forcings.map((x) => x.f));
  post({ type: 'forcings', list: forcings.map((x) => x.info) });
}
/** Drop forcings that have ended; returns true when the list changed. */
function expireForcings(): void {
  const t = modelNow(), n = forcings.length;
  forcings = forcings.filter((x) => x.info.until === null || x.info.until > t);
  if (forcings.length !== n) syncForcings();
}
/** Forget every forcing (new model, refinement to another grid, restored state). */
function clearForcings(): void { if (forcings.length) { forcings = []; syncForcings(); } }
/** The CPU model applies the lasting forcings at the start of each step (after the physics' own pre-step work). */
function hookForcing(model: RegionalModel): void {
  const prev = model.preStep;
  model.preStep = (mm): void => { prev?.(mm); if (forcings.length) applyWind(mm, forcings.map((x) => x.f), mm.c.dt); };
}

/**
 * Coarser states kept by refinements (newest last, at most 3): coarsening continues on the kept grid with the fine
 * run averaged into it. `box`: the refinement cut a smaller domain (the eye box, the 250 m tornado box), so only that
 * part is replaced (blended over its relaxation zone) and the rest is the state at the time of the refinement.
 */
interface Parent {
  setup: RegionalSetup; arrays: Float32Array[]; rain: Float32Array; snow: Float32Array; time: number; steps: number;
  frame: { u: number; v: number }; origin: { x: number; y: number }; dpEnv: number; tsk: Float64Array | null; wet: Float64Array | null;
  bnd: { u: Float64Array; v: Float64Array; th: Float64Array; qv: Float64Array | null; pp: Float64Array | null } | null; box: boolean;
}
let parents: Parent[] = [];
/** Keep the current (CPU-synchronised) state as the parent of a refinement. */
function keepParent(box: boolean): void {
  const mm = m!, sf = phys?.surface ?? null, b = mm.boundary;
  parents.push({ setup: { ...setup! }, arrays: [mm.u, mm.v, mm.w, mm.th, mm.pp, ...mm.scalars].map((a) => Float32Array.from(a)), rain: Float32Array.from(mp!.rainAcc), snow: Float32Array.from(mp!.snowAcc),
    time: mm.time, steps: mm.steps, frame: { ...frameVel }, origin: { ...origin }, dpEnv, tsk: sf ? Float64Array.from(sf.tsk) : null, wet: sf ? Float64Array.from(sf.wet) : null,
    bnd: b ? { u: Float64Array.from(b.u), v: Float64Array.from(b.v), th: Float64Array.from(b.th), qv: b.qv ? Float64Array.from(b.qv) : null, pp: b.pp ? Float64Array.from(b.pp) : null } : null, box });
  if (parents.length > 3) parents.shift();
}
/** Grid changes available now: refine to (dx), coarsen to (dx; `back`: a kept coarser state), the eye nest. */
function gridOptions(): { refineTo: number | null; coarsenTo: number | null; coarsenBack: boolean; nestOk: boolean } {
  if (!setup || axi || experiment === 'nest' || !m) return { refineTo: null, coarsenTo: null, coarsenBack: false, nestOk: false };
  const p = parents[parents.length - 1];
  return { refineTo: refinedSetup(setup)?.dx ?? null, coarsenTo: p ? p.setup.dx : coarsenedSetup(setup)?.dx ?? null, coarsenBack: !!p, nestOk: true };
}

/** the eye nest: a finer grid in a cylinder at the domain centre, coupled both ways (eyeNest.ts), or null */
let eye: EyeNest | null = null;
function nestInfo(): NestInfo | null {
  if (!eye) return null;
  const g = eye.g;
  return { R: g.R, dx: g.dx, dz: g.dz, r: g.r, rz: g.rz, nx: g.nx, nz: g.nz, cells: eye.cells, nsub: eye.nsub, x0: eye.x0, y0: eye.y0, L: g.nx * g.dx, cx: g.cx, cy: g.cy, Wf: g.Wf };
}
/** Drop the eye nest (the outer grid keeps what it was fed back). */
function stopNest(): void {
  if (!eye) return;
  eye.destroy(); eye = null;
  post({ type: 'nest', info: null });
}
/** Advance the models by n outer steps on the GPU (with the eye nest: its sub-steps and feedback in each). */
function stepGpu(n: number): void {
  const e = eye, link = e?.link;
  if (e && link) for (let s = 0; s < n; s++) link.step(e.nsub);
  else gpu!.step(n);
}
/** One outer step on the CPU (with the eye nest's sub-steps and feedback). */
function stepCpu(): void {
  if (eye) eye.cpuStep(m!, mp!);
  else { m!.step(); mp!.apply(m!.c.dt); }
}

/** One storm analysis of the current CPU model state (level-0 fields; column composites for cells). */
function analyseStorms(col: Float32Array | null): void {
  if (!m) return;
  const { nx, ny, dx, dy } = m.c;
  if (!col && Math.min(nx * dx, ny * dy) < 400000) col = columnDiagnostics(m);
  catalog.update(m.time, findStorms(m, col, frameVel), origin, m.c.lateral === 'open' ? null : { x: nx * dx, y: ny * dy });
}
let nestSpec: NestSpec | null = null;
/** how the current model was built (needed to rebuild it from a save) */
let builtGpu = false;
// pacing and display cadence
const pacer = new Pacer();
let frameMode: { kind: 'wall' | 'model' | 'fast'; every: number } = { kind: 'wall', every: 0 };
let lastFrameModel = 0;
// the axisymmetric tropical-cyclone experiment runs through its own driver (CPU, revolved for display)
let axi: AxiDriver | null = null;
/** the set-up of the current run (null: nest in the global model) */
let setup: RegionalSetup | null = null;
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
/** draw sub-grid cloud (display.ts) in the 3-D view; the cumulus scheme's rain rate per column (mm/h) of the last frame, null without it */
let subgrid = true, cuRate: Float32Array | null = null;
// ground-relative position of the domain origin: advances with the frame velocity, jumps with rolls and refinement
let origin = { x: 0, y: 0 }, originT = 0;
function advanceOrigin(): void {
  const t = modelNow();
  origin.x += frameVel.u * (t - originT); origin.y += frameVel.v * (t - originT);
  originT = t;
}
function resetOrigin(x = 0, y = 0): void { origin = { x, y }; originT = modelNow(); prevAcc = null; tcAcc = null; tcRain = null; undo = null; envDu = 0; }
// precipitation rate from the change of the accumulation between frames
let prevAcc: { t: number; rain: Float32Array } | null = null, lastRate: Float32Array | null = null;
const currentGpu = (): GpuRegional | null => gpu;
const isTc = (): boolean => !axi && !!setup && experiment !== 'nest' && tcLike(setup);
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
  tc_hr: '熱帶氣旋 5 km / tropical cyclone 5 km', tc_3: '熱帶氣旋 3 km / tropical cyclone 3 km', tornado: '龍捲超大胞 250 m / tornadic supercell 250 m', tornado_c: '龍捲超大胞 1 km / tornadic supercell 1 km', nest: '巢狀區域 / nest', tc_axi: '軸對稱颱風 / axisymmetric TC', custom: '自訂 / custom',
};

// Adaptive time step (GPU): dt = min(acoustic limit, CFL_TARGET / max(|u|/dx + |v|/dy + |w|/dz)),
// between the configured dt0 and 3 dt0; shrinks at once, grows by at most 10 % per check. The
// acoustic limit keeps the horizontal sound Courant number of the split steps c_s dt / (n_s dx) <= 0.45.
const CFL_TARGET = 0.8, CFL_MAX = 1.1;
let adaptive = true, dt0 = 0;
// Safety net for interactions: the state just before the last change (float32 copies, models up to 4 M cells) and the
// model time of the change; a blow-up within a model hour of it restores this state and pauses.
let undo: { t: number; arrays: Float32Array[]; rain: Float32Array; snow: Float32Array; ub: Float64Array; vb: Float64Array; bu: Float64Array | null; bqv: Float64Array | null; envDu: number } | null = null;
/** accumulated wind change of 'environment' interactions at 6 km (bounded to +-ENV_DU_MAX) */
let envDu = 0;
const ENV_DU_MAX = 30;
/** largest plausible |w| (m/s): beyond it the state is treated as numerically unstable */
const W_BLOWUP = 200;
/** CPU adaptive time step: the same Courant rule as the GPU, checked every tick */
function cpuAdaptDt(): void {
  if (!m || gpu || !adaptive) return;
  const { nx, ny, nz, dx, dy, dz } = m.c;
  let rate = 0;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const q = m.idx(i, j, k);
    rate = Math.max(rate, Math.abs(m.u[q]!) / dx + Math.abs(m.v[q]!) / dy + Math.abs(m.w[q]!) / dz);
  }
  if (!Number.isFinite(rate)) return;
  // the CPU never goes above the configured step (its acoustic sub-steps are sized for it)
  const cur = m.c.dt, lo = 0.25 * dt0;
  let next = Math.min(dt0, CFL_TARGET / Math.max(rate, 1e-9));
  if (rate * cur > CFL_MAX) next = Math.min(next, 0.7 / rate);
  else if (next > cur) next = Math.min(next, 1.1 * cur);
  next = Math.max(lo, next);
  if (Math.abs(next - cur) > 0.02 * cur) m.c.dt = next;
}
/** The eye nest's sub-steps for the outer step now (CPU: its Courant number computed here; GPU: read back). */
async function nestFit(): Promise<void> {
  const e = eye; if (!e || !m) return;
  const dt = gpu ? gpu.dt : m.c.dt;
  e.fit(dt, adaptive ? await e.courantRate() : null);
}
/** Keep the state before an interaction (see undo). */
function takeUndo(): void {
  if (!m || !mp || m.size * (5 + m.scalars.length) > 4e6 * 11) { undo = null; return; }
  undo = { t: m.time, arrays: [m.u, m.v, m.w, m.th, m.pp, ...m.scalars].map((a) => Float32Array.from(a)), rain: Float32Array.from(mp.rainAcc), snow: Float32Array.from(mp.snowAcc),
    ub: Float64Array.from(m.ub), vb: Float64Array.from(m.vb), bu: m.boundary ? Float64Array.from(m.boundary.u) : null, bqv: m.boundary?.qv ? Float64Array.from(m.boundary.qv) : null, envDu };
}
/** Restore the state before the last interaction (after a blow-up); returns false without one. */
async function restoreUndo(): Promise<boolean> {
  if (!undo || !m || !mp) return false;
  const u = undo; undo = null;
  [m.u, m.v, m.w, m.th, m.pp, ...m.scalars].forEach((a, f) => a.set(u.arrays[f]!));
  mp.rainAcc.set(u.rain); mp.snowAcc.set(u.snow);
  m.ub.set(u.ub); m.vb.set(u.vb);
  if (m.boundary && u.bu) { m.boundary.u.set(u.bu); if (m.boundary.qv && u.bqv) m.boundary.qv.set(u.bqv); }
  envDu = u.envDu;
  m.time = u.t; m.c.dt = dt0;
  if (gpu) {
    const device = gpu.device, tp = gpu.tracerCount ? await gpu.readTracers() : null;
    gpu.destroy();
    gpu = new GpuRegional(device, m, { moist: true, physics: physCfg, ice: true });
    gpu.setForcings(forcings.map((x) => x.f));
    gpu.uploadFrom(m, { rain: mp.rainAcc, snow: mp.snowAcc });
    gpu.setDt(dt0);
    if (tp) gpu.initTracers(tp);
  }
  // the eye nest starts again from the restored outer state
  if (eye) {
    const g = eye.g;
    stopNest();
    try { await startNest(g.R, g.dx, g.dz, false); } catch (e) { post({ type: 'log', text: `眼區細化無法重建，已停止 / the eye nest could not be rebuilt: ${String((e as Error).message ?? e)}` }); }
  }
  gpuBatch = 1;
  return true;
}
/** After a numerical blow-up: undo the last interaction if it was recent, else stop with an error. */
async function blowUp(): Promise<void> {
  running = false;
  if (undo && modelNow() - undo.t < 3600 && await restoreUndo()) {
    clearForcings();
    post({ type: 'paused', reason: '數值不穩定：上一個互動太強，已自動還原到互動前並暫停 / numerical instability right after the last change: the state before it was restored and the run paused' });
    await sendFrame();
  } else post({ type: 'error', message: '數值發散 / numerical blow-up' });
}
function dtLimits(): { lo: number; hi: number } {
  const c = m!.c;
  const ac = c.nsound * 0.45 * Math.min(c.dx, c.dy) / 350;
  return { lo: 0.25 * dt0, hi: Math.max(dt0, Math.min(3 * dt0, ac)) };
}
async function adaptDt(): Promise<void> {
  if (!gpu || !m) return;
  if (!adaptive) { await nestFit(); return; }
  const rate = await gpu.maxCourantRate();
  if (!Number.isFinite(rate)) return;
  const { lo, hi } = dtLimits(), cur = gpu.dt;
  let next = Math.min(hi, CFL_TARGET / Math.max(rate, 1e-9));
  if (rate * cur > CFL_MAX) next = Math.min(next, 0.7 / rate);          // overshoot: cut back hard
  else if (next > cur) next = Math.min(next, 1.1 * cur);
  next = Math.max(lo, next);
  if (Math.abs(next - cur) > 0.02 * cur) gpu.setDt(next);
  await nestFit();
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
    g.setForcings(forcings.map((x) => x.f));
    gpu = g;
  } catch (e) { reason = String(e); }
  for (let i = 0; i < scopes.length; i++) { const err = await device.popErrorScope(); if (err && !reason) reason = err.message; }
  if (reason) { gpu?.destroy(); gpu = null; }
  return reason;
}

const post = (msg: FromRegionalWorker, tr: Transferable[] = []): void => (self as unknown as Worker).postMessage(msg, tr);

/** Build the model of a set-up (presets included); sets the module state. */
function build(s0: RegionalSetup, gpuOk: boolean): { dt: number; description: string; land: Uint8Array | null } {
  const b = buildModel(s0, gpuOk);
  setup = b.setup;
  experiment = presetById(setup.preset) && setup.preset !== 'custom' ? setup.preset as RegionalExperiment : 'custom';
  m = b.model; mp = new IceMicrophysics(m);
  physCfg = b.physics; phys = physCfg ? new RegionalPhysics(m, physCfg) : null;
  frameVel = { ...b.frame }; if (physCfg) physCfg.frameVel = frameVel;
  tracker = b.tracker; lastTrack = 0;
  dpEnv = b.dpEnv;
  hookForcing(m);
  return { dt: m.c.dt, description: b.description, land: b.land };
}

/** Land mask of the current surface (painted land has wetness below 1), or null when all sea / no surface. */
function landMask(): Uint8Array | null {
  const sf = phys?.surface; if (!sf) return null;
  const a = Uint8Array.from(sf.wet, (w) => (w < 0.99 ? 1 : 0));
  return a.some((x) => x) ? a : null;
}

/** One-way nest inside the global model at (lat0, lon0): initial and lateral-boundary fields from the
 *  global snapshot, surface skin temperature and wetness from the global surface model. */
function buildNest(g: NestPayload, lat0: number, lon0: number, size: NestSize, gpuOk: boolean): { dt: number; description: string; land: Uint8Array | null } {
  experiment = 'nest'; setup = null; tracker = null;
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
  hookForcing(m);
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
    if (msg.type === 'init' && presetById(msg.setup.preset)?.axi) {
      running = false;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      busy = true;
      try {
        stopNest(); gpu?.destroy(); gpu = null; m = null; mp = null; tracker = null; nestSource = null;
        experiment = 'tc_axi'; frameVel = { u: 0, v: 0 }; setup = { ...msg.setup };
        axi = new AxiDriver(axiFromSetup(msg.setup));
        resetOrigin(); stormDomain = null; tracers = null; catalog.reset(); followId = null; clearForcings(); parents = [];
        post({ type: 'ready', land: null, experiment: 'tc_axi', nx: axi.N, ny: axi.N, nz: axi.ax.a.nz, dx: axi.dxv, dz: axi.ax.a.dz, dt: axi.dt, description: axi.description(), backend: 'cpu',
          note: '軸對稱模式在 CPU 上執行（很快）；3D 畫面是把半徑–高度場繞軸旋轉 / the axisymmetric model runs on the CPU (fast); the 3-D view revolves the radius-height fields', refineTo: null, tc: true, setup });
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
      stopNest(); gpu?.destroy(); gpu = null;
      let info = build(msg.setup, !!device);
      builtGpu = !!device; nestSource = null;
      let note = '';
      if (device) {
        const reason = await tryGpu(device);
        if (reason) { note = gpuFailNote(reason); info = build(msg.setup, false); builtGpu = false; }
      } else if (msg.backend !== 'cpu') note = 'WebGPU 不可用，改用 CPU / WebGPU unavailable, using CPU';
      const c = m!.c, cells = c.nx * c.ny * c.nz;
      if (!gpu && cells > 1.5e6) note += (note ? ' · ' : '') + `這個網格有 ${(cells / 1e6).toFixed(1)} M 格點，在 CPU 上非常慢 / ${(cells / 1e6).toFixed(1)} M cells: very slow on the CPU`;
      dt0 = c.dt; resetOrigin(); stormDomain = null; setupTracers(); catalog.reset(); followId = null; clearForcings(); parents = [];
      post({ type: 'ready', land: info.land, experiment, nx: c.nx, ny: c.ny, nz: c.nz, dx: c.dx, dz: c.dz, dt: info.dt, description: info.description, backend: gpu ? 'gpu' : 'cpu', note,
        tc: isTc(), setup, ...gridOptions() });
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
      stopNest(); gpu?.destroy(); gpu = null;
      let info = buildNest(msg.payload, msg.lat0, msg.lon0, msg.size, !!device);
      builtGpu = !!device; nestSource = { payload: msg.payload, lat0: msg.lat0, lon0: msg.lon0, size: msg.size };
      let note = '';
      if (device) {
        const reason = await tryGpu(device);
        if (reason) { note = gpuFailNote(reason); info = buildNest(msg.payload, msg.lat0, msg.lon0, msg.size, false); builtGpu = false; }
      } else note = (msg.backend !== 'cpu' ? 'WebGPU 不可用，改用 CPU / WebGPU unavailable, using CPU · ' : '') + 'CPU 使用較粗的網格 / the CPU uses a coarser grid';
      const c = m!.c;
      dt0 = c.dt; resetOrigin(); stormDomain = null; setupTracers(); catalog.reset(); followId = null; clearForcings(); parents = [];
      post({ type: 'ready', experiment: 'nest', nx: c.nx, ny: c.ny, nz: c.nz, dx: c.dx, dz: c.dz, dt: info.dt, description: info.description, backend: gpu ? 'gpu' : 'cpu', note, land: info.land, refineTo: null, tc: false, setup: null });
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
        if (gpu) { const d1 = gpu.dt; stepGpu(1); advectTracers(d1, 1); await gpu.device.queue.onSubmittedWorkDone(); await adaptDt(); }
        else { stepCpu(); advectTracers(m.c.dt, 1); cpuAdaptDt(); await nestFit(); }
        await sendFrame();
      } catch (e) { post({ type: 'error', message: String(e) }); }
      busy = false;
    }
    else if (msg.type === 'speed') stepsPerTick = Math.max(1, msg.stepsPerTick | 0);
    else if (msg.type === 'ground') { ground = msg.field; await refreshFrame(); }
    else if (msg.type === 'charts') { chartReq = msg.req; await refreshFrame(); }
    else if (msg.type === 'volMode') { volMode = msg.mode | 0; await refreshFrame(); }
    else if (msg.type === 'subgrid') { subgrid = msg.on; await refreshFrame(); }
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
        const meta = { kind: 'regional' as const, title: `${EXP_LABEL.tc_axi} · ${(t / 3600).toFixed(1)} h`, experiment: 'tc_axi', builtGpu: false, axi: axi.p, setup,
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
          time: t, steps: m.steps, dt: gpu ? gpu.dt : c.dt, frameVel, dpEnv, adaptive, nest, origin, setup };
        const buffer = packSave(meta, arrays);
        post({ type: 'saveData', meta, buffer }, [buffer]);
        if (eye) post({ type: 'log', text: '存檔只含外圍網格（細化區以其平均存在外圍網格裡），載入後可再開始眼區細化 / the save holds the outer grid (with the nest averaged into it); start the eye nest again after loading' });
      } catch (e) { post({ type: 'error', message: `存檔失敗 / save failed: ${String(e)}` }); }
      busy = false; running = wasRunning;
    }
    else if (msg.type === 'load') {
      running = false;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      busy = true;
      try {
        const { meta, arrays } = unpackSave(msg.buffer);
        const M = meta as unknown as { experiment: RegionalExperiment; builtGpu: boolean; grid: { nx: number; ny: number; nz: number; size: number; nsc: number }; time: number; steps: number; dt: number; frameVel: { u: number; v: number }; dpEnv: number; nest: Record<string, unknown> | null; origin?: { x: number; y: number };
          setup?: RegionalSetup | null; tcEnv?: Partial<AxiParams> | null; tornadoEnv?: TornadoEnv | null };
        if (meta.kind !== 'regional') throw new Error('不是區域模式存檔 / not a regional save');
        if (M.experiment === 'tc_axi') {
          stopNest(); gpu?.destroy(); gpu = null; m = null; mp = null; tracker = null; nestSource = null; experiment = 'tc_axi'; frameVel = { u: 0, v: 0 };
          axi = new AxiDriver({ ...AXI_DEFAULTS, ...LEGACY_TC, ...((meta as unknown as { axi?: AxiParams }).axi ?? {}) });
          axi.restore(arrays as Record<string, Float32Array>, M.time, M.steps);
          setup = M.setup ?? setupOf('tc_axi');
          resetOrigin(); stormDomain = null; tracers = null; catalog.reset(); followId = null; clearForcings(); parents = [];
          post({ type: 'ready', land: null, experiment: 'tc_axi', nx: axi.N, ny: axi.N, nz: axi.ax.a.nz, dx: axi.dxv, dz: axi.ax.a.dz, dt: axi.dt, description: axi.description(), backend: 'cpu', note: `已載入存檔 / save loaded (t = ${(M.time / 3600).toFixed(2)} h)`, refineTo: null, tc: true, setup });
          await sendFrame();
          busy = false;
          return;
        }
        axi = null;
        const device = msg.backend === 'cpu' ? null : await getGpu();
        stopNest(); gpu?.destroy(); gpu = null;
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
          // saves from before set-ups recorded the experiment and its environment (the oldest ones neither: the
          // neutral sounding with radiative relaxation)
          info = build(M.setup ?? setupFromLegacy(M.experiment, M.tcEnv, M.tornadoEnv), M.builtGpu); nestSource = null;
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
        // the environment's pressure comes from the rebuilt base state (saves before sea-level pressures stored the lowest level's)
        let note = `已載入存檔 / save loaded (t = ${(M.time / 3600).toFixed(2)} h)`;
        if (device) {
          const reason = await tryGpu(device);
          if (reason) note += ' · ' + gpuFailNote(reason);
          else if (gpu) (gpu as GpuRegional).setDt(M.dt);
        } else if (M.builtGpu) note += ' · 此存檔是 GPU 網格，在 CPU 上會很慢 / GPU-sized grid: very slow on the CPU';
        dt0 = mm.c.dt;
        resetOrigin(M.origin?.x ?? 0, M.origin?.y ?? 0); stormDomain = null; setupTracers(); catalog.reset(); followId = null; clearForcings(); parents = [];
        const c = mm.c;
        post({ type: 'ready', experiment, nx: c.nx, ny: c.ny, nz: c.nz, dx: c.dx, dz: c.dz, dt: info.dt, description: info.description, backend: gpu ? 'gpu' : 'cpu', note, land: info.land ?? null,
          tc: isTc(), setup: experiment === 'nest' ? null : setup, ...gridOptions() });
        await sendFrame();
      } catch (e) { post({ type: 'error', message: `載入失敗 / load failed: ${String((e as Error).message ?? e)}` }); }
      busy = false;
    }
    else if (msg.type === 'nestStart' || msg.type === 'nestStop') {
      if (!setup || experiment === 'nest' || axi || !m || !mp) return;
      if (msg.type === 'nestStop' && !eye) return;
      const wasRunning = running; running = false;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      busy = true;
      try {
        await syncFromGpu();
        advanceOrigin();
        stopNest();
        if (msg.type === 'nestStart') post({ type: 'log', text: await startNest(msg.R, msg.dx, msg.dz, true) });
        else post({ type: 'log', text: '已停止眼區細化；外圍網格保留細化區回饋的平均 / eye nest stopped; the outer grid keeps what the nest fed back' });
        await sendFrame();
      } catch (e) { post({ type: 'error', message: `眼區細化失敗 / eye nest failed: ${String((e as Error).message ?? e)}` }); }
      finally { busy = false; running = wasRunning; }
    }
    else if (msg.type === 'refine') {
      if (!setup || experiment === 'nest' || axi || !m || !mp) return;
      if (!refinedSetup(setup)) return;
      const wasRunning = running; running = false;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      busy = true;
      try {
        await syncFromGpu();
        advanceOrigin();
        stopNest();
        const target = refinedSetup(setup);
        if (!target) return;
        const mc = m, mpc = mp, fromDx = mc.c.dx, frameC = { ...frameVel }, trackC = tracker?.position ?? null, originC = { ...origin };
        const Lc = mc.c.nx * mc.c.dx, Lcy = mc.c.ny * mc.c.dy, Lf = target.L;
        keepParent(Lf < Lc - 1);
        const device = gpu ? gpu.device : null;
        gpu?.destroy(); gpu = null;
        const info = build(target, !!device);
        builtGpu = !!device;
        const mf = m!;
        // bring the fine model into the coarse model's current frame (the interpolated winds are relative to it)
        if (frameC.u !== frameVel.u || frameC.v !== frameVel.v) {
          mf.shiftFrame(frameC.u - frameVel.u, frameC.v - frameVel.v);
          frameVel = frameC; if (physCfg) physCfg.frameVel = frameVel;
        }
        // sub-box around the tracked storm when the fine domain is smaller
        const Lfx = mf.c.nx * mf.c.dx, Lfy = mf.c.ny * mf.c.dy;
        const cx = trackC?.x ?? Lc / 2, cy = trackC?.y ?? Lcy / 2;
        const open = mc.c.lateral === 'open';
        const x0 = Lfx >= Lc - 1 ? 0 : open ? Math.max(0, Math.min(Lc - Lfx, cx - Lfx / 2)) : cx - Lfx / 2;
        const y0 = Lfy >= Lcy - 1 ? 0 : open ? Math.max(0, Math.min(Lcy - Lfy, cy - Lfy / 2)) : cy - Lfy / 2;
        refineInto(mc, mf, x0, y0, { rain: mpc.rainAcc, snow: mpc.snowAcc }, { rain: mp!.rainAcc, snow: mp!.snowAcc });
        // an open box relaxes to its state at this moment
        if (mf.c.lateral === 'open') mf.boundary = { u: Float64Array.from(mf.u), v: Float64Array.from(mf.v), th: Float64Array.from(mf.th), qv: Float64Array.from(mf.scalars[QV]!), pp: Float64Array.from(mf.pp) };
        const km = (d: number): string => (d >= 1000 ? `${+(d / 1000).toFixed(2)} km` : `${d} m`);
        let note = `已從 Δx ${km(fromDx)} 細化 / refined from Δx ${km(fromDx)} at t = ${(mc.time / 60).toFixed(0)} min`;
        if (device) {
          const reason = await tryGpu(device);
          if (reason) note += ' · ' + gpuFailNote(reason);
        }
        dt0 = mf.c.dt;
        const wrapX = (x: number, L: number): number => (open ? x : x - Math.floor(x / L) * L);
        resetOrigin(originC.x + wrapX(x0, Lc), originC.y + wrapX(y0, Lcy)); stormDomain = null; setupTracers(); clearForcings();
        const c = mf.c;
        post({ type: 'ready', land: landMask() ?? info.land, experiment, nx: c.nx, ny: c.ny, nz: c.nz, dx: c.dx, dz: c.dz, dt: info.dt, description: info.description, backend: gpu ? 'gpu' : 'cpu', note,
          tc: isTc(), setup, ...gridOptions() });
        await sendFrame();
      } catch (e) { post({ type: 'error', message: `細化失敗 / refinement failed: ${String(e)}` }); }
      finally { busy = false; running = wasRunning; }
    }
    else if (msg.type === 'coarsen') {
      if (!setup || experiment === 'nest' || axi || !m || !mp) return;
      const parent = parents[parents.length - 1] ?? null, target = parent ? parent.setup : coarsenedSetup(setup);
      if (!target) return;
      const wasRunning = running; running = false;
      while (busy) await new Promise((r) => setTimeout(r, 5));
      busy = true;
      try {
        await syncFromGpu();
        advanceOrigin();
        stopNest();
        if (parent) parents.pop();
        const mf = m, mpf = mp, physF = phys, frameF = { ...frameVel }, originF = { ...origin }, fromDx = mf.c.dx, tNow = mf.time, sfF = physF?.surface ?? null;
        const device = gpu ? gpu.device : null;
        gpu?.destroy(); gpu = null;
        const info = build(target, !!device);
        builtGpu = !!device;
        const mc = m!;
        // the coarse model's frame: the kept state's, else the fine model's
        const frameC = parent ? parent.frame : frameF;
        if (frameC.u !== frameVel.u || frameC.v !== frameVel.v) {
          mc.shiftFrame(frameC.u - frameVel.u, frameC.v - frameVel.v);
          frameVel = { ...frameC }; if (physCfg) physCfg.frameVel = frameVel;
        }
        let originC = { ...originF };
        if (parent) {
          // the kept state (its outer part is the state at the time of the refinement)
          [mc.u, mc.v, mc.w, mc.th, mc.pp, ...mc.scalars].forEach((a, f) => a.set(parent.arrays[f]!));
          mp!.rainAcc.set(parent.rain); mp!.snowAcc.set(parent.snow);
          const sf = phys?.surface;
          if (sf && parent.tsk && parent.wet) { sf.tsk.set(parent.tsk); sf.wet.set(parent.wet); if (physCfg) physCfg.surface = { tsk: sf.tsk, wet: sf.wet }; }
          if (parent.bnd && mc.boundary) { mc.boundary.u.set(parent.bnd.u); mc.boundary.v.set(parent.bnd.v); mc.boundary.th.set(parent.bnd.th); if (parent.bnd.qv && mc.boundary.qv) mc.boundary.qv.set(parent.bnd.qv); if (parent.bnd.pp && mc.boundary.pp) mc.boundary.pp.set(parent.bnd.pp); }
          dpEnv = parent.dpEnv;
          // where its domain would be now (moving with its frame velocity since the refinement)
          originC = { x: parent.origin.x + parent.frame.u * (tNow - parent.time), y: parent.origin.y + parent.frame.v * (tNow - parent.time) };
        }
        // the fine run in the coarse model's frame, averaged into it (a box blended over its relaxation zone)
        if (frameF.u !== frameC.u || frameF.v !== frameC.v) mf.shiftFrame(frameC.u - frameF.u, frameC.v - frameF.v);
        const x0 = originF.x - originC.x, y0 = originF.y - originC.y;
        const margin = parent?.box ? (mf.c.relaxCells ?? 6) * mf.c.dx + 2 * mc.c.dx : 0;
        const sfC = phys?.surface ?? null;
        coarsenInto(mf, mc, x0, y0, margin, { rain: mpf.rainAcc, snow: mpf.snowAcc }, { rain: mp!.rainAcc, snow: mp!.snowAcc },
          sfF && sfC ? [sfF.tsk, sfF.wet] : undefined, sfF && sfC ? [sfC.tsk, sfC.wet] : undefined);
        if (sfC && physCfg) physCfg.surface = { tsk: sfC.tsk, wet: sfC.wet };
        mc.time = tNow;
        const km = (d: number): string => (d >= 1000 ? `${+(d / 1000).toFixed(2)} km` : `${d} m`);
        let note = `已從 Δx ${km(fromDx)} 粗化到 ${km(mc.c.dx)} / coarsened from Δx ${km(fromDx)} to ${km(mc.c.dx)}` +
          (parent?.box ? '；細化區已貼回，外圍是細化當時的狀態 / the refined box was put back; outside it the state is the one at the refinement' : '');
        if (device) {
          const reason = await tryGpu(device);
          if (reason) note += ' · ' + gpuFailNote(reason);
        }
        dt0 = mc.c.dt;
        resetOrigin(originC.x, originC.y); stormDomain = null; setupTracers(); clearForcings();
        const c = mc.c;
        post({ type: 'ready', land: landMask() ?? info.land, experiment, nx: c.nx, ny: c.ny, nz: c.nz, dx: c.dx, dz: c.dz, dt: info.dt, description: info.description, backend: gpu ? 'gpu' : 'cpu', note,
          tc: isTc(), setup, ...gridOptions() });
        await sendFrame();
      } catch (e) { post({ type: 'error', message: `粗化失敗 / coarsening failed: ${String(e)}` }); }
      busy = false; running = wasRunning;
    }
    else if (msg.type === 'adaptive') {
      adaptive = msg.on;
      if (!adaptive && gpu) { while (busy) await new Promise((r) => setTimeout(r, 5)); gpu.setDt(dt0); await nestFit(); }
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
    else if (msg.type === 'clearForcing') { clearForcings(); if (!running) await refreshFrame(); }
    else if (msg.type === 'perturb' || msg.type === 'paint' || msg.type === 'environment' || msg.type === 'moisture' || msg.type === 'wind') {
      if (!m || !mp) { post({ type: 'error', message: '這個實驗不支援此互動（軸對稱模式請改用參數面板）/ this experiment does not support this interaction (use the panel for the axisymmetric model)' }); return; }
      while (busy) await new Promise((r) => setTimeout(r, 5));
      busy = true;
      try {
        const note = await interact(msg);
        if (note) post({ type: 'log', text: note });
        if (gpu) await adaptDt(); else cpuAdaptDt();
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
          stepGpu(nb); rateSteps += nb;
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
          const dtb = m.c.dt;
          for (let s = 0; s < ns; s++) { stepCpu(); rateSteps++; }
          advectTracers(ns * dtb, ns);
          cpuAdaptDt();
          await nestFit();
        }
        advanceOrigin();
        if (pacer.reached(modelNow())) {
          running = false;
          await sendFrame();
          post({ type: 'paused', reason: `已到達設定時間，自動暫停 / reached the stop time (t = ${(modelNow() / 3600).toFixed(2)} h)` });
        }
        if (forcings.length) expireForcings();
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

/** Shift a [j][i] plane by (di, dj) cells, wrapping around (as RegionalModel.roll). */
function rollPlane(a: Float64Array, di: number, dj: number): void {
  const { nx, ny } = m!.c, t = a.slice(0, nx * ny), mod = (x: number, n: number): number => ((x % n) + n) % n;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) a[mod(j + dj, ny) * nx + mod(i + di, nx)] = t[j * nx + i]!;
}
/** Open boundaries: the columns that a roll wrapped around to the upstream edge enter from outside the domain; they take the
 *  environment (the boundary targets), a fresh surface of the set-up and no precipitation. */
function enterEnvironment(di: number, dj: number): void {
  const mm = m!, b = mm.boundary, { nx, ny, nz } = mm.c;
  if (!b) return;
  const entered = (i: number, n: number, d: number): boolean => (d > 0 ? i < d : d < 0 ? i >= n + d : false);
  const sf = phys?.surface, sea = setup?.surface !== 'land', tsk = sea ? (setup?.sst ?? 28) + 273.15 : mm.th0[0]! * mm.pi0[0]!;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    if (!entered(i, nx, di) && !entered(j, ny, dj)) continue;
    for (let k = 0; k <= nz; k++) {
      const q = mm.idx(i, j, k);
      mm.w[q] = 0;
      if (k === nz) continue;
      mm.u[q] = b.u[q]!; mm.v[q] = b.v[q]!; mm.th[q] = b.th[q]!; mm.pp[q] = b.pp ? b.pp[q]! : 0;
      mm.scalars.forEach((a, n) => { a[q] = n === QV ? (b.qv ? b.qv[q]! : mm.qv0[k]!) : 0; });
    }
    const c = j * nx + i;
    mp!.rainAcc[c] = 0; mp!.snowAcc[c] = 0;
    if (sf) { sf.tsk[c] = tsk; sf.wet[c] = sea ? 1 : 0.3; }
  }
}

/** Keep the storm inside the periodic domain: Galilean frame shift and whole-cell re-centring
 *  (exact symmetries); on the GPU the model is rebuilt with the new frame velocity. */
async function followStorm(): Promise<void> {
  if (!m || !mp || !tracker) return;
  await syncFromGpu();
  // a fresh analysis of the full state; keep following the same storm while it lives, else the main one (none before a storm exists)
  advanceOrigin();
  analyseStorms(null);
  const st = mainStorm();
  if (!st) return;
  if (st.id !== followId) { followId = st.id; tracker.forget(); }
  let a = tracker.follow(m, st.xd, st.yd, st.kind);
  if (eye) {
    // the eye nest is fixed at the domain centre: whole-cell re-centring as soon as the storm is more than a cell (or
    // 15 % of the cylinder's radius) off it
    const thr = Math.max(m.c.dx, 0.15 * eye.g.R), ox = st.xd - m.c.nx * m.c.dx / 2, oy = st.yd - m.c.ny * m.c.dy / 2;
    const di = Math.abs(ox) > thr ? -Math.round(ox / m.c.dx) : 0, dj = Math.abs(oy) > thr ? -Math.round(oy / m.c.dy) : 0;
    if (di !== a.di || dj !== a.dj) { tracker.shift((di - a.di) * m.c.dx, (dj - a.dj) * m.c.dy); a = { ...a, di, dj }; }
  }
  await moveDomain(a);
}

/** Move the domain with the storm: Galilean frame change (du, dv) and whole-cell roll (di, dj) of the fields (and the
 *  eye nest with them); the GPU models are rebuilt. The CPU state must be current (syncFromGpu). */
async function moveDomain(a: { du: number; dv: number; di: number; dj: number }): Promise<void> {
  if (!m || !mp) return;
  if (!a.du && !a.dv && !a.di && !a.dj) return;
  if (a.du || a.dv) { m.shiftFrame(a.du, a.dv); eye?.shiftFrame(a.du, a.dv); frameVel.u += a.du; frameVel.v += a.dv; if (physCfg) physCfg.frameVel = frameVel; }
  if (a.di || a.dj) {
    m.roll(a.di, a.dj, [mp.rainAcc, mp.snowAcc]); origin.x -= a.di * m.c.dx; origin.y -= a.dj * m.c.dy; prevAcc = null;
    // the ground stays put: painted sea and land move through the domain with the fields
    const sf = phys?.surface;
    if (sf && physCfg) { rollPlane(sf.tsk, a.di, a.dj); rollPlane(sf.wet, a.di, a.dj); physCfg.surface = { tsk: sf.tsk, wet: sf.wet }; }
    if (m.c.lateral === 'open') enterEnvironment(a.di, a.dj);
    eye?.roll(m, { rain: mp.rainAcc, snow: mp.snowAcc }, a.di, a.dj, sf ?? null);
    if (sf) post({ type: 'land', land: landMask() });
    if (forcings.length) {
      const Lx = m.c.nx * m.c.dx, Ly = m.c.ny * m.c.dy;
      for (const x of forcings) { x.f.x = ((x.f.x + a.di * m.c.dx) % Lx + Lx) % Lx; x.f.y = ((x.f.y + a.dj * m.c.dy) % Ly + Ly) % Ly; x.info.x = x.f.x; x.info.y = x.f.y; }
      eye?.setForcings(forcings.map((x) => x.f));
      post({ type: 'forcings', list: forcings.map((x) => x.info) });
    }
  }
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
    gpu.setForcings(forcings.map((x) => x.f));
    gpu.uploadFrom(m, { rain: mp.rainAcc, snow: mp.snowAcc });
    gpu.setDt(dtNow);
    await nestGpuUp();
  }
  if (tp && tracers) { tracers.pos.set(tp); gpu?.initTracers(tp); }
  else if (tp) setupTracers(tp);
}

/** Rebuild the eye nest's GPU model and coupling for a newly built outer GPU model (stops the nest if that fails). */
async function nestGpuUp(): Promise<void> {
  if (!eye || !gpu) return;
  const reason = await eye.gpuUp(gpu);
  if (reason) { stopNest(); post({ type: 'log', text: `眼區細化的 GPU 模式建立失敗，已停止 / the eye nest's GPU model failed, stopped: ${reason.slice(0, 200)}` }); }
  else eye.fit(gpu.dt);
}

/**
 * Start the eye nest: radius R, spacings near dx, dz (m). `recentre`: first roll the domain (whole cells) so the main storm
 * is at its centre, where the cylinder is. Returns a note for the page; throws with the reason when it cannot start.
 */
async function startNest(R: number, dx: number, dz: number, recentre: boolean): Promise<string> {
  const mm = m!, mpp = mp!, c = mm.c;
  let where = '';
  if (recentre) {
    // the cylinder is fixed at the domain centre, so the domain follows the storm from now on
    if (!tracker) { tracker = new StormTracker(setup?.init === 'vortex' ? 'vortex' : 'updraft'); lastTrack = mm.time; where = '區域從現在起跟著風暴走 / the domain follows the storm from now on；'; }
    analyseStorms(null);
    const st = mainStorm();
    if (!st) where += '還沒偵測到風暴，圓柱在區域中央 / no storm yet: the cylinder is at the domain centre';
    else {
      const di = -Math.round((st.xd - c.nx * c.dx / 2) / c.dx), dj = -Math.round((st.yd - c.ny * c.dy / 2) / c.dy);
      if (di || dj) { await moveDomain({ du: 0, dv: 0, di, dj }); tracker.shift(di * c.dx, dj * c.dy); }
      where += '以風暴為中心 / centred on the storm';
    }
  }
  const n = EyeNest.build(mm, { rain: mpp.rainAcc, snow: mpp.snowAcc }, setup!, frameVel, phys?.surface ?? null, R, dx, dz, !!gpu);
  if (typeof n === 'string') throw new Error(n);
  n.setForcings(forcings.map((x) => x.f));
  n.fit(gpu ? gpu.dt : c.dt);
  if (gpu) {
    const reason = await n.gpuUp(gpu);
    if (reason) { n.destroy(); throw new Error(`GPU: ${reason.slice(0, 300)}`); }
  }
  eye = n;
  post({ type: 'nest', info: nestInfo() });
  const g = n.g, km = (d: number): string => (d >= 1000 ? `${+(d / 1000).toFixed(2)} km` : `${Math.round(d)} m`);
  return `眼區細化：半徑 ${km(g.R)} 的圓柱，Δx ${km(g.dx)}（外圍的 1/${g.r}）、Δz ${km(g.dz)}（1/${g.rz}），${(n.cells / 1e6).toFixed(2)} M 格，每個外圍步 ${n.nsub} 個內部步；雙向耦合${where ? '；' + where : ''}` +
    ` / eye nest: cylinder of radius ${km(g.R)}, Δx ${km(g.dx)}, Δz ${km(g.dz)}, ${(n.cells / 1e6).toFixed(2)} M cells, ${n.nsub} inner steps per outer step, two-way`;
}

/** Interaction: change the conditions (never the outcome). Returns a log line. */
async function interact(msg: Extract<ToRegionalWorker, { type: 'perturb' | 'paint' | 'environment' | 'moisture' | 'wind' }>): Promise<string> {
  const mm = m!, { nx, ny, nz, dx, dy } = mm.c, L = Math.min(nx * dx, ny * dy);
  if (msg.type === 'paint') {
    const sf = phys?.surface;
    if (!sf || !physCfg) return '這個實驗沒有地面通量，不能塗海溫或陸地 / this experiment has no surface fluxes to paint';
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      if (Math.hypot((i + 0.5) * dx - msg.x, (j + 0.5) * dy - msg.y) > msg.radius) continue;
      const c = j * nx + i;
      const dT = Math.max(0.1, Math.min(5, msg.amount ?? 2));
      if (msg.kind === 'warmer') sf.tsk[c] = Math.min(310, sf.tsk[c]! + dT);
      else if (msg.kind === 'cooler') sf.tsk[c] = Math.max(271.35, sf.tsk[c]! - dT);
      else if (msg.kind === 'land') { if (sf.wet[c]! >= 0.99) sf.tsk[c] = mm.th0[0]! * mm.pi0[0]!; sf.wet[c] = 0.3; }
      else { if (sf.wet[c]! < 0.99) sf.tsk[c] = (setup?.sst ?? 28) + 273.15; sf.wet[c] = 1; }
    }
    if (msg.kind === 'land' || msg.kind === 'sea') post({ type: 'land', land: landMask() });
    // keep the painting when the GPU model is rebuilt (storm following) and in the configuration
    physCfg.surface = { tsk: sf.tsk, wet: sf.wet };
    gpu?.setSurface(sf.tsk, sf.wet);
    eye?.surfaceFrom(mm, sf);
    return '';
  }
  await syncFromGpu();
  takeUndo();
  gpuBatch = 1;
  const top = nz * mm.c.dz;
  const clampR = (r: number | undefined, d: number): number => Math.max(2 * dx, Math.min(0.45 * L, Number.isFinite(r) ? r! : d));
  if (msg.type === 'wind') {
    // a wind once (added now) or a lasting forcing; the push's vertical part is at most 15 m/s
    const R = clampR(msg.radius, 20 * dx), H = Math.max(mm.c.dz, Math.min(top / 2, 0.5 * (Number.isFinite(msg.depth) ? msg.depth : 3000)));
    const az = (msg.az || 0) * Math.PI / 180, el = Math.max(-90, Math.min(90, msg.el || 0)) * Math.PI / 180;
    let speed = Math.max(0, Math.min(40, msg.speed || 0));
    const dir: [number, number, number] = [Math.cos(el) * Math.sin(az), Math.cos(el) * Math.cos(az), Math.sin(el)];
    if (msg.form === 'push' && Math.abs(dir[2]) * speed > 15) speed = 15 / Math.abs(dir[2]);
    const f: WindForcing = { x: msg.x, y: msg.y, z: Math.max(0, Math.min(top, msg.z || 0)), R, H, speed, dir, form: msg.form, sign: msg.sign === -1 ? -1 : 1 };
    const what = `${msg.form === 'push' ? '推送 / push' : msg.form === 'rotate' ? (f.sign > 0 ? '逆時針旋轉 / counter-clockwise' : '順時針旋轉 / clockwise') : (f.sign > 0 ? '輻合 / converging' : '輻散 / diverging')} ${speed.toFixed(0)} m/s`;
    if (msg.minutes === 0) {
      applyWind(mm, [f], 'once');
      if (gpu) gpu.uploadFrom(mm, { rain: mp!.rainAcc, snow: mp!.snowAcc });
      if (eye) { applyWind(eye.m, [{ ...f, x: f.x - eye.x0, y: f.y - eye.y0 }], 'once'); eye.upload(); }
      return `一次風 / wind once: ${what} at (${(f.x / 1000).toFixed(0)}, ${(f.y / 1000).toFixed(0)}) km, ${(f.z / 1000).toFixed(1)} km high`;
    }
    if (forcings.length >= MAX_FORCINGS) forcings.shift();
    const until = msg.minutes > 0 ? modelNow() + msg.minutes * 60 : null;
    forcings.push({ f, info: { id: forcingId++, x: f.x, y: f.y, z: f.z, radius: R, depth: 2 * H, speed, az: msg.az || 0, el: msg.el || 0, form: msg.form, sign: f.sign, until } });
    syncForcings();
    return `持續風 / lasting wind: ${what}${until === null ? '，直到清除 / until cleared' : `，${msg.minutes} 模式分鐘 / model minutes`}`;
  }
  if (msg.type === 'moisture') {
    // multiply the vapour in the region (cos^2 envelope), capped at saturation
    const R = clampR(msg.radius, 20 * dx), H = Math.max(mm.c.dz, Math.min(top / 2, 0.5 * (Number.isFinite(msg.depth) ? msg.depth : 3000)));
    const fac = Math.max(0.3, Math.min(2, msg.factor || 1));
    // the outer grid and the eye nest (its origin at ox, oy in the outer coordinates)
    const moisten = (md: RegionalModel, ox: number, oy: number): void => {
      const { nx, ny, nz, dx, dy } = md.c, qv = md.scalars[QV]!, open = md.c.lateral === 'open';
      for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        let ex = ox + (i + 0.5) * dx - msg.x, ey = oy + (j + 0.5) * dy - msg.y;
        if (!open) { ex -= Math.round(ex / (nx * dx)) * nx * dx; ey -= Math.round(ey / (ny * dy)) * ny * dy; }
        const rh = Math.hypot(ex, ey) / R, rz = Math.abs(md.zc[k]! - msg.z) / H;
        if (rh >= 1 || rz >= 1) continue;
        const e = Math.cos(0.5 * Math.PI * rh) ** 2 * Math.cos(0.5 * Math.PI * rz) ** 2, q = md.idx(i, j, k);
        const pi = md.pi0[k]! + md.pp[q]!, T = md.th[q]! * pi, p = 1e5 * Math.pow(pi, 1004.5 / 287.05), es = 611.2 * Math.exp(17.67 * (T - 273.15) / (T - 29.65));
        qv[q] = Math.max(0, Math.min(qv[q]! * (1 + (fac - 1) * e), Math.max(qv[q]!, 0.622 * es / Math.max(p - es, 1))));
      }
    };
    moisten(mm, 0, 0);
    if (gpu) gpu.uploadFrom(mm, { rain: mp!.rainAcc, snow: mp!.snowAcc });
    if (eye) { moisten(eye.m, eye.x0, eye.y0); eye.upload(); }
    return `${fac >= 1 ? '增濕 / moistened' : '變乾 / dried'} ×${fac} at (${(msg.x / 1000).toFixed(0)}, ${(msg.y / 1000).toFixed(0)}) km, ${(msg.z / 1000).toFixed(1)} km high`;
  }
  if (msg.type === 'perturb') {
    // warm bubble (+3 K, centred at 1.5 km unless given) or cold pool (-6 K at the ground unless given), horizontal radius 10 km
    // (at least 4 cells, at most L/8) and vertical radius 1.5 km unless given; amplitude up to 15 K. Repeated clicks add up
    // only to twice the amplitude (at least 6 K warm / 10 K cold, at most 20 K) relative to the base state (they never
    // weaken an existing anomaly).
    const warm = msg.kind === 'warm', rh = msg.radius ? clampR(msg.radius, 10000) : Math.max(4 * dx, Math.min(10000, L / 8));
    const zc = msg.z !== undefined && Number.isFinite(msg.z) ? Math.max(0, Math.min(top, msg.z)) : warm ? 1500 : 0;
    const rz = msg.depth && Number.isFinite(msg.depth) ? Math.max(mm.c.dz, Math.min(top / 2, msg.depth / 2)) : 1500;
    const a0 = msg.amp !== undefined && Number.isFinite(msg.amp) ? Math.min(15, Math.abs(msg.amp)) : warm ? 3 : 6;
    const amp = warm ? a0 : -a0, cap = warm ? Math.min(20, Math.max(6, 2 * a0)) : -Math.min(20, Math.max(10, 2 * a0));
    const bubble = (md: RegionalModel, ox: number, oy: number): void => {
      const { nx, ny, nz, dx, dy } = md.c;
      for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const r = Math.sqrt(((ox + (i + 0.5) * dx - msg.x) / rh) ** 2 + ((oy + (j + 0.5) * dy - msg.y) / rh) ** 2 + ((md.zc[k]! - zc) / rz) ** 2);
        if (r >= 1) continue;
        const q = md.idx(i, j, k), d = amp * Math.cos(0.5 * Math.PI * r) ** 2, room = md.th0[k]! + cap - md.th[q]!;
        md.th[q] = md.th[q]! + (warm ? Math.max(0, Math.min(d, room)) : Math.min(0, Math.max(d, room)));
      }
    };
    bubble(mm, 0, 0);
    if (gpu) gpu.uploadFrom(mm, { rain: mp!.rainAcc, snow: mp!.snowAcc });
    if (eye) { bubble(eye.m, eye.x0, eye.y0); eye.upload(); }
    return `${warm ? '放暖泡 / warm bubble' : '放冷池 / cold pool'} ${amp >= 0 ? '+' : ''}${amp.toFixed(1)} K at (${(msg.x / 1000).toFixed(1)}, ${(msg.y / 1000).toFixed(1)}) km, ${(zc / 1000).toFixed(1)} km high`;
  }
  // environment: wind increment linear to 6 km (constant above) and a humidity factor tapered over 1-8 km;
  // the accumulated change at 6 km stays within +-ENV_DU_MAX
  const du6 = Math.max(-ENV_DU_MAX - envDu, Math.min(ENV_DU_MAX - envDu, msg.du6));
  envDu += du6;
  const du = (z: number): number => du6 * Math.min(1, z / 6000);
  const hf = (z: number): number => 1 + (msg.humidity - 1) * Math.max(0, Math.min(1, (z - 500) / 500, (8500 - z) / 500));
  // the outer grid (and its boundary targets) and the eye nest (its targets come from the outer grid)
  const change = (md: RegionalModel, targets: boolean): void => {
    const qv = md.scalars[QV]!, { nx, ny, nz } = md.c;
    for (let k = 0; k < nz; k++) {
      const d = du(md.zc[k]!), f = hf(md.zc[k]!);
      md.ub[k] = md.ub[k]! + d;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const q = md.idx(i, j, k);
        md.u[q] = md.u[q]! + d;
        if (f !== 1) {
          const pi = md.pi0[k]! + md.pp[q]!, T = md.th[q]! * pi, p = 1e5 * Math.pow(pi, 1004.5 / 287.05), es = 611.2 * Math.exp(17.67 * (T - 273.15) / (T - 29.65));
          qv[q] = Math.min(qv[q]! * f, 0.622 * es / Math.max(p - es, 1));
        }
        const b = md.boundary;
        if (targets && b) { b.u[q] = b.u[q]! + d; if (b.qv) b.qv[q] = b.qv[q]! * f; }
      }
    }
  };
  change(mm, true);
  if (eye) change(eye.m, false);
  if (gpu) {
    // the damping-layer wind and the boundary targets live in the GPU model's tables: rebuild it
    const device = gpu.device, dtNow = gpu.dt, tp = gpu.tracerCount ? await gpu.readTracers() : null;
    gpu.destroy();
    gpu = new GpuRegional(device, mm, { moist: true, physics: physCfg, ice: true });
    gpu.setForcings(forcings.map((x) => x.f));
    gpu.uploadFrom(mm, { rain: mp!.rainAcc, snow: mp!.snowAcc });
    gpu.setDt(dtNow);
    if (tp) gpu.initTracers(tp);
    await nestGpuUp();
  }
  return `改變環境 / environment changed: ${du6 >= 0 ? '+' : ''}${du6.toFixed(0)} m/s westerly at 6 km (total ${envDu >= 0 ? '+' : ''}${envDu.toFixed(0)}, limit ±${ENV_DU_MAX}), 1-8 km humidity x${msg.humidity}`;
}

/** Copy the GPU state into the CPU model arrays (display and diagnostics reuse the CPU code), the eye nest's too. */
async function syncFromGpu(): Promise<void> {
  if (!gpu || !m || !mp) return;
  await syncModel(gpu, m, mp);
  await eye?.sync();
}

/** set by sendFrameRaw when the state is numerically unstable */
let blowPending = false;
/** Post a frame of the current state; a numerically unstable state is handled by blowUp instead. */
async function sendFrame(): Promise<void> {
  blowPending = false;
  await sendFrameRaw();
  if (blowPending) { blowPending = false; await blowUp(); }
}
async function sendFrameRaw(): Promise<void> {
  if (axi) {
    lastFrame = performance.now(); lastFrameModel = modelNow();
    const { msg, transfer } = axi.frame(chartReq, ground, volMode, rate, { ...origin }, subgrid);
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
  let charts: ChartData | null = null, trOut: Float32Array | null = null, nestOut: NestFrame | null = null;
  const transfer = new Set<ArrayBuffer>([cloud.buffer, rain.buffer]);
  /** the eye nest's volume at this moment (before the GPU steps on); false when it is numerically unstable */
  const nestVolume = async (): Promise<boolean> => {
    const e = eye, info = nestInfo();
    if (!e || !info) return true;
    const v = await e.volume(volMode, subgrid);
    if (!Number.isFinite(v.wmax) || !Number.isFinite(v.wmin) || Math.max(v.wmax, -v.wmin) > W_BLOWUP) return false;
    nestOut = { ...info, cloud: v.cloud, rain: v.rain };
    transfer.add(v.cloud.buffer as ArrayBuffer); transfer.add(v.rain.buffer as ArrayBuffer);
    return true;
  };
  // the GPU model of this frame (the module variable can be replaced while a readback is awaited)
  const gpu = currentGpu();
  if (gpu) {
    // GPU: packed display bytes, column composites and the few horizontal planes the diagnostics need
    const levels = [...new Set([0, k15, ...(sliceK >= 0 ? [sliceK] : [])])];
    const d = await gpu.readDisplay(levels, volMode, subgrid);
    cuRate = d.cu ? Float32Array.from({ length: nx * ny }, (_, c) => d.cu![4 * c + 2]!) : null;
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
    if (!Number.isFinite(wmax) || !Number.isFinite(wmin) || Math.max(wmax, -wmin) > W_BLOWUP) { blowPending = true; return; }
    if (!await nestVolume()) { blowPending = true; return; }
    if (req) charts = await buildCharts(req, col, planeAt, sliceK, gpu);
    advanceOrigin();
    // keep the GPU busy while this frame is unpacked on the CPU
    if (running && gpu === currentGpu()) { const dtb = gpu.dt; stepGpu(gpuBatch); rateSteps += gpuBatch; advectTracers(gpuBatch * dtb, gpuBatch); }
    trOut = d.tracers ? d.tracers.slice() : null;
  } else {
    let bad = false;
    for (let q = 0; q < m.size && !bad; q += 97) if (!Number.isFinite(m.w[q]!) || Math.abs(m.w[q]!) > W_BLOWUP || !Number.isFinite(m.th[q]!)) bad = true;
    if (bad) { blowPending = true; return; }
    const cu = phys?.cu ?? null;
    cuRate = cu ? Float32Array.from(cu.rate) : null;
    // cloud: extinction of cloud water, sub-grid cloud, ice and snow (anvils are mostly snow); channel 2: extinction of
    // precipitation (rain + graupel), updraft or vorticity (display.ts; the GPU display kernel packs the same bytes)
    ({ wmax, wmin, qcmax, qrmax } = volumeBytes(m, cloud, rain, volMode, subgrid));
    if (!await nestVolume()) { blowPending = true; return; }
    if (req) {
      col = columnDiagnostics(m, subgrid);
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
  // storms: every vortex / cell with its own identity; nothing before a storm exists
  analyseStorms(col);
  const main = mainStorm();
  let storm: { x: number; y: number } | null = main ? { x: main.x, y: main.y } : null;
  if (isTc()) {
    const v = main?.kind === 'vortex' ? main : catalog.main();
    if (v && v.kind === 'vortex') {
      const ic = Math.min(nx - 1, Math.floor(v.xd / dx)), jc = Math.min(ny - 1, Math.floor(v.yd / m.c.dy));
      dp = v.pmin! - dpEnv; rmw = v.rmw ?? null;
      tcRainUpdate(ic, jc);
      const ew = eyewallProfile(m, 1500, { ic, jc }); eyewalls = ew.peaks;
      vtProfile = { dr: dx, vt: ew.vt.map((x) => +x.toFixed(2)) };
      storm = { x: v.x, y: v.y };
    }
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
  }
  if (charts) for (const a of chartArrays(charts)) transfer.add(a.buffer as ArrayBuffer);
  if (trOut) transfer.add(trOut.buffer as ArrayBuffer);
  stormDomain = main ? { x: main.xd, y: main.yd } : stormDomain;
  post({ type: 'frame', time: m.time, nx, ny, nz, dx, dz, cloud, rain, ground: g, groundField: ground, groundRange: [lo, hi],
    stats: { wmax, wmin, qcmax, qrmax, rainmax, vmax, dp, rmw, eyewalls, zetaMax, vGround, dbzMax, uhMax, uhMin, capeMax, storm, storms: catalog.active, mainId: main?.id ?? null, vtProfile, tornado, tcRain: isTc() ? tcRain : null },
    origin: { ...origin }, charts, tracers: trOut, nest: nestOut, stepsPerSecond: rate, dt: gpu ? gpu.dt : m.c.dt }, [...transfer]);
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
  if (req.maps.length) charts.maps = compositeMaps(mm, col, pl0, { rain: mpp.rainAcc, snow: mpp.snowAcc, rate: lastRate, cu: cuRate }, req.maps, frameVel);
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
    const v = mainStorm();
    if (req.rz === 'auto' && v?.kind === 'vortex') { xc = v.xd; yc = v.yd; }
    else if (req.rz === 'auto') { const r = tcMetrics(mm); xc = (r.ic + 0.5) * dx; yc = (r.jc + 0.5) * dy; }
    else { xc = req.rz.x; yc = req.rz.y; }
    const dr = dx, nr = Math.max(4, Math.min(Math.floor(Math.min(nx, ny) / 2), Math.ceil(Math.max(300000, 20 * dx) / dr)));
    const raw = gpu ? await gpu.readRZ(xc, yc, dr, nr) : azimuthalMeans(mm, xc, yc, dr, nr);
    charts.rz = { xc, yc, dr, nr, vars: unpackRZ(raw, nr, nz) };
  }
  return charts;
}
