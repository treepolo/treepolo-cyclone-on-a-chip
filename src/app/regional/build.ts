// Build a regional model from a set-up (setup.ts): grid, base state and environmental wind, surface and physics
// configuration, initial disturbance, frame velocity and storm following. Used for presets and custom runs alike.

import { RegionalModel, type RegionalConfig } from '../../regional/core.js';
import { weismanKlempQ } from '../../regional/kessler.js';
import { QV } from '../../regional/ice.js';
import type { RegionalPhysicsConfig } from '../../regional/physics.js';
import { tcSounding, insertVortex } from '../../regional/tropical.js';
import { quarterCircleWind, bunkersRightMover, StormTracker, TORNADO_WK82, type TornadoEnv } from '../../regional/supercell.js';
import { autoDt, describe, presetById, setupOf, type RegionalSetup } from './setup.js';
import type { AxiParams } from './axiDriver.js';
import type { RegionalExperiment } from './protocol.js';

export interface Built {
  model: RegionalModel;
  /** sub-grid / surface / radiation physics (null: none, the model's numerics only) */
  physics: RegionalPhysicsConfig | null;
  /** ground-relative velocity of the model frame (m/s) */
  frame: { u: number; v: number };
  /** keeps the storm near the domain centre (null: the domain stays put) */
  tracker: StormTracker | null;
  /** surface pressure of the undisturbed environment (hPa), for pressure deficits */
  dpEnv: number;
  /** the set-up actually built (a GPU-only grid runs coarser on the CPU) */
  setup: RegionalSetup;
  description: string;
  /** land mask per column (1 land), null when all sea or no surface */
  land: Uint8Array | null;
}

/** Environmental wind at height z (ground-relative) and the frame velocity that keeps a storm near the centre. */
export function envWind(s: RegionalSetup): { wind: (z: number) => { u: number; v: number }; frame: { u: number; v: number } } {
  if (s.wind === 'trade') {
    // easterly trade wind of windU below 3 km, weakening linearly to calm at 12 km
    return { wind: (z) => ({ u: -s.windU * Math.max(0, Math.min(1, (12000 - z) / 9000)), v: 0 }), frame: { u: 0, v: 0 } };
  }
  if (s.wind === 'shear') {
    // Weisman-Klemp straight-line shear: windU of speed change mostly below 6 km; the frame moves at half of it
    return { wind: (z) => ({ u: s.windU * Math.tanh(z / 3000), v: 0 }), frame: { u: s.windU / 2, v: 0 } };
  }
  if (s.wind === 'quarter') {
    const h = { R: s.windR, U6: s.windU, depth: s.windDepth }, w = (z: number): { u: number; v: number } => quarterCircleWind(z, h);
    return { wind: w, frame: bunkersRightMover(w) };
  }
  return { wind: () => ({ u: 0, v: 0 }), frame: { u: 0, v: 0 } };
}

export function buildModel(input: RegionalSetup, gpuOk: boolean): Built {
  let s = { ...input };
  // grids meant for a GPU run coarser on the CPU (as before: 3 km TC on the 5 km grid, 250 m tornado as a 500 m preview)
  if (!gpuOk && s.preset === 'tc_3' && s.dx === 3000) s = { ...s, dx: 5000, dt: 30 };
  if (!gpuOk && s.preset === 'tornado' && s.dx === 250) s = { ...s, dx: 500, L: 40000, dz: 400, dt: 3 };
  const nx = Math.round(s.L / s.dx), nz = Math.round(s.top / s.dz), dt = autoDt(s);
  const f = 2 * 7.292e-5 * Math.sin(s.lat * Math.PI / 180);
  const { wind, frame } = envWind(s);
  const open = s.boundary === 'open';
  const cfg: RegionalConfig = {
    nx, ny: nx, nz, dx: s.dx, dy: s.dx, dz: s.dz, dt, nsound: s.dx <= 250 ? 8 : 6, f, beta: s.sounding === 'wk82' ? 0.2 : 0.3, divDamp: 0.1,
    dampDepth: Math.min(6000, 0.3 * s.top), dampRate: 1 / 300, kdiff2: 0,
    lateral: open ? 'open' : 'periodic', relaxCells: Math.max(6, Math.round(2500 / s.dx)), relaxTau: 300,
    // a background wind on a rotating domain is geostrophic (balanced by the large-scale pressure gradient it implies)
    geostrophic: f !== 0 && s.wind !== 'calm',
  };
  const snd = s.sounding === 'wk82' ? weismanKlempQ(s.qvBL / 1000) : tcSounding(s.sounding === 're87' ? 're87' : 'unstable', s.sst + 273.15);
  const m = new RegionalModel(cfg, snd, 6);
  m.setBaseWind((z) => { const w = wind(z); return { u: w.u - frame.u, v: w.v - frame.v }; });
  for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) m.scalars[QV]![m.idx(i, j, k)] = m.qv0[k]!;
  // open boundaries relax toward the undisturbed environment
  if (open) m.boundary = { u: Float64Array.from(m.u), v: Float64Array.from(m.v), th: Float64Array.from(m.th), qv: Float64Array.from(m.scalars[QV]!), pp: new Float64Array(m.size) };
  // initial disturbance
  const L = nx * s.dx;
  if (s.init === 'vortex' && s.initAmp > 0) insertVortex(m, f, s.initAmp);
  else if (s.init === 'bubble' && s.initAmp > 0) {
    const rh = s.wind === 'quarter' || L < 100000 ? 5000 : 10000, xc = s.follow || open ? 0.5 * L : 0.35 * L, yc = 0.5 * L, zc = s.wind === 'quarter' ? 1500 : 1400;
    for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
      const r = Math.sqrt((((i + 0.5) * s.dx - xc) / rh) ** 2 + (((j + 0.5) * s.dx - yc) / rh) ** 2 + ((m.zc[k]! - zc) / zc) ** 2);
      if (r < 1) m.th[m.idx(i, j, k)] = m.th[m.idx(i, j, k)]! + s.initAmp * Math.cos(0.5 * Math.PI * r) ** 2;
    }
  }
  // physics: none for the classic numerics-only storm runs (no fluxes, no radiation); otherwise sub-grid mixing
  // (isotropic LES-type below 1.5 km grid spacing), surface and radiation
  let physics: RegionalPhysicsConfig | null = null;
  let land: Uint8Array | null = null;
  if (s.fluxes || s.radiation !== 'none') {
    const les = s.dx <= 1500, delta = Math.cbrt(s.dx * s.dx * s.dz);
    physics = {
      lh: les ? 0.21 * delta : 0.2 * s.dx, lv: les ? 0.21 * delta : 100, sst: 0, ck: 1.2e-3,
      radTau: s.radiation === 'relax' ? 12 * 3600 : s.radiation === 'const' ? 12 * 3600 : 0, radMax: (s.radiation === 'relax' ? s.radRate : 2) / 86400,
      radConst: s.radiation === 'const' ? s.radRate / 86400 : 0,
      vmin: s.vmin, gust: s.gust, blNoise: s.blNoise, frameVel: { ...frame },
    };
    if (s.fluxes) {
      const n2 = nx * nx, sea = s.surface === 'sea';
      physics.surface = { tsk: new Float64Array(n2).fill(sea ? s.sst + 273.15 : m.th0[0]! * m.pi0[0]!), wet: new Float64Array(n2).fill(sea ? 1 : 0.3) };
      if (!sea) { physics.z0 = 0.1; land = new Uint8Array(n2).fill(1); }
    }
  }
  const dpEnv = 1e5 * Math.pow(m.pi0[0]!, 1004.5 / 287.05) / 100;
  const tracker = s.follow ? new StormTracker(s.init === 'vortex' ? 'vortex' : 'updraft') : null;
  const p = presetById(s.preset);
  return { model: m, physics, frame, tracker, dpEnv, setup: s, land, description: `${p && s.preset !== 'custom' ? p.label.split(' / ')[0] + '：' : ''}${describe(s)}` };
}

/** The set-up of a save made before set-ups existed (experiment id and the environments it recorded). */
export function setupFromLegacy(exp: RegionalExperiment, tcEnv: Partial<AxiParams> | null | undefined, tornadoEnv: TornadoEnv | null | undefined): RegionalSetup {
  const s = setupOf(exp);
  if (exp === 'tc' || exp === 'tc_hr' || exp === 'tc_3') {
    const e = { snd: 're87', radConst: 0, radMax: 2, vmin: 1, blNoise: 0, sst: 301.15, vmax0: 15, f: 5e-5, ...(tcEnv ?? {}) };
    return { ...s, sounding: e.snd === 'unstable' ? 'tropical' : 're87', radiation: e.radConst ? 'const' : 'relax', radRate: e.radConst || e.radMax, vmin: e.vmin, blNoise: e.blNoise ?? 0,
      gust: false, sst: e.sst - 273.15, initAmp: e.vmax0, lat: Math.asin(Math.min(1, e.f / (2 * 7.292e-5))) * 180 / Math.PI };
  }
  if (exp === 'tornado' || exp === 'tornado_c') {
    const e = tornadoEnv ?? TORNADO_WK82;
    return { ...s, qvBL: e.qvMax * 1000, windR: e.R, windU: e.U6, windDepth: e.depth };
  }
  return s;
}

/** Parameters of the axisymmetric driver from a set-up. */
export function axiFromSetup(s: RegionalSetup): AxiParams {
  return { sst: s.sst + 273.15, dr: Math.max(1000, Math.min(4000, s.dx)), lh: 1000, lv: 100, ck: 1.2e-3, vmin: s.vmin, radMax: s.radiation === 'relax' ? s.radRate : 2,
    radConst: s.radiation === 'const' ? s.radRate : 0, rhTop: 0.4, snd: s.sounding === 're87' ? 're87' : 'unstable', blNoise: 0, vmax0: s.initAmp,
    f: 2 * 7.292e-5 * Math.sin(Math.max(1, Math.abs(s.lat)) * Math.PI / 180) };
}

/** The set-up a run continues with after refinement: a preset's finer preset grid (none for the finest presets), a custom
 *  grid the same domain at a third (coarse grids) or half the spacing. Returns null when there is nothing finer. */
export function refinedSetup(s: RegionalSetup): RegionalSetup | null {
  const p = presetById(s.preset);
  if (p && s.preset !== 'custom') {
    if (!p.refine) return null;
    const t = presetById(p.refine)!.setup;
    return { ...s, preset: t.preset, L: t.L, dx: t.dx, top: t.top, dz: t.dz, dt: t.dt };
  }
  if (s.dx <= 250) return null;
  const dx = s.dx >= 6000 ? s.dx / 3 : s.dx / 2;
  return { ...s, preset: 'custom', dx, dz: Math.min(s.dz, Math.max(100, dx / 4)), dt: 0 };
}
