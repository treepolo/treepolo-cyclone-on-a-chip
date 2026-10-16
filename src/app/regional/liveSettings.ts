// Settings changed while a run goes on (the "settings while running" panel of the regional page). The run is not restarted: the model keeps its state, its time,
// its history; only what the new setting means is changed in it (worker.ts then builds the GPU model anew from this state, as for a refinement).
//
//   lat                         the Coriolis parameter (the winds and pressures were balanced for the old one: they adjust)
//   boundary                    periodic <-> open (open: the edges relax to the undisturbed environment; periodic: what leaves one side enters the other)
//   wind, windU, windR, windDepth   the environmental wind: the difference between the new and the old environment is added to the whole wind field
//   sounding, qvBL              the environment (base state) of another sounding; the departures from the old one stay (the storm stays, the air around it changes: approximate)
//   surface, sst                sea or land everywhere / the sea temperature of all sea
//   fluxes, radiation, radRate, vmin, gust, blNoise, cumulus   the sub-grid, surface and radiation physics
//
// What is not here: the grid (the domain, spacings, top), the step, the starting disturbance, and following of the storm (its own switch).

import type { RegionalModel, BoundaryTargets } from '../../regional/core.js';
import { RegionalPhysics, type RegionalPhysicsConfig } from '../../regional/physics.js';
import { envWind, physicsOf, soundingOf } from './build.js';
import { sanitize, type RegionalSetup } from './setup.js';

export const LIVE_KEYS = ['lat', 'boundary', 'shape', 'surface', 'sst', 'fluxes', 'sounding', 'qvBL', 'wind', 'windU', 'windR', 'windDepth', 'radiation', 'radRate', 'vmin', 'gust', 'cumulus', 'blNoise'] as const;
export type LiveKey = (typeof LIVE_KEYS)[number];
const PHYS_KEYS: readonly LiveKey[] = ['fluxes', 'radiation', 'radRate', 'vmin', 'gust', 'blNoise', 'cumulus'];
const WIND_KEYS: readonly LiveKey[] = ['wind', 'windU', 'windR', 'windDepth'];

/** The current set-up with the live settings of `want` (sanitized); the rest of `cur` stays. */
export function liveMerge(cur: RegionalSetup, want: RegionalSetup): RegionalSetup {
  const s = { ...cur } as unknown as Record<string, unknown>;
  for (const k of LIVE_KEYS) s[k] = want[k];
  return sanitize(s as unknown as RegionalSetup);
}

/** the live settings that differ (numbers: by more than the rounding of the form) */
export const changedKeys = (a: RegionalSetup, b: RegionalSetup): LiveKey[] => LIVE_KEYS.filter((k) => {
  const x = a[k], y = b[k];
  return typeof x === 'number' && typeof y === 'number' ? Math.abs(x - y) > 1e-4 * Math.max(1, Math.abs(x)) : x !== y;
});
/** settings that change the environment the kept coarser states (refinement parents) were made in */
export const ENV_KEYS: readonly LiveKey[] = ['lat', 'boundary', 'wind', 'windU', 'windR', 'windDepth', 'sounding', 'qvBL'];

/** A model with its physics (null without sub-grid / surface physics): what the settings change. */
export interface LiveHolder { m: RegionalModel; cfg: RegionalPhysicsConfig | null; phys: RegionalPhysics | null }

/** Open-boundary targets of the undisturbed environment of a model. */
export function environmentTargets(m: RegionalModel): BoundaryTargets {
  const { nz } = m.c, z = (): Float64Array => new Float64Array(m.size);
  const b: BoundaryTargets = { u: z(), v: z(), th: z(), qv: z(), pp: z() };
  for (let k = 0; k < nz; k++) for (let o = k * m.plane; o < (k + 1) * m.plane; o++) { b.u[o] = m.ub[k]!; b.v[o] = m.vb[k]!; b.th[o] = m.th0[k]!; b.qv![o] = m.qv0[k]!; }
  return b;
}

/** Add a wind profile (per level, m/s) to the whole wind field, the base wind and the open-boundary targets. */
export function addWind(m: RegionalModel, dU: Float64Array, dV: Float64Array): void {
  const b = m.boundary;
  for (let k = 0; k < m.c.nz; k++) {
    m.ub[k] = m.ub[k]! + dU[k]!; m.vb[k] = m.vb[k]! + dV[k]!;
    for (let o = k * m.plane; o < (k + 1) * m.plane; o++) {
      m.u[o] = m.u[o]! + dU[k]!; m.v[o] = m.v[o]! + dV[k]!;
      if (b) { b.u[o] = b.u[o]! + dU[k]!; b.v[o] = b.v[o]! + dV[k]!; }
    }
  }
}

/**
 * Apply the change from set-up `old` to `next` to a model (the outer one or an inner grid: `child`, which is always open and takes its surface from the
 * outer one). `frame`: the velocity of the model frame over the ground now. The physics of the holder is made anew when anything it depends on changed
 * (the caller hooks the lasting winds again). Returns what was done, for the log.
 */
export function applyLive(h: LiveHolder, next: RegionalSetup, old: RegionalSetup, frame: { u: number; v: number }, child: boolean): { notes: string[]; physics: boolean } {
  const m = h.m, c = m.c as { f: number; geostrophic?: boolean; lateral?: 'periodic' | 'open'; relaxCells?: number; relaxTau?: number; beta?: number };
  const ch = new Set<LiveKey>(changedKeys(old, next)), notes: string[] = [];
  const windCh = WIND_KEYS.some((k) => ch.has(k)), soundingCh = ch.has('sounding') || (next.sounding === 'wk82' && ch.has('qvBL'));
  if (!ch.size) return { notes, physics: false };
  // rotation
  if (ch.has('lat') || ch.has('wind')) {
    const f = 2 * 7.292e-5 * Math.sin(next.lat * Math.PI / 180);
    c.f = f; c.geostrophic = f !== 0 && next.wind !== 'calm';
    if (ch.has('lat')) notes.push(`緯度 ${+next.lat.toFixed(1)}°（科氏參數 ${f.toExponential(2)} /s；風和氣壓會調整）/ latitude ${+next.lat.toFixed(1)}°`);
  }
  // lateral boundaries
  if (!child && ch.has('boundary')) {
    if (next.boundary === 'open') {
      c.lateral = 'open'; c.relaxCells = Math.max(6, Math.round(2500 / m.c.dx)); c.relaxTau = 300; m.boundary = environmentTargets(m);
      notes.push('側邊界改成開放：邊緣向環境鬆弛 / boundaries are open now: the edges relax to the environment');
    } else {
      c.lateral = 'periodic'; m.boundary = null;
      notes.push('側邊界改成週期：從一邊出去的空氣從對邊回來（兩邊原本不連續的地方會有一小段調整）/ boundaries are periodic now: what leaves one side enters the other');
    }
  }
  // the shape of the domain: a circle (the corners relax to the environment too) or the whole square
  if (!child && ch.has('shape')) {
    const cc = c as { relaxCyl?: { r: number; w: number } };
    if (next.shape === 'circle') {
      const nr = Math.max(6, c.relaxCells ?? Math.round(2500 / m.c.dx));
      c.lateral = 'open'; c.relaxCells = nr; c.relaxTau = c.relaxTau ?? 300; cc.relaxCyl = { r: Math.max(1, m.c.nx / 2 - nr), w: nr };
      if (!m.boundary) m.boundary = environmentTargets(m);
      notes.push('範圍改成圓形：圓外鬆弛到環境（邊界是開放的）/ the domain is a circle now: outside it the model relaxes to the environment');
    } else { delete cc.relaxCyl; notes.push('範圍改回方形 / the domain is the whole square again'); }
  }
  // environmental wind: the difference of the environments is added everywhere
  if (windCh) {
    const w = envWind(next).wind, { nz } = m.c, dU = new Float64Array(nz), dV = new Float64Array(nz);
    for (let k = 0; k < nz; k++) { const e = w(m.zc[k]!); dU[k] = e.u - frame.u - m.ub[k]!; dV[k] = e.v - frame.v - m.vb[k]!; }
    addWind(m, dU, dV);
    notes.push(`環境風改成 ${next.wind}${next.wind === 'calm' ? '' : ` ${next.windU} m/s`}（新舊環境風的差加到整個風場）/ environmental wind: ${next.wind}`);
  }
  // sounding: the base state of another environment, the departures stay
  if (soundingCh) {
    m.changeSounding(soundingOf(next)); c.beta = next.sounding === 'wk82' ? 0.2 : 0.3;
    notes.push('探空換掉：風暴的擾動留著，周圍的環境改成新的（近似，不是同一個實驗）/ sounding replaced: the storm stays, the air around it changes (approximate)');
  }
  // sea or land everywhere, the sea temperature of the sea
  const sf = h.phys?.surface;
  if (!child && sf && (ch.has('surface') || ch.has('sst'))) {
    if (ch.has('surface')) { const sea = next.surface === 'sea'; sf.tsk.fill(sea ? next.sst + 273.15 : m.th0[0]! * m.pi0[0]!); sf.wet.fill(sea ? 1 : 0.3); notes.push(sea ? `地表全部改成海洋 ${+next.sst.toFixed(1)} °C / surface: all sea` : '地表全部改成陸地 / surface: all land'); }
    else if (next.surface === 'sea') { for (let i = 0; i < sf.tsk.length; i++) if (sf.wet[i]! >= 0.99) sf.tsk[i] = next.sst + 273.15; notes.push(`海溫改成 ${+next.sst.toFixed(1)} °C / sea temperature ${+next.sst.toFixed(1)} °C`); }
  }
  // the sub-grid, surface and radiation physics
  const physics = PHYS_KEYS.some((k) => ch.has(k)) || ch.has('surface') || ch.has('sst') || soundingCh;
  if (physics) {
    const keep = next.fluxes && h.phys?.surface ? { tsk: h.phys.surface.tsk, wet: h.phys.surface.wet } : null;
    const made = physicsOf({ ...next, dx: m.c.dx, dz: m.c.dz }, m, frame, keep);
    m.physicsTend = null; m.preStep = null;
    h.cfg = made.physics; h.phys = made.physics ? new RegionalPhysics(m, made.physics) : null;
    if (PHYS_KEYS.some((k) => ch.has(k))) notes.push('物理設定已更新 / physics settings updated');
  }
  return { notes, physics };
}
