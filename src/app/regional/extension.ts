// An outer grid around the running model (the page's "范圍加大": a larger domain whose added part has its own, coarser spacing). The running model, with all its
// state, stays exactly as it is, as the inner grid; a coarse model of the larger domain is made around it, and the inner grid's lateral boundaries relax to it
// (one way: the outer grid drives the inner grid's edges, nothing flows back).
//
// The outer model is made from the set-up (the same environment: sounding, wind, sea), carries the inner grid's base wind, and inside the inner grid's box takes the
// coarse means of the inner state (refine.ts coarsenInto) - so the storm is in it, coarsely - and the plain environment elsewhere. Both grids have the same top and
// the same vertical spacing. The inner box is whole outer cells: the outer spacing is a whole multiple of the inner one that divides the inner domain.
//
// The outer model runs by itself, its own steps ahead of the inner grid's (worker.ts): for each inner step the outer model is stepped until its last step covers it,
// and the inner relaxation targets are the outer state interpolated in space and in time between the start and the end of that outer step (nestTargets, GpuNest.targets).
// The targets are needed only in an edge ring of the inner grid (and its sponge layer): NestGeom.ring.

import type { RegionalModel } from '../../regional/core.js';
import { IceMicrophysics } from '../../regional/ice.js';
import { RegionalPhysics, type RegionalPhysicsConfig } from '../../regional/physics.js';
import { coarsenInto } from '../../regional/refine.js';
import type { NestGeom } from '../../regional/twoway.js';
import { buildModel } from './build.js';
import { addWind } from './liveSettings.js';
import { autoDt, type RegionalSetup } from './setup.js';

/** most cells of the outer grid */
export const OUTER_MAX_CELLS = 30e6;

export interface Outer {
  model: RegionalModel; mp: IceMicrophysics; cfg: RegionalPhysicsConfig | null; phys: RegionalPhysics | null;
  /** the inner grid's box in it */
  geom: NestGeom;
  /** the set-up it was made from, and its own step (s) */
  setup: RegionalSetup; dt0: number;
}

/**
 * The outer spacing for a wished one: a whole multiple (at least 2, at most 32) of the inner spacing that divides the inner domain into at least 4 whole outer cells, the nearest to the
 * wish; a message when there is none.
 */
export function outerSpacing(innerNx: number, innerDx: number, want: number): number | string {
  const r0 = Math.max(2, Math.min(32, Math.round(want / innerDx)));
  let best = 0;
  for (let r = 2; r <= 32; r++) if (innerNx % r === 0 && innerNx / r >= 4 && (best === 0 || Math.abs(r - r0) < Math.abs(best - r0) || (Math.abs(r - r0) === Math.abs(best - r0) && r < best))) best = r;
  if (!best) return `內圈的 ${innerNx} 格沒有適合的整數倍（2 到 32，且至少 4 個外圈格）可以當外圈格距 / the ${innerNx} inner cells cannot be divided into whole outer cells (2 to 32 inner cells wide)`;
  return best * innerDx;
}

/**
 * The outer grid for the inner model `inner` of set-up `setup` (its frame velocity `frame` over the ground): the domain at least `wantL` m across, spacing near `wantDx` m.
 * `acc`: the inner model's accumulated rain and snow, `surface`: its skin temperature and wetness (painted sea and land). A message when it cannot be built.
 */
export function buildOuter(inner: RegionalModel, setup: RegionalSetup, frame: { u: number; v: number }, wantL: number, wantDx: number, gpuOk: boolean,
  acc: { rain: Float64Array; snow: Float64Array }, surface: { tsk: Float64Array; wet: Float64Array } | null): Outer | string {
  const c = inner.c, sp = outerSpacing(c.nx, c.dx, wantDx);
  if (typeof sp === 'string') return sp;
  if (c.nx !== c.ny || Math.abs(c.dx - c.dy) > 1e-9) return '內圈不是正方形 / the inner grid is not square';
  const np = c.nx / Math.round(sp / c.dx), nxp = Math.ceil(wantL / sp - 1e-9);
  const nT = nxp % 2 === np % 2 ? nxp : nxp + 1;
  if (nT <= np) return `外圈的總範圍要比內圈（${(c.nx * c.dx / 1000).toFixed(0)} km）大 / the outer domain must be larger than the inner one (${(c.nx * c.dx / 1000).toFixed(0)} km)`;
  if (nT * nT * c.nz > OUTER_MAX_CELLS) return `外圈 ${(nT * nT * c.nz / 1e6).toFixed(1)} M 格，上限 ${OUTER_MAX_CELLS / 1e6} M / the outer grid has ${(nT * nT * c.nz / 1e6).toFixed(1)} M cells, at most ${OUTER_MAX_CELLS / 1e6} M`;
  const i0 = (nT - np) / 2, r = Math.round(sp / c.dx);
  const os: RegionalSetup = { ...setup, preset: 'custom', L: nT * sp, dx: sp, dz: c.dz, top: c.nz * c.dz, dt: 0, init: 'none', initAmp: 0, follow: false, shape: 'square' };
  const b = buildModel(os, gpuOk), m = b.model;
  if (m.c.nx !== nT || m.c.nz !== c.nz) return `內部錯誤：外圈網格 ${m.c.nx} x ${m.c.nz}，應為 ${nT} x ${c.nz} / internal error: outer grid ${m.c.nx} x ${m.c.nz}, expected ${nT} x ${c.nz}`;
  // the inner grid's frame and base wind (a change of the environment by an interaction included)
  if (frame.u !== b.frame.u || frame.v !== b.frame.v) m.shiftFrame(frame.u - b.frame.u, frame.v - b.frame.v);
  const dU = Float64Array.from(inner.ub, (x, k) => x - m.ub[k]!), dV = Float64Array.from(inner.vb, (x, k) => x - m.vb[k]!);
  addWind(m, dU, dV);
  if (m.boundary) { const o = m.boundary; for (let k = 0; k < c.nz; k++) for (let q = k * m.plane; q < (k + 1) * m.plane; q++) { o.u[q] = m.ub[k]!; o.v[q] = m.vb[k]!; } }
  const cfg = b.physics;
  if (cfg) cfg.frameVel = { ...frame };
  const phys = cfg ? new RegionalPhysics(m, cfg) : null, mp = new IceMicrophysics(m);
  // inside the box: the coarse means of the inner state (and of its painted surface and accumulated rain)
  const x0 = i0 * sp, y0 = i0 * sp, sf = phys?.surface ?? null;
  coarsenInto(inner, m, x0, y0, 0, acc, { rain: mp.rainAcc, snow: mp.snowAcc }, surface && sf ? [surface.tsk, surface.wet] : undefined, surface && sf ? [sf.tsk, sf.wet] : undefined);
  m.time = inner.time; m.steps = 0;
  if (cfg && sf) cfg.surface = { tsk: sf.tsk, wet: sf.wet };
  const ring = (c.relaxCells ?? Math.max(6, Math.round(2500 / c.dx))) + 2;
  const geom: NestGeom = { r, rz: 1, i0, j0: i0, np, nx: c.nx, nz: c.nz, dx: c.dx, dz: c.dz, cx: (i0 + np / 2) * sp, cy: (i0 + np / 2) * sp, R: np * sp / 2, W: 0, Wf: 0, x0, y0, ring };
  return { model: m, mp, cfg, phys, geom, setup: os, dt0: autoDt(os) };
}

/** Make the inner model's lateral boundaries open, relaxing (like any open domain) to the targets that the outer grid sets; returns what to undo it with. */
export function openInner(inner: RegionalModel): { lateral: 'periodic' | 'open'; boundary: RegionalModel['boundary']; relaxCells?: number; relaxTau?: number } {
  const c = inner.c as { lateral?: 'periodic' | 'open'; relaxCells?: number; relaxTau?: number };
  const was = { lateral: c.lateral ?? 'periodic', boundary: inner.boundary, relaxCells: c.relaxCells, relaxTau: c.relaxTau };
  if (c.lateral !== 'open') {
    c.lateral = 'open'; c.relaxCells = Math.max(6, Math.round(2500 / inner.c.dx)); c.relaxTau = 300;
    const z = (): Float64Array => new Float64Array(inner.size);
    inner.boundary = { u: z(), v: z(), th: z(), qv: z(), pp: z() };
  }
  return was;
}
