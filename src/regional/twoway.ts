// Two-way nest: a finer grid (horizontally and vertically) in a cylinder at the centre of a running regional model, for
// the eye and eyewall of a tropical cyclone. The parent domain follows the storm, so the cylinder, fixed at the parent's
// centre, stays on the storm.
//
// The child grid is a square box aligned with the parent grid (integer refinement ratios r across and rz up), with the
// same top and the same sounding, so its base state is the parent's. Each parent step:
//   1. the parent advances from t to t + dt;
//   2. the child advances in n sub-steps; around the cylinder (radius R) its fields relax toward the parent's,
//      interpolated in space (trilinear, at each variable's staggered position) and in time (between the parent's
//      states at t and t + dt): no relaxation inside R, growing over W outside it, full beyond (the box's corners); the
//      lid's sponge layer also damps toward them;
//   3. feedback: inside the cylinder the parent takes the mean of the child over each of its cells (u and v over the
//      coincident faces, w over the coincident level, the rest over the r x r x rz cells), fully up to R - Wf and
//      blended out (cos^2) toward R, so the relaxation ring and the feedback region never overlap.
// This is the usual two-way nesting of storm models (e.g. WRF, HWRF), with a circular instead of a square interface.

import type { BoundaryTargets, RegionalConfig, RegionalModel } from './core.js';
import { makeSampler } from './refine.js';

export interface NestGeom {
  /** refinement ratios: child spacing = parent spacing / r across, / rz up */
  r: number; rz: number;
  /** the child box: its lower-left corner at parent cell (i0, j0), np parent cells across */
  i0: number; j0: number; np: number;
  /** child cells across and levels, spacings (m) */
  nx: number; nz: number; dx: number; dz: number;
  /** the cylinder: centre (parent coordinates, m) and radius (m); relaxation ring width outside it and feedback taper
   *  inside it (m) */
  cx: number; cy: number; R: number; W: number; Wf: number;
}

/** relaxation time scale of the child at full strength (s) */
export const NEST_TAU = 60;

/**
 * Geometry of a nest of radius R (m) with spacings near dx, dz (m) in the parent `pc`; a message when it does not fit.
 * The spacings are rounded to the parent's divided by whole numbers (2-32 across, 1-8 up).
 */
export function nestGeometry(pc: RegionalConfig, R: number, dx: number, dz: number): NestGeom | string {
  const r = Math.max(2, Math.min(32, Math.round(pc.dx / dx))), rz = Math.max(1, Math.min(8, Math.round(pc.dz / dz)));
  const dxc = pc.dx / r, dzc = pc.dz / rz;
  if (R < 2 * pc.dx) return `圓柱半徑至少要 ${(2 * pc.dx / 1000).toFixed(0)} km（外圍的兩格）/ the radius must be at least ${(2 * pc.dx / 1000).toFixed(0)} km (two outer cells)`;
  const W = Math.max(6 * dxc, 1.5 * pc.dx), Wf = Math.min(Math.max(pc.dx, 0.25 * R), 0.5 * R);
  const np = 2 * Math.ceil((R + W) / pc.dx) + 2;
  const margin = pc.lateral === 'open' ? (pc.relaxCells ?? 5) + 2 : 0;
  if (np > pc.nx - 2 * margin || np > pc.ny - 2 * margin) return '圓柱太大，外圍區域放不下 / the cylinder does not fit in the outer domain';
  if (pc.nz * rz > 200) return `垂直層太多（${pc.nz * rz} 層，上限 200），請加大 Δz / too many levels (${pc.nz * rz}, at most 200): use a larger Δz`;
  if (np * r > 1024) return `細化區太寬（${np * r} 格，上限 1024），請縮小半徑或加大 Δx / too many columns across (${np * r}, at most 1024)`;
  const i0 = Math.floor((pc.nx - np) / 2), j0 = Math.floor((pc.ny - np) / 2);
  return { r, rz, i0, j0, np, nx: np * r, nz: pc.nz * rz, dx: dxc, dz: dzc, cx: (i0 + np / 2) * pc.dx, cy: (j0 + np / 2) * pc.dy, R, W, Wf };
}

/** Child cells of a nest. */
export const nestCells = (g: NestGeom): number => g.nx * g.nx * g.nz;

/** The child's relaxation: a ring around the cylinder (in child cells, from the box centre) and its time scale. */
export function nestRelax(g: NestGeom): { relaxCyl: { r: number; w: number }; relaxTau: number } {
  return { relaxCyl: { r: g.R / g.dx, w: g.W / g.dx }, relaxTau: NEST_TAU };
}

/** Feedback weight at distance d (m) from the cylinder's axis: 1 up to R - Wf, cos^2 down to 0 at R. */
export function feedbackWeight(g: NestGeom, d: number): number {
  const a = g.R - g.Wf;
  return d <= a ? 1 : d >= g.R ? 0 : Math.cos(0.5 * Math.PI * (d - a) / g.Wf) ** 2;
}

/** The parent's state at the start of a step (the relaxation targets are interpolated in time from it). */
export interface ParentState { u: Float64Array; v: Float64Array; th: Float64Array; pp: Float64Array; qv: Float64Array | null }
export const parentState = (p: RegionalModel): ParentState =>
  ({ u: Float64Array.from(p.u), v: Float64Array.from(p.v), th: Float64Array.from(p.th), pp: Float64Array.from(p.pp), qv: p.scalars[0] ? Float64Array.from(p.scalars[0]) : null });

/** Empty relaxation targets for a child model. */
export const emptyTargets = (c: RegionalModel): BoundaryTargets =>
  ({ u: new Float64Array(c.size), v: new Float64Array(c.size), th: new Float64Array(c.size), qv: c.scalars[0] ? new Float64Array(c.size) : null, pp: new Float64Array(c.size) });

/**
 * Relaxation targets of the child at a fraction alpha of the parent step: the parent's fields interpolated in space to
 * each child point and in time between `old` (the parent at the start of the step; null: the parent now) and the parent
 * now. Only where the child uses them: its relaxation ring and beyond, and its sponge layer below the lid.
 */
export function nestTargets(p: RegionalModel, old: ParentState | null, child: RegionalModel, g: NestGeom, alpha: number, out: BoundaryTargets): void {
  const pc = p.c, c = child.c, sample = makeSampler(p), x0 = g.i0 * pc.dx, y0 = g.j0 * pc.dy;
  const zSponge = c.nz * c.dz - c.dampDepth, cyl = c.relaxCyl;
  const used = (i: number, j: number, k: number): boolean => {
    if (c.dampDepth > 0 && (k + 0.5) * c.dz > zSponge) return true;
    if (!cyl) return true;
    return Math.hypot(i + 0.5 - c.nx / 2, j + 0.5 - c.ny / 2) > cyl.r;
  };
  // theta, qv and the wind as departures from the parent's base profiles, the child's added back (makeSampler)
  const both = (now: Float64Array, before: Float64Array | null | undefined, x: number, y: number, z: number, sx: number, sy: number, base?: Float64Array): number => {
    const a = sample(now, x, y, z, sx, sy, false, base);
    return before && alpha < 1 ? (1 - alpha) * sample(before, x, y, z, sx, sy, false, base) + alpha * a : a;
  };
  for (let k = 0; k < c.nz; k++) {
    const z = (k + 0.5) * c.dz;
    for (let j = 0; j <= c.ny; j++) for (let i = 0; i <= c.nx; i++) {
      const q = child.idx(i, j, k), ic = Math.min(i, c.nx - 1), jc = Math.min(j, c.ny - 1);
      if (!used(ic, jc, k)) continue;
      const xf = x0 + i * c.dx, yf = y0 + j * c.dy, xc = x0 + (i + 0.5) * c.dx, yc = y0 + (j + 0.5) * c.dy;
      if (j < c.ny) out.u[q] = child.ub[k]! + both(p.u, old?.u, xf, yc, z, 0, 0.5, p.ub);
      if (i < c.nx) out.v[q] = child.vb[k]! + both(p.v, old?.v, xc, yf, z, 0.5, 0, p.vb);
      if (i < c.nx && j < c.ny) {
        out.th[q] = child.th0[k]! + both(p.th, old?.th, xc, yc, z, 0.5, 0.5, p.th0);
        if (out.pp) out.pp[q] = both(p.pp, old?.pp, xc, yc, z, 0.5, 0.5);
        if (out.qv && p.scalars[0]) out.qv[q] = Math.max(0, child.qv0[k]! + both(p.scalars[0], old?.qv, xc, yc, z, 0.5, 0.5, p.qv0));
      }
    }
  }
}

/**
 * Feedback: inside the cylinder the parent takes the child's mean over each of its cells (u, v over the coincident faces,
 * w over the coincident level, theta, pi' and the scalars over the r x r x rz child cells), blended as feedbackWeight.
 * theta, qv and the wind are averaged as departures from the child's base profiles and the parent's added back (as the
 * targets), so a child that is the interpolated parent gives the parent back.
 */
export function nestFeedback(child: RegionalModel, p: RegionalModel, g: NestGeom): void {
  const pc = p.c, { r, rz } = g, ns = Math.min(p.scalars.length, child.scalars.length);
  const none = new Float64Array(child.c.nz);
  for (let b = 0; b < g.np; b++) for (let a = 0; a < g.np; a++) {
    const ip = g.i0 + a, jp = g.j0 + b;
    const wt = feedbackWeight(g, Math.hypot((ip + 0.5) * pc.dx - g.cx, (jp + 0.5) * pc.dy - g.cy));
    if (wt <= 0) continue;
    const ic = a * r, jc = b * r;
    const put = (dst: Float64Array, q: number, mean: number): void => { dst[q] = (1 - wt) * dst[q]! + wt * mean; };
    for (let k = 0; k < pc.nz; k++) {
      const q = p.idx(ip, jp, k), kc = k * rz;
      let su = 0, sv = 0;
      for (let dk = 0; dk < rz; dk++) for (let d = 0; d < r; d++) {
        su += child.u[child.idx(ic, jc + d, kc + dk)]! - child.ub[kc + dk]!;
        sv += child.v[child.idx(ic + d, jc, kc + dk)]! - child.vb[kc + dk]!;
      }
      put(p.u, q, p.ub[k]! + su / (r * rz)); put(p.v, q, p.vb[k]! + sv / (r * rz));
      const cell = (arr: Float64Array, base: Float64Array): number => {
        let s = 0;
        for (let dk = 0; dk < rz; dk++) for (let dj = 0; dj < r; dj++) for (let di = 0; di < r; di++) s += arr[child.idx(ic + di, jc + dj, kc + dk)]! - base[kc + dk]!;
        return s / (r * r * rz);
      };
      put(p.th, q, p.th0[k]! + cell(child.th, child.th0)); put(p.pp, q, cell(child.pp, none));
      for (let s = 0; s < ns; s++) {
        put(p.scalars[s]!, q, s === 0 ? p.qv0[k]! + cell(child.scalars[0]!, child.qv0) : cell(child.scalars[s]!, none));
        if (p.scalars[s]![q]! < 0) p.scalars[s]![q] = 0;
      }
      if (k > 0) {
        let sw = 0;
        for (let dj = 0; dj < r; dj++) for (let di = 0; di < r; di++) sw += child.w[child.idx(ic + di, jc + dj, kc)]!;
        put(p.w, q, sw / (r * r));
      }
    }
  }
}

/** Surface arrays ([j][i], e.g. skin temperature, wetness) of the parent sampled (bilinearly) at the child's columns. */
export function nestSurface(p: RegionalModel, g: NestGeom, src: ArrayLike<number>): Float64Array {
  const pc = p.c, out = new Float64Array(g.nx * g.nx), x0 = g.i0 * pc.dx, y0 = g.j0 * pc.dy;
  const ax = (q: number, n: number): [number, number, number] => { const c = Math.max(0, Math.min(n - 1, q)), i0 = Math.min(Math.floor(c), Math.max(0, n - 2)); return n < 2 ? [0, 0, 0] : [i0, i0 + 1, c - i0]; };
  for (let j = 0; j < g.nx; j++) for (let i = 0; i < g.nx; i++) {
    const [a0, a1, wx] = ax((x0 + (i + 0.5) * g.dx) / pc.dx - 0.5, pc.nx), [b0, b1, wy] = ax((y0 + (j + 0.5) * g.dx) / pc.dy - 0.5, pc.ny);
    out[j * g.nx + i] = (1 - wy) * ((1 - wx) * src[b0 * pc.nx + a0]! + wx * src[b0 * pc.nx + a1]!) + wy * ((1 - wx) * src[b1 * pc.nx + a0]! + wx * src[b1 * pc.nx + a1]!);
  }
  return out;
}
