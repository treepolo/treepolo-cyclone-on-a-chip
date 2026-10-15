// A taller grid inside the running model ("local height extension"): a child of the model over a square part of its domain that reaches higher than the model's lid.
// Inside its footprint it replaces the model: it has the same or a finer grid (whole multiples, across and up), its own lid and Rayleigh layer at its own top Z2, and its
// side boundaries relax to the model below the model's top Z1 and to the undisturbed environment above it. The model takes the child's means back (below Z1) over the
// footprint, tapering to the footprint's edge. Coupled in both directions like the eye nest (twoway.ts), but a separate piece: a square footprint instead of a cylinder,
// levels above the model's top that the model does not have (their targets are the environment, not a continuation of the model's top level), and a feedback only
// for the levels the model has.
//
// Everything is in metres and whole cells of the model (the host); `x0, y0` is the footprint's lower-left corner in the host's coordinates.

import type { BoundaryTargets, RegionalConfig, RegionalModel } from './core.js';
import { makeSampler } from './refine.js';
import { feedbackWeight, type NestGeom, type ParentState } from './twoway.js';

/** most cells of the tall grid */
export const TALL_MAX_CELLS = 30e6;

export interface TallGeom {
  /** refinement ratios: child spacing = host spacing / r across, / rz up */
  r: number; rz: number;
  /** the footprint: lower-left corner at host cell (i0, j0), np host cells across */
  i0: number; j0: number; np: number;
  /** child cells across and levels, spacings (m) */
  nx: number; nz: number; dx: number; dz: number;
  /** the footprint's lower-left corner (m, host coordinates; a footprint that moves with a domain that follows a storm changes it) */
  x0: number; y0: number;
  /** the host's top and the child's (m) */
  Z1: number; Z2: number;
  /** width of the feedback taper inside the footprint's edge (m) */
  Wf: number;
  /** width (child cells) of the edge ring where the relaxation targets are needed */
  ring: number;
  /** the host's levels */
  nzHost: number;
}

/** what the page asks for: the footprint's centre (m, host coordinates) and width (m), the new top (m), the ratios */
export interface TallSpec { cx: number; cy: number; width: number; top: number; r: number; rz: number }

/** relaxation-zone width of the child (cells): as a model's open boundary (build.ts) */
export const tallRelaxCells = (dx: number): number => Math.max(6, Math.round(2500 / dx));

/** host cells between the host's edge and the footprint (the host's own relaxation band when it is open) */
export function tallMargin(host: RegionalConfig): number {
  return host.lateral === 'open' ? (host.relaxCells ?? 5) + 2 : 2;
}

/** The widest footprint (host cells across) the host's domain allows. */
export const tallMaxCols = (host: RegionalConfig): number => Math.min(host.nx, host.ny) - 2 * tallMargin(host);

/**
 * The geometry of a tall grid in the host `host`; a message when it does not fit. The footprint is whole host cells (at least enough for the child's relaxation zone
 * to leave an interior), moved inside the margin when it would cross it; the spacings are the host's divided by whole numbers.
 */
export function tallGeometry(host: RegionalConfig, spec: TallSpec): TallGeom | string {
  const r = Math.round(spec.r), rz = Math.round(spec.rz);
  if (!(r >= 1) || !(rz >= 1) || r > 32 || rz > 8) return `倍數要是整數：水平 1–32、垂直 1–8 / the ratios must be whole numbers: 1-32 across, 1-8 up (${spec.r}, ${spec.rz})`;
  const dx = host.dx / r, dz = host.dz / rz, Z1 = host.nz * host.dz;
  const nz = Math.round(spec.top / dz), Z2 = nz * dz;
  if (nz <= host.nz * rz) return `新頂要比原本的模式頂（${(Z1 / 1000).toFixed(1)} km）高至少一層 / the new top must be higher than the model's top (${(Z1 / 1000).toFixed(1)} km) by at least a level`;
  const margin = tallMargin(host), maxNp = tallMaxCols(host), rc = tallRelaxCells(dx);
  const np = Math.round(spec.width / host.dx), minNp = Math.max(8, Math.ceil(4 * rc / r));
  if (np < minNp) return `範圍至少要 ${minNp} 個外圍格（${(minNp * host.dx / 1000).toFixed(0)} km）：邊界鬆弛帶之內要留出內部 / the footprint must be at least ${minNp} host cells across`;
  if (np > maxNp) return `範圍太大：最多 ${maxNp} 個外圍格（${(maxNp * host.dx / 1000).toFixed(0)} km，扣掉邊緣的 ${margin} 格）；要整個範圍加高請用「模式頂」/ the footprint is too wide: at most ${maxNp} host cells (the edge keeps ${margin} cells); for the whole domain use the model top`;
  const nxc = np * r;
  if (nxc > 1024) return `太寬（${nxc} 格，上限 1024），請縮小範圍或降低水平倍數 / too many columns across (${nxc}, at most 1024)`;
  if (nz > 400) return `垂直層太多（${nz} 層，上限 400）/ too many levels (${nz}, at most 400)`;
  const cells = nxc * nxc * nz;
  if (cells > TALL_MAX_CELLS) return `${(cells / 1e6).toFixed(1)} M 格，上限 ${TALL_MAX_CELLS / 1e6} M / ${(cells / 1e6).toFixed(1)} M cells, at most ${TALL_MAX_CELLS / 1e6} M`;
  const i0 = Math.max(margin, Math.min(host.nx - margin - np, Math.round(spec.cx / host.dx - np / 2))), j0 = Math.max(margin, Math.min(host.ny - margin - np, Math.round(spec.cy / host.dy - np / 2)));
  const W = np * host.dx, Wf = Math.min(Math.max(2 * host.dx, 0.15 * W), 0.45 * W);
  return { r, rz, i0, j0, np, nx: nxc, nz, dx, dz, x0: i0 * host.dx, y0: j0 * host.dy, Z1, Z2, Wf, ring: rc + 2, nzHost: host.nz };
}

/** Child cells of a tall grid. */
export const tallCells = (g: TallGeom): number => g.nx * g.nx * g.nz;

/** The least new top (m) that keeps the child's Rayleigh layer (its depth: as the model's, min(6 km, 0.3 top)) above the host's top. */
export function tallTopAdvice(Z1: number): number { return Z1 + 6000; }

/** Feedback weight at a host cell of the footprint (indices relative to the host): 0 at the footprint's edge, 1 from the taper's width inward, sin^2 between. */
export function tallWeight(g: TallGeom, ip: number, jp: number, dxh: number, dyh: number): number {
  const a = g.i0 + g.np - ip - 0.5, b = ip - g.i0 + 0.5, c = g.j0 + g.np - jp - 0.5, d = jp - g.j0 + 0.5;
  const dist = Math.min(a * dxh, b * dxh, c * dyh, d * dyh);
  if (dist <= 0) return 0;
  return dist >= g.Wf ? 1 : Math.sin(0.5 * Math.PI * dist / g.Wf) ** 2;
}

/** Empty relaxation targets for the child. */
export const tallEmptyTargets = (c: RegionalModel): BoundaryTargets =>
  ({ u: new Float64Array(c.size), v: new Float64Array(c.size), th: new Float64Array(c.size), qv: c.scalars[0] ? new Float64Array(c.size) : null, pp: new Float64Array(c.size) });

/**
 * Relaxation targets of the child at a fraction alpha of the host's step: below the host's top the host's fields interpolated in space and in time between `old` (the host
 * at the start of the step; null: the host now) and the host now; above it the child's own base state (no departure). Within half a host level either side of Z1 the
 * departure is faded out linearly (the host has no level above its top). Only where the child uses them: its edge ring and its Rayleigh layer.
 */
export function tallTargets(p: RegionalModel, old: ParentState | null, child: RegionalModel, g: TallGeom, alpha: number, out: BoundaryTargets): void {
  const c = child.c, pc = p.c, sample = makeSampler(p), x0 = g.x0, y0 = g.y0;
  const zSponge = c.nz * c.dz - c.dampDepth;
  const used = (i: number, j: number, k: number): boolean => (c.dampDepth > 0 && (k + 0.5) * c.dz > zSponge) || Math.min(i, j, c.nx - 1 - i, c.ny - 1 - j) < g.ring;
  const both = (now: Float64Array, before: Float64Array | null | undefined, x: number, y: number, z: number, sx: number, sy: number, base?: Float64Array): number => {
    const a = sample(now, x, y, z, sx, sy, false, base);
    return before && alpha < 1 ? (1 - alpha) * sample(before, x, y, z, sx, sy, false, base) + alpha * a : a;
  };
  for (let k = 0; k < c.nz; k++) {
    const z = (k + 0.5) * c.dz;
    // the weight of the host's departure: 1 up to half a host level below its top, 0 from half a level above it
    const wh = Math.max(0, Math.min(1, (g.Z1 + 0.5 * pc.dz - z) / pc.dz));
    for (let j = 0; j <= c.ny; j++) for (let i = 0; i <= c.nx; i++) {
      const q = child.idx(i, j, k), ic = Math.min(i, c.nx - 1), jc = Math.min(j, c.ny - 1);
      if (!used(ic, jc, k)) continue;
      const xf = x0 + i * c.dx, yf = y0 + j * c.dy, xc = x0 + (i + 0.5) * c.dx, yc = y0 + (j + 0.5) * c.dy;
      if (j < c.ny) out.u[q] = child.ub[k]! + (wh > 0 ? wh * both(p.u, old?.u, xf, yc, z, 0, 0.5, p.ub) : 0);
      if (i < c.nx) out.v[q] = child.vb[k]! + (wh > 0 ? wh * both(p.v, old?.v, xc, yf, z, 0.5, 0, p.vb) : 0);
      if (i < c.nx && j < c.ny) {
        out.th[q] = child.th0[k]! + (wh > 0 ? wh * both(p.th, old?.th, xc, yc, z, 0.5, 0.5, p.th0) : 0);
        if (out.pp) out.pp[q] = wh > 0 ? wh * both(p.pp, old?.pp, xc, yc, z, 0.5, 0.5) : 0;
        if (out.qv && p.scalars[0]) out.qv[q] = Math.max(0, child.qv0[k]! + (wh > 0 ? wh * both(p.scalars[0], old?.qv, xc, yc, z, 0.5, 0.5, p.qv0) : 0));
      }
    }
  }
}

/** A child box over whole cells of its parent: lower-left cell, cells across, ratios across and up (what restrictInto needs). */
export interface BoxGeom { r: number; rz: number; i0: number; j0: number; np: number }

/**
 * Restriction of a child into its parent over a box of whole parent cells, blended by `weight(ip, jp)` (0..1; ip, jp: the parent's cell), for the parent's lowest `nk` levels: the parent takes
 * the child's mean over each of its cells (u, v over the coincident faces, w over the coincident level, theta, pi' and the scalars over the r x r x rz child cells; departures from the
 * child's base profiles with the parent's added back, as the targets). The top face's w is not touched.
 */
export function restrictInto(child: RegionalModel, p: RegionalModel, g: BoxGeom, weight: (ip: number, jp: number) => number, nk: number): void {
  const { r, rz } = g, ns = Math.min(p.scalars.length, child.scalars.length);
  const none = new Float64Array(child.c.nz);
  for (let b = 0; b < g.np; b++) for (let a = 0; a < g.np; a++) {
    const ip = g.i0 + a, jp = g.j0 + b;
    const wt = weight(ip, jp);
    if (wt <= 0) continue;
    const ic = a * r, jc = b * r;
    const put = (dst: Float64Array, q: number, mean: number): void => { dst[q] = (1 - wt) * dst[q]! + wt * mean; };
    for (let k = 0; k < nk; k++) {
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

/** Feedback of the tall grid into the host: over the footprint, tapering to its edge, for the levels the host has (restrictInto). */
export function tallFeedback(child: RegionalModel, p: RegionalModel, g: TallGeom): void {
  restrictInto(child, p, g, (ip, jp) => tallWeight(g, ip, jp, p.c.dx, p.c.dy), p.c.nz);
}

/** An eye nest (a cylinder in the host, twoway.ts) that lies inside a tall grid's footprint, as a child of the tall grid: its box in the tall grid's cells, and the levels of the tall grid it reaches. */
export interface EyeInTall { geom: NestGeom; nk: number }

/**
 * Whether the eye nest `e` (geometry in the host's cells) can feed back into the tall grid `t` as well as into the host: its ratios are whole multiples of the tall grid's, and its box lies inside
 * the footprint, clear of the tall grid's edge ring. A message when not (then the eye nest feeds the host only, and the tall grid's own feedback wins over it).
 */
export function eyeInTall(t: TallGeom, e: NestGeom): EyeInTall | string {
  if (e.r % t.r !== 0 || e.rz % t.rz !== 0) return `眼區的倍數（${e.r}、${e.rz}）不是局部加高的倍數（${t.r}、${t.rz}）的整數倍 / the eye nest's ratios are not whole multiples of the tall grid's`;
  const clear = Math.ceil(t.ring / t.r) + 1;
  if (e.i0 - t.i0 < clear || e.j0 - t.j0 < clear || t.i0 + t.np - (e.i0 + e.np) < clear || t.j0 + t.np - (e.j0 + e.np) < clear) return '眼區的框離局部加高的邊太近 / the eye nest lies too near the edge of the tall grid';
  const geom: NestGeom = { r: e.r / t.r, rz: e.rz / t.rz, i0: (e.i0 - t.i0) * t.r, j0: (e.j0 - t.j0) * t.r, np: e.np * t.r, nx: e.nx, nz: e.nz, dx: e.dx, dz: e.dz, cx: e.cx - t.x0, cy: e.cy - t.y0, R: e.R, W: e.W, Wf: e.Wf };
  return { geom, nk: t.nzHost * t.rz };
}

/** The eye nest's feedback into the tall grid (the cylinder's weights, the tall grid's levels below the host's top). */
export function eyeToTallFeedback(eye: RegionalModel, tall: RegionalModel, g: EyeInTall): void {
  const td = tall.c;
  restrictInto(eye, tall, g.geom, (ip, jp) => feedbackWeight(g.geom, Math.hypot((ip + 0.5) * td.dx - g.geom.cx, (jp + 0.5) * td.dy - g.geom.cy)), g.nk);
}

/** Surface arrays ([j][i]: skin temperature, wetness) of the host sampled (bilinearly) at the child's columns. */
export function tallSurface(p: RegionalModel, g: TallGeom, src: ArrayLike<number>): Float64Array {
  const pc = p.c, out = new Float64Array(g.nx * g.nx);
  const ax = (q: number, n: number): [number, number, number] => { const c = Math.max(0, Math.min(n - 1, q)), i0 = Math.min(Math.floor(c), Math.max(0, n - 2)); return n < 2 ? [0, 0, 0] : [i0, i0 + 1, c - i0]; };
  for (let j = 0; j < g.nx; j++) for (let i = 0; i < g.nx; i++) {
    const [a0, a1, wx] = ax((g.x0 + (i + 0.5) * g.dx) / pc.dx - 0.5, pc.nx), [b0, b1, wy] = ax((g.y0 + (j + 0.5) * g.dx) / pc.dy - 0.5, pc.ny);
    out[j * g.nx + i] = (1 - wy) * ((1 - wx) * src[b0 * pc.nx + a0]! + wx * src[b0 * pc.nx + a1]!) + wy * ((1 - wx) * src[b1 * pc.nx + a0]! + wx * src[b1 * pc.nx + a1]!);
  }
  return out;
}
