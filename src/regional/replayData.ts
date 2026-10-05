// What a recorded replay frame holds besides the 3-D display bytes, and its 16-bit packing (shared by the worker, which
// records and analyses, and the page, which keeps and saves them).
//
// Tiers of a frame: 'view' (only what the 3-D view draws), 'maps' (+ every composite / surface map), 'full' (+ the model
// state itself, 16-bit per value, from which any chart of any time can be computed again: slices, sections, soundings,
// radius-height means, the maps). A frame always carries the numbers of the time series (the frame statistics).

import { RegionalModel, type RegionalConfig } from './core.js';

export type ReplayTier = 'view' | 'maps' | 'full';
export const TIERS: ReplayTier[] = ['view', 'maps', 'full'];

/** an array packed to 16 bits between its extremes (NaN is kept as 65535) */
export interface Q16 { lo: number; hi: number; q: Uint16Array }

export function quant(a: ArrayLike<number>): Q16 {
  const n = a.length;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) { const v = a[i]!; if (v < lo) lo = v; if (v > hi) hi = v; }   // NaN never compares
  if (!(hi >= lo)) { lo = 0; hi = 0; }
  const q = new Uint16Array(n), k = hi > lo ? 65534 / (hi - lo) : 0;
  for (let i = 0; i < n; i++) { const v = a[i]!; q[i] = v === v ? Math.round((v - lo) * k) : 65535; }
  return { lo, hi, q };
}

/** Unpack into `out` (a new Float32Array when absent). */
export function dequant<T extends Float32Array | Float64Array = Float32Array>(p: Q16, out?: T): T {
  const n = p.q.length, o = (out ?? new Float32Array(n)) as T, s = (p.hi - p.lo) / 65534;
  for (let i = 0; i < n; i++) { const c = p.q[i]!; o[i] = c === 65535 ? NaN : p.lo + c * s; }
  return o;
}

/** Everything of the model that does not change in time and the charts need (a model to compute charts on is made from it). */
export interface ReplayMeta {
  c: RegionalConfig;
  nsc: number;
  /** base state per level: th0, qv0, pi0, rho0 (nz), th0f, rho0f, pi0f (nz + 1), ub, vb (nz) */
  base: { th0: number[]; qv0: number[]; pi0: number[]; rho0: number[]; th0f: number[]; rho0f: number[]; pi0f: number[]; ub: number[]; vb: number[] };
  experiment: string;
  tc: boolean; sea: boolean;
  /** land mask ([j][i], 1 land) when the surface has land */
  land: Uint8Array | null;
}

/** The horizontal fields ([j][i]) that are not part of the state: accumulations, rates, swaths, sea-surface temperature. */
export interface ReplayAux {
  rain: Float32Array; snow: Float32Array;
  rate: Float32Array | null; cu: Float32Array | null; uh: Float32Array | null; wind: Float32Array | null; sst: Float32Array | null;
  /** the model's frame velocity (m/s) */
  vel: { u: number; v: number };
}

/** The recorded part of a frame beyond the 3-D bytes. */
export interface ReplayRec {
  tier: ReplayTier;
  /** which meta (ids count up per model in a run) */
  metaId: number;
  /** only on the first frame of a model */
  meta?: ReplayMeta;
  aux: ReplayAux | null;
  /** 'maps' and 'full': every map, packed */
  maps: Record<string, Q16> | null;
  /** 'full': u, v, w, th, pp and the scalars, whole padded arrays */
  state: Q16[] | null;
  /** the centre of the radius-height means of a tropical cyclone (the main vortex; domain coordinates, m) */
  centre: { x: number; y: number } | null;
}

/** Bytes a recorded part holds (memory accounting and file size). */
export function recBytes(r: ReplayRec | null | undefined): number {
  if (!r) return 0;
  let n = 0;
  if (r.state) for (const s of r.state) n += s.q.byteLength;
  if (r.maps) for (const s of Object.values(r.maps)) n += s.q.byteLength;
  if (r.aux) { const a = r.aux; n += a.rain.byteLength + a.snow.byteLength + (a.rate?.byteLength ?? 0) + (a.cu?.byteLength ?? 0) + (a.uh?.byteLength ?? 0) + (a.wind?.byteLength ?? 0) + (a.sst?.byteLength ?? 0); }
  return n;
}

/** The meta of a model (what a run knows besides the model: its experiment, kind of surface and land mask). */
export function metaOf(m: RegionalModel, extra: { experiment: string; tc: boolean; sea: boolean; land: Uint8Array | null }): ReplayMeta {
  const a = (x: Float64Array): number[] => Array.from(x);
  return { c: JSON.parse(JSON.stringify(m.c)) as RegionalConfig, nsc: m.scalars.length,
    base: { th0: a(m.th0), qv0: a(m.qv0), pi0: a(m.pi0), rho0: a(m.rho0), th0f: a(m.th0f), rho0f: a(m.rho0f), pi0f: a(m.pi0f), ub: a(m.ub), vb: a(m.vb) }, ...extra };
}

/** A model with the grid and base state of a meta (its state is empty: loadState fills it); charts are computed on it. */
export function modelFromMeta(meta: ReplayMeta): RegionalModel {
  const m = new RegionalModel(meta.c, () => ({ theta: 300, qv: 0 }), meta.nsc), b = meta.base;
  m.th0.set(b.th0); m.qv0.set(b.qv0); m.pi0.set(b.pi0); m.rho0.set(b.rho0); m.th0f.set(b.th0f); m.rho0f.set(b.rho0f); m.pi0f.set(b.pi0f); m.ub.set(b.ub); m.vb.set(b.vb);
  return m;
}

/** The prognostic arrays of a model in the order of a packed state: u, v, w, theta, pi', the scalars. */
export const stateArrays = (m: RegionalModel): Float64Array[] => [m.u, m.v, m.w, m.th, m.pp, ...m.scalars];

/** Pack the state of a CPU model. */
export const packState = (m: RegionalModel): Q16[] => stateArrays(m).map((a) => quant(a));

/** Unpack a state into a model (a field the state lacks, or of another size, is zero). */
export function loadState(m: RegionalModel, state: ReadonlyArray<Q16>): void {
  stateArrays(m).forEach((a, f) => { const q = state[f]; if (q && q.q.length === a.length) dequant(q, a); else a.fill(0); });
}
