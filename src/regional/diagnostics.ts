// Diagnostics for the regional charts (display only): thermodynamic conversions, radar reflectivity,
// column composites, horizontal slices, cross-sections, soundings and azimuthal means. The GPU display
// kernels (src/gpu/regionalGpu.ts) compute the same quantities with the same definitions; the CPU
// versions here serve the CPU backend and the GPU-vs-CPU tests.

import { H, type RegionalModel } from './core.js';

const RD = 287.05, CP = 1004.5, P0 = 1e5, LV = 2.5e6, G = 9.80665;

/** Pressure (Pa) from the Exner function. */
export const pressure = (pi: number): number => P0 * Math.pow(Math.max(pi, 1e-6), CP / RD);
/** Saturation vapour pressure over water (Pa), Bolton (1980). */
export const esatW = (T: number): number => 611.2 * Math.exp(17.67 * (T - 273.15) / (T - 29.65));
/** Saturation mixing ratio over water (Tetens / Bolton), T in K, p in Pa. */
export function qsatW(T: number, p: number): number {
  const es = esatW(T);
  return 0.622 * es / Math.max(p - es, 1);
}
/** Equivalent potential temperature (K), simplified Bolton form. */
export const thetaE = (T: number, p: number, qv: number): number => T * Math.pow(P0 / p, RD / CP) * Math.exp(LV * qv / (CP * T));
/** Dew point (deg C) from the mixing ratio and pressure. */
export function dewPoint(qv: number, p: number): number {
  const e = Math.max(qv, 1e-9) * p / (0.622 + qv) / 100;     // hPa
  const l = Math.log(e / 6.112);
  return 243.5 * l / (17.67 - l);
}
/** Radar reflectivity (dBZ) from rain, snow and graupel mixing ratios (kg/kg) and air density. */
export function dbz(rho: number, qr: number, qs: number, qg: number): number {
  return 10 * Math.log10(Math.max(zFactor(rho, qr, qs, qg), 1e-3));
}
/** Reflectivity factor (mm^6 m^-3) from rain, snow and graupel contents (Smith 1984 exponential size distributions). */
const zFactor = (rho: number, qr: number, qs: number, qg: number): number =>
  3.63e9 * Math.pow(rho * Math.max(qr, 0), 1.75) + 9.80e8 * Math.pow(rho * Math.max(qs, 0), 1.75) + 4.33e10 * Math.pow(rho * Math.max(qg, 0), 1.75);

// ---------------------------------------------------------------- variables

/** Values per column in the display composites (GPU readDisplay and columnDiagnostics). */
export const COL = 10;
/** Offsets in a column record. */
export const C = { wmax: 0, wmin: 1, cmax: 2, pmax: 3, dbz: 4, ctopZ: 5, ctopT: 6, uh: 7, cape: 8, cin: 9 } as const;

/** Fields of a horizontal slice. */
export const SLICE_VARS = ['dbz', 'w', 'speed', 'u', 'v', 'thp', 'thetaE', 'rh', 'zeta', 'pp', 'qv', 'cloud', 'precip'] as const;
/** Fields of a cross-section or sounding (per level). */
export const SECTION_VARS = ['dbz', 'w', 'u', 'v', 'thp', 'thetaE', 'rh', 'cloud', 'precip', 'T', 'Td', 'p', 'qv', 'pp'] as const;
/** Composite (column / surface) maps. */
export const MAP_VARS = ['dbzMax', 'ctopT', 'ctopZ', 'uh', 'wMax', 'rainRate', 'rain', 'snow', 'slp', 'sfcWind', 'sfcU', 'sfcV', 'sfcThp', 'sfcThetaE', 'cape', 'cin'] as const;
/** Azimuthal-mean (radius-height) fields. */
export const RZ_VARS = ['vt', 'vr', 'w', 'thp', 'cond'] as const;
export type SliceVar = (typeof SLICE_VARS)[number];
export type SectionVar = (typeof SECTION_VARS)[number];
export type MapVar = (typeof MAP_VARS)[number];
export type RzVar = (typeof RZ_VARS)[number];

// ---------------------------------------------------------------- parcel (CAPE / CIN)

/**
 * Surface-based parcel: lifted from the lowest model level, dry adiabatically (theta, qv conserved)
 * until saturated, then pseudo-adiabatically (saturation adjustment at every level, condensate removed).
 * Buoyancy uses virtual temperatures. CAPE: positive area above the level of free convection (the first
 * level at or above the lifting condensation level with positive buoyancy); CIN: negative area below it.
 * Without a level of free convection both are 0. The GPU column kernel uses the same steps.
 */
export function parcelAscent(T: ArrayLike<number>, p: ArrayLike<number>, qv: ArrayLike<number>, dz: number, Tp?: Float64Array): { cape: number; cin: number; lcl: number; lfc: number; el: number } {
  const n = T.length;
  let th = T[0]! * Math.pow(P0 / p[0]!, RD / CP), q = Math.max(0, qv[0]!);
  let cape = 0, cin = 0, lcl = -1, lfc = -1, el = -1;
  if (Tp) Tp[0] = T[0]!;
  for (let k = 1; k < n; k++) {
    const pi = Math.pow(p[k]! / P0, RD / CP);
    let t = th * pi, d = 0;
    const qs0 = qsatW(t, p[k]!);
    if (q > qs0) {
      // condensation d: q - d = qs(t + Lv d / cp), Newton iterations
      for (let it = 0; it < 4; it++) {
        const tt = t + LV * d / CP, es = esatW(tt), qs = 0.622 * es / Math.max(p[k]! - es, 1);
        const dq = qs * p[k]! / Math.max(p[k]! - es, 1) * 17.67 * 243.5 / ((tt - 29.65) * (tt - 29.65));
        d += (q - d - qs) / (1 + LV / CP * dq);
      }
      d = Math.max(0, Math.min(q, d));
      if (lcl < 0) lcl = k;
    }
    t += LV * d / CP; q -= d; th = t / pi;
    if (Tp) Tp[k] = t;
    const tvp = t * (1 + 0.61 * q), tve = T[k]! * (1 + 0.61 * Math.max(0, qv[k]!));
    const b = G * (tvp - tve) / tve;
    if (lfc < 0) { if (lcl >= 0 && b > 0) { lfc = k; cape += b * dz; el = k; } else cin += Math.min(b, 0) * dz; }
    else if (b > 0) { cape += b * dz; el = k; }
  }
  if (lfc < 0) { cape = 0; cin = 0; }
  return { cape, cin, lcl, lfc, el };
}

// ---------------------------------------------------------------- per-cell values

export interface CellState { u: number; v: number; w: number; th: number; pp: number; q: ArrayLike<number> }
export interface LevelBase { th0: number; pi0: number; rho0: number }

/** Section / sounding values of one cell (SECTION_VARS order). q: qv, qc, qr[, qi, qs, qg]. */
export function sectionValues(c: CellState, b: LevelBase, out: Float32Array, o: number): void {
  const pi = b.pi0 + c.pp, p = pressure(pi), T = c.th * pi, q = c.q;
  const qv = q[0] ?? 0, qc = q[1] ?? 0, qr = q[2] ?? 0, qi = q[3] ?? 0, qs = q[4] ?? 0, qg = q[5] ?? 0;
  out[o] = dbz(b.rho0, qr, qs, qg); out[o + 1] = c.w; out[o + 2] = c.u; out[o + 3] = c.v; out[o + 4] = c.th - b.th0;
  out[o + 5] = thetaE(T, p, qv); out[o + 6] = 100 * qv / qsatW(T, p); out[o + 7] = 1e3 * Math.max(0, qc + qi); out[o + 8] = 1e3 * Math.max(0, qr + qs + qg);
  out[o + 9] = T - 273.15; out[o + 10] = dewPoint(qv, p); out[o + 11] = p / 100; out[o + 12] = 1e3 * Math.max(0, qv);
  out[o + 13] = (p - pressure(b.pi0)) / 100;
}

// ---------------------------------------------------------------- columns

/** Per-column composites of the CPU model (COL values per column, same definitions as the GPU kernel):
 *  w max / min, condensate and precipitation maxima, column-max reflectivity (dBZ), cloud-top height (m)
 *  and temperature (K; the lowest-level temperature where there is no cloud), 2-5 km updraft helicity
 *  (m^2/s^2), surface-based CAPE and CIN (J/kg). */
export function columnDiagnostics(m: RegionalModel): Float32Array {
  const { nx, ny, nz, dx, dy, dz } = m.c, sc = m.scalars, ns = sc.length, sx = m.sx, pl = m.plane;
  const out = new Float32Array(nx * ny * COL);
  const T = new Float64Array(nz), p = new Float64Array(nz), qv = new Float64Array(nz);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    let wmax = 0, wmin = 0, cmax = 0, pmax = 0, zmax = 0, ctz = 0, ctt = 0, uh = 0;
    for (let k = 0; k < nz; k++) {
      const q = m.idx(i, j, k);
      const qr = ns > 2 ? Math.max(sc[2]![q]!, 0) : 0, qs = ns > 5 ? Math.max(sc[4]![q]!, 0) : 0, qg = ns > 5 ? Math.max(sc[5]![q]!, 0) : 0;
      const cl = Math.max(0, (ns > 1 ? sc[1]![q]! : 0) + (ns > 5 ? sc[3]![q]! : 0)), pr = Math.max(0, (ns > 2 ? sc[2]![q]! : 0) + (ns > 5 ? qs + qg : 0));
      zmax = Math.max(zmax, zFactor(m.rho0[k]!, qr, qs, qg));
      const wc = 0.5 * (m.w[q]! + m.w[q + pl]!);
      const zeta = 0.25 * ((m.v[q + 1]! + m.v[q + 1 + sx]!) - (m.v[q - 1]! + m.v[q - 1 + sx]!)) / dx
        - 0.25 * ((m.u[q + sx]! + m.u[q + sx + 1]!) - (m.u[q - sx]! + m.u[q - sx + 1]!)) / dy;
      const zc = m.zc[k]!;
      if (zc >= 2000 && zc <= 5000) uh += wc * zeta * dz;
      const pi = m.pi0[k]! + m.pp[q]!;
      T[k] = m.th[q]! * pi; p[k] = pressure(pi); qv[k] = ns > 0 ? sc[0]![q]! : 0;
      if (cl > 1e-5) { ctz = zc; ctt = T[k]!; }
      const w = m.w[q]!;
      wmax = Math.max(wmax, w); wmin = Math.min(wmin, w); cmax = Math.max(cmax, cl); pmax = Math.max(pmax, pr);
    }
    if (ctz === 0) ctt = T[0]!;
    const pc = ns > 0 ? parcelAscent(T, p, qv, dz) : { cape: 0, cin: 0 };
    const o = COL * (j * nx + i);
    out[o] = wmax; out[o + 1] = wmin; out[o + 2] = cmax; out[o + 3] = pmax; out[o + 4] = 10 * Math.log10(Math.max(zmax, 1e-3));
    out[o + 5] = ctz; out[o + 6] = ctt; out[o + 7] = uh; out[o + 8] = pc.cape; out[o + 9] = pc.cin;
  }
  return out;
}

/** Column profiles at grid columns (i, j), same layout as GpuRegional.readColumns: per point, per level,
 *  cell-centred u, v, w, then theta, pi' and the moisture scalars (5 + scalars values). */
export function columnProfiles(m: RegionalModel, points: { i: number; j: number }[]): Float32Array {
  const { nx, ny, nz } = m.c, nf = 5 + m.scalars.length, out = new Float32Array(points.length * nz * nf);
  points.forEach((pt, n) => {
    const i = Math.max(0, Math.min(nx - 1, pt.i)), j = Math.max(0, Math.min(ny - 1, pt.j));
    for (let k = 0; k < nz; k++) {
      const q = m.idx(i, j, k), o = (n * nz + k) * nf;
      out[o] = 0.5 * (m.u[q]! + m.u[q + 1]!); out[o + 1] = 0.5 * (m.v[q]! + m.v[q + m.sx]!); out[o + 2] = 0.5 * (m.w[q]! + m.w[q + m.plane]!);
      out[o + 3] = m.th[q]!; out[o + 4] = m.pp[q]!;
      m.scalars.forEach((a, s) => { out[o + 5 + s] = a[q]!; });
    }
  });
  return out;
}

/** SECTION_VARS values (layout [var][k][point]) from column profiles (readColumns / columnProfiles layout).
 *  Winds are made ground-relative with the frame velocity. */
export function sectionFromColumns(m: RegionalModel, cols: Float32Array, np: number, nf: number, frame: { u: number; v: number }): Float32Array {
  const nz = m.c.nz, nv = SECTION_VARS.length, out = new Float32Array(nv * nz * np), tmp = new Float32Array(nv);
  const q = new Float64Array(nf - 5);
  for (let p = 0; p < np; p++) for (let k = 0; k < nz; k++) {
    const o = (p * nz + k) * nf;
    for (let s = 0; s < nf - 5; s++) q[s] = cols[o + 5 + s]!;
    sectionValues({ u: cols[o]! + frame.u, v: cols[o + 1]! + frame.v, w: cols[o + 2]!, th: cols[o + 3]!, pp: cols[o + 4]!, q }, { th0: m.th0[k]!, pi0: m.pi0[k]!, rho0: m.rho0[k]! }, tmp, 0);
    for (let v = 0; v < nv; v++) out[(v * nz + k) * np + p] = tmp[v]!;
  }
  return out;
}

// ---------------------------------------------------------------- horizontal slice

/** One level of every prognostic field (plane layout, halos included): u, v on faces, w at the bottom
 *  (w) and top (wTop) of the level, theta, pi' and the moisture scalars. */
export interface LevelPlanes { u: ArrayLike<number>; v: ArrayLike<number>; w: ArrayLike<number>; wTop: ArrayLike<number>; th: ArrayLike<number>; pp: ArrayLike<number>; sc: ArrayLike<number>[] }

/** The planes of level k of the CPU model (views, no copies). */
export function modelPlanes(m: RegionalModel, k: number): LevelPlanes {
  const o = k * m.plane, e = o + m.plane, s = (a: Float64Array, off = o): Float64Array => a.subarray(off, off + m.plane);
  return { u: s(m.u), v: s(m.v), w: s(m.w), wTop: s(m.w, e), th: s(m.th), pp: s(m.pp), sc: m.scalars.map((a) => s(a)) };
}

/** Horizontal slice fields ([j][i] per variable) at level k. Winds are ground-relative. */
export function sliceFields(m: RegionalModel, pl: LevelPlanes, k: number, vars: readonly SliceVar[], frame: { u: number; v: number }): Partial<Record<SliceVar, Float32Array>> {
  const { nx, ny, dx, dy } = m.c, sx = m.sx, n = nx * ny;
  const out: Partial<Record<SliceVar, Float32Array>> = {};
  for (const v of vars) out[v] = new Float32Array(n);
  const th0 = m.th0[k]!, pi0 = m.pi0[k]!, rho0 = m.rho0[k]!, sc = pl.sc, ns = sc.length, p0 = pressure(pi0);
  const g = (a: ArrayLike<number> | undefined, q: number): number => (a ? a[q]! : 0);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const q = (j + H) * sx + (i + H), c = j * nx + i;
    const uc = 0.5 * (pl.u[q]! + pl.u[q + 1]!) + frame.u, vc = 0.5 * (pl.v[q]! + pl.v[q + sx]!) + frame.v;
    const pi = pi0 + pl.pp[q]!, p = pressure(pi), T = pl.th[q]! * pi, qv = g(sc[0], q);
    for (const v of vars) {
      let x = 0;
      switch (v) {
        case 'dbz': x = dbz(rho0, g(sc[2], q), ns > 5 ? g(sc[4], q) : 0, ns > 5 ? g(sc[5], q) : 0); break;
        case 'w': x = 0.5 * (pl.w[q]! + pl.wTop[q]!); break;
        case 'speed': x = Math.hypot(uc, vc); break;
        case 'u': x = uc; break;
        case 'v': x = vc; break;
        case 'thp': x = pl.th[q]! - th0; break;
        case 'thetaE': x = thetaE(T, p, qv); break;
        case 'rh': x = 100 * qv / qsatW(T, p); break;
        case 'zeta': x = 0.25 * ((pl.v[q + 1]! + pl.v[q + 1 + sx]!) - (pl.v[q - 1]! + pl.v[q - 1 + sx]!)) / dx
          - 0.25 * ((pl.u[q + sx]! + pl.u[q + sx + 1]!) - (pl.u[q - sx]! + pl.u[q - sx + 1]!)) / dy; break;
        case 'pp': x = (p - p0) / 100; break;
        case 'qv': x = 1e3 * Math.max(0, qv); break;
        case 'cloud': x = 1e3 * Math.max(0, g(sc[1], q) + (ns > 5 ? g(sc[3], q) : 0)); break;
        case 'precip': x = 1e3 * Math.max(0, g(sc[2], q) + (ns > 5 ? g(sc[4], q) + g(sc[5], q) : 0)); break;
      }
      out[v]![c] = x;
    }
  }
  return out;
}

// ---------------------------------------------------------------- surface / composite maps

/** Composite and surface maps ([j][i]) from the column records, the lowest-level planes and the
 *  precipitation accumulations (mm) and rates (mm/h). */
export function compositeMaps(m: RegionalModel, col: Float32Array, pl0: LevelPlanes, acc: { rain: ArrayLike<number>; snow: ArrayLike<number>; rate: ArrayLike<number> | null },
  vars: readonly MapVar[], frame: { u: number; v: number }): Partial<Record<MapVar, Float32Array>> {
  const { nx, ny } = m.c, sx = m.sx, n = nx * ny, out: Partial<Record<MapVar, Float32Array>> = {};
  const th0 = m.th0[0]!, pi0 = m.pi0[0]!, z0 = m.zc[0]!;
  for (const v of vars) {
    const a = new Float32Array(n);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const c = j * nx + i, q = (j + H) * sx + (i + H), o = COL * c;
      let x = 0;
      switch (v) {
        case 'dbzMax': x = col[o + C.dbz]!; break;
        case 'ctopT': x = col[o + C.ctopT]! - 273.15; break;
        case 'ctopZ': x = col[o + C.ctopZ]! / 1000; break;
        case 'uh': x = col[o + C.uh]!; break;
        case 'wMax': x = col[o + C.wmax]!; break;
        case 'cape': x = col[o + C.cape]!; break;
        case 'cin': x = col[o + C.cin]!; break;
        case 'rain': x = acc.rain[c]!; break;
        case 'snow': x = acc.snow[c]!; break;
        case 'rainRate': x = acc.rate ? acc.rate[c]! : 0; break;
        case 'sfcU': x = 0.5 * (pl0.u[q]! + pl0.u[q + 1]!) + frame.u; break;
        case 'sfcV': x = 0.5 * (pl0.v[q]! + pl0.v[q + sx]!) + frame.v; break;
        case 'sfcWind': x = Math.hypot(0.5 * (pl0.u[q]! + pl0.u[q + 1]!) + frame.u, 0.5 * (pl0.v[q]! + pl0.v[q + sx]!) + frame.v); break;
        case 'sfcThp': x = pl0.th[q]! - th0; break;
        case 'slp': {
          const pi = pi0 + pl0.pp[q]!, T = pl0.th[q]! * pi, tv = T * (1 + 0.61 * Math.max(0, pl0.sc[0] ? pl0.sc[0][q]! : 0));
          x = pressure(pi) * Math.exp(G * z0 / (RD * tv)) / 100; break;
        }
        case 'sfcThetaE': { const pi = pi0 + pl0.pp[q]!; x = thetaE(pl0.th[q]! * pi, pressure(pi), pl0.sc[0] ? pl0.sc[0][q]! : 0); break; }
      }
      a[c] = x;
    }
    out[v] = a;
  }
  return out;
}

// ---------------------------------------------------------------- azimuthal means

/** Azimuthal means about (xc, yc) (m from the domain origin) in nr rings of width dr, per level (layout
 *  [ring + nr * level] per variable ... packed as 5 values per (level, ring): tangential wind, radial wind,
 *  w, theta - theta0, total condensate (kg/kg)). Same sampling as GpuRegional.readRZ: t = ring * nz + level. */
export function azimuthalMeans(m: RegionalModel, xc: number, yc: number, dr: number, nr: number): Float32Array {
  const { nx, ny, nz, dx, dy } = m.c, open = m.c.lateral === 'open', out = new Float32Array(5 * nr * nz);
  const cell = (x: number, y: number): [number, number] => {
    let i = Math.floor(x / dx), j = Math.floor(y / dy);
    if (open) { i = Math.max(0, Math.min(nx - 1, i)); j = Math.max(0, Math.min(ny - 1, j)); }
    else { i = ((i % nx) + nx) % nx; j = ((j % ny) + ny) % ny; }
    return [i, j];
  };
  for (let ir = 0; ir < nr; ir++) {
    const r = (ir + 0.5) * dr, na = Math.max(16, Math.min(256, Math.trunc(6.2832 * r / (0.5 * dx))));
    for (let k = 0; k < nz; k++) {
      let vt = 0, vr = 0, ww = 0, tp = 0, cd = 0;
      for (let a = 0; a < na; a++) {
        const ang = 6.2831853 * (a + 0.5) / na, ca = Math.cos(ang), sa = Math.sin(ang);
        const [i, j] = cell(xc + r * ca, yc + r * sa), q = m.idx(i, j, k);
        const uc = 0.5 * (m.u[q]! + m.u[q + 1]!), vc = 0.5 * (m.v[q]! + m.v[q + m.sx]!);
        vt += -uc * sa + vc * ca; vr += uc * ca + vc * sa;
        ww += 0.5 * (m.w[q]! + m.w[q + m.plane]!);
        tp += m.th[q]! - m.th0[k]!;
        for (let s = 1; s < m.scalars.length; s++) cd += Math.max(m.scalars[s]![q]!, 0);
      }
      const o = 5 * (ir * nz + k);
      out[o] = vt / na; out[o + 1] = vr / na; out[o + 2] = ww / na; out[o + 3] = tp / na; out[o + 4] = cd / na;
    }
  }
  return out;
}

/** Unpack readRZ / azimuthalMeans output into per-variable arrays of layout [k][ring]. */
export function unpackRZ(raw: Float32Array, nr: number, nz: number): Record<RzVar, Float32Array> {
  const out = {} as Record<RzVar, Float32Array>;
  RZ_VARS.forEach((v, f) => {
    const a = new Float32Array(nr * nz);
    for (let ir = 0; ir < nr; ir++) for (let k = 0; k < nz; k++) a[k * nr + ir] = raw[5 * (ir * nz + k) + f]! * (v === 'cond' ? 1e3 : 1);
    out[v] = a;
  });
  return out;
}
