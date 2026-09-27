// One-way nesting: initialise a regional non-hydrostatic model from a global-model state and use
// the global fields as lateral-boundary relaxation targets.
//
// Geometry: a local tangent plane centred at (lat0, lon0): x east, y north,
//   lat = lat0 + y / a,  lon = lon0 + x / (a cos lat0);   f = 2 Omega sin lat0 (f-plane).
// Vertical: global sigma-level heights from the hydrostatic relation (Simmons–Burridge, as in the
// dycore); theta, u, v, qv and ln p interpolated linearly in height to the regional levels.
// Base state: the domain-mean theta(z), qv(z); pi' from the interpolated pressure so that the global
// geostrophic balance is carried over.

import { DRY_AIR, EARTH } from '../core/constants.js';
import { RegionalModel, RegionalConfig, BoundaryTargets } from './core.js';

/** Global fields on a Gaussian grid: [k][lat][lon] (k top -> bottom), ps [lat][lon]. */
export interface GlobalSnapshot {
  nlat: number; nlon: number; K: number;
  lat: Float64Array; lon: Float64Array;          // radians, lat north -> south
  sigma: Float64Array; sigmaHalf: Float64Array;  // full / half levels
  u: ArrayLike<number>; v: ArrayLike<number>; T: ArrayLike<number>; ps: ArrayLike<number>;
  q: ArrayLike<number> | null;
  phis?: ArrayLike<number> | null;               // surface geopotential (m^2/s^2)
}

export interface NestSpec { lat0: number; lon0: number; L: number; dx: number; nz: number; dz: number; dt: number; nsound: number }

interface Column { z: Float64Array; th: Float64Array; u: Float64Array; v: Float64Array; q: Float64Array; lnp: Float64Array }

/** Bilinear sample of the global snapshot at (lat, lon) -> vertical profiles on global levels. */
function sampleColumn(g: GlobalSnapshot, lat: number, lon: number): Column {
  const { nlat, nlon, K } = g, ng = nlat * nlon;
  let j = 0;
  while (j < nlat - 2 && g.lat[j + 1]! > lat) j++;
  const wy = Math.max(0, Math.min(1, (g.lat[j]! - lat) / (g.lat[j]! - g.lat[j + 1]!)));
  let x = lon / (2 * Math.PI) * nlon;
  x -= Math.floor(x / nlon) * nlon;
  const i0 = Math.floor(x) % nlon, i1 = (i0 + 1) % nlon, wx = x - Math.floor(x);
  const pts = [[j, i0, (1 - wy) * (1 - wx)], [j, i1, (1 - wy) * wx], [j + 1, i0, wy * (1 - wx)], [j + 1, i1, wy * wx]] as const;
  const at = (f: ArrayLike<number>, k: number): number => { let s = 0; for (const [jj, ii, w] of pts) s += w * f[k * ng + jj * nlon + ii]!; return s; };
  const ps = at(g.ps, 0);
  const phis = g.phis ? at(g.phis, 0) : 0;
  const kap = DRY_AIR.kappa, R = DRY_AIR.rd, grav = EARTH.gravity;
  const c: Column = { z: new Float64Array(K), th: new Float64Array(K), u: new Float64Array(K), v: new Float64Array(K), q: new Float64Array(K), lnp: new Float64Array(K) };
  // hydrostatic heights (above sea level): half levels from the surface upward
  let zh = phis / grav;
  for (let k = K - 1; k >= 0; k--) {
    const T = at(g.T, k), sh = g.sigmaHalf[k]!, sb = g.sigmaHalf[k + 1]!;
    const alpha = k === 0 ? Math.LN2 : 1 - sh / (sb - sh) * Math.log(sb / sh);
    c.z[k] = zh + alpha * R * T / grav;
    if (k > 0) zh += R * T * Math.log(sb / sh) / grav;
    const p = g.sigma[k]! * ps;
    c.th[k] = T * Math.pow(DRY_AIR.pRef / p, kap);
    c.u[k] = at(g.u, k); c.v[k] = at(g.v, k); c.q[k] = g.q ? Math.max(0, at(g.q, k)) : 0;
    c.lnp[k] = Math.log(p);
  }
  return c;
}

/** Linear interpolation of a column profile to height z (levels ordered top -> bottom, z decreasing). */
function interp(c: Column, f: Float64Array, z: number, extrapLnp = false): number {
  const K = c.z.length;
  if (z <= c.z[K - 1]!) {
    if (!extrapLnp) return f[K - 1]!;
    const dz = c.z[K - 2]! - c.z[K - 1]!;
    return f[K - 1]! + (f[K - 1]! - f[K - 2]!) * (c.z[K - 1]! - z) / dz;
  }
  if (z >= c.z[0]!) return f[0]!;
  let k = K - 1;
  while (k > 0 && c.z[k - 1]! < z) k--;
  const w = (z - c.z[k]!) / (c.z[k - 1]! - c.z[k]!);
  return f[k]! + w * (f[k - 1]! - f[k]!);
}

/** Bilinear sample of a 2-D global field [lat][lon] at the regional cell centres (row-major j*nx+i). */
export function sampleSurface(g: GlobalSnapshot, spec: NestSpec, f: ArrayLike<number>): Float64Array {
  const a = EARTH.radius, nx = Math.round(spec.L / spec.dx), out = new Float64Array(nx * nx);
  const { nlat, nlon } = g;
  for (let jj = 0; jj < nx; jj++) for (let ii = 0; ii < nx; ii++) {
    const lat = spec.lat0 + ((jj + 0.5) * spec.dx - spec.L / 2) / a;
    const lon = spec.lon0 + ((ii + 0.5) * spec.dx - spec.L / 2) / (a * Math.cos(spec.lat0));
    let j = 0;
    while (j < nlat - 2 && g.lat[j + 1]! > lat) j++;
    const wy = Math.max(0, Math.min(1, (g.lat[j]! - lat) / (g.lat[j]! - g.lat[j + 1]!)));
    let x = lon / (2 * Math.PI) * nlon;
    x -= Math.floor(x / nlon) * nlon;
    const i0 = Math.floor(x) % nlon, i1 = (i0 + 1) % nlon, wx = x - Math.floor(x);
    out[jj * nx + ii] = (1 - wy) * ((1 - wx) * f[j * nlon + i0]! + wx * f[j * nlon + i1]!) + wy * ((1 - wx) * f[(j + 1) * nlon + i0]! + wx * f[(j + 1) * nlon + i1]!);
  }
  return out;
}

export interface NestedRegional { model: RegionalModel; boundary: BoundaryTargets; f: number; description: string }

/** Build a regional model (open boundaries) initialised from a global snapshot. nScalars = 3 for Kessler moisture. */
export function nestFromGlobal(g: GlobalSnapshot, spec: NestSpec, nScalars = 3, extra: Partial<RegionalConfig> = {}): NestedRegional {
  const a = EARTH.radius, nx = Math.round(spec.L / spec.dx), ny = nx, nz = spec.nz;
  const lat0 = spec.lat0, lon0 = spec.lon0;
  // sample columns on the regional grid (cell centres)
  const cols: Column[] = [];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const x = (i + 0.5) * spec.dx - spec.L / 2, y = (j + 0.5) * spec.dx - spec.L / 2;
    cols.push(sampleColumn(g, lat0 + y / a, lon0 + x / (a * Math.cos(lat0))));
  }
  // domain-mean base-state sounding
  const zc = Array.from({ length: nz }, (_, k) => (k + 0.5) * spec.dz);
  const thMean = new Float64Array(nz), qMean = new Float64Array(nz);
  for (const c of cols) for (let k = 0; k < nz; k++) { thMean[k] = thMean[k]! + interp(c, c.th, zc[k]!); qMean[k] = qMean[k]! + interp(c, c.q, zc[k]!); }
  for (let k = 0; k < nz; k++) { thMean[k] = thMean[k]! / cols.length; qMean[k] = qMean[k]! / cols.length; }
  const sounding = (z: number): { theta: number; qv: number } => {
    const x = Math.max(0, Math.min(nz - 1, z / spec.dz - 0.5)), k = Math.min(nz - 2, Math.floor(x)), w = x - k;
    return { theta: thMean[k]! * (1 - w) + thMean[k + 1]! * w, qv: qMean[k]! * (1 - w) + qMean[k + 1]! * w };
  };
  const f = 2 * EARTH.omega * Math.sin(lat0);
  const cfg: RegionalConfig = {
    nx, ny, nz, dx: spec.dx, dy: spec.dx, dz: spec.dz, dt: spec.dt, nsound: spec.nsound, f, beta: 0.3, divDamp: 0.1,
    dampDepth: Math.min(6000, 0.25 * nz * spec.dz), dampRate: 1 / 300, kdiff2: 0, lateral: 'open', relaxCells: 8, relaxTau: 300, ...extra,
  };
  const m = new RegionalModel(cfg, sounding, nScalars);
  const kap = DRY_AIR.kappa;
  // fill fields: theta, qv, pi' at centres; u at x-faces and v at y-faces (averaged from neighbouring centres)
  const uc = new Float64Array(nx * ny * nz), vc = new Float64Array(nx * ny * nz);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const c = cols[j * nx + i]!;
    for (let k = 0; k < nz; k++) {
      const z = zc[k]!, q = m.idx(i, j, k);
      m.th[q] = interp(c, c.th, z);
      if (nScalars > 0) m.scalars[0]![q] = interp(c, c.q, z);
      const p = Math.exp(interp(c, c.lnp, z, true));
      m.pp[q] = Math.pow(p / DRY_AIR.pRef, kap) - m.pi0[k]!;
      uc[(k * ny + j) * nx + i] = interp(c, c.u, z);
      vc[(k * ny + j) * nx + i] = interp(c, c.v, z);
    }
  }
  // Discrete hydrostatic balance in the regional model's own form: keep pi' from the interpolated
  // pressure only at a reference level (~3 km, where global levels are dense) and integrate
  //   cp theta_rho_face (pi'_k - pi'_{k-1}) / dz = g (theta_rho/theta_rho0 - 1)_face
  // up and down from it, so the initial columns are exactly balanced.
  {
    const G = EARTH.gravity, cpd = DRY_AIR.cp, dz = spec.dz;
    let kr = 0;
    for (let k = 0; k < nz; k++) if (Math.abs(zc[k]! - 3000) < Math.abs(zc[kr]! - 3000)) kr = k;
    const qv = nScalars > 0 ? m.scalars[0]! : null;
    const thr = (q: number): number => m.th[q]! * (1 + 0.61 * (qv ? qv[q]! : 0));
    const buoy = (q: number, k: number): number => { const t0 = m.th0[k]! * (1 + 0.61 * m.qv0[k]!); return G * (thr(q) - t0) / t0; };
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      for (let k = kr + 1; k < nz; k++) {
        const q = m.idx(i, j, k), qm = q - m.plane;
        const B = 0.5 * (buoy(q, k) + buoy(qm, k - 1)), tf = 0.5 * (thr(q) + thr(qm));
        m.pp[q] = m.pp[qm]! + B * dz / (cpd * tf);
      }
      for (let k = kr - 1; k >= 0; k--) {
        const q = m.idx(i, j, k), qp = q + m.plane;
        const B = 0.5 * (buoy(qp, k + 1) + buoy(q, k)), tf = 0.5 * (thr(qp) + thr(q));
        m.pp[q] = m.pp[qp]! - B * dz / (cpd * tf);
      }
    }
  }
  // remove the domain-mean pi' per level (the base state carries the mean)
  for (let k = 0; k < nz; k++) {
    let s = 0;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) s += m.pp[m.idx(i, j, k)]!;
    s /= nx * ny;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) m.pp[m.idx(i, j, k)] = m.pp[m.idx(i, j, k)]! - s;
  }
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const q = m.idx(i, j, k), o = (k * ny + j) * nx;
    m.u[q] = i > 0 ? 0.5 * (uc[o + i - 1]! + uc[o + i]!) : uc[o + i]!;
    m.v[q] = j > 0 ? 0.5 * (vc[((k * ny + j - 1) * nx) + i]! + vc[o + i]!) : vc[o + i]!;
  }
  for (let k = 0; k < nz; k++) { m.ub[k] = 0; m.vb[k] = 0; }
  const boundary: BoundaryTargets = { u: Float64Array.from(m.u), v: Float64Array.from(m.v), th: Float64Array.from(m.th), qv: nScalars > 0 ? Float64Array.from(m.scalars[0]!) : null, pp: Float64Array.from(m.pp) };
  m.boundary = boundary;
  const description = `nest at ${(lat0 * 180 / Math.PI).toFixed(1)}°, ${(lon0 * 180 / Math.PI).toFixed(1)}°, ${(spec.L / 1000).toFixed(0)} km, dx ${(spec.dx / 1000).toFixed(1)} km`;
  return { model: m, boundary, f, description };
}
