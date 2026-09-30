// Coarse-to-fine refinement: continue a running regional simulation on a finer grid.
//
// A storm is spun up cheaply on a coarse grid and, once it is interesting, every prognostic field is
// interpolated to a finer grid (the same domain, or a sub-box around the storm) and the run continues
// there. Each variable is interpolated trilinearly at its own staggered position on the C-grid: u on x
// faces, v on y faces, w on z faces, theta, pi' and the moisture species at cell centres. theta and
// qv are total values, pi' is relative to the base state, and both grids use the same sounding, so the
// interpolated state is consistent with the fine grid's base state. Trilinear interpolation keeps the
// moisture species non-negative. The small divergence it introduces is removed by the acoustic steps
// within the first minutes.

import type { RegionalModel } from './core.js';

export interface Accumulations { rain: Float64Array; snow: Float64Array }

/** Trilinear sampler of a model's fields at any point of its domain (m, the model's own coordinates), each variable at
 *  its staggered position: sx, sy horizontal staggering (0 = face, 0.5 = centre), faceZ for w levels. Periodic
 *  domains wrap, open ones clamp at the edges (and below the lowest / above the highest level). With `base` (a profile
 *  per level) it samples the departure from it: fields with a strong mean vertical gradient (theta, qv, the base wind)
 *  are interpolated as departures and the target grid's own profile added back. Below the lowest cell centre and above
 *  the highest the cell-centred fields continue linearly (at most half a level). */
export function makeSampler(mc: RegionalModel): (a: Float64Array, x: number, y: number, z: number, sx: number, sy: number, faceZ: boolean, base?: Float64Array) => number {
  const c = mc.c, periodic = c.lateral !== 'open';
  // sample index along one axis: fractional index -> two integer indices and a weight
  const axis = (p: number, n: number): [number, number, number] => {
    if (periodic) {
      const i0 = Math.floor(p), w = p - i0;
      return [((i0 % n) + n) % n, (((i0 + 1) % n) + n) % n, w];
    }
    const q = Math.max(0, Math.min(n - 1, p)), i0 = Math.min(Math.floor(q), n - 2 < 0 ? 0 : n - 2);
    return n < 2 ? [0, 0, 0] : [i0, i0 + 1, q - i0];
  };
  // vertical: faces span the column; cell centres continue linearly over the half level below the lowest centre and
  // above the highest (clamping would carry the value of 500 m, say, down to 250 m)
  const vaxis = (p: number, n: number, faces: boolean): [number, number, number] => {
    if (n < 2) return [0, 0, 0];
    const q = faces ? Math.max(0, Math.min(n - 1, p)) : Math.max(-0.5, Math.min(n - 0.5, p)), i0 = Math.max(0, Math.min(Math.floor(q), n - 2));
    return [i0, i0 + 1, q - i0];
  };
  return (a, x, y, z, sx, sy, faceZ, base) => {
    // number of distinct points along each axis (faces: nx + 1 with open boundaries, nx if periodic)
    const nI = sx === 0 && !periodic ? c.nx + 1 : c.nx, nJ = sy === 0 && !periodic ? c.ny + 1 : c.ny;
    const [i0, i1, wx] = axis(x / c.dx - sx, nI), [j0, j1, wy] = axis(y / c.dy - sy, nJ);
    const [k0, k1, wz] = faceZ ? vaxis(z / c.dz, c.nz + 1, true) : vaxis(z / c.dz - 0.5, c.nz, false);
    const g = base ? (i: number, j: number, k: number): number => a[mc.idx(i, j, k)]! - base[k]! : (i: number, j: number, k: number): number => a[mc.idx(i, j, k)]!;
    const l0 = (1 - wy) * ((1 - wx) * g(i0, j0, k0) + wx * g(i1, j0, k0)) + wy * ((1 - wx) * g(i0, j1, k0) + wx * g(i1, j1, k0));
    const l1 = (1 - wy) * ((1 - wx) * g(i0, j0, k1) + wx * g(i1, j0, k1)) + wy * ((1 - wx) * g(i0, j1, k1) + wx * g(i1, j1, k1));
    return (1 - wz) * l0 + wz * l1;
  };
}

/** Bilinear sample of a [j][i] surface array of a model at (x, y) m (periodic wrap or clamp as the model). */
function surfaceSampler(mc: RegionalModel): (a: ArrayLike<number>, x: number, y: number) => number {
  const c = mc.c, periodic = c.lateral !== 'open';
  const ax = (p: number, n: number): [number, number, number] => {
    if (periodic) { const i0 = Math.floor(p); return [((i0 % n) + n) % n, (((i0 + 1) % n) + n) % n, p - i0]; }
    const q = Math.max(0, Math.min(n - 1, p)), i0 = Math.min(Math.floor(q), Math.max(0, n - 2));
    return n < 2 ? [0, 0, 0] : [i0, i0 + 1, q - i0];
  };
  return (a, x, y) => {
    const [i0, i1, wx] = ax(x / c.dx - 0.5, c.nx), [j0, j1, wy] = ax(y / c.dy - 0.5, c.ny);
    return (1 - wy) * ((1 - wx) * a[j0 * c.nx + i0]! + wx * a[j0 * c.nx + i1]!) + wy * ((1 - wx) * a[j1 * c.nx + i0]! + wx * a[j1 * c.nx + i1]!);
  };
}

/** Interpolate the state of the coarse model `mc` into the fine model `mf`, whose domain starts at
 *  (x0, y0) metres in the coarse model's coordinates. Surface accumulations ([j][i]) are interpolated
 *  too. The fine model's time is set to the coarse model's. `cols`: only the fine columns (i, j) it accepts (faces
 *  on the far edge belong to the last column). */
export function refineInto(mc: RegionalModel, mf: RegionalModel, x0 = 0, y0 = 0, accC?: Accumulations, accF?: Accumulations, cols?: (i: number, j: number) => boolean): void {
  const f = mf.c, sample = makeSampler(mc), ssf = surfaceSampler(mc);
  const take = cols ? (i: number, j: number): boolean => cols(Math.min(i, f.nx - 1), Math.min(j, f.ny - 1)) : (): boolean => true;
  // theta, qv and the horizontal wind as departures from each grid's own base profile (makeSampler)
  const fill = (dst: Float64Array, src: Float64Array, sx: number, sy: number, faceZ: boolean, iMax: number, jMax: number, kMax: number, baseC?: Float64Array, baseF?: Float64Array): void => {
    for (let k = 0; k < kMax; k++) {
      const z = faceZ ? k * f.dz : (k + 0.5) * f.dz, b = baseF ? baseF[k]! : 0;
      for (let j = 0; j < jMax; j++) {
        const y = y0 + (j + sy) * f.dy;
        for (let i = 0; i < iMax; i++) if (take(i, j)) dst[mf.idx(i, j, k)] = b + sample(src, x0 + (i + sx) * f.dx, y, z, sx, sy, faceZ, baseC);
      }
    }
  };
  // smooth start: the trilinear interpolant is continuous but creased at every coarse cell face, and the finer grid (and
  // its picture) would show those creases as steps; a centred box filter one coarse cell wide in each direction turns it
  // into a smooth (piecewise quadratic) field, like the smooth interpolation nests usually start from. On departures from
  // the base profiles, continued linearly beyond the edges (linear fields stay exact); positive fields are clamped
  // there. Not for a partial refill.
  const cc = mc.c, fPer = f.lateral !== 'open';
  const ratio = (a: number, b: number): number => Math.max(1, Math.round(a / b));
  const rx = ratio(cc.dx, f.dx), ry = ratio(cc.dy, f.dy), rz = ratio(cc.dz, f.dz);
  const filt = (v: Float64Array, n: number, r: number, per: boolean, pos: boolean, o: Float64Array): void => {
    const h = Math.floor(r / 2), even = r % 2 === 0;
    const g = (i: number): number => {
      if (i >= 0 && i < n) return v[i]!;
      if (per) return v[((i % n) + n) % n]!;
      if (pos || n < 2) return v[i < 0 ? 0 : n - 1]!;
      return i < 0 ? v[0]! + i * (v[1]! - v[0]!) : v[n - 1]! + (i - n + 1) * (v[n - 1]! - v[n - 2]!);
    };
    for (let i = 0; i < n; i++) { let t = 0; for (let d = -h; d <= h; d++) t += (even && (d === h || d === -h) ? 0.5 : 1) * g(i + d); o[i] = t / r; }
  };
  const smooth = (a: Float64Array, nI: number, nJ: number, nK: number, base: Float64Array | null, pos: boolean): void => {
    if (cols) return;
    const at = (i: number, j: number, k: number): number => mf.idx(i, j, k), line = new Float64Array(Math.max(nI, nJ, nK)), out = new Float64Array(line.length);
    if (base) for (let k = 0; k < nK; k++) for (let j = 0; j < nJ; j++) for (let i = 0; i < nI; i++) a[at(i, j, k)] = a[at(i, j, k)]! - base[k]!;
    if (rx > 1) for (let k = 0; k < nK; k++) for (let j = 0; j < nJ; j++) {
      for (let i = 0; i < nI; i++) line[i] = a[at(i, j, k)]!;
      filt(line, nI, rx, fPer, pos, out);
      for (let i = 0; i < nI; i++) a[at(i, j, k)] = out[i]!;
    }
    if (ry > 1) for (let k = 0; k < nK; k++) for (let i = 0; i < nI; i++) {
      for (let j = 0; j < nJ; j++) line[j] = a[at(i, j, k)]!;
      filt(line, nJ, ry, fPer, pos, out);
      for (let j = 0; j < nJ; j++) a[at(i, j, k)] = out[j]!;
    }
    if (rz > 1) for (let j = 0; j < nJ; j++) for (let i = 0; i < nI; i++) {
      for (let k = 0; k < nK; k++) line[k] = a[at(i, j, k)]!;
      filt(line, nK, rz, false, pos, out);
      for (let k = 0; k < nK; k++) a[at(i, j, k)] = out[k]!;
    }
    if (base) for (let k = 0; k < nK; k++) for (let j = 0; j < nJ; j++) for (let i = 0; i < nI; i++) a[at(i, j, k)] = a[at(i, j, k)]! + base[k]!;
  };
  // condensate and precipitation (non-negative, with sharp edges): linear interpolation would spread them over whole
  // coarse cells, so their edges would follow the coarse cells' faces (steps); instead the cube root is interpolated and
  // cubed, so they fall off toward their edges as the coarse grid's own picture does (display.ts bytes are cube roots).
  // That takes some mass off the edges; one factor for the whole region gives it back (a factor per level or per cell
  // would show as steps again), so the mass in the region is the coarse cells' (sampled cell by cell)
  const cPer = cc.lateral !== 'open';
  const cell = (p: number, n: number): number => { const i = Math.floor(p); return cPer ? ((i % n) + n) % n : Math.max(0, Math.min(n - 1, i)); };
  const hydrometeor = (dst: Float64Array, src: Float64Array): void => {
    const root = src.map((v) => Math.cbrt(Math.max(0, v)));
    let T = 0, S = 0;
    for (let k = 0; k < f.nz; k++) {
      const z = (k + 0.5) * f.dz, K = Math.max(0, Math.min(cc.nz - 1, Math.floor(z / cc.dz)));
      for (let j = 0; j < f.ny; j++) {
        const y = y0 + (j + 0.5) * f.dy, jc = cell(y / cc.dy, cc.ny);
        for (let i = 0; i < f.nx; i++) {
          if (!take(i, j)) continue;
          const x = x0 + (i + 0.5) * f.dx;
          dst[mf.idx(i, j, k)] = sample(root, x, y, z, 0.5, 0.5, false) ** 3;
          T += Math.max(0, src[mc.idx(cell(x / cc.dx, cc.nx), jc, K)]!) * mc.rho0[K]!;
        }
      }
    }
    smooth(dst, f.nx, f.ny, f.nz, null, true);
    for (let k = 0; k < f.nz; k++) for (let j = 0; j < f.ny; j++) for (let i = 0; i < f.nx; i++) if (take(i, j)) S += dst[mf.idx(i, j, k)]! * mf.rho0[k]!;
    const g = S > 0 ? T / S : 0;
    for (let k = 0; k < f.nz; k++) for (let j = 0; j < f.ny; j++) for (let i = 0; i < f.nx; i++) if (take(i, j)) dst[mf.idx(i, j, k)] = dst[mf.idx(i, j, k)]! * g;
  };
  const fOpen = f.lateral === 'open', nu = f.nx + (fOpen ? 1 : 0), nv = f.ny + (fOpen ? 1 : 0);
  fill(mf.u, mc.u, 0, 0.5, false, nu, f.ny, f.nz, mc.ub, mf.ub); smooth(mf.u, nu, f.ny, f.nz, mf.ub, false);
  fill(mf.v, mc.v, 0.5, 0, false, f.nx, nv, f.nz, mc.vb, mf.vb); smooth(mf.v, f.nx, nv, f.nz, mf.vb, false);
  fill(mf.w, mc.w, 0.5, 0.5, true, f.nx, f.ny, f.nz + 1); smooth(mf.w, f.nx, f.ny, f.nz + 1, null, false);
  fill(mf.th, mc.th, 0.5, 0.5, false, f.nx, f.ny, f.nz, mc.th0, mf.th0); smooth(mf.th, f.nx, f.ny, f.nz, mf.th0, false);
  fill(mf.pp, mc.pp, 0.5, 0.5, false, f.nx, f.ny, f.nz); smooth(mf.pp, f.nx, f.ny, f.nz, null, false);
  const ns = Math.min(mc.scalars.length, mf.scalars.length);
  for (let s = 0; s < ns; s++) {
    if (s === 0) { fill(mf.scalars[0]!, mc.scalars[0]!, 0.5, 0.5, false, f.nx, f.ny, f.nz, mc.qv0, mf.qv0); smooth(mf.scalars[0]!, f.nx, f.ny, f.nz, mf.qv0, false); }
    else hydrometeor(mf.scalars[s]!, mc.scalars[s]!);
    const a = mf.scalars[s]!;
    for (let i = 0; i < a.length; i++) if (a[i]! < 0) a[i] = 0;
  }
  // no vertical motion through the ground and the lid
  for (let j = 0; j < f.ny; j++) for (let i = 0; i < f.nx; i++) if (take(i, j)) { mf.w[mf.idx(i, j, 0)] = 0; mf.w[mf.idx(i, j, f.nz)] = 0; }
  if (accC && accF) {
    for (const key of ['rain', 'snow'] as const) {
      const src = accC[key], dst = accF[key];
      for (let j = 0; j < f.ny; j++) for (let i = 0; i < f.nx; i++) if (take(i, j)) dst[j * f.nx + i] = ssf(src, x0 + (i + 0.5) * f.dx, y0 + (j + 0.5) * f.dy);
    }
  }
  mf.time = mc.time;
}

/**
 * Fine-to-coarse (the reverse of refineInto): every point of the coarse model `mc` that lies in the fine model's
 * domain (starting at (x0, y0) m in the coarse coordinates; a periodic fine domain as large as the coarse one covers
 * all of it) takes the mean of the fine fields over its cell (sampled at the fine spacing), blended with the coarse
 * model's own value over `margin` m inside the fine domain's edge (0: no blending). Surface accumulations and
 * `surface` arrays ([j][i], e.g. skin temperature and wetness) likewise. The coarse model's time becomes the fine one's.
 */
export function coarsenInto(mf: RegionalModel, mc: RegionalModel, x0: number, y0: number, margin: number,
  accF?: Accumulations, accC?: Accumulations, surfF?: ArrayLike<number>[], surfC?: Float64Array[]): void {
  const c = mc.c, f = mf.c, sample = makeSampler(mf), ssf = surfaceSampler(mf);
  const Lxf = f.nx * f.dx, Lyf = f.ny * f.dy, Lxc = c.nx * c.dx, Lyc = c.ny * c.dy, cPer = c.lateral !== 'open';
  const whole = f.lateral !== 'open' && Math.abs(Lxf - Lxc) < 1 && Math.abs(Lyf - Lyc) < 1;
  const rx = Math.max(1, Math.round(c.dx / f.dx)), ry = Math.max(1, Math.round(c.dy / f.dy)), rz = Math.max(1, Math.round(c.dz / f.dz));
  /** position in the fine domain of a coarse point, and its blending weight (0 outside) */
  const local = (x: number, y: number): { lx: number; ly: number; w: number } => {
    let lx = x - x0, ly = y - y0;
    if (whole) return { lx, ly, w: 1 };
    if (cPer) { lx -= Math.floor(lx / Lxc) * Lxc; ly -= Math.floor(ly / Lyc) * Lyc; }
    if (lx < 0 || ly < 0 || lx > Lxf || ly > Lyf) return { lx, ly, w: 0 };
    const d = Math.min(lx, Lxf - lx, ly, Lyf - ly);
    return { lx, ly, w: margin > 0 && d < margin ? Math.sin(0.5 * Math.PI * d / margin) ** 2 : 1 };
  };
  const mean = (a: Float64Array, lx: number, ly: number, z: number, sx: number, sy: number, faceZ: boolean): number => {
    let s = 0, n = 0;
    for (let b = 0; b < ry; b++) for (let aa = 0; aa < rx; aa++) {
      const px = lx + ((aa + 0.5) / rx - 0.5) * c.dx * (sx === 0 ? 0 : 1), py = ly + ((b + 0.5) / ry - 0.5) * c.dy * (sy === 0 ? 0 : 1);
      if (faceZ) { s += sample(a, px, py, z, sx, sy, true); n++; continue; }
      for (let q = 0; q < rz; q++) { s += sample(a, px, py, z + ((q + 0.5) / rz - 0.5) * c.dz, sx, sy, false); n++; }
    }
    return s / n;
  };
  const fill = (dst: Float64Array, src: Float64Array, sx: number, sy: number, faceZ: boolean, iMax: number, jMax: number, kMax: number): void => {
    for (let j = 0; j < jMax; j++) for (let i = 0; i < iMax; i++) {
      const { lx, ly, w } = local((i + sx) * c.dx, (j + sy) * c.dy);
      if (w <= 0) continue;
      for (let k = 0; k < kMax; k++) {
        const z = faceZ ? k * c.dz : (k + 0.5) * c.dz, q = mc.idx(i, j, k);
        dst[q] = w * mean(src, lx, ly, z, sx, sy, faceZ) + (1 - w) * dst[q]!;
      }
    }
  };
  const cOpen = !cPer;
  fill(mc.u, mf.u, 0, 0.5, false, c.nx + (cOpen ? 1 : 0), c.ny, c.nz);
  fill(mc.v, mf.v, 0.5, 0, false, c.nx, c.ny + (cOpen ? 1 : 0), c.nz);
  fill(mc.w, mf.w, 0.5, 0.5, true, c.nx, c.ny, c.nz + 1);
  fill(mc.th, mf.th, 0.5, 0.5, false, c.nx, c.ny, c.nz);
  fill(mc.pp, mf.pp, 0.5, 0.5, false, c.nx, c.ny, c.nz);
  const ns = Math.min(mc.scalars.length, mf.scalars.length);
  for (let s = 0; s < ns; s++) {
    fill(mc.scalars[s]!, mf.scalars[s]!, 0.5, 0.5, false, c.nx, c.ny, c.nz);
    const a = mc.scalars[s]!;
    for (let i = 0; i < a.length; i++) if (a[i]! < 0) a[i] = 0;
  }
  for (let j = 0; j < c.ny; j++) for (let i = 0; i < c.nx; i++) { mc.w[mc.idx(i, j, 0)] = 0; mc.w[mc.idx(i, j, c.nz)] = 0; }
  const plane = (src: ArrayLike<number>, dst: Float64Array): void => {
    for (let j = 0; j < c.ny; j++) for (let i = 0; i < c.nx; i++) {
      const { lx, ly, w } = local((i + 0.5) * c.dx, (j + 0.5) * c.dy);
      if (w <= 0) continue;
      let s = 0; for (let b = 0; b < ry; b++) for (let a = 0; a < rx; a++) s += ssf(src, lx + ((a + 0.5) / rx - 0.5) * c.dx, ly + ((b + 0.5) / ry - 0.5) * c.dy);
      dst[j * c.nx + i] = w * s / (rx * ry) + (1 - w) * dst[j * c.nx + i]!;
    }
  };
  if (accF && accC) { plane(accF.rain, accC.rain); plane(accF.snow, accC.snow); }
  if (surfF && surfC) surfF.forEach((a, n) => { if (surfC[n]) plane(a, surfC[n]!); });
  mc.time = mf.time; mc.steps = mf.steps;
}
