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

/** Interpolate the state of the coarse model `mc` into the fine model `mf`, whose domain starts at
 *  (x0, y0) metres in the coarse model's coordinates. Surface accumulations ([j][i]) are interpolated
 *  too. The fine model's time is set to the coarse model's. */
export function refineInto(mc: RegionalModel, mf: RegionalModel, x0 = 0, y0 = 0, accC?: Accumulations, accF?: Accumulations): void {
  const c = mc.c, f = mf.c;
  const periodic = c.lateral !== 'open';
  // coarse sample index along one axis: fractional index -> two integer indices and a weight
  const axis = (p: number, n: number): [number, number, number] => {
    if (periodic) {
      const i0 = Math.floor(p), w = p - i0;
      return [((i0 % n) + n) % n, (((i0 + 1) % n) + n) % n, w];
    }
    const q = Math.max(0, Math.min(n - 1, p)), i0 = Math.min(Math.floor(q), n - 2 < 0 ? 0 : n - 2);
    return n < 2 ? [0, 0, 0] : [i0, i0 + 1, q - i0];
  };
  const vaxis = (p: number, n: number): [number, number, number] => {
    const q = Math.max(0, Math.min(n - 1, p)), i0 = Math.min(Math.floor(q), Math.max(0, n - 2));
    return n < 2 ? [0, 0, 0] : [i0, i0 + 1, q - i0];
  };
  /** trilinear sample; sx, sy: horizontal staggering (0 = face, 0.5 = centre); faceZ: w levels */
  const sample = (a: Float64Array, x: number, y: number, z: number, sx: number, sy: number, faceZ: boolean): number => {
    // number of distinct points along each axis (faces: nx + 1 with open boundaries, nx if periodic)
    const nI = sx === 0 && !periodic ? c.nx + 1 : c.nx, nJ = sy === 0 && !periodic ? c.ny + 1 : c.ny;
    const [i0, i1, wx] = axis(x / c.dx - sx, nI), [j0, j1, wy] = axis(y / c.dy - sy, nJ);
    const [k0, k1, wz] = faceZ ? vaxis(z / c.dz, c.nz + 1) : vaxis(z / c.dz - 0.5, c.nz);
    const g = (i: number, j: number, k: number): number => a[mc.idx(i, j, k)]!;
    const l0 = (1 - wy) * ((1 - wx) * g(i0, j0, k0) + wx * g(i1, j0, k0)) + wy * ((1 - wx) * g(i0, j1, k0) + wx * g(i1, j1, k0));
    const l1 = (1 - wy) * ((1 - wx) * g(i0, j0, k1) + wx * g(i1, j0, k1)) + wy * ((1 - wx) * g(i0, j1, k1) + wx * g(i1, j1, k1));
    return (1 - wz) * l0 + wz * l1;
  };
  const fill = (dst: Float64Array, src: Float64Array, sx: number, sy: number, faceZ: boolean, iMax: number, jMax: number, kMax: number): void => {
    for (let k = 0; k < kMax; k++) {
      const z = faceZ ? k * f.dz : (k + 0.5) * f.dz;
      for (let j = 0; j < jMax; j++) {
        const y = y0 + (j + sy) * f.dy;
        for (let i = 0; i < iMax; i++) dst[mf.idx(i, j, k)] = sample(src, x0 + (i + sx) * f.dx, y, z, sx, sy, faceZ);
      }
    }
  };
  const fOpen = f.lateral === 'open';
  fill(mf.u, mc.u, 0, 0.5, false, f.nx + (fOpen ? 1 : 0), f.ny, f.nz);
  fill(mf.v, mc.v, 0.5, 0, false, f.nx, f.ny + (fOpen ? 1 : 0), f.nz);
  fill(mf.w, mc.w, 0.5, 0.5, true, f.nx, f.ny, f.nz + 1);
  fill(mf.th, mc.th, 0.5, 0.5, false, f.nx, f.ny, f.nz);
  fill(mf.pp, mc.pp, 0.5, 0.5, false, f.nx, f.ny, f.nz);
  const ns = Math.min(mc.scalars.length, mf.scalars.length);
  for (let s = 0; s < ns; s++) {
    fill(mf.scalars[s]!, mc.scalars[s]!, 0.5, 0.5, false, f.nx, f.ny, f.nz);
    const a = mf.scalars[s]!;
    for (let i = 0; i < a.length; i++) if (a[i]! < 0) a[i] = 0;
  }
  // no vertical motion through the ground and the lid
  for (let j = 0; j < f.ny; j++) for (let i = 0; i < f.nx; i++) { mf.w[mf.idx(i, j, 0)] = 0; mf.w[mf.idx(i, j, f.nz)] = 0; }
  if (accC && accF) {
    for (const key of ['rain', 'snow'] as const) {
      const src = accC[key], dst = accF[key];
      for (let j = 0; j < f.ny; j++) for (let i = 0; i < f.nx; i++) {
        const [i0, i1, wx] = axis((x0 + (i + 0.5) * f.dx) / c.dx - 0.5, c.nx), [j0, j1, wy] = axis((y0 + (j + 0.5) * f.dy) / c.dy - 0.5, c.ny);
        dst[j * f.nx + i] = (1 - wy) * ((1 - wx) * src[j0 * c.nx + i0]! + wx * src[j0 * c.nx + i1]!) + wy * ((1 - wx) * src[j1 * c.nx + i0]! + wx * src[j1 * c.nx + i1]!);
      }
    }
  }
  mf.time = mc.time;
}
