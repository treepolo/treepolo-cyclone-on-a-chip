// What the 3-D view draws of an outer grid (extension.ts): the outer model's display volume, reduced horizontally when it is large, and where it lies around the
// inner domain. Pure functions (the worker reads the bytes; protocol.ts has the frame).

import type { OuterFrame } from './protocol.js';

/** most voxels of the outer volume sent to the 3-D view */
export const OUTER_DISPLAY_MAX = 4e6;

/** The horizontal reduction (whole cells per side, at least 1) that brings an nx x nx x nz volume to at most `cap` voxels. */
export function decimationFor(nx: number, nz: number, cap: number = OUTER_DISPLAY_MAX): number {
  let f = 1;
  while (Math.ceil(nx / f) ** 2 * nz > cap) f++;
  return f;
}

/** A byte of display extinction (cube root coding: the extinction is the byte cubed) */
const cube = (b: number): number => { const x = b / 255; return x * x * x; };
const uncube = (e: number): number => Math.round(255 * Math.cbrt(Math.max(0, e)));

/**
 * Reduce a [k][j][i] byte volume of nx x nx columns by f across (blocks of f x f columns; the last blocks are narrower): the extinction (the byte cubed) is averaged,
 * so a uniform volume stays as it is and a lone cloud keeps the mass it had. Returns the volume and its width.
 */
export function decimateVolume(a: Uint8Array, nx: number, nz: number, f: number): { data: Uint8Array; nx: number } {
  if (f <= 1) return { data: a, nx };
  const m = Math.ceil(nx / f), out = new Uint8Array(m * m * nz);
  for (let k = 0; k < nz; k++) for (let J = 0; J < m; J++) for (let I = 0; I < m; I++) {
    let s = 0, n = 0;
    for (let j = J * f; j < Math.min(nx, (J + 1) * f); j++) for (let i = I * f; i < Math.min(nx, (I + 1) * f); i++) { s += cube(a[(k * nx + j) * nx + i]!); n++; }
    out[(k * m + J) * m + I] = uncube(s / n);
  }
  return { data: out, nx: m };
}

/**
 * The frame of an outer grid for the 3-D view. `onx`, `onz`, `odx`, `odz`: the outer model's grid; `x0`, `y0`: where the inner box lies in the outer grid (m; the outer corner is
 * at minus that in the inner grid's coordinates); `ring`: the inner edge band (cells) that relaxes to the outer grid; `idx`: the inner spacing (m).
 */
export function outerFrameOf(cloud: Uint8Array, rain: Uint8Array, onx: number, onz: number, odx: number, odz: number, x0: number, y0: number, ring: number, idx: number, cap: number = OUTER_DISPLAY_MAX): OuterFrame {
  const f = decimationFor(onx, onz, cap);
  const c = decimateVolume(cloud, onx, onz, f), r = decimateVolume(rain, onx, onz, f);
  return { nx: c.nx, nz: onz, f, dx: odx * f, dz: odz, x0: -x0, y0: -y0, L: onx * odx, top: onz * odz, taper: ring * idx, cloud: c.data, rain: r.data };
}
