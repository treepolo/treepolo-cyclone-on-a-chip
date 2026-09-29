// Wind interactions of the regional model: a wind added once, or a lasting forcing that nudges the wind in a region
// toward a target (a steady push, a rotation or a convergence). Only the wind is touched; pressure, temperature, cloud
// and rain respond through the equations.
//
// Region: centre (x, y, z), horizontal radius R and half-depth H; envelope e = cos^2(pi r / 2R) cos^2(pi |z - zc| / 2H)
// inside, 0 outside. Target wind (departure from the background wind ub, vb):
//   push:     speed * (unit 3-D direction), the same everywhere in the region;
//   rotate:   speed * s(r) * tangential unit (sign +1 counter-clockwise seen from above, -1 clockwise);
//   converge: speed * s(r) * radial unit toward the centre (sign +1) or away from it (sign -1);
// with s(r) = sin(pi r / R) (calm at the centre and the edge). A one-time wind adds e * target; a lasting one relaxes the
// departure toward the target: d(u - ub)/dt = e (target - (u - ub)) / FORCING_TAU. w stays 0 at the ground and the lid.

import type { RegionalModel } from './core.js';

export const FORCING_TAU = 600;
/** at most this many lasting forcings at once (the GPU kernel's table) */
export const MAX_FORCINGS = 8;

export interface WindForcing {
  x: number; y: number; z: number; R: number; H: number;
  /** target speed (m/s); push: unit direction (east, north, up) */
  speed: number; dir: [number, number, number];
  form: 'push' | 'rotate' | 'converge'; sign: 1 | -1;
}
export const FORM_ID = { push: 0, rotate: 1, converge: 2 } as const;

/** Envelope and target wind (u, v, w departure) of a forcing at a point (m); periodic domains wrap the offsets. */
export function forcingAt(f: WindForcing, px: number, py: number, pz: number, L: { x: number; y: number } | null): { e: number; tu: number; tv: number; tw: number } {
  let dx = px - f.x, dy = py - f.y;
  if (L) { dx -= Math.round(dx / L.x) * L.x; dy -= Math.round(dy / L.y) * L.y; }
  const r = Math.hypot(dx, dy), rh = r / f.R, rz = Math.abs(pz - f.z) / f.H;
  if (rh >= 1 || rz >= 1) return { e: 0, tu: 0, tv: 0, tw: 0 };
  const e = Math.cos(0.5 * Math.PI * rh) ** 2 * Math.cos(0.5 * Math.PI * rz) ** 2;
  if (f.form === 'push') return { e, tu: f.speed * f.dir[0], tv: f.speed * f.dir[1], tw: f.speed * f.dir[2] };
  if (r < 1e-6) return { e, tu: 0, tv: 0, tw: 0 };
  const s = f.speed * Math.sin(Math.PI * rh), cx = dx / r, cy = dy / r;
  if (f.form === 'rotate') return { e, tu: -f.sign * s * cy, tv: f.sign * s * cx, tw: 0 };
  return { e, tu: -f.sign * s * cx, tv: -f.sign * s * cy, tw: 0 };
}

/** Apply forcings to the CPU model: once (add e * target) or as a relaxation over dt (lasting). */
export function applyWind(m: RegionalModel, list: readonly WindForcing[], dt: number | 'once'): void {
  if (!list.length) return;
  const { nx, ny, nz, dx, dy } = m.c, L = m.c.lateral === 'open' ? null : { x: nx * dx, y: ny * dy };
  const once = dt === 'once', a = once ? 0 : (dt as number) / FORCING_TAU;
  for (const f of list) {
    // the cells the region can touch (with wrapping on periodic domains)
    const ri = Math.ceil(f.R / dx) + 1, rj = Math.ceil(f.R / dy) + 1, ic = Math.floor(f.x / dx), jc = Math.floor(f.y / dy);
    for (let jj = jc - rj; jj <= jc + rj; jj++) for (let ii = ic - ri; ii <= ic + ri; ii++) {
      let i = ii, j = jj;
      if (L) { i = ((i % nx) + nx) % nx; j = ((j % ny) + ny) % ny; } else if (i < 0 || j < 0 || i >= nx || j >= ny) continue;
      const xs = ii * dx, ys = jj * dy, xc = (ii + 0.5) * dx, yc = (jj + 0.5) * dy;
      for (let k = 0; k <= nz; k++) {
        const q = m.idx(i, j, k);
        if (k < nz) {
          const zc = m.zc[k]!;
          const fu = forcingAt(f, xs, yc, zc, L);
          if (fu.e > 0) m.u[q] = m.u[q]! + (once ? fu.e * fu.tu : a * fu.e * (fu.tu - (m.u[q]! - m.ub[k]!)));
          const fv = forcingAt(f, xc, ys, zc, L);
          if (fv.e > 0) m.v[q] = m.v[q]! + (once ? fv.e * fv.tv : a * fv.e * (fv.tv - (m.v[q]! - m.vb[k]!)));
        }
        if (k > 0 && k < nz) {
          const fw = forcingAt(f, xc, yc, m.zf[k]!, L);
          if (fw.e > 0) m.w[q] = m.w[q]! + (once ? fw.e * fw.tw : a * fw.e * (fw.tw - m.w[q]!));
        }
      }
    }
  }
}

/** Forcing table for the GPU kernel: per forcing 3 vec4 (x, y, z, R | H, speed, form, sign | dir x, y, z, 0). */
export function forcingTable(list: readonly WindForcing[]): Float32Array {
  const t = new Float32Array(12 * MAX_FORCINGS);
  list.slice(0, MAX_FORCINGS).forEach((f, n) => {
    t.set([f.x, f.y, f.z, f.R, f.H, f.speed, FORM_ID[f.form], f.sign, f.dir[0], f.dir[1], f.dir[2], 0], 12 * n);
  });
  return t;
}
