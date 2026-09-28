// Massless tracer particles for the 3-D view (display only; they never feed back on the model): advected
// by the resolved wind, trilinearly interpolated from the C-grid, with midpoint (RK2) steps in the wind
// of the current state. Particles that leave an open domain, reach the damping layer or exceed their
// lifetime are re-seeded in the lowest 2 km (half of them near the storm, half anywhere), so the picture
// shows where boundary-layer air goes. The GPU kernel (GpuRegional.advectTracers) uses the same steps
// and the same hash-based random numbers.

import { H, type RegionalModel } from './core.js';

/** PCG hash (32-bit), as in the GPU kernel. */
export function pcg(v: number): number {
  const s = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
  const w = Math.imul(((s >>> ((s >>> 28) + 4)) ^ s) >>> 0, 277803737) >>> 0;
  return ((w >>> 22) ^ w) >>> 0;
}
export const rnd = (a: number): number => pcg(a) / 4294967296;

export interface TracerParams {
  /** lifetime (s) before a particle is re-seeded */
  life: number;
  /** seeding centre (m, domain coordinates) and radius for half of the particles (radius 0: anywhere) */
  cx: number; cy: number; rad: number;
  /** seeding layer depth (m) */
  zSeed: number;
}

/** Wind (u, v, w) at (x, y, z) in m from the domain origin, trilinear on the staggered grid. */
export function windAt(m: RegionalModel, x: number, y: number, z: number): [number, number, number] {
  const { dx, dy, dz, nz } = m.c;
  return [
    sample(m, m.u, x / dx, y / dy - 0.5, z / dz - 0.5, nz),
    sample(m, m.v, x / dx - 0.5, y / dy, z / dz - 0.5, nz),
    sample(m, m.w, x / dx - 0.5, y / dy - 0.5, z / dz, nz + 1),
  ];
}
function sample(m: RegionalModel, a: Float64Array, gi: number, gj: number, gk: number, nk: number): number {
  const { nx, ny } = m.c;
  const i0 = Math.max(-2, Math.min(nx, Math.floor(gi))), j0 = Math.max(-2, Math.min(ny, Math.floor(gj)));
  const kc = Math.max(0, Math.min(nk - 1, gk)), k0 = Math.min(nk - 2, Math.floor(kc));
  const fi = Math.max(0, Math.min(1, gi - i0)), fj = Math.max(0, Math.min(1, gj - j0)), fk = Math.max(0, Math.min(1, kc - k0));
  const q = k0 * m.plane + (j0 + H) * m.sx + (i0 + H), sx = m.sx, pl = m.plane;
  const l = (o: number): number => a[q + o]! * (1 - fi) + a[q + o + 1]! * fi;
  const b = l(0) * (1 - fj) + l(sx) * fj, t = l(pl) * (1 - fj) + l(pl + sx) * fj;
  return b * (1 - fk) + t * fk;
}

export class Tracers {
  /** x, y, z (m) and age (s) per particle */
  readonly pos: Float32Array;
  /** re-seeding counter (advances the random sequence) */
  seed = 1;
  constructor(readonly n: number, private readonly m: RegionalModel, public params: TracerParams) {
    this.pos = new Float32Array(4 * n);
    for (let p = 0; p < n; p++) { this.spawn(p, 0x9e3779b9); this.pos[4 * p + 3] = rnd(p * 7919 + 17) * params.life; }
  }

  /** Re-seed particle p (same formula as the GPU kernel). */
  spawn(p: number, seed: number): void { spawnInto(this.pos, p, seed, this.m, this.params); }

  /** Advance by dt in nsub midpoint steps. */
  advect(dt: number, nsub: number): void {
    const m = this.m, { nx, ny, nz, dx, dy, dz } = m.c, Lx = nx * dx, Ly = ny * dy, top = nz * dz - m.c.dampDepth, open = m.c.lateral === 'open';
    const h = dt / Math.max(1, nsub), P = this.pos;
    this.seed = (this.seed + 1) >>> 0;
    for (let p = 0; p < this.n; p++) {
      let x = P[4 * p]!, y = P[4 * p + 1]!, z = P[4 * p + 2]!;
      for (let s = 0; s < nsub; s++) {
        const a = windAt(m, x, y, z);
        const b = windAt(m, x + 0.5 * h * a[0], y + 0.5 * h * a[1], Math.max(0, z + 0.5 * h * a[2]));
        x += h * b[0]; y += h * b[1]; z = Math.max(0, z + h * b[2]);
        if (!open) { x -= Math.floor(x / Lx) * Lx; y -= Math.floor(y / Ly) * Ly; }
      }
      P[4 * p] = x; P[4 * p + 1] = y; P[4 * p + 2] = z; P[4 * p + 3] = P[4 * p + 3]! + dt;
      if (P[4 * p + 3]! > this.params.life || z > top || (open && (x < 0 || y < 0 || x > Lx || y > Ly))) this.spawn(p, Math.imul(this.seed, 2654435761) >>> 0);
    }
  }
}

/** Seed particle p: position from hashes of (p, seed), age 0. */
export function spawnInto(P: Float32Array, p: number, seed: number, m: RegionalModel, pr: TracerParams): void {
  const { nx, ny, dx, dy } = m.c, Lx = nx * dx, Ly = ny * dy;
  const h = (k: number): number => rnd((Math.imul(p, 4) + k + seed) >>> 0);
  let x: number, y: number;
  if (pr.rad > 0 && (p & 1) === 0) {
    const a = 6.2831853 * h(0), r = pr.rad * Math.sqrt(h(1));
    x = Math.max(0, Math.min(Lx, pr.cx + r * Math.cos(a))); y = Math.max(0, Math.min(Ly, pr.cy + r * Math.sin(a)));
  } else { x = h(0) * Lx; y = h(1) * Ly; }
  P[4 * p] = x; P[4 * p + 1] = y; P[4 * p + 2] = 50 + h(2) * (pr.zSeed - 50); P[4 * p + 3] = 0;
}
