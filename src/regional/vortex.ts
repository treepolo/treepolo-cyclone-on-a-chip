// A vortex added to the state by an interaction (the 3-D view's warm-core and cold-core vortex tools): a tangential wind of a chosen strength and size, with the
// pressure that balances it (gradient wind: dpi'/dr = (v^2/r + f v) / (cp theta_v)) and the temperature that balances that (hydrostatic: theta' = cp theta_v^2 / g dpi'/dz).
//
// The wind is vt(r, z) = dir vmax g(r) h(z): g is a bell, (r/rm) exp((1 - (r/rm)^2) / 2) times cos^2(pi r / 2R), scaled so that its largest value is 1 (the wind
// is vmax at the radius of maximum wind, which is rm up to the window), zero beyond the outer radius R; h is the cos^2 envelope of the other tools, 1 at the height z0
// of the strongest wind and zero H above and below it. A wind that decreases upwards (z0 at the ground) is a warm-core vortex, as a tropical cyclone; one that increases
// upwards (z0 aloft) is cold below that height, as a cold low in the upper troposphere. The result is an increment: it is added to the state (to winds, theta and the
// Exner perturbation), so vortices add up, and an existing storm keeps its own flow. Nothing is remembered: it is a start, not a forcing.
import type { RegionalModel } from './core.js';
import { pressure } from './diagnostics.js';

const G = 9.80665, CP = 1004.5;

export interface VortexSpec {
  /** centre (m, in the coordinates of the interactions) */
  x: number; y: number;
  /** height of the strongest wind (m) */
  z: number;
  /** outer radius (m): no wind beyond it */
  R: number;
  /** radius of maximum wind (m) */
  rm: number;
  /** half-depth (m): the wind is zero this far above and below z */
  H: number;
  /** strongest wind (m/s) */
  vmax: number;
  /** +1 counter-clockwise seen from above, -1 clockwise */
  dir: 1 | -1;
}

/** the increments of a patch of the grid, [k][jj][ii] (ii, jj from the cell (i0, j0); wrapped round a periodic domain) */
export interface VortexPatch {
  i0: number; j0: number; ni: number; nj: number; nz: number;
  du: Float32Array; dv: Float32Array; dth: Float32Array; dpp: Float32Array;
  /** change of surface pressure at the centre (hPa; negative: a low) and the strongest wind of the increment (m/s) */
  dpCentre: number; vmax: number;
}

/** what a model grid must show: its geometry and base state (a RegionalModel does) */
type GridLike = Pick<RegionalModel, 'c' | 'zc' | 'th0' | 'qv0' | 'pi0'>;

/** The increments of the vortex on a grid whose origin is at (ox, oy) in the coordinates of the interaction. */
export function vortexPatch(m: GridLike, s: VortexSpec, ox = 0, oy = 0): VortexPatch {
  const { nx, ny, nz, dx, dy, f } = m.c, periodic = m.c.lateral !== 'open', R = Math.max(s.R, 2 * Math.min(dx, dy)), rm = Math.min(Math.max(s.rm, Math.min(dx, dy)), 0.8 * R), H = Math.max(s.H, 1);
  // the bell, scaled to a largest value of 1
  const bell = (r: number): number => (r >= R ? 0 : (r / rm) * Math.exp(0.5 * (1 - (r / rm) ** 2)) * Math.cos(0.5 * Math.PI * r / R) ** 2);
  let peak = 0;
  for (let n = 1; n <= 2000; n++) peak = Math.max(peak, bell(n * R / 2000));
  const vt = (r: number, k: number): number => {
    const h = Math.abs(m.zc[k]! - s.z) < H ? Math.cos(0.5 * Math.PI * (m.zc[k]! - s.z) / H) ** 2 : 0;
    return s.dir * s.vmax * bell(r) / peak * h;
  };
  // pressure: gradient-wind balance integrated inward from R (where it is zero), per level
  const nr = Math.min(4000, Math.max(200, Math.ceil(2 * R / Math.min(dx, dy)))), dr = R / nr;
  const piR = new Float64Array(nz * (nr + 1));
  for (let k = 0; k < nz; k++) {
    const thv = m.th0[k]! * (1 + 0.61 * m.qv0[k]!);
    let pi = 0;
    for (let n = nr - 1; n >= 0; n--) {
      const r = (n + 0.5) * dr, v = vt(r, k);
      pi -= (v * v / r + f * v) / (CP * thv) * dr;
      piR[k * (nr + 1) + n] = pi;
    }
  }
  const piAt = (k: number, r: number): number => { const x = Math.min(nr - 1e-6, r / dr), n = Math.floor(x), w = x - n; return piR[k * (nr + 1) + n]! * (1 - w) + piR[k * (nr + 1) + n + 1]! * w; };
  // the patch of cells the vortex reaches
  let i0 = Math.floor((s.x - R - ox) / dx) - 1, i1 = Math.ceil((s.x + R - ox) / dx) + 1, j0 = Math.floor((s.y - R - oy) / dy) - 1, j1 = Math.ceil((s.y + R - oy) / dy) + 1;
  if (periodic) { if (i1 - i0 + 1 >= nx) { i0 = 0; i1 = nx - 1; } if (j1 - j0 + 1 >= ny) { j0 = 0; j1 = ny - 1; } }
  else { i0 = Math.max(0, i0); i1 = Math.min(nx - 1, i1); j0 = Math.max(0, j0); j1 = Math.min(ny - 1, j1); }
  const ni = Math.max(0, i1 - i0 + 1), nj = Math.max(0, j1 - j0 + 1), N = ni * nj * nz;
  const du = new Float32Array(N), dv = new Float32Array(N), dth = new Float32Array(N), dpp = new Float32Array(N);
  const Lx = nx * dx, Ly = ny * dy;
  const rel = (px: number, py: number): { ex: number; ey: number; r: number } => {
    let ex = px - s.x, ey = py - s.y;
    if (periodic) { ex -= Math.round(ex / Lx) * Lx; ey -= Math.round(ey / Ly) * Ly; }
    return { ex, ey, r: Math.hypot(ex, ey) };
  };
  let vmaxSeen = 0;
  const col = new Float64Array(nz);
  for (let jj = 0; jj < nj; jj++) for (let ii = 0; ii < ni; ii++) {
    const i = i0 + ii, j = j0 + jj;
    const a = rel(ox + i * dx, oy + (j + 0.5) * dy), b = rel(ox + (i + 0.5) * dx, oy + j * dy), c = rel(ox + (i + 0.5) * dx, oy + (j + 0.5) * dy);
    for (let k = 0; k < nz; k++) col[k] = c.r < R ? piAt(k, c.r) : 0;
    for (let k = 0; k < nz; k++) {
      const o = (k * nj + jj) * ni + ii;
      if (a.r > 0 && a.r < R) { const v = vt(a.r, k); du[o] = -v * a.ey / a.r; vmaxSeen = Math.max(vmaxSeen, Math.abs(v)); }
      if (b.r > 0 && b.r < R) { const v = vt(b.r, k); dv[o] = v * b.ex / b.r; }
      dpp[o] = col[k]!;
      const kp = Math.min(k + 1, nz - 1), km = Math.max(k - 1, 0), thv = m.th0[k]! * (1 + 0.61 * m.qv0[k]!);
      dth[o] = kp === km ? 0 : CP * thv * thv / G * (col[kp]! - col[km]!) / (m.zc[kp]! - m.zc[km]!);
    }
  }
  const pi00 = m.pi0[0]!;
  return { i0, j0, ni, nj, nz, du, dv, dth, dpp, dpCentre: (pressure(pi00 + piAt(0, 0)) - pressure(pi00)) / 100, vmax: vmaxSeen };
}

/** Add a patch to a CPU model (the cells of a periodic domain wrap round). */
export function applyVortexPatch(m: RegionalModel, p: VortexPatch): void {
  const { nx, ny } = m.c, periodic = m.c.lateral !== 'open';
  for (let k = 0; k < p.nz; k++) for (let jj = 0; jj < p.nj; jj++) for (let ii = 0; ii < p.ni; ii++) {
    let i = p.i0 + ii, j = p.j0 + jj;
    if (periodic) { i = ((i % nx) + nx) % nx; j = ((j % ny) + ny) % ny; } else if (i < 0 || i >= nx || j < 0 || j >= ny) continue;
    const o = (k * p.nj + jj) * p.ni + ii, q = m.idx(i, j, k);
    m.u[q] = m.u[q]! + p.du[o]!; m.v[q] = m.v[q]! + p.dv[o]!; m.th[q] = m.th[q]! + p.dth[o]!; m.pp[q] = m.pp[q]! + p.dpp[o]!;
  }
}
