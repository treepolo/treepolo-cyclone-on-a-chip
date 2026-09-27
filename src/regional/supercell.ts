// Tornado-scale supercell environment (initial and boundary conditions only; the storm, its
// mesocyclone and any tornado-like vortex must develop from the equations).
//   * thermodynamics: Weisman & Klemp (1982) sounding (14 g/kg low-level vapour cap)
//   * wind: quarter-circle hodograph of radius R over the lowest 2 km, then a straight segment
//     to u = U6 at 6 km, constant above (a common idealised tornadic-supercell hodograph shape)
//   * the domain translates with the Bunkers et al. (2000) right-mover motion so that the storm
//     stays inside it; surface drag uses the ground-relative wind (log law over land, z0)
//   * initiation: a warm bubble

import { RegionalModel, RegionalConfig } from './core.js';
import { weismanKlemp } from './kessler.js';
import { QV } from './ice.js';
import type { RegionalPhysicsConfig } from './physics.js';

export interface Hodograph { R: number; U6: number }

/** Ground-relative environmental wind at height z (m). */
export function quarterCircleWind(z: number, h: Hodograph = { R: 10, U6: 30 }): { u: number; v: number } {
  if (z <= 2000) { const a = 0.5 * Math.PI * z / 2000; return { u: h.R * (1 - Math.cos(a)), v: h.R * Math.sin(a) }; }
  const f = Math.min(1, (z - 2000) / 4000);
  return { u: h.R + f * (h.U6 - h.R), v: h.R };
}

/** Bunkers et al. (2000) right-moving supercell motion: 0-6 km mean wind plus 7.5 m/s to the right
 *  of the shear vector between the 0-0.5 km and 5.5-6 km layer means. */
export function bunkersRightMover(wind: (z: number) => { u: number; v: number }): { u: number; v: number } {
  const mean = (z0: number, z1: number): { u: number; v: number } => {
    let u = 0, v = 0; const n = 200;
    for (let i = 0; i < n; i++) { const w = wind(z0 + (i + 0.5) * (z1 - z0) / n); u += w.u; v += w.v; }
    return { u: u / n, v: v / n };
  };
  const m = mean(0, 6000), lo = mean(0, 500), hi = mean(5500, 6000);
  const sx = hi.u - lo.u, sy = hi.v - lo.v, s = Math.hypot(sx, sy) || 1;
  return { u: m.u + 7.5 * sy / s, v: m.v - 7.5 * sx / s };
}

/** Build the experiment: model (6 moisture species for the ice scheme), physics configuration and
 *  the frame velocity. dx is the horizontal grid spacing, L the domain width. */
export function tornadoExperiment(dx: number, L: number, nz: number, dz: number, dt: number, nsound = 6): { model: RegionalModel; physics: RegionalPhysicsConfig; frame: { u: number; v: number }; description: string } {
  const nx = Math.round(L / dx);
  const cfg: RegionalConfig = { nx, ny: nx, nz, dx, dy: dx, dz, dt, nsound, f: 0, beta: 0.2, divDamp: 0.1, dampDepth: Math.min(5000, 0.3 * nz * dz), dampRate: 1 / 300, kdiff2: 0 };
  const m = new RegionalModel(cfg, weismanKlemp, 6);
  const frame = bunkersRightMover((z) => quarterCircleWind(z));
  m.setBaseWind((z) => { const w = quarterCircleWind(z); return { u: w.u - frame.u, v: w.v - frame.v }; });
  for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) m.scalars[QV]![m.idx(i, j, k)] = m.qv0[k]!;
  // warm bubble: 3 K, 5 km horizontal and 1.5 km vertical radius, centred at 1.5 km in the domain's left third
  const xc = 0.4 * L, yc = 0.4 * L;
  for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
    const r = Math.sqrt((((i + 0.5) * dx - xc) / 5000) ** 2 + (((j + 0.5) * dx - yc) / 5000) ** 2 + ((m.zc[k]! - 1500) / 1500) ** 2);
    if (r < 1) m.th[m.idx(i, j, k)] = m.th[m.idx(i, j, k)]! + 3 * Math.cos(0.5 * Math.PI * r) ** 2;
  }
  // isotropic Smagorinsky (mixing length 0.21 x filter width), land surface with log-law drag
  const delta = Math.cbrt(dx * dx * dz), n2 = nx * nx;
  const tsk = new Float64Array(n2).fill(m.th0[0]! * m.pi0[0]!), wet = new Float64Array(n2).fill(0.3);
  const physics: RegionalPhysicsConfig = { lh: 0.21 * delta, lv: 0.21 * delta, sst: 0, ck: 1.2e-3, radTau: 0, radMax: 0, surface: { tsk, wet }, z0: 0.1, frameVel: frame };
  return {
    model: m, physics, frame,
    description: `龍捲尺度超大胞 / tornado-scale supercell（Δx ${dx} m）：WK82 探空、四分之一圓風徑圖、地面摩擦；區域隨右移胞移動 (${frame.u.toFixed(1)}, ${frame.v.toFixed(1)}) m/s / WK82 sounding, quarter-circle hodograph, surface drag; domain moves with the right-mover`,
  };
}

/**
 * Keeps a storm inside a periodic domain (measurement + exact symmetries only): locates the main
 * updraft at ~4 km, estimates its motion from successive positions, then (a) changes the frame
 * velocity by the storm's relative motion (Galilean shift) and (b) rolls the fields by whole cells
 * when the storm is more than L/8 from the centre. The caller applies the returned actions to the
 * model (RegionalModel.shiftFrame / roll) and to the physics frame velocity.
 */
export class StormTracker {
  private last: { x: number; y: number; t: number } | null = null;

  update(m: RegionalModel): { du: number; dv: number; di: number; dj: number; x: number; y: number } | null {
    const { nx, ny, nz, dx, dy } = m.c;
    let k4 = 0; for (let k = 0; k < nz; k++) if (Math.abs(m.zc[k]! - 4000) < Math.abs(m.zc[k4]! - 4000)) k4 = k;
    let wmax = 0, im = 0, jm = 0;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const w = m.w[m.idx(i, j, k4)]!; if (w > wmax) { wmax = w; im = i; jm = j; } }
    if (wmax < 5) { this.last = null; return null; }
    // w-weighted centroid of strong updraft within 5 km of the maximum (periodic distances)
    const Lx = nx * dx, Ly = ny * dy, rad = 5000;
    let sw = 0, sx = 0, sy = 0;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const w = m.w[m.idx(i, j, k4)]!;
      if (w < 0.5 * wmax) continue;
      let ddx = (i - im) * dx, ddy = (j - jm) * dy;
      ddx -= Math.round(ddx / Lx) * Lx; ddy -= Math.round(ddy / Ly) * Ly;
      if (Math.hypot(ddx, ddy) > rad) continue;
      sw += w; sx += w * ddx; sy += w * ddy;
    }
    const x = (im + 0.5) * dx + sx / sw, y = (jm + 0.5) * dy + sy / sw, t = m.time;
    let du = 0, dv = 0;
    if (this.last && t > this.last.t) {
      let ddx = x - this.last.x, ddy = y - this.last.y;
      ddx -= Math.round(ddx / Lx) * Lx; ddy -= Math.round(ddy / Ly) * Ly;
      // apply half of the measured relative motion per update (damps jumps of the updraft centroid)
      du = 0.5 * ddx / (t - this.last.t); dv = 0.5 * ddy / (t - this.last.t);
    }
    // re-centre when more than L/8 from the middle
    const offx = x - Lx / 2, offy = y - Ly / 2;
    const di = Math.abs(offx) > Lx / 8 ? -Math.round(offx / dx) : 0, dj = Math.abs(offy) > Ly / 8 ? -Math.round(offy / dy) : 0;
    this.last = { x: x + di * dx, y: y + dj * dy, t };
    return { du, dv, di, dj, x, y };
  }
}
