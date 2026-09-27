// Regional-model sub-grid and surface physics for convection / tropical-cyclone experiments
// (after Rotunno & Emanuel 1987; Bryan & Rotunno 2009, CM1):
//   * Smagorinsky–Lilly eddy viscosity with separate horizontal / vertical mixing lengths and a
//     Richardson-number stability correction; eddy diffusion of u, v, w, theta and moisture
//     (Prandtl number 1/3 for scalars)
//   * bulk sea-surface fluxes of momentum, sensible heat and water vapour over a fixed SST
//     (Cd from Donelan et al. 2004-style capping, constant Ck)
//   * Newtonian radiative relaxation of theta toward the base state, capped at 2 K/day of cooling
// Everything is applied as a slow tendency of the regional core through its `physicsTend` hook.

import { DRY_AIR } from '../core/constants.js';
import { RegionalModel } from './core.js';
import { QV } from './kessler.js';

export interface RegionalPhysicsConfig {
  lh: number;              // horizontal mixing length (m)
  lv: number;              // vertical mixing length (m)
  sst: number;             // K (0 = no surface fluxes)
  ck: number;              // enthalpy exchange coefficient
  radTau: number;          // s (0 = no radiation)
  radMax: number;          // max cooling K/s
  /** per-column surface (nx*ny, row-major j*nx+i): skin temperature (K) and moisture availability
   *  (1 = sea, bucket fraction over land). Overrides the uniform sst (nested runs). */
  surface?: { tsk: ArrayLike<number>; wet: ArrayLike<number> } | null;
  /** land roughness length (m): if set, the drag coefficient is the neutral log-law value
   *  (kappa / ln(z1/z0))^2 at the lowest model level z1 instead of the sea-surface formula */
  z0?: number;
  /** ground-relative velocity of the model frame (m/s) when the domain translates with a storm:
   *  surface drag and fluxes use the ground-relative wind */
  frameVel?: { u: number; v: number };
}

/** Drag coefficient at the lowest level: neutral log law over land (z0 given) or the Donelan-type
 *  sea-surface formula 1e-3 (1 + 0.07 U) capped at 2.4e-3. */
export function dragCoefficient(c: RegionalPhysicsConfig, z1: number, spd: number): number {
  if (c.z0 && c.z0 > 0) return (0.4 / Math.log(z1 / c.z0)) ** 2;
  return Math.min(2.4e-3, 1e-3 * (1 + 0.07 * spd));
}

/** Surface state used by the bulk fluxes: per-column skin temperature and wetness, surface Exner
 *  function and pressure. A uniform SST uses pi_s = 1, p_s = 1000 hPa (the idealised TC set-up);
 *  a per-column surface extrapolates the base state hydrostatically to z = 0. */
export function surfaceState(m: RegionalModel, c: RegionalPhysicsConfig): { tsk: Float64Array; wet: Float64Array; pis: number; psfc: number } | null {
  const n = m.c.nx * m.c.ny;
  if (c.surface) {
    const pis = m.pi0[0]! + 9.80665 * 0.5 * m.c.dz / (DRY_AIR.cp * m.th0[0]! * (1 + 0.61 * m.qv0[0]!));
    return { tsk: Float64Array.from(c.surface.tsk), wet: Float64Array.from(c.surface.wet), pis, psfc: DRY_AIR.pRef * Math.pow(pis, DRY_AIR.cp / DRY_AIR.rd) };
  }
  if (c.sst > 0) return { tsk: new Float64Array(n).fill(c.sst), wet: new Float64Array(n).fill(1), pis: 1, psfc: DRY_AIR.pRef };
  return null;
}

export class RegionalPhysics {
  readonly Km: Float64Array;   // eddy viscosity at cell centres
  /** surface enthalpy flux diagnostics (W m^-2), last call */
  readonly shf: Float64Array; readonly lhf: Float64Array;

  private readonly sfc: ReturnType<typeof surfaceState>;

  constructor(private readonly m: RegionalModel, readonly cfg: RegionalPhysicsConfig) {
    this.sfc = surfaceState(m, cfg);
    this.Km = new Float64Array(m.size);
    this.shf = new Float64Array(m.c.nx * m.c.ny);
    this.lhf = new Float64Array(m.c.nx * m.c.ny);
    m.physicsTend = (mm, t): void => this.tendencies(mm, t);
  }

  private tendencies(m: RegionalModel, t: { fu: Float64Array; fv: Float64Array; fw: Float64Array; fth: Float64Array; fsc: Float64Array[] }): void {
    const { nx, ny, nz, dx, dy, dz } = m.c, sx = m.sx, pl = m.plane, c = this.cfg;
    const u = m.u, v = m.v, w = m.w, th = m.th;
    const g = 9.80665;
    // ---- eddy viscosity from the deformation and N^2 (cell centres)
    const K = this.Km;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const q = m.idx(i, j, k);
      const dudx = (u[q + 1]! - u[q]!) / dx, dvdy = (v[q + sx]! - v[q]!) / dy, dwdz = (w[q + pl]! - w[q]!) / dz;
      const dudy = 0.25 * ((u[q + sx]! + u[q + sx + 1]!) - (u[q - sx]! + u[q - sx + 1]!)) / dy;
      const dvdx = 0.25 * ((v[q + 1]! + v[q + 1 + sx]!) - (v[q - 1]! + v[q - 1 + sx]!)) / dx;
      const kp = Math.min(k + 1, nz - 1), km = Math.max(k - 1, 0), dzz = (kp - km) * dz || dz;
      const dudz = 0.5 * ((u[m.idx(i, j, kp)]! + u[m.idx(i, j, kp) + 1]!) - (u[m.idx(i, j, km)]! + u[m.idx(i, j, km) + 1]!)) / dzz;
      const dvdz = 0.5 * ((v[m.idx(i, j, kp)]! + v[m.idx(i, j, kp) + sx]!) - (v[m.idx(i, j, km)]! + v[m.idx(i, j, km) + sx]!)) / dzz;
      const S2 = 2 * (dudx * dudx + dvdy * dvdy + dwdz * dwdz) + (dudy + dvdx) ** 2 + dudz * dudz + dvdz * dvdz;
      const N2 = g * (th[m.idx(i, j, kp)]! - th[m.idx(i, j, km)]!) / (dzz * th[q]!);
      const ri = N2 / Math.max(S2, 1e-10);
      const stab = Math.sqrt(Math.max(0, 1 - 3 * ri));    // Pr = 1/3
      // anisotropic: use lh for horizontal and lv for vertical diffusion (stored as scaled deformation)
      K[q] = Math.sqrt(S2) * stab;
    }
    m.fillHalo(K, nz);
    const lh2 = c.lh * c.lh, lv2 = c.lv * c.lv;
    const diff = (a: Float64Array, out: Float64Array, prandtl: number, base: Float64Array | null): void => {
      for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const q = m.idx(i, j, k);
        const b0 = base ? base[k]! : 0;
        const ac = a[q]! - b0;
        const kh = (qq: number): number => lh2 * K[qq]! / prandtl;
        const fxr = 0.5 * (kh(q) + kh(q + 1)) * (a[q + 1]! - a[q]!) / dx, fxl = 0.5 * (kh(q) + kh(q - 1)) * (a[q]! - a[q - 1]!) / dx;
        const fyr = 0.5 * (kh(q) + kh(q + sx)) * (a[q + sx]! - a[q]!) / dy, fyl = 0.5 * (kh(q) + kh(q - sx)) * (a[q]! - a[q - sx]!) / dy;
        let fzt = 0, fzb = 0;
        if (k < nz - 1) { const kv = 0.5 * (K[q]! + K[q + pl]!) * lv2 / prandtl * m.rho0f[k + 1]!; fzt = kv * ((a[q + pl]! - (base ? base[k + 1]! : 0)) - ac) / dz; }
        if (k > 0) { const kv = 0.5 * (K[q]! + K[q - pl]!) * lv2 / prandtl * m.rho0f[k]!; fzb = kv * (ac - (a[q - pl]! - (base ? base[k - 1]! : 0))) / dz; }
        out[q] = out[q]! + (fxr - fxl) / dx + (fyr - fyl) / dy + (fzt - fzb) / (m.rho0[k]! * dz);
      }
    };
    diff(u, t.fu, 1, null);
    diff(v, t.fv, 1, null);
    diff(th, t.fth, 1 / 3, m.th0);
    for (let s = 0; s < t.fsc.length; s++) diff(m.scalars[s]!, t.fsc[s]!, 1 / 3, s === QV ? m.qv0 : null);
    // w: horizontal + vertical diffusion at w levels
    for (let k = 1; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const q = m.idx(i, j, k);
      const kc = 0.5 * (K[q]! + K[q - pl]!);
      t.fw[q] = t.fw[q]! + lh2 * kc * ((w[q + 1]! - 2 * w[q]! + w[q - 1]!) / (dx * dx) + (w[q + sx]! - 2 * w[q]! + w[q - sx]!) / (dy * dy))
        + lv2 * kc * (w[q + pl]! - 2 * w[q]! + w[q - pl]!) / (dz * dz);
    }

    // ---- surface fluxes (lowest model level)
    const sf = this.sfc;
    if (sf) {
      const cp = DRY_AIR.cp;
      const qv = m.scalars[QV];
      const pi1 = m.pi0[0]!, rho1 = m.rho0[0]!;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const q = m.idx(i, j, 0), c2 = j * nx + i;
        const tsk = sf.tsk[c2]!;
        const esS = 611.2 * Math.exp(17.67 * (tsk - 273.15) / (tsk - 29.65));
        const qsS = 0.622 * esS / (sf.psfc - 0.378 * esS);
        const thS = tsk / sf.pis;
        // ground-relative wind (the model frame may translate with a storm at frameVel)
        const ua = 0.5 * (u[q]! + u[q + 1]!) + (c.frameVel?.u ?? 0), va = 0.5 * (v[q]! + v[q + sx]!) + (c.frameVel?.v ?? 0);
        const spd = Math.max(Math.hypot(ua, va), 1);
        // drag coefficient: 1e-3 (1 + 0.07 U) capped at 2.4e-3 (Donelan-type saturation)
        const cd = dragCoefficient(c, 0.5 * dz, spd);
        const taux = cd * spd * ua, tauy = cd * spd * va;
        t.fu[q] = t.fu[q]! - 0.5 * taux / dz; t.fu[q + 1] = t.fu[q + 1]! - 0.5 * taux / dz;
        t.fv[q] = t.fv[q]! - 0.5 * tauy / dz; t.fv[q + sx] = t.fv[q + sx]! - 0.5 * tauy / dz;
        const fth = c.ck * spd * (thS - th[q]!);             // K m/s
        t.fth[q] = t.fth[q]! + fth / dz;
        this.shf[c2] = rho1 * cp * pi1 * fth;
        if (qv) {
          let fq = c.ck * spd * (qsS - qv[q]!);
          if (fq > 0) fq *= sf.wet[c2]!;                     // evaporation limited by surface wetness
          t.fsc[QV]![q] = t.fsc[QV]![q]! + fq / dz;
          this.lhf[c2] = rho1 * 2.5e6 * fq;
        }
      }
    }

    // ---- Newtonian radiative relaxation, capped cooling
    if (c.radTau > 0) {
      for (let k = 0; k < nz; k++) {
        const pi = m.pi0[k]!;
        for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
          const q = m.idx(i, j, k);
          let r = -(th[q]! - m.th0[k]!) / c.radTau;
          r = Math.max(r, -c.radMax / pi);
          t.fth[q] = t.fth[q]! + r;
        }
      }
    }
  }
}
