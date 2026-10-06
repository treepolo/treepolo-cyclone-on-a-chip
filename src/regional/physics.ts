// Regional-model sub-grid and surface physics for convection / tropical-cyclone experiments
// (after Rotunno & Emanuel 1987; Bryan & Rotunno 2009, CM1):
//   * Smagorinsky–Lilly eddy viscosity with separate horizontal / vertical mixing lengths and a
//     Richardson-number stability correction; eddy diffusion of u, v, w, theta and moisture
//     (Prandtl number 1/3 for scalars)
//   * bulk sea-surface fluxes of momentum, sensible heat and water vapour over a fixed SST
//     (Cd from Donelan et al. 2004-style capping, constant Ck)
//   * Newtonian radiative relaxation of theta toward the base state, capped at 2 K/day of cooling
// Everything is applied as a slow tendency of the regional core through its `physicsTend` hook.

import { DRY_AIR, T_FLOOR } from '../core/constants.js';
import { RegionalModel } from './core.js';
import { QV, QR } from './kessler.js';
import { pcg, rnd } from './tracers.js';
import { applyCumulus, cumulusScale, type CumulusInfo } from './cumulus.js';

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
  /** minimum wind speed in the surface fluxes (m/s, gustiness; default 1) */
  vmin?: number;
  /** constant clear-sky tropospheric cooling (K/s) instead of the relaxation toward the base state (0 or
   *  undefined: RE87 relaxation); the stratosphere (base-state T < 210 K) still relaxes */
  radConst?: number;
  /** stochastic boundary-layer perturbations (K; 0 or undefined: none): every BL_NOISE_PERIOD seconds each column's
   *  theta below BL_NOISE_DEPTH changes by blNoise * (2 r - 1), r uniform in [0, 1). They stand for the turbulent thermals
   *  a 3-15 km grid cannot resolve, which start convection wherever the air is unstable (periodic domains only). */
  blNoise?: number;
  /** surface gustiness: the bulk fluxes use sqrt(U^2 + (1.2 w*)^2 + Ug^2), with the free-convection velocity w* from the
   *  surface buoyancy flux (Beljaars 1995, boundary layer 1 km deep) and Ug from the near-surface rain rate (convective
   *  downdraft gusts, Redelsperger et al. 2000), never below vmin: calm air over a warm sea still evaporates */
  gust?: boolean;
  /** scale-aware simplified Betts-Miller cumulus parameterization (cumulus.ts), applied at the start of every step */
  cumulus?: boolean;
}

/** Effective wind speed of the bulk fluxes (m/s): mean wind `spd`, gustiness from the surface buoyancy flux (surface minus
 *  air potential temperature dth K, specific-humidity difference dq, air theta th, exchange coefficient ck) and the rain
 *  rate from the lowest-level rain water qr (kg/kg) at density rho. The GPU flux kernel uses the same formula. */
export function gustSpeed(spd: number, vmin: number, ck: number, dth: number, dq: number, th: number, qr: number, rho: number): number {
  const u0 = Math.max(spd, vmin), b = ck * u0 * (dth + 0.61 * th * dq);
  const ws = b > 0 ? Math.cbrt(9.80665 / th * b * 1000) : 0;
  // rain rate (cm/day) from Kessler-Wilhelmson fall speed, Redelsperger et al. (2000) gustiness, capped at its 7 cm/day maximum
  const rq = Math.max(0, rho * qr), rcd = rq > 0 ? Math.min(7, rq * 36.34 * Math.pow(1e-3 * rq, 0.1364) * 3600 * 2.4) : 0;
  const ug = Math.log(1 + 6.69 * rcd - 0.476 * rcd * rcd);
  return Math.max(vmin, Math.sqrt(spd * spd + 1.44 * ws * ws + ug * ug));
}

/**
 * Saturation mixing ratio (kg/kg) of the air at the surface, from the skin temperature (K) and the surface pressure (Pa). Any skin
 * temperature gives a number: below T_FLOOR the formula says nothing but zero (the vapour pressure there is 1e-17 Pa), and above the
 * boiling point the vapour pressure would pass the pressure of the air, so it stops at 0.9 of it (air of nearly pure vapour).
 */
export function surfaceQs(tsk: number, psfc: number): number {
  const T = Math.max(tsk, T_FLOOR), es = Math.min(611.2 * Math.exp(17.67 * (T - 273.15) / (T - 29.65)), 0.9 * psfc);
  return 0.622 * es / (psfc - 0.378 * es);
}

export const BL_NOISE_PERIOD = 600, BL_NOISE_DEPTH = 1000;
/** Random number in [-1, 1) of column (i, j) in noise interval `epoch` (the GPU kernel uses the same hash). */
export function blNoiseValue(i: number, j: number, nx: number, epoch: number): number {
  return 2 * rnd(((i + nx * j) ^ pcg(epoch >>> 0)) >>> 0) - 1;
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
  /** eddy viscosity at cell centres (allocated with the cache on the first CPU step) */
  Km: Float64Array = new Float64Array(0);
  /** surface enthalpy flux diagnostics (W m^-2), last call */
  readonly shf: Float64Array; readonly lhf: Float64Array;

  private readonly sfc: ReturnType<typeof surfaceState>;
  /** Current per-column surface (skin temperature K, wetness), or null without surface fluxes. */
  get surface(): { tsk: Float64Array; wet: Float64Array } | null { return this.sfc ? { tsk: this.sfc.tsk, wet: this.sfc.wet } : null; }

  constructor(private readonly m: RegionalModel, readonly cfg: RegionalPhysicsConfig) {
    this.sfc = surfaceState(m, cfg);
    this.shf = new Float64Array(m.c.nx * m.c.ny);
    this.lhf = new Float64Array(m.c.nx * m.c.ny);
    m.physicsTend = (mm, t, stage): void => this.tendencies(mm, t, stage);
    const noise = (cfg.blNoise ?? 0) > 0, cu = !!cfg.cumulus && cumulusScale(m.c.dx) > 0;
    if (cu) { const n = m.c.nx * m.c.ny; this.cu = { kb: new Int16Array(n).fill(-1), kt: new Int16Array(n).fill(-1), rate: new Float32Array(n) }; }
    if (noise || cu) m.preStep = (mm): void => { if (noise) this.noise(mm); if (this.cu) applyCumulus(mm, mm.c.dt, mm.scalars.length >= 6, this.cu); };
  }
  /** parameterized convection of the last step (cloud base / top level, rain rate), null without the scheme */
  cu: CumulusInfo | null = null;

  /** last noise interval applied (the first step only records it) */
  private noiseEpoch = -1;
  /** Stochastic boundary-layer perturbations at the start of each new noise interval. */
  private noise(m: RegionalModel): void {
    const e = Math.floor(m.time / BL_NOISE_PERIOD + 1e-6);
    if (this.noiseEpoch < 0 || e <= this.noiseEpoch) { this.noiseEpoch = Math.max(this.noiseEpoch, e); return; }
    this.noiseEpoch = e;
    const { nx, ny, nz } = m.c, a = this.cfg.blNoise!;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const d = a * blNoiseValue(i, j, nx, e);
      for (let k = 0; k < nz && m.zc[k]! < BL_NOISE_DEPTH; k++) { const q = m.idx(i, j, k); m.th[q] = m.th[q]! + d; }
    }
  }

  /** sub-grid turbulence + surface-flux tendencies of the first RK stage, reused in stages 2 and 3 (as in WRF);
   *  allocated on the first CPU step (a model that only mirrors a GPU run never needs it) */
  private cache: { fu: Float64Array; fv: Float64Array; fw: Float64Array; fth: Float64Array; fsc: Float64Array[] } | null = null;

  private tendencies(m: RegionalModel, out: { fu: Float64Array; fv: Float64Array; fw: Float64Array; fth: Float64Array; fsc: Float64Array[] }, stage: number): void {
    if (!this.cache) {
      const z = (): Float64Array => new Float64Array(m.size);
      this.cache = { fu: z(), fv: z(), fw: z(), fth: z(), fsc: m.scalars.map(z) };
      this.Km = z();
    }
    const t = this.cache;
    if (stage === 0) {
      for (const a of [t.fu, t.fv, t.fw, t.fth, ...t.fsc]) a.fill(0);
      this.turbulenceAndSurface(m, t);
    }
    const add = (o: Float64Array, a: Float64Array): void => { for (let i = 0; i < o.length; i++) o[i] = o[i]! + a[i]!; };
    add(out.fu, t.fu); add(out.fv, t.fv); add(out.fw, t.fw); add(out.fth, t.fth);
    for (let s = 0; s < out.fsc.length; s++) add(out.fsc[s]!, t.fsc[s]!);
    this.radiation(m, out);
  }

  private turbulenceAndSurface(m: RegionalModel, t: { fu: Float64Array; fv: Float64Array; fw: Float64Array; fth: Float64Array; fsc: Float64Array[] }): void {
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
        const qsS = surfaceQs(tsk, sf.psfc);
        const thS = tsk / sf.pis;
        // ground-relative wind (the model frame may translate with a storm at frameVel)
        const ua = 0.5 * (u[q]! + u[q + 1]!) + (c.frameVel?.u ?? 0), va = 0.5 * (v[q]! + v[q + sx]!) + (c.frameVel?.v ?? 0);
        const spd = c.gust
          ? gustSpeed(Math.hypot(ua, va), c.vmin ?? 1, c.ck, thS - th[q]!, qv ? (qsS - qv[q]!) * sf.wet[c2]! : 0, th[q]!, m.scalars[QR]?.[q] ?? 0, rho1)
          : Math.max(Math.hypot(ua, va), c.vmin ?? 1);
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

  }

  /** Newtonian radiative relaxation, capped cooling (every stage) */
  private radiation(m: RegionalModel, t: { fth: Float64Array }): void {
    const { nx, ny, nz } = m.c, c = this.cfg, th = m.th, rc = c.radConst ?? 0;
    if (c.radTau > 0 || rc > 0) {
      for (let k = 0; k < nz; k++) {
        const pi = m.pi0[k]!, trop = rc > 0 && m.th0[k]! * pi > 210;
        if (!trop && !(c.radTau > 0)) continue;
        for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
          const q = m.idx(i, j, k);
          t.fth[q] = t.fth[q]! + (trop ? -rc / pi : Math.max(-(th[q]! - m.th0[k]!) / c.radTau, -c.radMax / pi));
        }
      }
    }
  }
}
