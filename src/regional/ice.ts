// Single-moment six-class bulk microphysics with ice: qv, qc, qr, qi (cloud ice), qs (snow),
// qg (graupel), in RegionalModel.scalars slots 0..5.
//
// Warm-rain processes are exactly those of the Kessler scheme (kessler.ts: autoconversion, accretion,
// saturation adjustment over water, KW78 rain evaporation and fall speed), so above 0 °C the scheme
// reduces to Kessler. The ice processes follow Lin, Farley & Orville (1983, JCAM) and Rutledge & Hobbs
// (1983, JAS) with the cloud-ice relations of Hong, Dudhia & Chen (2004, MWR):
//   * precipitating ice (snow, graupel) and rain have inverse-exponential size distributions
//     N(D) = n0 exp(-lambda D) and fall speeds V = a D^b (rho_0/rho)^1/2 (snow n0 increasing as the
//     temperature falls, Houze et al. 1979 as used in HDC04)
//   * ice nucleation (Fletcher-type, capped), depositional growth / sublimation of cloud ice, snow and
//     graupel with ventilation, driven by the supersaturation over ice (this drives the
//     Wegener–Bergeron–Findeisen process: cloud water evaporates while ice grows)
//   * collection: snow and graupel rime cloud water, collect cloud ice; rain collects cloud ice
//   * autoconversion ice -> snow and snow -> graupel (Lin 1983 forms); Bergeron-process snow production from
//     cloud ice and cloud water in mixed-phase cloud (Lin 1983 Psfi, Psfw)
//   * Bigg (1953) freezing of rain to graupel; homogeneous freezing below -40 °C; melting of snow and
//     graupel by conduction and ventilation (Rutledge & Hobbs 1983); instant melting of cloud ice above 0 °C
//   * mass-weighted sedimentation of rain, snow, graupel and cloud ice (flux form, sub-stepped)
// Latent heating uses Lv (vapour/liquid), Ls (vapour/ice) and Lf = Ls - Lv (liquid/ice). All transfers
// are between species, limited so that no species goes negative: total water is conserved exactly.

import { DRY_AIR, T_FLOOR, VT_MAX } from '../core/constants.js';
import type { RegionalModel } from './core.js';

/** The model members the microphysics uses (the 3-D regional model and the axisymmetric model both have them). */
export type MicroHost = Pick<RegionalModel, 'c' | 'idx' | 'th' | 'pp' | 'scalars' | 'rho0' | 'pi0'>;

export const QV = 0, QC = 1, QR = 2, QI = 3, QS = 4, QG = 5;

export const ICE = {
  T0: 273.15, LV: 2.5e6, LS: 2.834e6, RV: 461.5, RHOW: 1000,
  N0R: 8e6, AR: 841.99667, BR: 0.8,                 // rain (collection kernels)
  RHOS: 100, AS: 11.72, BS: 0.41,                   // snow
  N0G: 4e6, RHOG: 500, AG: 330, BG: 0.8,            // graupel
  KA: 2.4e-2, MU: 1.718e-5,                         // thermal conductivity of air, dynamic viscosity
  MI0: 1e-12,                                       // mass of a newly nucleated ice crystal (kg)
  QI0: 1e-3, QS0: 6e-4,                             // autoconversion thresholds ice -> snow, snow -> graupel (Lin et al. 1983)
  BIGG_B: 100, BIGG_A: 0.66,                        // Bigg (1953) freezing: B' (m^-3 s^-1), A' (K^-1)
} as const;
export const LF = ICE.LS - ICE.LV;

/** Koenig (1971) depositional growth of ice crystals, dm/dt = a1 m^a2 (cgs), at -1 ... -31 °C (as tabulated for
 *  Hsie et al. 1980 and Lin et al. 1983); used by the Bergeron-process snow production below. */
export const KOENIG_A1 = [0.7939e-7, 0.7841e-6, 0.3369e-5, 0.4336e-5, 0.5285e-5, 0.3728e-5, 0.1852e-5, 0.2991e-6, 0.4248e-6, 0.7434e-6, 0.1812e-5, 0.4394e-5, 0.9145e-5,
  0.1725e-4, 0.3348e-4, 0.1725e-4, 0.9175e-5, 0.4412e-5, 0.2252e-5, 0.9115e-6, 0.4876e-6, 0.3473e-6, 0.4758e-6, 0.6306e-6, 0.8573e-6, 0.7868e-6, 0.7192e-6, 0.6513e-6, 0.5956e-6, 0.5333e-6, 0.4834e-6];
export const KOENIG_A2 = [0.4006, 0.4831, 0.5320, 0.5307, 0.5319, 0.5249, 0.4888, 0.3894, 0.4047, 0.4318, 0.4771, 0.5183, 0.5463, 0.5651, 0.5813, 0.5655, 0.5478, 0.5203, 0.4906, 0.4447,
  0.4126, 0.3960, 0.4149, 0.4320, 0.4506, 0.4483, 0.4460, 0.4433, 0.4413, 0.4382, 0.4361];

/**
 * Bergeron-process snow production (Lin et al. 1983, after Hsie et al. 1980) in mixed-phase cloud (cloud water present,
 * -31 °C < T < 0 °C): cloud-ice crystals growing by deposition from 40 to 50 um radius in dt1 seconds become snow
 * (psfi = qi / dt1), and those crystals collect cloud water on the way (psfw). kg/kg/s. Without liquid water (anvils,
 * cirrus) cloud ice only becomes snow by autoconversion above QI0 and by collection.
 */
export function bergeron(T: number, qi: number, qc: number, rho: number, dt: number): { psfi: number; psfw: number } {
  if (!(T < ICE.T0 && T > ICE.T0 - 31) || qc <= 1e-8 || qi <= 1e-12) return { psfi: 0, psfw: 0 };
  const it = Math.min(30, Math.max(0, Math.round(ICE.T0 - T) - 1)), a1 = KOENIG_A1[it]!, a2 = KOENIG_A2[it]!;
  const mi40 = 2.46e-7, mi50 = 4.8e-7;                                                     // crystal masses (g)
  const dt1 = (Math.pow(mi50, 1 - a2) - Math.pow(mi40, 1 - a2)) / (a1 * (1 - a2));        // growth time (s)
  const ni50 = qi / 4.8e-10 * Math.min(1, dt / dt1);                                       // 50 um crystals per kg of air
  return { psfi: qi / dt1, psfw: ni50 * (a1 * Math.pow(mi50, a2) * 1e-3 + Math.PI * rho * qc * 2.5e-9 * 1.0) };
}

/** Lanczos approximation of the gamma function (x > 0). */
export function gammaFn(x: number): number {
  const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.PI / (Math.sin(Math.PI * x) * gammaFn(1 - x));
  x -= 1;
  let a = c[0]!;
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i]! / (x + i);
  return Math.sqrt(2 * Math.PI) * Math.pow(t, x + 0.5) * Math.exp(-t) * a;
}

const G3R = gammaFn(3 + ICE.BR), G3S = gammaFn(3 + ICE.BS), G3G = gammaFn(3 + ICE.BG);
const G4S = gammaFn(4 + ICE.BS), G4G = gammaFn(4 + ICE.BG);
const G5S = gammaFn((ICE.BS + 5) / 2), G5G = gammaFn((ICE.BG + 5) / 2);

/** Saturation mixing ratios over water (the Kessler / KW78 form) and over ice (Tetens ice constants). */
export function qvsWater(T: number, p: number): number { return 380 / p * Math.exp(17.27 * (T - 273.15) / (T - 35.86)); }
export function qvsIce(T: number, p: number): number { return 380 / p * Math.exp(21.875 * (T - 273.15) / (T - 7.66)); }

/** Diagnostics of one microphysics step (per column sums, kg m^-2). */
export class IceMicrophysics {
  /** accumulated surface precipitation (kg m^-2 = mm water) per column [j][i]: total, and the frozen part */
  readonly rainAcc: Float64Array;
  readonly snowAcc: Float64Array;
  clipped = 0;
  /** the saturation switches (the shared microOpts: changed between steps by the worker) */
  opts: MicroOpts = microOpts;

  constructor(private readonly m: MicroHost) {
    if (m.scalars.length < 6) throw new Error('IceMicrophysics needs 6 scalars (qv, qc, qr, qi, qs, qg)');
    this.rainAcc = new Float64Array(m.c.nx * m.c.ny);
    this.snowAcc = new Float64Array(m.c.nx * m.c.ny);
  }

  apply(dt: number, opts: { sediment?: boolean } = {}): void {
    const m = this.m, { nx, ny, nz, dz } = m.c, cp = DRY_AIR.cp, rd = DRY_AIR.rd;
    const S = m.scalars, rho = m.rho0, pi0 = m.pi0, rhoSfc = rho[0]!;
    const col = new Float64Array(nz), vt = new Float64Array(nz), flux = new Float64Array(nz + 1);
    const sediment = opts.sediment ?? true;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const c2 = j * nx + i;
      for (let k = 0; k < nz; k++) {
        const q = m.idx(i, j, k);
        for (let s = 0; s < 6; s++) { const a = S[s]!; if (a[q]! < 0) { this.clipped -= a[q]! * rho[k]! * dz; a[q] = 0; } }
      }
      // ---------------- sedimentation (mass-weighted fall speeds)
      if (sediment) for (const sp of [QR, QS, QG, QI]) {
        const a = S[sp]!;
        let vmax = 0;
        for (let k = 0; k < nz; k++) {
          const q = m.idx(i, j, k), qq = a[q]!;
          col[k] = qq;
          vt[k] = Math.min(fallSpeed(sp, qq, rho[k]!, rhoSfc, m.th[q]! * (pi0[k]! + m.pp[q]!)), VT_MAX);
          vmax = Math.max(vmax, vt[k]!);
        }
        if (vmax === 0) continue;
        const nsub = Math.max(1, Math.ceil(vmax * dt / (0.8 * dz))), dts = dt / nsub;
        for (let s = 0; s < nsub; s++) {
          for (let k = 0; k < nz; k++) flux[k] = rho[k]! * vt[k]! * col[k]!;
          flux[nz] = 0;
          this.rainAcc[c2] = this.rainAcc[c2]! + flux[0]! * dts;
          if (sp !== QR) this.snowAcc[c2] = this.snowAcc[c2]! + flux[0]! * dts;
          for (let k = 0; k < nz; k++) col[k] = col[k]! + dts * (flux[k + 1]! - flux[k]!) / (rho[k]! * dz);
        }
        for (let k = 0; k < nz; k++) a[m.idx(i, j, k)] = Math.max(0, col[k]!);
      }
      // ---------------- local processes
      for (let k = 0; k < nz; k++) {
        const q = m.idx(i, j, k);
        const pi = Math.max(pi0[k]! + m.pp[q]!, 0.05);
        const st = { th: m.th[q]!, qv: S[QV]![q]!, qc: S[QC]![q]!, qr: S[QR]![q]!, qi: S[QI]![q]!, qs: S[QS]![q]!, qg: S[QG]![q]! };
        cellProcesses(st, rho[k]!, rhoSfc, pi, DRY_AIR.pRef * Math.pow(pi, cp / rd), dt, this.opts);
        m.th[q] = st.th; S[QV]![q] = st.qv; S[QC]![q] = st.qc; S[QR]![q] = st.qr; S[QI]![q] = st.qi; S[QS]![q] = st.qs; S[QG]![q] = st.qg;
      }
    }
  }
}

/** Mass-weighted terminal velocity (m/s) of species sp at mixing ratio q. */
export function fallSpeed(sp: number, q: number, rho: number, rhoSfc: number, T: number): number {
  if (q <= 1e-12) return 0;
  const rq = rho * q, dens = Math.sqrt(rhoSfc / rho);
  if (sp === QR) return 36.34 * Math.pow(0.001 * rq, 0.1364) * dens;                 // KW78, as in Kessler
  if (sp === QS) { const lam = lambda(ICE.RHOS, n0Snow(T), rq); return ICE.AS * G4S / (6 * Math.pow(lam, ICE.BS)) * dens; }
  if (sp === QG) { const lam = lambda(ICE.RHOG, ICE.N0G, rq); return ICE.AG * G4G / (6 * Math.pow(lam, ICE.BG)) * dens; }
  // cloud ice (HDC04): N_i = 5.38e7 (rho q_i)^0.75, D_i = 11.9 sqrt(m_i), V = 1.49e4 D_i^1.31
  const ni = iceNumber(rq), di = 11.9 * Math.sqrt(rq / ni);
  return Math.min(1.49e4 * Math.pow(di, 1.31), 3);
}

/** Cloud-ice number concentration (m^-3) from the ice content (HDC04), bounded to [1e3, 1e6] as in WSM6. */
function iceNumber(rqi: number): number { return Math.min(1e6, Math.max(1e3, 5.38e7 * Math.pow(rqi, 0.75))); }

/** Slope of the exponential size distribution: rho q = pi rho_x n0 / lambda^4. */
function lambda(rhoX: number, n0: number, rq: number): number { return Math.pow(Math.PI * rhoX * n0 / Math.max(rq, 1e-15), 0.25); }
/** Snow intercept parameter increasing at lower temperatures (HDC04), capped. */
function n0Snow(T: number): number { return Math.min(2e8, 2e6 * Math.exp(0.12 * Math.max(0, ICE.T0 - T))); }

interface CellState { th: number; qv: number; qc: number; qr: number; qi: number; qs: number; qg: number }

/** Switches of the saturation handling, changed from the page while the model runs (worker.ts 'micro' message; the GPU kernel reads the same two numbers). */
export interface MicroOpts {
  /** Ice supersaturation. On: below -40 °C vapour is brought back only to the saturation ratio at which solution droplets freeze by themselves (iceSCrit),
   *  so the air can hold ice supersaturation as real cirrus does; off: it is brought to ice saturation at once. (Between -40 and 0 °C the ice grows at a finite rate either way.) */
  iceSS: boolean;
  /** Liquid-water supersaturation: the time (s) in which vapour above water saturation condenses (an exponential approach, stable for any step). 0: at once (saturation
   *  adjustment, as always before; the air never holds more than saturation). Evaporation of cloud water stays instant. */
  liqTau: number;
}
export const MICRO_DEFAULT: Readonly<MicroOpts> = { iceSS: true, liqTau: 0 };
/** The switches of this page's models (the worker changes them; every IceMicrophysics and GPU model reads them). */
export const microOpts: MicroOpts = { ...MICRO_DEFAULT };
/** Saturation ratio over ice at which solution droplets freeze homogeneously, 2.349 - T/259 (Koop et al. 2000 as fitted by Kärcher & Lohmann 2002): 1.45 at -40 °C, 1.6 at -75 °C. */
export const iceSCrit = (T: number): number => Math.max(1, 2.349 - T / 259);

/**
 * All local microphysical processes of one grid cell over dt (process-split: phase changes forced by
 * temperature, then explicit ice-process rates with sink limiting, then the Kessler warm-rain step).
 */
export function cellProcesses(s: CellState, rho: number, rhoSfc: number, pi: number, p: number, dt: number, opt: Readonly<MicroOpts> = MICRO_DEFAULT): void {
  const cp = DRY_AIR.cp, T0 = ICE.T0, hv = ICE.LV / (cp * pi), hs = ICE.LS / (cp * pi), hf = LF / (cp * pi);
  let T = Math.max(T_FLOOR, s.th * pi);
  // ---- temperature-forced phase changes
  if (T > T0 && s.qi > 0) { s.qc += s.qi; s.th -= hf * s.qi; s.qi = 0; T = Math.max(T_FLOOR, s.th * pi); }                   // instant melting of cloud ice
  if (T < T0 - 40) {                                                                                  // homogeneous freezing
    if (s.qc > 0) { s.qi += s.qc; s.th += hf * s.qc; s.qc = 0; }
    if (s.qr > 0) { s.qg += s.qr; s.th += hf * s.qr; s.qr = 0; }
    T = Math.max(T_FLOOR, s.th * pi);
  }
  // ---- ice-process rates (kg/kg/s)
  const dens = Math.sqrt(rhoSfc / rho), dens4 = Math.sqrt(dens);
  const psi = 2.26e-5 * Math.pow(T / T0, 1.81) * (1e5 / p);                // water-vapour diffusivity (m^2/s)
  const nu = ICE.MU / rho, sc3 = Math.cbrt(nu / psi);
  const qsi = qvsIce(T, p), si = s.qv / qsi - 1;
  const Ai = (ICE.LS / (ICE.RV * T) - 1) * ICE.LS / (ICE.KA * T), Bi = 1 / (psi * rho * qsi);
  const rqs = rho * s.qs, rqg = rho * s.qg, rqr = rho * s.qr;
  const n0s = n0Snow(T);
  const lamS = s.qs > 1e-12 ? lambda(ICE.RHOS, n0s, rqs) : 0, lamG = s.qg > 1e-12 ? lambda(ICE.RHOG, ICE.N0G, rqg) : 0, lamR = s.qr > 1e-12 ? lambda(ICE.RHOW, ICE.N0R, rqr) : 0;
  // collection kernels (pi/4) n0 a Gamma(3+b) / lambda^(3+b) (rho0/rho)^1/2, per unit of the collected mixing ratio
  const kS = lamS > 0 ? Math.PI / 4 * n0s * ICE.AS * G3S / Math.pow(lamS, 3 + ICE.BS) * dens : 0;
  const kG = lamG > 0 ? Math.PI / 4 * ICE.N0G * ICE.AG * G3G / Math.pow(lamG, 3 + ICE.BG) * dens : 0;
  const kR = lamR > 0 ? Math.PI / 4 * ICE.N0R * ICE.AR * G3R / Math.pow(lamR, 3 + ICE.BR) * dens : 0;
  // ventilated vapour / heat exchange integrals: 0.78/lambda^2 + 0.31 Sc^1/3 (a/nu)^1/2 Gamma((b+5)/2) lambda^-(b+5)/2 (rho0/rho)^1/4
  const vS = lamS > 0 ? 0.78 / (lamS * lamS) + 0.31 * sc3 * Math.sqrt(ICE.AS / nu) * G5S * dens4 / Math.pow(lamS, (ICE.BS + 5) / 2) : 0;
  const vG = lamG > 0 ? 0.78 / (lamG * lamG) + 0.31 * sc3 * Math.sqrt(ICE.AG / nu) * G5G * dens4 / Math.pow(lamG, (ICE.BG + 5) / 2) : 0;

  let pidep = 0, pigen = 0, psdep = 0, pgdep = 0, psaut = 0, pgaut = 0, psaci = 0, pgaci = 0, praci = 0;
  let psacw = 0, pgacw = 0, pgfrz = 0, psmlt = 0, pgmlt = 0, psfi = 0, psfw = 0;
  if (T < T0) {
    const eci = Math.exp(0.05 * (T - T0));
    // nucleation: N = 1e3 exp(0.1 (T0 - T)) m^-3 (HDC04-type, capped at 1e6), crystals of mass MI0
    if (si > 0) {
      const nNuc = Math.min(1e6, 1e3 * Math.exp(0.1 * (T0 - T)));
      pigen = Math.max(0, Math.min(ICE.MI0 * nNuc / rho - s.qi, (s.qv - qsi) / (1 + ICE.LS * ICE.LS * qsi / (cp * ICE.RV * T * T)))) / dt;
    }
    // cloud-ice deposition / sublimation: 4 D_i N_i (S_i - 1) / (A + B) per volume (HDC04)
    if (s.qi > 1e-12) {
      const rqi = rho * s.qi, ni = iceNumber(rqi), di = 11.9 * Math.sqrt(rqi / ni);
      pidep = 4 * di * ni * si / (rho * (Ai + Bi));
      psaut = Math.max(0, 1e-3 * Math.exp(0.025 * (T - T0)) * (s.qi - ICE.QI0));                      // Lin 1983
      const b = bergeron(T, s.qi, s.qc, rho, dt); psfi = b.psfi; psfw = b.psfw;
      psaci = kS * eci * s.qi;
      pgaci = kG * eci * s.qi;
      praci = kR * s.qi;
    }
    // snow and graupel deposition / sublimation (plates: 4 D; spheres: 2 pi D), ventilated
    if (lamS > 0) psdep = 4 * n0s * si * vS / (rho * (Ai + Bi));
    if (lamG > 0) pgdep = 2 * Math.PI * ICE.N0G * si * vG / (rho * (Ai + Bi));
    if (s.qs > ICE.QS0) pgaut = 1e-3 * Math.exp(0.09 * (T - T0)) * (s.qs - ICE.QS0);                  // Lin 1983
    // riming of cloud water
    psacw = kS * s.qc;
    pgacw = kG * s.qc;
    // Bigg freezing of rain -> graupel: 20 pi^2 B' n0r (rho_w/rho) [exp(A'(T0-T)) - 1] / lambda_r^7
    if (lamR > 0) pgfrz = 20 * Math.PI * Math.PI * ICE.BIGG_B * ICE.N0R * (ICE.RHOW / rho) * (Math.exp(ICE.BIGG_A * (T0 - T)) - 1) / Math.pow(lamR, 7);
  } else {
    // melting by conduction with ventilation: 2 pi n0 Ka (T - T0) / (rho Lf) x ventilation integral
    if (lamS > 0) psmlt = 2 * Math.PI * n0s * ICE.KA * (T - T0) * vS / (rho * LF);
    if (lamG > 0) pgmlt = 2 * Math.PI * ICE.N0G * ICE.KA * (T - T0) * vG / (rho * LF);
    // snow / graupel collecting cloud water above 0 °C shed it as rain
    psacw = kS * s.qc; pgacw = kG * s.qc;
  }
  // ---- sink limiting: scale each species' outflow so that it cannot go negative over dt
  const dep = (x: number): number => Math.max(x, 0), sub = (x: number): number => Math.max(-x, 0);
  const vSink = dep(pidep) + pigen + dep(psdep) + dep(pgdep);
  const fv = vSink * dt > s.qv ? s.qv / (vSink * dt) : 1;
  const iSink = sub(pidep) + psaut + psfi + psaci + pgaci + praci;
  const fi = iSink * dt > s.qi ? s.qi / (iSink * dt) : 1;
  const sSink = sub(psdep) + pgaut + psmlt;
  const fs = sSink * dt > s.qs ? s.qs / (sSink * dt) : 1;
  const gSink = sub(pgdep) + pgmlt;
  const fg = gSink * dt > s.qg ? s.qg / (gSink * dt) : 1;
  const cSink = psacw + psfw + pgacw;
  const fc = cSink * dt > s.qc ? s.qc / (cSink * dt) : 1;
  const rSink = pgfrz;
  const fr = rSink * dt > s.qr ? s.qr / (rSink * dt) : 1;
  // transfers over dt (kg/kg)
  const cold = T < T0;
  const t_vi = (dep(pidep) + pigen) * fv * dt, t_iv = sub(pidep) * fi * dt;
  const t_vs = dep(psdep) * fv * dt, t_sv = sub(psdep) * fs * dt;
  const t_vg = dep(pgdep) * fv * dt, t_gv = sub(pgdep) * fg * dt;
  const t_is = (psaut + psfi + psaci) * fi * dt, t_ig = (pgaci + praci) * fi * dt;
  const t_sg = pgaut * fs * dt, t_sr = psmlt * fs * dt, t_gr = pgmlt * fg * dt;
  const t_cs = (psacw + (cold ? psfw : 0)) * fc * dt, t_cg = pgacw * fc * dt, t_rg = pgfrz * fr * dt;
  s.qv += -t_vi + t_iv - t_vs + t_sv - t_vg + t_gv;
  s.qi += t_vi - t_iv - t_is - t_ig;
  s.qs += t_vs - t_sv + t_is - t_sg - t_sr + (cold ? t_cs : 0);
  s.qg += t_vg - t_gv + t_ig + t_sg - t_gr + (cold ? t_cg : 0) + t_rg;
  s.qc -= t_cs + t_cg;
  s.qr += t_sr + t_gr - t_rg + (cold ? 0 : t_cs + t_cg);
  s.th += hs * (t_vi - t_iv + t_vs - t_sv + t_vg - t_gv) + hf * ((cold ? t_cs + t_cg : 0) + t_rg - t_sr - t_gr);
  s.qv = Math.max(s.qv, 0); s.qi = Math.max(s.qi, 0); s.qs = Math.max(s.qs, 0); s.qg = Math.max(s.qg, 0); s.qc = Math.max(s.qc, 0); s.qr = Math.max(s.qr, 0);
  // ---- warm rain and saturation adjustment over water (identical to the Kessler scheme)
  T = Math.max(T_FLOOR, s.th * pi);
  let vv = s.qv, cc = s.qc, rr = s.qr;
  const factorn = 1 / (1 + 2.2 * dt * Math.pow(Math.max(0, rr), 0.875));
  const qrprod = cc - (cc - dt * Math.max(0.001 * (cc - 0.001), 0)) * factorn;
  cc = Math.max(cc - qrprod, 0);
  rr = rr + qrprod;
  const qvs = qvsWater(T, p);
  const f5 = 237.3 * 17.27 * ICE.LV / cp;
  let prod = (vv - qvs) / (1 + qvs * f5 / (T - 35.86) ** 2);
  if (prod > 0 && opt.liqTau > 0) prod *= -Math.expm1(-dt / opt.liqTau);      // liquid supersaturation: condensation takes liqTau
  if (T < T0 - 40) prod = Math.min(prod, 0);                                  // no liquid condensation below -40 °C
  const rq = Math.max(rho * rr, 0);
  const ern = Math.min(dt * (((1.6 + 124.9 * Math.pow(rq, 0.2046)) * Math.pow(rq, 0.525)) / (2.55e8 / (p * qvs) + 5.4e5)) * (Math.max(qvs - vv, 0) / (rho * qvs)),
    Math.max(-prod - cc, 0), rr);
  const product = Math.max(prod, -cc);
  s.th += hv * (product - ern);
  vv = Math.max(vv - product + ern, 0);
  s.qv = vv; s.qc = cc + product; s.qr = Math.max(rr - ern, 0);
  // below -40 °C vapour in excess of ice saturation (of the freezing threshold, with ice supersaturation on) deposits directly (fast adjustment)
  if (T < T0 - 40) {
    const Tn = Math.max(T_FLOOR, s.th * pi), qsi2 = qvsIce(Tn, p) * (opt.iceSS ? iceSCrit(Tn) : 1);
    if (s.qv > qsi2) {
      const d = (s.qv - qsi2) / (1 + ICE.LS * ICE.LS * qsi2 / (cp * ICE.RV * Tn * Tn));
      s.qv -= d; s.qi += d; s.th += hs * d;
    }
  }
}
