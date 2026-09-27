// Radiation with diagnostic clouds for the global column physics (radiation: 'cloudy').
//
// Longwave: the Byrne & O'Gorman (2013) gray water-vapour optical depth of the 'byrne' scheme, with
//   cloud layers acting as gray absorbers/emitters: layer transmissivity x (1 - f eps_c).
// Shortwave: water-vapour absorption with the Lacis & Hansen (1974) absorptivity
//     A(y) = 2.9 y / ((1 + 141.5 y)^0.635 + 5.925 y),  y = pressure-scaled water path (cm) x M,
//   M = 35 / sqrt(1224 mu0^2 + 1) (magnification for mean zenith cosine mu0), and conservatively
//   scattering cloud layers of reflectance R = (1-g) tau / (2 + (1-g) tau) (two-stream, g = 0.85),
//   combined with the surface albedo by the adding method.
// Clouds (diagnostic):
//   * stratiform, per layer: f = ((RH - RHc) / (1 - RHc))^2 for RH > RHc (Slingo 1987), below 100 hPa
//   * convective: fraction Cc = 0.245 + 0.125 ln(Pconv [mm/day]) (Slingo 1987), capped at 0.8, from the
//     previous step's convective precipitation, as one effective layer at the convective top carrying the
//     condensate path of the whole tower
//   * in-cloud condensate density rho_l(z) = rho_l0 exp(-z / h_l), h_l = 700 m ln(1 + PW [cm])
//     (CCM2, Hack 1993); liquid above 263 K, ice below 243 K, linear in between
//   * optical depth tau = 1.5 CWP / (rho_w r_e) (r_e = 10 um liquid, 30 um ice); longwave emissivity
//     1 - exp(-k CWP) with k = 0.090361 m^2/g (liquid) and 0.005 + 1/r_e[um] m^2/g (ice) as in CCM3

import { MOIST, qsat } from './thermo.js';

export const CLOUD = { RHC: 0.8, RHOL0: 0.21e-3 /* kg m^-3 */, G: 0.85, RE_L: 10e-6, RE_I: 30e-6, KL: 0.090361, MU0: 0.5 };

export interface CloudRadIn {
  K: number; T: Float64Array; q: Float64Array; pf: Float64Array; ph: Float64Array; zh: Float64Array; zf: Float64Array;
  ts: number; albedoSfc: number; insol: number; lwTau: Float64Array; convRate: number; convTop: number; g: number;
}
export interface CloudRadOut {
  heat: Float64Array;          // K/s per layer (SW + LW)
  swSfcNet: number;            // absorbed SW at the surface (W m^-2)
  lwDownSfc: number;           // W m^-2
  olr: number; swUpTop: number; cloudCover: number; lwp: number;
}

const lacisHansen = (y: number): number => 2.9 * y / (Math.pow(1 + 141.5 * y, 0.635) + 5.925 * y);

/** Work arrays for one column (reuse between calls). */
export function cloudWork(K: number): Record<string, Float64Array> {
  const c = (n: number): Float64Array => new Float64Array(n);
  return { cf: c(K), cwp: c(K), fice: c(K), R: c(K), Tr: c(K), Rb: c(K + 1), dn: c(K + 1), up: c(K + 1), trl: c(K), lwu: c(K + 1), lwd: c(K + 1), heat: c(K), Y: c(K + 1) };
}

export function cloudRadiation(inp: CloudRadIn, w: Record<string, Float64Array>, stefan = MOIST.stefan, cp = 1004.64): CloudRadOut {
  const { K, T, q, pf, ph, zf, g } = inp;
  const cf = w.cf!, cwp = w.cwp!, fice = w.fice!, R = w.R!, Tr = w.Tr!, Rb = w.Rb!, dn = w.dn!, up = w.up!, trl = w.trl!, lwu = w.lwu!, lwd = w.lwd!, heat = w.heat!, Y = w.Y!;
  // precipitable water (cm) and condensate scale height
  let pw = 0;
  for (let k = 0; k < K; k++) pw += q[k]! * (ph[k + 1]! - ph[k]!) / g;
  const hl = 700 * Math.log(1 + pw / 10);
  // ---- stratiform clouds
  for (let k = 0; k < K; k++) {
    const rh = q[k]! / qsat(T[k]!, pf[k]!);
    const f = pf[k]! > 1e4 && rh > CLOUD.RHC ? Math.min(1, ((rh - CLOUD.RHC) / (1 - CLOUD.RHC)) ** 2) : 0;
    const dz = (ph[k + 1]! - ph[k]!) / g / (pf[k]! / (287.04 * T[k]!));
    cf[k] = f;
    cwp[k] = CLOUD.RHOL0 * Math.exp(-zf[k]! / Math.max(hl, 1)) * dz * 1e3;        // in-cloud path, g m^-2
    fice[k] = Math.max(0, Math.min(1, (263 - T[k]!) / 20));
  }
  // ---- convective cloud: one effective layer at the convective top
  const pc = inp.convRate * 86400;
  const cc = pc > 0.14 && inp.convTop >= 0 ? Math.min(0.8, 0.245 + 0.125 * Math.log(pc)) : 0;
  let convK = -1, convCwp = 0;
  if (cc > 0) {
    convK = inp.convTop;
    for (let k = inp.convTop; k < K; k++) convCwp += cwp[k]!;
  }
  // ---- layer shortwave reflectance / transmittance, longwave cloud emissivity
  const tauOf = (path: number, fi: number): number => 1.5 * path * 1e-3 / (1000 * ((1 - fi) * CLOUD.RE_L + fi * CLOUD.RE_I));
  const epsOf = (path: number, fi: number): number => 1 - Math.exp(-((1 - fi) * CLOUD.KL + fi * (0.005 + 1 / (CLOUD.RE_I * 1e6))) * path);
  const rOf = (tau: number): number => { const x = (1 - CLOUD.G) * tau; return x / (2 + x); };
  // water-vapour absorption: cumulative scaled path (cm) at the interfaces
  const M = 35 / Math.sqrt(1224 * CLOUD.MU0 * CLOUD.MU0 + 1);
  Y[0] = 0;
  for (let k = 0; k < K; k++) Y[k + 1] = Y[k]! + q[k]! * (ph[k + 1]! - ph[k]!) / g * (pf[k]! / 1e5) * 0.1 * M;
  let clear = 1;
  for (let k = 0; k < K; k++) {
    let rc = cf[k]! * rOf(tauOf(cwp[k]!, fice[k]!));
    let ec = cf[k]! * epsOf(cwp[k]!, fice[k]!);
    let fk = cf[k]!;
    if (k === convK) {
      const r2 = cc * rOf(tauOf(convCwp, fice[k]!)), e2 = cc * epsOf(convCwp, fice[k]!);
      rc = 1 - (1 - rc) * (1 - r2); ec = 1 - (1 - ec) * (1 - e2); fk = 1 - (1 - fk) * (1 - cc);
    }
    clear *= 1 - fk;
    const twv = (1 - lacisHansen(Y[k + 1]!)) / (1 - lacisHansen(Y[k]!));
    R[k] = rc;
    Tr[k] = (1 - rc) * twv;
    trl[k] = inp.lwTau[k]! * (1 - ec);
  }
  // ---- shortwave adding: reflectance of everything below interface i
  Rb[K] = inp.albedoSfc;
  for (let k = K - 1; k >= 0; k--) Rb[k] = R[k]! + Tr[k]! * Tr[k]! * Rb[k + 1]! / (1 - R[k]! * Rb[k + 1]!);
  dn[0] = inp.insol; up[0] = inp.insol * Rb[0]!;
  for (let k = 0; k < K; k++) { dn[k + 1] = dn[k]! * Tr[k]! / (1 - R[k]! * Rb[k + 1]!); up[k + 1] = dn[k + 1]! * Rb[k + 1]!; }
  // ---- longwave (gray two-stream with cloud-modified transmissivities)
  lwd[0] = 0;
  for (let k = 0; k < K; k++) { const B = stefan * T[k]! ** 4; lwd[k + 1] = lwd[k]! * trl[k]! + B * (1 - trl[k]!); }
  lwu[K] = stefan * inp.ts ** 4;
  for (let k = K - 1; k >= 0; k--) { const B = stefan * T[k]! ** 4; lwu[k] = lwu[k + 1]! * trl[k]! + B * (1 - trl[k]!); }
  let lwpCol = 0;
  for (let k = 0; k < K; k++) {
    const netTop = (dn[k]! - up[k]!) - (lwu[k]! - lwd[k]!), netBot = (dn[k + 1]! - up[k + 1]!) - (lwu[k + 1]! - lwd[k + 1]!);
    heat[k] = g * (netTop - netBot) / (cp * (ph[k + 1]! - ph[k]!));
    lwpCol += cf[k]! * cwp[k]!;
  }
  return { heat, swSfcNet: dn[K]! - up[K]!, lwDownSfc: lwd[K]!, olr: lwu[0]!, swUpTop: up[0]!, cloudCover: 1 - clear, lwp: lwpCol + cc * convCwp };
}
