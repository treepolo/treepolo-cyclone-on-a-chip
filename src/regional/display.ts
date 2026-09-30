// Optical quantities for pictures of the regional model (3-D view, visible satellite image): measurement only, the
// model's equations never see them.
//
// Extinction (1/m) from the condensate with the geometric-optics estimate beta = 1.5 rho q / (rho_p r_e): cloud water
// (r_e 10 um) 150 m^2/kg, cloud ice (30 um) 55, snow (60 um, standing in for most of the anvil ice of this scheme) 27,
// graupel (500 um, 400 kg/m^3) 7.5, rain (500 um) 3.
//
// Sub-grid cloud (Smith 1990, triangular distribution of total water within the grid box): a grid box whose mean
// relative humidity is between RHc and 100 % already holds cloud in its moister part; its mean condensate is
// (1 - RHc) qsat (1 + Q)^3 / 6 with Q = (RH - 1) / (1 - RHc). RHc is scale-aware: 0.85 on 15 km grids, closer to 1 on
// finer grids (0.96 at 1 km), since a finer box varies less within itself. This is the partial cloudiness every
// coarse model diagnoses (stratocumulus and stratus decks below saturation); here it is drawn, not fed back.

import type { RegionalModel } from './core.js';

export const EXT = { liquid: 150, ice: 55, snow: 27, graupel: 7.5, rain: 3 } as const;
/** largest extinction a display byte holds (1/m); bytes store (beta / EXT_MAX)^(1/3) */
export const EXT_MAX = 0.3;
/** Display byte of an extinction coefficient (1/m). */
export const extByte = (beta: number): number => (beta > 0 ? Math.min(255, Math.round(255 * Math.cbrt(beta / EXT_MAX))) : 0);

/** Critical relative humidity of sub-grid cloud on a grid of spacing dx (m). */
export const subgridRHc = (dx: number): number => 1 - 0.15 * Math.sqrt(Math.min(1, dx / 15000));
const esat = (T: number): number => 611.2 * Math.exp(17.67 * (T - 273.15) / (T - 29.65));
/** Saturation mixing ratio over water (kg/kg) at T (K) and p (Pa). */
export const qsatW = (T: number, p: number): number => { const es = esat(T); return 0.622 * es / Math.max(p - es, 1); };
/** Mean sub-grid condensate (kg/kg) of an unsaturated box (vapour qv, saturation qs, critical humidity rhc). */
export function subgridCloud(qv: number, qs: number, rhc: number): number {
  const rh = qv / qs;
  if (rh <= rhc || rh >= 1) return 0;
  const Q = (rh - 1) / (1 - rhc), a = 1 + Q;
  return (1 - rhc) * qs * a * a * a / 6;
}

/** Cloud extinction (1/m) at density rho from cloud water, sub-grid cloud (liquid above -20 °C, else ice), ice and snow. */
export function cloudExtinction(rho: number, qc: number, qsub: number, qi: number, qs: number, T: number): number {
  return rho * (EXT.liquid * Math.max(0, qc) + (T > 253.15 ? EXT.liquid : EXT.ice) * qsub + EXT.ice * Math.max(0, qi) + EXT.snow * Math.max(0, qs));
}
/** Precipitation extinction (1/m) from rain and graupel. */
export const precipExtinction = (rho: number, qr: number, qg: number): number => rho * (EXT.rain * Math.max(0, qr) + EXT.graupel * Math.max(0, qg));


/** Visible-channel cloud albedo of a column optical depth (two-stream, asymmetry 0.85). */
export const albedo = (tau: number): number => tau / (tau + 7.7);

/**
 * The 3-D view bytes of a CPU model ([k][j][i]): cloud (cloud water, sub-grid cloud when `subgrid`, ice, snow) and the
 * second channel (precipitation: rain and graupel; mode 1 updraft; mode 2 cyclonic vertical vorticity), as the GPU display
 * kernel packs them. Returns the extremes of w and of condensate / precipitation.
 */
export function volumeBytes(m: RegionalModel, cloud: Uint8Array, rain: Uint8Array, mode: number, subgrid: boolean): { wmax: number; wmin: number; qcmax: number; qrmax: number } {
  const { nx, ny, nz, dx } = m.c, sc = m.scalars, rhc = subgridRHc(dx);
  const qv = sc[0]!, qc = sc[1]!, qr = sc[2]!, qi = sc[3]!, qs = sc[4]!, qg = sc[5]!;
  let wmax = 0, wmin = 0, qcmax = 0, qrmax = 0;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const q = m.idx(i, j, k), o = (k * ny + j) * nx + i;
    const cl = Math.max(0, qc[q]! + qi[q]!), pr = Math.max(0, qr[q]! + qs[q]! + qg[q]!), rho = m.rho0[k]!;
    const pi = m.pi0[k]! + m.pp[q]!, T = m.th[q]! * pi;
    const qsub = subgrid && qc[q]! <= 1e-8 ? subgridCloud(qv[q]!, qsatW(T, 1e5 * Math.pow(pi, 1004.5 / 287.05)), rhc) : 0;
    cloud[o] = extByte(cloudExtinction(rho, qc[q]!, qsub, qi[q]!, qs[q]!, T));
    let v2 = extByte(precipExtinction(rho, qr[q]!, qg[q]!)) / 255;
    if (mode === 1) v2 = Math.sqrt(Math.max(0.5 * (m.w[q]! + m.w[q + m.plane]!), 0) / 40);
    else if (mode === 2) {
      const zeta = 0.25 * ((m.v[q + 1]! + m.v[q + 1 + m.sx]!) - (m.v[q - 1]! + m.v[q - 1 + m.sx]!)) / dx - 0.25 * ((m.u[q + m.sx]! + m.u[q + m.sx + 1]!) - (m.u[q - m.sx]! + m.u[q - m.sx + 1]!)) / m.c.dy;
      v2 = Math.sqrt(Math.max(zeta, 0) / 0.05);
    }
    rain[o] = Math.min(255, Math.round(v2 * 255));
    qcmax = Math.max(qcmax, cl); qrmax = Math.max(qrmax, pr);
    const w = m.w[q]!; wmax = Math.max(wmax, w); wmin = Math.min(wmin, w);
  }
  return { wmax, wmin, qcmax, qrmax };
}
