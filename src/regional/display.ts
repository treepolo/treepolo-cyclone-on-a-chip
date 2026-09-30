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
