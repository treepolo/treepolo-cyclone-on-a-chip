// Moist thermodynamic constants and saturation functions (SI).
// Saturation vapour pressure uses the simplified Clausius–Clapeyron form of
// Frierson et al. (2006): e_s = e0 exp[-(L/R_v)(1/T - 1/T0)], constant latent heat.

import { DRY_AIR } from '../../core/constants.js';

export const MOIST = {
  Lv: 2.5e6,        // J kg^-1
  Rv: 461.5,        // J kg^-1 K^-1
  e0: 610.78,       // Pa
  T0: 273.16,       // K
  rhoWater: 1000,   // kg m^-3
  cpWater: 4186,    // J kg^-1 K^-1
  stefan: 5.670374e-8,
  vonKarman: 0.4,
};
export const EPS = DRY_AIR.rd / MOIST.Rv;        // 0.622

export function satVaporPressure(T: number): number {
  return MOIST.e0 * Math.exp(-(MOIST.Lv / MOIST.Rv) * (1 / T - 1 / MOIST.T0));
}

/** Saturation specific humidity (kg/kg) at temperature T (K) and pressure p (Pa). */
export function qsat(T: number, p: number): number {
  const es = satVaporPressure(T);
  return EPS * es / Math.max(p - (1 - EPS) * es, 1e-3 * p);
}

/** d qsat / dT (K^-1). */
export function dqsatdT(T: number, p: number): number {
  const es = satVaporPressure(T);
  const den = Math.max(p - (1 - EPS) * es, 1e-3 * p);
  const des = es * MOIST.Lv / (MOIST.Rv * T * T);
  return EPS * des * p / (den * den);
}

/** Mixing ratio from vapour pressure. */
export function mixingRatio(e: number, p: number): number {
  return EPS * e / Math.max(p - e, 1e-3 * p);
}
