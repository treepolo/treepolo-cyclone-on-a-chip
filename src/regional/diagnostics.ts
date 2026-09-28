// Diagnostics for the regional charts (display only): thermodynamic conversions, radar reflectivity and
// derived fields from the prognostic variables of a cell.

const RD = 287.05, CP = 1004.5, P0 = 1e5, LV = 2.5e6;

/** Pressure (Pa) from the Exner function. */
export const pressure = (pi: number): number => P0 * Math.pow(Math.max(pi, 1e-6), CP / RD);
/** Saturation mixing ratio over water (Tetens / Bolton), T in K, p in Pa. */
export function qsatW(T: number, p: number): number {
  const es = 611.2 * Math.exp(17.67 * (T - 273.15) / (T - 29.65));
  return 0.622 * es / Math.max(p - es, 1);
}
/** Equivalent potential temperature (K), simplified Bolton form. */
export const thetaE = (T: number, p: number, qv: number): number => T * Math.pow(P0 / p, RD / CP) * Math.exp(LV * qv / (CP * T));
/** Dew point (deg C) from the mixing ratio and pressure. */
export function dewPoint(qv: number, p: number): number {
  const e = Math.max(qv, 1e-9) * p / (0.622 + qv) / 100;     // hPa
  const l = Math.log(e / 6.112);
  return 243.5 * l / (17.67 - l);
}
/** Radar reflectivity (dBZ) from rain, snow and graupel mixing ratios (kg/kg) and air density. */
export function dbz(rho: number, qr: number, qs: number, qg: number): number {
  const z = 3.63e9 * Math.pow(rho * Math.max(qr, 0), 1.75) + 9.80e8 * Math.pow(rho * Math.max(qs, 0), 1.75) + 4.33e10 * Math.pow(rho * Math.max(qg, 0), 1.75);
  return 10 * Math.log10(Math.max(z, 1e-3));
}

/** Fields of a horizontal slice, in this order (see sliceFields). */
export const SLICE_VARS = ['dbz', 'w', 'speed', 'u', 'v', 'thp', 'thetaE', 'rh', 'zeta', 'pp', 'qv', 'cloud', 'precip'] as const;
/** Fields of a cross-section, in this order. */
export const SECTION_VARS = ['dbz', 'w', 'u', 'v', 'thp', 'thetaE', 'rh', 'cloud', 'precip', 'T', 'Td', 'p'] as const;
/** Composite (column / surface) maps, in this order. */
export const MAP_VARS = ['dbzMax', 'ctopT', 'uh', 'sfcWind', 'sfcPp', 'sfcThp', 'rain', 'snow', 'wMax'] as const;

export interface CellState { u: number; v: number; w: number; th: number; pp: number; q: ArrayLike<number> }
export interface LevelBase { th0: number; pi0: number; rho0: number }

/** Section / sounding values of one cell (SECTION_VARS order). q: qv, qc, qr[, qi, qs, qg]. */
export function sectionValues(c: CellState, b: LevelBase, out: Float32Array, o: number): void {
  const pi = b.pi0 + c.pp, p = pressure(pi), T = c.th * pi, q = c.q;
  const qv = q[0] ?? 0, qc = q[1] ?? 0, qr = q[2] ?? 0, qi = q[3] ?? 0, qs = q[4] ?? 0, qg = q[5] ?? 0;
  out[o] = dbz(b.rho0, qr, qs, qg); out[o + 1] = c.w; out[o + 2] = c.u; out[o + 3] = c.v; out[o + 4] = c.th - b.th0;
  out[o + 5] = thetaE(T, p, qv); out[o + 6] = 100 * qv / qsatW(T, p); out[o + 7] = 1e3 * Math.max(0, qc + qi); out[o + 8] = 1e3 * Math.max(0, qr + qs + qg);
  out[o + 9] = T - 273.15; out[o + 10] = dewPoint(qv, p); out[o + 11] = p / 100;
}
