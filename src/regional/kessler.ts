// Kessler warm-rain microphysics (Kessler 1969; Klemp & Wilhelmson 1978), following the WRF / CM1
// implementation: autoconversion and accretion (implicit), saturation adjustment with latent heating,
// evaporation of rain in subsaturated air, and rain sedimentation with the KW78 terminal velocity.
// Species are mixing ratios (kg/kg) in RegionalModel.scalars: [qv, qc, qr].

import { DRY_AIR, VT_MAX } from '../core/constants.js';
import { RegionalModel } from './core.js';

export const QV = 0, QC = 1, QR = 2;
const XLV = 2.5e6;
const RV = 461.5;

export class KesslerMicrophysics {
  /** accumulated surface rain (kg m^-2 = mm) per column [j][i] */
  readonly rainAcc: Float64Array;
  /** clipped negative water from advection (kg m^-2, domain sum) */
  clipped = 0;

  constructor(private readonly m: RegionalModel) {
    this.rainAcc = new Float64Array(m.c.nx * m.c.ny);
    // moist buoyancy is computed exactly in the core from theta_rho = theta (1 + 0.61 qv - qc - qr)
  }

  /** Apply microphysics over dt (call after each dynamics step). */
  apply(dt: number): void {
    const m = this.m, { nx, ny, nz, dz } = m.c, cp = DRY_AIR.cp, rd = DRY_AIR.rd;
    const qv = m.scalars[QV]!, qc = m.scalars[QC]!, qr = m.scalars[QR]!;
    const rho = m.rho0, pi0 = m.pi0;
    const rhoSfc = rho[0]!;
    const vt = new Float64Array(nz), col = new Float64Array(nz), flux = new Float64Array(nz + 1);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      // ---- clip negatives produced by advection (bookkeeping)
      for (let k = 0; k < nz; k++) {
        const q = m.idx(i, j, k);
        for (const a of [qv, qc, qr]) if (a[q]! < 0) { this.clipped -= a[q]! * rho[k]! * dz; a[q] = 0; }
      }
      // ---- rain sedimentation (flux form, upwind, sub-stepped for CFL)
      let vmax = 0;
      for (let k = 0; k < nz; k++) {
        const q = m.idx(i, j, k);
        col[k] = qr[q]!;
        vt[k] = qr[q]! > 1e-12 ? Math.min(36.34 * Math.pow(0.001 * rho[k]! * qr[q]!, 0.1364) * Math.sqrt(rhoSfc / rho[k]!), VT_MAX) : 0;
        vmax = Math.max(vmax, vt[k]!);
      }
      const nsub = Math.max(1, Math.ceil(vmax * dt / (0.8 * dz)));
      const dts = dt / nsub;
      for (let s = 0; s < nsub; s++) {
        for (let k = 0; k < nz; k++) flux[k] = rho[k]! * vt[k]! * col[k]!;   // downward flux out of cell k through its bottom
        flux[nz] = 0;
        this.rainAcc[j * nx + i] = this.rainAcc[j * nx + i]! + flux[0]! * dts;
        for (let k = 0; k < nz; k++) col[k] = col[k]! + dts * (flux[k + 1]! - flux[k]!) / (rho[k]! * dz);
      }
      // ---- warm-rain processes and saturation adjustment
      for (let k = 0; k < nz; k++) {
        const q = m.idx(i, j, k);
        let vv = qv[q]!, cc = qc[q]!, rr = col[k]!;
        const r = rho[k]!;
        // autoconversion (threshold 1 g/kg, rate 1e-3 s^-1) and accretion, implicit in qr
        const factorn = 1 / (1 + 2.2 * dt * Math.pow(Math.max(0, rr), 0.875));
        const qrprod = cc - (cc - dt * Math.max(0.001 * (cc - 0.001), 0)) * factorn;
        cc = Math.max(cc - qrprod, 0);
        rr = rr + qrprod;
        // saturation (Teten's formula, as in KW78 / WRF Kessler)
        const pi = pi0[k]! + m.pp[q]!;
        const p = DRY_AIR.pRef * Math.pow(pi, cp / rd);
        const T = m.th[q]! * pi;
        const qvs = 380 / p * Math.exp(17.27 * (T - 273.15) / (T - 35.86));
        const f5 = 237.3 * 17.27 * XLV / cp;
        const prod = (vv - qvs) / (1 + qvs * f5 / (T - 35.86) ** 2);
        // evaporation of rain in subsaturated air (Klemp & Wilhelmson 1978)
        const rq = Math.max(r * rr, 0);
        const ern = Math.min(dt * (((1.6 + 124.9 * Math.pow(rq, 0.2046)) * Math.pow(rq, 0.525)) / (2.55e8 / (p * qvs) + 5.4e5)) * (Math.max(qvs - vv, 0) / (r * qvs)),
          Math.max(-prod - cc, 0), rr);
        const product = Math.max(prod, -cc);
        m.th[q] = m.th[q]! + XLV / (cp * pi) * (product - ern);
        vv = Math.max(vv - product + ern, 0);
        cc = cc + product;
        rr = rr - ern;
        qv[q] = vv; qc[q] = cc; qr[q] = Math.max(rr, 0);
      }
    }
    void RV;
  }
}

/** Weisman & Klemp (1982) sounding: theta(z), qv(z) (with the 14 g/kg surface cap used in CM1). */
export function weismanKlemp(z: number): { theta: number; qv: number } { return weismanKlempQ(0.014)(z); }
/** The WK82 sounding with another boundary-layer mixing-ratio cap (kg/kg): a larger cap gives a moister
 *  boundary layer, a lower cloud base and more CAPE (e.g. 0.016 for a tornadic environment). */
export function weismanKlempQ(qvMax: number): (z: number) => { theta: number; qv: number } {
  return (z: number) => {
    const theta = wkTheta(z), rh = z <= 12000 ? 1 - 0.75 * Math.pow(z / 12000, 1.25) : 0.25;
    const pi = wkExner(z), p = DRY_AIR.pRef * Math.pow(pi, 1 / DRY_AIR.kappa);
    const T = theta * pi;
    const qvs = 380 / p * Math.exp(17.27 * (T - 273.15) / (T - 35.86));
    return { theta, qv: Math.min(qvMax, rh * qvs) };
  };
}
function wkTheta(z: number): number {
  const ztr = 12000, th0 = 300, thtr = 343, Ttr = 213;
  return z <= ztr ? th0 + (thtr - th0) * Math.pow(z / ztr, 1.25) : thtr * Math.exp(9.80665 * (z - ztr) / (DRY_AIR.cp * Ttr));
}
/** Exner function of the (dry) WK82 sounding by hydrostatic integration from 1000 hPa, tabulated every 50 m. */
let wkTable: Float64Array | null = null;
function wkExner(z: number): number {
  if (!wkTable) {
    wkTable = new Float64Array(1001);
    let pi = 1;
    wkTable[0] = 1;
    for (let n = 1; n <= 1000; n++) { const zm = (n - 0.5) * 50; pi -= 9.80665 / (DRY_AIR.cp * wkTheta(zm)) * 50; wkTable[n] = pi; }
  }
  const x = Math.max(0, Math.min(999.999, z / 50)), n = Math.floor(x), w = x - n;
  return wkTable[n]! * (1 - w) + wkTable[n + 1]! * w;
}
