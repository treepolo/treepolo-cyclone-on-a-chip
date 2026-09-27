// Simplified Betts–Miller convection (Frierson 2007, JAS 64, 1959–1976), ported from the
// Isca/GFDL qe_moist_convection module. Relaxes T and q toward a moist-adiabatic reference
// profile (parcel lifted from the lowest level) with relative humidity rhbm, timescale tauBm,
// with the "change Tref" enthalpy-conserving closure and shallow convection.
// Column arrays are indexed top (0) -> bottom (K-1); pHalf has K+1 entries.

import { DRY_AIR } from '../../core/constants.js';
import { EPS, MOIST, mixingRatio, satVaporPressure } from './thermo.js';

export const SBM = { tauBm: 7200, rhbm: 0.7, Tmin: 160 };

const kappa = DRY_AIR.kappa, cp = DRY_AIR.cp, rd = DRY_AIR.rd, Lv = MOIST.Lv, Rv = MOIST.Rv;

function virtualTemp(T: number, r: number): number {
  const q = r / (1 + r);
  return T * (1 + q * (Rv / rd - 1));
}

/** Temperature at the lifting condensation level from value = ln(e_s(T)) - ln(T)/kappa. */
function lclTemperature(value: number): number {
  let T = 250;
  for (let it = 0; it < 30; it++) {
    const f = Math.log(satVaporPressure(T)) - Math.log(T) / kappa - value;
    const df = Lv / (Rv * T * T) - 1 / (kappa * T);
    const dT = f / df;
    T -= dT;
    if (T < 100) T = 100;
    if (Math.abs(dT) < 1e-6) break;
  }
  return T;
}

export interface SbmWork {
  Tp: Float64Array; rp: Float64Array; rin: Float64Array; Tv: Float64Array;
  Tref: Float64Array; qref: Float64Array; dT: Float64Array; dq: Float64Array;
}
export function sbmWork(K: number): SbmWork {
  const f = (): Float64Array => new Float64Array(K);
  return { Tp: f(), rp: f(), rin: f(), Tv: f(), Tref: f(), qref: f(), dT: f(), dq: f() };
}

/**
 * Apply SBM convection to one column over dt. Modifies T and q in place.
 * Returns convective precipitation (kg m^-2 over dt) and CAPE (J/kg).
 */
export function sbmColumn(T: Float64Array, q: Float64Array, pFull: Float64Array, pHalf: Float64Array,
                          dt: number, g: number, w: SbmWork): { rain: number; cape: number } {
  const K = T.length, ks = K - 1;
  const { Tp, rp, rin, Tv, Tref, qref, dT, dq } = w;
  const tau = SBM.tauBm;
  // guard the model top (p = 0) inside logarithms
  const ph = (k: number): number => (k === 0 ? Math.max(pHalf[0]!, 0.1 * pFull[0]!) : pHalf[k]!);
  for (let k = 0; k < K; k++) {
    rin[k] = q[k]! / (1 - q[k]!);
    Tv[k] = virtualTemp(T[k]!, rin[k]!);
    Tp[k] = T[k]!; rp[k] = rin[k]!;
    dT[k] = 0; dq[k] = 0; Tref[k] = T[k]!; qref[k] = q[k]!;
  }
  // ---------- CAPE of a parcel from the lowest level
  let cape = 0, nocape = true, kLZB = -1, kLCL = ks;
  const T0 = T[ks]!, r0 = rin[ks]!;
  const rs0 = mixingRatio(satVaporPressure(T0), pFull[ks]!);
  let skip = false;
  const noCapeReset = (): void => { kLZB = -1; for (let k = 0; k < K; k++) { Tp[k] = T[k]!; rp[k] = rin[k]!; } };
  if (r0 >= rs0) {
    kLCL = ks;
    Tp[ks] = T0 + (r0 - rs0) / (cp / Lv + Lv * rs0 / Rv / (T0 * T0));
    rp[ks] = mixingRatio(satVaporPressure(Tp[ks]!), pFull[ks]!);
  } else {
    const theta0 = T0 * Math.pow(DRY_AIR.pRef / pFull[ks]!, kappa);
    if (r0 <= 0) {
      skip = true;
    } else {
      const value = Math.log(Math.pow(theta0, -1 / kappa) * DRY_AIR.pRef * r0 / (EPS + r0));
      let TLCL = lclTemperature(value);
      let pLCL = DRY_AIR.pRef * Math.pow(TLCL / theta0, 1 / kappa);
      if (pLCL < pFull[0]!) { pLCL = pFull[0]!; TLCL = theta0 * Math.pow(pLCL / DRY_AIR.pRef, kappa); }
      let k = ks;
      while (k > 0 && pFull[k]! > pLCL) {
        Tp[k] = theta0 * Math.pow(pFull[k]! / DRY_AIR.pRef, kappa);
        rp[k] = mixingRatio(satVaporPressure(Tp[k]!), pFull[k]!);
        k--;
      }
      kLCL = k;
      let a = kappa * TLCL + (Lv / cp) * r0;
      let b = Lv * Lv * r0 / (cp * Rv * TLCL * TLCL);
      Tp[kLCL] = TLCL + (a / (1 + b)) * Math.log(pFull[kLCL]! / pLCL) / 2;
      if (Tp[kLCL]! < SBM.Tmin) { skip = true; noCapeReset(); }
      else {
        rp[kLCL] = mixingRatio(satVaporPressure(Tp[kLCL]!), (pFull[kLCL]! + pLCL) / 2);
        a = kappa * Tp[kLCL]! + (Lv / cp) * rp[kLCL]!;
        b = Lv * Lv * rp[kLCL]! / (cp * Rv * Tp[kLCL]! ** 2);
        Tp[kLCL] = TLCL + (a / (1 + b)) * Math.log(pFull[kLCL]! / pLCL);
        if (Tp[kLCL]! < SBM.Tmin) { skip = true; noCapeReset(); }
        else {
          rp[kLCL] = mixingRatio(satVaporPressure(Tp[kLCL]!), pFull[kLCL]!);
          const buoy = virtualTemp(Tp[kLCL]!, rp[kLCL]!) - Tv[kLCL]!;
          if (buoy > 0) { cape += rd * buoy * Math.log(ph(kLCL + 1) / ph(kLCL)); nocape = false; }
        }
      }
    }
  }
  if (!skip) {
    for (let k = kLCL - 1; k >= 0; k--) {
      let a = kappa * Tp[k + 1]! + (Lv / cp) * rp[k + 1]!;
      let b = Lv * Lv * rp[k + 1]! / (cp * Rv * Tp[k + 1]! ** 2);
      Tp[k] = Tp[k + 1]! + (a / (1 + b)) * Math.log(pFull[k]! / pFull[k + 1]!) / 2;
      if (Tp[k]! < SBM.Tmin) { if (nocape) noCapeReset(); break; }
      rp[k] = mixingRatio(satVaporPressure(Tp[k]!), (pFull[k]! + pFull[k + 1]!) / 2);
      a = kappa * Tp[k]! + (Lv / cp) * rp[k]!;
      b = Lv * Lv * rp[k]! / (cp * Rv * Tp[k]! ** 2);
      Tp[k] = Tp[k + 1]! + (a / (1 + b)) * Math.log(pFull[k]! / pFull[k + 1]!);
      if (Tp[k]! < SBM.Tmin) { if (nocape) noCapeReset(); break; }
      rp[k] = mixingRatio(satVaporPressure(Tp[k]!), pFull[k]!);
      const buoy = virtualTemp(Tp[k]!, rp[k]!) - Tv[k]!;
      if (buoy < 0) {
        if (!nocape) { kLZB = k + 1; break; }
      } else {
        cape += rd * buoy * Math.log(ph(k + 1) / ph(k));
        nocape = false;
      }
    }
  }
  if (!(cape > 0)) return { rain: 0, cape: 0 };
  if (kLZB < 0) kLZB = 0;

  // ---------- reference profiles
  for (let k = 0; k < K; k++) Tref[k] = Tp[k]!;
  for (let k = kLZB; k <= ks; k++) {
    const eref = SBM.rhbm * pFull[k]! * rp[k]! / (rp[k]! + EPS);
    const r = mixingRatio(eref, pFull[k]!);
    qref[k] = r / (1 + r);
  }
  for (let k = 0; k < kLZB; k++) { Tref[k] = T[k]!; qref[k] = q[k]!; }

  // precipitation from moisture relaxation (Pq) and from temperature relaxation (Pt)
  let Pq = 0, Pt = 0;
  for (let k = kLZB; k <= ks; k++) {
    const dp = pHalf[k + 1]! - pHalf[k]!;
    dq[k] = -(q[k]! - qref[k]!) * dt / tau;
    Pq += dq[k]! * (pHalf[k]! - pHalf[k + 1]!);
    dT[k] = -(T[k]! - Tref[k]!) * dt / tau;
    Pt += (cp / Lv) * dT[k]! * dp;
  }
  Pq /= g; Pt /= g;

  if (Pq > 0 && Pt > 0) {
    // deep convection
    if (Pq > Pt) {
      const f = Pt / Pq;
      for (let k = kLZB; k <= ks; k++) dq[k] = dq[k]! * f;
      Pq = Pt;
    } else {
      let dk = 0;
      for (let k = kLZB; k <= ks; k++) dk -= (dT[k]! + (Lv / cp) * dq[k]!) * (pHalf[k + 1]! - pHalf[k]!);
      dk /= pHalf[ks + 1]! - pHalf[kLZB]!;
      for (let k = kLZB; k <= ks; k++) dT[k] = dT[k]! + dk;
    }
  } else if (Pt > 0) {
    // shallow convection: find the level of zero precipitation
    let k = kLZB;
    while (Pq < 0 && k <= ks) { Pq -= dq[k]! * (pHalf[k]! - pHalf[k + 1]!) / g; k++; }
    const kTop = k - 1;
    const found = Pq > 0;
    if (kTop > kLZB) for (let kk = kLZB; kk <= kTop - 1; kk++) { dT[kk] = 0; dq[kk] = 0; }
    if (found) {
      const c = Pq * g / (dq[kTop]! * (pHalf[kTop + 1]! - pHalf[kTop]!));
      dq[kTop] = dq[kTop]! * c; dT[kTop] = dT[kTop]! * c;
      let dk = 0;
      for (let kk = kTop; kk <= ks; kk++) dk += dT[kk]! * (pHalf[kk]! - pHalf[kk + 1]!);
      dk /= pHalf[ks + 1]! - pHalf[kTop]!;
      if (kTop !== ks) for (let kk = kTop; kk <= ks; kk++) dT[kk] = dT[kk]! + dk;
    } else {
      const k1 = kTop === kLZB ? ks : kLZB, k2 = kTop === kLZB ? ks : kTop;
      for (let kk = k1; kk <= k2; kk++) { dT[kk] = 0; dq[kk] = 0; }
    }
    Pq = 0;
  } else {
    return { rain: 0, cape };
  }
  for (let k = 0; k < K; k++) { T[k] = T[k]! + dT[k]!; q[k] = q[k]! + dq[k]!; }
  return { rain: Math.max(0, Pq), cape };
}
