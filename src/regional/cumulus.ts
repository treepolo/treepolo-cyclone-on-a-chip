// Scale-aware cumulus parameterization of the regional model: the simplified Betts-Miller scheme of Frierson (2007,
// J. Atmos. Sci. 64, 1959-1976), after Betts (1986) and Betts & Miller (1986).
//
// Each column lifts a parcel from the lowest level (pseudo-adiabatic, the same steps as the CAPE diagnostic). Where
// it has CAPE and a cloud at least CU_MIN_DEPTH deep, temperature and humidity from the ground to the level of
// neutral buoyancy relax toward the parcel's temperature and CU_RH of its saturation humidity over CU_TAU:
//   dT/dt = -(T - Tref) / tau + c,   dq/dt = -(q - qref) / tau,
// with the constant c making the column's heating equal the latent heat of the vapour removed (energy conserved:
// Frierson's adjustment of the reference temperature). The removed vapour is precipitation: CU_DETRAIN of it is left
// as cloud ice (liquid in warm columns) in the top CU_DETRAIN_DEPTH of the cloud, where convection detrains its
// condensate (anvils, the outflow cirrus); the rest becomes rain water below the cloud base, from where the
// microphysics lets it fall and partly evaporate.
// Shallow convection (Frierson's "shallower" scheme): where the column is too dry for deep convection (the vapour
// removed over the whole cloud would be negative), the cloud top is lowered to the highest level at which it is still
// non-negative; the same relaxation there moves boundary-layer vapour up into the lower troposphere (moistening it for
// later deep convection) with almost no rain.
//
// Scale awareness: the tendencies are multiplied by cumulusScale(dx) = clamp((dx - 3 km) / (12 km - 3 km), 0, 1):
// full strength on 12 km grids and coarser, none at 3 km and finer, where the model resolves convection itself.

import type { RegionalModel } from './core.js';

export const CU_TAU = 7200, CU_RH = 0.7, CU_MIN_DEPTH = 3000, CU_DETRAIN = 0.2, CU_DETRAIN_DEPTH = 2000;
const CP = 1004.5, LV = 2.5e6, RD = 287.05, G = 9.80665;

/** Strength of the parameterized convection on a grid of spacing dx (m). */
export const cumulusScale = (dx: number): number => Math.max(0, Math.min(1, (dx - 3000) / 9000));

const esat = (T: number): number => 611.2 * Math.exp(17.67 * (T - 273.15) / (T - 29.65));

/**
 * One column: temperatures T, pressures p (Pa), vapour q, density rho per level, level spacing dz, time step dt,
 * strength sc. Returns null without convection, else the changes of T and q per level (from level 0 to kt), the
 * precipitation (kg/m^2 over dt), the cloud base kb (first saturated parcel level) and top kt (last buoyant level, or
 * the lowered top of shallow convection), the parcel's CAPE (J/kg) and whether the convection is deep.
 */
export function cumulusColumn(T: ArrayLike<number>, p: ArrayLike<number>, q: ArrayLike<number>, rho: ArrayLike<number>, dz: number, dt: number, sc: number,
  dT: Float64Array, dq: Float64Array): { P: number; kb: number; kt: number; cape: number; deep: boolean } | null {
  const nz = T.length;
  if (sc <= 0 || nz < 3) return null;
  // parcel (parcelAscent in diagnostics.ts / the GPU display kernel): pseudo-adiabatic, 4 Newton steps of saturation adjustment
  const Tp = new Float64Array(nz);
  let pth = T[0]! * Math.pow(1e5 / p[0]!, RD / CP), pq = Math.max(0, q[0]!), kb = -1, lfc = -1, kt = -1, cape = 0;
  Tp[0] = T[0]!;
  for (let k = 1; k < nz; k++) {
    const pk = p[k]!, pik = Math.pow(pk / 1e5, RD / CP);
    let tp = pth * pik, d = 0;
    const es0 = esat(tp);
    if (pq > 0.622 * es0 / Math.max(pk - es0, 1)) {
      for (let it = 0; it < 4; it++) {
        const tt = tp + LV * d / CP, es = esat(tt), qs = 0.622 * es / Math.max(pk - es, 1);
        const dqs = qs * pk / Math.max(pk - es, 1) * 17.67 * 243.5 / ((tt - 29.65) * (tt - 29.65));
        d += (pq - d - qs) / (1 + LV / CP * dqs);
      }
      d = Math.max(0, Math.min(pq, d));
      if (kb < 0) kb = k;
    }
    tp += LV * d / CP; pq -= d; pth = tp / pik; Tp[k] = tp;
    const tve = T[k]! * (1 + 0.61 * Math.max(0, q[k]!)), b = G * (tp * (1 + 0.61 * pq) - tve) / tve;
    if (lfc < 0) { if (kb >= 0 && b > 0) { lfc = k; cape += b * dz; kt = k; } }
    else if (kt === k - 1 && b > 0) { cape += b * dz; kt = k; }
  }
  if (lfc < 0 || cape <= 0 || kt <= kb) return null;
  // relaxation toward the reference profiles
  const a = dt / CU_TAU;
  let Pq = 0, kq = -1;
  for (let k = 0; k <= kt; k++) {
    const Tr = Tp[k]!, es = esat(Tr), qr = CU_RH * 0.622 * es / Math.max(p[k]! - es, 1);
    dT[k] = -(T[k]! - Tr) * a; dq[k] = -(q[k]! - qr) * a;
    Pq -= dq[k]! * rho[k]! * dz;
    if (Pq >= 0) kq = k;
  }
  // deep convection needs a deep cloud and a net loss of vapour; otherwise shallow convection up to the highest level
  // where the loss is still non-negative (at least one level above the cloud base)
  const deep = Pq > 0 && (kt - kb) * dz >= CU_MIN_DEPTH;
  if (!deep) { if (kq <= kb) return null; kt = kq; }
  let P = 0, PT = 0, mass = 0;
  for (let k = 0; k <= kt; k++) { const md = rho[k]! * dz; P -= dq[k]! * md; PT += CP * dT[k]! * md / LV; mass += md; }
  if (!(P >= 0)) return null;
  // energy: the column's heating equals the latent heat of the vapour removed
  const c = (P - PT) * LV / (CP * mass);
  for (let k = 0; k <= kt; k++) { dT[k] = sc * (dT[k]! + c); dq[k] = sc * dq[k]!; }
  return { P: sc * P, kb, kt, cape, deep };
}

/** Per-column state of the scheme for display: cloud base level, top level (shallow convection: -top - 2; none: -1),
 *  convective rain rate (mm/h). */
export interface CumulusInfo { kb: Int16Array; kt: Int16Array; rate: Float32Array }

/**
 * Apply the scheme to every column of the CPU model over dt (theta, vapour; precipitation as rain water below the
 * cloud base, detrained condensate at the top). `ice`: the model carries cloud ice (scalar 3), else detrainment
 * goes to cloud water (scalar 1). Fills `info` when given.
 */
export function applyCumulus(m: RegionalModel, dt: number, ice: boolean, info: CumulusInfo | null): void {
  const { nx, ny, nz, dz } = m.c, sc = cumulusScale(m.c.dx);
  if (info) { info.kb.fill(-1); info.kt.fill(-1); info.rate.fill(0); }
  if (sc <= 0) return;
  const T = new Float64Array(nz), p = new Float64Array(nz), q = new Float64Array(nz), dT = new Float64Array(nz), dq = new Float64Array(nz), pi = new Float64Array(nz);
  const qv = m.scalars[0]!, qc = m.scalars[1]!, qr = m.scalars[2]!, qi = ice ? m.scalars[3]! : null;
  const nd = Math.max(1, Math.round(CU_DETRAIN_DEPTH / dz));
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    for (let k = 0; k < nz; k++) {
      const s = m.idx(i, j, k); pi[k] = m.pi0[k]! + m.pp[s]!;
      T[k] = m.th[s]! * pi[k]!; p[k] = 1e5 * Math.pow(pi[k]!, CP / RD); q[k] = qv[s]!;
    }
    const r = cumulusColumn(T, p, q, m.rho0, dz, dt, sc, dT, dq);
    if (!r) continue;
    for (let k = 0; k <= r.kt; k++) { const s = m.idx(i, j, k); m.th[s] = m.th[s]! + dT[k]! / pi[k]!; qv[s] = Math.max(0, qv[s]! + dq[k]!); }
    if (info) { const c = j * nx + i; info.kb[c] = r.kb; info.kt[c] = r.deep ? r.kt : -r.kt - 2; info.rate[c] = r.P / dt * 3600; }
    if (!(r.P > 0)) continue;
    // detrained condensate in the top of the cloud, rain water below the cloud base
    const k0 = Math.max(r.kb, r.kt - nd + 1);
    let mt = 0; for (let k = k0; k <= r.kt; k++) mt += m.rho0[k]! * dz;
    for (let k = k0; k <= r.kt; k++) { const s = m.idx(i, j, k), add = CU_DETRAIN * r.P / mt; if (qi && T[k]! < 273.15) qi[s] = qi[s]! + add; else qc[s] = qc[s]! + add; }
    const kr = Math.max(1, r.kb);
    let mb = 0; for (let k = 0; k < kr; k++) mb += m.rho0[k]! * dz;
    for (let k = 0; k < kr; k++) { const s = m.idx(i, j, k); qr[s] = qr[s]! + (1 - CU_DETRAIN) * r.P / mb; }
  }
}
