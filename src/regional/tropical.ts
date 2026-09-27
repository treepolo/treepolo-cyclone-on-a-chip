// Idealised tropical-cyclone environment: moist tropical sounding and the Rotunno & Emanuel (1987)
// initial vortex in gradient-wind and hydrostatic balance.

import { DRY_AIR } from '../core/constants.js';
import { RegionalModel } from './core.js';
import { QV } from './kessler.js';

const G = 9.80665, CP = DRY_AIR.cp, RD = DRY_AIR.rd, LV = 2.5e6, EPS = 0.622;
const esat = (T: number): number => 611.2 * Math.exp(17.67 * (T - 273.15) / (T - 29.65));
const qsat = (T: number, p: number): number => { const e = esat(T); return EPS * e / (p - 0.378 * e); };

/**
 * Convectively near-neutral tropical sounding: surface T = sst - 1 K, RH 80%, dry adiabat to the LCL,
 * pseudo-adiabat above, isothermal (Ttrop) stratosphere; RH decreasing from 80% to 40% at 12 km.
 */
export function tropicalSounding(sst = 301.15, Ttrop = 200): (z: number) => { theta: number; qv: number } {
  const dz = 10, n = 3000;
  const th = new Float64Array(n + 1), qv = new Float64Array(n + 1);
  let T = sst - 1, p = 1e5;
  const rh0 = 0.8;
  const q0 = rh0 * qsat(T, p);
  let saturated = false;
  for (let i = 0; i <= n; i++) {
    const z = i * dz;
    const rh = z < 12000 ? rh0 - 0.4 * z / 12000 : 0.4 * Math.exp(-(z - 12000) / 3000);
    th[i] = T * Math.pow(1e5 / p, DRY_AIR.kappa);
    qv[i] = Math.min(rh * qsat(T, p), 0.02);
    // lapse rate for the next step
    let gamma: number;
    if (T <= Ttrop) gamma = 0;
    else {
      if (!saturated && q0 >= qsat(T, p)) saturated = true;
      if (!saturated) gamma = G / CP;
      else { const rs = qsat(T, p); gamma = G * (1 + LV * rs / (RD * T)) / (CP + LV * LV * rs * EPS / (RD * T * T)); }
    }
    const Tn = Math.max(Ttrop, T - gamma * dz);
    p *= Math.exp(-G * dz / (RD * 0.5 * (T + Tn)));
    T = Tn;
  }
  return (z: number) => {
    const x = Math.max(0, Math.min(n - 1e-6, z / dz)), i = Math.floor(x), w = x - i;
    return { theta: th[i]! * (1 - w) + th[i + 1]! * w, qv: qv[i]! * (1 - w) + qv[i + 1]! * w };
  };
}

/** Rotunno & Emanuel (1987) vortex tangential wind at the surface (m/s). */
export function re87Wind(r: number, f: number, vmax = 15, rm = 82500, r0 = 412500): number {
  if (r >= r0) return 0;
  const a = (2 * rm / (r + rm)) ** 3 - (2 * rm / (r0 + rm)) ** 3;
  return Math.sqrt(Math.max(0, vmax * vmax * (r / rm) ** 2 * a + f * f * r * r / 4)) - f * r / 2;
}

/**
 * Insert a balanced RE87 vortex centred in the domain: wind decreases linearly to zero at zTop;
 * pi' from gradient-wind balance (integrated inward from the outer radius), theta' from hydrostatic balance.
 */
export function insertVortex(m: RegionalModel, f: number, vmax = 15, zTop = 20000): void {
  const { nx, ny, nz, dx, dy } = m.c;
  const xc = nx * dx / 2, yc = ny * dy / 2;
  const nr = 1000, dr = 500;
  const piR = new Float64Array(nz * (nr + 1));
  for (let k = 0; k < nz; k++) {
    const z = m.zc[k]!, fac = Math.max(0, 1 - z / zTop);
    const thv = m.th0[k]! * (1 + 0.61 * m.qv0[k]!);
    let pi = 0;
    piR[k * (nr + 1) + nr] = 0;
    for (let n = nr - 1; n >= 0; n--) {
      const r = (n + 0.5) * dr;
      const vv = re87Wind(r, f, vmax) * fac;
      pi -= (vv * vv / r + f * vv) / (CP * thv) * dr;
      piR[k * (nr + 1) + n] = pi;
    }
  }
  const piAt = (k: number, r: number): number => { const x = Math.min(nr - 1e-6, r / dr), n = Math.floor(x), w = x - n; return piR[k * (nr + 1) + n]! * (1 - w) + piR[k * (nr + 1) + n + 1]! * w; };
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    for (let k = 0; k < nz; k++) {
      const z = m.zc[k]!, fac = Math.max(0, 1 - z / zTop);
      const q = m.idx(i, j, k);
      // u at the x-face, v at the y-face
      const xu = i * dx - xc, yu = (j + 0.5) * dy - yc, ru = Math.hypot(xu, yu);
      const xv = (i + 0.5) * dx - xc, yv = j * dy - yc, rv = Math.hypot(xv, yv);
      const vu = re87Wind(ru, f, vmax) * fac, vvv = re87Wind(rv, f, vmax) * fac;
      m.u[q] = ru > 0 ? -vu * yu / ru : 0;
      m.v[q] = rv > 0 ? vvv * xv / rv : 0;
      const xc2 = (i + 0.5) * dx - xc, yc2 = (j + 0.5) * dy - yc, rc = Math.hypot(xc2, yc2);
      m.pp[q] = piAt(k, rc);
    }
    // hydrostatic theta' = cp thv0^2 / g * d pi'/dz (centred differences)
    for (let k = 0; k < nz; k++) {
      const kp = Math.min(k + 1, nz - 1), km = Math.max(k - 1, 0);
      const q = m.idx(i, j, k);
      const dpdz = (m.pp[m.idx(i, j, kp)]! - m.pp[m.idx(i, j, km)]!) / ((kp - km) * m.c.dz);
      const thv = m.th0[k]! * (1 + 0.61 * m.qv0[k]!);
      m.th[q] = m.th0[k]! + CP * thv * thv / G * dpdz;
    }
  }
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const q = m.idx(i, j, k);
    if (m.scalars[QV]) m.scalars[QV]![q] = m.qv0[k]!;
  }
}

/** Minimum surface pressure (hPa) and maximum lowest-level wind speed (m/s) in the domain, plus centre indices. */
export function tcMetrics(m: RegionalModel): { pmin: number; vmax: number; ic: number; jc: number; rmw: number } {
  const { nx, ny, dx } = m.c, sx = m.sx;
  let pmin = Infinity, ic = 0, jc = 0, vmax = 0;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const q = m.idx(i, j, 0);
    const pi = m.pi0[0]! + m.pp[q]!;
    const p = DRY_AIR.pRef * Math.pow(pi, 1 / DRY_AIR.kappa) / 100;
    if (p < pmin) { pmin = p; ic = i; jc = j; }
    const sp = Math.hypot(0.5 * (m.u[q]! + m.u[q + 1]!), 0.5 * (m.v[q]! + m.v[q + sx]!));
    vmax = Math.max(vmax, sp);
  }
  // radius of maximum azimuthal-mean tangential wind at the lowest level
  const nb = Math.floor(nx / 2), vt = new Float64Array(nb), cnt = new Float64Array(nb);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const x = (i - ic) * dx, y = (j - jc) * dx, r = Math.hypot(x, y);
    const b = Math.floor(r / dx);
    if (b >= nb || r === 0) continue;
    const q = m.idx(i, j, 0);
    const ua = 0.5 * (m.u[q]! + m.u[q + 1]!), va = 0.5 * (m.v[q]! + m.v[q + sx]!);
    vt[b] = vt[b]! + (-ua * y + va * x) / r; cnt[b] = cnt[b]! + 1;
  }
  let best = 0, rmw = 0;
  for (let b = 0; b < nb; b++) if (cnt[b]! > 0 && vt[b]! / cnt[b]! > best) { best = vt[b]! / cnt[b]!; rmw = (b + 0.5) * dx; }
  return { pmin, vmax, ic, jc, rmw };
}
