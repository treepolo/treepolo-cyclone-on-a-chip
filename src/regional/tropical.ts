// Idealised tropical-cyclone environment: moist tropical sounding and the Rotunno & Emanuel (1987)
// initial vortex in gradient-wind and hydrostatic balance.

import { DRY_AIR } from '../core/constants.js';
import { RegionalModel } from './core.js';
import { QV } from './kessler.js';
import { rng } from '../core/random.js';

const G = 9.80665, CP = DRY_AIR.cp, RD = DRY_AIR.rd, LV = 2.5e6, EPS = 0.622;
const esat = (T: number): number => 611.2 * Math.exp(17.67 * (T - 273.15) / (T - 29.65));
const qsat = (T: number, p: number): number => { const e = esat(T); return EPS * e / (p - 0.378 * e); };

/**
 * Convectively near-neutral tropical sounding: surface T = sst - 1 K, RH 80%, dry adiabat to the LCL,
 * pseudo-adiabat above, isothermal (Ttrop) stratosphere; RH decreasing linearly from 80% to rhTop at 12 km
 * (0.4 by default; about 0.6 matches the moist-tropical hurricane-season mean of Dunion 2011).
 */
export function tropicalSounding(sst = 301.15, Ttrop = 200, rhTop = 0.4): (z: number) => { theta: number; qv: number } {
  return tropicalProfile(sst, Ttrop, rhTop, 0, false);
}

/**
 * Tropical sounding builder. The reference parcel (surface air, T = sst - 1 K, RH 80%) rises dry-adiabatically to
 * its LCL and pseudo-adiabatically above; the environment is that parcel path, colder by
 * cool * sin(pi (z - 1 km) / 14 km) between 1 and 15 km, isothermal (Ttrop) in the stratosphere. Water vapour follows the
 * relative-humidity profile at the environment temperature; with `mixed`, the lowest kilometre also holds the surface
 * air's water vapour as far as 90% relative humidity allows (a well-mixed sub-cloud layer that is not saturated).
 */
function tropicalProfile(sst: number, Ttrop: number, rhTop: number, cool: number, mixed: boolean): (z: number) => { theta: number; qv: number } {
  const dz = 10, n = 3000;
  const th = new Float64Array(n + 1), qv = new Float64Array(n + 1);
  let Tp = sst - 1, p = 1e5;
  const rh0 = 0.8;
  const q0 = rh0 * qsat(Tp, p);
  const off = (z: number): number => (z > 1000 && z < 15000 ? cool * Math.sin(Math.PI * (z - 1000) / 14000) : 0);
  let saturated = false;
  for (let i = 0; i <= n; i++) {
    const z = i * dz, T = Math.max(Ttrop, Tp - off(z));
    const rh = z < 12000 ? rh0 - (rh0 - rhTop) * z / 12000 : rhTop * Math.exp(-(z - 12000) / 3000);
    th[i] = T * Math.pow(1e5 / p, DRY_AIR.kappa);
    qv[i] = Math.min(rh * qsat(T, p), 0.02);
    if (mixed && z < 1000) qv[i] = Math.max(qv[i]!, Math.min(q0, 0.9 * qsat(T, p)));
    // lapse rate of the reference parcel for the next step
    let gamma: number;
    if (Tp <= Ttrop) gamma = 0;
    else {
      if (!saturated && q0 >= qsat(Tp, p)) saturated = true;
      if (!saturated) gamma = G / CP;
      else { const rs = qsat(Tp, p); gamma = G * (1 + LV * rs / (RD * Tp)) / (CP + LV * LV * rs * EPS / (RD * Tp * Tp)); }
    }
    const Tpn = Math.max(Ttrop, Tp - gamma * dz), Tn = Math.max(Ttrop, Tpn - off(z + dz));
    p *= Math.exp(-G * dz / (RD * 0.5 * (T + Tn)));
    Tp = Tpn;
  }
  return (z: number) => {
    const x = Math.max(0, Math.min(n - 1e-6, z / dz)), i = Math.floor(x), w = x - i;
    return { theta: th[i]! * (1 - w) + th[i + 1]! * w, qv: qv[i]! * (1 - w) + qv[i + 1]! * w };
  };
}

/** Environment of the tropical-cyclone experiments: 'unstable' a conditionally unstable tropical sounding like the
 *  observed mean over tropical oceans, 're87' the convectively neutral sounding above (no CAPE). */
export type TcSounding = 'unstable' | 're87';

/**
 * Conditionally unstable tropical sounding (tropicalProfile with cool = 3 K and a mixed sub-cloud layer): a free
 * troposphere up to 3 K cooler than the surface air's moist adiabat and a well-mixed, unsaturated boundary layer.
 * The neutral sounding's relative-humidity profile leaves the lowest model level about 2 g/kg drier than the surface
 * air, so a lifted parcel is colder than its surroundings (CAPE 0) and only the storm's strong surface fluxes can build
 * deep convection: an eyewall but no rainbands or outer convection. Here the surface-parcel CAPE from the lowest level of
 * a 500 m grid is about 1000 J/kg at 28 °C, as over tropical oceans (roughly 1000-2000 J/kg).
 */
export function unstableTropicalSounding(sst = 301.15, cool = 3): (z: number) => { theta: number; qv: number } {
  return tropicalProfile(sst, 200, 0.4, cool, true);
}

/** The sounding of the tropical-cyclone experiments. */
export function tcSounding(kind: TcSounding, sst: number, rhTop = 0.4): (z: number) => { theta: number; qv: number } {
  return kind === 'unstable' ? unstableTropicalSounding(sst) : tropicalSounding(sst, 200, rhTop);
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
export function insertVortex(m: RegionalModel, f: number, vmax = 15, zTop = 20000, noiseK = 0.1, seed = 7): void {
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
  // random theta perturbations in the lowest 1 km break the grid's 4-fold symmetry
  const r = rng(seed);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const q = m.idx(i, j, k);
    if (m.scalars[QV]) m.scalars[QV]![q] = m.qv0[k]!;
    if (m.zc[k]! < 1000) m.th[q] = m.th[q]! + noiseK * (2 * r() - 1);
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

/**
 * Eyewall diagnostics (measurement only): azimuthal-mean tangential wind vt(r) around the surface
 * pressure minimum at height z, lightly smoothed, and its local maxima. Two maxima separated by a
 * minimum ("moat") at least 10 % below the weaker one indicate concentric eyewalls; an inner maximum
 * decaying while the outer one strengthens is the signature of an eyewall replacement cycle.
 */
export function eyewallProfile(m: RegionalModel, z = 1500, centre?: { ic: number; jc: number }): { r: number[]; vt: number[]; peaks: { r: number; v: number }[]; concentric: boolean } {
  const { nx, ny, dx, nz } = m.c, sx = m.sx;
  const { ic, jc } = centre ?? tcMetrics(m);
  let k = 0; for (let kk = 0; kk < nz; kk++) if (Math.abs(m.zc[kk]! - z) < Math.abs(m.zc[k]! - z)) k = kk;
  const nb = Math.floor(Math.min(nx, ny) / 2), sum = new Float64Array(nb), cnt = new Float64Array(nb);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const x = (i - ic) * dx, y = (j - jc) * dx, r = Math.hypot(x, y);
    const b = Math.floor(r / dx);
    if (b >= nb || r === 0) continue;
    const q = m.idx(i, j, k);
    const ua = 0.5 * (m.u[q]! + m.u[q + 1]!), va = 0.5 * (m.v[q]! + m.v[q + sx]!);
    sum[b] = sum[b]! + (-ua * y + va * x) / r; cnt[b] = cnt[b]! + 1;
  }
  const raw = Array.from(sum, (s, b) => (cnt[b]! > 0 ? s / cnt[b]! : 0));
  const vt = raw.map((_, b) => (raw[Math.max(0, b - 1)]! + 2 * raw[b]! + raw[Math.min(nb - 1, b + 1)]!) / 4);
  const r = vt.map((_, b) => (b + 0.5) * dx);
  const peaks = eyewallPeaks(r, vt);
  return { r, vt, peaks, concentric: peaks.length >= 2 };
}

/** Local maxima of a (smoothed) tangential-wind profile vt(r) above 10 m/s and half the maximum, keeping
 *  only maxima separated by a moat at least 10 % below the weaker neighbouring peak. */
export function eyewallPeaks(r: ArrayLike<number>, vt: ArrayLike<number>): { r: number; v: number }[] {
  const nb = vt.length;
  let vmax = -Infinity; for (let b = 0; b < nb; b++) vmax = Math.max(vmax, vt[b]!);
  const peaks: { r: number; v: number; b: number }[] = [];
  for (let b = 1; b < nb - 1; b++) if (vt[b]! > vt[b - 1]! && vt[b]! >= vt[b + 1]! && vt[b]! > 0.5 * vmax && vt[b]! > 10) peaks.push({ r: r[b]!, v: vt[b]!, b });
  const kept: typeof peaks = [];
  for (const p of peaks) {
    const last = kept[kept.length - 1];
    if (!last) { kept.push(p); continue; }
    let moat = Infinity; for (let b = last.b; b <= p.b; b++) moat = Math.min(moat, vt[b]!);
    if (moat < 0.9 * Math.min(last.v, p.v)) kept.push(p);
    else if (p.v > last.v) kept[kept.length - 1] = p;
  }
  return kept.map(({ r: rr, v }) => ({ r: rr, v }));
}
