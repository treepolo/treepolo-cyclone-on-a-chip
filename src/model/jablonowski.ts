// Jablonowski & Williamson (2006) baroclinic-instability test (QJRMS 132, 2943–2975).
// A steady, balanced, zonally symmetric midlatitude jet; an optional 1 m/s Gaussian bump in u
// triggers a baroclinic wave that develops into extratropical cyclones with fronts by day 8–9.
// With ps = 1000 hPa initially, eta = sigma, so the analytic state applies directly in sigma.

import { DRY_AIR, EARTH } from '../core/constants.js';
import { Dycore } from './dycore.js';
import { uniformSigmaHalf } from './vertical.js';

const JW = { eta0: 0.252, etaT: 0.2, u0: 35, deltaT: 4.8e5, gamma: 0.005, T0: 288, up: 1, lonC: Math.PI / 9, latC: 2 * Math.PI / 9 };

export interface JwConfig { trunc: number; levels: number; dt: number; perturb: boolean }

export function jwState(lat: number, lon: number, eta: number, perturb: boolean): { u: number; T: number } {
  const { eta0, etaT, u0, deltaT, gamma, T0 } = JW;
  const a = EARTH.radius, om = EARTH.omega, rd = DRY_AIR.rd, g = EARTH.gravity;
  const ev = (eta - eta0) * Math.PI / 2;
  const s = Math.sin(lat), c = Math.cos(lat);
  let u = u0 * Math.pow(Math.cos(ev), 1.5) * Math.sin(2 * lat) ** 2;
  if (perturb) {
    const r = a * Math.acos(Math.max(-1, Math.min(1, Math.sin(JW.latC) * s + Math.cos(JW.latC) * c * Math.cos(lon - JW.lonC))));
    u += JW.up * Math.exp(-((r / (a / 10)) ** 2));
  }
  let Tbar = T0 * Math.pow(eta, rd * gamma / g);
  if (eta < etaT) Tbar += deltaT * Math.pow(etaT - eta, 5);
  const A = (-2 * s ** 6 * (c * c + 1 / 3) + 10 / 63) * 2 * u0 * Math.pow(Math.cos(ev), 1.5);
  const B = (8 / 5 * c ** 3 * (s * s + 2 / 3) - Math.PI / 4) * a * om;
  const T = Tbar + 0.75 * eta * Math.PI * u0 / rd * Math.sin(ev) * Math.sqrt(Math.cos(ev)) * (A + B);
  return { u, T };
}

export function jwSurfaceGeopotential(lat: number): number {
  const { eta0, u0 } = JW;
  const a = EARTH.radius, om = EARTH.omega;
  const s = Math.sin(lat), c = Math.cos(lat);
  const cv = Math.pow(Math.cos((1 - eta0) * Math.PI / 2), 1.5);
  return u0 * cv * ((-2 * s ** 6 * (c * c + 1 / 3) + 10 / 63) * u0 * cv + (8 / 5 * c ** 3 * (s * s + 2 / 3) - Math.PI / 4) * a * om);
}

export function createJablonowski(cfg: JwConfig): Dycore {
  const model = new Dycore({
    trunc: cfg.trunc, sigmaHalf: uniformSigmaHalf(cfg.levels), dt: cfg.dt,
    planet: EARTH, air: DRY_AIR, tRef: 300, hyperdiffTau: cfg.trunc <= 42 ? 0.1 * 86400 : 0.05 * 86400,
  });
  const tr = model.tr, ng = model.ng, K = model.K, nlon = tr.nlon;
  const T = new Float64Array(ng * K), u = new Float64Array(ng * K), v = new Float64Array(ng * K);
  const ps = new Float64Array(ng).fill(1e5), phis = new Float64Array(ng);
  for (let j = 0; j < tr.nlat; j++) for (let i = 0; i < nlon; i++) {
    const p = j * nlon + i;
    phis[p] = jwSurfaceGeopotential(tr.lat[j]!);
    for (let k = 0; k < K; k++) {
      const st = jwState(tr.lat[j]!, tr.lon[i]!, model.lev.sigma[k]!, cfg.perturb);
      T[k * ng + p] = st.T; u[k * ng + p] = st.u;
    }
  }
  model.setSurfaceGeopotential(phis);
  model.setFromGrid(T, ps, u, v);
  return model;
}
