// Dynamical-core regressions.
import { DAY, DRY_AIR, EARTH } from '../core/constants.js';
import { Dycore } from '../model/dycore.js';
import { uniformSigmaHalf } from '../model/vertical.js';
import { check, summary, rng } from './assert.js';
import { createJablonowski } from '../model/jablonowski.js';

function makeModel(trunc = 21, K = 10, dt = 1800): Dycore {
  return new Dycore({ trunc, sigmaHalf: uniformSigmaHalf(K), dt, planet: EARTH, air: DRY_AIR, hyperdiffTau: 1e30, robert: 0.02 });
}

/** Total energy integral (J / m^2 global mean): (c_p T + K) dp/g. */
function totalEnergy(m: Dycore): { E: number; M: number; AAM: number } {
  const g = m.refreshGrid(), tr = m.tr, ng = m.ng, lev = m.lev;
  let E = 0, M = 0, A = 0;
  for (let j = 0; j < tr.nlat; j++) {
    let e = 0, ms = 0, am = 0;
    const cs = tr.coslat[j]!, a = EARTH.radius;
    for (let i = 0; i < tr.nlon; i++) {
      const p = j * tr.nlon + i, ps = g.ps[p]!;
      ms += ps;
      for (let k = 0; k < m.K; k++) {
        const q = k * ng + p, dp = lev.dsigma[k]! * ps;
        e += (DRY_AIR.cp * g.T[q]! + 0.5 * (g.u[q]! ** 2 + g.v[q]! ** 2)) * dp;
        am += (g.u[q]! + EARTH.omega * a * cs) * a * cs * dp;
      }
    }
    E += tr.weight[j]! * e / tr.nlon; M += tr.weight[j]! * ms / tr.nlon; A += tr.weight[j]! * am / tr.nlon;
  }
  return { E: E / 2 / EARTH.gravity, M: M / 2 / EARTH.gravity, AAM: A / 2 / EARTH.gravity };
}

// 1. Resting stratified atmosphere with uniform surface pressure stays at rest.
{
  const m = makeModel();
  const ng = m.ng, T = new Float64Array(ng * m.K), ps = new Float64Array(ng).fill(1e5);
  for (let k = 0; k < m.K; k++) T.fill(200 + 90 * m.lev.sigma[k]!, k * ng, (k + 1) * ng);
  m.setFromGrid(T, ps);
  for (let s = 0; s < 480; s++) m.step();
  const g = m.refreshGrid();
  let umax = 0;
  for (let q = 0; q < g.u.length; q++) umax = Math.max(umax, Math.abs(g.u[q]!), Math.abs(g.v[q]!));
  check('stratified resting atmosphere stays at rest for 10 days', umax < 1e-9, umax);
}

// 2. Unforced baroclinic flow: conservation of mass, total energy and axial angular momentum.
{
  const m = makeModel(21, 10, 1200);
  const tr = m.tr, ng = m.ng, K = m.K, r = rng(4);
  const T = new Float64Array(ng * K), ps = new Float64Array(ng).fill(1e5);
  const u = new Float64Array(ng * K), v = new Float64Array(ng * K);
  for (let k = 0; k < K; k++) for (let j = 0; j < tr.nlat; j++) for (let i = 0; i < tr.nlon; i++) {
    const q = k * ng + j * tr.nlon + i, mu = tr.mu[j]!, sig = m.lev.sigma[k]!;
    T[q] = 230 + 60 * sig - 40 * mu * mu * sig + 0.5 * (r() - 0.5);
    u[q] = 30 * (1 - sig) * (1 - mu * mu) * Math.sin(Math.PI * Math.abs(mu)) ** 2;
  }
  m.setFromGrid(T, ps, u, v);
  const e0 = totalEnergy(m);
  let finite = true;
  for (let s = 0; s < 20 * 72; s++) { m.step(); if (s % 72 === 0 && !m.isFinite()) { finite = false; break; } }
  const e1 = totalEnergy(m);
  check('unforced run stays finite for 20 days', finite);
  const dM = Math.abs(e1.M - e0.M) / e0.M, dE = Math.abs(e1.E - e0.E) / e0.E, dA = Math.abs(e1.AAM - e0.AAM) / e0.AAM;
  check('dry mass drift over 20 days < 1e-4', dM < 1e-4, dM);
  check('total energy drift over 20 days < 1e-4', dE < 1e-4, dE);
  check('axial angular momentum drift over 20 days < 1e-3', dA < 1e-3, dA);
  console.log(`  info: 20 days  dE/E=${dE.toExponential(2)}  dM/M=${dM.toExponential(2)}  dAAM/AAM=${dA.toExponential(2)}`);
}

// 3. Semi-implicit gravity-wave stability at a long timestep (external-mode CFL >> 1).
{
  const m = makeModel(42, 10, 3600);
  const ng = m.ng, T = new Float64Array(ng * m.K), ps = new Float64Array(ng);
  const tr = m.tr;
  for (let k = 0; k < m.K; k++) T.fill(250, k * ng, (k + 1) * ng);
  for (let j = 0; j < tr.nlat; j++) for (let i = 0; i < tr.nlon; i++) {
    const lat = tr.lat[j]!, lon = tr.lon[i]!;
    const d = Math.acos(Math.cos(lat) * Math.cos(lon - Math.PI));
    ps[j * tr.nlon + i] = 1e5 + 500 * Math.exp(-((d / 0.2) ** 2));
  }
  m.setFromGrid(T, ps);
  for (let s = 0; s < 5 * 24; s++) m.step();
  check('semi-implicit: 1 h timestep with pressure bump stays finite (5 days)', m.isFinite());
}

// 4. Jablonowski–Williamson balanced steady state (with surface geopotential) stays steady.
{
  const m = createJablonowski({ trunc: 21, levels: 15, dt: 1800, perturb: false });
  for (let s = 0; s < 5 * 48; s++) m.step();
  const g = m.refreshGrid();
  let dev = 0;
  for (let q = 0; q < g.ps.length; q++) dev = Math.max(dev, Math.abs(g.ps[q]! - 1e5));
  check('JW steady state: |ps - 1000 hPa| < 0.5 hPa after 5 days', dev < 50, dev / 100);
}

void DAY;
summary('dycore');
