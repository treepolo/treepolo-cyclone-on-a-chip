// Moisture transport and moist-physics regressions.
import { DRY_AIR, EARTH, DAY } from '../core/constants.js';
import { SemiLagrangian } from '../model/semiLagrangian.js';
import { SpectralTransform } from '../spectral/transform.js';
import { buildSigmaLevels } from '../model/vertical.js';
import { FRIERSON_SIGMA_HALF, createAquaplanet, AQUA_PRESETS } from '../model/presets.js';
import { sbmColumn, sbmWork } from '../model/moist/sbm.js';
import { GrayAquaplanet } from '../model/moist/aquaplanet.js';
import { MOIST, qsat } from '../model/moist/thermo.js';
import { check, summary, rng } from './assert.js';

const tr = new SpectralTransform(21);
const lev = buildSigmaLevels(FRIERSON_SIGMA_HALF, DRY_AIR, 300);
const K = lev.K, ng = tr.gridSize, nlon = tr.nlon, nlat = tr.nlat;
const sl = new SemiLagrangian({ nlat, nlon, K, lat: tr.lat, lon: tr.lon, sigma: lev.sigma, radius: EARTH.radius });

// 1. uniform tracer stays exactly uniform under an arbitrary smooth 3-D flow
{
  const u = new Float64Array(ng * K), v = new Float64Array(ng * K), w = new Float64Array(ng * K);
  for (let k = 0; k < K; k++) for (let j = 0; j < nlat; j++) for (let i = 0; i < nlon; i++) {
    const q = k * ng + j * nlon + i, la = tr.lat[j]!, lo = tr.lon[i]!;
    u[q] = 40 * Math.cos(la) + 15 * Math.sin(3 * lo) * Math.cos(la);
    v[q] = 10 * Math.cos(2 * lo) * Math.cos(la);
    w[q] = 2e-5 * Math.sin(lo) * Math.sin(Math.PI * lev.sigma[k]!);
  }
  const src = new Float64Array(ng * K).fill(0.0123), dst = new Float64Array(ng * K);
  sl.advect(u, v, w, 3600, [src], [dst]);
  let e = 0;
  for (const x of dst) e = Math.max(e, Math.abs(x - 0.0123));
  check('semi-Lagrangian: uniform tracer preserved', e < 1e-15, e);
}

// 2. solid-body rotation over the poles: cosine bell, one revolution (12 days)
{
  const u0 = 2 * Math.PI * EARTH.radius / (12 * DAY), alpha = Math.PI / 2 - 0.05; // nearly pole-crossing
  const u = new Float64Array(ng * K), v = new Float64Array(ng * K), w = new Float64Array(ng * K);
  const bell = (la: number, lo: number): number => {
    const lc = 0, pc = 3 * Math.PI / 2, R = EARTH.radius / 3;
    const r = EARTH.radius * Math.acos(Math.max(-1, Math.min(1, Math.sin(lc) * Math.sin(la) + Math.cos(lc) * Math.cos(la) * Math.cos(lo - pc))));
    return r < R ? 0.5 * (1 + Math.cos(Math.PI * r / R)) : 0;
  };
  const q0 = new Float64Array(ng * K);
  for (let k = 0; k < K; k++) for (let j = 0; j < nlat; j++) for (let i = 0; i < nlon; i++) {
    const q = k * ng + j * nlon + i, la = tr.lat[j]!, lo = tr.lon[i]!;
    u[q] = u0 * (Math.cos(la) * Math.cos(alpha) + Math.sin(la) * Math.cos(lo) * Math.sin(alpha));
    v[q] = -u0 * Math.sin(lo) * Math.sin(alpha);
    q0[q] = bell(la, lo);
  }
  let a = Float64Array.from(q0), b = new Float64Array(ng * K);
  const dt = 1800, steps = Math.round(12 * DAY / dt);
  for (let s = 0; s < steps; s++) { sl.advect(u, v, w, dt, [a], [b]); const t = a; a = b; b = t; }
  let l2n = 0, l2d = 0, mx = -Infinity, mn = Infinity, m0 = 0, m1 = 0;
  const k = K - 1;
  for (let j = 0; j < nlat; j++) for (let i = 0; i < nlon; i++) {
    const q = k * ng + j * nlon + i;
    l2n += tr.weight[j]! * (a[q]! - q0[q]!) ** 2; l2d += tr.weight[j]! * q0[q]! ** 2;
    mx = Math.max(mx, a[q]!); mn = Math.min(mn, a[q]!);
    m0 += tr.weight[j]! * q0[q]!; m1 += tr.weight[j]! * a[q]!;
  }
  const l2 = Math.sqrt(l2n / l2d);
  check('semi-Lagrangian: no new extrema after one pole-crossing revolution (min >= 0, max <= 1)', mn >= 0 && mx <= 1 + 1e-12, `min ${mn.toExponential(2)} max ${mx.toFixed(4)}`);
  // cubic SL is diffusive on the coarse T21 grid (bell ~7 cells wide, 576 interpolations); regression bound only
  check('semi-Lagrangian: cosine-bell relative L2 error < 0.65 at T21 after 12 days (regression bound)', l2 < 0.65, l2);
  console.log(`  info: cosine bell T21 L2=${l2.toFixed(3)}  peak=${mx.toFixed(3)}  mass ratio (before fixer)=${(m1 / m0).toFixed(4)}`);
}

// 3. SBM conserves enthalpy and closes the water budget in a convectively unstable column
{
  const ps = 1e5, pf = new Float64Array(K), ph = new Float64Array(K + 1);
  for (let k = 0; k <= K; k++) ph[k] = FRIERSON_SIGMA_HALF[k]! * ps;
  for (let k = 0; k < K; k++) pf[k] = lev.sigma[k]! * ps;
  const T = new Float64Array(K), q = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    T[k] = Math.max(200, 302 * Math.pow(lev.sigma[k]!, 0.19));
    q[k] = (lev.sigma[k]! > 0.85 ? 0.9 : 0.4) * qsat(T[k]!, pf[k]!);
  }
  // put the free troposphere 1.5 K below its own moist adiabat (near convective quasi-equilibrium)
  const w = sbmWork(K);
  sbmColumn(Float64Array.from(T), Float64Array.from(q), pf, ph, 900, EARTH.gravity, w);
  for (let k = 0; k < K; k++) if (pf[k]! > 15000 && pf[k]! < 85000) { T[k] = w.Tp[k]! - 1.5; q[k] = 0.8 * qsat(T[k]!, pf[k]!); }
  const T0 = Float64Array.from(T), q0 = Float64Array.from(q);
  const res = sbmColumn(T, q, pf, ph, 900, EARTH.gravity, w);
  let dH = 0, dW = 0;
  for (let k = 0; k < K; k++) {
    const m = (ph[k + 1]! - ph[k]!) / EARTH.gravity;
    dH += (DRY_AIR.cp * (T[k]! - T0[k]!) + MOIST.Lv * (q[k]! - q0[k]!)) * m;
    dW += (q[k]! - q0[k]!) * m;
  }
  check('SBM: convection triggers in a moist unstable column', res.rain > 0 && res.cape > 0, `rain ${res.rain.toExponential(2)} kg/m2, CAPE ${res.cape.toFixed(0)} J/kg`);
  check('SBM: column moist enthalpy conserved (cp dT + L dq = 0)', Math.abs(dH) < 1e-6 * MOIST.Lv * res.rain + 1e-6, dH);
  check('SBM: water removed equals precipitation', Math.abs(-dW - res.rain) < 1e-12, -dW - res.rain);
}

// 4. aquaplanet column physics: water budget per column dq_column = (E - P) dt exactly
{
  const phys = new GrayAquaplanet(EARTH, DRY_AIR, nlat, nlon, K, tr.lat);
  const r = rng(9), nK = ng * K;
  const st = { nlat, nlon, K, mu: tr.mu, sigma: lev.sigma, sigmaHalf: lev.sigmaHalf,
    u: new Float64Array(nK), v: new Float64Array(nK), T: new Float64Array(nK), q: new Float64Array(nK), ps: new Float64Array(ng) };
  for (let p = 0; p < ng; p++) st.ps[p] = 1e5 * (0.98 + 0.04 * r());
  for (let k = 0; k < K; k++) for (let p = 0; p < ng; p++) {
    const q = k * ng + p, j = Math.floor(p / nlon), sig = lev.sigma[k]!;
    const ts = phys.f.sst[p]!;
    st.T[q] = Math.max(200, (ts - 2) * Math.pow(sig, 0.19)) + (r() - 0.5);
    st.q[q] = (0.3 + 0.8 * r()) * qsat(st.T[q]!, sig * st.ps[p]!);
    st.u[q] = 10 * Math.cos(tr.lat[j]!) * (1 - sig) + 3 * (r() - 0.5); st.v[q] = 3 * (r() - 0.5);
  }
  const q0 = Float64Array.from(st.q);
  phys.resetAccumulators();
  phys.negativeWater = 0;
  phys.apply(st, 900);
  let worst = 0;
  for (let p = 0; p < ng; p++) {
    let dW = 0;
    for (let k = 0; k < K; k++) dW += (st.q[k * ng + p]! - q0[k * ng + p]!) * lev.dsigma[k]! * st.ps[p]! / EARTH.gravity;
    const budget = phys.f.evap[p]! - phys.f.precipConv[p]! - phys.f.precipLS[p]!;
    worst = Math.max(worst, Math.abs(dW - budget));
  }
  check('aquaplanet physics: per-column water budget closes (|dW - (E-P)dt| < 1e-9 kg/m2)', worst < 1e-9 && phys.negativeWater < 1e-9, worst);
}

// 5. short coupled run stays finite and moist
{
  const { model } = createAquaplanet(AQUA_PRESETS.AQUA_T21!);
  for (let s = 0; s < 72 * 5; s++) model.step();
  let qmin = Infinity;
  for (const x of model.q) qmin = Math.min(qmin, x);
  check('aquaplanet T21: 5 days finite, q >= 0', model.isFinite() && qmin >= 0, qmin);
}

summary('moist');
