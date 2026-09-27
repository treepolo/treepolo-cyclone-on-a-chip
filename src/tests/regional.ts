// Regional compressible non-hydrostatic core regressions (dry).
import { DRY_AIR, EARTH } from '../core/constants.js';
import { RegionalModel, RegionalConfig } from '../regional/core.js';
import { check, summary } from './assert.js';
import { KesslerMicrophysics, weismanKlemp, QV, QC, QR } from '../regional/kessler.js';

const base: RegionalConfig = { nx: 64, ny: 1, nz: 32, dx: 200, dy: 200, dz: 200, dt: 1, nsound: 6, f: 0, beta: 0.2, divDamp: 0.1, dampDepth: 0, dampRate: 0, kdiff2: 0 };

// 1. stratified resting atmosphere stays at rest
{
  const N2 = 1e-4;
  const m = new RegionalModel({ ...base, nx: 16, ny: 16, nz: 40, dx: 1000, dy: 1000, dz: 400, dt: 6, nsound: 6, dampDepth: 4000, dampRate: 0.01 },
    (z) => ({ theta: 300 * Math.exp(N2 * z / EARTH.gravity), qv: 0 }));
  for (let s = 0; s < 300; s++) m.step();
  const wmax = m.maxAbs(m.w, m.c.nz + 1), umax = m.maxAbs(m.u);
  check('regional: stratified rest stays at rest (30 min)', wmax < 1e-10 && umax < 1e-10, Math.max(wmax, umax));
}

// 2. Straka et al. (1993) density current: 900 s, dx = dz = 100 m, nu = 75 m^2/s
{
  const nx = 512, nz = 64, dx = 100;
  const m = new RegionalModel({ ...base, nx, ny: 1, nz, dx, dy: dx, dz: dx, dt: 0.5, nsound: 6, kdiff2: 75 }, () => ({ theta: 300, qv: 0 }));
  const x0 = -nx * dx / 2;
  for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) {
    const x = x0 + (i + 0.5) * dx, z = m.zc[k]!;
    const L = Math.sqrt(((x - 0) / 4000) ** 2 + ((z - 3000) / 2000) ** 2);
    if (L <= 1) {
      const dT = -7.5 * (Math.cos(Math.PI * L) + 1);
      m.th[m.idx(i, 0, k)] = 300 + dT / m.pi0[k]!;
    }
  }
  const t0 = Date.now();
  while (m.time < 900 - 1e-9) m.step();
  let thMin = Infinity, front = 0, umax = 0, wmin = 0, wmax = 0;
  for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) thMin = Math.min(thMin, m.th[m.idx(i, 0, k)]! - 300);
  // front: rightmost surface point with theta' < -1 K
  for (let i = nx / 2; i < nx; i++) if (m.th[m.idx(i, 0, 0)]! - 300 < -1) front = x0 + (i + 0.5) * dx;
  umax = m.maxAbs(m.u);
  for (let k = 0; k <= nz; k++) for (let i = 0; i < nx; i++) { const w = m.w[m.idx(i, 0, k)]!; wmin = Math.min(wmin, w); wmax = Math.max(wmax, w); }
  // symmetry about x = 0
  let asym = 0;
  for (let k = 0; k < nz; k++) for (let i = 0; i < nx / 2; i++) asym = Math.max(asym, Math.abs(m.th[m.idx(i, 0, k)]! - m.th[m.idx(nx - 1 - i, 0, k)]!));
  console.log(`  info: Straka 900 s: theta'min ${thMin.toFixed(2)} K, front ${(front / 1000).toFixed(2)} km, max|u| ${umax.toFixed(2)}, w ${wmin.toFixed(2)}..${wmax.toFixed(2)} m/s, asym ${asym.toExponential(2)}, wall ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  // reference (Straka et al. 1993, 100 m): theta'min -9.77 K, front 15.54 km, w about -16 .. +13 m/s
  check('Straka density current: theta\' min within 0.5 K of -9.77 K', Math.abs(thMin + 9.77) < 0.5, thMin);
  check('Straka density current: front position within 0.5 km of 15.54 km', Math.abs(front / 1000 - 15.54) < 0.5, front / 1000);
  check('Straka density current: w extremes within 15% of (-16, +13) m/s', Math.abs(wmin + 16) < 2.4 && Math.abs(wmax - 13) < 2.0, `${wmin.toFixed(2)}..${wmax.toFixed(2)}`);
  check('Straka density current: left-right symmetric', asym < 1e-6, asym);
}

// 3. Moist convection (Weisman–Klemp sounding, Kessler): a warm bubble grows into a deep precipitating
//    updraft within 25 min, and total water (vapour + condensate + surface rain) is conserved.
{
  const nx = 24, nz = 40, dx = 3000, dz = 500;
  const m = new RegionalModel({ ...base, nx, ny: nx, nz, dx, dy: dx, dz, dt: 6, nsound: 6, dampDepth: 5000, dampRate: 1 / 300 }, weismanKlemp, 3);
  const mp = new KesslerMicrophysics(m);
  for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) m.scalars[QV]![m.idx(i, j, k)] = m.qv0[k]!;
  const c = nx * dx / 2;
  for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
    const r = Math.sqrt((((i + 0.5) * dx - c) / 10000) ** 2 + (((j + 0.5) * dx - c) / 10000) ** 2 + ((m.zc[k]! - 1400) / 1400) ** 2);
    if (r < 1) m.th[m.idx(i, j, k)] = m.th[m.idx(i, j, k)]! + 2 * Math.cos(0.5 * Math.PI * r) ** 2;
  }
  const water = (): number => {
    let t = 0;
    for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
      const q = m.idx(i, j, k);
      t += (m.scalars[QV]![q]! + m.scalars[QC]![q]! + m.scalars[QR]![q]!) * m.rho0[k]! * dz;
    }
    for (const r of mp.rainAcc) t += r;
    return t + mp.clipped * 0;
  };
  const W0 = water();
  let wmax = 0, qrmax = 0;
  while (m.time < 25 * 60 - 1e-9) {
    m.step(); mp.apply(6);
    wmax = Math.max(wmax, m.maxAbs(m.w, nz + 1));
    qrmax = Math.max(qrmax, m.maxAbs(m.scalars[QR]!));
  }
  const drift = Math.abs(water() - W0) / W0;
  check('moist regional: bubble grows into a deep updraft (> 15 m/s) with rain (> 1 g/kg)', wmax > 15 && qrmax > 1e-3, `w ${wmax.toFixed(1)} m/s, qr ${(qrmax * 1000).toFixed(2)} g/kg`);
  check('moist regional: total water conserved within 0.5% (clipping included)', drift < 5e-3, drift);
}

void DRY_AIR;
summary('regional');
