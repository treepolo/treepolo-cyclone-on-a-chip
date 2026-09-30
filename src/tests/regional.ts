// Regional compressible non-hydrostatic core regressions (dry).
import { DRY_AIR, EARTH } from '../core/constants.js';
import { RegionalModel, RegionalConfig } from '../regional/core.js';
import { check, summary } from './assert.js';
import { KesslerMicrophysics, weismanKlemp, QV, QC, QR } from '../regional/kessler.js';
import { IceMicrophysics, cellProcesses, ICE, QI, QS, QG } from '../regional/ice.js';
import { eyewallProfile, tropicalSounding, insertVortex } from '../regional/tropical.js';
import { refineInto } from '../regional/refine.js';

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

// 4. Ice microphysics, local processes: exact water conservation and conservation of the
//    liquid-ice potential temperature theta - (Lv q_liq + Ls q_ice) / (cp pi) over mixed-phase states
{
  let worstW = 0, worstH = 0;
  const cp = DRY_AIR.cp, LF = ICE.LS - ICE.LV;
  for (let n = 0; n < 400; n++) {
    const T = 225 + (n % 40) * 1.5, pi = 0.6 + 0.4 * ((n * 7) % 13) / 13, p = 1e5 * Math.pow(pi, cp / DRY_AIR.rd);
    const s = { th: T / pi, qv: 0.9e-3 * Math.exp(0.06 * (T - 273)) * (0.8 + 0.4 * ((n * 3) % 5) / 5), qc: 5e-4 * ((n * 5) % 3), qr: 1e-3 * ((n * 11) % 4) / 3,
      qi: 3e-4 * ((n * 13) % 3), qs: 1e-3 * ((n * 17) % 3), qg: 1e-3 * ((n * 19) % 3) / 2 };
    const w0 = s.qv + s.qc + s.qr + s.qi + s.qs + s.qg;
    const h0 = s.th - (ICE.LV * (s.qc + s.qr) + ICE.LS * (s.qi + s.qs + s.qg)) / (cp * pi) - ICE.LS * s.qv * 0;
    const hv = (st: typeof s): number => st.th - (ICE.LV * (st.qc + st.qr) + ICE.LS * (st.qi + st.qs + st.qg)) / (cp * pi);
    cellProcesses(s, p / (DRY_AIR.rd * T), 1.1, pi, p, 20);
    worstW = Math.max(worstW, Math.abs(s.qv + s.qc + s.qr + s.qi + s.qs + s.qg - w0) / w0);
    worstH = Math.max(worstH, Math.abs(hv(s) - h0));
    void LF;
  }
  check('ice microphysics: water conserved by local processes (rel < 1e-12)', worstW < 1e-12, worstW);
  check('ice microphysics: liquid-ice potential temperature conserved (< 1e-9 K)', worstH < 1e-9, worstH);
}

// 5. Ice microphysics reduces to Kessler in a warm atmosphere (all T > 0 °C)
{
  const warm = (z: number): { theta: number; qv: number } => ({ theta: 300 + 3e-3 * z, qv: 0.016 * Math.exp(-z / 2500) });
  const cfg: RegionalConfig = { ...base, nx: 10, ny: 10, nz: 12, dx: 2000, dy: 2000, dz: 250, dt: 5, nsound: 6 };
  const mk = (n: number): RegionalModel => {
    const m = new RegionalModel(cfg, warm, n);
    for (let k = 0; k < 12; k++) for (let j = 0; j < 10; j++) for (let i = 0; i < 10; i++) {
      const q = m.idx(i, j, k);
      m.scalars[QV]![q] = m.qv0[k]! * (1.02 + 0.01 * Math.sin(i + 2 * j));
      m.scalars[QC]![q] = 1.5e-3 * Math.max(0, Math.sin(0.5 * i) * Math.cos(0.4 * j));
      m.scalars[QR]![q] = 1e-3 * Math.max(0, Math.cos(0.3 * i + k));
      m.th[q] = m.th[q]! + 0.5 * Math.sin(0.7 * i + 0.3 * k);
    }
    return m;
  };
  const a = mk(3), b = mk(6);
  const ka = new KesslerMicrophysics(a), kb = new IceMicrophysics(b);
  for (let s = 0; s < 20; s++) { ka.apply(5); kb.apply(5); }
  let d = 0;
  for (let q = 0; q < a.size; q++) d = Math.max(d, Math.abs(a.th[q]! - b.th[q]!), Math.abs(a.scalars[QR]![q]! - b.scalars[QR]![q]!) * 1e3, Math.abs(a.scalars[QC]![q]! - b.scalars[QC]![q]!) * 1e3);
  let dr = 0; for (let c = 0; c < ka.rainAcc.length; c++) dr = Math.max(dr, Math.abs(ka.rainAcc[c]! - kb.rainAcc[c]!));
  check('ice microphysics: identical to Kessler above 0 °C (theta, qc, qr, surface rain)', d < 1e-12 && dr < 1e-12, Math.max(d, dr));
}

// 6. Deep convection with ice (Weisman–Klemp): ice, snow and graupel form aloft, no liquid below -40 °C,
//    rain and graupel/snow reach the ground, total water conserved
{
  const nx = 24, nz = 40, dx = 3000, dz = 500;
  const m = new RegionalModel({ ...base, nx, ny: nx, nz, dx, dy: dx, dz, dt: 6, nsound: 6, dampDepth: 5000, dampRate: 1 / 300 }, weismanKlemp, 6);
  const mp = new IceMicrophysics(m);
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
      for (let s = 0; s < 6; s++) t += m.scalars[s]![q]! * m.rho0[k]! * dz;
    }
    for (const r of mp.rainAcc) t += r;
    return t;
  };
  const W0 = water();
  let wmax = 0, qimax = 0, qsmax = 0, qgmax = 0, liqCold = 0;
  while (m.time < 40 * 60 - 1e-9) {
    m.step(); mp.apply(6);
    wmax = Math.max(wmax, m.maxAbs(m.w, nz + 1));
    qimax = Math.max(qimax, m.maxAbs(m.scalars[QI]!)); qsmax = Math.max(qsmax, m.maxAbs(m.scalars[QS]!)); qgmax = Math.max(qgmax, m.maxAbs(m.scalars[QG]!));
  }
  for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
    const q = m.idx(i, j, k), T = m.th[q]! * (m.pi0[k]! + m.pp[q]!);
    if (T < 233.15) liqCold = Math.max(liqCold, m.scalars[QC]![q]! + m.scalars[QR]![q]!);
  }
  let rain = 0, frozen = 0; for (let n = 0; n < mp.rainAcc.length; n++) { rain += mp.rainAcc[n]!; frozen += mp.snowAcc[n]!; }
  const drift = Math.abs(water() - W0) / W0;
  if (process.env.VERBOSE) console.log(`    water drift ${(water() - W0) / W0}, clipped ${mp.clipped / W0}`);
  check('ice regional: deep updraft with cloud ice, snow and graupel aloft', wmax > 15 && qimax > 1e-4 && qsmax > 1e-4 && qgmax > 1e-4, `w ${wmax.toFixed(1)} m/s, qi ${(qimax * 1e3).toFixed(2)} qs ${(qsmax * 1e3).toFixed(2)} qg ${(qgmax * 1e3).toFixed(2)} g/kg`);
  check('ice regional: no liquid water colder than -40 °C', liqCold < 1e-9, liqCold);
  check('ice regional: precipitation reaches the ground', rain > 0, `total ${(rain / (nx * nx)).toFixed(2)} mm mean, frozen part ${(100 * frozen / Math.max(rain, 1e-30)).toFixed(0)}%`);
  check('ice regional: total water conserved within 0.5%', drift < 5e-3, drift);
}

// 7. Eyewall diagnostic: single and concentric wind maxima of a prescribed vortex are recognised
{
  const cfg: RegionalConfig = { ...base, nx: 120, ny: 120, nz: 6, dx: 2000, dy: 2000, dz: 500, dt: 10 };
  const mk = (vt: (r: number) => number): RegionalModel => {
    const m = new RegionalModel(cfg, (z) => ({ theta: 300 + 0.004 * z, qv: 0 }), 0);
    const c = 60 * 2000;
    for (let k = 0; k < 6; k++) for (let j = 0; j < 120; j++) for (let i = 0; i < 120; i++) {
      const q = m.idx(i, j, k);
      const xu = i * 2000 - c + 1000, yu = (j + 0.5) * 2000 - c;           // u point: west face
      const xv = (i + 0.5) * 2000 - c, yv = j * 2000 - c + 1000;           // v point: south face
      const ru = Math.hypot(xu, yu) || 1, rv = Math.hypot(xv, yv) || 1;
      m.u[q] = -vt(ru) * yu / ru; m.v[q] = vt(rv) * xv / rv;
      const xc = (i + 0.5) * 2000 - c - 1000, yc = (j + 0.5) * 2000 - c - 1000;
      m.pp[q] = -1e-3 * Math.exp(-(xc * xc + yc * yc) / (2 * 30000 ** 2));
    }
    return m;
  };
  const single = eyewallProfile(mk((r) => 50 * (r < 25000 ? r / 25000 : Math.pow(25000 / r, 0.6))), 1000);
  const dbl = eyewallProfile(mk((r) => {
    const inner = 45 * Math.exp(-(((r - 20000) / 8000) ** 2)), outer = 40 * Math.exp(-(((r - 70000) / 12000) ** 2));
    return Math.max(inner, outer, 12 * Math.min(1, r / 20000));
  }), 1000);
  check('eyewall diagnostic: single eyewall at ~25 km', single.peaks.length === 1 && !single.concentric && Math.abs(single.peaks[0]!.r - 25000) < 4000, single.peaks.map((p) => (p.r / 1000).toFixed(0)).join(','));
  check('eyewall diagnostic: concentric eyewalls at ~20 and ~70 km', dbl.concentric && Math.abs(dbl.peaks[0]!.r - 20000) < 4000 && Math.abs(dbl.peaks[1]!.r - 70000) < 4000, dbl.peaks.map((p) => (p.r / 1000).toFixed(0)).join(','));
}

// 8. Whole-cell roll commutes with a model step (periodic translation invariance, used by the storm tracker)
{
  const cfg: RegionalConfig = { ...base, nx: 16, ny: 12, nz: 10, dx: 1000, dy: 1000, dz: 500, dt: 4, nsound: 6 };
  const mk = (): RegionalModel => {
    const m = new RegionalModel(cfg, weismanKlemp, 3);
    m.setBaseWind((z) => ({ u: 5 + z / 1000, v: -2 }));
    for (let k = 0; k < 10; k++) for (let j = 0; j < 12; j++) for (let i = 0; i < 16; i++) {
      const q = m.idx(i, j, k);
      m.th[q] = m.th[q]! + Math.exp(-(((i - 5) / 2) ** 2) - (((j - 4) / 2) ** 2) - (((k - 2) / 1.5) ** 2));
      m.scalars[QV]![q] = m.qv0[k]!;
    }
    return m;
  };
  const a = mk(), b = mk();
  a.roll(3, -2); a.step(); a.step();
  b.step(); b.step(); b.roll(3, -2);
  let d = 0;
  for (const [x, y] of [[a.u, b.u], [a.w, b.w], [a.th, b.th], [a.pp, b.pp]] as const) for (let k = 0; k < 10; k++) for (let j = 0; j < 12; j++) for (let i = 0; i < 16; i++) d = Math.max(d, Math.abs(x[a.idx(i, j, k)]! - y[a.idx(i, j, k)]!));
  check('regional: periodic roll commutes with time stepping (exact)', d < 1e-12, d);
}

// coarse-to-fine refinement: (a) linear fields are reproduced exactly at the staggered positions of a
// sub-box; (b) a tropical-cyclone vortex keeps its strength and continues stably on a 3x finer grid
{
  const open: RegionalConfig = { ...base, lateral: 'open', nx: 20, ny: 20, nz: 10, dx: 3000, dy: 3000, dz: 1000, dt: 10, nsound: 6 };
  const mc = new RegionalModel(open, (z) => ({ theta: 300 + 0.004 * z, qv: 0 }), 1);
  const lin = (x: number, y: number, z: number): number => 1 + 2e-5 * x - 3e-5 * y + 1e-4 * z;
  for (let k = 0; k <= 10; k++) for (let j = -2; j < 23; j++) for (let i = -2; i < 23; i++) {
    const q = mc.idx(i, j, k);
    if (q < 0 || q >= mc.size) continue;
    mc.u[q] = lin(i * 3000, (j + 0.5) * 3000, (k + 0.5) * 1000);
    mc.w[q] = lin((i + 0.5) * 3000, (j + 0.5) * 3000, k * 1000);
    mc.scalars[0]![q] = lin((i + 0.5) * 3000, (j + 0.5) * 3000, (k + 0.5) * 1000);
  }
  const mf = new RegionalModel({ ...open, nx: 30, ny: 30, nz: 20, dx: 1000, dy: 1000, dz: 500 }, (z) => ({ theta: 300 + 0.004 * z, qv: 0 }), 1);
  const x0 = 12000, y0 = 15000;
  refineInto(mc, mf, x0, y0);
  let e = 0;
  for (let k = 1; k < 19; k++) for (let j = 1; j < 29; j++) for (let i = 1; i < 29; i++) {
    e = Math.max(e, Math.abs(mf.u[mf.idx(i, j, k)]! - lin(x0 + i * 1000, y0 + (j + 0.5) * 1000, (k + 0.5) * 500)));
    e = Math.max(e, Math.abs(mf.w[mf.idx(i, j, k)]! - lin(x0 + (i + 0.5) * 1000, y0 + (j + 0.5) * 1000, k * 500)));
    e = Math.max(e, Math.abs(mf.scalars[0]![mf.idx(i, j, k)]! - lin(x0 + (i + 0.5) * 1000, y0 + (j + 0.5) * 1000, (k + 0.5) * 500)));
  }
  check('refine: linear fields reproduced exactly at staggered sub-box points', e < 1e-12, e);
  // a sharp-edged rain shaft (coarse cells 8-11 x 8-11, levels 0-5): the fine field keeps the mass (each wet level's
  // within 40 %: the smooth start spreads the shaft's top by a level), stays non-negative and falls off toward the shaft's edge (cube-root interpolation) instead of filling
  // whole coarse cells beside it
  const m2 = new RegionalModel({ ...open, lateral: 'periodic' }, (z) => ({ theta: 300 + 0.004 * z, qv: 0 }), 3);
  for (let k = 0; k < 6; k++) for (let j = 8; j < 12; j++) for (let i = 8; i < 12; i++) m2.scalars[2]![m2.idx(i, j, k)] = 1e-3 * (1 + 0.1 * k);
  const f2 = new RegionalModel({ ...open, lateral: 'periodic', nx: 60, ny: 60, nz: 20, dx: 1000, dy: 1000, dz: 500 }, (z) => ({ theta: 300 + 0.004 * z, qv: 0 }), 3);
  refineInto(m2, f2);
  let eLevel = 0, neg = 0, tc = 0, tf = 0;
  for (let K = 0; K < 10; K++) {
    let sc = 0, sf = 0;
    for (let j = 0; j < 20; j++) for (let i = 0; i < 20; i++) sc += m2.scalars[2]![m2.idx(i, j, K)]! * m2.rho0[K]! * 9;
    for (let k = 2 * K; k < 2 * K + 2; k++) for (let j = 0; j < 60; j++) for (let i = 0; i < 60; i++) { const v = f2.scalars[2]![f2.idx(i, j, k)]!; sf += v * f2.rho0[k]! * 0.5; if (v < 0) neg++; }
    if (K < 6) eLevel = Math.max(eLevel, Math.abs(sf / sc - 1));
    tc += sc; tf += sf;
  }
  const eMass = Math.abs(tf / tc - 1);
  // along a row through the shaft at mid-height: inside, at the last wet coarse centre, halfway to the dry one
  const row = (i: number): number => f2.scalars[2]![f2.idx(i, 30, 4)]!;
  const inside = row(30), rim = row(34) /* x = 34.5 km: coarse centre 11 at 34.5 km */, half = row(36) /* 36.5 km */;
  check('refine: rain shaft keeps its mass (1e-12; each wet level within 40 %), stays >= 0, falls off to under a quarter of its rim value two thirds of the way to the dry neighbour',
    eMass < 1e-12 && eLevel < 0.4 && neg === 0 && half < 0.25 * rim && rim > 0 && inside > rim * 0.9, `mass ${eMass.toExponential(1)}, levels ${(100 * eLevel).toFixed(0)} %, negative ${neg}, inside ${inside.toExponential(2)} rim ${rim.toExponential(2)} at 2/3 ${half.toExponential(2)}`);
}
{
  const f = 5e-5, sst = 301.15;
  const cfg: RegionalConfig = { ...base, nx: 30, ny: 30, nz: 12, dx: 12000, dy: 12000, dz: 1500, dt: 45, nsound: 6, f, dampDepth: 5000, dampRate: 1 / 300 };
  const mc = new RegionalModel(cfg, tropicalSounding(sst), 3);
  insertVortex(mc, f, 15);
  for (let s = 0; s < 4; s++) mc.step();
  const mf = new RegionalModel({ ...cfg, nx: 90, ny: 90, nz: 24, dx: 4000, dy: 4000, dz: 750, dt: 15 }, tropicalSounding(sst), 3);
  refineInto(mc, mf);
  const vmax = (m: RegionalModel): number => { let v = 0; for (let j = 0; j < m.c.ny; j++) for (let i = 0; i < m.c.nx; i++) { const q = m.idx(i, j, 0); v = Math.max(v, Math.hypot(0.5 * (m.u[q]! + m.u[q + 1]!), 0.5 * (m.v[q]! + m.v[q + m.sx]!))); } return v; };
  const ppmin = (m: RegionalModel): number => { let p = 0; for (let j = 0; j < m.c.ny; j++) for (let i = 0; i < m.c.nx; i++) p = Math.min(p, m.pp[m.idx(i, j, 0)]!); return p; };
  const vc = vmax(mc), vf = vmax(mf), pc = ppmin(mc), pf = ppmin(mf);
  check('refine: vortex max wind carried over (within 10 %)', Math.abs(vf - vc) < 0.1 * vc, `${vf.toFixed(2)} vs ${vc.toFixed(2)} m/s`);
  check('refine: central pi\' deficit carried over (within 10 %)', Math.abs(pf - pc) < 0.1 * Math.abs(pc), `${pf.toExponential(3)} vs ${pc.toExponential(3)}`);
  for (let s = 0; s < 8; s++) mf.step();
  const v2 = vmax(mf);
  check('refine: fine run continues stably (2 min), vortex kept', Number.isFinite(v2) && Math.abs(v2 - vf) < 0.15 * vf && mf.time === mc.time + 120, `${v2.toFixed(2)} m/s at t = ${mf.time} s`);
}

void DRY_AIR;
// Chart diagnostics (display only)
{
  const { parcelAscent, qsatW, dewPoint, pressure, columnDiagnostics, azimuthalMeans, C, COL } = await import('../regional/diagnostics.js');
  // Weisman-Klemp (1982) sounding, 14 g/kg boundary layer: surface-based CAPE of about 2000-2500 J/kg
  const nz = 40, dz = 400;
  const m = new RegionalModel({ ...base, nx: 4, ny: 4, nz, dx: 1000, dy: 1000, dz, dt: 6 }, weismanKlemp, 3);
  const T = new Float64Array(nz), p = new Float64Array(nz), q = new Float64Array(nz);
  for (let k = 0; k < nz; k++) { T[k] = m.th0[k]! * m.pi0[k]!; p[k] = pressure(m.pi0[k]!); q[k] = m.qv0[k]!; }
  const pc = parcelAscent(T, p, q, dz);
  check('diagnostics: WK82 surface-based CAPE 1500-3000 J/kg, small CIN, LCL below 2 km', pc.cape > 1500 && pc.cape < 3000 && pc.cin > -100 && pc.cin <= 0 && pc.lcl >= 0 && (pc.lcl + 0.5) * dz < 2000,
    `CAPE ${pc.cape.toFixed(0)} J/kg, CIN ${pc.cin.toFixed(1)} J/kg, LCL ${((pc.lcl + 0.5) * dz / 1000).toFixed(1)} km, EL ${((pc.el + 0.5) * dz / 1000).toFixed(1)} km`);
  const dry = parcelAscent(T, p, new Float64Array(nz), dz);
  check('diagnostics: no CAPE without moisture', dry.cape === 0 && dry.cin === 0, dry.cape);
  let worst = 0;
  for (const [Tk, pk] of [[300, 1e5], [273.15, 7e4], [250, 4e4]] as const) worst = Math.max(worst, Math.abs(dewPoint(qsatW(Tk, pk), pk) + 273.15 - Tk));
  check('diagnostics: dew point of saturated air equals the temperature (< 0.1 K)', worst < 0.1, worst);
  // solid-body rotation (vertical vorticity 0.01 s^-1) and w = 10 m/s: 2-5 km updraft helicity = 10 * 0.01 * 3000 m
  const r = new RegionalModel({ ...base, nx: 12, ny: 12, nz: 20, dx: 1000, dy: 1000, dz: 500, dt: 6 }, weismanKlemp, 3);
  const om = 0.005, xc = 6000, yc = 6000;
  for (let k = 0; k <= 20; k++) for (let j = -3; j < 15; j++) for (let i = -3; i < 15; i++) {
    const o = k * r.plane + (j + 3) * r.sx + (i + 3);
    r.u[o] = -om * ((j + 0.5) * 1000 - yc); r.v[o] = om * ((i + 0.5) * 1000 - xc); r.w[o] = 10;
  }
  const cd = columnDiagnostics(r);
  check('diagnostics: updraft helicity of a rotating updraft (300 m2/s2)', Math.abs(cd[COL * (6 * 12 + 6) + C.uh]! - 300) < 1e-6, cd[COL * (6 * 12 + 6) + C.uh]!);
  const az = azimuthalMeans(r, xc, yc, 1000, 4);
  // ring 2 (r = 2.5 km): tangential wind om * r, no radial wind (cell-sampled, so within one cell's offset)
  check('diagnostics: azimuthal-mean tangential wind of solid-body rotation (om r within 10 %)', Math.abs(az[5 * (2 * 20 + 3)]! - om * 2500) < 0.1 * om * 2500 && Math.abs(az[5 * (2 * 20 + 3) + 1]!) < 0.1 * om * 2500, `${az[5 * (2 * 20 + 3)]!.toFixed(3)} m/s vs ${(om * 2500).toFixed(3)}`);
}

// Tropical-cyclone environments: the neutral RE87 sounding has no surface-parcel CAPE; the unstable one has a
// well-mixed moist boundary layer and CAPE like the tropical oceans (so the outer region can sustain convection)
{
  const { parcelAscent, pressure } = await import('../regional/diagnostics.js');
  const { tcSounding } = await import('../regional/tropical.js');
  const nz = 50, dz = 500;
  const env = (kind: 'unstable' | 're87') => {
    const m = new RegionalModel({ ...base, nx: 4, ny: 4, nz, dx: 5000, dy: 5000, dz, dt: 10 }, tcSounding(kind, 301.15), 6);
    const T = new Float64Array(nz), p = new Float64Array(nz), q = new Float64Array(nz);
    for (let k = 0; k < nz; k++) { T[k] = m.th0[k]! * m.pi0[k]!; p[k] = pressure(m.pi0[k]!); q[k] = m.qv0[k]!; }
    return { ...parcelAscent(T, p, q, dz), q0: q[0]! };
  };
  const n = env('re87'), u = env('unstable');
  check('tropical: the RE87 sounding is neutral (no surface-parcel CAPE)', n.cape === 0, n.cape);
  check('tropical: the unstable sounding has CAPE 700-1500 J/kg and a moist mixed boundary layer (qv > 17 g/kg at 250 m)', u.cape > 700 && u.cape < 1500 && u.q0 > 0.017,
    `CAPE ${u.cape.toFixed(0)} J/kg, CIN ${u.cin.toFixed(0)}, qv ${(u.q0 * 1e3).toFixed(1)} g/kg`);
}

// Surface gustiness: free convection adds to a light wind (Beljaars 1995: w* of about 0.7 m/s at 5 m/s over a warm sea),
// heavy rain adds downdraft gusts of a few m/s even in calm air, and a strong mean wind is barely changed
{
  const { gustSpeed } = await import('../regional/physics.js');
  const g = (spd: number, qr: number): number => gustSpeed(spd, 1, 1.2e-3, 1, 4e-3, 300, qr, 1.15);
  const calm = g(0, 0), light = g(5, 0), rainy = g(0, 3e-3), windy = g(20, 0);
  check('gustiness: light wind +0.5-6 %, heavy rain in calm air > 2.5 m/s, strong wind within 1 %, calm dry air at the floor', light > 5.025 && light < 5.3 && rainy > 2.5 && windy < 20.2 && calm === 1,
    `${calm.toFixed(2)}, ${light.toFixed(2)}, ${rainy.toFixed(2)}, ${windy.toFixed(2)} m/s`);
}

// Set-ups: every small preset builds and steps with finite values; the trade wind is geostrophic (Coriolis balanced)
{
  const { buildModel } = await import('../app/regional/build.js');
  const { setupOf } = await import('../app/regional/setup.js');
  const { RegionalPhysics } = await import('../regional/physics.js');
  let ok = true; const bad: string[] = [];
  for (const id of ['tc', 'supercell', 'tornado_c']) {
    const b = buildModel(setupOf(id), false), mm = b.model, mp = new IceMicrophysics(mm);
    if (b.physics) new RegionalPhysics(mm, b.physics);
    for (let n = 0; n < 2; n++) { mm.step(); mp.apply(mm.c.dt); }
    if (![mm.u, mm.v, mm.w, mm.th, mm.pp, ...mm.scalars].every((a) => a.every(Number.isFinite))) { ok = false; bad.push(id); }
  }
  check('set-ups: the tc, supercell and tornado_c presets build and step with finite values', ok, bad.join(' ') || 'all finite');
  // a trade wind on the f-plane without surface physics stays put (geostrophic balance), no inertial oscillation
  const s = { ...setupOf('tc'), wind: 'trade' as const, windU: 8, fluxes: false, radiation: 'none' as const, init: 'none' as const, L: 240000 };
  const b = buildModel(s, false), mm = b.model;
  for (let n = 0; n < 60; n++) mm.step();
  let dv = 0; for (let q = 0; q < mm.size; q++) dv = Math.max(dv, Math.abs(mm.v[q]!));
  check('set-ups: an 8 m/s trade wind stays geostrophic (|v| < 0.05 m/s after 1 h)', dv < 0.05, dv);
}

// Storm catalogue: nothing before a storm exists; two vortices get two lasting identities while they move (also
// across a domain roll); two updraft areas are two cells
{
  const { findVortices, findCells, StormCatalog } = await import('../regional/storms.js');
  const { COL, C } = await import('../regional/diagnostics.js');
  const { tcSounding } = await import('../regional/tropical.js');
  const nx = 60, dx = 15000, f = 5e-5;
  const mm = new RegionalModel({ ...base, nx, ny: nx, nz: 10, dx, dy: dx, dz: 1000, dt: 60, f }, tcSounding('unstable', 301.15), 6);
  const vortex = (xc: number, yc: number, v: number): void => {
    for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
      const x = (i + 0.5) * dx - xc, y = (j + 0.5) * dx - yc, r = Math.hypot(x, y), R = 80000, e = Math.exp(-(r * r) / (R * R));
      const q = mm.idx(i, j, 0);
      mm.pp[q] = mm.pp[q]! - v * v * e / (1004.5 * 300);            // cyclostrophic-sized pressure dip
      if (r > 0) { const vt = v * (r / R) * Math.exp(0.5 - 0.5 * (r * r) / (R * R)); mm.u[q] = mm.u[q]! - vt * y / r; mm.v[q] = mm.v[q]! + vt * x / r; }
    }
  };
  const cat = new StormCatalog(), frame = { u: 0, v: 0 }, L = { x: nx * dx, y: nx * dx };
  cat.update(0, findVortices(mm, frame), { x: 0, y: 0 }, L);
  const none = cat.active.length;
  vortex(250000, 300000, 20); vortex(650000, 500000, 30);
  cat.update(600, findVortices(mm, frame), { x: 0, y: 0 }, L);
  cat.update(1200, findVortices(mm, frame), { x: 0, y: 0 }, L);
  const ids1 = cat.active.map((s) => s.id).sort().join(',');
  // the domain rolls by 5 cells (the origin moves back by the same distance): the storms keep their identities
  mm.roll(5, 0); cat.update(1800, findVortices(mm, frame), { x: -5 * dx, y: 0 }, L);
  const ids2 = cat.active.map((s) => s.id).sort().join(',');
  const strongest = cat.main();
  check('storms: no storm before one exists, two vortices keep their identities across a roll, the deeper is the main one', none === 0 && ids1 === '1,2' && ids2 === ids1 && (strongest?.dp ?? 0) > 8 && Math.abs((strongest?.x ?? 0) - 650000) < 20000,
    `before ${none}, ids ${ids1} -> ${ids2}, main ${strongest?.name} (${strongest?.dp?.toFixed(1)} hPa)`);
  const n = 40, col = new Float32Array(n * n * COL);
  for (const [ic, jc] of [[10, 10], [30, 25]] as const) for (let j = jc - 1; j <= jc + 1; j++) for (let i = ic - 1; i <= ic + 1; i++) col[COL * (j * n + i) + C.wmax] = 15;
  const cells = findCells(n, n, 1000, 1000, col, true);
  check('storms: two updraft areas are two cells', cells.length === 2 && Math.abs(cells[0]!.xd - 10500) < 600, cells.map((c) => `(${(c.xd / 1000).toFixed(1)}, ${(c.yd / 1000).toFixed(1)})`).join(' '));
}

// Satellite-like columns: a clear tropical column has no cloud albedo, about 50-70 mm of precipitable water and a
// water-vapour channel temperature of the upper troposphere (-45 to -10 °C)
{
  const { columnDiagnostics, C, COL } = await import('../regional/diagnostics.js');
  const { tcSounding } = await import('../regional/tropical.js');
  const mm = new RegionalModel({ ...base, nx: 4, ny: 4, nz: 50, dx: 5000, dy: 5000, dz: 500, dt: 10 }, tcSounding('unstable', 301.15), 6);
  for (let k = 0; k < mm.c.nz; k++) for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) mm.scalars[0]![mm.idx(i, j, k)] = mm.qv0[k]!;
  const cd = columnDiagnostics(mm, false), o = COL * 5;
  const vis = cd[o + C.vis]!, pw = cd[o + C.pw]!, wv = cd[o + C.wvT]! - 273.15;
  check('satellite columns: clear tropical column: albedo 0, precipitable water 45-75 mm, water-vapour channel -45 to -10 °C', vis === 0 && pw > 45 && pw < 75 && wv > -45 && wv < -10,
    `albedo ${vis.toFixed(2)}, PW ${pw.toFixed(1)} mm, WV ${wv.toFixed(1)} °C`);
  // sub-grid cloud: none below RHc, a thin deck at 95 % on a 15 km grid, the visible top at that level
  const { subgridCloud, subgridRHc, qsatW: qsw } = await import('../regional/display.js');
  const qs0 = qsw(290, 9e4), rhc = subgridRHc(15000), c90 = subgridCloud(0.84 * qs0, qs0, rhc), c95 = subgridCloud(0.95 * qs0, qs0, rhc);
  for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) { const q = mm.idx(i, j, 3); const pi = mm.pi0[3]! + mm.pp[q]!; mm.scalars[0]![q] = 0.97 * qsw(mm.th[q]! * pi, 1e5 * Math.pow(pi, 1004.5 / 287.05)); }
  const cs = columnDiagnostics(mm, true);
  check('sub-grid cloud: none at 84 % (RHc 0.85), 0.05-0.3 g/kg at 95 %; a humid layer gives albedo > 0.1 with its top at that level',
    c90 === 0 && c95 > 5e-5 && c95 < 3e-4 && cs[o + C.vis]! > 0.1 && Math.abs(cs[o + C.visZ]! - mm.zc[3]!) < 1,
    `${(c95 * 1e3).toFixed(3)} g/kg, albedo ${cs[o + C.vis]!.toFixed(2)}, top ${cs[o + C.visZ]!.toFixed(0)} m`);
}

// Wind interactions: a push once adds its speed at the centre; a lasting push relaxes the wind toward the target
// (after 5 tau within 1 %); a counter-clockwise rotation blows toward +y east of the centre; w stays 0 at the ground
{
  const { applyWind, FORCING_TAU } = await import('../regional/forcing.js');
  const mk = (): RegionalModel => new RegionalModel({ ...base, nx: 20, ny: 20, nz: 10, dx: 1000, dy: 1000, dz: 500, dt: 5 }, weismanKlemp, 3);
  const m1 = mk(), c = 10000, zc = m1.zc[2]!;
  const push = { x: c, y: c + 500, z: zc, R: 5000, H: 1500, speed: 10, dir: [1, 0, 0] as [number, number, number], form: 'push' as const, sign: 1 as const };
  applyWind(m1, [push], 'once');
  const u1 = m1.u[m1.idx(10, 10, 2)]!;
  const m2 = mk();
  for (let n = 0; n < 5 * FORCING_TAU / 5; n++) applyWind(m2, [push], 5);
  const u2 = m2.u[m2.idx(10, 10, 2)]!;
  const m3 = mk();
  applyWind(m3, [{ ...push, x: c, y: c, form: 'rotate', speed: 10, dir: [0, 0, 0] }], 'once');
  const vEast = m3.v[m3.idx(12, 10, 2)]!;
  const m4 = mk();
  applyWind(m4, [{ ...push, z: 0, dir: [0, 0, 1], H: 2000 }], 'once');
  let w0 = 0; for (let j = 0; j < 20; j++) for (let i = 0; i < 20; i++) w0 = Math.max(w0, Math.abs(m4.w[m4.idx(i, j, 0)]!));
  const wUp = m4.w[m4.idx(10, 10, 1)]!;
  check('wind interactions: push once ~10 m/s at the centre, lasting push -> target within 1 %, CCW rotation northward east of centre, no w at the ground',
    Math.abs(u1 - 10) < 0.2 && Math.abs(u2 - 10 * (1 - Math.exp(-5))) < 0.15 && vEast > 1 && w0 === 0 && wUp > 1,
    `${u1.toFixed(2)}, ${u2.toFixed(2)}, v ${vEast.toFixed(2)}, w0 ${w0}, w1 ${wUp.toFixed(2)}`);
}

// Cumulus parameterization (simplified Betts-Miller): active in the unstable tropical column, energy-consistent (heating
// = latent heat of the vapour removed), total water conserved (the removed vapour becomes rain water and detrained
// ice), inactive over the neutral RE87 column and on a 3 km grid
{
  const { applyCumulus, cumulusScale } = await import('../regional/cumulus.js');
  const { tcSounding } = await import('../regional/tropical.js');
  const { IceMicrophysics: Ice } = await import('../regional/ice.js');
  const mk = (kind: 'unstable' | 're87', dx: number): RegionalModel => {
    const mm = new RegionalModel({ ...base, nx: 3, ny: 3, nz: 25, dx, dy: dx, dz: 1000, dt: 60 }, tcSounding(kind, 301.15), 6);
    new Ice(mm);
    for (let k = 0; k < 25; k++) for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) mm.scalars[0]![mm.idx(i, j, k)] = mm.qv0[k]!;
    return mm;
  };
  const col = (mm: RegionalModel): { h: number; w: number } => {
    let h = 0, w = 0;
    for (let k = 0; k < 25; k++) {
      const q = mm.idx(1, 1, k), md = mm.rho0[k]! * 1000, T = mm.th[q]! * (mm.pi0[k]! + mm.pp[q]!);
      h += 1004.5 * T * md; w += md * (mm.scalars[0]![q]! + mm.scalars[1]![q]! + mm.scalars[2]![q]! + mm.scalars[3]![q]!);
    }
    return { h, w };
  };
  const u = mk('unstable', 15000), a0 = col(u), info = { kb: new Int16Array(9), kt: new Int16Array(9), rate: new Float32Array(9) };
  applyCumulus(u, 60, true, info);
  const a1 = col(u), P = info.rate[4]! / 3600 * 60, lat = 2.5e6 * P;
  const n = mk('re87', 15000), info2 = { kb: new Int16Array(9), kt: new Int16Array(9), rate: new Float32Array(9) };
  applyCumulus(n, 60, true, info2);
  const f = mk('unstable', 3000), f0 = col(f); applyCumulus(f, 60, true, null);
  check('cumulus: active in the unstable column (rain > 0, top above 8 km), heating = latent heat (1 %), water conserved, none over RE87 or at 3 km',
    P > 0 && info.kt[4]! >= 8 && Math.abs((a1.h - a0.h) - lat) < 0.01 * lat && Math.abs(a1.w - a0.w) < 1e-9 * a0.w && info2.kt[4] === -1 && col(f).h === f0.h && cumulusScale(3000) === 0,
    `rate ${info.rate[4]!.toFixed(2)} mm/h, top level ${info.kt[4]}, heating ${((a1.h - a0.h) / lat).toFixed(4)} x latent, water ${((a1.w - a0.w) / a0.w).toExponential(1)}`);
}

// Coarsening: refining to half the spacing and averaging back returns the smooth coarse fields (within 1 %); a
// sub-box put back replaces only the box (outside unchanged) and blends at its edge
{
  const { refineInto: rin, coarsenInto: cin } = await import('../regional/refine.js');
  const mk = (n: number, dx: number, lateral: 'periodic' | 'open'): RegionalModel => new RegionalModel({ ...base, nx: n, ny: n, nz: 10, dx, dy: dx, dz: 1000, dt: 10, lateral }, weismanKlemp, 3);
  const a = mk(30, 4000, 'periodic'), L = 120000;
  for (let k = 0; k < 10; k++) for (let j = 0; j < 30; j++) for (let i = 0; i < 30; i++) {
    const q = a.idx(i, j, k), x = (i + 0.5) * 4000, y = (j + 0.5) * 4000;
    a.th[q] = a.th[q]! + 2 * Math.sin(2 * Math.PI * x / L) * Math.cos(2 * Math.PI * y / L);
    a.u[q] = 5 * Math.cos(2 * Math.PI * y / L); a.v[q] = 3 * Math.sin(2 * Math.PI * (i * 4000) / L);
  }
  const th0 = Float64Array.from(a.th), u0 = Float64Array.from(a.u);
  const f = mk(60, 2000, 'periodic');
  rin(a, f, 0, 0);
  a.th.fill(300); a.u.fill(0);
  cin(f, a, 0, 0, 0);
  let eth = 0, eu = 0;
  for (let k = 0; k < 10; k++) for (let j = 0; j < 30; j++) for (let i = 0; i < 30; i++) { const q = a.idx(i, j, k); eth = Math.max(eth, Math.abs(a.th[q]! - th0[q]!)); eu = Math.max(eu, Math.abs(a.u[q]! - u0[q]!)); }
  // a 40 km open box in the middle, warmed by 1 K, put back with a 10 km blend
  const box = mk(20, 2000, 'open'), before = Float64Array.from(a.th);
  rin(a, box, 40000, 40000);
  for (let q = 0; q < box.size; q++) box.th[q] = box.th[q]! + 1;
  cin(box, a, 40000, 40000, 10000);
  const out = Math.abs(a.th[a.idx(2, 2, 3)]! - before[a.idx(2, 2, 3)]!), mid = a.th[a.idx(15, 15, 3)]! - before[a.idx(15, 15, 3)]!, edge = a.th[a.idx(10, 15, 3)]! - before[a.idx(10, 15, 3)]!;
  check('coarsening: refine then average back within 2 % of the smooth fields; a box put back changes only the box, fully at its centre, partly at its edge',
    eth < 0.04 && eu < 0.1 && out === 0 && Math.abs(mid - 1) < 0.05 && edge > 0 && edge < 0.9,
    `theta ${eth.toExponential(1)} K, u ${eu.toExponential(1)} m/s; box: outside ${out}, centre ${mid.toFixed(3)}, edge ${edge.toFixed(3)}`);
}

// Two-way nest in a cylinder (twoway.ts): geometry, relaxation targets, feedback and a short coupled run
{
  const { nestGeometry, nestTargets, nestFeedback, emptyTargets, parentState, nestRelax, feedbackWeight } = await import('../regional/twoway.js');
  const { refineInto: rin } = await import('../regional/refine.js');
  const pcfg: RegionalConfig = { ...base, nx: 40, ny: 40, nz: 10, dx: 4000, dy: 4000, dz: 1000, dt: 10, lateral: 'periodic' };
  const p = new RegionalModel(pcfg, weismanKlemp, 1);
  const g0 = nestGeometry(p.c, 30000, 2000, 500);
  if (typeof g0 === 'string') throw new Error(g0);
  const g = g0;
  check('two-way nest: geometry (ratios 2 and 2, 24 outer cells across centred in the domain, 48 x 48 x 20 inner cells)',
    g.r === 2 && g.rz === 2 && g.np === 24 && g.i0 === 8 && g.j0 === 8 && g.nx === 48 && g.nz === 20 && g.cx === 80000 && g.cy === 80000,
    `r ${g.r} rz ${g.rz} np ${g.np} i0 ${g.i0} nx ${g.nx} nz ${g.nz} centre ${g.cx}`);
  // departures from the base state linear in x, y, z (at each variable's staggered position): interpolation
  // reproduces them exactly away from the lowest and highest levels
  const lin = (x: number, y: number, z: number): number => 2e-5 * x - 1e-5 * y + 3e-4 * z;
  for (let k = 0; k < 10; k++) for (let j = 0; j < 40; j++) for (let i = 0; i < 40; i++) {
    const q = p.idx(i, j, k), z = (k + 0.5) * 1000;
    p.th[q] = p.th0[k]! + lin((i + 0.5) * 4000, (j + 0.5) * 4000, z); p.u[q] = lin(i * 4000, (j + 0.5) * 4000, z); p.v[q] = 0.5 * lin((i + 0.5) * 4000, j * 4000, z);
    p.pp[q] = 1e-4 * lin((i + 0.5) * 4000, (j + 0.5) * 4000, z); p.scalars[0]![q] = p.qv0[k]! * (1 + 0.01 * lin((i + 0.5) * 4000, (j + 0.5) * 4000, z));
  }
  const old = parentState(p);
  for (const a of [old.th, old.u]) for (let i = 0; i < a.length; i++) a[i] = a[i]! - 1;
  const ccfg: RegionalConfig = { ...pcfg, nx: g.nx, ny: g.nx, nz: g.nz, dx: g.dx, dy: g.dx, dz: g.dz, dt: 5, lateral: 'open', ...nestRelax(g) };
  const ch = new RegionalModel(ccfg, weismanKlemp, 1), tg = emptyTargets(ch);
  nestTargets(p, old, ch, g, 0.25, tg);
  // a child point in the relaxation ring (a corner of the box), mid-troposphere
  const ci = 1, cj = 2, ck = 9, x0 = g.i0 * 4000, y0 = g.j0 * 4000, zc = (ck + 0.5) * g.dz;
  const eTh = Math.abs(tg.th[ch.idx(ci, cj, ck)]! - (ch.th0[ck]! + lin(x0 + (ci + 0.5) * g.dx, y0 + (cj + 0.5) * g.dx, zc) - 0.75));
  const eU = Math.abs(tg.u[ch.idx(ci, cj, ck)]! - (lin(x0 + ci * g.dx, y0 + (cj + 0.5) * g.dx, zc) - 0.75));
  check('two-way nest: relaxation targets are the outer fields interpolated in space and time (exact for linear fields)', eTh < 1e-9 && eU < 1e-9, `theta ${eTh.toExponential(1)}, u ${eU.toExponential(1)}`);
  // feedback: a child that is the interpolated parent gives the parent back; a warmer child warms the parent inside only
  rin(p, ch, x0, y0);
  const th0 = Float64Array.from(p.th), u0 = Float64Array.from(p.u);
  nestFeedback(ch, p, g);
  // (the lowest and highest levels: the child's outer half-levels hold the departure of the nearest outer level)
  let eBack = 0, eEdge = 0;
  for (let k = 0; k < 10; k++) for (let j = 0; j < 40; j++) for (let i = 0; i < 40; i++) {
    const q = p.idx(i, j, k), e = Math.max(Math.abs(p.th[q]! - th0[q]!), Math.abs(p.u[q]! - u0[q]!));
    if (k === 0 || k === 9) eEdge = Math.max(eEdge, e); else eBack = Math.max(eBack, e);
  }
  for (let i = 0; i < ch.th.length; i++) ch.th[i] = ch.th[i]! + 1;
  nestFeedback(ch, p, g);
  let inside = 0, outside = 0, taper = 0;
  for (let j = 0; j < 40; j++) for (let i = 0; i < 40; i++) {
    const d = Math.hypot((i + 0.5) * 4000 - g.cx, (j + 0.5) * 4000 - g.cy), dth = p.th[p.idx(i, j, 5)]! - th0[p.idx(i, j, 5)]!;
    if (d <= g.R - g.Wf) inside = Math.max(inside, Math.abs(dth - 1));
    else if (d >= g.R) outside = Math.max(outside, Math.abs(dth));
    else taper = Math.max(taper, Math.abs(dth - feedbackWeight(g, d)));
  }
  check('two-way nest: feedback returns an unchanged child exactly (lowest / highest level within 0.04); a 1 K warmer child warms the outer grid by 1 K inside, blended at the rim, not at all outside',
    eBack < 1e-9 && eEdge < 0.04 && inside < 1e-9 && outside === 0 && taper < 1e-9,
    `unchanged ${eBack.toExponential(1)} (lowest / highest level ${eEdge.toFixed(4)}), inside ${inside.toExponential(1)}, rim ${taper.toExponential(1)}, outside ${outside}`);
  // a short coupled run: a warm bubble in the middle of a resting atmosphere
  const q2 = new RegionalModel(pcfg, weismanKlemp, 1);
  for (let k = 0; k < 10; k++) for (let j = 0; j < 40; j++) for (let i = 0; i < 40; i++) {
    const rr = Math.hypot(((i + 0.5) * 4000 - 80000) / 12000, ((j + 0.5) * 4000 - 80000) / 12000, ((k + 0.5) * 1000 - 1500) / 1500);
    q2.scalars[0]![q2.idx(i, j, k)] = q2.qv0[k]!;
    if (rr < 1) q2.th[q2.idx(i, j, k)] = q2.th[q2.idx(i, j, k)]! + 2 * Math.cos(0.5 * Math.PI * rr) ** 2;
  }
  const c2 = new RegionalModel(ccfg, weismanKlemp, 1);
  c2.boundary = emptyTargets(c2);
  rin(q2, c2, x0, y0);
  let finite = true;
  for (let s = 0; s < 24; s++) {
    const st = parentState(q2);
    q2.step();
    for (let n = 0; n < 2; n++) { nestTargets(q2, st, c2, g, (n + 0.5) / 2, c2.boundary); c2.step(); }
    nestFeedback(c2, q2, g);
    if (!Number.isFinite(q2.maxAbs(q2.w, 11)) || !Number.isFinite(c2.maxAbs(c2.w, 21))) finite = false;
  }
  // after the feedback the centre of the outer grid is the child's mean there
  const pc = q2.idx(20, 20, 2);
  // (as departures from each grid's base profile)
  let mean = 0; for (let dk = 0; dk < 2; dk++) for (let dj = 0; dj < 2; dj++) for (let di = 0; di < 2; di++) mean += c2.th[c2.idx(24 + di, 24 + dj, 4 + dk)]! - c2.th0[4 + dk]!;
  mean /= 8;
  const wIn = c2.maxAbs(c2.w, 21), wOut = q2.maxAbs(q2.w, 11);
  check('two-way nest: a 4-minute coupled run of a warm bubble stays finite, rises in both grids, and the outer centre holds the child mean',
    finite && wIn > 0.3 && wIn < 30 && wOut > 0.1 && Math.abs(q2.th[pc]! - (q2.th0[2]! + mean)) < 1e-9 && c2.time === q2.time,
    `w inner ${wIn.toFixed(2)} outer ${wOut.toFixed(2)} m/s, centre ${(q2.th[pc]! - q2.th0[2]! - mean).toExponential(1)}, times ${c2.time} ${q2.time}`);
}

summary('regional');
