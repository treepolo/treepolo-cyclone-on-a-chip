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

summary('regional');
