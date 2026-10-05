// GPU regional model vs CPU Float64 regional model.
import { RegionalModel } from '../../regional/core.js';
import { KesslerMicrophysics, weismanKlemp, QV, QC, QR } from '../../regional/kessler.js';
import { RegionalPhysics } from '../../regional/physics.js';
import { tropicalSounding, insertVortex, tcSounding } from '../../regional/tropical.js';
import { GpuRegional, COL } from '../regionalGpu.js';
import { gcheck, getDevice } from './harness.js';
import { nestFromGlobal, GlobalSnapshot } from '../../regional/nest.js';
import { IceMicrophysics, QI, QS, QG } from '../../regional/ice.js';
import { C, columnDiagnostics, columnProfiles, azimuthalMeans } from '../../regional/diagnostics.js';
import { Tracers } from '../../regional/tracers.js';
import { applyWind, type WindForcing } from '../../regional/forcing.js';
import { cloudExtinction, precipExtinction, extByte, subgridCloud, subgridRHc, qsatW } from '../../regional/display.js';

function cmp(m: RegionalModel, g: Float32Array, f: number, a: Float64Array, nk: number): number {
  let e = 0, s = 0;
  for (let k = 0; k < nk; k++) for (let j = 0; j < m.c.ny; j++) for (let i = 0; i < m.c.nx; i++) {
    const q = m.idx(i, j, k);
    e += (g[f * m.size + q]! - a[q]!) ** 2; s += a[q]! ** 2;
  }
  return Math.sqrt(e / Math.max(s, 1e-300));
}

export async function regionalTests(): Promise<void> {
  const device = await getDevice();
  // ---- moist convection (no sub-grid physics)
  {
    const nx = 20, nz = 24, dx = 3000, dz = 700;
    const m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt: 6, nsound: 6, f: 0, beta: 0.2, divDamp: 0.1, dampDepth: 4000, dampRate: 1 / 300, kdiff2: 0 }, weismanKlemp, 3);
    const mp = new KesslerMicrophysics(m);
    m.setBaseWind((z) => ({ u: 10 * Math.tanh(z / 3000) - 5, v: 2 }));
    for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) m.scalars[QV]![m.idx(i, j, k)] = m.qv0[k]!;
    const c = nx * dx / 2;
    for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
      const r = Math.sqrt((((i + 0.5) * dx - c) / 9000) ** 2 + (((j + 0.5) * dx - c) / 9000) ** 2 + ((m.zc[k]! - 1400) / 1400) ** 2);
      if (r < 1) m.th[m.idx(i, j, k)] = m.th[m.idx(i, j, k)]! + 2 * Math.cos(0.5 * Math.PI * r) ** 2;
    }
    for (let s = 0; s < 100; s++) { m.step(); mp.apply(6); }     // 10 min: convection underway
    const g = new GpuRegional(device, m, { moist: true, physics: null });
    g.uploadFrom(m);
    for (const n of [1, 10]) {
      const todo = n === 1 ? 1 : 9;
      for (let s = 0; s < todo; s++) { m.step(); mp.apply(6); }
      g.step(todo);
      const st = await g.readState();
      const du = cmp(m, st, 0, m.u, nz), dw = cmp(m, st, 2, m.w, nz + 1), dth = cmp(m, st, 3, m.th, nz);
      const dq = cmp(m, st, 5, m.scalars[QV]!, nz), dc = cmp(m, st, 6, m.scalars[QC]!, nz), dr = cmp(m, st, 7, m.scalars[QR]!, nz);
      const tol = n === 1 ? 1e-4 : 2e-3;
      gcheck(`regional moist, ${n} steps: u rel L2 < ${tol}`, du < tol, du);
      gcheck(`regional moist, ${n} steps: w rel L2 < ${tol * 10}`, dw < tol * 10, dw);
      gcheck(`regional moist, ${n} steps: theta rel L2 < ${tol / 10}`, dth < tol / 10, dth);
      gcheck(`regional moist, ${n} steps: qv / qc / qr rel L2 < ${tol * 20}`, dq < tol * 20 && dc < tol * 20 && dr < tol * 20, `${dq.toExponential(2)} ${dc.toExponential(2)} ${dr.toExponential(2)}`);
    }
  }
  // ---- tropical-cyclone physics (turbulence, surface fluxes, radiation, Coriolis)
  {
    const nx = 16, nz = 16, dx = 20000, dz = 1250, f = 5e-5;
    const m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt: 60, nsound: 6, f, beta: 0.3, divDamp: 0.1, dampDepth: 5000, dampRate: 1 / 300, kdiff2: 0 }, tropicalSounding(301.15), 3);
    const mp = new KesslerMicrophysics(m);
    const cfg = { lh: 4000, lv: 100, sst: 301.15, ck: 1.2e-3, radTau: 12 * 3600, radMax: 2 / 86400 };
    new RegionalPhysics(m, cfg);
    insertVortex(m, f, 15);
    for (let s = 0; s < 30; s++) { m.step(); mp.apply(60); }
    const g = new GpuRegional(device, m, { moist: true, physics: cfg });
    g.uploadFrom(m);
    for (let s = 0; s < 5; s++) { m.step(); mp.apply(60); }
    g.step(5);
    const st = await g.readState();
    const du = cmp(m, st, 0, m.u, nz), dth = cmp(m, st, 3, m.th, nz), dq = cmp(m, st, 5, m.scalars[QV]!, nz);
    gcheck('regional TC physics, 5 steps: u rel L2 < 1e-3', du < 1e-3, du);
    gcheck('regional TC physics, 5 steps: theta rel L2 < 1e-5', dth < 1e-5, dth);
    gcheck('regional TC physics, 5 steps: qv rel L2 < 1e-3', dq < 1e-3, dq);
  }
  // ---- the same with constant clear-sky cooling and a 4 m/s minimum wind in the surface fluxes
  {
    const nx = 16, nz = 16, dx = 20000, dz = 1250, f = 5e-5;
    const m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt: 60, nsound: 6, f, beta: 0.3, divDamp: 0.1, dampDepth: 5000, dampRate: 1 / 300, kdiff2: 0 }, tropicalSounding(301.15, 200, 0.6), 3);
    const mp = new KesslerMicrophysics(m);
    const cfg = { lh: 4000, lv: 100, sst: 301.15, ck: 1.2e-3, radTau: 12 * 3600, radMax: 2 / 86400, vmin: 4, radConst: 1.5 / 86400 };
    new RegionalPhysics(m, cfg);
    insertVortex(m, f, 15);
    for (let s = 0; s < 30; s++) { m.step(); mp.apply(60); }
    const g = new GpuRegional(device, m, { moist: true, physics: cfg });
    g.uploadFrom(m);
    for (let s = 0; s < 5; s++) { m.step(); mp.apply(60); }
    g.step(5);
    const st = await g.readState();
    const du = cmp(m, st, 0, m.u, nz), dth = cmp(m, st, 3, m.th, nz), dq = cmp(m, st, 5, m.scalars[QV]!, nz);
    gcheck('regional TC physics with constant cooling and vmin 4 m/s, 5 steps: u < 1e-3, theta < 1e-5, qv < 1e-3', du < 1e-3 && dth < 1e-5 && dq < 1e-3, `${du.toExponential(1)} ${dth.toExponential(1)} ${dq.toExponential(1)}`);
  }
  // ---- surface gustiness and a geostrophic trade wind (Coriolis on the departure from the background wind)
  {
    const nx = 16, nz = 16, dx = 20000, dz = 1250, f = 5e-5;
    const m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt: 60, nsound: 6, f, beta: 0.3, divDamp: 0.1, dampDepth: 5000, dampRate: 1 / 300, kdiff2: 0, geostrophic: true }, tropicalSounding(301.15, 200, 0.6), 3);
    const mp = new KesslerMicrophysics(m);
    m.setBaseWind((z) => ({ u: -7 * Math.max(0, Math.min(1, (12000 - z) / 9000)), v: 0 }));
    const cfg = { lh: 4000, lv: 100, sst: 301.15, ck: 1.2e-3, radTau: 12 * 3600, radMax: 2 / 86400, vmin: 1, radConst: 1.5 / 86400, gust: true };
    new RegionalPhysics(m, cfg);
    insertVortex(m, f, 15);
    for (let s = 0; s < 30; s++) { m.step(); mp.apply(60); }
    const g = new GpuRegional(device, m, { moist: true, physics: cfg });
    g.uploadFrom(m);
    for (let s = 0; s < 5; s++) { m.step(); mp.apply(60); }
    g.step(5);
    const st = await g.readState();
    const du = cmp(m, st, 0, m.u, nz), dth = cmp(m, st, 3, m.th, nz), dq = cmp(m, st, 5, m.scalars[QV]!, nz);
    gcheck('regional TC physics with gustiness and a geostrophic trade wind, 5 steps: u < 1e-3, theta < 1e-5, qv < 1e-3', du < 1e-3 && dth < 1e-5 && dq < 1e-3, `${du.toExponential(1)} ${dth.toExponential(1)} ${dq.toExponential(1)}`);
  }
  // ---- cumulus parameterization (cumulus.ts) with ice microphysics on a 15 km grid
  {
    const nx = 12, nz = 25, dx = 15000, dz = 1000, f = 5e-5;
    const m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt: 60, nsound: 6, f, beta: 0.3, divDamp: 0.1, dampDepth: 5000, dampRate: 1 / 300, kdiff2: 0 }, tcSounding('unstable', 301.15), 6);
    const mp = new IceMicrophysics(m);
    for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
      const q = m.idx(i, j, k);
      // columns of different moisture: some deep, some shallow, some none
      m.scalars[QV]![q] = m.qv0[k]! * (0.85 + 0.3 * (((i * 7 + j * 3) % 5) / 4)) * (i % 3 === 0 && m.zc[k]! > 2000 ? 0.5 : 1);
    }
    const cfg = { lh: 3000, lv: 100, sst: 301.15, ck: 1.2e-3, radTau: 0, radMax: 0, cumulus: true };
    const phy = new RegionalPhysics(m, cfg);
    const g = new GpuRegional(device, m, { moist: true, physics: cfg, ice: true });
    g.uploadFrom(m);
    for (let s = 0; s < 3; s++) { m.step(); mp.apply(60); }
    g.step(3);
    const st = await g.readState();
    const dth = cmp(m, st, 3, m.th, nz), dq = cmp(m, st, 5, m.scalars[QV]!, nz), dr = cmp(m, st, 7, m.scalars[QR]!, nz), di = cmp(m, st, 8, m.scalars[QI]!, nz);
    const cu = phy.cu!, deep = Array.from(cu.kt).filter((x) => x >= 0).length, shallow = Array.from(cu.kt).filter((x) => x < -1).length;
    gcheck('regional cumulus + ice, 3 steps: theta < 1e-5, qv < 1e-3, qr and qi < 2e-2 (rel L2); deep and shallow columns present',
      dth < 1e-5 && dq < 1e-3 && dr < 2e-2 && di < 2e-2 && deep > 0 && shallow > 0, `${dth.toExponential(1)} ${dq.toExponential(1)} ${dr.toExponential(1)} ${di.toExponential(1)} (deep ${deep}, shallow ${shallow})`);
  }
  // ---- lasting wind forcings (forcing.ts): a tilted push across the periodic edge and a clockwise rotation
  {
    const nx = 24, nz = 16, dx = 2000, dz = 500;
    const m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt: 6, nsound: 6, f: 1e-4, beta: 0.2, divDamp: 0.1, dampDepth: 3000, dampRate: 1 / 300, kdiff2: 0 }, weismanKlemp, 3);
    m.setBaseWind((z) => ({ u: 5 * Math.tanh(z / 3000), v: 1 }));
    for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) m.scalars[QV]![m.idx(i, j, k)] = m.qv0[k]!;
    const list: WindForcing[] = [
      { x: 2000, y: 24000, z: 1500, R: 9000, H: 1500, speed: 15, dir: [0.6, 0.64, 0.48], form: 'push', sign: 1 },
      { x: 30000, y: 20000, z: 3000, R: 12000, H: 2500, speed: 12, dir: [0, 0, 0], form: 'rotate', sign: -1 },
    ];
    m.preStep = (mm): void => applyWind(mm, list, mm.c.dt);
    const g = new GpuRegional(device, m, { moist: true, physics: null });
    g.uploadFrom(m); g.setForcings(list);
    for (let s = 0; s < 5; s++) m.step();
    g.step(5);
    const st = await g.readState();
    const du = cmp(m, st, 0, m.u, nz), dv = cmp(m, st, 1, m.v, nz), dw = cmp(m, st, 2, m.w, nz + 1);
    let wm = 0; for (let q = 0; q < m.size; q++) wm = Math.max(wm, Math.abs(m.w[q]!));
    gcheck('regional wind forcing (push across the periodic edge, rotation), 5 steps: u, v rel L2 < 1e-4, w < 1e-3', du < 1e-4 && dv < 1e-4 && dw < 1e-3 && wm > 0.05, `${du.toExponential(1)} ${dv.toExponential(1)} ${dw.toExponential(1)} (w max ${wm.toFixed(2)} m/s)`);
  }
}

/** Analytic global state for nesting tests: midlatitude jet with a wave, moist lower troposphere. */
function syntheticSnapshot(): GlobalSnapshot {
  const nlat = 48, nlon = 96, K = 20;
  const lat = Float64Array.from({ length: nlat }, (_, j) => (90 - (j + 0.5) * 180 / nlat) * Math.PI / 180);
  const lon = Float64Array.from({ length: nlon }, (_, i) => i * 2 * Math.PI / nlon);
  const sigmaHalf = Float64Array.from({ length: K + 1 }, (_, k) => k / K);
  const sigma = Float64Array.from({ length: K }, (_, k) => (k + 0.5) / K);
  const ng = nlat * nlon, u = new Float64Array(K * ng), v = new Float64Array(K * ng), T = new Float64Array(K * ng), q = new Float64Array(K * ng), ps = new Float64Array(ng);
  for (let j = 0; j < nlat; j++) for (let i = 0; i < nlon; i++) {
    const p = j * nlon + i, la = lat[j]!, lo = lon[i]!;
    ps[p] = 1e5 + 800 * Math.sin(la * 2) * Math.cos(5 * lo);
    for (let k = 0; k < K; k++) {
      const s = sigma[k]!, o = k * ng + p;
      u[o] = 25 * Math.sin(Math.PI * s) * Math.exp(-((((la * 180 / Math.PI) - 40) / 15) ** 2)) + 3 * Math.sin(5 * lo);
      v[o] = 4 * Math.cos(5 * lo) * Math.cos(la);
      T[o] = Math.max(210, (300 - 30 * Math.sin(la) ** 2) * Math.pow(s, 0.19));
      q[o] = 0.012 * Math.cos(la) ** 2 * s ** 3;
    }
  }
  return { nlat, nlon, K, lat, lon, sigma, sigmaHalf, u, v, T, ps, q };
}

export async function regionalNestTests(): Promise<void> {
  const device = await getDevice();
  const g0 = syntheticSnapshot();
  const { model: m, boundary } = nestFromGlobal(g0, { lat0: 35 * Math.PI / 180, lon0: 1.0, L: 400000, dx: 20000, nz: 16, dz: 1000, dt: 60, nsound: 6 }, 3);
  const mp = new KesslerMicrophysics(m);
  const n2 = m.c.nx * m.c.ny;
  const tsk = Float64Array.from({ length: n2 }, (_, c) => 290 + 8 * (c % m.c.nx) / m.c.nx);
  const wet = Float64Array.from({ length: n2 }, (_, c) => (Math.floor(c / m.c.nx) < m.c.ny / 2 ? 1 : 0.3));
  const cfg = { lh: 4000, lv: 100, sst: 0, ck: 1.2e-3, radTau: 0, radMax: 0, surface: { tsk, wet }, z0: 0.1, frameVel: { u: 3, v: -2 } };
  new RegionalPhysics(m, cfg);
  for (let s = 0; s < 20; s++) { m.step(); mp.apply(60); }
  const g = new GpuRegional(device, m, { moist: true, physics: cfg, boundary });
  g.uploadFrom(m);
  for (let s = 0; s < 10; s++) { m.step(); mp.apply(60); }
  g.step(10);
  const st = await g.readState(), nz = m.c.nz;
  const du = cmp(m, st, 0, m.u, nz), dv = cmp(m, st, 1, m.v, nz), dw = cmp(m, st, 2, m.w, nz + 1), dth = cmp(m, st, 3, m.th, nz), dpp = cmp(m, st, 4, m.pp, nz), dq = cmp(m, st, 5, m.scalars[QV]!, nz);
  gcheck('regional nest (open BC, land/sea surface), 10 steps: u, v rel L2 < 1e-3', du < 1e-3 && dv < 1e-3, `${du.toExponential(2)} ${dv.toExponential(2)}`);
  gcheck('regional nest (open BC, land/sea surface), 10 steps: w rel L2 < 1e-2', dw < 1e-2, dw);
  gcheck('regional nest (open BC, land/sea surface), 10 steps: theta rel L2 < 1e-5', dth < 1e-5, dth);
  gcheck('regional nest (open BC, land/sea surface), 10 steps: pi\' rel L2 < 1e-3', dpp < 1e-3, dpp);
  gcheck('regional nest (open BC, land/sea surface), 10 steps: qv rel L2 < 1e-3', dq < 1e-3, dq);
}

/** Six-class ice microphysics: GPU vs CPU in a deep convective cloud (Weisman–Klemp sounding). */
/** Two-way nest in a cylinder (twoway.ts, nestGpu.ts): GPU targets and feedback against the CPU, and a coupled run. */
export async function regionalTwoWayTest(): Promise<void> {
  const device = await getDevice();
  const { nestGeometry, nestTargets, nestFeedback, emptyTargets, parentState, nestRelax } = await import('../../regional/twoway.js');
  const { refineInto } = await import('../../regional/refine.js');
  const { GpuNest } = await import('../nestGpu.js');
  const pcfg = { nx: 32, ny: 32, nz: 12, dx: 4000, dy: 4000, dz: 1000, dt: 12, nsound: 6, f: 5e-5, beta: 0.2, divDamp: 0.1, dampDepth: 3000, dampRate: 1 / 300, kdiff2: 0, lateral: 'periodic' as const };
  const mk = (): { p: RegionalModel; mp: KesslerMicrophysics } => {
    const p = new RegionalModel(pcfg, weismanKlemp, 3), mp = new KesslerMicrophysics(p);
    p.setBaseWind((z) => ({ u: 5 * Math.tanh(z / 3000), v: -2 }));
    for (let k = 0; k < pcfg.nz; k++) for (let j = 0; j < 32; j++) for (let i = 0; i < 32; i++) {
      const q = p.idx(i, j, k), r = Math.hypot(((i + 0.5) * 4000 - 64000) / 10000, ((j + 0.5) * 4000 - 64000) / 10000, (p.zc[k]! - 1500) / 1500);
      p.scalars[QV]![q] = p.qv0[k]!;
      if (r < 1) p.th[q] = p.th[q]! + 2.5 * Math.cos(0.5 * Math.PI * r) ** 2;
    }
    for (let s = 0; s < 40; s++) { p.step(); mp.apply(12); }       // 8 min: a growing cloud at the centre
    return { p, mp };
  };
  const { p, mp } = mk();
  const g = nestGeometry(p.c, 24000, 2000, 500);
  if (typeof g === 'string') throw new Error(g);
  const mkChild = (): { c: RegionalModel; cp: KesslerMicrophysics } => {
    const c = new RegionalModel({ ...pcfg, nx: g.nx, ny: g.nx, nz: g.nz, dx: g.dx, dy: g.dx, dz: g.dz, dt: 6, lateral: 'open', ...nestRelax(g) }, weismanKlemp, 3);
    c.setBaseWind((z) => ({ u: 5 * Math.tanh(z / 3000), v: -2 }));
    c.boundary = emptyTargets(c);
    refineInto(p, c, g.i0 * 4000, g.j0 * 4000);
    nestTargets(p, null, c, g, 1, c.boundary);
    return { c, cp: new KesslerMicrophysics(c) };
  };
  const { c, cp } = mkChild();
  const gp = new GpuRegional(device, p, { moist: true, physics: null }), gc = new GpuRegional(device, c, { moist: true, physics: null, boundary: c.boundary });
  gp.uploadFrom(p); gc.uploadFrom(c);
  const link = new GpuNest(gp, gc, g);
  // targets after one parent step, half-way through it; the CPU reference from the GPU parent's own states at the start
  // and the end of the step (the kernel alone: CPU and GPU parents differ by f32 rounding, most visibly in pi')
  p.step(); mp.apply(12);
  gp.step(1);
  const gNow = await gp.readState(), gOld = await gp.readBuffer(gp.S0, gp.nf * p.size * 4);
  const pg = new RegionalModel(pcfg, weismanKlemp, 3);
  pg.setBaseWind((z) => ({ u: 5 * Math.tanh(z / 3000), v: -2 }));
  [pg.u, pg.v, pg.w, pg.th, pg.pp, ...pg.scalars].forEach((a, f) => { for (let i = 0; i < p.size; i++) a[i] = gNow[f * p.size + i]!; });
  const fOld = (f: number): Float64Array => Float64Array.from(gOld.subarray(f * p.size, (f + 1) * p.size));
  const st = { u: fOld(0), v: fOld(1), th: fOld(3), pp: fOld(4), qv: fOld(5) };
  const tg = emptyTargets(c);
  nestTargets(pg, st, c, g, 0.5, tg);
  link.targets(0.5);
  const B = await gc.readBuffer(gc.B!, 5 * c.size * 4);
  const rel = (f: number, a: Float64Array): number => {
    let e = 0, s = 0;
    for (let k = 0; k < c.c.nz; k++) for (let j = 0; j < c.c.ny; j++) for (let i = 0; i < c.c.nx; i++) {
      if (Math.hypot(i + 0.5 - c.c.nx / 2, j + 0.5 - c.c.ny / 2) <= c.c.relaxCyl!.r) continue;
      const q = c.idx(i, j, k); e += (B[f * c.size + q]! - a[q]!) ** 2; s += a[q]! ** 2;
    }
    return Math.sqrt(e / Math.max(s, 1e-300));
  };
  const et = [rel(0, tg.u), rel(1, tg.v), rel(2, tg.th), rel(3, tg.qv!), rel(4, tg.pp!)];
  gcheck('two-way nest: GPU relaxation targets match the CPU (u, v, theta, qv, pi\'; rel L2 < 1e-4)', et.every((x) => x < 1e-4), et.map((x) => x.toExponential(1)).join(' '));
  // feedback of a changed child (warmer and moister in the middle) into the parent
  for (let k = 0; k < c.c.nz; k++) for (let j = 0; j < c.c.ny; j++) for (let i = 0; i < c.c.nx; i++) {
    const q = c.idx(i, j, k), r = Math.hypot(i + 0.5 - c.c.nx / 2, j + 0.5 - c.c.ny / 2) / 8;
    if (r < 1) { c.th[q] = c.th[q]! + 1.5 * (1 - r); c.scalars[QV]![q] = c.scalars[QV]![q]! * (1 + 0.1 * (1 - r)); c.u[q] = c.u[q]! + 3 * (1 - r); }
  }
  gc.uploadFrom(c); gp.uploadFrom(p);
  nestFeedback(c, p, g);
  link.feedback();
  const sp = await gp.readState();
  const ef = [cmp(p, sp, 0, p.u, pcfg.nz), cmp(p, sp, 2, p.w, pcfg.nz + 1), cmp(p, sp, 3, p.th, pcfg.nz), cmp(p, sp, 5, p.scalars[QV]!, pcfg.nz), cmp(p, sp, 6, p.scalars[QC]!, pcfg.nz)];
  gcheck('two-way nest: GPU feedback matches the CPU (u, w, theta, qv, qc; rel L2 < 1e-5, qc < 1e-4)', ef[0]! < 1e-5 && ef[1]! < 1e-5 && ef[2]! < 1e-7 && ef[3]! < 1e-5 && ef[4]! < 1e-4, ef.map((x) => x.toExponential(1)).join(' '));
  // a coupled run of 10 parent steps (2 child sub-steps each) on both
  const a = mk(), b = mkChild();
  const P = a.p, Pm = a.mp;
  // (mkChild used the first parent; rebuild the child from this one)
  refineInto(P, b.c, g.i0 * 4000, g.j0 * 4000); nestTargets(P, null, b.c, g, 1, b.c.boundary!);
  const gP = new GpuRegional(device, P, { moist: true, physics: null }), gC = new GpuRegional(device, b.c, { moist: true, physics: null, boundary: b.c.boundary });
  gP.uploadFrom(P); gC.uploadFrom(b.c);
  const L2 = new GpuNest(gP, gC, g);
  for (let s = 0; s < 10; s++) {
    const s0 = parentState(P);
    P.step(); Pm.apply(12);
    for (let n = 0; n < 2; n++) { nestTargets(P, s0, b.c, g, (n + 0.5) / 2, b.c.boundary!); b.c.step(); b.cp.apply(6); }
    nestFeedback(b.c, P, g);
  }
  for (let s = 0; s < 10; s++) L2.step(2);
  const sP = await gP.readState(), sC = await gC.readState();
  const dP = [cmp(P, sP, 2, P.w, pcfg.nz + 1), cmp(P, sP, 3, P.th, pcfg.nz), cmp(P, sP, 6, P.scalars[QC]!, pcfg.nz)];
  const dC = [cmp(b.c, sC, 2, b.c.w, b.c.c.nz + 1), cmp(b.c, sC, 3, b.c.th, b.c.c.nz), cmp(b.c, sC, 6, b.c.scalars[QC]!, b.c.c.nz)];
  let wmax = 0; for (let q = 0; q < b.c.size; q++) wmax = Math.max(wmax, Math.abs(b.c.w[q]!));
  gcheck('two-way nest: 10 coupled steps on the GPU follow the CPU (w < 2e-2, theta < 1e-5, qc < 5e-2 rel L2, outer and inner) with a rising cloud in the nest',
    dP[0]! < 2e-2 && dP[1]! < 1e-5 && dP[2]! < 5e-2 && dC[0]! < 2e-2 && dC[1]! < 1e-5 && dC[2]! < 5e-2 && wmax > 1,
    `outer ${dP.map((x) => x.toExponential(1)).join(' ')}, inner ${dC.map((x) => x.toExponential(1)).join(' ')}, inner w max ${wmax.toFixed(1)} m/s`);
  for (const x of [gp, gc, gP, gC]) x.destroy();
  link.destroy(); L2.destroy();
}

export async function regionalIceTests(): Promise<void> {
  const device = await getDevice();
  const nx = 20, nz = 30, dx = 3000, dz = 600;
  const m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt: 6, nsound: 6, f: 0, beta: 0.2, divDamp: 0.1, dampDepth: 5000, dampRate: 1 / 300, kdiff2: 0 }, weismanKlemp, 6);
  const mp = new IceMicrophysics(m);
  m.setBaseWind((z) => ({ u: 10 * Math.tanh(z / 3000) - 5, v: 2 }));
  for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) m.scalars[QV]![m.idx(i, j, k)] = m.qv0[k]!;
  const c = nx * dx / 2;
  for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
    const r = Math.sqrt((((i + 0.5) * dx - c) / 9000) ** 2 + (((j + 0.5) * dx - c) / 9000) ** 2 + ((m.zc[k]! - 1400) / 1400) ** 2);
    if (r < 1) m.th[m.idx(i, j, k)] = m.th[m.idx(i, j, k)]! + 2 * Math.cos(0.5 * Math.PI * r) ** 2;
  }
  for (let s = 0; s < 200; s++) { m.step(); mp.apply(6); }     // 20 min: glaciating anvil
  const g = new GpuRegional(device, m, { moist: true, physics: null, ice: true });
  g.uploadFrom(m);
  for (const n of [1, 10]) {
    const todo = n === 1 ? 1 : 9;
    for (let s = 0; s < todo; s++) { m.step(); mp.apply(6); }
    g.step(todo);
    const st = await g.readState();
    const du = cmp(m, st, 0, m.u, nz), dth = cmp(m, st, 3, m.th, nz);
    const d = [QV, QC, QR, QI, QS, QG].map((sp) => cmp(m, st, 5 + sp, m.scalars[sp]!, nz));
    const tol = n === 1 ? 1e-4 : 2e-3;
    gcheck(`regional ice, ${n} steps: u rel L2 < ${tol}`, du < tol, du);
    gcheck(`regional ice, ${n} steps: theta rel L2 < ${tol / 10}`, dth < tol / 10, dth);
    gcheck(`regional ice, ${n} steps: qv qc qr qi qs qg rel L2 < ${tol * 20}`, d.every((x) => x < tol * 20), d.map((x) => x.toExponential(1)).join(' '));
  }
  let qi = 0, qs = 0, qg = 0;
  for (let q = 0; q < m.size; q++) { qi = Math.max(qi, m.scalars[QI]![q]!); qs = Math.max(qs, m.scalars[QS]![q]!); qg = Math.max(qg, m.scalars[QG]![q]!); }
  gcheck('regional ice test case exercises the ice paths (qi, qg > 0.05 g/kg; snow present)', qi > 5e-5 && qs > 1e-8 && qg > 5e-5, `qi ${(qi * 1e3).toFixed(2)} qs ${(qs * 1e3).toFixed(4)} qg ${(qg * 1e3).toFixed(2)} g/kg`);
}

/** Timing of one GPU step for the app's supercell configuration (Kessler vs ice). */
export async function regionalPerf(): Promise<void> {
  const device = await getDevice();
  for (const [ice, env] of [[false, false], [true, false], [true, true]] as const) {
    const nx = 60, nz = 40, dx = 2000;
    const m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz: 500, dt: 6, nsound: 6, f: 0, beta: 0.2, divDamp: 0.1, dampDepth: 6000, dampRate: 1 / 300, kdiff2: 0 }, weismanKlemp, ice ? 6 : 3);
    if (env) {   // the app's supercell environment: shear and a moist boundary layer
      m.setBaseWind((z) => ({ u: 30 * Math.tanh(z / 3000) - 15, v: 0 }));
      for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) m.scalars[QV]![m.idx(i, j, k)] = m.qv0[k]!;
    }
    const g = new GpuRegional(device, m, { moist: true, physics: null, ice });
    g.uploadFrom(m);
    g.step(1); await device.queue.onSubmittedWorkDone();
    const t0 = performance.now();
    g.step(3); await device.queue.onSubmittedWorkDone();
    gcheck(`perf supercell 60x60x40 ice=${ice} env=${env}: ms per step`, true, ((performance.now() - t0) / 3).toFixed(0));
  }
}

/** Adaptive time stepping (as in the app's regional worker) vs the fixed configured dt: a sheared
 *  warm-bubble storm run for 40 model minutes should give a similar storm with fewer steps. */
export async function regionalAdaptiveTest(): Promise<void> {
  const device = await getDevice();
  const run = async (adaptive: boolean): Promise<{ wmax: number; cmax: number; rain: number; steps: number; dtEnd: number }> => {
    const nx = 40, nz = 30, dx = 2000, dz = 500, dt0 = 6;
    const m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt: dt0, nsound: 6, f: 0, beta: 0.2, divDamp: 0.1, dampDepth: 5000, dampRate: 1 / 300, kdiff2: 0 }, weismanKlemp, 6);
    m.setBaseWind((z) => ({ u: 30 * Math.tanh(z / 3000) - 15, v: 0 }));
    for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) m.scalars[QV]![m.idx(i, j, k)] = m.qv0[k]!;
    const xc = nx * dx * 0.4, yc = nx * dx / 2;
    for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
      const r = Math.sqrt((((i + 0.5) * dx - xc) / 10000) ** 2 + (((j + 0.5) * dx - yc) / 10000) ** 2 + ((m.zc[k]! - 1400) / 1400) ** 2);
      if (r < 1) m.th[m.idx(i, j, k)] = m.th[m.idx(i, j, k)]! + 2 * Math.cos(0.5 * Math.PI * r) ** 2;
    }
    const g = new GpuRegional(device, m, { moist: true, physics: null, ice: true });
    g.uploadFrom(m);
    const hi = Math.max(dt0, Math.min(3 * dt0, 6 * 0.45 * dx / 350));
    while (g.time < 2400 - 1e-6) {
      const n = Math.max(1, Math.min(20, Math.ceil(Math.min(300, 2400 - g.time) / g.dt)));
      g.step(n);
      await device.queue.onSubmittedWorkDone();
      if (adaptive) {
        const rate = await g.maxCourantRate(), cur = g.dt;
        let next = Math.min(hi, 0.8 / Math.max(rate, 1e-9));
        if (rate * cur > 1.1) next = Math.min(next, 0.7 / rate); else if (next > cur) next = Math.min(next, 1.1 * cur);
        next = Math.max(0.25 * dt0, Math.min(next, Math.max(0.5, 2400 - g.time)));
        if (Math.abs(next - cur) > 0.02 * cur) g.setDt(next);
      }
    }
    const d = await g.readDisplay([0]);
    let wmax = 0, cmax = 0, rain = 0;
    for (let c = 0; c < nx * nx; c++) { wmax = Math.max(wmax, d.col[COL * c]!); cmax = Math.max(cmax, d.col[COL * c + 2]!); }
    for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) rain += d.rain[m.idx(i, j, 0)]!;
    const out = { wmax, cmax, rain, steps: g.steps, dtEnd: g.dt };
    g.destroy();
    return out;
  };
  const a = await run(false), b = await run(true);
  gcheck('adaptive dt: storm develops (fixed dt: w max > 10 m/s)', a.wmax > 10, a.wmax.toFixed(1));
  gcheck('adaptive dt: w max within 20 % of fixed dt', Math.abs(b.wmax - a.wmax) < 0.2 * a.wmax, `${b.wmax.toFixed(1)} vs ${a.wmax.toFixed(1)}`);
  gcheck('adaptive dt: max condensate within 25 %', Math.abs(b.cmax - a.cmax) < 0.25 * a.cmax, `${(b.cmax * 1e3).toFixed(2)} vs ${(a.cmax * 1e3).toFixed(2)} g/kg`);
  gcheck('adaptive dt: domain rain within 30 %', Math.abs(b.rain - a.rain) < 0.3 * Math.max(a.rain, 1e-9), `${b.rain.toFixed(1)} vs ${a.rain.toFixed(1)}`);
  gcheck('adaptive dt: fewer steps', b.steps < a.steps, `${b.steps} vs ${a.steps} steps, final dt ${b.dtEnd.toFixed(1)} s`);
}

/** Debug: fraction of columns flagged as containing condensate, and the condensate magnitudes. */
export async function regionalFlagDebug(): Promise<void> {
  const device = await getDevice();
  const nx = 60, nz = 40, dx = 2000;
  const m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz: 500, dt: 6, nsound: 6, f: 0, beta: 0.2, divDamp: 0.1, dampDepth: 6000, dampRate: 1 / 300, kdiff2: 0 }, weismanKlemp, 6);
  m.setBaseWind((z) => ({ u: 30 * Math.tanh(z / 3000) - 15, v: 0 }));
  for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) m.scalars[QV]![m.idx(i, j, k)] = m.qv0[k]!;
  const g = new GpuRegional(device, m, { moist: true, physics: null, ice: true });
  g.uploadFrom(m);
  for (const n of [1, 5, 20]) {
    g.step(n); await device.queue.onSubmittedWorkDone();
    const st = await g.readState();
    const hist = new Map<number, number>();
    let raw = 0;
    for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
      let mx = 0;
      for (let k = 0; k < nz; k++) for (let f = 6; f < g.nf; f++) mx = Math.max(mx, Math.abs(st[f * m.size + m.idx(i, j, k)]!));
      if (mx > 0) raw++;
      const b = mx === 0 ? -99 : Math.floor(Math.log10(mx));
      hist.set(b, (hist.get(b) ?? 0) + 1);
    }
    gcheck(`flag debug after ${g.steps} steps: columns with any condensate`, true, `${raw}/${nx * nx}; log10 max histogram ${[...hist].sort((a, b) => a[0] - b[0]).map(([b, c]) => `${b}:${c}`).join(' ')}`);
  }
}

/** Coarse-to-fine refinement of a developed storm (2 km -> 1 km on the GPU): the updraft and the
 *  condensate carry over and the fine run continues without a jump or blow-up. */
export async function regionalRefineTest(): Promise<void> {
  const { refineInto } = await import('../../regional/refine.js');
  const device = await getDevice();
  const L = 80000;
  const mk = (dx: number, nz: number, dz: number, dt: number): RegionalModel => {
    const nx = L / dx;
    const m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt, nsound: 6, f: 0, beta: 0.2, divDamp: 0.1, dampDepth: 5000, dampRate: 1 / 300, kdiff2: 0 }, weismanKlemp, 6);
    m.setBaseWind((z) => ({ u: 30 * Math.tanh(z / 3000) - 15, v: 0 }));
    for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) m.scalars[QV]![m.idx(i, j, k)] = m.qv0[k]!;
    return m;
  };
  const mc = mk(2000, 30, 500, 6);
  for (let k = 0; k < 30; k++) for (let j = 0; j < 40; j++) for (let i = 0; i < 40; i++) {
    const r = Math.sqrt((((i + 0.5) * 2000 - 0.4 * L) / 10000) ** 2 + (((j + 0.5) * 2000 - 0.5 * L) / 10000) ** 2 + ((mc.zc[k]! - 1400) / 1400) ** 2);
    if (r < 1) mc.th[mc.idx(i, j, k)] = mc.th[mc.idx(i, j, k)]! + 2 * Math.cos(0.5 * Math.PI * r) ** 2;
  }
  const stats = async (g: GpuRegional, nx: number): Promise<{ w: number; c: number }> => {
    const d = await g.readDisplay([0]); let w = 0, c = 0;
    for (let i = 0; i < nx * nx; i++) { w = Math.max(w, d.col[COL * i]!); c = Math.max(c, d.col[COL * i + 2]!); }
    return { w, c };
  };
  const gc = new GpuRegional(device, mc, { moist: true, physics: null, ice: true });
  gc.uploadFrom(mc);
  gc.step(300); await device.queue.onSubmittedWorkDone();                 // 30 min
  const before = await stats(gc, 40);
  const st = await gc.readState();
  [mc.u, mc.v, mc.w, mc.th, mc.pp, ...mc.scalars].forEach((a, f) => { for (let i = 0; i < mc.size; i++) a[i] = st[f * mc.size + i]!; });
  mc.time = gc.time; gc.destroy();
  const mf = mk(1000, 45, 333.3333, 3);
  refineInto(mc, mf);
  const gf = new GpuRegional(device, mf, { moist: true, physics: null, ice: true });
  gf.uploadFrom(mf); gf.time = mf.time;
  gf.step(1); await device.queue.onSubmittedWorkDone();
  const just = await stats(gf, 80);
  gf.step(99); await device.queue.onSubmittedWorkDone();                  // +5 min
  const after = await stats(gf, 80);
  gf.destroy();
  gcheck('refine GPU: storm developed on the coarse grid (w max > 15 m/s at 30 min)', before.w > 15, before.w.toFixed(1));
  gcheck('refine GPU: updraft carried over (w max within 30 % right after refining)', Math.abs(just.w - before.w) < 0.3 * before.w, `${just.w.toFixed(1)} vs ${before.w.toFixed(1)} m/s`);
  gcheck('refine GPU: condensate carried over (within 30 %)', Math.abs(just.c - before.c) < 0.3 * before.c, `${(just.c * 1e3).toFixed(2)} vs ${(before.c * 1e3).toFixed(2)} g/kg`);
  gcheck('refine GPU: fine run continues (5 min later: storm alive, no blow-up)', Number.isFinite(after.w) && after.w > 0.5 * before.w && after.w < 80, `${after.w.toFixed(1)} m/s at t = ${(gf.time / 60).toFixed(0)} min`);
}

/** Chart diagnostics: the GPU display kernels (column composites with CAPE / CIN, column profiles,
 *  azimuthal means) agree with the CPU versions on the same state (a storm in shear, ice microphysics). */
export async function regionalChartsTest(): Promise<void> {
  const device = await getDevice();
  const nx = 24, nz = 30, dx = 2500, dz = 500;
  const m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt: 6, nsound: 6, f: 0, beta: 0.2, divDamp: 0.1, dampDepth: 5000, dampRate: 1 / 300, kdiff2: 0 }, weismanKlemp, 6);
  const mp = new IceMicrophysics(m);
  m.setBaseWind((z) => ({ u: 20 * Math.tanh(z / 3000) - 10, v: 4 * Math.sin(z / 2000) }));
  for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) m.scalars[QV]![m.idx(i, j, k)] = m.qv0[k]!;
  const c = nx * dx / 2;
  for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
    const r = Math.sqrt((((i + 0.5) * dx - c) / 9000) ** 2 + (((j + 0.5) * dx - c) / 9000) ** 2 + ((m.zc[k]! - 1400) / 1400) ** 2);
    if (r < 1) m.th[m.idx(i, j, k)] = m.th[m.idx(i, j, k)]! + 2 * Math.cos(0.5 * Math.PI * r) ** 2;
  }
  for (let s = 0; s < 250; s++) { m.step(); mp.apply(6); }     // 25 min: mature cell with rain, graupel and anvil
  const g = new GpuRegional(device, m, { moist: true, physics: null, ice: true });
  g.uploadFrom(m);
  const d = await g.readDisplay([0], 0);
  const cpu = columnDiagnostics(m);
  {
    // 3-D view bytes (cloud, precipitation) as the worker computes them on the CPU
    const qc = m.scalars[QC]!, qr = m.scalars[QR]!, qi = m.scalars[QI]!, qs = m.scalars[QS]!, qg = m.scalars[QG]!, qv = m.scalars[QV]!, rhc = subgridRHc(dx);
    const bad = [0, 0]; let ncl = 0, nice = 0;
    for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
      const q = m.idx(i, j, k), o = (k * nx + j) * nx + i, rho = m.rho0[k]!, pi = m.pi0[k]! + m.pp[q]!, T = m.th[q]! * pi;
      const qsub = qc[q]! <= 1e-8 ? subgridCloud(qv[q]!, qsatW(T, 1e5 * Math.pow(pi, 1004.5 / 287.05)), rhc) : 0;
      const bc = cloudExtinction(rho, qc[q]!, qsub, qi[q]!, qs[q]!, T);
      const want = [extByte(bc), extByte(precipExtinction(rho, qr[q]!, qg[q]!))];
      const v = d.packed[o]!, got = [v & 255, (v >>> 8) & 255];
      for (let c = 0; c < 2; c++) if (Math.abs(got[c]! - want[c]!) > 1) bad[c]!++;
      if (want[0]! > 0) ncl++; if (want[0]! > 0 && qi[q]! + qs[q]! > qc[q]!) nice++;
      if (v >>> 16) bad[0]!++;
    }
    const lim = 0.001 * nx * nx * nz;
    gcheck('charts: GPU 3-D view bytes (cloud, precipitation) match the CPU within 1 in 99.9 % of cells; liquid and ice cloud present',
      bad.every((b) => b <= lim) && ncl > 100 && nice > 10, `cells off by > 1: ${bad.join(' ')} of ${nx * nx * nz}; cloudy ${ncl}, ice-dominated ${nice}`);
  }
  const rel = (f: number): number => {
    let e = 0, n = 0;
    for (let q = 0; q < nx * nx; q++) { e += (d.col[COL * q + f]! - cpu[COL * q + f]!) ** 2; n += cpu[COL * q + f]! ** 2; }
    return Math.sqrt(e / Math.max(n, 1e-30));
  };
  const basic = [C.wmax, C.wmin, C.cmax, C.pmax, C.ctopT].map(rel);
  gcheck('charts: GPU column extremes and cloud-top temperature match the CPU (rel L2 < 1e-4)', basic.every((x) => x < 1e-4), basic.map((x) => x.toExponential(1)).join(' '));
  const sat = [C.vis, C.pw, C.wvT, C.visZ].map(rel);
  gcheck('charts: GPU cloud albedo, precipitable water, water-vapour channel and visible cloud top match the CPU (rel L2 < 1e-3)', sat.every((x) => x < 1e-3), sat.map((x) => x.toExponential(1)).join(' '));
  let dz2 = 0, ctzMis = 0;
  for (let q = 0; q < nx * nx; q++) {
    if (cpu[COL * q + C.dbz]! > 0) dz2 = Math.max(dz2, Math.abs(d.col[COL * q + C.dbz]! - cpu[COL * q + C.dbz]!));
    if (d.col[COL * q + C.ctopZ] !== cpu[COL * q + C.ctopZ]) ctzMis++;
  }
  gcheck('charts: column-max reflectivity within 0.05 dBZ, cloud-top height identical in 99 % of columns', dz2 < 0.05 && ctzMis <= 0.01 * nx * nx, `${dz2.toExponential(2)} dB, ${ctzMis} columns differ`);
  const uh = rel(C.uh), cape = rel(C.cape), cin = rel(C.cin);
  gcheck('charts: updraft helicity rel L2 < 1e-3, CAPE < 1e-2, CIN < 5e-2', uh < 1e-3 && cape < 1e-2 && cin < 5e-2, `${uh.toExponential(1)} ${cape.toExponential(1)} ${cin.toExponential(1)}`);
  {
    // severe-weather indices: wind indices from the same interpolated samples, echo top, VIL, LCL and lifted index
    const wind = [C.shear, C.srh1, C.srh3, C.u850, C.v850, C.u200, C.v200, C.shear01].map(rel), vil = rel(C.vil);
    let lclMis = 0, etMis = 0, liErr = 0, srhMax = 0, shMax = 0, vilMax = 0, etMax = 0;
    for (let q = 0; q < nx * nx; q++) {
      const o = COL * q;
      if (d.col[o + C.lcl] !== cpu[o + C.lcl]) lclMis++;
      if (d.col[o + C.etop] !== cpu[o + C.etop]) etMis++;
      liErr = Math.max(liErr, Math.abs(d.col[o + C.li]! - cpu[o + C.li]!));
      srhMax = Math.max(srhMax, Math.abs(cpu[o + C.srh3]!)); shMax = Math.max(shMax, cpu[o + C.shear]!); vilMax = Math.max(vilMax, cpu[o + C.vil]!); etMax = Math.max(etMax, cpu[o + C.etop]!);
    }
    gcheck('charts: GPU 0-6 and 0-1 km shear, 0-1 / 0-3 km helicity, 850 / 200 hPa winds (rel L2 < 1e-4) and VIL (< 1e-3) match the CPU; LCL and echo top identical in 98 % of columns, lifted index within 0.02 K; all present',
      wind.every((x) => x < 1e-4) && vil < 1e-3 && lclMis <= 0.02 * nx * nx && etMis <= 0.02 * nx * nx && liErr < 0.02 && srhMax > 10 && shMax > 10 && vilMax > 1 && etMax > 5000,
      `${wind.map((x) => x.toExponential(1)).join(' ')} ${vil.toExponential(1)}; LCL ${lclMis}, echo top ${etMis} columns differ, LI ${liErr.toExponential(1)} K; SRH ${srhMax.toFixed(0)} m2/s2, shear ${shMax.toFixed(1)} m/s, VIL ${vilMax.toFixed(1)} kg/m2, echo top ${(etMax / 1000).toFixed(1)} km`);
  }
  let capeMax = 0, uhMax = 0, dbzMax = 0;
  for (let q = 0; q < nx * nx; q++) { capeMax = Math.max(capeMax, cpu[COL * q + C.cape]!); uhMax = Math.max(uhMax, Math.abs(cpu[COL * q + C.uh]!)); dbzMax = Math.max(dbzMax, cpu[COL * q + C.dbz]!); }
  gcheck('charts: test storm exercises the paths (CAPE > 1000 J/kg, |UH| > 1 m2/s2, > 40 dBZ)', capeMax > 1000 && uhMax > 1 && dbzMax > 40, `CAPE ${capeMax.toFixed(0)} J/kg, UH ${uhMax.toFixed(1)}, ${dbzMax.toFixed(1)} dBZ`);
  const pts = [{ i: 0, j: 0 }, { i: 12, j: 12 }, { i: 23, j: 5 }, { i: 7, j: 23 }];
  const gc = await g.readColumns(pts), cc = columnProfiles(m, pts);
  let e = 0, n = 0;
  for (let i = 0; i < cc.length; i++) { e += (gc[i]! - cc[i]!) ** 2; n += cc[i]! ** 2; }
  gcheck('charts: GPU column profiles match the CPU (rel L2 < 1e-6)', gc.length === cc.length && Math.sqrt(e / n) < 1e-6, Math.sqrt(e / n));
  const xc = 0.55 * nx * dx, yc = 0.45 * nx * dx, dr = dx, nr = 10;
  const gr = await g.readRZ(xc, yc, dr, nr), cr = azimuthalMeans(m, xc, yc, dr, nr);
  const errs = [0, 1, 2, 3, 4].map((f) => {
    let ee = 0, nn = 0;
    for (let t = 0; t < nr * nz; t++) { ee += (gr[5 * t + f]! - cr[5 * t + f]!) ** 2; nn += cr[5 * t + f]! ** 2; }
    return Math.sqrt(ee / Math.max(nn, 1e-30));
  });
  gcheck('charts: GPU azimuthal means (vt, vr, w, theta\', condensate) match the CPU (rel L2 < 1e-3)', gr.length === cr.length && errs.every((x) => x < 1e-3), errs.map((x) => x.toExponential(1)).join(' '));
  // tracer particles: advection (no re-seeding) and re-seeding agree with the CPU version
  const pr = { life: 1e9, cx: 0.5 * nx * dx, cy: 0.5 * nx * dx, rad: 15000, zSeed: 2000 };
  const tc = new Tracers(512, m, pr);
  for (let p = 0; p < 512; p++) tc.pos[4 * p + 2] = 200 + (p % 37) * 250;      // spread through the troposphere
  g.initTracers(Float32Array.from(tc.pos));
  tc.seed = 5; tc.advect(120, 12);
  g.advectTracers(120, 12, pr, Math.imul(6, 2654435761) >>> 0);
  let tg = await g.readTracers(), dmax = 0, moved = 0;
  for (let p = 0; p < 512; p++) { for (let c = 0; c < 3; c++) dmax = Math.max(dmax, Math.abs(tg[4 * p + c]! - tc.pos[4 * p + c]!)); }
  for (let p = 0; p < 512; p++) moved = Math.max(moved, Math.abs(tc.pos[4 * p + 2]! - (200 + (p % 37) * 250)));
  gcheck('tracers: GPU advection matches the CPU (max position difference < 1 m after 2 min)', dmax < 1 && moved > 50, `${dmax.toExponential(2)} m (max vertical displacement ${moved.toFixed(0)} m)`);
  const pr2 = { ...pr, life: 60 };
  tc.params = pr2; tc.seed = 9; tc.advect(120, 4);
  g.advectTracers(120, 4, pr2, Math.imul(10, 2654435761) >>> 0);
  tg = await g.readTracers(); dmax = 0;
  let inLayer = true;
  let worst = '';
  for (let p = 0; p < 512; p++) {
    for (let c = 0; c < 3; c++) { const d = Math.abs(tg[4 * p + c]! - tc.pos[4 * p + c]!); if (d > dmax) { dmax = d; worst = `p ${p} c ${c}: ${tg[4 * p + c]} vs ${tc.pos[4 * p + c]}`; } }
    if (tg[4 * p + 2]! > 2000 || tg[4 * p + 3] !== 0) inLayer = false;
  }
  // WGSL sin / cos are only accurate to about 2^-11 (absolute): metres over a 15 km seeding radius
  gcheck('tracers: re-seeding matches the CPU (positions < 10 m, all re-seeded in the lowest 2 km)', dmax < 10 && inLayer, `${dmax.toExponential(2)} m ${worst}`);
  g.destroy();
}

// Interactions made on the GPU (GpuRegional.edit, windOnce) against the same changes to the CPU model (worker.ts), and
// the GPU copy of the state for the undo
export async function regionalEditTest(): Promise<void> {
  const device = await getDevice();
  const cfg = { nx: 24, ny: 24, nz: 16, dx: 3000, dy: 3000, dz: 800, dt: 8, nsound: 6, f: 5e-5, beta: 0.2, divDamp: 0.1, dampDepth: 3000, dampRate: 1 / 300, kdiff2: 0, lateral: 'periodic' as const };
  const m = new RegionalModel(cfg, weismanKlemp, 3), mp = new KesslerMicrophysics(m);
  m.setBaseWind((z) => ({ u: 4 * Math.tanh(z / 3000), v: -1 }));
  for (let k = 0; k < cfg.nz; k++) for (let j = 0; j < 24; j++) for (let i = 0; i < 24; i++) {
    const q = m.idx(i, j, k), r = Math.hypot(((i + 0.5) * 3000 - 36000) / 9000, ((j + 0.5) * 3000 - 36000) / 9000, (m.zc[k]! - 1500) / 1500);
    m.scalars[QV]![q] = m.qv0[k]!;
    if (r < 1) m.th[q] = m.th[q]! + 2 * Math.cos(0.5 * Math.PI * r) ** 2;
  }
  for (let s = 0; s < 20; s++) { m.step(); mp.apply(8); }
  const g = new GpuRegional(device, m, { moist: true, physics: null });
  g.uploadFrom(m);
  const before = await g.readState();
  // the CPU changes (as worker.ts interact): a warm bubble seen from an origin offset (ox, oy), drying capped at
  // saturation across the periodic edge, a one-time rotating wind
  const bub = { x: 40000, y: 30000, z: 1800, rh: 12000, rz: 1500, amp: 3, cap: 6, ox: -4000, oy: 2000 };
  for (let k = 0; k < cfg.nz; k++) for (let j = 0; j < 24; j++) for (let i = 0; i < 24; i++) {
    const r = Math.sqrt(((bub.ox + (i + 0.5) * 3000 - bub.x) / bub.rh) ** 2 + ((bub.oy + (j + 0.5) * 3000 - bub.y) / bub.rh) ** 2 + ((m.zc[k]! - bub.z) / bub.rz) ** 2);
    if (r >= 1) continue;
    const q = m.idx(i, j, k), d = bub.amp * Math.cos(0.5 * Math.PI * r) ** 2, room = m.th0[k]! + bub.cap - m.th[q]!;
    m.th[q] = m.th[q]! + Math.max(0, Math.min(d, room));
  }
  const mo = { x: 70000, y: 5000, z: 2500, R: 15000, H: 2000, fac: 0.6 }, qv = m.scalars[QV]!;
  for (let k = 0; k < cfg.nz; k++) for (let j = 0; j < 24; j++) for (let i = 0; i < 24; i++) {
    let ex = (i + 0.5) * 3000 - mo.x, ey = (j + 0.5) * 3000 - mo.y;
    ex -= Math.round(ex / 72000) * 72000; ey -= Math.round(ey / 72000) * 72000;
    const rh = Math.hypot(ex, ey) / mo.R, rz = Math.abs(m.zc[k]! - mo.z) / mo.H;
    if (rh >= 1 || rz >= 1) continue;
    const e = Math.cos(0.5 * Math.PI * rh) ** 2 * Math.cos(0.5 * Math.PI * rz) ** 2, q = m.idx(i, j, k);
    const pi = m.pi0[k]! + m.pp[q]!, T = m.th[q]! * pi, p = 1e5 * Math.pow(pi, 1004.5 / 287.05), es = 611.2 * Math.exp(17.67 * (T - 273.15) / (T - 29.65));
    qv[q] = Math.max(0, Math.min(qv[q]! * (1 + (mo.fac - 1) * e), Math.max(qv[q]!, 0.622 * es / Math.max(p - es, 1))));
  }
  const wf: WindForcing = { x: 30000, y: 40000, z: 2000, R: 15000, H: 1500, speed: 8, dir: [0, 0, 0], form: 'rotate', sign: 1 };
  applyWind(m, [wf], 'once');
  g.edit({ kind: 'bubble', ...bub, warm: true });
  g.edit({ kind: 'moisture', ...mo, ox: 0, oy: 0 });
  g.windOnce(wf);
  const st = await g.readState();
  const eTh = cmp(m, st, 3, m.th, cfg.nz), eQv = cmp(m, st, 5, qv, cfg.nz), eU = cmp(m, st, 0, m.u, cfg.nz), eV = cmp(m, st, 1, m.v, cfg.nz);
  let mU = 0, mV = 0, mQ = 0, wV = 0, wQ = 0;
  for (let q = 0; q < m.size; q++) {
    const a = Math.abs(st[q]! - m.u[q]!), b = Math.abs(st[m.size + q]! - m.v[q]!), c = Math.abs(st[5 * m.size + q]! - qv[q]!);
    if (a > mU) mU = a; if (b > mV) { mV = b; wV = q; } if (c > mQ) { mQ = c; wQ = q; }
  }
  const detail = `max abs: u ${mU.toExponential(2)} v ${mV.toExponential(2)} (k ${Math.floor(wV / m.plane)}: gpu ${st[m.size + wV]!.toFixed(5)} cpu ${m.v[wV]!.toFixed(5)} before ${before[m.size + wV]!.toFixed(5)}) qv ${mQ.toExponential(2)} (k ${Math.floor(wQ / m.plane)}: gpu ${st[5 * m.size + wQ]!.toExponential(4)} cpu ${qv[wQ]!.toExponential(4)} before ${before[5 * m.size + wQ]!.toExponential(4)})`;
  let dTh = 0, dQv = 0;
  for (let q = 0; q < m.size; q++) { dTh = Math.max(dTh, Math.abs(st[3 * m.size + q]! - before[3 * m.size + q]!)); dQv = Math.max(dQv, before[5 * m.size + q]! - st[5 * m.size + q]!); }
  // (WGSL's sin and cos may err by 2^-11 absolute: the envelope, so each change, by about 5e-4 of itself)
  gcheck('interactions on the GPU: warm bubble (offset origin), drying across the periodic edge and a one-time rotating wind match the CPU changes (theta rel L2 < 1e-5, qv, u, v < 1e-4)',
    eTh < 1e-5 && eQv < 1e-4 && eU < 1e-4 && eV < 1e-4 && dTh > 1 && dQv > 1e-4, `theta ${eTh.toExponential(1)}, qv ${eQv.toExponential(1)}, u ${eU.toExponential(1)}, v ${eV.toExponential(1)}; changes ${dTh.toFixed(2)} K, ${(dQv * 1000).toFixed(2)} g/kg; ${detail}`);
  // undo: a copy on the GPU, restored exactly with the model time
  g.snapshot();
  const t0 = g.time, snap = await g.readState();
  g.step(3); g.edit({ kind: 'bubble', ...bub, warm: false, amp: -5, cap: -10 });
  const ok = g.restoreSnapshot(), back = await g.readState();
  let eBack = 0; for (let q = 0; q < back.length; q++) eBack = Math.max(eBack, Math.abs(back[q]! - snap[q]!));
  gcheck('interactions on the GPU: the copy of the state is restored exactly (and the time)', ok && eBack === 0 && g.time === t0, `max diff ${eBack}, time ${g.time} vs ${t0}`);
  g.destroy();
}
