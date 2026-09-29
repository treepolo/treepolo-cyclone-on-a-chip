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
  const rel = (f: number): number => {
    let e = 0, n = 0;
    for (let q = 0; q < nx * nx; q++) { e += (d.col[COL * q + f]! - cpu[COL * q + f]!) ** 2; n += cpu[COL * q + f]! ** 2; }
    return Math.sqrt(e / Math.max(n, 1e-30));
  };
  const basic = [C.wmax, C.wmin, C.cmax, C.pmax, C.ctopT].map(rel);
  gcheck('charts: GPU column extremes and cloud-top temperature match the CPU (rel L2 < 1e-4)', basic.every((x) => x < 1e-4), basic.map((x) => x.toExponential(1)).join(' '));
  const sat = [C.vis, C.pw, C.wvT].map(rel);
  gcheck('charts: GPU cloud albedo, precipitable water and water-vapour channel match the CPU (rel L2 < 1e-3)', sat.every((x) => x < 1e-3), sat.map((x) => x.toExponential(1)).join(' '));
  let dz2 = 0, ctzMis = 0;
  for (let q = 0; q < nx * nx; q++) {
    if (cpu[COL * q + C.dbz]! > 0) dz2 = Math.max(dz2, Math.abs(d.col[COL * q + C.dbz]! - cpu[COL * q + C.dbz]!));
    if (d.col[COL * q + C.ctopZ] !== cpu[COL * q + C.ctopZ]) ctzMis++;
  }
  gcheck('charts: column-max reflectivity within 0.05 dBZ, cloud-top height identical in 99 % of columns', dz2 < 0.05 && ctzMis <= 0.01 * nx * nx, `${dz2.toExponential(2)} dB, ${ctzMis} columns differ`);
  const uh = rel(C.uh), cape = rel(C.cape), cin = rel(C.cin);
  gcheck('charts: updraft helicity rel L2 < 1e-3, CAPE < 1e-2, CIN < 5e-2', uh < 1e-3 && cape < 1e-2 && cin < 5e-2, `${uh.toExponential(1)} ${cape.toExponential(1)} ${cin.toExponential(1)}`);
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
