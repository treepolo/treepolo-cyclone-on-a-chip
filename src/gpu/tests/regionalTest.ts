// GPU regional model vs CPU Float64 regional model.
import { RegionalModel } from '../../regional/core.js';
import { KesslerMicrophysics, weismanKlemp, QV, QC, QR } from '../../regional/kessler.js';
import { RegionalPhysics } from '../../regional/physics.js';
import { tropicalSounding, insertVortex } from '../../regional/tropical.js';
import { GpuRegional } from '../regionalGpu.js';
import { gcheck, getDevice } from './harness.js';
import { nestFromGlobal, GlobalSnapshot } from '../../regional/nest.js';
import { IceMicrophysics, QI, QS, QG } from '../../regional/ice.js';

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
    for (let c = 0; c < nx * nx; c++) { wmax = Math.max(wmax, d.col[4 * c]!); cmax = Math.max(cmax, d.col[4 * c + 2]!); }
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
