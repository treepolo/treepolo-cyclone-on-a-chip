// GPU regional model vs CPU Float64 regional model.
import { RegionalModel } from '../../regional/core.js';
import { KesslerMicrophysics, weismanKlemp, QV, QC, QR } from '../../regional/kessler.js';
import { RegionalPhysics } from '../../regional/physics.js';
import { tropicalSounding, insertVortex } from '../../regional/tropical.js';
import { GpuRegional } from '../regionalGpu.js';
import { gcheck, getDevice } from './harness.js';

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
