// GPU moist aquaplanet vs CPU Float64 (T21 L25).
import { createAquaplanet, AQUA_PRESETS, createEarth, EARTH_PRESETS, EarthData } from '../../model/presets.js';
import { GpuDycore } from '../dycoreGpu.js';
import { GpuMoist, SFC } from '../moistGpu.js';
import { gcheck, getDevice } from './harness.js';

function rel(a: ArrayLike<number>, b: ArrayLike<number>, off = 0, n = b.length): number {
  let e = 0, s = 0;
  for (let i = 0; i < n; i++) { e += (a[off + i]! - b[i]!) ** 2; s += b[i]! ** 2; }
  return Math.sqrt(e / Math.max(s, 1e-300));
}

export async function moistTests(): Promise<void> {
  const device = await getDevice();
  const { model: cpu, physics } = createAquaplanet(AQUA_PRESETS.AQUA_T21!);
  for (let i = 0; i < 72 * 15; i++) cpu.step();      // 15 days: convection and storms active
  const gd = new GpuDycore(device, cpu, { heldSuarez: false });
  const gm = new GpuMoist(device, gd, cpu, physics);
  gd.uploadFrom(cpu);
  gm.uploadFrom(cpu);
  const K = cpu.K, ng = cpu.ng, nspec = cpu.tr.nspec;
  let done = 0;
  for (const n of [1, 10, 72]) {
    for (let i = done; i < n; i++) cpu.step();
    gd.step(n - done);
    done = n;
    const g = await gd.readCurrent();
    const q = await gm.readQ();
    const s = await gm.readSurface();
    const k = K - 3;
    const tv = rel(g, Float32Array.from(cpu.cur.tmp[k]!.re), (2 * K + k) * nspec * 2, 0) ;
    void tv;
    // spectral temperature (real parts only interleaved) — compare via reconstruction on grid would be costlier; use re/im
    let e = 0, sN = 0;
    for (let si = 0; si < nspec; si++) {
      e += (g[((2 * K + k) * nspec + si) * 2]! - cpu.cur.tmp[k]!.re[si]!) ** 2 + (g[((2 * K + k) * nspec + si) * 2 + 1]! - cpu.cur.tmp[k]!.im[si]!) ** 2;
      sN += cpu.cur.tmp[k]!.re[si]! ** 2 + cpu.cur.tmp[k]!.im[si]! ** 2;
    }
    const dT = Math.sqrt(e / sN);
    const dq = rel(q, cpu.q.subarray(k * ng, (k + 1) * ng), k * ng, ng);
    const dsst = rel(s, physics.f.sst, SFC.ts * ng, ng);
    const tol = n <= 10 ? 1e-3 : 5e-2;
    gcheck(`aquaplanet T21 after ${n} steps: T (sigma ${cpu.lev.sigma[k]!.toFixed(2)}) GPU vs CPU rel L2 < ${tol}`, dT < tol, dT);
    gcheck(`aquaplanet T21 after ${n} steps: q GPU vs CPU rel L2 < ${tol * 20}`, dq < tol * 20, dq);
    gcheck(`aquaplanet T21 after ${n} steps: SST GPU vs CPU rel L2 < ${tol}`, dsst < tol, dsst);
  }
  // GPU-only: 20 more days, water budget and finiteness
  const q0 = await gm.readQ();
  const s0 = await gm.readSurface();
  gd.step(72 * 20);
  const q1 = await gm.readQ();
  const s1 = await gm.readSurface();
  let finite = true, qmin = Infinity;
  for (const x of q1) { if (!Number.isFinite(x)) finite = false; qmin = Math.min(qmin, x); }
  const gmean = (arr: Float32Array, off: number): number => {
    let t = 0;
    for (let j = 0; j < cpu.tr.nlat; j++) { let r = 0; for (let i = 0; i < cpu.tr.nlon; i++) r += arr[off + j * cpu.tr.nlon + i]!; t += cpu.tr.weight[j]! * r / cpu.tr.nlon; }
    return t / 2;
  };
  const P = gmean(s1, SFC.precipConv * ng) + gmean(s1, SFC.precipLS * ng) - gmean(s0, SFC.precipConv * ng) - gmean(s0, SFC.precipLS * ng);
  const E = gmean(s1, SFC.evap * ng) - gmean(s0, SFC.evap * ng);
  gcheck('aquaplanet T21 GPU: 20 days finite, q >= 0', finite && qmin >= 0, qmin);
  gcheck('aquaplanet T21 GPU: precipitation 2–7 mm/day, E ≈ P within 15%', P / 20 > 2 && P / 20 < 7 && Math.abs(E - P) / P < 0.15, `P ${(P / 20).toFixed(2)} E ${(E / 20).toFixed(2)} mm/day`);
  void q0; void K;
}

/** Earth configuration (land, orography, seasons, sea ice): GPU vs CPU after a CPU spin-up. */
export async function earthTests(): Promise<void> {
  const device = await getDevice();
  const data = await (await fetch('data/earth_t42.json')).json() as EarthData;
  const { model: cpu, physics } = createEarth(EARTH_PRESETS.EARTH_T21!, data);
  for (let i = 0; i < 72 * 10; i++) cpu.step();      // 10 days: sea ice forming at high latitudes
  const ng = cpu.ng;
  let iceN = 0; for (let p = 0; p < ng; p++) if (physics.f.ice[p]! > 0) iceN++;
  gcheck('Earth T21 CPU: sea ice present after 10 days', iceN > 0, `${iceN} points`);
  const gd = new GpuDycore(device, cpu, { heldSuarez: false });
  const gm = new GpuMoist(device, gd, cpu, physics);
  gd.uploadFrom(cpu);
  gm.uploadFrom(cpu);
  let done = 0;
  for (const n of [1, 10]) {
    for (let i = done; i < n; i++) cpu.step();
    gd.step(n - done);
    done = n;
    const s = await gm.readSurface();
    const dts = rel(s, physics.f.sst, SFC.ts * ng, ng), dice = rel(s, physics.f.ice, SFC.ice * ng, ng), db = rel(s, physics.f.bucket, SFC.bucket * ng, ng);
    gcheck(`Earth T21 after ${n} steps: surface temperature GPU vs CPU rel L2 < 1e-4`, dts < 1e-4, dts);
    gcheck(`Earth T21 after ${n} steps: sea-ice thickness rel L2 < 1e-2`, dice < 1e-2, dice);
    gcheck(`Earth T21 after ${n} steps: soil water rel L2 < 1e-3`, db < 1e-3, db);
  }
}
