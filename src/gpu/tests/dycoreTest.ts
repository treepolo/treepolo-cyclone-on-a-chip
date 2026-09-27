// GPU dynamical core vs CPU Float64 core (Held–Suarez, T21 L20).
import { createHeldSuarez, HS_PRESETS } from '../../model/presets.js';
import { GpuDycore } from '../dycoreGpu.js';
import { gcheck, getDevice } from './harness.js';

function relDiff(g: Float32Array, cpu: { re: Float64Array; im: Float64Array }, f: number, nspec: number): number {
  let e = 0, n = 0;
  for (let s = 0; s < nspec; s++) {
    e += (g[(f * nspec + s) * 2]! - cpu.re[s]!) ** 2 + (g[(f * nspec + s) * 2 + 1]! - cpu.im[s]!) ** 2;
    n += cpu.re[s]! ** 2 + cpu.im[s]! ** 2;
  }
  return Math.sqrt(e / Math.max(n, 1e-300));
}

export async function dycoreTests(): Promise<void> {
  const device = await getDevice();
  const cpu = createHeldSuarez(HS_PRESETS.T21L20!);
  // spin the CPU model up a little so the flow is non-trivial, then hand the state to the GPU
  for (let i = 0; i < 36 * 20; i++) cpu.step();
  const gpu = new GpuDycore(device, cpu, { heldSuarez: true });
  gpu.uploadFrom(cpu);
  const K = cpu.K, nspec = cpu.tr.nspec;
  const checkpoints = [1, 10, 100];
  let done = 0;
  for (const n of checkpoints) {
    const todo = n - done;
    for (let i = 0; i < todo; i++) cpu.step();
    gpu.step(todo);
    done = n;
    const g = await gpu.readCurrent();
    const kMid = Math.floor(K / 2), kLow = K - 1;
    const dv = relDiff(g, cpu.cur.vor[kMid]!, kMid, nspec);
    const dt = relDiff(g, cpu.cur.tmp[kLow]!, 2 * K + kLow, nspec);
    const dp = relDiff(g, cpu.cur.lnps, 3 * K, nspec);
    const tol = n <= 10 ? 1e-4 : 2e-3;
    gcheck(`HS T21 after ${n} steps: vorticity (mid level) GPU vs CPU rel L2 < ${tol}`, dv < tol, dv);
    gcheck(`HS T21 after ${n} steps: temperature (lowest level) GPU vs CPU rel L2 < ${tol}`, dt < tol, dt);
    gcheck(`HS T21 after ${n} steps: ln ps GPU vs CPU rel L2 < ${tol}`, dp < tol, dp);
  }
  // longer GPU-only integration: stays finite and bounded
  gpu.step(36 * 30);
  const g = await gpu.readCurrent();
  let finite = true, vmax = 0;
  for (let i = 0; i < g.length; i++) { if (!Number.isFinite(g[i]!)) { finite = false; break; } }
  for (let s = 0; s < nspec; s++) vmax = Math.max(vmax, Math.abs(g[(Math.floor(K / 2) * nspec + s) * 2]!));
  gcheck('HS T21 GPU: 30 further days finite', finite, vmax);
}
