// GPU spectral transform vs CPU Float64 transform.
import { SpectralTransform } from '../../spectral/transform.js';
import { GpuTransform } from '../transformGpu.js';
import { gcheck, getDevice } from './harness.js';
import { rng } from '../../core/random.js';

export async function transformTests(): Promise<void> {
  const device = await getDevice();
  for (const T of [21, 42]) {
    const cpu = new SpectralTransform(T);
    const g = new GpuTransform(device, cpu);
    const r = rng(T);
    const a = 6.371e6;
    const vor = cpu.newSpec(), div = cpu.newSpec(), sc = cpu.newSpec();
    for (let s = 0; s < cpu.nspec; s++) {
      const m0 = cpu.mOf[s] === 0;
      vor.re[s] = (r() - 0.5) * 1e-5; vor.im[s] = m0 ? 0 : (r() - 0.5) * 1e-5;
      div.re[s] = (r() - 0.5) * 1e-6; div.im[s] = m0 ? 0 : (r() - 0.5) * 1e-6;
      sc.re[s] = r() - 0.5; sc.im[s] = m0 ? 0 : r() - 0.5;
    }
    vor.re[0] = 0; div.re[0] = 0;
    // spectral buffer fields: 0 scalar, 1 vor, 2 div, 3 psi, 4 chi
    const specData = new Float32Array(5 * cpu.nspec * 2);
    const put = (f: number, x: { re: Float64Array; im: Float64Array }): void => { for (let s = 0; s < cpu.nspec; s++) { specData[(f * cpu.nspec + s) * 2] = x.re[s]!; specData[(f * cpu.nspec + s) * 2 + 1] = x.im[s]!; } };
    put(0, sc); put(1, vor); put(2, div);
    const psi = cpu.newSpec(), chi = cpu.newSpec();
    for (let s = 0; s < cpu.nspec; s++) { const f = cpu.nn1[s]! > 0 ? -a * a / cpu.nn1[s]! : 0; psi.re[s] = vor.re[s]! * f; psi.im[s] = vor.im[s]! * f; chi.re[s] = div.re[s]! * f; chi.im[s] = div.im[s]! * f; }
    put(3, psi); put(4, chi);
    const spec = g.specBuffer(5), four = g.fourierBuffer(5), grid = g.gridBuffer(5), spec2 = g.specBuffer(3);
    g.write(spec, specData);
    const ia = 1 / a;
    // outputs: 0 scalar, 1 dX/dlambda, 2 (1-mu^2) dX/dmu, 3 U, 4 V
    const synth = g.prepareSynth(spec, four, [
      { output: 0, terms: [{ input: 0, kind: 'P', im: false, scale: 1 }] },
      { output: 1, terms: [{ input: 0, kind: 'P', im: true, scale: 1 }] },
      { output: 2, terms: [{ input: 0, kind: 'H', im: false, scale: 1 }] },
      { output: 3, terms: [{ input: 4, kind: 'P', im: true, scale: ia }, { input: 3, kind: 'H', im: false, scale: -ia }] },
      { output: 4, terms: [{ input: 3, kind: 'P', im: true, scale: ia }, { input: 4, kind: 'H', im: false, scale: ia }] },
    ]);
    const inv = g.prepareFftInverse(four, grid, 5);
    g.run(synth, inv);
    const gg = await g.read(grid, 5 * cpu.gridSize * 4);
    const ref = [cpu.newGrid(), cpu.newGrid(), cpu.newGrid(), cpu.newGrid(), cpu.newGrid()];
    cpu.synth(sc, ref[0]!); cpu.synthGrad(sc, ref[1]!, ref[2]!); cpu.synthUV(vor, div, a, ref[3]!, ref[4]!);
    const names = ['scalar', 'd/dlambda', '(1-mu^2) d/dmu', 'U', 'V'];
    for (let f = 0; f < 5; f++) {
      let e = 0, mx = 0;
      for (let i = 0; i < cpu.gridSize; i++) { e = Math.max(e, Math.abs(gg[f * cpu.gridSize + i]! - ref[f]![i]!)); mx = Math.max(mx, Math.abs(ref[f]![i]!)); }
      gcheck(`T${T} synthesis ${names[f]} matches CPU (rel)`, e / mx < 2e-5, e / mx);
    }
    // analysis: scalar round trip, and div/curl of (U,V) back to div/vor
    const fwd0 = g.prepareFftForward(grid, four, 1, 0, 0, false);
    const fwdUV = g.prepareFftForward(grid, four, 2, 3, 1, true);
    const anal = g.prepareAnal(four, spec2, [
      { output: 0, terms: [{ input: 0, kind: 'P', im: false, scale: 1 }] },
      { output: 1, terms: [{ input: 2, kind: 'P', im: true, scale: ia }, { input: 1, kind: 'H', im: false, scale: ia }] },   // curl -> vor
      { output: 2, terms: [{ input: 1, kind: 'P', im: true, scale: ia }, { input: 2, kind: 'H', im: false, scale: -ia }] },  // div
    ]);
    g.run(fwd0, fwdUV, anal);
    const s2 = await g.read(spec2, 3 * cpu.nspec * 8);
    const cmp = (f: number, x: { re: Float64Array; im: Float64Array }): number => {
      let e = 0, mx = 0;
      for (let s = 0; s < cpu.nspec; s++) { e = Math.max(e, Math.abs(s2[(f * cpu.nspec + s) * 2]! - x.re[s]!), Math.abs(s2[(f * cpu.nspec + s) * 2 + 1]! - x.im[s]!)); mx = Math.max(mx, Math.abs(x.re[s]!), Math.abs(x.im[s]!)); }
      return e / mx;
    };
    gcheck(`T${T} scalar grid->spectral round trip (rel)`, cmp(0, sc) < 2e-5, cmp(0, sc));
    gcheck(`T${T} curl(U,V) recovers vorticity (rel)`, cmp(1, vor) < 2e-4, cmp(1, vor));
    gcheck(`T${T} div(U,V) recovers divergence (rel)`, cmp(2, div) < 2e-4, cmp(2, div));
  }
}
