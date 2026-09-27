import { RegionalModel } from '../../regional/core.js';
import { weismanKlemp } from '../../regional/kessler.js';
import { GpuRegional } from '../regionalGpu.js';
import { getDevice } from './harness.js';
export async function regionalDebug(): Promise<void> {
  const device = await getDevice();
  const nx = 12, nz = 16, dx = 3000, dz = 900;
  const m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt: 6, nsound: 6, f: 0, beta: 0.2, divDamp: 0.1, dampDepth: 0, dampRate: 0, kdiff2: 0 }, weismanKlemp, 0);
  m.setBaseWind((z) => ({ u: 10 * Math.tanh(z / 3000) - 5, v: 2 }));
  const c = nx * dx / 2;
  for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
    const r = Math.sqrt((((i + 0.5) * dx - c) / 9000) ** 2 + (((j + 0.5) * dx - c) / 9000) ** 2 + ((m.zc[k]! - 1400) / 1400) ** 2);
    if (r < 1) m.th[m.idx(i, j, k)] = m.th[m.idx(i, j, k)]! + 2 * Math.cos(0.5 * Math.PI * r) ** 2;
  }
  for (let s = 0; s < 20; s++) m.step();
  const g = new GpuRegional(device, m, { moist: false, physics: null });
  g.uploadFrom(m);
  // slow tendencies: passes = [save, haloAll, mom, sca, ...]
  g.debugSlow(4);
  const F = await g.readBuffer(g.F, 8 * m.size * 4);
  const mm = m as unknown as { slowTendencies(): void; fu: Float64Array; fv: Float64Array; fw: Float64Array; fth: Float64Array; fpp: Float64Array };
  mm.slowTendencies();
  const rel = (f: number, a: Float64Array, nk: number): string => {
    let e = 0, s = 0, worst = 0, at = '';
    for (let k = 0; k < nk; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) { const q = m.idx(i, j, k); const d = Math.abs(F[f * m.size + q]! - a[q]!); e += d * d; s += a[q]! ** 2; if (d > worst) { worst = d; at = `${i},${j},${k} gpu ${F[f * m.size + q]!.toExponential(3)} cpu ${a[q]!.toExponential(3)}`; } }
    return `${Math.sqrt(e / s).toExponential(2)} worst ${at}`;
  };
  console.log('GPUTEST INFO fu ' + rel(0, mm.fu, nz));
  console.log('GPUTEST INFO fv ' + rel(1, mm.fv, nz));
  console.log('GPUTEST INFO fw ' + rel(2, mm.fw, nz + 1));
  console.log('GPUTEST INFO fth ' + rel(3, mm.fth, nz));
  console.log('GPUTEST INFO fpp ' + rel(4, mm.fpp, nz));
}
