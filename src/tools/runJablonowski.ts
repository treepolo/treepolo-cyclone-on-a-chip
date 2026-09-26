// Jablonowski–Williamson baroclinic wave. Usage:
//   node dist/tools/runJablonowski.js [trunc=42] [levels=26] [dt=900] [days=10] [outDir]
// Reports minimum surface pressure per day and writes ps / T850-level maps at days 7, 8, 9, 10.

import { mkdirSync, writeFileSync } from 'node:fs';
import { DAY } from '../core/constants.js';
import { createJablonowski } from '../model/jablonowski.js';
import { mapSvg } from './plot.js';

const trunc = Number(process.argv[2] ?? 42), levels = Number(process.argv[3] ?? 26), dt = Number(process.argv[4] ?? 900);
const days = Number(process.argv[5] ?? 10), outDir = process.argv[6] ?? `results/JW_T${trunc}L${levels}`;
mkdirSync(outDir, { recursive: true });

const run = (perturb: boolean): string[] => {
  const m = createJablonowski({ trunc, levels, dt, perturb });
  const tr = m.tr, lines: string[] = [];
  const latDeg = Array.from(tr.lat, (x) => x * 180 / Math.PI), lonDeg = Array.from(tr.lon, (x) => x * 180 / Math.PI);
  let k850 = 0;
  for (let k = 0; k < m.K; k++) if (Math.abs(m.lev.sigma[k]! - 0.85) < Math.abs(m.lev.sigma[k850]! - 0.85)) k850 = k;
  const stepsPerDay = Math.round(DAY / dt);
  for (let d = 1; d <= days; d++) {
    for (let s = 0; s < stepsPerDay; s++) m.step();
    const g = m.refreshGrid();
    let pmin = Infinity, pmax = -Infinity, at = [0, 0];
    for (let j = 0; j < tr.nlat; j++) for (let i = 0; i < tr.nlon; i++) {
      const p = g.ps[j * tr.nlon + i]!;
      if (p < pmin) { pmin = p; at = [latDeg[j]!, lonDeg[i]!]; }
      pmax = Math.max(pmax, p);
    }
    // symmetry diagnostic for the unperturbed case: rms of zonal deviations of ps
    let dev = 0;
    for (let j = 0; j < tr.nlat; j++) {
      let mean = 0;
      for (let i = 0; i < tr.nlon; i++) mean += g.ps[j * tr.nlon + i]!;
      mean /= tr.nlon;
      for (let i = 0; i < tr.nlon; i++) dev += (g.ps[j * tr.nlon + i]! - mean) ** 2;
    }
    dev = Math.sqrt(dev / (tr.nlat * tr.nlon));
    lines.push(`day ${d}: min ps ${(pmin / 100).toFixed(2)} hPa at ${at[0]!.toFixed(1)}N ${at[1]!.toFixed(1)}E, max ps ${(pmax / 100).toFixed(2)} hPa, rms zonal ps deviation ${(dev / 100).toFixed(3)} hPa`);
    console.log((perturb ? 'perturbed ' : 'steady    ') + lines[lines.length - 1]);
    if (perturb && d >= 7) {
      const ng = m.ng;
      writeFileSync(`${outDir}/ps_day${d}.svg`, mapSvg({ title: `Surface pressure day ${d}, JW baroclinic wave T${trunc}L${levels}`, lat: latDeg, lon: lonDeg, values: Array.from(g.ps, (x) => x / 100), units: 'hPa', contourStep: 10, latRange: [0, 90] }));
      writeFileSync(`${outDir}/T850_day${d}.svg`, mapSvg({ title: `Temperature at sigma=${m.lev.sigma[k850]!.toFixed(3)} day ${d}, T${trunc}L${levels}`, lat: latDeg, lon: lonDeg, values: Array.from(g.T.subarray(k850 * ng, (k850 + 1) * ng)), units: 'K', contourStep: 5, latRange: [0, 90] }));
      const vor = new Float64Array(ng);
      m.vorticityGrid(k850, vor);
      writeFileSync(`${outDir}/vor850_day${d}.svg`, mapSvg({ title: `Relative vorticity at sigma=${m.lev.sigma[k850]!.toFixed(3)} day ${d}, T${trunc}L${levels}`, lat: latDeg, lon: lonDeg, values: Array.from(vor, (x) => x * 1e5), units: '1e-5 s^-1', contourStep: 5, latRange: [0, 90], diverging: true }));
    }
  }
  return lines;
};

const steady = run(false);
const wave = run(true);
writeFileSync(`${outDir}/summary.txt`, `Jablonowski–Williamson T${trunc}L${levels} dt=${dt}s\n\nSteady state (no perturbation):\n${steady.join('\n')}\n\nBaroclinic wave (1 m/s perturbation):\n${wave.join('\n')}\n`);
