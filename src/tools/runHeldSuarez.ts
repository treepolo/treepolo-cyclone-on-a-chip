// Held–Suarez climate run (Node). Usage:
//   node dist/tools/runHeldSuarez.js [preset=T42L20] [spinupDays=200] [avgDays=300] [outDir=results/<preset>]
// Writes climate.json, summary.txt and SVG sections of [u], [T], psi and eddy fluxes.

import { mkdirSync, writeFileSync, existsSync, readFileSync } from 'node:fs';
import { DAY, EARTH } from '../core/constants.js';
import { HS_PRESETS, createHeldSuarez } from '../model/presets.js';
import { ZonalMeanAccumulator, summarize, ZonalMeanClimate } from '../model/diagnostics.js';
import { sectionSvg } from './plot.js';

const presetName = process.argv[2] ?? 'T42L20';
const spinupDays = Number(process.argv[3] ?? 200);
const avgDays = Number(process.argv[4] ?? 300);
const outDir = process.argv[5] ?? `results/${presetName}`;
const cfg = HS_PRESETS[presetName];
if (!cfg) throw new Error(`unknown preset ${presetName}; options: ${Object.keys(HS_PRESETS).join(', ')}`);
mkdirSync(outDir, { recursive: true });

const model = createHeldSuarez(cfg);
const ckpt = `${outDir}/checkpoint.bin`;
if (existsSync(ckpt)) {
  const buf = readFileSync(ckpt);
  const header = new Float64Array(buf.buffer, buf.byteOffset, 2);
  model.importState({ time: header[0]!, steps: header[1]!, data: new Float64Array(buf.buffer.slice(buf.byteOffset + 16, buf.byteOffset + buf.byteLength)) });
  console.log(`resumed from checkpoint at day ${(model.time / DAY).toFixed(2)}`);
}
const saveCheckpoint = (): void => {
  const s = model.exportState();
  const out = new Float64Array(2 + s.data.length);
  out[0] = s.time; out[1] = s.steps; out.set(s.data, 2);
  writeFileSync(ckpt, Buffer.from(out.buffer));
};

const tr = model.tr;
const acc = new ZonalMeanAccumulator(tr.nlat, tr.nlon, model.K);
const stepsPerDay = Math.round(DAY / cfg.dt);
const sampleEvery = Math.max(1, Math.round(6 * 3600 / cfg.dt)); // 6-hourly samples
const totalDays = spinupDays + avgDays;
const ps0 = model.meanSurfacePressure();
const t0 = Date.now();
const log: string[] = [];
const say = (s: string): void => { console.log(s); log.push(s); };
say(`Held–Suarez ${presetName}: T${cfg.trunc} (${tr.nlon}x${tr.nlat}), L${cfg.levels}, dt=${cfg.dt}s, spin-up ${spinupDays} d, average ${avgDays} d`);

while (model.time < totalDays * DAY - 1) {
  model.step();
  const day = model.time / DAY;
  if (day > spinupDays && model.steps % sampleEvery === 0) acc.add(model.refreshGrid());
  if (model.steps % stepsPerDay === 0) {
    const d = Math.round(day);
    if (!model.isFinite()) { say(`NaN detected at day ${d}`); process.exit(1); }
    if (d % 10 === 0) {
      const g = model.refreshGrid();
      let umax = 0;
      for (let q = 0; q < g.u.length; q++) umax = Math.max(umax, Math.hypot(g.u[q]!, g.v[q]!));
      const sec = (Date.now() - t0) / 1000;
      say(`day ${d}: max|V|=${umax.toFixed(1)} m/s  mean ps drift=${((model.meanSurfacePressure() - ps0) / ps0).toExponential(2)}  wall=${sec.toFixed(0)}s`);
      saveCheckpoint();
    }
    if (d % 50 === 0 && acc.samples > 0) writeOutputs(acc.result(tr.lat, model.lev.sigma, model.lev.sigmaHalf, EARTH), false);
  }
}
writeOutputs(acc.result(tr.lat, model.lev.sigma, model.lev.sigmaHalf, EARTH), true);

function writeOutputs(c: ZonalMeanClimate, final: boolean): void {
  writeFileSync(`${outDir}/climate.json`, JSON.stringify(c));
  const s = summarize(c);
  const lines = [
    `Held–Suarez ${presetName}  samples=${c.samples}  (${final ? 'final' : 'partial'})`,
    `NH jet max: ${s.jetMaxNH.u.toFixed(1)} m/s at ${s.jetMaxNH.lat.toFixed(1)}°, sigma=${s.jetMaxNH.sigma.toFixed(3)}`,
    `SH jet max: ${s.jetMaxSH.u.toFixed(1)} m/s at ${s.jetMaxSH.lat.toFixed(1)}°, sigma=${s.jetMaxSH.sigma.toFixed(3)}`,
    `psi range: ${(s.minPsi / 1e9).toFixed(1)} .. ${(s.maxPsi / 1e9).toFixed(1)} x 1e9 kg/s`,
    `Overturning cells (north -> south, column max |psi|):`,
    ...s.cells.map((x) => `  ${x.fromLat.toFixed(1)}° .. ${x.toLat.toFixed(1)}°  peak ${(x.peak / 1e9).toFixed(1)}e9 kg/s at sigma ${x.sigma.toFixed(2)}`),
    `Surface zonal wind (lowest level):`,
    ...s.surfaceU.filter((_, j) => j % Math.max(1, Math.floor(c.nlat / 16)) === 0).map((x) => `  ${x.lat.toFixed(1)}°: ${x.u.toFixed(2)} m/s`),
    `Equator-pole surface-level dT: ${s.eqPoleDeltaT.toFixed(1)} K`,
  ];
  writeFileSync(`${outDir}/summary.txt`, lines.join('\n') + '\n\nRun log:\n' + log.join('\n') + '\n');
  if (final) console.log(lines.join('\n'));
  const K = c.K;
  writeFileSync(`${outDir}/u.svg`, sectionSvg({ title: `[u] zonal-mean zonal wind, ${presetName}`, lat: c.lat, sigma: c.sigma, values: c.u, units: 'm/s', diverging: true, contourStep: 5 }));
  writeFileSync(`${outDir}/T.svg`, sectionSvg({ title: `[T] zonal-mean temperature, ${presetName}`, lat: c.lat, sigma: c.sigma, values: c.T, units: 'K', diverging: false, contourStep: 10 }));
  writeFileSync(`${outDir}/psi.svg`, sectionSvg({ title: `ψ mean meridional mass streamfunction, ${presetName}`, lat: c.lat, sigma: c.sigmaHalf.slice(1, K), values: c.psi.slice(c.nlat, K * c.nlat).map((x) => x / 1e9), units: '1e9 kg/s', diverging: true, contourStep: 10 }));
  writeFileSync(`${outDir}/uv_eddy.svg`, sectionSvg({ title: `[u'v'] eddy momentum flux, ${presetName}`, lat: c.lat, sigma: c.sigma, values: c.uvEddy, units: 'm²/s²', diverging: true, contourStep: 10 }));
  writeFileSync(`${outDir}/vT_eddy.svg`, sectionSvg({ title: `[v'T'] eddy heat flux, ${presetName}`, lat: c.lat, sigma: c.sigma, values: c.vTEddy, units: 'K m/s', diverging: true, contourStep: 5 }));
  writeFileSync(`${outDir}/eke.svg`, sectionSvg({ title: `eddy kinetic energy, ${presetName}`, lat: c.lat, sigma: c.sigma, values: c.eke, units: 'm²/s²', diverging: false, contourStep: 50 }));
}
