// Moist gray-radiation aquaplanet climate run (Node). Usage:
//   node dist/tools/runAquaplanet.js [preset=AQUA_T42] [spinupDays=300] [avgDays=300] [outDir]
// Writes summary.txt, climate.json and SVG sections / zonal-mean profiles.

import { mkdirSync, writeFileSync, existsSync, readFileSync } from 'node:fs';
import { DAY, EARTH } from '../core/constants.js';
import { AQUA_PRESETS, createAquaplanet } from '../model/presets.js';
import { ZonalMeanAccumulator, summarize } from '../model/diagnostics.js';
import { sectionSvg, lineSvg } from './plot.js';

const presetName = process.argv[2] ?? 'AQUA_T42';
const spinupDays = Number(process.argv[3] ?? 300);
const avgDays = Number(process.argv[4] ?? 300);
const outDir = process.argv[5] ?? `results/${presetName}`;
const cfg = AQUA_PRESETS[presetName];
if (!cfg) throw new Error(`unknown preset ${presetName}`);
mkdirSync(outDir, { recursive: true });

const { model, physics } = createAquaplanet(cfg);
const tr = model.tr, ng = model.ng, K = model.K, nlat = tr.nlat, nlon = tr.nlon;
const ckpt = `${outDir}/checkpoint.bin`;
if (existsSync(ckpt)) {
  const buf = readFileSync(ckpt);
  const all = new Float64Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const nState = all.length - 2 - ng;
  model.importState({ time: all[0]!, steps: all[1]!, data: all.subarray(2, 2 + nState) });
  physics.f.sst.set(all.subarray(2 + nState));
  console.log(`resumed from checkpoint at day ${(model.time / DAY).toFixed(2)}`);
}
const saveCheckpoint = (): void => {
  const s = model.exportState();
  const out = new Float64Array(2 + s.data.length + ng);
  out[0] = s.time; out[1] = s.steps; out.set(s.data, 2); out.set(physics.f.sst, 2 + s.data.length);
  writeFileSync(ckpt, Buffer.from(out.buffer));
};

const acc = new ZonalMeanAccumulator(nlat, nlon, K);
const qZonal = new Float64Array(K * nlat), rhZonal = new Float64Array(K * nlat);
let qSamples = 0;
const stepsPerDay = Math.round(DAY / cfg.dt);
const sampleEvery = Math.max(1, Math.round(6 * 3600 / cfg.dt));
const totalDays = spinupDays + avgDays;
const log: string[] = [];
const say = (s: string): void => { console.log(s); log.push(s); };
say(`Gray aquaplanet ${presetName}: T${cfg.trunc} (${nlon}x${nlat}), L${K}, dt=${cfg.dt}s, spin-up ${spinupDays} d, average ${avgDays} d`);

const globalMean = (f: Float64Array): number => {
  let s = 0;
  for (let j = 0; j < nlat; j++) { let r = 0; for (let i = 0; i < nlon; i++) r += f[j * nlon + i]!; s += tr.weight[j]! * r / nlon; }
  return s / 2;
};
let W0 = model.totalWater(), budgetStartE = 0, budgetStartP = 0;
const cumE = new Float64Array(1), cumP = new Float64Array(1);
let avgStarted = model.time / DAY >= spinupDays;
if (avgStarted) physics.resetAccumulators();
const t0 = Date.now();
physics.resetAccumulators();
let lastE = 0, lastP = 0;

while (model.time < totalDays * DAY - 1) {
  model.step();
  const day = model.time / DAY;
  if (!avgStarted && day >= spinupDays) { avgStarted = true; physics.resetAccumulators(); acc.reset(); lastE = 0; lastP = 0; }
  if (avgStarted && model.steps % sampleEvery === 0) {
    const g = model.refreshGrid();
    acc.add(g);
    for (let k = 0; k < K; k++) for (let j = 0; j < nlat; j++) {
      let qs = 0, rs = 0;
      for (let i = 0; i < nlon; i++) {
        const idx = k * ng + j * nlon + i;
        const qv = model.q[idx]!;
        const p = g.ps[j * nlon + i]! * model.lev.sigma[k]!;
        qs += qv;
        rs += qv / qsatLocal(g.T[idx]!, p);
      }
      qZonal[k * nlat + j] = qZonal[k * nlat + j]! + qs / nlon;
      rhZonal[k * nlat + j] = rhZonal[k * nlat + j]! + rs / nlon;
    }
    qSamples++;
  }
  if (model.steps % stepsPerDay === 0) {
    const d = Math.round(day);
    if (!model.isFinite()) { say(`NaN detected at day ${d}`); process.exit(1); }
    const E = globalMean(physics.f.evap), P = globalMean(physics.f.precipConv) + globalMean(physics.f.precipLS);
    cumE[0] = cumE[0]! + (E - lastE); cumP[0] = cumP[0]! + (P - lastP);
    lastE = E; lastP = P;
    if (d % 10 === 0) {
      const g = model.refreshGrid();
      let umax = 0;
      for (let q = 0; q < g.u.length; q++) umax = Math.max(umax, Math.hypot(g.u[q]!, g.v[q]!));
      let smin = Infinity, smax = -Infinity;
      for (const x of physics.f.sst) { smin = Math.min(smin, x); smax = Math.max(smax, x); }
      const W = model.totalWater();
      const tacc = Math.max(physics.f.accTime, 1);
      say(`day ${d}: max|V|=${umax.toFixed(1)} m/s  SST ${smin.toFixed(1)}..${smax.toFixed(1)} K  W=${W.toFixed(2)} kg/m2  ` +
        `P=${(P / tacc * DAY).toFixed(2)} E=${(E / tacc * DAY).toFixed(2)} mm/day  OLR=${(globalMean(physics.f.olr) / tacc).toFixed(1)} W/m2  ` +
        `budget dW-(E-P)=${(W - W0 - (cumE[0]! - cumP[0]!)).toFixed(3)} fixer=${model.waterFixer.toFixed(3)}  wall=${((Date.now() - t0) / 1000).toFixed(0)}s`);
      saveCheckpoint();
    }
    if (avgStarted && d % 50 === 0) writeOutputs(false);
  }
}
writeOutputs(true);
void budgetStartE; void budgetStartP;

function qsatLocal(T: number, p: number): number {
  const es = 610.78 * Math.exp(-(2.5e6 / 461.5) * (1 / T - 1 / 273.16));
  const eps = 287.05 / 461.5;
  return eps * es / Math.max(p - (1 - eps) * es, 1e-3 * p);
}

function zonal(f: Float64Array, scale: number): number[] {
  const out: number[] = [];
  for (let j = 0; j < nlat; j++) { let s = 0; for (let i = 0; i < nlon; i++) s += f[j * nlon + i]!; out.push(s / nlon * scale); }
  return out;
}

function writeOutputs(final: boolean): void {
  const c = acc.result(tr.lat, model.lev.sigma, model.lev.sigmaHalf, EARTH);
  const s = summarize(c);
  const tacc = Math.max(physics.f.accTime, 1);
  const toMmDay = DAY / tacc;
  const precipC = zonal(physics.f.precipConv, toMmDay), precipL = zonal(physics.f.precipLS, toMmDay);
  const evap = zonal(physics.f.evap, toMmDay), sst = zonal(physics.f.sst, 1), olr = zonal(physics.f.olr, 1 / tacc);
  const precip = precipC.map((x, j) => x + precipL[j]!);
  const n = Math.max(1, qSamples);
  const q = Array.from(qZonal, (x) => x / n * 1000), rh = Array.from(rhZonal, (x) => x / n * 100);
  const gm = (arr: number[]): number => arr.reduce((a, x, j) => a + tr.weight[j]! * x, 0) / 2;
  writeFileSync(`${outDir}/climate.json`, JSON.stringify({ ...c, precip, precipConv: precipC, precipLS: precipL, evap, sst, olr, q, rh }));
  let pmax = 0, pmaxLat = 0;
  precip.forEach((x, j) => { if (x > pmax) { pmax = x; pmaxLat = c.lat[j]!; } });
  const lines = [
    `Gray aquaplanet ${presetName}  samples=${c.samples}  averaging ${(tacc / DAY).toFixed(0)} days  (${final ? 'final' : 'partial'})`,
    `Global mean: P=${gm(precip).toFixed(2)} mm/day (conv ${gm(precipC).toFixed(2)}, large-scale ${gm(precipL).toFixed(2)}), E=${gm(evap).toFixed(2)} mm/day, OLR=${gm(olr).toFixed(1)} W/m2, SST=${gm(sst).toFixed(1)} K`,
    `Precipitation max: ${pmax.toFixed(1)} mm/day at ${pmaxLat.toFixed(1)}°`,
    `NH jet max: ${s.jetMaxNH.u.toFixed(1)} m/s at ${s.jetMaxNH.lat.toFixed(1)}°, sigma=${s.jetMaxNH.sigma.toFixed(3)}`,
    `SH jet max: ${s.jetMaxSH.u.toFixed(1)} m/s at ${s.jetMaxSH.lat.toFixed(1)}°, sigma=${s.jetMaxSH.sigma.toFixed(3)}`,
    `psi range: ${(s.minPsi / 1e9).toFixed(1)} .. ${(s.maxPsi / 1e9).toFixed(1)} x 1e9 kg/s`,
    `Overturning cells (north -> south, column max |psi|):`,
    ...s.cells.map((x) => `  ${x.fromLat.toFixed(1)}° .. ${x.toLat.toFixed(1)}°  peak ${(x.peak / 1e9).toFixed(1)}e9 kg/s at sigma ${x.sigma.toFixed(2)}`),
    `Zonal means (lat: P, E, SST, surface u):`,
    ...c.lat.map((lat, j) => ({ lat, j })).filter((x) => x.j % Math.max(1, Math.floor(nlat / 16)) === 0)
      .map(({ lat, j }) => `  ${lat.toFixed(1)}°: P=${precip[j]!.toFixed(2)} E=${evap[j]!.toFixed(2)} mm/day  SST=${sst[j]!.toFixed(1)} K  u_sfc=${c.u[(K - 1) * nlat + j]!.toFixed(2)} m/s`),
  ];
  writeFileSync(`${outDir}/summary.txt`, lines.join('\n') + '\n\nRun log:\n' + log.join('\n') + '\n');
  if (final) console.log(lines.join('\n'));
  writeFileSync(`${outDir}/u.svg`, sectionSvg({ title: `[u] zonal-mean zonal wind, ${presetName}`, lat: c.lat, sigma: c.sigma, values: c.u, units: 'm/s', diverging: true, contourStep: 5 }));
  writeFileSync(`${outDir}/T.svg`, sectionSvg({ title: `[T] zonal-mean temperature, ${presetName}`, lat: c.lat, sigma: c.sigma, values: c.T, units: 'K', diverging: false, contourStep: 10 }));
  writeFileSync(`${outDir}/psi.svg`, sectionSvg({ title: `ψ mean meridional mass streamfunction, ${presetName}`, lat: c.lat, sigma: c.sigmaHalf.slice(1, K), values: c.psi.slice(nlat, K * nlat).map((x) => x / 1e9), units: '1e9 kg/s', diverging: true, contourStep: 20 }));
  writeFileSync(`${outDir}/q.svg`, sectionSvg({ title: `[q] specific humidity, ${presetName}`, lat: c.lat, sigma: c.sigma, values: q, units: 'g/kg', diverging: false, contourStep: 2 }));
  writeFileSync(`${outDir}/rh.svg`, sectionSvg({ title: `relative humidity, ${presetName}`, lat: c.lat, sigma: c.sigma, values: rh, units: '%', diverging: false, contourStep: 10 }));
  writeFileSync(`${outDir}/uv_eddy.svg`, sectionSvg({ title: `[u'v'] eddy momentum flux, ${presetName}`, lat: c.lat, sigma: c.sigma, values: c.uvEddy, units: 'm²/s²', diverging: true, contourStep: 10 }));
  writeFileSync(`${outDir}/precip.svg`, lineSvg({ title: `Zonal-mean precipitation and evaporation, ${presetName}`, x: c.lat, series: [
    { name: 'P total', values: precip, color: '#1f5fbf' }, { name: 'P convective', values: precipC, color: '#6aa0e8', dashed: true },
    { name: 'P large-scale', values: precipL, color: '#9b59b6', dashed: true }, { name: 'E', values: evap, color: '#d9822b' },
  ], units: 'mm/day' }));
  writeFileSync(`${outDir}/sst.svg`, lineSvg({ title: `Zonal-mean SST, ${presetName}`, x: c.lat, series: [{ name: 'SST', values: sst, color: '#c0392b' }], units: 'K' }));
}
