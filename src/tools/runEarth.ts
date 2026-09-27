// Earth-like climate run with real land, orography and seasons (Node). Usage:
//   node dist/tools/runEarth.js [preset=EARTH_T21] [spinupYears=2] [avgYears=1] [outDir]
// Model time 0 is the northern spring equinox (~20 March). Writes seasonal (JJA / DJF) maps of
// precipitation, surface temperature and low-level wind, a zonal-mean precipitation Hovmoeller
// diagram, a monsoon index and zonal-mean circulation sections.

import { mkdirSync, writeFileSync, existsSync, readFileSync } from 'node:fs';
import { DAY, EARTH } from '../core/constants.js';
import { EARTH_PRESETS, EarthData, createEarth } from '../model/presets.js';
import { ZonalMeanAccumulator, summarize } from '../model/diagnostics.js';
import { mapSvg, sectionSvg } from './plot.js';

const presetName = process.argv[2] ?? 'EARTH_T21';
const spinupYears = Number(process.argv[3] ?? 2);
const avgYears = Number(process.argv[4] ?? 1);
const outDir = process.argv[5] ?? `results/${presetName}`;
const cfgMaybe = EARTH_PRESETS[presetName];
if (!cfgMaybe) throw new Error(`unknown preset ${presetName}`);
const cfg = cfgMaybe;
mkdirSync(outDir, { recursive: true });
const data = JSON.parse(readFileSync('data/earth_t42.json', 'utf8')) as EarthData;

const { model, physics } = createEarth(cfg, data);
const tr = model.tr, ng = model.ng, K = model.K, nlat = tr.nlat, nlon = tr.nlon;
const year = physics.cfg.yearLength;
const ckpt = `${outDir}/checkpoint.bin`;
if (existsSync(ckpt)) {
  const buf = readFileSync(ckpt);
  const all = new Float64Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const nState = all.length - 2 - 2 * ng;
  model.importState({ time: all[0]!, steps: all[1]!, data: all.subarray(2, 2 + nState) });
  physics.f.sst.set(all.subarray(2 + nState, 2 + nState + ng));
  physics.f.bucket.set(all.subarray(2 + nState + ng));
  console.log(`resumed from checkpoint at day ${(model.time / DAY).toFixed(2)}`);
}
const saveCheckpoint = (): void => {
  const s = model.exportState();
  const out = new Float64Array(2 + s.data.length + 2 * ng);
  out[0] = s.time; out[1] = s.steps; out.set(s.data, 2); out.set(physics.f.sst, 2 + s.data.length); out.set(physics.f.bucket, 2 + s.data.length + ng);
  writeFileSync(ckpt, Buffer.from(out.buffer));
};

let k850 = 0;
for (let k = 0; k < K; k++) if (Math.abs(model.lev.sigma[k]! - 0.85) < Math.abs(model.lev.sigma[k850]! - 0.85)) k850 = k;
// season windows in days after the spring equinox (~20 March)
const SEASONS = { JJA: [72, 164], DJF: [256, 346] } as const;
type Season = keyof typeof SEASONS;
const comp: Record<Season, { P: Float64Array; Ts: Float64Array; u: Float64Array; v: Float64Array; n: number; t: number }> = {
  JJA: { P: new Float64Array(ng), Ts: new Float64Array(ng), u: new Float64Array(ng), v: new Float64Array(ng), n: 0, t: 0 },
  DJF: { P: new Float64Array(ng), Ts: new Float64Array(ng), u: new Float64Array(ng), v: new Float64Array(ng), n: 0, t: 0 },
};
const NB = 12, hov = new Float64Array(NB * nlat), hovN = new Float64Array(NB);
const acc = new ZonalMeanAccumulator(nlat, nlon, K);
const log: string[] = [];
const say = (s: string): void => { console.log(s); log.push(s); };
const t0 = Date.now();
const startAvg = spinupYears * year, endT = (spinupYears + avgYears) * year;
const stepsPerDay = Math.round(DAY / cfg.dt), sampleEvery = Math.max(1, Math.round(6 * 3600 / cfg.dt));
say(`Earth ${presetName}: T${cfg.trunc} (${nlon}x${nlat}), L${K}, dt=${cfg.dt}s, land fraction ${(physics.surface.land.reduce((a, b) => a + b, 0) / ng).toFixed(3)}, spin-up ${spinupYears} y, average ${avgYears} y`);
const gmean = (f: Float64Array): number => { let s = 0; for (let j = 0; j < nlat; j++) { let r = 0; for (let i = 0; i < nlon; i++) r += f[j * nlon + i]!; s += tr.weight[j]! * r / nlon; } return s / 2; };

while (model.time < endT - 1) {
  model.step();
  const t = model.time;
  if (t > startAvg) {
    const doy = (t % year) / DAY;
    const pr = physics.f.precipRate;
    for (const s of Object.keys(SEASONS) as Season[]) {
      const [a, b] = SEASONS[s];
      if (doy >= a && doy < b) {
        const c = comp[s];
        for (let p = 0; p < ng; p++) { c.P[p] = c.P[p]! + pr[p]! * cfg.dt; c.Ts[p] = c.Ts[p]! + physics.f.sst[p]! * cfg.dt; }
        c.t += cfg.dt;
        if (model.steps % sampleEvery === 0) {
          const g = model.refreshGrid();
          for (let p = 0; p < ng; p++) { c.u[p] = c.u[p]! + g.u[k850 * ng + p]!; c.v[p] = c.v[p]! + g.v[k850 * ng + p]!; }
          c.n++;
        }
      }
    }
    const bin = Math.min(NB - 1, Math.floor(doy / (year / DAY / NB)));
    for (let j = 0; j < nlat; j++) { let r = 0; for (let i = 0; i < nlon; i++) r += pr[j * nlon + i]!; hov[bin * nlat + j] = hov[bin * nlat + j]! + r / nlon * cfg.dt; }
    hovN[bin] = hovN[bin]! + cfg.dt;
    if (model.steps % sampleEvery === 0) acc.add(model.refreshGrid());
  }
  if (model.steps % stepsPerDay === 0) {
    const d = Math.round(t / DAY);
    if (!model.isFinite()) { say(`NaN detected at day ${d}`); process.exit(1); }
    if (d % 15 === 0) {
      let tmin = Infinity, tmax = -Infinity;
      for (const x of physics.f.sst) { tmin = Math.min(tmin, x); tmax = Math.max(tmax, x); }
      const g = model.refreshGrid();
      let umax = 0;
      for (let q = 0; q < g.u.length; q++) umax = Math.max(umax, Math.hypot(g.u[q]!, g.v[q]!));
      say(`day ${d} (doy ${((t % year) / DAY).toFixed(0)}): Ts ${tmin.toFixed(1)}..${tmax.toFixed(1)} K (mean ${gmean(physics.f.sst).toFixed(1)})  max|V| ${umax.toFixed(1)} m/s  W=${model.totalWater().toFixed(2)} kg/m2  decl ${(physics.declination * 180 / Math.PI).toFixed(1)}°  wall=${((Date.now() - t0) / 1000).toFixed(0)}s`);
      saveCheckpoint();
    }
  }
}
writeOutputs();

function writeOutputs(): void {
  const latDeg = Array.from(tr.lat, (x) => x * 180 / Math.PI), lonDeg = Array.from(tr.lon, (x) => x * 180 / Math.PI);
  const land = Array.from(physics.surface.land);
  const lines: string[] = [`Earth ${presetName} seasonal climate (average of ${avgYears} year(s) after ${spinupYears} year(s) spin-up)`];
  const boxMean = (f: Float64Array, la0: number, la1: number, lo0: number, lo1: number): number => {
    let s = 0, w = 0;
    for (let j = 0; j < nlat; j++) { const la = latDeg[j]!; if (la < la0 || la > la1) continue; for (let i = 0; i < nlon; i++) { const lo = lonDeg[i]!; if (lo < lo0 || lo > lo1) continue; const c = Math.cos(tr.lat[j]!); s += f[j * nlon + i]! * c; w += c; } }
    return w ? s / w : NaN;
  };
  const P: Record<string, Float64Array> = {}, Ts: Record<string, Float64Array> = {};
  for (const s of Object.keys(SEASONS) as Season[]) {
    const c = comp[s], tt = Math.max(c.t, 1), n = Math.max(c.n, 1);
    P[s] = Float64Array.from(c.P, (x) => x / tt * DAY);
    Ts[s] = Float64Array.from(c.Ts, (x) => x / tt);
    const u = Array.from(c.u, (x) => x / n), v = Array.from(c.v, (x) => x / n);
    writeFileSync(`${outDir}/precip_${s}.svg`, mapSvg({ title: `${s} precipitation, ${presetName}`, lat: latDeg, lon: lonDeg, values: Array.from(P[s]!), units: 'mm/day', contourStep: 4, latRange: [-60, 70], outline: land, range: [0, 16] }, 900, 360));
    writeFileSync(`${outDir}/ts_${s}.svg`, mapSvg({ title: `${s} surface temperature, ${presetName}`, lat: latDeg, lon: lonDeg, values: Array.from(Ts[s]!), units: 'K', contourStep: 10, latRange: [-90, 90], outline: land }, 900, 420));
    writeFileSync(`${outDir}/wind850_${s}.svg`, mapSvg({ title: `${s} wind at sigma=${model.lev.sigma[k850]!.toFixed(2)} (arrows) and zonal wind (colour), ${presetName}`, lat: latDeg, lon: lonDeg, values: u, units: 'm/s', contourStep: 5, latRange: [-60, 70], outline: land, diverging: true, vectors: { u, v, stride: cfg.trunc > 21 ? 3 : 2, scale: 1.1 } }, 900, 360));
  }
  const monsoon = [
    ['South Asia (10–30N, 70–100E)', 10, 30, 70, 100], ['West Africa (5–20N, 340–360E+0–20E)', 5, 20, 340, 360],
    ['East Asia (20–40N, 105–125E)', 20, 40, 105, 125], ['Australia (20–10S, 120–150E)', -20, -10, 120, 150],
    ['South America (20–5S, 300–320E)', -20, -5, 300, 320],
  ] as const;
  lines.push('Monsoon precipitation (mm/day), JJA vs DJF:');
  for (const [name, a, b, c, d] of monsoon) lines.push(`  ${name}: JJA ${boxMean(P.JJA!, a, b, c, d).toFixed(2)}  DJF ${boxMean(P.DJF!, a, b, c, d).toFixed(2)}`);
  lines.push('Surface temperature (K), JJA vs DJF:');
  for (const [name, a, b, c, d] of [['Siberia (55–70N, 90–130E)', 55, 70, 90, 130], ['Sahara (18–30N, 0–30E)', 18, 30, 0, 30], ['Equatorial Pacific (5S–5N, 180–260E)', -5, 5, 180, 260], ['Antarctica (70–90S)', -90, -70, 0, 360]] as const)
    lines.push(`  ${name}: JJA ${boxMean(Ts.JJA!, a, b, c, d).toFixed(1)}  DJF ${boxMean(Ts.DJF!, a, b, c, d).toFixed(1)}`);
  const hovVals: number[] = [];
  for (let b = 0; b < NB; b++) for (let j = 0; j < nlat; j++) hovVals.push(hov[b * nlat + j]! / Math.max(hovN[b]!, 1) * DAY);
  // Hovmoeller: rows = 30-day bins after the equinox (drawn with sectionSvg; vertical axis = season fraction)
  writeFileSync(`${outDir}/precip_hovmoller.svg`, sectionSvg({ title: `zonal-mean precipitation vs season (rows: months after 20 Mar), ${presetName}`, lat: latDeg, sigma: Array.from({ length: NB }, (_, b) => (b + 0.5) / NB), values: hovVals, units: 'mm/day', diverging: false, contourStep: 2 }));
  const c = acc.result(tr.lat, model.lev.sigma, model.lev.sigmaHalf, EARTH);
  const s = summarize(c);
  lines.push(`Annual-mean jets: NH ${s.jetMaxNH.u.toFixed(1)} m/s at ${s.jetMaxNH.lat.toFixed(1)}°, SH ${s.jetMaxSH.u.toFixed(1)} m/s at ${s.jetMaxSH.lat.toFixed(1)}°`);
  lines.push('Annual-mean overturning cells:', ...s.cells.map((x) => `  ${x.fromLat.toFixed(1)}° .. ${x.toLat.toFixed(1)}°  peak ${(x.peak / 1e9).toFixed(1)}e9 kg/s`));
  writeFileSync(`${outDir}/u.svg`, sectionSvg({ title: `annual-mean [u], ${presetName}`, lat: c.lat, sigma: c.sigma, values: c.u, units: 'm/s', diverging: true, contourStep: 5 }));
  writeFileSync(`${outDir}/psi.svg`, sectionSvg({ title: `annual-mean ψ, ${presetName}`, lat: c.lat, sigma: c.sigmaHalf.slice(1, K), values: c.psi.slice(nlat, K * nlat).map((x) => x / 1e9), units: '1e9 kg/s', diverging: true, contourStep: 20 }));
  writeFileSync(`${outDir}/climate.json`, JSON.stringify({ lat: latDeg, lon: lonDeg, land, P: { JJA: Array.from(P.JJA!), DJF: Array.from(P.DJF!) }, Ts: { JJA: Array.from(Ts.JJA!), DJF: Array.from(Ts.DJF!) } }));
  writeFileSync(`${outDir}/summary.txt`, lines.join('\n') + '\n\nRun log:\n' + log.join('\n') + '\n');
  console.log(lines.join('\n'));
}
