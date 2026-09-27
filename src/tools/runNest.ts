// One-way nested regional forecast inside the Earth climate model (Node). Usage:
//   node dist/tools/runNest.js <checkpointDir> <latDeg> <lonDegE> [hours=24] [dxKm=20] [advanceDays=0] [outDir] [preset=EARTH_T21]
// Loads a global checkpoint written by runEarth, optionally advances the global model, then runs the
// regional non-hydrostatic model over a 1200 km square centred at (lat, lon). The global model keeps
// running alongside; the regional lateral-boundary targets are updated every global step (linear in
// time between 3-hourly global states). Writes hourly diagnostics and maps of rain, low-level wind,
// cloud and a comparison with the global model's own precipitation over the same box.

import { mkdirSync, writeFileSync, readFileSync, existsSync } from 'node:fs';
import { DAY, EARTH } from '../core/constants.js';
import { EARTH_PRESETS, EarthData, createEarth } from '../model/presets.js';
import type { Dycore } from '../model/dycore.js';
import { nestFromGlobal, nestTargets, sampleSurface, GlobalSnapshot, NestSpec } from '../regional/nest.js';
import { IceMicrophysics, QC, QR, QI, QS, QG } from '../regional/ice.js';
import { RegionalPhysics } from '../regional/physics.js';
import type { BoundaryTargets } from '../regional/core.js';
import { xySvg } from './plot.js';

const [ckDir, latS, lonS] = process.argv.slice(2);
if (!ckDir || latS === undefined || lonS === undefined) throw new Error('usage: runNest <checkpointDir> <latDeg> <lonDegE> [hours] [dxKm] [advanceDays] [outDir] [preset]');
const hours = Number(process.argv[5] ?? 24), dxKm = Number(process.argv[6] ?? 20), advanceDays = Number(process.argv[7] ?? 0);
const outDir = process.argv[8] ?? `results/nest_${latS}_${lonS}`;
const presetName = process.argv[9] ?? 'EARTH_T21';
const cfg = EARTH_PRESETS[presetName];
if (!cfg) throw new Error(`unknown preset ${presetName}`);
mkdirSync(outDir, { recursive: true });

// ---------------- global model from checkpoint
const data = JSON.parse(readFileSync('data/earth_t42.json', 'utf8')) as EarthData;
const { model: gm, physics } = createEarth(cfg, data);
const ng = gm.ng;
const ckpt = `${ckDir}/checkpoint.bin`;
if (!existsSync(ckpt)) throw new Error(`no checkpoint at ${ckpt}`);
{
  const buf = readFileSync(ckpt);
  const all = new Float64Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const nState = gm.exportState().data.length;
  gm.importState({ time: all[0]!, steps: all[1]!, data: all.subarray(2, 2 + nState) });
  physics.f.sst.set(all.subarray(2 + nState, 2 + nState + ng));
  physics.f.bucket.set(all.subarray(2 + nState + ng, 2 + nState + 2 * ng));
  if (all.length >= 2 + nState + 3 * ng) physics.f.ice.set(all.subarray(2 + nState + 2 * ng, 2 + nState + 3 * ng));
}
const log: string[] = [];
const say = (s: string): void => { console.log(s); log.push(s); };
const t0 = Date.now();
const year = physics.cfg.yearLength;
const doy = (): number => ((gm.time % year) + year) % year / DAY;
say(`global ${presetName} resumed at day ${(gm.time / DAY).toFixed(1)} (day ${doy().toFixed(0)} after the March equinox)`);
for (let d = 0; d < advanceDays; d++) {
  for (let s = 0; s < Math.round(DAY / cfg.dt); s++) gm.step();
  if ((d + 1) % 10 === 0) say(`  advanced to day ${doy().toFixed(0)} after the equinox  wall=${((Date.now() - t0) / 1000).toFixed(0)}s`);
}

function snapshot(m: Dycore): GlobalSnapshot {
  const g = m.refreshGrid(), phis = new Float64Array(m.ng);
  m.surfaceGeopotentialGrid(phis);
  return {
    nlat: m.tr.nlat, nlon: m.tr.nlon, K: m.K, lat: m.tr.lat, lon: m.tr.lon, sigma: m.lev.sigma, sigmaHalf: m.lev.sigmaHalf,
    u: Float64Array.from(g.u), v: Float64Array.from(g.v), T: Float64Array.from(g.T), ps: Float64Array.from(g.ps), q: Float64Array.from(m.q), phis,
  };
}

// ---------------- nest
// 'auto' location: the grid point (|lat| < latS-limit given as lonS, default 30) with the largest
// global-model precipitation over the last 6 h, i.e. where the global model is convecting now
let lat0 = Number(latS) * Math.PI / 180, lon0 = Number(lonS) * Math.PI / 180;
if (latS === 'auto') {
  const lim = (Number(lonS) || 30) * Math.PI / 180, acc0 = Float64Array.from(physics.f.precipConv, (x, p) => x + physics.f.precipLS[p]!);
  for (let s = 0; s < Math.round(6 * 3600 / cfg.dt); s++) gm.step();
  let best = -1, bp = 0;
  for (let p = 0; p < ng; p++) {
    const la = gm.tr.lat[Math.floor(p / gm.tr.nlon)]!;
    if (Math.abs(la) > lim) continue;
    const r = physics.f.precipConv[p]! + physics.f.precipLS[p]! - acc0[p]!;
    if (r > best) { best = r; bp = p; }
  }
  lat0 = gm.tr.lat[Math.floor(bp / gm.tr.nlon)]!; lon0 = gm.tr.lon[bp % gm.tr.nlon]!;
  say(`auto location: global precipitation maximum ${(best / 6 * 24).toFixed(1)} mm/day at ${(lat0 * 180 / Math.PI).toFixed(1)}°, ${(lon0 * 180 / Math.PI).toFixed(1)}°E`);
}
const fine = dxKm < 15;
const spec: NestSpec = { lat0, lon0, L: 1200000, dx: dxKm * 1000, nz: fine ? 30 : 24, dz: fine ? 600 : 750, dt: fine ? 40 : 60, nsound: 6 };
let snap = snapshot(gm);
const nest = nestFromGlobal(snap, spec, 6);
const m = nest.model, { nx, ny, nz } = m.c, n2 = nx * ny;
// diagnostics exclude the lateral relaxation zone
const nr = m.c.relaxCells ?? 8, nInterior = (nx - 2 * nr) * (ny - 2 * nr);
const mp = new IceMicrophysics(m);
const wet = Float64Array.from({ length: ng }, (_, p) => physics.surface.land[p] ? Math.min(1, physics.f.bucket[p]! / (0.75 * physics.cfg.bucketMax)) : 1);
const tsk = sampleSurface(snap, spec, physics.f.sst), wetR = sampleSurface(snap, spec, wet);
const land: number[] = Array.from(sampleSurface(snap, spec, physics.surface.land), (x) => (x > 0.5 ? 1 : 0));
new RegionalPhysics(m, { lh: 0.2 * spec.dx, lv: 100, sst: 0, ck: 1.2e-3, radTau: 0, radMax: 0, surface: { tsk, wet: wetR } });
say(`${nest.description}; ${nx}x${ny}x${nz}, dz ${spec.dz} m, dt ${spec.dt} s; land fraction ${(land.reduce((a, b) => a + b, 0) / n2).toFixed(2)}; skin T ${Math.min(...tsk).toFixed(1)}..${Math.max(...tsk).toFixed(1)} K`);

// boundary targets: linear in time between global states 3 h apart
const tgap = 3 * 3600, stepsPerGap = Math.round(tgap / cfg.dt);
let bA: BoundaryTargets = nest.boundary, bB: BoundaryTargets = bA, tA = 0;
const cur = m.boundary!;
const cloneB = (b: BoundaryTargets): BoundaryTargets => ({ u: Float64Array.from(b.u), v: Float64Array.from(b.v), th: Float64Array.from(b.th), qv: b.qv ? Float64Array.from(b.qv) : null, pp: b.pp ? Float64Array.from(b.pp) : null });
m.boundary = cloneB(cur);
let globalAdvanced = 0;
function advanceGlobal(): void {
  globalAdvanced += tgap;
  for (let s = 0; s < stepsPerGap; s++) gm.step();
  snap = snapshot(gm);
  bA = bB; bB = nestTargets(snap, spec, m);
}
bA = cloneB(nest.boundary);
// global precipitation over the box: mean over global points inside the domain
const boxPts: number[] = [];
{
  const half = (spec.L / 2 - nr * spec.dx) / EARTH.radius;
  for (let j = 0; j < gm.tr.nlat; j++) for (let i = 0; i < gm.tr.nlon; i++) {
    const la = gm.tr.lat[j]!;
    let dl = gm.tr.lon[i]! - lon0; dl -= Math.round(dl / (2 * Math.PI)) * 2 * Math.PI;
    if (Math.abs(la - lat0) <= half && Math.abs(dl * Math.cos(lat0)) <= half) boxPts.push(j * gm.tr.nlon + i);
  }
}
const gPrecip = (): number => { let s = 0; for (const p of boxPts) s += physics.f.precipConv[p]! + physics.f.precipLS[p]!; return boxPts.length ? s / boxPts.length : NaN; };
const gP0 = gPrecip();
advanceGlobal();

const series: { t: number[]; wmax: number[]; rain: number[]; vmax: number[]; cloud: number[] } = { t: [], wmax: [], rain: [], vmax: [], cloud: [] };
let rainPrev = 0;
const nSteps = Math.round(hours * 3600 / spec.dt);
const lerpB = (w: number): void => {
  const b = m.boundary!;
  const mix = (o: Float64Array, a: Float64Array, c: Float64Array): void => { for (let i = 0; i < o.length; i++) o[i] = a[i]! + w * (c[i]! - a[i]!); };
  mix(b.u, bA.u, bB.u); mix(b.v, bA.v, bB.v); mix(b.th, bA.th, bB.th);
  if (b.qv && bA.qv && bB.qv) mix(b.qv, bA.qv, bB.qv);
  if (b.pp && bA.pp && bB.pp) mix(b.pp, bA.pp, bB.pp);
};
function maps(tag: string): void {
  const rain = Array.from(mp.rainAcc), u = new Array<number>(n2), v = new Array<number>(n2), cw = new Array<number>(n2), spd = new Array<number>(n2);
  let k1 = 0;
  for (let k = 0; k < nz; k++) if (Math.abs(m.zc[k]! - 1000) < Math.abs(m.zc[k1]! - 1000)) k1 = k;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const q = m.idx(i, j, k1), c = j * nx + i;
    u[c] = 0.5 * (m.u[q]! + m.u[q + 1]!); v[c] = 0.5 * (m.v[q]! + m.v[q + m.sx]!); spd[c] = Math.hypot(u[c]!, v[c]!);
    let s = 0;
    for (let k = 0; k < nz; k++) { const qq = m.idx(i, j, k); for (const sp of [QC, QR, QI, QS, QG]) s += m.rho0[k]! * m.scalars[sp]![qq]! * spec.dz; }
    cw[c] = s;
  }
  writeFileSync(`${outDir}/rain_${tag}.svg`, xySvg({ title: `accumulated rain, ${tag}`, nx, ny, dx: spec.dx, values: rain, units: 'mm', diverging: false, contourStep: 20, outline: land }));
  writeFileSync(`${outDir}/wind1km_${tag}.svg`, xySvg({ title: `wind at ${(m.zc[k1]! / 1000).toFixed(1)} km, ${tag}`, nx, ny, dx: spec.dx, values: spd, units: 'm/s', diverging: false, contourStep: 5, outline: land, vectors: { u, v, stride: 4, scale: 1.2 } }));
  writeFileSync(`${outDir}/condensate_${tag}.svg`, xySvg({ title: `column condensate (cloud, rain, ice, snow, graupel), ${tag}`, nx, ny, dx: spec.dx, values: cw, units: 'kg/m2', diverging: false, contourStep: 2, outline: land }));
}

for (let s = 1; s <= nSteps; s++) {
  const t = s * spec.dt;
  while (t > tA + tgap) { tA += tgap; advanceGlobal(); }
  lerpB((t - tA) / tgap);
  m.step(); mp.apply(spec.dt);
  if (!Number.isFinite(m.th[m.idx(nx >> 1, ny >> 1, 1)]!)) { say(`blow-up at t=${(t / 3600).toFixed(2)} h`); break; }
  if (t % 3600 < spec.dt / 2 || s === nSteps) {
    let wmax = 0, qcm = 0, vmax = 0, rsum = 0, cloudy = 0;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const q = m.idx(i, j, k); wmax = Math.max(wmax, m.w[q]!); qcm = Math.max(qcm, m.scalars[QC]![q]!); }
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const q = m.idx(i, j, 0);
      vmax = Math.max(vmax, Math.hypot(0.5 * (m.u[q]! + m.u[q + 1]!), 0.5 * (m.v[q]! + m.v[q + m.sx]!)));
      if (i >= nr && j >= nr && i < nx - nr && j < ny - nr) rsum += mp.rainAcc[j * nx + i]!;
      let c = 0; for (let k = 0; k < nz; k++) c = Math.max(c, m.scalars[QC]![m.idx(i, j, k)]!);
      if (c > 1e-4) cloudy++;
    }
    const rmean = rsum / nInterior, rate = (rmean - rainPrev) / ((t - (series.t.length ? series.t[series.t.length - 1]! * 3600 : 0)) / 3600);
    rainPrev = rmean;
    series.t.push(t / 3600); series.wmax.push(wmax); series.rain.push(rate * 24); series.vmax.push(vmax); series.cloud.push(cloudy / n2);
    say(`t=${(t / 3600).toFixed(0)} h: wmax ${wmax.toFixed(1)} m/s  qc max ${(qcm * 1000).toFixed(2)} g/kg  cloud cover ${(100 * cloudy / n2).toFixed(0)}%  domain rain ${(rate * 24).toFixed(1)} mm/day  sfc vmax ${vmax.toFixed(1)} m/s  wall ${((Date.now() - t0) / 1000).toFixed(0)} s`);
    if ((t / 3600) % 12 < 0.01) maps(`${(t / 3600).toFixed(0)}h`);
    if (process.env.PROFILE && (t / 3600) % 3 < 0.01) {
      for (let k = 0; k < nz; k++) {
        let th = 0, qv = 0, qc = 0, rh = 0;
        for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
          const q = m.idx(i, j, k), pi = m.pi0[k]! + m.pp[q]!, p = 1e5 * Math.pow(pi, 1004.5 / 287.05), T = m.th[q]! * pi;
          const qs = 380 / p * Math.exp(17.27 * (T - 273.15) / (T - 35.86));
          th += m.th[q]!; qv += m.scalars[0]![q]!; qc += m.scalars[QC]![q]!; rh += m.scalars[0]![q]! / qs;
        }
        say(`   z=${(m.zc[k]! / 1000).toFixed(2)} km  th ${(th / n2).toFixed(2)} (th0 ${m.th0[k]!.toFixed(2)})  qv ${(qv / n2 * 1000).toFixed(2)} g/kg  qc ${(qc / n2 * 1000).toFixed(3)}  RH ${(rh / n2 * 100).toFixed(0)}%`);
      }
    }
  }
}
const gP = (gPrecip() - gP0) / globalAdvanced * DAY;
let rtot = 0; for (let j = nr; j < ny - nr; j++) for (let i = nr; i < nx - nr; i++) rtot += mp.rainAcc[j * nx + i]!;
say(`summary: regional interior-mean rain (relaxation zone excluded) ${(rtot / nInterior / (series.t[series.t.length - 1]! / 24)).toFixed(2)} mm/day; global model over the same box ${gP.toFixed(2)} mm/day (${boxPts.length} global points)`);
writeFileSync(`${outDir}/summary.txt`, log.join('\n') + '\n');
