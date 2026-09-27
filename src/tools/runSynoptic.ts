// Synoptic snapshots of the Earth climate model (Node): sea-level pressure over 850-hPa temperature,
// precipitation and 250-hPa wind, from a runEarth checkpoint. Usage:
//   node dist/tools/runSynoptic.js <checkpointDir> [advanceDays=0] [snapshots=4] [everyHours=24] [outDir] [preset=EARTH_T21]
// Nothing here prescribes weather: the maps show whatever cyclones, fronts, air masses and jets the
// model produces.

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { DAY, DRY_AIR, EARTH } from '../core/constants.js';
import { EARTH_PRESETS, EarthData, createEarth } from '../model/presets.js';
import { mapSvg } from './plot.js';

const ckDir = process.argv[2];
if (!ckDir) throw new Error('usage: runSynoptic <checkpointDir> [advanceDays] [snapshots] [everyHours] [outDir] [preset]');
const advanceDays = Number(process.argv[3] ?? 0), nSnap = Number(process.argv[4] ?? 4), every = Number(process.argv[5] ?? 24);
const outDir = process.argv[6] ?? `${ckDir}/synoptic`;
const presetName = process.argv[7] ?? 'EARTH_T21';
const cfg = EARTH_PRESETS[presetName]!;
mkdirSync(outDir, { recursive: true });
const data = JSON.parse(readFileSync('data/earth_t42.json', 'utf8')) as EarthData;
const { model, physics } = createEarth(cfg, data);
const ng = model.ng, K = model.K, nlat = model.tr.nlat, nlon = model.tr.nlon;
{
  const buf = readFileSync(`${ckDir}/checkpoint.bin`);
  const all = new Float64Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const nState = model.exportState().data.length;
  model.importState({ time: all[0]!, steps: all[1]!, data: all.subarray(2, 2 + nState) });
  physics.f.sst.set(all.subarray(2 + nState, 2 + nState + ng));
  physics.f.bucket.set(all.subarray(2 + nState + ng, 2 + nState + 2 * ng));
  if (all.length >= 2 + nState + 3 * ng) physics.f.ice.set(all.subarray(2 + nState + 2 * ng, 2 + nState + 3 * ng));
}
const stepsPerDay = Math.round(DAY / cfg.dt);
for (let s = 0; s < advanceDays * stepsPerDay; s++) model.step();
const year = physics.cfg.yearLength;
const latDeg = Array.from(model.tr.lat, (x) => x * 180 / Math.PI), lonDeg = Array.from(model.tr.lon, (x) => x * 180 / Math.PI);
const land = Array.from(physics.surface.land);
const phis = new Float64Array(ng);
model.surfaceGeopotentialGrid(phis);
const lev = model.lev.sigma;
const nearest = (s: number): number => { let k = 0; for (let i = 0; i < K; i++) if (Math.abs(lev[i]! - s) < Math.abs(lev[k]! - s)) k = i; return k; };
const k250 = nearest(0.25);
const months = ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb'];
for (let n = 0; n < nSnap; n++) {
  if (n > 0) for (let s = 0; s < Math.round(every * 3600 / cfg.dt); s++) model.step();
  const acc0 = Float64Array.from(physics.f.precipConv, (x, p) => x + physics.f.precipLS[p]!);
  for (let s = 0; s < Math.round(6 * 3600 / cfg.dt); s++) model.step();      // 6-h precipitation accumulation
  const g = model.refreshGrid();
  const slp = new Array<number>(ng), t850 = new Array<number>(ng), pr = new Array<number>(ng), u250 = new Array<number>(ng), v250 = new Array<number>(ng), spd = new Array<number>(ng);
  for (let p = 0; p < ng; p++) {
    const ps = g.ps[p]!;
    // 850 hPa temperature: linear in ln p between model levels (extrapolated below ground with 6.5 K/km)
    let T8 = NaN;
    for (let k = 0; k < K - 1; k++) {
      const pa = lev[k]! * ps, pb = lev[k + 1]! * ps;
      if (pa <= 85000 && pb >= 85000) { const w = Math.log(85000 / pa) / Math.log(pb / pa); T8 = g.T[k * ng + p]! * (1 - w) + g.T[(k + 1) * ng + p]! * w; break; }
    }
    const Tb = g.T[(K - 1) * ng + p]!;
    if (!Number.isFinite(T8)) T8 = Tb * Math.pow(85000 / (lev[K - 1]! * ps), DRY_AIR.rd * 0.0065 / EARTH.gravity);
    t850[p] = T8 - 273.15;
    // mean-sea-level pressure: hydrostatic reduction with a 6.5 K/km lapse rate from the lowest level
    const Ts = Tb * Math.pow(1 / lev[K - 1]!, DRY_AIR.rd * 0.0065 / EARTH.gravity);
    const zs = phis[p]! / EARTH.gravity;
    // not meaningful over high terrain (standard practice: masked above 1500 m)
    slp[p] = zs > 1500 ? NaN : ps * Math.pow(1 + 0.0065 * zs / Ts, EARTH.gravity / (DRY_AIR.rd * 0.0065)) / 100;
    pr[p] = (physics.f.precipConv[p]! + physics.f.precipLS[p]! - acc0[p]!) / 6 * 24;
    u250[p] = g.u[k250 * ng + p]!; v250[p] = g.v[k250 * ng + p]!; spd[p] = Math.hypot(u250[p]!, v250[p]!);
  }
  const doy = ((model.time % year) + year) % year / DAY, mon = months[Math.floor(doy / (year / DAY / 12)) % 12];
  const tag = `d${(model.time / DAY).toFixed(0)}`;
  const when = `day ${(model.time / DAY).toFixed(1)} (${mon}, ${doy.toFixed(0)} d after the March equinox)`;
  writeFileSync(`${outDir}/slp_t850_${tag}.svg`, mapSvg({ title: `sea-level pressure (contours, 4 hPa) over 850-hPa temperature, ${when}`, lat: latDeg, lon: lonDeg, values: t850, units: '°C', contourStep: 5, latRange: [-80, 80], outline: land, diverging: true, range: [-30, 30], overlay: { values: slp, step: 4 } }, 1000, 480));
  writeFileSync(`${outDir}/precip_${tag}.svg`, mapSvg({ title: `precipitation (6-h mean), ${when}`, lat: latDeg, lon: lonDeg, values: pr, units: 'mm/day', contourStep: 10, latRange: [-80, 80], outline: land, range: [0, 40] }, 1000, 480));
  writeFileSync(`${outDir}/jet250_${tag}.svg`, mapSvg({ title: `wind speed and vectors at sigma=${lev[k250]!.toFixed(2)}, ${when}`, lat: latDeg, lon: lonDeg, values: spd, units: 'm/s', contourStep: 20, latRange: [-80, 80], outline: land, range: [0, 80], vectors: { u: u250, v: v250, stride: nlon > 64 ? 3 : 2, scale: 0.5 } }, 1000, 480));
  let slpMin = Infinity, pMin = 0; for (let p = 0; p < ng; p++) if (slp[p]! < slpMin) { slpMin = slp[p]!; pMin = p; }
  console.log(`${when}: SLP min ${slpMin.toFixed(1)} hPa at ${latDeg[Math.floor(pMin / nlon)]!.toFixed(0)}°, ${lonDeg[pMin % nlon]!.toFixed(0)}°E, T850 ${Math.min(...t850).toFixed(1)}..${Math.max(...t850).toFixed(1)} °C, max 250-hPa wind ${Math.max(...spd).toFixed(1)} m/s`);
}
