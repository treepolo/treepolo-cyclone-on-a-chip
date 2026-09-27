// Write a spun-up Earth state for the app (src/model/spinup.ts): run an Earth preset with the
// observation-derived ocean heat transport for a number of days after the spring equinox.
// Usage: node dist/tools/spinupEarth.js [preset=EARTH_T42] [days=110] [out=data/spinup_earth_t42q.bin]
import { readFileSync, writeFileSync } from 'node:fs';
import { EARTH_PRESETS, EarthData, MonthlyLatLon, createEarth } from '../model/presets.js';
import { encodeSpinup } from '../model/spinup.js';

const preset = process.argv[2] ?? 'EARTH_T42', days = Number(process.argv[3] ?? 110), out = process.argv[4] ?? 'data/spinup_earth_t42q.bin';
const cfg = EARTH_PRESETS[preset];
if (!cfg) throw new Error(`unknown preset ${preset}`);
const data = JSON.parse(readFileSync(cfg.trunc >= 85 ? 'data/earth_512.json' : 'data/earth_t42.json', 'utf8')) as EarthData;
const qflux = JSON.parse(readFileSync('data/qflux_gray_t21.json', 'utf8')) as MonthlyLatLon;
const { model, physics } = createEarth(cfg, data, { qflux: false }, { qflux });
const t0 = Date.now(), end = days * 86400;
let nextLog = 0;
while (model.time < end - 1) {
  model.step();
  if (model.time >= nextLog) {
    nextLog += 5 * 86400;
    console.log(`day ${(model.time / 86400).toFixed(1)}  mean ps ${(model.meanSurfacePressure() / 100).toFixed(2)} hPa  ${((Date.now() - t0) / 60000).toFixed(1)} min`);
  }
}
writeFileSync(out, Buffer.from(encodeSpinup(model, physics)));
console.log(`wrote ${out}`);
