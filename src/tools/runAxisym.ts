// Axisymmetric tropical-cyclone runs from the command line, one or many at once (one worker thread per
// run, up to the number of CPU cores). Prints the intensity every 3 model hours and writes a CSV per run,
// with early stopping when a run blows up. For screening parameters before a 3-D run.
//
// Usage: node dist/tools/runAxisym.js days=4 [key=value ...] [sweep=key:v1,v2,...] [out=results/axisym]
//   keys: sst (deg C), dr (m), lh, lv (m), ck (x1e-3), vmin (m/s), rad (K/day cap of the relaxation),
//         radc (K/day constant tropospheric cooling; 0 = relaxation), rh12 (RH at 12 km, 0-1), vmax0 (m/s), lat (deg)
// The CSV also has the mean precipitation rate (mm/h) in the core (r < 60 km) and in the outer region
// (100-300 km) over each 3-hour interval, and the number of rain rings (local maxima above 0.5 mm/h) beyond 80 km.
// Examples: node dist/tools/runAxisym.js days=5 sweep=vmin:1,3,5
//           node dist/tools/runAxisym.js days=6 cases='base|radc:1|rh12:0.6|radc:1+rh12:0.6'
import { mkdirSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { AxisymModel } from '../regional/axisym.js';
import { IceMicrophysics } from '../regional/ice.js';
import { tropicalSounding, eyewallPeaks } from '../regional/tropical.js';

interface RunSpec { name: string; days: number; sst: number; dr: number; lh: number; lv: number; ck: number; vmin: number; rad: number; radc: number; rh12: number; vmax0: number; lat: number; nz: number; out: string }

/** progress line: printed by the main thread (worker-thread console output is not flushed reliably to files) */
const say = (line: string): void => { if (isMainThread) console.log(line); else parentPort!.postMessage({ line }); };

function run(p: RunSpec): string {
  const nr = Math.round(800000 / p.dr), nz = p.nz, dz = 25000 / nz, f = 2 * 7.292e-5 * Math.sin(p.lat * Math.PI / 180);
  const m = new AxisymModel({ nr, nz, dr: p.dr, dz, dt: Math.min(20, 7.5 * p.dr / 1000), nsound: 6, f, dampDepth: 6000, dampRate: 1 / 300, spongeWidth: 150000, spongeRate: 1 / 900,
    lh: p.lh, lv: p.lv, sst: p.sst + 273.15, ck: p.ck * 1e-3, vmin: p.vmin, radTau: 12 * 3600, radMax: p.rad / 86400, radConst: p.radc / 86400 }, tropicalSounding(p.sst + 273.15, 200, p.rh12));
  const mp = new IceMicrophysics(m);
  m.insertVortex(p.vmax0, 20000);
  const every = Math.round(3 * 3600 / m.a.dt), rows = ['t_h,dp_hPa,vmax_ms,rmw_km,core_mmh,outer_mmh,rings,eyewalls'], t0 = Date.now();
  let acc0 = Float64Array.from(mp.rainAcc), tAcc = 0;
  // area-weighted mean precipitation rate (mm/h) over [r0, r1) since the last sample; rings: rain-rate maxima beyond 80 km
  const rainStats = (): { core: number; outer: number; rings: number } => {
    const hrs = (m.time - tAcc) / 3600, rate = Array.from(mp.rainAcc, (a, i) => (a - acc0[i]!) / Math.max(hrs, 1e-9));
    const mean = (r0: number, r1: number): number => { let s = 0, w = 0; for (let i = 0; i < nr; i++) { const r = m.rc[i + 3]!; if (r >= r0 && r < r1) { s += rate[i]! * r; w += r; } } return w ? s / w : 0; };
    let rings = 0; for (let i = 1; i < nr - 1; i++) if (m.rc[i + 3]! > 80000 && rate[i]! > 0.5 && rate[i]! > rate[i - 1]! && rate[i]! >= rate[i + 1]!) rings++;
    acc0 = Float64Array.from(mp.rainAcc); tAcc = m.time;
    return { core: mean(0, 60000), outer: mean(100000, 300000), rings };
  };
  let k15 = 0; for (let k = 0; k < nz; k++) if (Math.abs(m.zc[k]! - 1500) < Math.abs(m.zc[k15]! - 1500)) k15 = k;
  let last = '';
  while (m.time < p.days * 86400 - 1e-9) {
    m.step(); mp.apply(m.a.dt);
    if (m.steps % every === 0) {
      const r = m.metrics();
      if (!Number.isFinite(r.dp) || r.vmax > 150) { last = `${p.name}: blew up at t = ${(m.time / 3600).toFixed(1)} h`; break; }
      const rr: number[] = [], vt: number[] = [];
      for (let i = 0; i < Math.min(nr, Math.ceil(300000 / p.dr)); i++) { rr.push(m.rc[i + 3]!); vt.push(m.v[m.idx(i, 0, k15)]!); }
      const ew = eyewallPeaks(rr, vt).map((e) => `${(e.r / 1000).toFixed(0)}km/${e.v.toFixed(0)}`).join(' ');
      const rs = rainStats();
      rows.push(`${(m.time / 3600).toFixed(1)},${r.dp.toFixed(2)},${r.vmax.toFixed(2)},${(r.rmw / 1000).toFixed(1)},${rs.core.toFixed(2)},${rs.outer.toFixed(3)},${rs.rings},${ew}`);
      last = `${p.name}: t ${(m.time / 3600).toFixed(0)} h  dp ${r.dp.toFixed(1)} hPa  vmax ${r.vmax.toFixed(1)} m/s  RMW ${(r.rmw / 1000).toFixed(0)} km  rain core ${rs.core.toFixed(1)} outer ${rs.outer.toFixed(2)} mm/h, ${rs.rings} rings  eyewalls ${ew || '-'}  (${((Date.now() - t0) / 1000).toFixed(0)} s)`;
      say(last);
    }
  }
  mkdirSync(p.out, { recursive: true });
  writeFileSync(`${p.out}/${p.name}.csv`, rows.join('\n') + '\n');
  return last;
}

if (isMainThread) {
  const args = Object.fromEntries(process.argv.slice(2).map((a) => a.split('=')).filter((kv) => kv.length === 2) as [string, string][]);
  const base: RunSpec = { name: 'base', days: Number(args.days ?? 4), sst: Number(args.sst ?? 28), dr: Number(args.dr ?? 2000), lh: Number(args.lh ?? 1000), lv: Number(args.lv ?? 100), ck: Number(args.ck ?? 1.2),
    vmin: Number(args.vmin ?? 1), rad: Number(args.rad ?? 2), radc: Number(args.radc ?? 0), rh12: Number(args.rh12 ?? 0.4), vmax0: Number(args.vmax0 ?? 15), lat: Number(args.lat ?? 20), nz: Number(args.nz ?? 50), out: args.out ?? 'results/axisym' };
  const specs: RunSpec[] = [];
  if (args.cases) {
    // cases=base|radc:1|rh12:0.6+radc:1 : each case sets one or more keys ('+'-joined key:value pairs)
    for (const c of args.cases.split('|')) {
      const spec: RunSpec = { ...base, name: c === 'base' ? 'base' : c.replace(/[:+]/g, '_') };
      if (c !== 'base') for (const kv of c.split('+')) { const [k, v] = kv.split(':') as [keyof RunSpec, string]; (spec as unknown as Record<string, number>)[k] = Number(v); }
      specs.push(spec);
    }
  } else if (args.sweep) {
    const [key, list] = args.sweep.split(':') as [keyof RunSpec, string];
    for (const v of list.split(',')) specs.push({ ...base, [key]: Number(v), name: `${String(key)}_${v}` });
  } else specs.push(base);
  const slots = Math.max(1, Math.min(cpus().length, specs.length));
  let next = 0;
  const done: string[] = [];
  const launch = (): void => {
    if (next >= specs.length) return;
    const spec = specs[next++]!;
    const w = new Worker(new URL(import.meta.url), { workerData: spec });
    w.on('message', (msg: string | { line: string }) => { if (typeof msg === 'string') done.push(msg); else console.log(msg.line); });
    w.on('error', (e) => { done.push(`${spec.name}: error ${String(e)}`); launch(); });
    w.on('exit', () => { launch(); if (done.length === specs.length) console.log('\n' + done.join('\n')); });
  };
  for (let i = 0; i < slots; i++) launch();
} else {
  parentPort!.postMessage(run(workerData as RunSpec));
}
