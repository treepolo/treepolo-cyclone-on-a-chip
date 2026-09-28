// Axisymmetric tropical-cyclone runs from the command line, one or many at once (one worker thread per
// run, up to the number of CPU cores). Prints the intensity every 3 model hours and writes a CSV per run,
// with early stopping when a run blows up. For screening parameters before a 3-D run.
//
// Usage: node dist/tools/runAxisym.js days=4 [key=value ...] [sweep=key:v1,v2,...] [out=results/axisym]
//   keys: sst (deg C), dr (m), lh, lv (m), ck (x1e-3), vmin (m/s), rad (K/day), vmax0 (m/s), lat (deg)
// Example: node dist/tools/runAxisym.js days=5 sweep=vmin:1,3,5
import { mkdirSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { AxisymModel } from '../regional/axisym.js';
import { IceMicrophysics } from '../regional/ice.js';
import { tropicalSounding, eyewallPeaks } from '../regional/tropical.js';

interface RunSpec { name: string; days: number; sst: number; dr: number; lh: number; lv: number; ck: number; vmin: number; rad: number; vmax0: number; lat: number; out: string }

function run(p: RunSpec): string {
  const nr = Math.round(800000 / p.dr), nz = 50, dz = 500, f = 2 * 7.292e-5 * Math.sin(p.lat * Math.PI / 180);
  const m = new AxisymModel({ nr, nz, dr: p.dr, dz, dt: Math.min(20, 7.5 * p.dr / 1000), nsound: 6, f, dampDepth: 6000, dampRate: 1 / 300, spongeWidth: 150000, spongeRate: 1 / 900,
    lh: p.lh, lv: p.lv, sst: p.sst + 273.15, ck: p.ck * 1e-3, vmin: p.vmin, radTau: 12 * 3600, radMax: p.rad / 86400 }, tropicalSounding(p.sst + 273.15));
  const mp = new IceMicrophysics(m);
  m.insertVortex(p.vmax0, 20000);
  const every = Math.round(3 * 3600 / m.a.dt), rows = ['t_h,dp_hPa,vmax_ms,rmw_km,eyewalls'], t0 = Date.now();
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
      rows.push(`${(m.time / 3600).toFixed(1)},${r.dp.toFixed(2)},${r.vmax.toFixed(2)},${(r.rmw / 1000).toFixed(1)},${ew}`);
      last = `${p.name}: t ${(m.time / 3600).toFixed(0)} h  dp ${r.dp.toFixed(1)} hPa  vmax ${r.vmax.toFixed(1)} m/s  RMW ${(r.rmw / 1000).toFixed(0)} km  eyewalls ${ew || '-'}  (${((Date.now() - t0) / 1000).toFixed(0)} s)`;
      console.log(last);
    }
  }
  mkdirSync(p.out, { recursive: true });
  writeFileSync(`${p.out}/${p.name}.csv`, rows.join('\n') + '\n');
  return last;
}

if (isMainThread) {
  const args = Object.fromEntries(process.argv.slice(2).map((a) => a.split('=')).filter((kv) => kv.length === 2) as [string, string][]);
  const base: RunSpec = { name: 'base', days: Number(args.days ?? 4), sst: Number(args.sst ?? 28), dr: Number(args.dr ?? 2000), lh: Number(args.lh ?? 1000), lv: Number(args.lv ?? 100), ck: Number(args.ck ?? 1.2),
    vmin: Number(args.vmin ?? 1), rad: Number(args.rad ?? 2), vmax0: Number(args.vmax0 ?? 15), lat: Number(args.lat ?? 20), out: args.out ?? 'results/axisym' };
  const specs: RunSpec[] = [];
  if (args.sweep) {
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
    w.on('message', (s: string) => done.push(s));
    w.on('error', (e) => { done.push(`${spec.name}: error ${String(e)}`); launch(); });
    w.on('exit', () => { launch(); if (done.length === specs.length) console.log('\n' + done.join('\n')); });
  };
  for (let i = 0; i < slots; i++) launch();
} else {
  parentPort!.postMessage(run(workerData as RunSpec));
}
