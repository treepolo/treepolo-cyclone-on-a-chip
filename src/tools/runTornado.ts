// Tornado-scale supercell environment (src/regional/supercell.ts) on the CPU (Node). Usage:
//   node dist/tools/runTornado.js [minutes=90] [dx=500] [L=30000] [nz=40] [dz=400] [dt=3] [outDir] [env=default|wk82]
// env: the default strong low-level-shear environment, or the former WK82 one (TORNADO_WK82).
// At 500 m this checks the set-up (a rotating supercell that stays inside the moving domain);
// tornado-like vortices need ~100-200 m grids (GPU).
import { mkdirSync, writeFileSync } from 'node:fs';
import { tornadoExperiment, StormTracker, TORNADO_DEFAULT, TORNADO_WK82 } from '../regional/supercell.js';
import { IceMicrophysics, QC, QI } from '../regional/ice.js';
import { RegionalPhysics } from '../regional/physics.js';
import { xySvg } from './plot.js';

const a = process.argv.slice(2).map(Number);
const minutes = a[0] || 90, dx = a[1] || 500, L = a[2] || 40000, nz = a[3] || 40, dz = a[4] || 400, dt = a[5] || 3;
const outDir = process.argv[8] ?? `results/tornado_${dx}m`;
mkdirSync(outDir, { recursive: true });
const e = tornadoExperiment(dx, L, nz, dz, dt, 6, process.argv[9] === 'wk82' ? TORNADO_WK82 : TORNADO_DEFAULT);
const m = e.model, mp = new IceMicrophysics(m);
new RegionalPhysics(m, e.physics);
const { nx, ny } = m.c;
const lines: string[] = [];
const say = (s: string): void => { console.log(s); lines.push(s); };
say(`${e.description}; ${nx}x${ny}x${nz}, dz ${dz} m, dt ${dt} s; frame ${e.frame.u.toFixed(2)}, ${e.frame.v.toFixed(2)} m/s`);
const t0 = Date.now();
const kAt = (z: number): number => { let k = 0; for (let i = 0; i < nz; i++) if (Math.abs(m.zc[i]! - z) < Math.abs(m.zc[k]! - z)) k = i; return k; };
const k3 = kAt(3000);
const zeta = (k: number, i: number, j: number): number => { const q = m.idx(i, j, k); return (m.v[q]! - m.v[q - 1]!) / dx - (m.u[q]! - m.u[q - m.sx]!) / dx; };
const tracker = new StormTracker();
const frame = e.physics.frameVel!;
while (m.time < minutes * 60 - 1e-9) {
  m.step(); mp.apply(dt);
  if (m.steps % Math.round(600 / dt) === 0) {
    const a = tracker.update(m);
    if (a) {
      if (a.du || a.dv) { m.shiftFrame(a.du, a.dv); frame.u += a.du; frame.v += a.dv; }
      if (a.di || a.dj) m.roll(a.di, a.dj, [mp.rainAcc, mp.snowAcc]);
      say(`  tracker: storm at (${(a.x / 1000).toFixed(1)}, ${(a.y / 1000).toFixed(1)}) km; frame velocity now (${frame.u.toFixed(1)}, ${frame.v.toFixed(1)}) m/s; roll (${a.di}, ${a.dj})`);
    }
  }
  if (m.steps % Math.round(300 / dt) === 0) {
    let wmax = 0, z0 = 0, z3 = 0, iw = 0, jw = 0, vg = 0, cloudTop = 0;
    for (let j = 1; j < ny; j++) for (let i = 1; i < nx; i++) {
      z0 = Math.max(z0, zeta(0, i, j)); z3 = Math.max(z3, zeta(k3, i, j));
      const q = m.idx(i, j, 0);
      vg = Math.max(vg, Math.hypot(0.5 * (m.u[q]! + m.u[q + 1]!) + frame.u, 0.5 * (m.v[q]! + m.v[q + m.sx]!) + frame.v));
    }
    for (let k = 0; k <= nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const w = m.w[m.idx(i, j, k)]!; if (w > wmax) { wmax = w; iw = i; jw = j; } }
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const q = m.idx(i, j, k); if (m.scalars[QC]![q]! + m.scalars[QI]![q]! > 1e-5) cloudTop = Math.max(cloudTop, m.zc[k]!); }
    const tmin = Math.round(m.time / 60);
    say(`t=${tmin} min: w max ${wmax.toFixed(1)} m/s at (${(iw * dx / 1000).toFixed(1)}, ${(jw * dx / 1000).toFixed(1)}) km  zeta max: surface ${z0.toFixed(4)}, 3 km ${z3.toFixed(4)} s^-1  ground-rel. wind max ${vg.toFixed(1)} m/s  cloud top ${(cloudTop / 1000).toFixed(1)} km  wall ${((Date.now() - t0) / 1000).toFixed(0)} s`);
    if (!Number.isFinite(wmax)) { say('NaN'); break; }
    if (tmin % 30 === 0) {
      const w3: number[] = [], z3m: number[] = [];
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const q = m.idx(i, j, k3); w3.push(0.5 * (m.w[q]! + m.w[q + m.plane]!)); z3m.push(i > 0 && j > 0 ? zeta(k3, i, j) * 1e3 : 0); }
      writeFileSync(`${outDir}/w3km_${tmin}.svg`, xySvg({ title: `w at 3 km, t=${tmin} min`, nx, ny, dx, values: w3, units: 'm/s', diverging: true, contourStep: 5 }));
      writeFileSync(`${outDir}/vort3km_${tmin}.svg`, xySvg({ title: `vertical vorticity at 3 km, t=${tmin} min`, nx, ny, dx, values: z3m, units: '1e-3 s^-1', diverging: true, contourStep: 5 }));
    }
  }
}
writeFileSync(`${outDir}/summary.txt`, lines.join('\n') + '\n');
