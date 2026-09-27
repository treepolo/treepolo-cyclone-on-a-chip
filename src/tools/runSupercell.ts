// Weisman & Klemp (1982) supercell in the regional non-hydrostatic model, with the six-class ice
// microphysics (default) or Kessler warm rain.
// Usage: node dist/tools/runSupercell.js [minutes=120] [dx=2000] [outDir=results/supercell] [ice|kessler]
import { mkdirSync, writeFileSync } from 'node:fs';
import { RegionalModel } from '../regional/core.js';
import { KesslerMicrophysics, weismanKlemp, QV, QC, QR } from '../regional/kessler.js';
import { IceMicrophysics, QI, QS, QG } from '../regional/ice.js';
import { xySvg, sectionSvg } from './plot.js';

const minutes = Number(process.argv[2] ?? 120), dx = Number(process.argv[3] ?? 2000);
const outDir = process.argv[4] ?? 'results/supercell';
const ice = (process.argv[5] ?? 'ice') === 'ice';
mkdirSync(outDir, { recursive: true });
const L = 120000, nx = Math.round(L / dx), nz = 40, dz = 500;
const dt = Math.min(6, 3 * dx / 1000);
const m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt, nsound: 6, f: 0, beta: 0.2, divDamp: 0.1, dampDepth: 6000, dampRate: 1 / 300, kdiff2: 0 }, weismanKlemp, ice ? 6 : 3);
const mp: { rainAcc: Float64Array; apply(dt: number): void } = ice ? new IceMicrophysics(m) : new KesslerMicrophysics(m);
// wind: unidirectional tanh shear (Us = 30 m/s over zs = 3 km), minus an approximate storm motion
m.setBaseWind((z) => ({ u: 30 * Math.tanh(z / 3000) - 15, v: 0 }));
for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) m.scalars[QV]![m.idx(i, j, k)] = m.qv0[k]!;
// warm bubble: 2 K, horizontal radius 10 km, vertical radius 1.4 km, centred at 1.4 km
const xc = L * 0.35, yc = L / 2;
for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
  const x = (i + 0.5) * dx, y = (j + 0.5) * dx, z = m.zc[k]!;
  const r = Math.sqrt(((x - xc) / 10000) ** 2 + ((y - yc) / 10000) ** 2 + ((z - 1400) / 1400) ** 2);
  if (r < 1) m.th[m.idx(i, j, k)] = m.th[m.idx(i, j, k)]! + 2 * Math.cos(0.5 * Math.PI * r) ** 2;
}
const lines: string[] = [];
const say = (s: string): void => { console.log(s); lines.push(s); };
say(`Weisman–Klemp supercell: ${nx}x${nx}x${nz}, dx=${dx} m, dz=${dz} m, dt=${dt} s, ${ice ? 'six-class ice' : 'Kessler warm-rain'} microphysics`);
const t0 = Date.now();
let k4 = 0; for (let k = 0; k < nz; k++) if (Math.abs(m.zc[k]! - 4000) < Math.abs(m.zc[k4]! - 4000)) k4 = k;
const snap = (tag: string): void => {
  const w4: number[] = [], vort: number[] = [], rain: number[] = [], qr1: number[] = [];
  for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
    const q = m.idx(i, j, k4);
    w4.push(0.5 * (m.w[q]! + m.w[q + m.plane]!));
    const dvdx = (m.v[m.idx((i + 1) % nx, j, k4)]! - m.v[q]!) / dx, dudy = (m.u[m.idx(i, (j + 1) % nx, k4)]! - m.u[q]!) / dx;
    vort.push((dvdx - dudy) * 1e3);
    rain.push(mp.rainAcc[j * nx + i]!);
    qr1.push(m.scalars[QR]![m.idx(i, j, 1)]! * 1000);
  }
  writeFileSync(`${outDir}/w4km_${tag}.svg`, xySvg({ title: `w at z=${(m.zc[k4]! / 1000).toFixed(1)} km, t=${tag} min`, nx, ny: nx, dx, values: w4, units: 'm/s', diverging: true, contourStep: 5 }));
  writeFileSync(`${outDir}/vort4km_${tag}.svg`, xySvg({ title: `vertical vorticity at ${(m.zc[k4]! / 1000).toFixed(1)} km, t=${tag} min`, nx, ny: nx, dx, values: vort, units: '1e-3 s^-1', diverging: true, contourStep: 2 }));
  writeFileSync(`${outDir}/rain_${tag}.svg`, xySvg({ title: `accumulated surface rain, t=${tag} min`, nx, ny: nx, dx, values: rain, units: 'mm', diverging: false, contourStep: 5 }));
  if (ice) {
    // west-east section through the column of maximum 4-km updraft: liquid (qc + qr) and ice (qi + qs + qg)
    let jm = 0, wm = -1;
    for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) { const w = m.w[m.idx(i, j, k4)]!; if (w > wm) { wm = w; jm = j; } }
    const xs = Array.from({ length: nx }, (_, i) => (i + 0.5) * dx / 1000), lev = Array.from({ length: nz }, (_, k) => 1 - (k + 0.5) / nz);
    const liq: number[] = [], icev: number[] = [];
    for (let k = nz - 1; k >= 0; k--) for (let i = 0; i < nx; i++) {
      const q = m.idx(i, jm, k);
      liq.push((m.scalars[QC]![q]! + m.scalars[QR]![q]!) * 1e3);
      icev.push((m.scalars[QI]![q]! + m.scalars[QS]![q]! + m.scalars[QG]![q]!) * 1e3);
    }
    writeFileSync(`${outDir}/xz_liquid_${tag}.svg`, sectionSvg({ title: `cloud + rain water, x-z through the updraft (y=${((jm + 0.5) * dx / 1000).toFixed(0)} km; x 0-${L / 1000} km, z 0-${nz * dz / 1000} km), t=${tag} min`, lat: xs, sigma: lev, values: liq, units: 'g/kg', diverging: false, contourStep: 1 }));
    writeFileSync(`${outDir}/xz_ice_${tag}.svg`, sectionSvg({ title: `cloud ice + snow + graupel, same section, t=${tag} min`, lat: xs, sigma: lev, values: icev, units: 'g/kg', diverging: false, contourStep: 1 }));
  }
  writeFileSync(`${outDir}/qr_sfc_${tag}.svg`, xySvg({ title: `rain water near the surface (reflectivity proxy), t=${tag} min`, nx, ny: nx, dx, values: qr1, units: 'g/kg', diverging: false, contourStep: 1 }));
};
while (m.time < minutes * 60 - 1e-9) {
  m.step();
  mp.apply(dt);
  if (m.steps % Math.round(300 / dt) === 0) {
    const tmin = Math.round(m.time / 60);
    let wmax = 0, wmin = 0, qcmax = 0, qrmax = 0, cloudTop = 0, qimax = 0, qsmax = 0, qgmax = 0;
    for (let k = 0; k <= nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
      const w = m.w[m.idx(i, j, k)]!; wmax = Math.max(wmax, w); wmin = Math.min(wmin, w);
      if (k < nz) {
        const q = m.idx(i, j, k), qc = m.scalars[QC]![q]!, cl = qc + (ice ? m.scalars[QI]![q]! : 0);
        qcmax = Math.max(qcmax, qc); qrmax = Math.max(qrmax, m.scalars[QR]![q]!); if (cl > 1e-5) cloudTop = Math.max(cloudTop, m.zc[k]!);
        if (ice) { qimax = Math.max(qimax, m.scalars[QI]![q]!); qsmax = Math.max(qsmax, m.scalars[QS]![q]!); qgmax = Math.max(qgmax, m.scalars[QG]![q]!); }
      }
    }
    let rmax = 0, rsum = 0; for (const r of mp.rainAcc) { rmax = Math.max(rmax, r); rsum += r; }
    say(`t=${tmin} min: w ${wmin.toFixed(1)}..${wmax.toFixed(1)} m/s  qc max ${(qcmax * 1000).toFixed(2)} g/kg  qr max ${(qrmax * 1000).toFixed(2)} g/kg  ${ice ? `qi ${(qimax * 1000).toFixed(2)} qs ${(qsmax * 1000).toFixed(2)} qg ${(qgmax * 1000).toFixed(2)} g/kg  ` : ''}cloud top ${(cloudTop / 1000).toFixed(1)} km  precip max ${rmax.toFixed(1)} mm  wall ${((Date.now() - t0) / 1000).toFixed(0)} s`);
    if (!Number.isFinite(wmax)) { say('NaN'); process.exit(1); }
    if (tmin % 30 === 0) snap(String(tmin));
  }
}
writeFileSync(`${outDir}/summary.txt`, lines.join('\n') + '\n');
