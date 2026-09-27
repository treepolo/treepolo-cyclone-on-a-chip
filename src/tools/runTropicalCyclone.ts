// Idealised tropical cyclone on an f-plane in the regional non-hydrostatic model (RE87 / BR09 style).
// Usage: node dist/tools/runTropicalCyclone.js [days=8] [dx=15000] [L=1200000] [outDir=results/tc]
import { mkdirSync, writeFileSync } from 'node:fs';
import { RegionalModel } from '../regional/core.js';
import { KesslerMicrophysics, QR } from '../regional/kessler.js';
import { RegionalPhysics } from '../regional/physics.js';
import { tropicalSounding, insertVortex, tcMetrics, eyewallProfile } from '../regional/tropical.js';
import { xySvg, sectionSvg } from './plot.js';

const days = Number(process.argv[2] ?? 8), dx = Number(process.argv[3] ?? 15000), L = Number(process.argv[4] ?? 1200000);
const outDir = process.argv[5] ?? 'results/tc';
mkdirSync(outDir, { recursive: true });
const nx = Math.round(L / dx), nz = 25, dz = 1000, f = 5e-5, sst = 301.15;
const dt = Math.min(60, dx / 250);
const m = new RegionalModel({ nx, ny: nx, nz, dx, dy: dx, dz, dt, nsound: 6, f, beta: 0.3, divDamp: 0.1, dampDepth: 6000, dampRate: 1 / 300, kdiff2: 0 }, tropicalSounding(sst), 3);
const mp = new KesslerMicrophysics(m);
const ph = new RegionalPhysics(m, { lh: 0.2 * dx, lv: 100, sst, ck: 1.2e-3, radTau: 12 * 3600, radMax: 2 / 86400 });
insertVortex(m, f, 15);
const lines: string[] = [];
const say = (s: string): void => { console.log(s); lines.push(s); };
say(`Tropical cyclone: ${nx}x${nx}x${nz}, dx=${dx / 1000} km, dz=${dz} m, dt=${dt} s, f=${f}, SST=${sst} K`);
const t0 = Date.now();
const every = Math.round(3 * 3600 / dt);
while (m.time < days * 86400 - 1e-9) {
  m.step();
  mp.apply(dt);
  if (m.steps % every === 0) {
    const r = tcMetrics(m);
    let lhmax = 0; for (const x of ph.lhf) lhmax = Math.max(lhmax, x);
    say(`t=${(m.time / 3600).toFixed(0)} h: pmin ${r.pmin.toFixed(1)} hPa  vmax ${r.vmax.toFixed(1)} m/s  RMW ${(r.rmw / 1000).toFixed(0)} km  max LHF ${lhmax.toFixed(0)} W/m2  eyewalls ${eyewallProfile(m).peaks.map((e) => `${(e.r / 1000).toFixed(0)} km/${e.v.toFixed(0)}`).join(' + ') || '-'}  wall ${((Date.now() - t0) / 1000).toFixed(0)} s`);
    if (!Number.isFinite(r.pmin)) { say('NaN'); process.exit(1); }
    if (Math.round(m.time / 3600) % 24 === 0) snapshot(`d${Math.round(m.time / 86400)}`, r.ic, r.jc);
  }
}
writeFileSync(`${outDir}/summary.txt`, lines.join('\n') + '\n');

function snapshot(tag: string, ic: number, jc: number): void {
  const spd: number[] = [], qr: number[] = [];
  for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
    const q = m.idx(i, j, 0);
    spd.push(Math.hypot(0.5 * (m.u[q]! + m.u[q + 1]!), 0.5 * (m.v[q]! + m.v[q + m.sx]!)));
    qr.push(m.scalars[QR]![q]! * 1000);
  }
  writeFileSync(`${outDir}/wind_sfc_${tag}.svg`, xySvg({ title: `surface wind speed, ${tag}`, nx, ny: nx, dx, values: spd, units: 'm/s', diverging: false, contourStep: 10 }));
  writeFileSync(`${outDir}/rain_${tag}.svg`, xySvg({ title: `near-surface rain water, ${tag}`, nx, ny: nx, dx, values: qr, units: 'g/kg', diverging: false, contourStep: 0.5 }));
  // azimuthal-mean radius-height sections of tangential wind and w
  const nb = Math.floor(Math.min(nx / 2, 300000 / dx)), vt = new Float64Array(nb * nz), ww = new Float64Array(nb * nz), cnt = new Float64Array(nb * nz);
  for (let k = 0; k < nz; k++) for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) {
    const x = (i - ic) * dx, y = (j - jc) * dx, r = Math.hypot(x, y), b = Math.floor(r / dx);
    if (b >= nb || r === 0) continue;
    const q = m.idx(i, j, k);
    const ua = 0.5 * (m.u[q]! + m.u[q + 1]!), va = 0.5 * (m.v[q]! + m.v[q + m.sx]!);
    vt[k * nb + b] = vt[k * nb + b]! + (-ua * y + va * x) / r;
    ww[k * nb + b] = ww[k * nb + b]! + 0.5 * (m.w[q]! + m.w[q + m.plane]!);
    cnt[k * nb + b] = cnt[k * nb + b]! + 1;
  }
  // draw with sectionSvg: 'lat' axis reused as radius (km, mapped to 90..-90), 'sigma' as 1 - z/ztop
  const rad = Array.from({ length: nb }, (_, b) => 90 - 180 * (b + 0.5) / nb);
  const lev = Array.from({ length: nz }, (_, k) => 1 - m.zc[nz - 1 - k]! / (nz * dz));
  const flip = (a: Float64Array): number[] => { const o: number[] = []; for (let k = nz - 1; k >= 0; k--) for (let b = 0; b < nb; b++) o.push(cnt[k * nb + b]! ? a[k * nb + b]! / cnt[k * nb + b]! : 0); return o; };
  writeFileSync(`${outDir}/vt_rz_${tag}.svg`, sectionSvg({ title: `azimuthal-mean tangential wind (x: radius 0-${(nb * dx / 1000).toFixed(0)} km, y: height 0-${nz} km) ${tag}`, lat: rad, sigma: lev, values: flip(vt), units: 'm/s', diverging: true, contourStep: 5 }));
  writeFileSync(`${outDir}/w_rz_${tag}.svg`, sectionSvg({ title: `azimuthal-mean vertical velocity ${tag}`, lat: rad, sigma: lev, values: flip(ww), units: 'm/s', diverging: true, contourStep: 0.2 }));
}
