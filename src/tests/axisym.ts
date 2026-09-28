// Axisymmetric (radius-height) regional model regressions.
import { AxisymModel, type AxisymConfig } from '../regional/axisym.js';
import { IceMicrophysics, QR } from '../regional/ice.js';
import { tropicalSounding } from '../regional/tropical.js';
import { weismanKlemp } from '../regional/kessler.js';
import { check, summary } from './assert.js';

const base: AxisymConfig = { nr: 150, nz: 25, dr: 4000, dz: 1000, dt: 20, nsound: 6, f: 5e-5, dampDepth: 5000, dampRate: 1 / 300, spongeWidth: 100000, spongeRate: 1 / 600,
  lh: 0, lv: 0, sst: 0, ck: 1.2e-3, vmin: 1, radTau: 0, radMax: 0 };
const dry = (z: number): { theta: number; qv: number } => ({ theta: 300 * Math.exp(1e-4 * z / 9.80665), qv: 0 });

// 1. a moist resting atmosphere stays exactly at rest
{
  const m = new AxisymModel(base, tropicalSounding(301.15));
  for (let s = 0; s < 100; s++) m.step();
  const e = Math.max(m.maxAbs(m.w, 26), m.maxAbs(m.u), m.maxAbs(m.pp), m.maxAbs(m.v));
  check('axisym: resting atmosphere stays at rest (100 steps)', e < 1e-12, e);
}

// 2. a dry vortex in gradient-wind and hydrostatic balance stays steady; angular momentum is conserved
{
  const m = new AxisymModel(base, dry, 1);
  m.insertVortex(15, 18000);
  const v0 = m.metrics().vmax, am0 = m.angularMomentum();
  let wmax = 0;
  for (let s = 0; s < 540; s++) { m.step(); wmax = Math.max(wmax, m.maxAbs(m.w, 26)); }
  const v1 = m.metrics().vmax, drift = Math.abs(m.angularMomentum() - am0) / am0;
  check('axisym: balanced vortex steady for 3 h (vmax within 1 %, |w| < 0.05 m/s)', Math.abs(v1 - v0) < 0.01 * v0 && wmax < 0.05, `${v0.toFixed(3)} -> ${v1.toFixed(3)} m/s, max |w| ${wmax.toExponential(2)}`);
  check('axisym: absolute angular momentum conserved (drift < 1e-4 in 3 h)', drift < 1e-4, drift);
}

// 3. turbulent mixing redistributes angular momentum without creating or destroying it
{
  const m = new AxisymModel({ ...base, lh: 1500, lv: 200 }, dry, 1);
  m.insertVortex(15, 18000);
  const am0 = m.angularMomentum();
  for (let s = 0; s < 180; s++) m.step();
  const drift = Math.abs(m.angularMomentum() - am0) / am0;
  check('axisym: mixing conserves angular momentum (drift < 1e-4 in 1 h)', drift < 1e-4, drift);
}

// 4. an axisymmetric warm bubble in a conditionally unstable sounding becomes a deep, raining updraft
{
  const m = new AxisymModel({ ...base, nr: 40, dr: 1000, nz: 40, dz: 500, dt: 5, f: 0, spongeWidth: 8000, spongeRate: 1 / 300, dampDepth: 4000 }, weismanKlemp, 6);
  const mp = new IceMicrophysics(m);
  for (let k = 0; k < 40; k++) for (let i = 0; i < 40; i++) {
    const r = Math.sqrt(((i + 0.5) * 1000 / 10000) ** 2 + ((m.zc[k]! - 1400) / 1400) ** 2);
    if (r < 1) m.th[m.idx(i, 0, k)] = m.th[m.idx(i, 0, k)]! + 2 * Math.cos(0.5 * Math.PI * r) ** 2;
  }
  let wmax = 0, qrmax = 0;
  for (let s = 0; s < 360; s++) {
    m.step(); mp.apply(5);
    wmax = Math.max(wmax, m.maxAbs(m.w, 41)); qrmax = Math.max(qrmax, m.maxAbs(m.scalars[QR]!));
  }
  let rain = 0; for (let i = 0; i < 40; i++) rain = Math.max(rain, mp.rainAcc[i]!);
  check('axisym: warm bubble grows into a deep updraft (> 15 m/s) with rain reaching the ground', wmax > 15 && qrmax > 1e-3 && rain > 0.1, `w ${wmax.toFixed(1)} m/s, qr ${(qrmax * 1e3).toFixed(2)} g/kg, rain ${rain.toFixed(2)} mm`);
  // the axis is a symmetry line: no flow through it
  let ua = 0; for (let k = 0; k < 40; k++) ua = Math.max(ua, Math.abs(m.u[m.idx(0, 0, k)]!));
  check('axisym: no radial flow through the axis', ua === 0, ua);
}

summary('axisym');
