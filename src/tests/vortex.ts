// The vortex tools of the 3-D view (src/regional/vortex.ts).
import { RegionalModel } from '../regional/core.js';
import { tropicalSounding } from '../regional/tropical.js';
import { check, summary } from './assert.js';

// The vortex tools (vortex.ts): the wind peaks at the strength asked, at about the radius asked, and is zero beyond the outer radius; a wind that decreases upwards is
// warm in the core and a low at the ground, one that increases upwards is cold below its strongest level; the direction flips the wind; the vortex is balanced
// (it holds for 20 minutes), and it wraps round the edge of a periodic domain
{
  const { vortexPatch, applyVortexPatch } = await import('../regional/vortex.js');
  const mk = (): RegionalModel => new RegionalModel({ nx: 48, ny: 48, nz: 30, dx: 5000, dy: 5000, dz: 600, dt: 20, nsound: 6, f: 5e-5, beta: 0.2, divDamp: 0.1, dampDepth: 0, dampRate: 0, kdiff2: 0, lateral: 'periodic' }, tropicalSounding(301.15), 6);
  const wind = (m: RegionalModel, cx: number, cy: number, k: number): { vmax: number; rmax: number; beyond: number } => {
    let vmax = 0, rmax = 0, beyond = 0;
    for (let j = 0; j < 48; j++) for (let i = 0; i < 48; i++) {
      const q = m.idx(i, j, k), x = (i + 0.5) * 5000 - cx, y = (j + 0.5) * 5000 - cy, r = Math.hypot(x, y);
      const u = 0.5 * (m.u[q]! + m.u[q + 1]!), v = 0.5 * (m.v[q]! + m.v[q + m.sx]!), vt = r > 0 ? (-u * y + v * x) / r : 0;
      if (Math.abs(vt) > vmax) { vmax = Math.abs(vt); rmax = r; }
      if (r > 100000 + 7500 && Math.hypot(u, v) > 1e-9) beyond++;
    }
    return { vmax, rmax, beyond };
  };
  // warm-core: strongest wind at the ground
  const w = mk(), spec = { x: 120000, y: 120000, z: 0, R: 100000, rm: 20000, H: 9000, vmax: 40, dir: 1 as const };
  const pw = vortexPatch(w, spec); applyVortexPatch(w, pw);
  const kc = 4, iw = w.idx(24, 24, kc), ic = w.idx(24, 24, 0);
  const ww = wind(w, 120000, 120000, 0);
  const warmOk = Math.abs(ww.vmax - 40) < 5 && Math.abs(ww.rmax - 20000) < 10000 && ww.beyond === 0 && pw.dpCentre < -1 && w.th[iw]! - w.th0[kc]! > 0.2 && w.pp[ic]! < 0;
  // the clockwise one flips the wind; east of the centre the northward wind is negative
  const cw = mk(); applyVortexPatch(cw, vortexPatch(cw, { ...spec, dir: -1 }));
  const east = cw.idx(30, 24, 0), eastW = w.idx(30, 24, 0);
  // cold-core: strongest wind at 10 km, cold below that height at the centre
  const c = mk(), cs = { x: 120000, y: 120000, z: 10000, R: 100000, rm: 25000, H: 5000, vmax: 25, dir: 1 as const };
  const pc = vortexPatch(c, cs); applyVortexPatch(c, pc);
  const k7 = Math.round(7000 / 600 - 0.5), kTop = Math.round(10000 / 600 - 0.5), cc7 = c.idx(24, 24, k7);
  const wc = wind(c, 120000, 120000, kTop);
  const coldOk = Math.abs(wc.vmax - 25) < 4 && c.th[cc7]! - c.th0[k7]! < -0.2 && Math.abs(c.u[c.idx(24, 24, 0)]!) < 1e-9 && pc.dpCentre > -1;
  // balanced: 20 minutes of the model keep the warm vortex
  const b = mk(); applyVortexPatch(b, vortexPatch(b, spec));
  for (let n = 0; n < 60; n++) b.step();
  const wb = wind(b, 120000, 120000, 0), finiteB = [b.u, b.v, b.w, b.th, b.pp].every((a) => a.every(Number.isFinite));
  // round the edge of the periodic domain: a vortex at x = 10 km wraps to the other side
  const e = mk(); applyVortexPatch(e, vortexPatch(e, { ...spec, x: 10000 }));
  const we = wind(e, 10000, 120000, 0);
  let wrapped = 0; for (let j = 0; j < 48; j++) for (let k = 0; k < 1; k++) wrapped += Math.abs(e.v[e.idx(46, j, k)]!);
  check('vortex tools: the warm-core vortex has the strongest wind asked at about the radius asked, none beyond the outer radius, a low and a warm core; clockwise flips the wind',
    warmOk && Math.sign(cw.v[east]!) === -Math.sign(w.v[eastW]!) && cw.v[east]! !== 0, `wind ${ww.vmax.toFixed(1)} m/s at ${(ww.rmax / 1000).toFixed(0)} km, beyond ${ww.beyond}, pressure ${pw.dpCentre.toFixed(1)} hPa, theta' ${(w.th[iw]! - w.th0[kc]!).toFixed(2)} K`);
  check('vortex tools: the cold-core vortex has its strongest wind aloft, none at the ground, and is cold below it',
    coldOk, `wind ${wc.vmax.toFixed(1)} m/s at 10 km, theta' at 7 km ${(c.th[cc7]! - c.th0[k7]!).toFixed(2)} K, ground wind ${Math.abs(c.u[c.idx(24, 24, 0)]!).toExponential(1)}`);
  check('vortex tools: the vortex is balanced (20 minutes of the model keep more than 70 % of its wind) and wraps round the edge of a periodic domain',
    finiteB && wb.vmax > 0.7 * 40 && wb.vmax < 1.3 * 40 && we.vmax > 35 && wrapped > 1, `after 20 min ${wb.vmax.toFixed(1)} m/s; at the edge ${we.vmax.toFixed(1)} m/s, wrapped wind ${wrapped.toFixed(1)}`);
}

summary('vortex');
