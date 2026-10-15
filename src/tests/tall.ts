// Tests without time steps: the tall grid (src/regional/tall.ts, src/app/regional/tallNest.ts): its geometry, the state it starts with, the relaxation targets and the feedback.
import type { RegionalModel } from '../regional/core.js';
import { buildModel } from '../app/regional/build.js';
import { setupOf, type RegionalSetup } from '../app/regional/setup.js';
import { TallNest } from '../app/regional/tallNest.js';
import { tallGeometry, tallEmptyTargets, tallTargets, tallFeedback, tallWeight, tallMaxCols, eyeInTall, type TallGeom } from '../regional/tall.js';
import { nestGeometry, parentState } from '../regional/twoway.js';
import { check, summary } from './assert.js';

const S: RegionalSetup = { ...setupOf('tc'), preset: 'custom', L: 150000, dx: 5000, top: 15000, dz: 1000, dt: 0, init: 'none', initAmp: 0, boundary: 'open' };
const acc = { rain: new Float64Array(30 * 30), snow: new Float64Array(30 * 30) };
/** a host with a smooth departure from its environment, linear in x, y, z (each variable at its own staggered position) */
function host(): { m: RegionalModel; frame: { u: number; v: number } } {
  const b = buildModel(S, false), m = b.model, { nx, ny, nz, dx, dy } = m.c;
  const dep = (x: number, y: number, z: number): number => 1e-3 * (x + 0.5 * y) / 1000 + 2e-4 * z / 1000;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const q = m.idx(i, j, k), z = m.zc[k]!;
    m.th[q] = m.th0[k]! + dep((i + 0.5) * dx, (j + 0.5) * dy, z);
    m.u[q] = m.ub[k]! + 3 * dep(i * dx, (j + 0.5) * dy, z); m.v[q] = m.vb[k]! - 2 * dep((i + 0.5) * dx, j * dy, z);
    m.pp[q] = 5 * dep((i + 0.5) * dx, (j + 0.5) * dy, z);
  }
  return { m, frame: b.frame };
}
const spec = { cx: 75000, cy: 75000, width: 60000, top: 24000, r: 2, rz: 1 };
const { m: H, frame } = host();
const hc = H.c;

// geometry
{
  const g = tallGeometry(hc, spec) as TallGeom;
  check('tall: a 60 km footprint at the centre, ratios 2 and 1, top 24 km: 12 host cells, 24 child cells of 2.5 km, 24 levels', typeof g !== 'string' && g.np === 12 && g.i0 === 9 && g.nx === 24 && g.dx === 2500 && g.nz === 24 && g.Z1 === 15000 && g.Z2 === 24000 && g.x0 === 45000, typeof g === 'string' ? g : '');
  check('tall: a top that is not above the model\'s is refused', typeof tallGeometry(hc, { ...spec, top: 15000 }) === 'string');
  check('tall: a footprint that is too small is refused', typeof tallGeometry(hc, { ...spec, width: 20000 }) === 'string');
  const max = tallMaxCols(hc), gm = tallGeometry(hc, { ...spec, width: max * hc.dx });
  check('tall: the widest footprint is the domain less the edge margin (a whole-domain extension is the model top\'s)', typeof gm !== 'string' && gm.np === max && gm.i0 === (hc.nx - max) / 2 && typeof tallGeometry(hc, { ...spec, width: (max + 1) * hc.dx }) === 'string', `${max}`);
  const edge = tallGeometry(hc, { ...spec, cx: 1000, cy: 149000 }) as TallGeom;
  check('tall: a footprint centred near the edge is moved inside the margin', edge.i0 === 2 + 6 && edge.j0 === hc.ny - 8 - edge.np, `${edge.i0}, ${edge.j0}`);
  check('tall: ratios must be whole numbers in range; too many cells are refused', typeof tallGeometry(hc, { ...spec, r: 0 }) === 'string' && typeof tallGeometry(hc, { ...spec, r: 40 }) === 'string' && typeof tallGeometry(hc, { ...spec, r: 32, rz: 8, top: 40000 }) === 'string');
  const w = (ip: number, jp: number): number => tallWeight(g, ip, jp, hc.dx, hc.dy);
  check('tall: the feedback weight is 0 outside, small at the footprint\'s edge cell, 1 deep inside', w(g.i0 - 1, 15) === 0 && w(g.i0, 15) > 0 && w(g.i0, 15) < 0.5 && w(15, 15) === 1 && w(g.i0 + g.np - 1, 15) === w(g.i0, 15), `${w(g.i0, 15).toFixed(3)}`);
}

// the state it starts with
const T = TallNest.build(H, acc, S, frame, null, spec, false);
if (typeof T === 'string') check('tall: the grid can be built', false, T);
else {
  const g = T.g, c = T.m, cc = c.c, nzH = hc.nz;
  const dep = (x: number, y: number, z: number): number => 1e-3 * (x + 0.5 * y) / 1000 + 2e-4 * z / 1000;
  check('tall: the grid has the footprint\'s cells, the levels to the new top and a lid with a Rayleigh layer of 6 km at the new top', cc.nx === 24 && cc.nz === 24 && cc.dampDepth === 6000 && Math.abs(cc.nz * cc.dz - 24000) < 1e-9 && cc.lateral === 'open');
  let worst = 0;
  for (const [i, j, k] of [[3, 4, 2], [12, 12, 8], [20, 7, 13]] as const) {
    const q = c.idx(i, j, k), x = g.x0 + (i + 0.5) * cc.dx, y = g.y0 + (j + 0.5) * cc.dy;
    worst = Math.max(worst, Math.abs(c.th[q]! - (c.th0[k]! + dep(x, y, c.zc[k]!))), Math.abs(c.pp[q]! - 5 * dep(x, y, c.zc[k]!)));
  }
  check('tall: below the model\'s top the state is the model\'s (theta and pi\' of a smooth field to 1e-9)', worst < 1e-9, worst);
  let water = 0; for (let k = nzH; k < cc.nz; k++) for (let j = 0; j < cc.ny; j++) for (let i = 0; i < cc.nx; i++) for (let s = 1; s < c.scalars.length; s++) water += Math.abs(c.scalars[s]![c.idx(i, j, k)]!);
  check('tall: above the model\'s top there is no cloud, rain or ice', water === 0);
  let base = 0; for (let k = 0; k < nzH; k++) base = Math.max(base, Math.abs(c.th0[k]! - H.th0[k]!), Math.abs(c.qv0[k]! - H.qv0[k]!), Math.abs(c.ub[k]! - H.ub[k]!));
  check('tall: the base state is the model\'s on the levels they share (the same sounding to the new top)', base < 1e-9, base);
  check('tall: w is 0 at the new lid and at the ground', c.w[c.idx(10, 10, cc.nz)] === 0 && c.w[c.idx(10, 10, 0)] === 0);

  // targets: the edge ring and the Rayleigh layer only; the host's fields below its top, the environment above
  const out = c.boundary!;
  const kM = 5, kA = 18, kS = 22, e = c.idx(0, 12, kM), mid = c.idx(12, 12, kM);
  check('tall: targets in the edge ring, none in the middle below the Rayleigh layer, some in it', out.th[e]! !== 0 && out.th[mid] === 0 && out.th[c.idx(12, 12, kS)]! !== 0);
  const wantTh = c.th0[kM]! + dep(g.x0 + 0.5 * cc.dx, g.y0 + 12.5 * cc.dy, c.zc[kM]!);
  check('tall: the edge target below the model\'s top is the host interpolated there (theta to 1e-9)', Math.abs(out.th[e]! - wantTh) < 1e-9, Math.abs(out.th[e]! - wantTh));
  const ea = c.idx(0, 12, kA);
  check('tall: above the model\'s top the target is the grid\'s own environment (theta, u, v, qv, pi\')', out.th[ea] === c.th0[kA] && out.u[ea] === c.ub[kA] && out.v[ea] === c.vb[kA] && out.qv![ea] === c.qv0[kA] && out.pp![ea] === 0);
  const eLo = c.idx(0, 12, 14), eHi = c.idx(0, 12, 15);
  check('tall: the host\'s departure counts up to half a host level below its top and not at all from half a level above it (the first level above the top: the environment)', Math.abs(out.th[eLo]! - (c.th0[14]! + dep(g.x0 + 0.5 * cc.dx, g.y0 + 12.5 * cc.dy, c.zc[14]!))) < 1e-9 && out.th[eHi] === c.th0[15]);
  // time interpolation: between the host at the start and the end of its step
  const old = parentState(H); H.th.forEach((_, q) => { H.th[q] = H.th[q]! + 1; });
  const o2 = tallEmptyTargets(c); tallTargets(H, old, c, g, 0.25, o2);
  check('tall: the targets are interpolated in time between the host\'s states (alpha 0.25: a quarter of the change)', Math.abs(o2.th[e]! - (wantTh + 0.25)) < 1e-9, o2.th[e]! - wantTh);
  H.th.forEach((_, q) => { H.th[q] = H.th[q]! - 1; });

  // feedback: the grid made from the host gives the host back; a change in the grid goes into the host over the footprint (below its top) only
  const before = Float64Array.from(H.th), bu = Float64Array.from(H.u);
  tallFeedback(c, H, g);
  let d = 0; for (let q = 0; q < before.length; q++) d = Math.max(d, Math.abs(H.th[q]! - before[q]!), Math.abs(H.u[q]! - bu[q]!));
  check('tall: the feedback of the grid made from a smooth host gives the host back (theta, u to 1e-9)', d < 1e-9, d);
  for (let q = 0; q < c.size; q++) c.th[q] = c.th[q]! + 1;
  tallFeedback(c, H, g);
  const inside = H.idx(15, 15, 5), edgeCell = H.idx(g.i0, 15, 5), outside = H.idx(g.i0 - 1, 15, 5), top = H.idx(15, 15, nzH - 1);
  check('tall: a change of 1 K in the grid reaches the host fully inside the footprint, partly at its edge, not outside it', Math.abs(H.th[inside]! - before[inside]! - 1) < 1e-9 && H.th[edgeCell]! - before[edgeCell]! > 0 && H.th[edgeCell]! - before[edgeCell]! < 0.5 && H.th[outside] === before[outside], `${(H.th[edgeCell]! - before[edgeCell]!).toFixed(3)}`);
  check('tall: the host\'s top level takes the grid\'s too (the levels the host has)', Math.abs(H.th[top]! - before[top]! - 1) < 1e-9);
  // the top face's w of the host is not touched
  check('tall: the host\'s lid keeps w = 0', H.w[H.idx(15, 15, nzH)] === 0);
  H.th.set(before);

  // the roll: the columns that come in are the host's
  const acc2 = { rain: new Float64Array(30 * 30), snow: new Float64Array(30 * 30) };
  T.roll(H, acc2, 1, 0, null);
  const qIn = c.idx(0, 12, 6), xIn = g.x0 + 0.5 * cc.dx, yIn = g.y0 + 12.5 * cc.dy;
  check('tall: after a roll by a host cell the columns that come in are the host\'s state', Math.abs(c.th[qIn]! - (c.th0[6]! + dep(xIn, yIn, c.zc[6]!))) < 1e-9, Math.abs(c.th[qIn]! - (c.th0[6]! + dep(xIn, yIn, c.zc[6]!))));
  T.fit(10, null);
  check('tall: the child steps per host step follow the child\'s own step', T.nsub === Math.max(1, Math.ceil(10 / T.dt0 - 1e-6)) && Math.abs(c.c.dt * T.nsub - 10) < 1e-9, `${T.nsub}`);

  // an eye nest inside the footprint feeds the grid too when its ratios are whole multiples of the grid's and its box is clear of the grid's edge
  const g1: TallGeom = { r: 1, rz: 1, i0: 2, j0: 2, np: 26, nx: 26, nz: 24, dx: 5000, dz: 1000, x0: 10000, y0: 10000, Z1: 15000, Z2: 24000, Wf: 10000, ring: 4, nzHost: 15 };
  const eg = nestGeometry({ ...hc, lateral: 'open', relaxCells: 6 }, 10000, 1000, 500);
  if (typeof eg === 'string') check('tall: an eye nest geometry for the test', false, eg);
  else {
    const ei = eyeInTall(g1, eg);
    check('tall: an eye nest in the middle of a footprint with ratio 1 feeds it too (its box in the grid\'s cells, the levels below the host\'s top)', typeof ei !== 'string' && ei.geom.r === eg.r && ei.geom.i0 === eg.i0 - g1.i0 && ei.geom.np === eg.np && ei.nk === hc.nz, typeof ei === 'string' ? ei : '');
    check('tall: not when the eye nest\'s ratio is not a multiple of the grid\'s', typeof eyeInTall({ ...g1, r: 2, ring: 4 }, { ...eg, r: 5 }) === 'string');
    check('tall: an eye nest at the footprint\'s edge is refused', typeof eyeInTall({ ...g1, i0: eg.i0 }, eg) === 'string');
  }
}
summary('tall');
