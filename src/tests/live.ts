// Tests that do not run the model forward (no time steps): the wind interactions (what a forcing adds), and the settings changed while a run goes on
// (liveSettings.ts: what each does to the state and the configuration of a small model). Cheap: they run in a moment.
import { RegionalModel, type RegionalConfig } from '../regional/core.js';
import { weismanKlemp } from '../regional/kessler.js';
import { applyWind, FORCING_TAU } from '../regional/forcing.js';
import { RegionalPhysics } from '../regional/physics.js';
import { buildModel, envWind, soundingOf } from '../app/regional/build.js';
import { setupOf, type RegionalSetup } from '../app/regional/setup.js';
import { applyLive, changedKeys, liveMerge, type LiveHolder } from '../app/regional/liveSettings.js';
import { refineInto, makeSampler } from '../regional/refine.js';
import { nestTargets } from '../regional/twoway.js';
import { buildOuter, openInner, outerSpacing } from '../app/regional/extension.js';
import { check, summary } from './assert.js';

const base: RegionalConfig = { nx: 64, ny: 1, nz: 32, dx: 200, dy: 200, dz: 200, dt: 1, nsound: 6, f: 0, beta: 0.2, divDamp: 0.1, dampDepth: 0, dampRate: 0, kdiff2: 0 };

// Wind interactions (all added to the wind there is): a push once adds its speed at the centre; a lasting push adds its speed every tau
// (5 tau: 5 times the speed); a counter-clockwise rotation blows toward +y east of the centre; a cyclone also blows inward there and an anticyclone
// clockwise and outward; w stays 0 at the ground; a lasting push on top of a wind keeps that wind
{
  const mk = (): RegionalModel => new RegionalModel({ ...base, nx: 20, ny: 20, nz: 10, dx: 1000, dy: 1000, dz: 500, dt: 5 }, weismanKlemp, 3);
  const m1 = mk(), c = 10000, zc = m1.zc[2]!;
  const push = { x: c, y: c + 500, z: zc, R: 5000, H: 1500, speed: 10, dir: [1, 0, 0] as [number, number, number], form: 'push' as const, sign: 1 as const };
  applyWind(m1, [push], 'once');
  const u1 = m1.u[m1.idx(10, 10, 2)]!;
  const m2 = mk();
  for (let n = 0; n < 5 * FORCING_TAU / 5; n++) applyWind(m2, [push], 5);
  const u2 = m2.u[m2.idx(10, 10, 2)]!;
  const m3 = mk();
  applyWind(m3, [{ ...push, x: c, y: c, form: 'rotate', speed: 10, dir: [0, 0, 0] }], 'once');
  const vEast = m3.v[m3.idx(12, 10, 2)]!;
  const mc = mk(), ma = mk();
  applyWind(mc, [{ ...push, x: c, y: c, form: 'vortex', sign: 1, speed: 10, ratio: 0.5, dir: [0, 0, 0] }], 'once');
  applyWind(ma, [{ ...push, x: c, y: c, form: 'vortex', sign: -1, speed: 10, ratio: 0.5, dir: [0, 0, 0] }], 'once');
  const cycV = mc.v[mc.idx(12, 10, 2)]!, cycU = mc.u[mc.idx(12, 10, 2)]!, antV = ma.v[ma.idx(12, 10, 2)]!, antU = ma.u[ma.idx(12, 10, 2)]!;
  const m5 = mk(); m5.u.fill(3);
  for (let n = 0; n < 5 * FORCING_TAU / 5; n++) applyWind(m5, [push], 5);
  const u5 = m5.u[m5.idx(10, 10, 2)]!;
  const m4 = mk();
  applyWind(m4, [{ ...push, z: 0, dir: [0, 0, 1], H: 2000 }], 'once');
  let w0 = 0; for (let j = 0; j < 20; j++) for (let i = 0; i < 20; i++) w0 = Math.max(w0, Math.abs(m4.w[m4.idx(i, j, 0)]!));
  const wUp = m4.w[m4.idx(10, 10, 1)]!;
  check('wind: push once ~10 m/s at the centre', Math.abs(u1 - 10) < 0.2, u1);
  check('wind: a lasting push adds 5 x its speed in 5 tau, also on top of an existing wind (3 + 50)', Math.abs(u2 - 50) < 0.5 && Math.abs(u5 - 53) < 0.5, `${u2.toFixed(2)}, ${u5.toFixed(2)}`);
  check('wind: counter-clockwise rotation blows northward east of the centre', vEast > 1, vEast);
  check('wind: a cyclone blows counter-clockwise and inward, an anticyclone clockwise and outward (east of the centre)', cycV > 9 && cycU < -3 && antV < -9 && antU > 3,
    `cyclone v ${cycV.toFixed(1)} u ${cycU.toFixed(1)}, anticyclone v ${antV.toFixed(1)} u ${antU.toFixed(1)}`);
  check('wind: no w at the ground, w above', w0 === 0 && wUp > 1, `${w0}, ${wUp.toFixed(2)}`);
}

// Settings while running: each changes the state or the configuration of a small model as it says
{
  const small: RegionalSetup = { ...setupOf('tc'), preset: 'custom', L: 300000, dx: 15000, top: 15000, dz: 1000, dt: 0, init: 'vortex', initAmp: 15 };
  const mk = (s: RegionalSetup): { m: RegionalModel; h: LiveHolder; frame: { u: number; v: number } } => {
    const b = buildModel(s, true);
    return { m: b.model, h: { m: b.model, cfg: b.physics, phys: b.physics ? new RegionalPhysics(b.model, b.physics) : null }, frame: b.frame };
  };
  check('live settings: changedKeys sees a change, not the rounding of a form', changedKeys(small, { ...small, lat: 20.00001 }).length === 0 && changedKeys(small, { ...small, lat: 25 }).join() === 'lat');
  check('live settings: liveMerge keeps the grid and the starting disturbance', (() => { const s = liveMerge(small, { ...small, dx: 3000, init: 'none', lat: 30 }); return s.dx === 15000 && s.init === 'vortex' && s.lat === 30; })());

  // latitude: the Coriolis parameter
  { const t = mk(small); applyLive(t.h, { ...small, lat: 0 }, small, t.frame, false);
    check('live settings: latitude 0 gives no rotation', t.m.c.f === 0); }
  { const t = mk(small); applyLive(t.h, { ...small, lat: 30 }, small, t.frame, false);
    check('live settings: latitude 30 gives f = 2 Omega sin 30', Math.abs(t.m.c.f - 7.292e-5) < 1e-9, t.m.c.f); }

  // environmental wind: the difference of the environments is added at every level, everywhere (the storm's own wind stays)
  { const t = mk(small), m = t.m, next = { ...small, wind: 'trade' as const, windU: 8 };
    const cells = [m.idx(3, 4, 2), m.idx(10, 10, 0), m.idx(17, 2, 9)], ks = [2, 0, 9];
    const uBefore = cells.map((q) => m.u[q]!), vBefore = cells.map((q) => m.v[q]!), ubBefore = ks.map((k) => m.ub[k]!);
    applyLive(t.h, next, small, t.frame, false);
    const w = envWind(next).wind;
    const baseOk = ks.every((k) => Math.abs(m.ub[k]! - (w(m.zc[k]!).u - t.frame.u)) < 1e-9);
    const addOk = cells.every((q, n) => Math.abs((m.u[q]! - uBefore[n]!) - ((w(m.zc[ks[n]!]!).u - t.frame.u) - ubBefore[n]!)) < 1e-9 && Math.abs(m.v[q]! - vBefore[n]!) < 1e-9);
    check('live settings: the environmental wind: the base wind is the new one and the difference is added to the whole wind field', baseOk && addOk,
      `ub at 2.5 km ${m.ub[2]!.toFixed(2)}, added there ${(m.u[cells[0]!]! - uBefore[0]!).toFixed(2)}`);
    check('live settings: a wind on a rotating domain is geostrophic', m.c.geostrophic === true); }

  // sounding: the departures from the base state stay, the base state is the new sounding's
  { const t = mk(small), m = t.m, next = { ...small, sounding: 're87' as const }, ks = [1, 6, 12];
    const cells = ks.map((k) => m.idx(10, 10, k)), dth = cells.map((q, n) => m.th[q]! - m.th0[ks[n]!]!);
    applyLive(t.h, next, small, t.frame, false);
    const snd = soundingOf(next);
    const baseOk = ks.every((k) => Math.abs(m.th0[k]! - snd(m.zc[k]!).theta) < 1e-9 && Math.abs(m.qv0[k]! - snd(m.zc[k]!).qv) < 1e-12);
    const keptOk = cells.every((q, n) => Math.abs((m.th[q]! - m.th0[ks[n]!]!) - dth[n]!) < 1e-9);
    check('live settings: a new sounding replaces the base state and keeps the departures from it', baseOk && keptOk && m.scalars[0]!.every((x) => x >= 0)); }

  // boundaries: periodic <-> open
  { const t = mk(small), m = t.m;
    applyLive(t.h, { ...small, boundary: 'open' as const }, small, t.frame, false);
    const q = m.idx(5, 5, 3);
    check('live settings: open boundaries relax to the undisturbed environment', m.c.lateral === 'open' && !!m.boundary && Math.abs(m.boundary.th[q]! - m.th0[3]!) < 1e-12 && m.boundary.pp![q] === 0 && Math.abs(m.boundary.u[q]! - m.ub[3]!) < 1e-12);
    const open = { ...small, boundary: 'open' as const };
    applyLive(t.h, { ...open, boundary: 'periodic' as const }, open, t.frame, false);
    check('live settings: periodic boundaries again', m.c.lateral === 'periodic' && m.boundary === null); }

  // physics: made anew, the sea painted before stays
  { const t = mk(small), m = t.m;
    t.h.phys!.surface!.tsk[7] = 305; t.h.phys!.surface!.wet[7] = 0.3;
    const next = { ...small, vmin: 3, gust: false, cumulus: false, radRate: 2 };
    const r = applyLive(t.h, next, small, t.frame, false);
    check('live settings: physics settings are made anew (new values) and keep the painted surface',
      r.physics && t.h.cfg!.vmin === 3 && t.h.cfg!.gust === false && !t.h.cfg!.cumulus && Math.abs(t.h.cfg!.radConst! - 2 / 86400) < 1e-12 && t.h.phys!.surface!.tsk[7] === 305 && t.h.phys!.surface!.wet[7] === 0.3);
    const off = { ...next, fluxes: false, radiation: 'none' as const };
    applyLive(t.h, off, next, t.frame, false);
    check('live settings: no fluxes and no radiation: no physics, no hooks', t.h.cfg === null && t.h.phys === null && m.physicsTend === null && m.preStep === null);
    const on = { ...off, fluxes: true, radiation: 'const' as const };
    applyLive(t.h, on, off, t.frame, false);
    check('live settings: fluxes on again: a surface at the set sea temperature', !!t.h.phys && Math.abs(t.h.phys.surface!.tsk[0]! - (on.sst + 273.15)) < 1e-9 && m.physicsTend !== null); }

  // sea and land everywhere, the sea temperature of the sea
  { const t = mk(small), m = t.m, sf = (): { tsk: Float64Array; wet: Float64Array } => t.h.phys!.surface!;
    applyLive(t.h, { ...small, sst: 30 }, small, t.frame, false);
    check('live settings: the sea temperature of all sea', Math.abs(sf().tsk[0]! - 303.15) < 1e-9 && Math.abs(sf().tsk[100]! - 303.15) < 1e-9);
    applyLive(t.h, { ...small, sst: 30, surface: 'land' as const }, { ...small, sst: 30 }, t.frame, false);
    check('live settings: all land: ground at the air temperature, dry', Math.abs(sf().tsk[0]! - m.th0[0]! * m.pi0[0]!) < 1e-9 && sf().wet[0] === 0.3); }

  // an inner grid (the eye nest) takes the rotation, never the boundaries or the surface
  { const t = mk(small), r = applyLive(t.h, { ...small, lat: 10, boundary: 'open' as const, sst: 31 }, small, t.frame, true);
    check('live settings: an inner grid takes the rotation, not the boundaries or the surface', t.m.c.lateral !== 'open' && Math.abs(t.m.c.f - 2 * 7.292e-5 * Math.sin(10 * Math.PI / 180)) < 1e-12 && Math.abs(t.h.phys!.surface!.tsk[0]! - 301.15) < 1e-9 && r.physics); }
}

// Grid changes while running (the domain smaller, the model top higher or lower, another vertical spacing): the state is interpolated into the new grid (refineInto)
{
  const A: RegionalSetup = { ...setupOf('tc'), preset: 'custom', L: 300000, dx: 15000, top: 15000, dz: 1000, dt: 0, init: 'vortex', initAmp: 20 };
  const ma = buildModel(A, true).model;
  // a state: the vortex and a cloud in the middle levels, all of it smooth in height
  const n = A.L / A.dx;
  for (let k = 0; k < ma.c.nz; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const q = ma.idx(i, j, k), z = ma.zc[k]!;
    ma.th[q] = ma.th[q]! + 0.0004 * z + 0.3 * Math.sin(i * 0.5 + j * 0.3);
    ma.scalars[1]![q] = z > 3000 && z < 6000 ? 5e-4 : 0;
    ma.scalars[3]![q] = k === ma.c.nz - 1 ? 1e-4 : 0;            // ice in the topmost level
  }
  const maxDiff = (a: Float64Array, b: Float64Array, ka: (k: number) => number, kb: number, nI: number, nJ: number, dI = 0, dJ = 0, mA = ma, mB: RegionalModel): number => {
    let d = 0;
    for (let k = 0; k < kb; k++) for (let j = 0; j < nJ; j++) for (let i = 0; i < nI; i++) d = Math.max(d, Math.abs(b[mB.idx(i, j, k)]! - a[mA.idx(i + dI, j + dJ, ka(k))]!));
    return d;
  };
  const regrid = (t: RegionalSetup, x0 = 0, y0 = 0): RegionalModel => { const mf = buildModel(t, true).model; refineInto(ma, mf, x0, y0); return mf; };

  // the same grid: nothing changes
  { const mf = regrid(A);
    check('regrid: the same grid gives the same state (theta, u, v, qc to 1e-9)', maxDiff(ma.th, mf.th, (k) => k, ma.c.nz, n, n, 0, 0, ma, mf) < 1e-9 && maxDiff(ma.u, mf.u, (k) => k, ma.c.nz, n, n, 0, 0, ma, mf) < 1e-9 && maxDiff(ma.scalars[1]!, mf.scalars[1]!, (k) => k, ma.c.nz, n, n, 0, 0, ma, mf) < 1e-12); }
  // a higher top: the lower levels are the same, the new levels carry the departure of the old top, no cloud or rain above the old top, w = 0 at the new lid
  { const mf = regrid({ ...A, top: 20000 });
    const lower = maxDiff(ma.th, mf.th, (k) => k, ma.c.nz, n, n, 0, 0, ma, mf) < 1e-9;
    const depTop = ma.th[ma.idx(4, 4, ma.c.nz - 1)]! - ma.th0[ma.c.nz - 1]!, depNew = mf.th[mf.idx(4, 4, 18)]! - mf.th0[18]!;
    let cloudAbove = 0; for (let k = ma.c.nz; k < mf.c.nz; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) cloudAbove += mf.scalars[1]![mf.idx(i, j, k)]! + mf.scalars[3]![mf.idx(i, j, k)]!;
    let wLid = 0; for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) wLid = Math.max(wLid, Math.abs(mf.w[mf.idx(i, j, mf.c.nz)]!));
    // (the new levels all carry the departure at the old lid, a half level beyond the topmost centre: the same in every new level)
    const sameAbove = [15, 16, 17, 18, 19].every((k) => Math.abs(mf.th[mf.idx(4, 4, k)]! - mf.th0[k]! - depNew) < 1e-9);
    check('regrid: a higher top keeps the lower levels, the new ones carry the departure of the old lid (the same in all of them), start clear, no w at the lid',
      mf.c.nz === 20 && lower && sameAbove && Math.abs(depNew - depTop) < 0.3 && cloudAbove === 0 && wLid === 0,
      `departure at the old top level ${depTop.toFixed(3)}, in the new levels ${depNew.toFixed(3)}`); }
  // a lower top: the levels that stay are the same
  { const mf = regrid({ ...A, top: 10000 });
    check('regrid: a lower top keeps the levels that stay', mf.c.nz === 10 && maxDiff(ma.th, mf.th, (k) => k, 10, n, n, 0, 0, ma, mf) < 1e-9 && maxDiff(ma.v, mf.v, (k) => k, 10, n, n, 0, 0, ma, mf) < 1e-9); }
  // another vertical spacing: a departure that is linear in height stays exact (a model without the vortex, so that the field is smooth in height)
  { const mb = buildModel({ ...A, init: 'none' }, true).model;
    for (let k = 0; k < mb.c.nz; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) mb.th[mb.idx(i, j, k)] = mb.th0[k]! + 0.0004 * mb.zc[k]! + 0.3 * Math.sin(i * 0.5 + j * 0.3);
    const mf = buildModel({ ...A, init: 'none', dz: 500 }, true).model; refineInto(mb, mf, 0, 0);
    let ok = mf.c.nz === 30, worst = 0;
    for (let k = 1; k < 13; k++) { const z = mf.zc[k]!, dep = mf.th[mf.idx(6, 7, k)]! - mf.th0[k]!, expect = 0.0004 * z + 0.3 * Math.sin(6 * 0.5 + 7 * 0.3); worst = Math.max(worst, Math.abs(dep - expect)); }
    if (worst > 2e-3) ok = false;
    check('regrid: another vertical spacing interpolates a departure that is linear in height (to 2e-3 K)', ok, worst); }
  // a smaller domain: the cells of the box are the old cells (same spacing)
  { const mf = regrid({ ...A, L: 150000 }, 75000, 45000);
    check('regrid: a smaller domain takes the cells of the box (theta, u to 1e-9)', mf.c.nx === 10 && maxDiff(ma.th, mf.th, (k) => k, ma.c.nz, 10, 10, 5, 3, ma, mf) < 1e-9 && maxDiff(ma.u, mf.u, (k) => k, ma.c.nz, 10, 10, 5, 3, ma, mf) < 1e-9); }
}

// An outer grid around the running model (extension.ts): the spacing, the outer state, and the relaxation targets it sets for the inner grid (no time steps)
{
  const S: RegionalSetup = { ...setupOf('tc'), preset: 'custom', L: 150000, dx: 5000, top: 15000, dz: 1000, dt: 0, init: 'vortex', initAmp: 20 };
  check('outer grid: the spacing is a whole multiple that divides the inner domain (30 cells: 15 km from 15 km; 30 km from 40 km)',
    outerSpacing(30, 5000, 15000) === 15000 && outerSpacing(30, 5000, 40000) === 30000 && typeof outerSpacing(31, 5000, 15000) === 'string');
  const b = buildModel(S, true), inner = b.model, n = 30;
  inner.ub[3] = inner.ub[3]! + 2;                                         // a change of the environment by an interaction: the outer grid carries it
  const acc = { rain: new Float64Array(n * n), snow: new Float64Array(n * n) };
  const out = buildOuter(inner, S, b.frame, 300000, 15000, true, acc, null);
  if (typeof out === 'string') { check('outer grid: it can be built', false, out); }
  else {
    const o = out.model, g = out.geom, k = 3;
    check('outer grid: 20 x 20 columns of 15 km, the inner box of 10 x 10 of them in the middle, the same levels', o.c.nx === 20 && o.c.dx === 15000 && o.c.nz === inner.c.nz && g.np === 10 && g.i0 === 5 && g.r === 3 && g.x0 === 75000);
    // inside the box the outer cell is the mean of its 3 x 3 inner cells; outside it is the plain environment
    let mean = 0; for (let dj = 0; dj < 3; dj++) for (let di = 0; di < 3; di++) mean += inner.th[inner.idx(15 + di, 15 + dj, k)]! / 9;
    check('outer grid: inside the box the outer cell is the mean of its inner cells (theta to 1e-9)', Math.abs(o.th[o.idx(10, 10, k)]! - mean) < 1e-9, `${o.th[o.idx(10, 10, k)]!.toFixed(4)} vs ${mean.toFixed(4)}`);
    check('outer grid: outside the box it is the plain environment, with the inner grid\'s base wind', Math.abs(o.th[o.idx(2, 2, k)]! - o.th0[k]!) < 1e-9 && Math.abs(o.ub[3]! - inner.ub[3]!) < 1e-12 && Math.abs(o.u[o.idx(2, 2, 3)]! - inner.ub[3]!) < 1e-9);
    // the targets: only in the edge ring and the sponge layer, the outer state interpolated to the inner edge
    openInner(inner);
    nestTargets(o, null, inner, g, 1, inner.boundary!);
    const sample = makeSampler(o), z = (k + 0.5) * inner.c.dz;
    const want = (i: number, j: number, x0 = g.x0!, y0 = g.y0!): number => inner.th0[k]! + sample(o.th, x0 + (i + 0.5) * inner.c.dx, y0 + (j + 0.5) * inner.c.dy, z, 0.5, 0.5, false, o.th0);
    check('outer grid: the edge cells of the inner grid relax to the outer state interpolated there', Math.abs(inner.boundary!.th[inner.idx(0, 7, k)]! - want(0, 7)) < 1e-9 && Math.abs(inner.boundary!.th[inner.idx(12, 29, k)]! - want(12, 29)) < 1e-9);
    check('outer grid: no targets in the middle (below the sponge layer), targets in the sponge layer', inner.boundary!.th[inner.idx(15, 15, k)] === 0 && inner.boundary!.th[inner.idx(15, 15, inner.c.nz - 1)]! !== 0);
    // the box moves with a domain that follows the storm
    g.x0 = g.x0! - 10000;
    nestTargets(o, null, inner, g, 1, inner.boundary!);
    check('outer grid: the targets follow the inner box when it moves over the outer grid', Math.abs(inner.boundary!.th[inner.idx(0, 7, k)]! - want(0, 7, g.x0!, g.y0!)) < 1e-9);
  }
}

summary('live');
