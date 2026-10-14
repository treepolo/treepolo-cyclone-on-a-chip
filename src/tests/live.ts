// Tests that do not run the model forward (no time steps): the wind interactions (what a forcing adds), and the settings changed while a run goes on
// (liveSettings.ts: what each does to the state and the configuration of a small model). Cheap: they run in a moment.
import { RegionalModel, type RegionalConfig } from '../regional/core.js';
import { weismanKlemp } from '../regional/kessler.js';
import { applyWind, FORCING_TAU } from '../regional/forcing.js';
import { RegionalPhysics } from '../regional/physics.js';
import { buildModel, envWind, soundingOf } from '../app/regional/build.js';
import { setupOf, type RegionalSetup } from '../app/regional/setup.js';
import { applyLive, changedKeys, liveMerge, type LiveHolder } from '../app/regional/liveSettings.js';
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

summary('live');
