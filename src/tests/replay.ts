// Replay data: the 16-bit packing, charts from a recorded state, the replay file, the store's policies.
import { check, summary } from './assert.js';
import { quant, dequant, metaOf, modelFromMeta, loadState, packState, type ReplayRec } from '../regional/replayData.js';
import { IceMicrophysics } from '../regional/ice.js';
import { RegionalPhysics } from '../regional/physics.js';
import { columnDiagnostics, compositeMaps, sliceFields, modelPlanes, azimuthalMeans, columnProfiles, sectionFromColumns, MAP_VARS, SLICE_VARS } from '../regional/diagnostics.js';
import { buildModel } from '../app/regional/build.js';
import { setupOf } from '../app/regional/setup.js';
import { ReplayStore, size, type ReplayFrame } from '../app/regional/replay.js';
import { chooseFrames, packReplay, unpackReplay } from '../app/regional/replayFile.js';
import type { RegionalFrame } from '../app/regional/protocol.js';

// 1. packing: the error is at most half a step; NaN stays NaN; a constant array is exact
{
  const a = Float64Array.from({ length: 5000 }, (_, i) => (i === 17 ? NaN : -5 + 12 * Math.abs(Math.sin(i * 12.9898) * 43758.5453 % 1)));
  const q = quant(a), b = dequant<Float64Array>(q, new Float64Array(a.length));
  let err = 0, nanOk = Number.isNaN(b[17]!);
  for (let i = 0; i < a.length; i++) if (i !== 17) err = Math.max(err, Math.abs(a[i]! - b[i]!));
  const step = (q.hi - q.lo) / 65534;
  const flat = dequant(quant(new Float32Array(10).fill(3)));
  check('replay: 16-bit packing: error <= half a step, NaN kept, constant exact', err <= step / 2 + 1e-12 && nanOk && flat.every((v) => v === 3), `error ${err.toExponential(2)} (step ${step.toExponential(2)})`);
}

// 2. a recorded state gives the charts of the model it came from (to the packing's precision)
let recFull: ReplayRec | null = null, metaTc: ReturnType<typeof metaOf> | null = null;
{
  const b = buildModel(setupOf('tc'), false), m = b.model, mp = new IceMicrophysics(m);
  if (b.physics) new RegionalPhysics(m, b.physics);
  for (let n = 0; n < 6; n++) { m.step(); mp.apply(m.c.dt); }
  const meta = metaOf(m, { experiment: 'tc', tc: true, sea: true, land: null });
  metaTc = meta;
  const m2 = modelFromMeta(meta);
  const state = packState(m);
  loadState(m2, state);
  const vel = { u: 0, v: 0 }, zero = new Float32Array(m.c.nx * m.c.ny);
  const acc = { rain: mp.rainAcc, snow: mp.snowAcc, rate: zero, cu: null, uhMax: null, windMax: null, sst: null };
  const c1 = columnDiagnostics(m, true), c2 = columnDiagnostics(m2, true);
  const maps1 = compositeMaps(m, c1, modelPlanes(m, 0), acc, MAP_VARS, vel), maps2 = compositeMaps(m2, c2, modelPlanes(m2, 0), acc, MAP_VARS, vel);
  const worst = (x: ArrayLike<number>, y: ArrayLike<number>): number => {
    let lo = Infinity, hi = -Infinity, d = 0;
    for (let i = 0; i < x.length; i++) { const a = x[i]!, c = y[i]!; if (Number.isNaN(a) || Number.isNaN(c)) { if (Number.isNaN(a) !== Number.isNaN(c)) d = Math.max(d, 1); continue; } lo = Math.min(lo, a); hi = Math.max(hi, a); d = Math.max(d, Math.abs(a - c)); }
    return d / Math.max(hi - lo, 1e-9);
  };
  // (maps that jump with a threshold - the level of the visible cloud top, of the cloud top, of the condensation level - move a level
  // where the packing tips a value over it: they are judged by the whole map's mean level instead)
  const STEP = new Set(['visZ', 'ctopZ', 'ctopT', 'etop', 'lcl']);
  let wm = 0, wn = '', all = '';
  for (const v of MAP_VARS) {
    const e = worst(maps1[v]!, maps2[v]!);
    if (e > 0.01) all += ` ${v} ${(100 * e).toFixed(1)}%`;
    if (!STEP.has(v) && e > wm) { wm = e; wn = v; }
  }
  check('replay: the maps of a recorded state agree with those of the model (< 3 % of each map\'s range; the level-threshold maps aside)', wm < 0.03, `worst ${wn} ${(100 * wm).toFixed(2)} %; over 1 %:${all || ' none'}`);
  let ws = 0, sn = '';
  const k = 4, pl1 = modelPlanes(m, k), pl2 = modelPlanes(m2, k), s1 = sliceFields(m, pl1, k, SLICE_VARS, vel), s2 = sliceFields(m2, pl2, k, SLICE_VARS, vel);
  for (const v of SLICE_VARS) { const e = worst(s1[v]!, s2[v]!); if (e > ws) { ws = e; sn = v; } }
  check('replay: the horizontal slices of a recorded state agree with those of the model (< 3 %)', ws < 0.03, `worst ${sn} ${(100 * ws).toFixed(2)} %`);
  const pts = Array.from({ length: 20 }, (_, i) => ({ i: 10 + i, j: 40 })), nf = 5 + m.scalars.length;
  const e1 = sectionFromColumns(m, columnProfiles(m, pts), pts.length, nf, vel), e2 = sectionFromColumns(m2, columnProfiles(m2, pts), pts.length, nf, vel);
  const r1 = azimuthalMeans(m, 40 * m.c.dx, 40 * m.c.dy, m.c.dx, 20), r2 = azimuthalMeans(m2, 40 * m.c.dx, 40 * m.c.dy, m.c.dx, 20);
  check('replay: sections and radius-height means of a recorded state agree with those of the model (< 3 %)', worst(e1, e2) < 0.03 && worst(r1, r2) < 0.03, `section ${(100 * worst(e1, e2)).toFixed(2)} %, rz ${(100 * worst(r1, r2)).toFixed(2)} %`);
  const packedMaps: Record<string, ReturnType<typeof quant>> = {};
  for (const v of MAP_VARS) packedMaps[v] = quant(maps1[v]!);
  recFull = { tier: 'full', metaId: 1, meta, aux: { rain: Float32Array.from(mp.rainAcc), snow: Float32Array.from(mp.snowAcc), rate: null, cu: null, uh: null, wind: null, sst: Float32Array.from({ length: zero.length }, () => 27.5), vel }, maps: packedMaps, state, centre: { x: 600000, y: 600000 } };
}

// 3. the replay file: frames come back as they went in; segments, every n-th frame and the content tier choose what goes in
{
  const mkFrame = (t: number, rec: ReplayRec | null, nest = false): ReplayFrame => {
    const nx = 4, ny = 3, nz = 2;
    return { t, nx, ny, nz, dx: 15000, dz: 500, Lx: nx * 15000, Ly: ny * 15000, top: nz * 500, cloud: Uint8Array.from({ length: nx * ny * nz }, (_, i) => (i * 7 + t) & 255), rain: Uint8Array.from({ length: nx * ny * nz }, (_, i) => (i * 3) & 255),
      ground: Uint8Array.from({ length: nx * ny * 4 }, (_, i) => i & 255), storms: [], origin: { x: 1000 * t, y: 5 },
      nest: nest ? { R: 1, dx: 1, dz: 1, r: 1, rz: 1, nx: 2, nz: 2, cells: 8, nsub: 1, x0: 0, y0: 0, L: 1, cx: 0, cy: 0, Wf: 1, cloud: Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8]), rain: Uint8Array.from([8, 7, 6, 5, 4, 3, 2, 1]) } : null,
      stats: { wmax: 1, wmin: -1, qcmax: 0, qrmax: 0, rainmax: 2, vmax: 30 + t, dp: 5, rmw: 40000, eyewalls: null, zetaMax: 0, vGround: 30, dbzMax: 40, uhMax: 0, uhMin: 0, capeMax: 1000, storm: { x: 5, y: 6 }, storms: [], mainId: null, vtProfile: { dr: 15000, vt: [1, 2, 3] }, tornado: null } as RegionalFrame['stats'],
      rec };
  };
  const frames = [0, 600, 1200, 1800, 2400, 3000].map((t, i) => mkFrame(t, i % 2 ? recFull : { ...recFull!, tier: 'maps', state: null, aux: null }, i === 1));
  const metas = new Map([[1, metaTc!]]);
  const { data } = packReplay(frames, metas, 'test');
  const back = unpackReplay(data);
  let same = back.frames.length === frames.length && back.title === 'test' && back.metas.get(1)?.c.nx === metaTc!.c.nx && back.metas.get(1)?.base.th0.length === metaTc!.base.th0.length;
  frames.forEach((f, i) => {
    const g = back.frames[i]!;
    same = same && g.t === f.t && g.nx === f.nx && g.cloud.every((v, j) => v === f.cloud[j]) && g.ground.every((v, j) => v === f.ground[j]) && g.stats.vmax === f.stats.vmax && g.origin.x === f.origin.x
      && (f.nest ? g.nest?.cloud[3] === 4 && g.nest.R === 1 : g.nest === null) && g.rec?.tier === f.rec?.tier
      && Object.keys(f.rec!.maps!).every((k) => g.rec!.maps![k]!.lo === f.rec!.maps![k]!.lo && g.rec!.maps![k]!.q.every((v, j) => v === f.rec!.maps![k]!.q[j]))
      && (f.rec!.state ? g.rec!.state!.every((s, n) => s.q.length === f.rec!.state![n]!.q.length && s.q[1000] === f.rec!.state![n]!.q[1000] && s.hi === f.rec!.state![n]!.hi) && g.rec!.aux!.sst![5] === 27.5 && g.rec!.centre!.x === 600000 : g.rec!.state === null);
  });
  check('replay file: frames, packed maps and states, aux fields, nest bytes and the model meta survive pack and unpack', same);
  const sel = chooseFrames(frames, { segments: [{ t0: 0, t1: 700 }, { t0: 2400, t1: 3000 }], tier: 'view', stride: 1 });
  const sel2 = chooseFrames(frames, { segments: [{ t0: 0, t1: 3000 }], tier: 'maps', stride: 2 });
  check('replay file: segments (two ranges), every n-th frame, and the content tier choose the frames', sel.map((f) => f.t).join() === '0,600,2400,3000' && sel.every((f) => f.rec === null)
    && sel2.map((f) => f.t).join() === '0,1200,2400' && sel2.every((f) => !f.rec || f.rec.state === null), `${sel.map((f) => f.t)} | ${sel2.map((f) => f.t)}`);
  let thrown = false;
  try { unpackReplay(new ArrayBuffer(64)); } catch { thrown = true; }
  check('replay file: something else is refused', thrown);

  // 4. the store: thinning keeps the first and the newest and the run covered, 'oldest' keeps the newest, 'stop' refuses
  const per = size(mkFrame(0, null));
  const fill = (policy: 'thin' | 'oldest' | 'stop'): { st: ReplayStore; refused: number } => {
    const st = new ReplayStore(12.5 * per); st.policy = policy;
    let refused = 0;
    for (let i = 0; i < 40; i++) if (!st.push({ ...mkFrame(i * 60, null) })) refused++;
    return { st, refused };
  };
  const th = fill('thin'), ol = fill('oldest'), sp = fill('stop');
  check('replay store: thin keeps the first and the newest frame within the budget; oldest keeps the newest; stop refuses frames when full',
    th.st.frames[0]!.t === 0 && th.st.frames[th.st.length - 1]!.t === 39 * 60 && th.st.length >= 4 && th.st.dropped > 0 && th.st.megabytes * 1e6 <= th.st.budget
    && ol.st.frames[ol.st.length - 1]!.t === 39 * 60 && ol.st.frames[0]!.t > 0 && ol.st.megabytes * 1e6 <= ol.st.budget && sp.refused > 0 && sp.st.full && sp.st.frames[0]!.t === 0,
    `thin ${th.st.length} frames (${th.st.dropped} dropped), oldest ${ol.st.length}, stop ${sp.st.length} (${sp.refused} refused)`);
}

summary('replay');
