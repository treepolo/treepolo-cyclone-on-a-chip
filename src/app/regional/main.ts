// Regional-model page: controls, 3-D volume view, statistics.
import { VolumeView } from './volume.js';
import { sequential, diverging } from '../colormap.js';
import { mountSavesPanel, storeSave, type SaveMeta } from '../saves.js';
import { UnattendedRun } from './runner.js';
import { RegionalCharts } from './charts.js';
import { SatelliteRenderer } from './satellite.js';
import { Missions } from './missions.js';
import { SetupForm } from './setupForm.js';
import { defaultDt, type RegionalSetup } from './setup.js';
import type { StormNow } from '../../regional/storms.js';
import { Tools3D } from './tools3d.js';
import { ReplayStore, type ReplayFrame } from './replay.js';
import { ReplayCharts } from './replayCharts.js';
import { mountReplayPanel } from './replayPanel.js';
import type { ReplayMeta } from '../../regional/replayData.js';
import type { ChartData, FromRegionalWorker, GroundField, RegionalFrame, NestInfo, NestPayload, NestSize, RegionalExperiment, ToRegionalWorker } from './protocol.js';
import { nestGeometry, nestCells } from '../../regional/twoway.js';
import type { RegionalConfig } from '../../regional/core.js';

const presetAxi = (s: RegionalSetup): boolean => s.preset === 'tc_axi';
const km = (m: number): string => (m >= 1000 ? `${+(m / 1000).toFixed(2)} km` : `${+m.toFixed(0)} m`);

// Opened on its own without the artifact runtime: regional.html redirects to the main page (see the
// inline script there); stop here instead of starting a model that is about to be unloaded.
if (window.parent === window && !(window as unknown as { claude?: unknown }).claude && !/[?&]standalone/.test(location.search)) await new Promise(() => {});
const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const log = (s: string): void => { const el = $('log'); el.textContent = `${s}\n${el.textContent ?? ''}`.slice(0, 3000); };
const view = new VolumeView($<HTMLCanvasElement>('view'));
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const send = (m: ToRegionalWorker): void => worker.postMessage(m);
const tools = new Tools3D(view, $('tools'), document.getElementById('fx') as unknown as SVGSVGElement, send, (s) => log(s));
let running = false, dt = 6;
// vertical exaggeration of the 3-D view (null: automatic, from the domain size)
let exag: number | null = null;
const autoExag = (L: number): number => (L > 500000 ? 8 : 2.5);
let land: Uint8Array | null = null;
let nest: { payload: NestPayload; lat0: number; lon0: number; size: NestSize } | null = null;
// charts: the main view is either the 3-D volume or one of the 2-D charts
const missions = new Missions($('missions'), (s) => log(s));
let curExp: RegionalExperiment = 'supercell', curTc = false, curSea = false;
let lastVol: ReplayFrame | null = null, refining = false;
// replay: frames the worker marks to be kept (the recording settings are in the replay panel); a replay file loaded from disk
// is shown from a store of its own (fileMode), apart from the live recording
const replay = new ReplayStore(400e6), fileStore = new ReplayStore(400e6);
let fileMode = false;
const rs = (): ReplayStore => (fileMode ? fileStore : replay);
let replayIdx: number | null = null, replayPlaying = false, replayAcc = 0;
const replayCharts = new ReplayCharts((m) => send(m));
/** satellite picture renderer (undefined: not made yet, null: unavailable) and the 3-D view's second channel */
let sat: SatelliteRenderer | null | undefined, volMode = 0;
/** Show a stored or live display volume in the 3-D view. */
function showVolume(f: ReplayFrame): void {
  view.setVolume(f.nx, f.ny, f.nz, f.cloud, f.rain, Math.min(0.8, f.top / f.Lx * (exag ?? autoExag(f.Lx))), f.top, f.nest ? { ...f.nest, Lx: f.Lx, Ly: f.Ly } : null);
  view.setGround(f.nx, f.ny, f.ground);
}
const charts = new RegionalCharts($<HTMLCanvasElement>('chart'), $('chartBar'), {
  request: (req) => { if (replayIdx !== null) void replayChartsNow(); else send({ type: 'charts', req }); },
  volMode: (mode) => { volMode = mode; send({ type: 'volMode', mode }); view.setMode(mode); },
  tracers: (n) => send({ type: 'tracers', n }),
  interact: (kind, x, y, radius) => {
    if (replayIdx !== null) { log('回放中不能互動；先按「回到即時」/ no interaction during a replay: press Live first'); return; }
    if (kind === 'warm' || kind === 'cold') send({ type: 'perturb', kind, x, y });
    else if (kind !== 'inspect') send({ type: 'paint', kind, x, y, radius });
  },
  camera: (mode) => { view.setCamera(mode); $('flyPad').hidden = mode !== 'fly'; },
  cut: (c) => view.setCut(c),
  // satellite pictures of the live frame (the renderer is made when first wanted)
  satellite: (r) => {
    const f = replayIdx !== null ? rs().frames[replayIdx] ?? null : lastVol; if (!f) return null;
    if (sat === undefined) sat = SatelliteRenderer.create();
    return sat ? sat.render(f, { ...r, precip: volMode === 0, land, sea: curSea }) : null;
  },
  view: (v) => {
    $('view').hidden = v !== '3d'; $('chart').hidden = v === '3d';
    if (v === '3d') { const f = replayIdx !== null ? rs().frames[replayIdx] : lastVol; if (f) showVolume(f); }
    else if (replayIdx !== null) void replayChartsNow();
  },
});

worker.onmessage = (ev: MessageEvent<FromRegionalWorker>): void => {
  const m = ev.data;
  if (m.type === 'ready') {
    // a new model ends the replay (a refinement keeps the recording, but not a replay file shown)
    if (!refining || fileMode) { replay.clear(); endReplay(); }
    dt = m.dt;
    land = m.land;
    gridNow = { nx: m.nx, ny: m.ny, nz: m.nz, dx: m.dx, dz: m.dz, dt: m.dt, open: m.setup?.boundary === 'open' };
    // vertical exaggeration so that the troposphere is visible (unless chosen)
    if (exag === null) { const e = autoExag(m.nx * m.dx); $<HTMLInputElement>('exag').value = String(e); $('exagV').textContent = `${e}×`; }
    $('grid').textContent = `${m.nx}×${m.ny}×${m.nz}, Δx ${m.dx >= 1000 ? `${(m.dx / 1000).toFixed(1)} km` : `${m.dx.toFixed(0)} m`}, Δz ${m.dz.toFixed(0)} m, 步長 ${m.dt} s`;
    $('backend').textContent = m.backend === 'gpu' ? 'WebGPU（f32）' : 'CPU（Float64）';
    if (m.note) log(m.note);
    $('desc').textContent = m.description;
    const rb = $<HTMLButtonElement>('refine');
    rb.disabled = !m.refineTo;
    rb.textContent = m.refineTo ? `細化到 ${km(m.refineTo)} / Refine to ${km(m.refineTo)}` : '細化 / Refine';
    const cb = $<HTMLButtonElement>('coarsen');
    cb.disabled = !m.coarsenTo;
    cb.textContent = m.coarsenTo ? `粗化到 ${km(m.coarsenTo)} / Coarsen` : '粗化 / Coarsen';
    cb.title = m.coarsenBack ? '回到細化前的網格（細化的部分平均回去）/ back to the grid before the refinement (the refined run averaged into it)' : '平均到較粗的網格接著算 / continue on a coarser grid (averaged)';
    nestOk = !!m.nestOk;
    $<HTMLButtonElement>('eyeGo').disabled = !nestOk;
    eyeInfo();
    if (m.setup) form.set(m.setup);
    curExp = m.experiment; curTc = m.tc;
    // a nest has its land mask from the global model (sea elsewhere); a set-up says which surface it has
    curSea = m.setup ? m.setup.surface === 'sea' : m.experiment === 'nest';
    view.setOutside(curSea ? [0.10, 0.17, 0.30] : [0.16, 0.22, 0.16]);
    document.querySelectorAll<HTMLElement>('.tcOnly').forEach((el) => { el.hidden = !curTc; });
    document.querySelectorAll<HTMLElement>('.stormOnly').forEach((el) => { el.hidden = curTc; });
    charts.setGrid({ nx: m.nx, ny: m.ny, nz: m.nz, dx: m.dx, dy: m.dx, dz: m.dz, experiment: m.experiment, land: m.land, tc: m.tc, sea: curSea }, refining);
    showSurface();
    tools.setGrid({ Lx: m.nx * m.dx, Ly: m.ny * m.dx, top: m.nz * m.dz, dx: m.dx, dz: m.dz, paint: m.setup ? m.setup.fluxes && !presetAxi(m.setup) : m.experiment === 'nest', interact: m.experiment !== 'tc_axi' });
    missions.reset({ e: m.experiment, tc: m.tc });
    refining = false;
    replayPanel.onReady(m.tc, m.experiment === 'tc_axi');
    log(`就緒 / Ready: ${m.description}`);
  } else if (m.type === 'frame') {
    runner.sample(m);
    charts.onFrame(m);
    missions.check(m, { e: curExp, tc: curTc });
    const in3d = charts.view === '3d';
    const [lo, hi] = m.groundRange, rgba = new Uint8Array(m.nx * m.ny * 4);
    // plain surface: sea (tropical cyclones, sea points of a nest) or land (convective storms, land points)
    const bare = (i: number): [number, number, number] => ((land ? !land[i] : curSea) ? [0.10, 0.17, 0.30] : [0.16, 0.22, 0.16]);
    for (let i = 0; i < m.ground.length; i++) {
      const v = m.ground[i]!;
      let c: [number, number, number];
      if (m.groundField === 'theta') c = diverging(v / hi);
      else if (m.groundField === 'none') c = bare(i);
      else if (m.groundField === 'rain' || m.groundField === 'snow') { const t = hi > 0 ? Math.sqrt(Math.max(0, v) / hi) : 0; c = t < 0.02 ? bare(i) : sequential(0.15 + 0.85 * t); }
      else c = sequential((v - lo) / ((hi - lo) || 1));
      rgba[4 * i] = c[0] * 255; rgba[4 * i + 1] = c[1] * 255; rgba[4 * i + 2] = c[2] * 255; rgba[4 * i + 3] = 255;
    }
    lastVol = { t: m.time, nx: m.nx, ny: m.ny, nz: m.nz, dx: m.dx, dz: m.dz, Lx: m.nx * m.dx, Ly: m.ny * m.dx, top: m.nz * m.dz, cloud: m.cloud, rain: m.rain, ground: rgba, storms: m.stats.storms ?? [], nest: m.nest,
      stats: m.stats, origin: m.origin };
    if (m.nest) showNest(m.nest);
    // the frames the worker marks (recording interval in model time) go to the replay store
    if (m.keep && !fileMode) { if (!replay.push({ ...lastVol, rec: m.rec ?? null })) replayPanel.onFull(); replayPanel.refresh(); }
    if (in3d && replayIdx === null) { showVolume(lastVol); view.setTracers(m.tracers, m.nx * m.dx, m.ny * m.dx, m.nz * m.dz); }
    replayBar();
    const s = m.stats, t = m.time;
    $('time').textContent = t < 7200 * 3 ? `${(t / 60).toFixed(0)} min` : `${(t / 3600).toFixed(1)} h (${(t / 86400).toFixed(2)} d)`;
    $('ovTime').textContent = t < 7200 * 3 ? `${(t / 60).toFixed(0)} min` : `${(t / 3600).toFixed(1)} h`;
    dt = m.dt;
    showStep(m.step, m.dt);
    const ms = m.stepsPerSecond * m.dt;
    $('rate').textContent = `${m.stepsPerSecond.toFixed(m.stepsPerSecond < 10 ? 2 : 1)} 步/s steps/s · ${ms < 60 ? `${ms.toFixed(1)} 模式秒/s model-s/s` : `${(ms / 60).toFixed(1)} 模式分/s model-min/s`} · 步長 ${m.dt.toFixed(1)} s`;
    $('w').textContent = `${s.wmin.toFixed(1)} … ${s.wmax.toFixed(1)} m/s`;
    $('qc').textContent = `${(s.qcmax * 1000).toFixed(2)} g/kg`;
    $('qr').textContent = `${(s.qrmax * 1000).toFixed(2)} g/kg`;
    $('rainmax').textContent = `${s.rainmax.toFixed(1)} mm`;
    $('vmax').textContent = `${s.vmax.toFixed(1)} m/s`;
    $('dp').textContent = s.dp === null ? '—' : `${s.dp.toFixed(1)} hPa${s.rmw ? ` · RMW ${(s.rmw / 1000).toFixed(0)} km` : ''}`;
    const dirName = (d: number): string => ['北', '東北', '東', '東南', '南', '西南', '西', '西北'][Math.round(d / 45) % 8]!;
    $('envShear').textContent = s.shear ? `${s.shear.mag.toFixed(1)} m/s（指向${dirName(s.shear.dir)} ${s.shear.dir.toFixed(0)}°）` : '—';
    $('eyewalls').textContent = !s.eyewalls ? '—' : s.eyewalls.length === 0 ? '未形成 / none yet'
      : (s.eyewalls.length >= 2 ? '雙眼牆 / concentric: ' : '') + s.eyewalls.map((e) => `${(e.r / 1000).toFixed(0)} km (${e.v.toFixed(0)} m/s)`).join(' · ');
    $('zeta').textContent = `${s.zetaMax.toFixed(3)} s⁻¹ · ${s.vGround.toFixed(1)} m/s`;
    $('tcrain').textContent = !s.tcRain ? '—' : `${s.tcRain.core.toFixed(1)} · ${s.tcRain.outer.toFixed(2)} mm/h（外圍 >1 mm/h ${(100 * s.tcRain.wet).toFixed(1)}%）`;
    tornadoWatch(m.time, s.tornado, m.dx);
    showStorms(s.storms ?? [], s.mainId ?? null, m);
    $('legend').textContent = m.groundField === 'none' ? '深藍 = 海、深綠 = 陸地 / dark blue = sea, dark green = land' : `${lo.toFixed(1)} … ${hi.toFixed(1)} ${m.groundField === 'rain' || m.groundField === 'snow' ? 'mm' : m.groundField === 'wind' ? 'm/s' : 'K'}`;
  } else if (m.type === 'error') { failSave?.(new Error(m.message)); log(`錯誤 / Error: ${m.message}`); running = false; sync(); $<HTMLButtonElement>('eyeGo').disabled = !nestOk; if (runner.running) void runner.abort(`error: ${m.message}`); }
  else if (m.type === 'nest') { showNest(m.info); $<HTMLButtonElement>('eyeGo').disabled = !nestOk; if (!m.info) view.clearNest(); }
  else if (m.type === 'saveData') pendingSave?.({ meta: m.meta, data: m.buffer });
  else if (m.type === 'paused') { log(m.reason); running = false; $('run').textContent = '執行 / Run'; $('ovRun').textContent = '▶'; if (runner.running) void runner.end('done'); }
  else if (m.type === 'log') log(m.text);
  else if (m.type === 'land') { land = m.land; charts.setLand(m.land); showSurface(); }
  else if (m.type === 'replayChart') replayCharts.onAnswer(m);
  else if (m.type === 'forcings') tools.setForcings(m.list);
  else if (m.type === 'profile') { $('profileOut').textContent = m.text; $<HTMLButtonElement>('profile').disabled = false; }
};
// tornado events: a detection that lasts at least one model minute; logged when it starts and when it ends
let tEvent: { start: number; last: number; ef: number; v: number; zeta: number; logged: boolean } | null = null;
function tornadoWatch(t: number, d: { zeta: number; v: number; ef: number; x: number; y: number } | null, dx: number): void {
  if (d) {
    if (!tEvent || t - tEvent.last > 120) tEvent = { start: t, last: t, ef: d.ef, v: d.v, zeta: d.zeta, logged: false };
    tEvent.last = t; tEvent.ef = Math.max(tEvent.ef, d.ef); tEvent.v = Math.max(tEvent.v, d.v); tEvent.zeta = Math.max(tEvent.zeta, d.zeta);
    if (!tEvent.logged && t - tEvent.start >= 60) { tEvent.logged = true; log(`龍捲偵測 / Tornado detected at t = ${(t / 60).toFixed(0)} min: EF${d.ef}, ${d.v.toFixed(0)} m/s, ζ ${d.zeta.toFixed(2)} s⁻¹ at (${(d.x / 1000).toFixed(1)}, ${(d.y / 1000).toFixed(1)}) km`); }
    $('tornado').textContent = `EF${d.ef} · ${d.v.toFixed(0)} m/s · ζ ${d.zeta.toFixed(2)} s⁻¹ · ${((t - tEvent.start) / 60).toFixed(0)} min`;
  } else {
    if (tEvent && tEvent.logged && t - tEvent.last > 120) { log(`龍捲結束 / Tornado ended: lasted ${((tEvent.last - tEvent.start) / 60).toFixed(0)} min, peak EF${tEvent.ef} (${tEvent.v.toFixed(0)} m/s)`); tEvent = null; }
    $('tornado').textContent = dx > 500 ? '（網格太粗 / grid too coarse）' : tEvent?.logged ? `上次 / last: EF${tEvent.ef}` : '無 / none';
  }
}
function sync(): void { $('run').textContent = running ? '暫停 / Pause' : '執行 / Run'; $('ovRun').textContent = running ? '⏸' : '▶'; send({ type: 'run', running }); }
$('run').onclick = (): void => { running = !running; sync(); };
$('ovRun').onclick = (): void => { running = !running; sync(); };
$('ovStep').onclick = (): void => { if (!running) send({ type: 'step1' }); };
// space bar: run / pause (not while typing in a form field)
window.addEventListener('keydown', (e) => {
  const a = document.activeElement;
  if (e.key !== ' ' || (a && (a.tagName === 'INPUT' || a.tagName === 'SELECT' || a.tagName === 'TEXTAREA' || a.tagName === 'BUTTON'))) return;
  e.preventDefault(); running = !running; sync();
});
// free-flight buttons: hold to move
document.querySelectorAll<HTMLButtonElement>('#flyPad button').forEach((b) => {
  const k = b.dataset.k!, on = (e: PointerEvent): void => { e.preventDefault(); b.setPointerCapture(e.pointerId); view.setKey(k, true); }, off = (): void => view.setKey(k, false);
  b.addEventListener('pointerdown', on); b.addEventListener('pointerup', off); b.addEventListener('pointercancel', off); b.addEventListener('lostpointercapture', off);
});
// ---------------- storms: list in the diagnostics (click: its numbers in the time series) and labels in the 3-D view
let stormsNow: StormNow[] = [], stormFrame: { nx: number; dx: number; ny: number; nz: number; dz: number } | null = null, stormMain: number | null = null;
const compass = (u: number, v: number): string => {
  const sp = Math.hypot(u, v); if (sp < 0.5) return '幾乎不動 / nearly still';
  const d = ['東 E', '東北 NE', '北 N', '西北 NW', '西 W', '西南 SW', '南 S', '東南 SE'][Math.round(Math.atan2(v, u) / (Math.PI / 4) + 8) % 8]!;
  return `往${d} ${sp.toFixed(0)} m/s`;
};
const galeKm = (r: number | undefined): string => (r === undefined ? '—' : r > 0 ? `${(r / 1000).toFixed(0)} km` : '無');
function stormLine(st: StormNow): string {
  const age = st.age < 7200 ? `${(st.age / 60).toFixed(0)} min` : `${(st.age / 3600).toFixed(1)} h`;
  if (st.kind === 'vortex') return `${st.pmin?.toFixed(0) ?? '—'} hPa（−${st.dp?.toFixed(1) ?? '—'}）· ${st.vmax?.toFixed(0) ?? '—'} m/s · RMW ${st.rmw ? (st.rmw / 1000).toFixed(0) : '—'} km · 七級 ${galeKm(st.r7)} · 十級 ${galeKm(st.r10)} · ${compass(st.u, st.v)} · ${age}`;
  return `w ${st.wmax?.toFixed(0) ?? '—'} m/s · UH ${st.uh?.toFixed(0) ?? '—'} · ${st.dbz?.toFixed(0) ?? '—'} dBZ · ${compass(st.u, st.v)} · ${age}`;
}
function showStorms(list: StormNow[], mainId: number | null, f: { nx: number; ny: number; nz: number; dx: number; dz: number }): void {
  stormsNow = list; stormFrame = f; stormMain = mainId;
  const mv = list.find((s) => s.id === mainId && s.kind === 'vortex') ?? list.find((s) => s.kind === 'vortex');
  $('gale').textContent = mv ? `${galeKm(mv.r7)} · ${galeKm(mv.r10)}` : '—';
  const box = $('stormList'), sel = charts.selectedStorm;
  $('stormCount').textContent = list.length ? `${list.length} 個 / ${list.length}` : '尚未形成 / none yet';
  // vortices first, strongest first; rows are reused (a click must survive the next frame's update)
  const order = [...list].sort((a, b) => (a.kind === b.kind ? (b.dp ?? Math.abs(b.uh ?? 0) + 10 * (b.wmax ?? 0)) - (a.dp ?? Math.abs(a.uh ?? 0) + 10 * (a.wmax ?? 0)) : a.kind === 'vortex' ? -1 : 1)).slice(0, 12);
  while (box.childElementCount < order.length + 1) {
    const b = document.createElement('button'), sw = document.createElement('span'), nm = document.createElement('span'), dt = document.createElement('span');
    sw.className = 'sw'; nm.className = 'nm'; dt.className = 'dt'; b.append(sw, nm, dt);
    b.onclick = (): void => { const id = Number(b.dataset.id); charts.selectStorm(charts.selectedStorm === id ? null : id); if (stormFrame) showStorms(stormsNow, stormMain, stormFrame); };
    box.append(b);
  }
  const kids = box.children;
  order.forEach((st, n) => {
    const b = kids[n] as HTMLButtonElement, [sw, nm, dt] = Array.from(b.children) as HTMLElement[];
    b.hidden = false; b.dataset.id = String(st.id); b.className = st.id === sel ? 'sel' : '';
    sw!.style.background = charts.stormColour(st.id);
    nm!.textContent = st.name + (st.id === mainId ? ' ★' : '');
    dt!.textContent = stormLine(st);
    b.title = st.id === mainId ? '主要風暴（跟隨、診斷用它）/ main storm (followed, used by the diagnostics)' : '點選：時間序列顯示它的數據 / click: its numbers in the time series';
  });
  for (let n = order.length; n < kids.length; n++) (kids[n] as HTMLElement).hidden = true;
  // the last row is the note on weaker storms
  const note = kids[order.length] as HTMLButtonElement;
  if (list.length > 12) { note.hidden = false; note.disabled = true; note.className = ''; note.dataset.id = ''; (note.children[0] as HTMLElement).style.background = 'transparent'; (note.children[1] as HTMLElement).textContent = ''; (note.children[2] as HTMLElement).textContent = `另有 ${list.length - 12} 個較弱的 / ${list.length - 12} weaker ones not listed`; }
  for (let n = 0; n < order.length; n++) (kids[n] as HTMLButtonElement).disabled = false;
}
/** Labels above the storms in the 3-D view (redrawn every animation frame: the camera moves). */
function placeStormLabels(): void {
  const host = $('stormLabels'), rf = replayIdx !== null ? rs().frames[replayIdx] ?? null : null;
  const list = rf ? rf.storms : stormsNow;
  if (charts.view !== '3d' || !$<HTMLInputElement>('showLabels').checked || (!rf && !stormFrame) || !list.length) { if (host.childElementCount) host.replaceChildren(); return; }
  const [bx, by, bz] = view.boxSize, Lx = rf ? rf.Lx : stormFrame!.nx * stormFrame!.dx, Ly = rf ? rf.Ly : stormFrame!.ny * stormFrame!.dx, off = $('view').offsetTop;
  while (host.childElementCount < Math.min(12, list.length)) host.append(document.createElement('div'));
  while (host.childElementCount > Math.min(12, list.length)) host.lastElementChild!.remove();
  list.slice(0, 12).forEach((st, n) => {
    const el = host.children[n] as HTMLElement, p = view.project([st.xd / Lx * bx, st.yd / Ly * by, bz * 0.92]);
    if (!p) { el.hidden = true; return; }
    el.hidden = false;
    el.style.left = `${p.x}px`; el.style.top = `${p.y + off}px`; el.style.borderColor = charts.stormColour(st.id);
    el.textContent = st.kind === 'vortex' ? `${st.name} ${st.pmin?.toFixed(0) ?? ''} hPa ${st.vmax?.toFixed(0) ?? ''} m/s` : `${st.name} w ${st.wmax?.toFixed(0) ?? ''}`;
  });
}
/** Sea / land shown in the diagnostics: the set-up's surface and the painted share. */
function showSurface(): void {
  const n = land ? land.reduce((a, x) => a + x, 0) : 0, frac = land ? n / land.length : curSea ? 0 : 1;
  $('surface').textContent = frac <= 0 ? '海洋 / sea' : frac >= 1 ? '陸地 / land' : `海洋 ${(100 * (1 - frac)).toFixed(0)}%、陸地 ${(100 * frac).toFixed(0)}% / sea and land`;
}
const form = new SetupForm($<HTMLSelectElement>('preset'), { grid: $('setupGrid'), env: $('setupEnv'), init: $('setupInit') }, $('setupInfo'), $<HTMLButtonElement>('apply'), $<HTMLButtonElement>('resetPreset'));
const backend = (): 'auto' | 'cpu' => $<HTMLSelectElement>('backendSel').value as 'auto' | 'cpu';
function start(s: RegionalSetup): void {
  running = false; sync();
  log(`建立模式中 / Building the model…`);
  send({ type: 'init', setup: s, backend: backend() });
}
function startNest(): void {
  running = false; sync();
  if (!nest) { log('等待全球模式傳送資料… / Waiting for the global model state…'); return; }
  log('建立巢狀區域模式中 / Building the nested regional model…');
  send({ type: 'initNest', payload: nest.payload, lat0: nest.lat0, lon0: nest.lon0, size: nest.size, backend: backend() });
}
form.onApply = start;
form.onNest = startNest;
const init = (): void => { if (form.isNest) startNest(); else start(form.value()); };
$('envApply').onclick = (): void => {
  const du6 = Number($<HTMLInputElement>('envDu').value), humidity = Number($<HTMLInputElement>('envRh').value);
  if (!Number.isFinite(du6) || !(humidity >= 0)) return;
  send({ type: 'environment', du6, humidity });
};
// remember which sections are open (per-viewer convenience)
document.querySelectorAll<HTMLDetailsElement>('details[id]').forEach((d) => {
  try { const v = localStorage.getItem(`regional-${d.id}`); if (v !== null) d.open = v === '1'; } catch { /* storage unavailable */ }
  d.addEventListener('toggle', () => { try { localStorage.setItem(`regional-${d.id}`, d.open ? '1' : '0'); } catch { /* storage unavailable */ } });
});
$('backendSel').onchange = init;
$('ground').onchange = (): void => send({ type: 'ground', field: $<HTMLSelectElement>('ground').value as GroundField });
$('subgrid').onchange = (): void => send({ type: 'subgrid', on: $<HTMLInputElement>('subgrid').checked });
// cloud-physics switches: ice supersaturation, and the time liquid water takes to condense (off: at once)
const sendMicro = (): void => {
  const on = $<HTMLInputElement>('liqSS').checked, tau = Number($<HTMLInputElement>('liqTau').value);
  $<HTMLInputElement>('liqTau').disabled = !on;
  send({ type: 'micro', iceSS: $<HTMLInputElement>('iceSS').checked, liqTau: on && tau > 0 ? tau * 60 : 0 });
};
$('iceSS').onchange = sendMicro; $('liqSS').onchange = sendMicro; $('liqTau').onchange = sendMicro;
// (a browser may restore the form: tell the worker unless it is as the worker starts)
if (!$<HTMLInputElement>('iceSS').checked || $<HTMLInputElement>('liqSS').checked) sendMicro();
$('exag').oninput = (): void => {
  exag = Number($<HTMLInputElement>('exag').value); $('exagV').textContent = `${exag}×`;
  const f = replayIdx !== null ? rs().frames[replayIdx] : lastVol;
  if (f && charts.view === '3d') showVolume(f);
};
$('fov').oninput = (): void => { const d = Number($<HTMLInputElement>('fov').value); view.setFov(d); $('fovV').textContent = `${d}°`; };
$('speed').oninput = (): void => send({ type: 'speed', stepsPerTick: Number($<HTMLInputElement>('speed').value) });
// Step size (Δt): automatic with a target Courant number, or a manual value (editing the value makes it manual)
const dtModeEl = $<HTMLSelectElement>('dtMode'), dtValEl = $<HTMLInputElement>('dtVal'), dtCflEl = $<HTMLInputElement>('dtCfl');
const stepNow = (): number => { const v = Number(dtValEl.value); return Number.isFinite(v) && v > 0 ? v : dt; };
let stepSeq = 0;
const sendCfl = (): void => { const c = Number(dtCflEl.value); if (Number.isFinite(c) && c > 0) send({ type: 'step', seq: ++stepSeq, mode: dtModeEl.value as 'auto' | 'manual', cfl: c, ...(dtModeEl.value === 'manual' ? { dt: stepNow() } : {}) }); };
dtModeEl.onchange = (): void => { if (dtModeEl.value === 'auto') sendCfl(); else send({ type: 'step', seq: ++stepSeq, mode: 'manual', dt: stepNow() }); };
dtCflEl.onchange = sendCfl;
const setManual = (v: number): void => { if (!(v > 0)) return; dtModeEl.value = 'manual'; dtValEl.value = String(+v.toPrecision(3)); send({ type: 'step', seq: ++stepSeq, mode: 'manual', dt: v }); };
dtValEl.onchange = (): void => setManual(Number(dtValEl.value));
$('dtHalf').onclick = (): void => setManual(stepNow() / 2);
$('dtDouble').onclick = (): void => setManual(stepNow() * 2);
/** Show the step size control of a frame (not over a field being edited). */
function showStep(s: { mode: 'auto' | 'manual'; cfl: number; dt0: number; adv: number; ac: number; seq: number }, dtNow: number): void {
  // (a frame sent before the worker handled the last change still shows the old state: its controls are not taken)
  const fixed = curExp === 'tc_axi', active = document.activeElement, stale = s.seq < stepSeq;
  dtModeEl.disabled = dtValEl.disabled = dtCflEl.disabled = fixed;
  $<HTMLButtonElement>('dtHalf').disabled = $<HTMLButtonElement>('dtDouble').disabled = fixed;
  if (active !== dtModeEl && !stale) dtModeEl.value = fixed ? 'manual' : s.mode;
  if (active !== dtCflEl && !stale && s.cfl > 0) dtCflEl.value = String(s.cfl);
  if (active !== dtValEl && !stale) dtValEl.value = String(+dtNow.toPrecision(3));
  $('dtCflRow').hidden = dtModeEl.value === 'manual';
  const info = $('dtInfo');
  if (fixed) { info.textContent = '固定 / fixed'; info.style.color = ''; return; }
  const bad = s.adv > 1.2 || s.ac > 0.7;
  info.textContent = `平流 ${s.adv.toFixed(2)} · 聲波 ${s.ac.toFixed(2)}${s.mode === 'auto' ? ` · 基準 ${s.dt0.toFixed(1)} s` : ''}`;
  info.style.color = bad ? '#e86a4a' : '';
}
// Nesting: embedded by the global page (in-page overlay, #nest...) or opened with ?nest=1; the
// parent/opener posts the global state and the chosen point.
const host = window.parent !== window ? window.parent : (window.opener as Window | null);
if (location.hash.startsWith('#nest') || new URLSearchParams(location.search).has('nest')) {
  form.addNest(true);
  window.addEventListener('message', (ev: MessageEvent) => {
    if (!host || ev.source !== host || !ev.data || ev.data.type !== 'nest') return;
    nest = { payload: ev.data.payload as NestPayload, lat0: ev.data.lat0 as number, lon0: ev.data.lon0 as number, size: (ev.data.size as NestSize) ?? 'meso' };
    form.addNest(true);
    startNest();
  });
  if (host) host.postMessage({ type: 'nest-ready' }, '*');
  else log('找不到全球模式視窗；請從全球模式頁面點選地點後開啟 / No global-model window found; open this page from the global model after picking a point');
}
if (window.parent !== window) { const back = document.getElementById('backLink'); if (back) back.hidden = true; }
init();
// ---------------- replay
const fmtT = (t: number): string => (t < 7200 * 3 ? `${(t / 60).toFixed(0)} min` : `${(t / 3600).toFixed(1)} h`);
function replayBar(): void {
  const st = rs(), n = st.length, sl = $<HTMLInputElement>('rpSlider');
  sl.max = String(Math.max(0, n - 1));
  if (replayIdx === null) return;
  replayIdx = Math.min(replayIdx, n - 1);
  sl.value = String(replayIdx);
  const f = st.frames[replayIdx];
  $('rpTime').textContent = f ? `${fileMode ? '檔案 ' : ''}${fmtT(f.t)}（${replayIdx + 1}/${n}）` : '—';
}
/** the chart data of the replayed frame for the charts' current request (one at a time; the newest request waits its turn) */
let rpBusy = false, rpAgain = false;
async function replayChartsNow(): Promise<void> {
  if (rpBusy) { rpAgain = true; return; }
  rpBusy = true;
  try {
    do {
      rpAgain = false;
      const st = rs(), i = replayIdx, f = i === null ? undefined : st.frames[i];
      if (!f || charts.view === '3d') continue;
      let charts_: ChartData, note: string;
      try { ({ charts: charts_, note } = await replayCharts.compute(f, st, charts.currentRequest())); }
      catch (e) { log(`回放圖表失敗 / replay chart failed: ${String((e as Error).message ?? e)}`); continue; }
      if (replayIdx === null || st.frames[replayIdx] !== f) continue;                     // moved on meanwhile
      const frame = { type: 'frame', time: f.t, nx: f.nx, ny: f.ny, nz: f.nz, dx: f.dx, dz: f.dz, cloud: f.cloud, rain: f.rain, ground: new Float32Array(0), groundField: 'rain', groundRange: [0, 1],
        stats: f.stats, origin: f.origin, charts: charts_, tracers: null, nest: f.nest ?? null, stepsPerSecond: 0, dt: 0, step: { mode: 'manual', cfl: 0, dt0: 0, adv: 0, ac: 0, seq: 0 } } as RegionalFrame;
      charts.showReplay(frame, note);
    } while (rpAgain);
  } finally { rpBusy = false; }
}
function showReplay(i: number): void {
  const st = rs();
  if (!st.length) return;
  replayIdx = Math.max(0, Math.min(st.length - 1, i));
  const f = st.frames[replayIdx]!;
  if (charts.view === '3d') { showVolume(f); view.setTracers(null, f.Lx, f.Ly, f.top); }
  else void replayChartsNow();
  replayBar();
  replayPanel.refresh();
}
function endReplay(): void {
  const was = replayIdx !== null || fileMode;
  replayIdx = null; replayPlaying = false; $('replayBar').hidden = true; $('stage').classList.remove('rpOn'); $('rpPlay').textContent = '▶';
  if (fileMode) { fileMode = false; fileStore.clear(); replayPanel.apply(); }
  if (was) charts.endReplay();
  if (lastVol && charts.view === '3d') showVolume(lastVol);
  replayPanel.refresh();
}
function startReplay(): void {
  if (replayIdx !== null) { endReplay(); return; }
  if (!rs().length) { log('還沒有可以重播的畫面（要先讓模擬跑一下，或載入回放資料）/ nothing to replay yet'); return; }
  $('replayBar').hidden = false; $('stage').classList.add('rpOn'); replayPlaying = false; showReplay(0);
  log(`重播：${rs().length} 個畫面，${rs().megabytes.toFixed(0)} MB / replay: ${rs().length} frames`);
}
$('ovReplay').onclick = startReplay;
/** A replay file was read: its frames are shown (with its own grid and history in the charts) until Live is pressed. */
function enterFile(frames: ReplayFrame[], metas: Map<number, ReplayMeta>, title: string): void {
  if (replayIdx !== null || fileMode) endReplay();
  fileStore.load(frames, metas); fileMode = true;
  send({ type: 'record', on: false, tier: replayPanel.settings.tier, every: replayPanel.settings.every });   // the live frames are not kept meanwhile
  const f = frames[0]!, meta = [...metas.values()][0];
  const nx = Math.round(f.Lx / f.dx), ny = Math.round(f.Ly / f.dx), nz = Math.round(f.top / f.dz);
  const tc = meta ? meta.tc : f.stats.vtProfile !== null || f.stats.dp !== null;
  charts.enterFile({ nx, ny, nz, dx: f.dx, dy: f.dx, dz: f.dz, experiment: (meta?.experiment ?? 'custom') as RegionalExperiment, land: meta?.land ?? null, tc, sea: meta?.sea ?? false },
    frames.map((x) => ({ time: x.t, stats: x.stats })));
  $('replayBar').hidden = false; $('stage').classList.add('rpOn'); replayPlaying = false; showReplay(0);
  log(`回放資料「${title}」：${frames.length} 格；按「回到即時」離開 / replay data loaded; press Live to leave`);
}
const replayPanel = mountReplayPanel({ $, send, log, live: replay, file: fileStore, fileMode: () => fileMode, step: () => dt,
  grid: () => (gridNow.nx ? gridNow : null), replayTime: () => (replayIdx === null ? null : rs().frames[replayIdx]?.t ?? null),
  openReplay: startReplay, loaded: enterFile, cleared: () => { if (replayIdx !== null && !fileMode) endReplay(); } });
$('rpSlider').oninput = (): void => { replayPlaying = false; $('rpPlay').textContent = '▶'; showReplay(Number($<HTMLInputElement>('rpSlider').value)); };
$('rpPlay').onclick = (): void => {
  if (replayIdx === null) return;
  if (!replayPlaying && replayIdx >= rs().length - 1) showReplay(0);
  replayPlaying = !replayPlaying; replayAcc = 0; $('rpPlay').textContent = replayPlaying ? '⏸' : '▶';
};
$('rpLive').onclick = endReplay;
$('compass').onclick = (): void => view.faceNorth();
let lastTick = performance.now();
function tick(): void {
  const in3d = charts.view === '3d', now = performance.now(), dts = Math.min(0.2, (now - lastTick) / 1000); lastTick = now;
  if (replayIdx !== null && replayPlaying) {
    replayAcc += dts * Number($<HTMLSelectElement>('rpSpeed').value);
    if (replayAcc >= 1) {
      const next = replayIdx + Math.floor(replayAcc); replayAcc -= Math.floor(replayAcc);
      if (next >= rs().length - 1) { showReplay(rs().length - 1); replayPlaying = false; $('rpPlay').textContent = '▶'; } else showReplay(next);
    }
  }
  if (in3d) view.render(Number($<HTMLInputElement>('cloudK').value));
  const cp = $('compass'); cp.hidden = !in3d;
  if (in3d) { cp.style.top = `${$('view').offsetTop + 8}px`; $('compassG').setAttribute('transform', `rotate(${(view.northAngle * 180 / Math.PI).toFixed(1)})`); }
  placeStormLabels();
  // tools and their overlay sit on the 3-D view (below the chart bar)
  const v = $('view'), fx = $('fx'), tl = $('tools');
  tl.hidden = !in3d; fx.style.display = in3d ? '' : 'none';
  if (in3d) {
    tl.style.top = `${v.offsetTop + 8}px`; fx.style.top = `${v.offsetTop}px`; fx.style.height = `${v.clientHeight}px`;
    tools.drawOverlay();
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
$('coarsen').onclick = (): void => { $<HTMLButtonElement>('coarsen').disabled = true; refining = true; log('粗化中… / Coarsening…'); send({ type: 'coarsen' }); };
/** the grid of the current run (the nest is built on it) and whether the eye nest applies */
let gridNow = { nx: 0, ny: 0, nz: 0, dx: 15000, dz: 500, dt: 60, open: false }, nestOk = false;
/** Eye nest from the panel (m). */
function eyeVals(): { R: number; dx: number; dz: number } {
  const num = (id: string, d: number): number => { const v = Number($<HTMLInputElement>(id).value); return Number.isFinite(v) && v > 0 ? v : d; };
  return { R: num('eyeR', 75) * 1000, dx: num('eyeDx', 1) * 1000, dz: num('eyeDz', 250) };
}
/** radius of maximum wind (m) of the main vortex, if one is detected */
function eyeRmw(): number | null {
  const v = stormsNow.find((s) => s.id === stormMain && s.kind === 'vortex') ?? stormsNow.find((s) => s.kind === 'vortex');
  return v && v.rmw ? v.rmw : null;
}
const kmTxt = (d: number): string => (d >= 1000 ? `${+(d / 1000).toFixed(2)} km` : `${Math.round(d)} m`);
/** The nest the panel asks for on the current grid (exact spacings, cells, inner steps) and its work per model second
 *  relative to the current grid. */
function eyeInfo(): void {
  const { R, dx, dz } = eyeVals(), g = gridNow;
  if (!g.nx) { $('eyeInfo').textContent = ''; return; }
  const pc = { nx: g.nx, ny: g.ny, nz: g.nz, dx: g.dx, dy: g.dx, dz: g.dz, lateral: g.open ? 'open' : 'periodic', relaxCells: Math.max(6, Math.round(2500 / g.dx)) } as unknown as RegionalConfig;
  const geo = nestGeometry(pc, R, dx, dz);
  if (typeof geo === 'string') { $('eyeInfo').textContent = `⚠ ${geo}`; return; }
  const cells = nestCells(geo), outer = g.nx * g.ny * g.nz, nsub = Math.max(1, Math.ceil(g.dt / defaultDt({ dx: geo.dx, dz: geo.dz }) - 1e-6));
  const cost = 1 + cells * nsub / outer, rmw = eyeRmw(), notes: string[] = [];
  if (rmw && R < 2 * rmw) notes.push(`⚠ 半徑小於最大風半徑 ${(rmw / 1000).toFixed(0)} km 的 2 倍，眼牆會碰到圓柱邊緣 / radius under twice the radius of maximum wind (${(rmw / 1000).toFixed(0)} km)`);
  if (cells > 8e6) notes.push('⚠ 需要較強的顯卡 / needs a strong GPU');
  const c = cost < 10 ? cost.toFixed(1) : cost.toFixed(0);
  $('eyeInfo').textContent = `實際 / actual: Δx ${kmTxt(geo.dx)}（外圍 ${kmTxt(g.dx)} 的 1/${geo.r}）、Δz ${kmTxt(geo.dz)}（1/${geo.rz}），` +
    `細網格 ${kmTxt(geo.nx * geo.dx)} 見方，${geo.nx}×${geo.nx}×${geo.nz} = ${(cells / 1e6).toFixed(2)} M 格，每個外圍步約 ${nsub} 個內部步；每模式秒的計算量約為目前的 ${c} 倍` +
    ` / fine box ${kmTxt(geo.nx * geo.dx)} across, ${(cells / 1e6).toFixed(2)} M cells, about ${nsub} inner steps per outer step: about ${c}× the work per model second` +
    (notes.length ? ` · ${notes.join(' · ')}` : '');
}
/** Suggest a nest for the current eyewall: radius about 2.5 times the radius of maximum wind (20-300 km), spacing about
 *  a twentieth of it (0.5 km to half the outer spacing), half the outer level spacing. */
function eyeFit(): void {
  const rmw = eyeRmw() ?? 30000;
  const R = Math.max(20, Math.min(300, Math.ceil(2.5 * rmw / 5000) * 5));
  const dx = Math.max(0.5, Math.min(gridNow.dx / 2000, Math.round(rmw / 20000 * 4) / 4));
  $<HTMLInputElement>('eyeR').value = String(R); $<HTMLInputElement>('eyeDx').value = String(dx); $<HTMLInputElement>('eyeDz').value = String(Math.max(50, gridNow.dz / 2));
  eyeInfo();
}
/** The running nest (from the worker; null: none). */
function showNest(n: NestInfo | null): void {
  $<HTMLButtonElement>('eyeStop').disabled = !n;
  $('eyeNow').textContent = n ? `細化中 / running: 半徑 ${kmTxt(n.R)}，Δx ${kmTxt(n.dx)}、Δz ${kmTxt(n.dz)}，${(n.cells / 1e6).toFixed(2)} M 格，每個外圍步 ${n.nsub} 個內部步 / inner steps per outer step`
    : '未細化 / not running';
}
for (const id of ['eyeR', 'eyeDx', 'eyeDz']) $(id).oninput = eyeInfo;
$('eyeFit').onclick = eyeFit;
$('secEye').addEventListener('toggle', () => { if (($('secEye') as HTMLDetailsElement).open) eyeFit(); });
$('eyeGo').onclick = (): void => {
  $<HTMLButtonElement>('eyeGo').disabled = true; log('眼區細化建立中… / Starting the eye nest…');
  send({ type: 'nestStart', ...eyeVals() });
};
$('eyeStop').onclick = (): void => { $<HTMLButtonElement>('eyeStop').disabled = true; send({ type: 'nestStop' }); };
$('profile').onclick = (): void => { $<HTMLButtonElement>('profile').disabled = true; $('profileOut').textContent = '量測中… / Measuring…'; worker.postMessage({ type: 'profile' } satisfies ToRegionalWorker); };
$('refine').onclick = (): void => { $<HTMLButtonElement>('refine').disabled = true; refining = true; log('細化中… / Refining…'); send({ type: 'refine' }); };
// ---------------- saved simulations
let pendingSave: ((r: { meta: SaveMeta; data: ArrayBuffer }) => void) | null = null, failSave: ((e: Error) => void) | null = null;
const captureSave = (): Promise<{ meta: SaveMeta; data: ArrayBuffer }> => new Promise((resolve, reject) => {
  if (pendingSave) { reject(new Error('存檔進行中 / a save is already in progress')); return; }
  const end = (): void => { pendingSave = null; failSave = null; clearTimeout(timer); };
  const timer = window.setTimeout(() => { end(); reject(new Error('逾時 / timed out')); }, 120000);
  pendingSave = (r): void => { end(); resolve(r); };
  failSave = (e): void => { end(); reject(e); };
  send({ type: 'save' });
});
const savesPanel = mountSavesPanel($('saves'), 'regional', captureSave,
  (_meta, data) => { running = false; sync(); send({ type: 'load', buffer: data.slice(0), backend: backend() }); },
  log);
// ---------------- pacing
$('pace').onchange = (): void => send({ type: 'pace', target: Number($<HTMLSelectElement>('pace').value) });
$('frames').onchange = (): void => {
  const v = $<HTMLSelectElement>('frames').value;
  send(v.startsWith('model:') ? { type: 'frames', kind: 'model', every: Number(v.slice(6)) } : { type: 'frames', kind: v as 'wall' | 'fast' });
};
$('step1').onclick = (): void => { if (!running) send({ type: 'step1' }); };
$('untilGo').onclick = (): void => {
  const h = Number($<HTMLInputElement>('untilH').value);
  if (!(h > 0)) return;
  send({ type: 'runUntil', hours: h });
  running = true; sync();
  log(`執行 ${h} 模式小時後自動暫停 / running for ${h} model hours`);
};
// ---------------- unattended runs
const runner = new UnattendedRun({
  experiment: () => `${curExp}: ${$('desc').textContent ?? ''}`,
  grid: () => $('grid').textContent ?? '',
  start: (cfg) => {
    send({ type: 'pace', target: 0 }); $<HTMLSelectElement>('pace').value = '0';
    send({ type: 'frames', kind: 'model', every: cfg.everyMin * 60 });
    send({ type: 'runUntil', hours: cfg.hours });
    running = true; sync();
    $<HTMLButtonElement>('runStart').disabled = true; $<HTMLButtonElement>('runStop').disabled = false;
  },
  finish: () => {
    $('frames').dispatchEvent(new Event('change'));
    $<HTMLButtonElement>('runStart').disabled = false; $<HTMLButtonElement>('runStop').disabled = true; $<HTMLButtonElement>('runCopy').disabled = false;
  },
  save: async () => { const { meta, data } = await captureSave(); await storeSave(meta, data); await savesPanel.refresh(); return meta.title; },
  log, status: (t) => { $('runStatus').textContent = t; },
});
$('runStart').onclick = (): void => {
  const hours = Number($<HTMLInputElement>('runH').value);
  if (!(hours > 0)) return;
  void runner.begin({ hours, everyMin: Number($<HTMLSelectElement>('runEvery').value), note: $<HTMLInputElement>('runNote').value.slice(0, 200), autoSave: $<HTMLInputElement>('runSave').checked });
};
$('runStop').onclick = (): void => { send({ type: 'runUntil', hours: 0 }); running = false; sync(); void runner.abort('stopped'); };
$('runCopy').onclick = async (): Promise<void> => {
  try { await navigator.clipboard.writeText(runner.reportText()); log('已複製報告，可以貼給開發者 / report copied'); }
  catch { log('無法存取剪貼簿 / clipboard unavailable'); }
};
