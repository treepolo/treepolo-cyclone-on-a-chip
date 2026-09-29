// Regional-model page: controls, 3-D volume view, statistics.
import { VolumeView } from './volume.js';
import { sequential, diverging } from '../colormap.js';
import { mountSavesPanel, storeSave, type SaveMeta } from '../saves.js';
import { UnattendedRun } from './runner.js';
import { RegionalCharts } from './charts.js';
import { Missions } from './missions.js';
import { SetupForm } from './setupForm.js';
import type { RegionalSetup } from './setup.js';
import type { StormNow } from '../../regional/storms.js';
import { Tools3D } from './tools3d.js';
import { ReplayStore, type ReplayFrame } from './replay.js';
import type { FromRegionalWorker, GroundField, NestPayload, NestSize, RegionalExperiment, ToRegionalWorker } from './protocol.js';

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
// replay of the 3-D view: frames kept by the page (memory budget from the device memory when the browser tells it)
const replay = new ReplayStore(Math.min(400, 64 * ((navigator as unknown as { deviceMemory?: number }).deviceMemory ?? 4)) * 1e6);
let replayIdx: number | null = null, replayPlaying = false, replayAcc = 0;
/** Show a stored or live display volume in the 3-D view. */
function showVolume(f: ReplayFrame): void {
  view.setVolume(f.nx, f.ny, f.nz, f.cloud, f.rain, f.aux, Math.min(0.8, f.top / f.Lx * (exag ?? autoExag(f.Lx))), f.top, f.Lx / f.nx);
  view.setGround(f.nx, f.ny, f.ground);
}
const charts = new RegionalCharts($<HTMLCanvasElement>('chart'), $('chartBar'), {
  request: (req) => send({ type: 'charts', req }),
  volMode: (mode) => { send({ type: 'volMode', mode }); view.setMode(mode); },
  tracers: (n) => send({ type: 'tracers', n }),
  interact: (kind, x, y, radius) => {
    if (kind === 'warm' || kind === 'cold') send({ type: 'perturb', kind, x, y });
    else if (kind !== 'inspect') send({ type: 'paint', kind, x, y, radius });
  },
  camera: (mode) => { view.setCamera(mode); $('flyPad').hidden = mode !== 'fly'; },
  view: (v) => {
    $('view').hidden = v !== '3d'; $('chart').hidden = v === '3d';
    if (v === '3d') { const f = replayIdx !== null ? replay.frames[replayIdx] : lastVol; if (f) showVolume(f); }
  },
});

worker.onmessage = (ev: MessageEvent<FromRegionalWorker>): void => {
  const m = ev.data;
  if (m.type === 'ready') {
    dt = m.dt;
    land = m.land;
    // vertical exaggeration so that the troposphere is visible (unless chosen)
    if (exag === null) { const e = autoExag(m.nx * m.dx); $<HTMLInputElement>('exag').value = String(e); $('exagV').textContent = `${e}×`; }
    $('grid').textContent = `${m.nx}×${m.ny}×${m.nz}, Δx ${m.dx >= 1000 ? `${(m.dx / 1000).toFixed(1)} km` : `${m.dx.toFixed(0)} m`}, Δz ${m.dz.toFixed(0)} m, Δt ${m.dt} s`;
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
    $<HTMLButtonElement>('eyeGo').disabled = !m.eyeOk;
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
    if (!refining) { replay.clear(); endReplay(); }
    refining = false;
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
    lastVol = { t: m.time, nx: m.nx, ny: m.ny, nz: m.nz, Lx: m.nx * m.dx, Ly: m.ny * m.dx, top: m.nz * m.dz, cloud: m.cloud, rain: m.rain, aux: m.aux, ground: rgba, storms: m.stats.storms ?? [] };
    replay.push(lastVol);
    if (in3d && replayIdx === null) { showVolume(lastVol); view.setTracers(m.tracers, m.nx * m.dx, m.ny * m.dx, m.nz * m.dz); }
    replayBar();
    const s = m.stats, t = m.time;
    $('time').textContent = t < 7200 * 3 ? `${(t / 60).toFixed(0)} min` : `${(t / 3600).toFixed(1)} h (${(t / 86400).toFixed(2)} d)`;
    $('ovTime').textContent = t < 7200 * 3 ? `${(t / 60).toFixed(0)} min` : `${(t / 3600).toFixed(1)} h`;
    const ms = m.stepsPerSecond * m.dt;
    $('rate').textContent = `${m.stepsPerSecond.toFixed(m.stepsPerSecond < 10 ? 2 : 1)} 步/s steps/s · ${ms < 60 ? `${ms.toFixed(1)} 模式秒/s model-s/s` : `${(ms / 60).toFixed(1)} 模式分/s model-min/s`} · Δt ${m.dt.toFixed(1)} s`;
    $('w').textContent = `${s.wmin.toFixed(1)} … ${s.wmax.toFixed(1)} m/s`;
    $('qc').textContent = `${(s.qcmax * 1000).toFixed(2)} g/kg`;
    $('qr').textContent = `${(s.qrmax * 1000).toFixed(2)} g/kg`;
    $('rainmax').textContent = `${s.rainmax.toFixed(1)} mm`;
    $('vmax').textContent = `${s.vmax.toFixed(1)} m/s`;
    $('dp').textContent = s.dp === null ? '—' : `${s.dp.toFixed(1)} hPa${s.rmw ? ` · RMW ${(s.rmw / 1000).toFixed(0)} km` : ''}`;
    $('eyewalls').textContent = !s.eyewalls ? '—' : s.eyewalls.length === 0 ? '未形成 / none yet'
      : (s.eyewalls.length >= 2 ? '雙眼牆 / concentric: ' : '') + s.eyewalls.map((e) => `${(e.r / 1000).toFixed(0)} km (${e.v.toFixed(0)} m/s)`).join(' · ');
    $('zeta').textContent = `${s.zetaMax.toFixed(3)} s⁻¹ · ${s.vGround.toFixed(1)} m/s`;
    $('tcrain').textContent = !s.tcRain ? '—' : `${s.tcRain.core.toFixed(1)} · ${s.tcRain.outer.toFixed(2)} mm/h（外圍 >1 mm/h ${(100 * s.tcRain.wet).toFixed(1)}%）`;
    tornadoWatch(m.time, s.tornado, m.dx);
    showStorms(s.storms ?? [], s.mainId ?? null, m);
    $('legend').textContent = m.groundField === 'none' ? '深藍 = 海、深綠 = 陸地 / dark blue = sea, dark green = land' : `${lo.toFixed(1)} … ${hi.toFixed(1)} ${m.groundField === 'rain' || m.groundField === 'snow' ? 'mm' : m.groundField === 'wind' ? 'm/s' : 'K'}`;
  } else if (m.type === 'error') { log(`錯誤 / Error: ${m.message}`); running = false; sync(); if (runner.running) void runner.abort(`error: ${m.message}`); }
  else if (m.type === 'saveData') { pendingSave?.({ meta: m.meta, data: m.buffer }); pendingSave = null; }
  else if (m.type === 'paused') { log(m.reason); running = false; $('run').textContent = '執行 / Run'; $('ovRun').textContent = '▶'; if (runner.running) void runner.end('done'); }
  else if (m.type === 'log') { log(m.text); if (/no vortex|eye box too large/.test(m.text)) { refining = false; $<HTMLButtonElement>('eyeGo').disabled = false; } }
  else if (m.type === 'land') { land = m.land; charts.setLand(m.land); showSurface(); }
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
  const host = $('stormLabels'), rf = replayIdx !== null ? replay.frames[replayIdx] ?? null : null;
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
  if (!Number.isFinite(du6) || !(humidity > 0)) return;
  send({ type: 'environment', du6: Math.max(-30, Math.min(30, du6)), humidity: Math.max(0.3, Math.min(2, humidity)) });
};
// remember which sections are open (per-viewer convenience)
document.querySelectorAll<HTMLDetailsElement>('details[id]').forEach((d) => {
  try { const v = localStorage.getItem(`regional-${d.id}`); if (v !== null) d.open = v === '1'; } catch { /* storage unavailable */ }
  d.addEventListener('toggle', () => { try { localStorage.setItem(`regional-${d.id}`, d.open ? '1' : '0'); } catch { /* storage unavailable */ } });
});
$('backendSel').onchange = init;
$('ground').onchange = (): void => send({ type: 'ground', field: $<HTMLSelectElement>('ground').value as GroundField });
$('subgrid').onchange = (): void => send({ type: 'subgrid', on: $<HTMLInputElement>('subgrid').checked });
$('exag').oninput = (): void => {
  exag = Number($<HTMLInputElement>('exag').value); $('exagV').textContent = `${exag}×`;
  const f = replayIdx !== null ? replay.frames[replayIdx] : lastVol;
  if (f && charts.view === '3d') showVolume(f);
};
$('fov').oninput = (): void => { const d = Number($<HTMLInputElement>('fov').value); view.setFov(d); $('fovV').textContent = `${d}°`; };
$('speed').oninput = (): void => send({ type: 'speed', stepsPerTick: Number($<HTMLInputElement>('speed').value) });
$('adaptive').onchange = (): void => send({ type: 'adaptive', on: $<HTMLInputElement>('adaptive').checked });
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
// ---------------- replay of the 3-D view
const fmtT = (t: number): string => (t < 7200 * 3 ? `${(t / 60).toFixed(0)} min` : `${(t / 3600).toFixed(1)} h`);
function replayBar(): void {
  const n = replay.length, sl = $<HTMLInputElement>('rpSlider');
  sl.max = String(Math.max(0, n - 1));
  if (replayIdx === null) return;
  replayIdx = Math.min(replayIdx, n - 1);
  sl.value = String(replayIdx);
  const f = replay.frames[replayIdx];
  $('rpTime').textContent = f ? `${fmtT(f.t)}（${replayIdx + 1}/${n}）` : '—';
}
function showReplay(i: number): void {
  if (!replay.length) return;
  replayIdx = Math.max(0, Math.min(replay.length - 1, i));
  const f = replay.frames[replayIdx]!;
  if (charts.view === '3d') { showVolume(f); view.setTracers(null, f.Lx, f.Ly, f.top); }
  replayBar();
}
function endReplay(): void {
  replayIdx = null; replayPlaying = false; $('replayBar').hidden = true; $('rpPlay').textContent = '▶';
  if (lastVol && charts.view === '3d') showVolume(lastVol);
}
$('ovReplay').onclick = (): void => {
  if (replayIdx !== null) { endReplay(); return; }
  if (!replay.length) { log('還沒有可以重播的畫面 / nothing to replay yet'); return; }
  $('replayBar').hidden = false; replayPlaying = false; showReplay(0);
  log(`重播：${replay.length} 個畫面，${replay.megabytes.toFixed(0)} MB / replay: ${replay.length} frames`);
};
$('rpSlider').oninput = (): void => { replayPlaying = false; $('rpPlay').textContent = '▶'; showReplay(Number($<HTMLInputElement>('rpSlider').value)); };
$('rpPlay').onclick = (): void => {
  if (replayIdx === null) return;
  if (!replayPlaying && replayIdx >= replay.length - 1) showReplay(0);
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
      if (next >= replay.length - 1) { showReplay(replay.length - 1); replayPlaying = false; $('rpPlay').textContent = '▶'; } else showReplay(next);
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
/** cells of the eye box and its time step */
function eyeInfo(): void {
  const L = Number($<HTMLSelectElement>('eyeL').value), dx = Number($<HTMLSelectElement>('eyeDx').value), dz = Number($<HTMLSelectElement>('eyeDz').value), top = form.value().top;
  const n = Math.round(L / dx) ** 2 * Math.round(top / dz);
  $('eyeInfo').textContent = `${Math.round(L / dx)}×${Math.round(L / dx)}×${Math.round(top / dz)} = ${(n / 1e6).toFixed(1)} M 格點 / cells${n > 8e6 ? ' · ⚠ 需要較強的顯卡 / needs a strong GPU' : ''}`;
}
for (const id of ['eyeL', 'eyeDx', 'eyeDz']) $(id).onchange = eyeInfo;
$('eyeGo').onclick = (): void => {
  $<HTMLButtonElement>('eyeGo').disabled = true; refining = true; log('眼區細化中… / Refining the eye…');
  send({ type: 'refineEye', L: Number($<HTMLSelectElement>('eyeL').value), dx: Number($<HTMLSelectElement>('eyeDx').value), dz: Number($<HTMLSelectElement>('eyeDz').value) });
};
$('profile').onclick = (): void => { $<HTMLButtonElement>('profile').disabled = true; $('profileOut').textContent = '量測中… / Measuring…'; worker.postMessage({ type: 'profile' } satisfies ToRegionalWorker); };
$('refine').onclick = (): void => { $<HTMLButtonElement>('refine').disabled = true; refining = true; log('細化中… / Refining…'); send({ type: 'refine' }); };
// ---------------- saved simulations
let pendingSave: ((r: { meta: SaveMeta; data: ArrayBuffer }) => void) | null = null;
const savesPanel = mountSavesPanel($('saves'), 'regional',
  () => new Promise((resolve, reject) => {
    if (pendingSave) { reject(new Error('存檔進行中 / a save is already in progress')); return; }
    pendingSave = resolve; send({ type: 'save' });
    setTimeout(() => { if (pendingSave === resolve) { pendingSave = null; reject(new Error('逾時 / timed out')); } }, 120000);
  }),
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
const captureSave = (): Promise<{ meta: SaveMeta; data: ArrayBuffer }> => new Promise((resolve, reject) => {
  if (pendingSave) { reject(new Error('存檔進行中 / a save is already in progress')); return; }
  pendingSave = resolve; send({ type: 'save' });
  setTimeout(() => { if (pendingSave === resolve) { pendingSave = null; reject(new Error('逾時 / timed out')); } }, 120000);
});
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
