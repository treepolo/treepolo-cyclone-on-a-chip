// Regional-model page: controls, 3-D volume view, statistics.
import { VolumeView } from './volume.js';
import { sequential, diverging } from '../colormap.js';
import { mountSavesPanel, storeSave, type SaveMeta } from '../saves.js';
import { UnattendedRun } from './runner.js';
import { RegionalCharts } from './charts.js';
import { Missions } from './missions.js';
import type { AxiParams } from './axiDriver.js';
import { quarterCircleWind, bunkersRightMover, type TornadoEnv } from '../../regional/supercell.js';
import { weismanKlempQ } from '../../regional/kessler.js';
import { parcelAscent } from '../../regional/diagnostics.js';
import type { FromRegionalWorker, GroundField, NestPayload, NestSize, RegionalExperiment, ToRegionalWorker } from './protocol.js';

const REFINE_LABEL: Partial<Record<RegionalExperiment, string>> = { supercell_hr: '1 km', tc_hr: '5 km', tc_3: '3 km', tornado: '250 m' };

// Opened on its own without the artifact runtime: regional.html redirects to the main page (see the
// inline script there); stop here instead of starting a model that is about to be unloaded.
if (window.parent === window && !(window as unknown as { claude?: unknown }).claude && !/[?&]standalone/.test(location.search)) await new Promise(() => {});
const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const log = (s: string): void => { const el = $('log'); el.textContent = `${s}\n${el.textContent ?? ''}`.slice(0, 3000); };
const view = new VolumeView($<HTMLCanvasElement>('view'));
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const send = (m: ToRegionalWorker): void => worker.postMessage(m);
let running = false, dt = 6, aspect = 0.3;
let land: Uint8Array | null = null;
let nest: { payload: NestPayload; lat0: number; lon0: number; size: NestSize } | null = null;
// charts: the main view is either the 3-D volume or one of the 2-D charts
const missions = new Missions($('missions'), (s) => log(s));
let curExp: RegionalExperiment = 'supercell';
let lastVol: { nx: number; ny: number; nz: number; cloud: Uint8Array; rain: Uint8Array; ground: Uint8Array } | null = null, refining = false;
const charts = new RegionalCharts($<HTMLCanvasElement>('chart'), $('chartBar'), {
  request: (req) => send({ type: 'charts', req }),
  volMode: (mode) => send({ type: 'volMode', mode }),
  tracers: (n) => send({ type: 'tracers', n }),
  interact: (kind, x, y, radius) => {
    if (kind === 'warm' || kind === 'cold') send({ type: 'perturb', kind, x, y });
    else if (kind !== 'inspect') send({ type: 'paint', kind, x, y, radius });
  },
  camera: (mode) => view.setCamera(mode),
  view: (v) => {
    $('view').hidden = v !== '3d'; $('chart').hidden = v === '3d';
    if (v === '3d' && lastVol) { view.setVolume(lastVol.nx, lastVol.ny, lastVol.nz, lastVol.cloud, lastVol.rain, aspect); view.setGround(lastVol.nx, lastVol.ny, lastVol.ground); }
  },
});

worker.onmessage = (ev: MessageEvent<FromRegionalWorker>): void => {
  const m = ev.data;
  if (m.type === 'ready') {
    dt = m.dt;
    land = m.land;
    // vertical exaggeration so that the troposphere is visible
    aspect = Math.min(0.45, (m.nz * m.dz) / (m.nx * m.dx) * (m.nx * m.dx > 500000 ? 12 : 2.5));
    $('grid').textContent = `${m.nx}×${m.ny}×${m.nz}, Δx ${m.dx >= 1000 ? `${(m.dx / 1000).toFixed(1)} km` : `${m.dx.toFixed(0)} m`}, Δz ${m.dz.toFixed(0)} m, Δt ${m.dt} s`;
    $('backend').textContent = m.backend === 'gpu' ? 'WebGPU（f32）' : 'CPU（Float64）';
    if (m.note) log(m.note);
    $('desc').textContent = m.description;
    const rb = $<HTMLButtonElement>('refine');
    rb.disabled = !m.refineTo;
    rb.textContent = m.refineTo ? `細化到 ${REFINE_LABEL[m.refineTo] ?? ''} / Refine to ${REFINE_LABEL[m.refineTo] ?? ''}` : '細化 / Refine';
    if ((m.experiment as string) !== 'nest') $<HTMLSelectElement>('exp').value = m.experiment;
    tcPanel(m.experiment);
    charts.setGrid({ nx: m.nx, ny: m.ny, nz: m.nz, dx: m.dx, dy: m.dx, dz: m.dz, experiment: m.experiment, land: m.land }, refining);
    curExp = m.experiment; missions.reset(m.experiment);
    refining = false;
    log(`就緒 / Ready: ${m.description}`);
  } else if (m.type === 'frame') {
    runner.sample(m);
    charts.onFrame(m);
    missions.check(m, curExp);
    const in3d = charts.view === '3d';
    if (in3d) { view.setVolume(m.nx, m.ny, m.nz, m.cloud, m.rain, aspect); view.setTracers(m.tracers, m.nx * m.dx, m.ny * m.dx, m.nz * m.dz); }
    const [lo, hi] = m.groundRange, rgba = new Uint8Array(m.nx * m.ny * 4);
    for (let i = 0; i < m.ground.length; i++) {
      const v = m.ground[i]!;
      let c: [number, number, number];
      if (m.groundField === 'theta') c = diverging(v / hi);
      else if (m.groundField === 'rain' || m.groundField === 'snow') { const t = Math.sqrt(Math.max(0, v) / hi); c = t < 0.02 ? (land && !land[i] ? [0.10, 0.17, 0.30] : [0.16, 0.22, 0.16]) : sequential(0.15 + 0.85 * t); }
      else c = sequential((v - lo) / ((hi - lo) || 1));
      rgba[4 * i] = c[0] * 255; rgba[4 * i + 1] = c[1] * 255; rgba[4 * i + 2] = c[2] * 255; rgba[4 * i + 3] = 255;
    }
    if (in3d) view.setGround(m.nx, m.ny, rgba);
    lastVol = { nx: m.nx, ny: m.ny, nz: m.nz, cloud: m.cloud, rain: m.rain, ground: rgba };
    const s = m.stats, t = m.time;
    $('time').textContent = t < 7200 * 3 ? `${(t / 60).toFixed(0)} min` : `${(t / 3600).toFixed(1)} h (${(t / 86400).toFixed(2)} d)`;
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
    $('legend').textContent = `${lo.toFixed(1)} … ${hi.toFixed(1)} ${m.groundField === 'rain' || m.groundField === 'snow' ? 'mm' : m.groundField === 'wind' ? 'm/s' : 'K'}`;
  } else if (m.type === 'error') { log(`錯誤 / Error: ${m.message}`); running = false; sync(); if (runner.running) void runner.abort(`error: ${m.message}`); }
  else if (m.type === 'saveData') { pendingSave?.({ meta: m.meta, data: m.buffer }); pendingSave = null; }
  else if (m.type === 'paused') { log(m.reason); running = false; $('run').textContent = '執行 / Run'; if (runner.running) void runner.end('done'); }
  else if (m.type === 'log') log(m.text);
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
function sync(): void { $('run').textContent = running ? '暫停 / Pause' : '執行 / Run'; send({ type: 'run', running }); }
$('run').onclick = (): void => { running = !running; sync(); };
const init = (): void => {
  running = false; sync();
  const exp = $<HTMLSelectElement>('exp').value as RegionalExperiment, backend = $<HTMLSelectElement>('backendSel').value as 'auto' | 'cpu';
  if (exp === 'nest') {
    if (!nest) { log('等待全球模式傳送資料… / Waiting for the global model state…'); return; }
    log('建立巢狀區域模式中 / Building the nested regional model…');
    send({ type: 'initNest', payload: nest.payload, lat0: nest.lat0, lon0: nest.lon0, size: nest.size, backend });
  } else if (exp === 'tc_axi' || exp === 'tc' || exp === 'tc_hr' || exp === 'tc_3') send({ type: 'init', experiment: exp, backend, axi: axiParams() });
  else if (exp === 'tornado' || exp === 'tornado_c') send({ type: 'init', experiment: exp, backend, tornado: tornadoParams() });
  else send({ type: 'init', experiment: exp, backend });
};
/** Parameters of the axisymmetric experiment from the panel. */
function axiParams(): AxiParams {
  const num = (id: string, d: number): number => { const v = Number($<HTMLInputElement>(id).value); return Number.isFinite(v) ? v : d; };
  const lat = Math.max(1, Math.min(60, Math.abs(num('axLat', 20))));
  return { sst: num('axSst', 28) + 273.15, dr: num('axDr', 2000), lh: Math.max(0, num('axLh', 1000)), lv: Math.max(0, num('axLv', 100)), ck: Math.max(0, num('axCk', 1.2)) * 1e-3,
    vmin: Math.max(0, num('axVmin', 1)), vmax0: Math.max(1, num('axV0', 15)), f: 2 * 7.292e-5 * Math.sin(lat * Math.PI / 180),
    radMax: Math.max(0, num('axRad', 2)), radConst: $<HTMLSelectElement>('axRadMode').value === 'const' ? Math.max(0, num('axRad', 1)) : 0, rhTop: Math.max(0.05, Math.min(1, num('axRh', 40) / 100)) };
}
/** Tornado environment from the panel, and its 0-1 / 0-3 km storm-relative helicity, shear and CAPE. */
function tornadoParams(): TornadoEnv {
  const num = (id: string, d: number): number => { const v = Number($<HTMLInputElement>(id).value); return Number.isFinite(v) ? v : d; };
  return { R: Math.max(0, num('toR', 10)), depth: Math.max(250, Math.min(5500, 1000 * num('toDepth', 2))), U6: num('toU6', 30), qvMax: Math.max(0.008, Math.min(0.02, num('toQ', 14) / 1000)) };
}
function tornadoInfo(): void {
  const e = tornadoParams(), h = { R: e.R, U6: e.U6, depth: e.depth }, rm = bunkersRightMover((z) => quarterCircleWind(z, h));
  const srh = (top: number): number => { let s = 0; const n = 200; for (let i = 0; i < n; i++) { const a = quarterCircleWind(i * top / n, h), b = quarterCircleWind((i + 1) * top / n, h); s += (b.u - rm.u) * (a.v - rm.v) - (a.u - rm.u) * (b.v - rm.v); } return s; };
  const snd = weismanKlempQ(e.qvMax), nz = 80, dz = 250, T = new Float64Array(nz), p = new Float64Array(nz), q = new Float64Array(nz);
  // WK82 is tabulated as theta and qv: integrate the Exner function hydrostatically from 1000 hPa
  let pi = 1;
  for (let k = 0; k < nz; k++) { const z = (k + 0.5) * dz, s = snd(z); if (k > 0) { const sb = snd(z - dz); pi -= 9.80665 / (1004.5 * 0.5 * (s.theta * (1 + 0.61 * s.qv) + sb.theta * (1 + 0.61 * sb.qv))) * dz; } else pi -= 9.80665 / (1004.5 * s.theta * (1 + 0.61 * s.qv)) * 0.5 * dz; T[k] = s.theta * pi; p[k] = 1e5 * Math.pow(pi, 1004.5 / 287.05); q[k] = s.qv; }
  const pc = parcelAscent(T, p, q, dz);
  $('toInfo').textContent = `SRH 0–1 km ${srh(1000).toFixed(0)}、0–3 km ${srh(3000).toFixed(0)} m²/s² · CAPE ${pc.cape.toFixed(0)} J/kg · LCL ${pc.lcl >= 0 ? ((pc.lcl + 0.5) * dz / 1000).toFixed(1) : '—'} km`;
}
for (const id of ['toR', 'toDepth', 'toU6', 'toQ']) $(id).oninput = tornadoInfo;
tornadoInfo();
$('envApply').onclick = (): void => {
  const du6 = Number($<HTMLInputElement>('envDu').value), humidity = Number($<HTMLInputElement>('envRh').value);
  if (!Number.isFinite(du6) || !(humidity > 0)) return;
  send({ type: 'environment', du6: Math.max(-30, Math.min(30, du6)), humidity: Math.max(0.3, Math.min(2, humidity)) });
};
$('toApply').onclick = (): void => { if (/^tornado/.test($<HTMLSelectElement>('exp').value)) init(); };
$('axApply').onclick = (): void => { if (/^tc/.test($<HTMLSelectElement>('exp').value)) init(); };
const tcPanel = (e: string): void => { $('tornadoPanel').hidden = !/^tornado/.test(e); $('axiPanel').hidden = !/^tc/.test(e); document.querySelectorAll<HTMLElement>('.axOnly').forEach((el) => { el.hidden = e !== 'tc_axi'; }); };
$('exp').onchange = (): void => { tcPanel($<HTMLSelectElement>('exp').value); init(); };
$('backendSel').onchange = init;
$('ground').onchange = (): void => send({ type: 'ground', field: $<HTMLSelectElement>('ground').value as GroundField });
$('speed').oninput = (): void => send({ type: 'speed', stepsPerTick: Number($<HTMLInputElement>('speed').value) });
$('adaptive').onchange = (): void => send({ type: 'adaptive', on: $<HTMLInputElement>('adaptive').checked });
// Nesting: embedded by the global page (in-page overlay, #nest...) or opened with ?nest=1; the
// parent/opener posts the global state and the chosen point.
const host = window.parent !== window ? window.parent : (window.opener as Window | null);
if (location.hash.startsWith('#nest') || new URLSearchParams(location.search).has('nest')) {
  const sel = $<HTMLSelectElement>('exp'), opt = document.createElement('option');
  opt.value = 'nest'; opt.textContent = '全球模式巢狀區域 / Nest in the global model';
  sel.prepend(opt); sel.value = 'nest';
  window.addEventListener('message', (ev: MessageEvent) => {
    if (!host || ev.source !== host || !ev.data || ev.data.type !== 'nest') return;
    nest = { payload: ev.data.payload as NestPayload, lat0: ev.data.lat0 as number, lon0: ev.data.lon0 as number, size: (ev.data.size as NestSize) ?? 'meso' };
    sel.value = 'nest';
    init();
  });
  if (host) host.postMessage({ type: 'nest-ready' }, '*');
  else log('找不到全球模式視窗；請從全球模式頁面點選地點後開啟 / No global-model window found; open this page from the global model after picking a point');
}
if (window.parent !== window) { const back = document.getElementById('backLink'); if (back) back.hidden = true; }
init();
function tick(): void { if (charts.view === '3d') view.render(Number($<HTMLInputElement>('cloudK').value), Number($<HTMLInputElement>('cloudK').value) * 1.5); requestAnimationFrame(tick); }
requestAnimationFrame(tick);
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
  (_meta, data) => { running = false; sync(); send({ type: 'load', buffer: data.slice(0), backend: $<HTMLSelectElement>('backendSel').value as 'auto' | 'cpu' }); },
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
  experiment: () => $<HTMLSelectElement>('exp').value,
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
