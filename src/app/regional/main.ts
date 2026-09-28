// Regional-model page: controls, 3-D volume view, statistics.
import { VolumeView } from './volume.js';
import { sequential, diverging } from '../colormap.js';
import { mountSavesPanel, type SaveMeta } from '../saves.js';
import type { FromRegionalWorker, GroundField, NestPayload, NestSize, RegionalExperiment, ToRegionalWorker } from './protocol.js';

const REFINE_LABEL: Partial<Record<RegionalExperiment, string>> = { supercell_hr: '1 km', tc_hr: '5 km', tornado: '250 m' };

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const log = (s: string): void => { const el = $('log'); el.textContent = `${s}\n${el.textContent ?? ''}`.slice(0, 3000); };
const view = new VolumeView($<HTMLCanvasElement>('view'));
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const send = (m: ToRegionalWorker): void => worker.postMessage(m);
let running = false, dt = 6, aspect = 0.3;
let land: Uint8Array | null = null;
let nest: { payload: NestPayload; lat0: number; lon0: number; size: NestSize } | null = null;

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
    log(`就緒 / Ready: ${m.description}`);
  } else if (m.type === 'frame') {
    view.setVolume(m.nx, m.ny, m.nz, m.cloud, m.rain, aspect);
    const [lo, hi] = m.groundRange, rgba = new Uint8Array(m.nx * m.ny * 4);
    for (let i = 0; i < m.ground.length; i++) {
      const v = m.ground[i]!;
      let c: [number, number, number];
      if (m.groundField === 'theta') c = diverging(v / hi);
      else if (m.groundField === 'rain' || m.groundField === 'snow') { const t = Math.sqrt(Math.max(0, v) / hi); c = t < 0.02 ? (land && !land[i] ? [0.10, 0.17, 0.30] : [0.16, 0.22, 0.16]) : sequential(0.15 + 0.85 * t); }
      else c = sequential((v - lo) / ((hi - lo) || 1));
      rgba[4 * i] = c[0] * 255; rgba[4 * i + 1] = c[1] * 255; rgba[4 * i + 2] = c[2] * 255; rgba[4 * i + 3] = 255;
    }
    view.setGround(m.nx, m.ny, rgba);
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
    $('legend').textContent = `${lo.toFixed(1)} … ${hi.toFixed(1)} ${m.groundField === 'rain' || m.groundField === 'snow' ? 'mm' : m.groundField === 'wind' ? 'm/s' : 'K'}`;
  } else if (m.type === 'error') { log(`錯誤 / Error: ${m.message}`); running = false; sync(); }
  else if (m.type === 'saveData') { pendingSave?.({ meta: m.meta, data: m.buffer }); pendingSave = null; }
  else if (m.type === 'paused') { log(m.reason); running = false; $('run').textContent = '執行 / Run'; }
  else if (m.type === 'profile') { $('profileOut').textContent = m.text; $<HTMLButtonElement>('profile').disabled = false; }
};
function sync(): void { $('run').textContent = running ? '暫停 / Pause' : '執行 / Run'; send({ type: 'run', running }); }
$('run').onclick = (): void => { running = !running; sync(); };
const init = (): void => {
  running = false; sync();
  const exp = $<HTMLSelectElement>('exp').value as RegionalExperiment, backend = $<HTMLSelectElement>('backendSel').value as 'auto' | 'cpu';
  if (exp === 'nest') {
    if (!nest) { log('等待全球模式傳送資料… / Waiting for the global model state…'); return; }
    log('建立巢狀區域模式中 / Building the nested regional model…');
    send({ type: 'initNest', payload: nest.payload, lat0: nest.lat0, lon0: nest.lon0, size: nest.size, backend });
  } else send({ type: 'init', experiment: exp, backend });
};
$('exp').onchange = init;
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
function tick(): void { view.render(Number($<HTMLInputElement>('cloudK').value), Number($<HTMLInputElement>('cloudK').value) * 1.5); requestAnimationFrame(tick); }
requestAnimationFrame(tick);
$('profile').onclick = (): void => { $<HTMLButtonElement>('profile').disabled = true; $('profileOut').textContent = '量測中… / Measuring…'; worker.postMessage({ type: 'profile' } satisfies ToRegionalWorker); };
$('refine').onclick = (): void => { $<HTMLButtonElement>('refine').disabled = true; log('細化中… / Refining…'); send({ type: 'refine' }); };
// ---------------- saved simulations
let pendingSave: ((r: { meta: SaveMeta; data: ArrayBuffer }) => void) | null = null;
mountSavesPanel($('saves'), 'regional',
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
