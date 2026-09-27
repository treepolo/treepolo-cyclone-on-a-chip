// Regional-model page: controls, 3-D volume view, statistics.
import { VolumeView } from './volume.js';
import { sequential, diverging } from '../colormap.js';
import type { FromRegionalWorker, GroundField, NestPayload, NestSize, RegionalExperiment, ToRegionalWorker } from './protocol.js';

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
    $('grid').textContent = `${m.nx}×${m.ny}×${m.nz}, Δx ${(m.dx / 1000).toFixed(1)} km, Δz ${m.dz.toFixed(0)} m, Δt ${m.dt} s`;
    $('backend').textContent = m.backend === 'gpu' ? 'WebGPU（f32）' : 'CPU（Float64）';
    if (m.note) log(m.note);
    $('desc').textContent = m.description;
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
    $('rate').textContent = `${m.stepsPerSecond.toFixed(1)} 步/s steps/s · ${(m.stepsPerSecond * dt / 60).toFixed(1)} 模式分/s model-min/s`;
    $('w').textContent = `${s.wmin.toFixed(1)} … ${s.wmax.toFixed(1)} m/s`;
    $('qc').textContent = `${(s.qcmax * 1000).toFixed(2)} g/kg`;
    $('qr').textContent = `${(s.qrmax * 1000).toFixed(2)} g/kg`;
    $('rainmax').textContent = `${s.rainmax.toFixed(1)} mm`;
    $('vmax').textContent = `${s.vmax.toFixed(1)} m/s`;
    $('dp').textContent = s.dp === null ? '—' : `${s.dp.toFixed(1)} hPa${s.rmw ? ` · RMW ${(s.rmw / 1000).toFixed(0)} km` : ''}`;
    $('legend').textContent = `${lo.toFixed(1)} … ${hi.toFixed(1)} ${m.groundField === 'rain' || m.groundField === 'snow' ? 'mm' : m.groundField === 'wind' ? 'm/s' : 'K'}`;
  } else if (m.type === 'error') { log(`錯誤 / Error: ${m.message}`); running = false; sync(); }
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
// Nesting: opened from the global page with ?nest=1; the opener posts the global state and the chosen point.
if (new URLSearchParams(location.search).has('nest')) {
  const sel = $<HTMLSelectElement>('exp'), opt = document.createElement('option');
  opt.value = 'nest'; opt.textContent = '全球模式巢狀區域 / Nest in the global model';
  sel.prepend(opt); sel.value = 'nest';
  window.addEventListener('message', (ev: MessageEvent) => {
    if (ev.origin !== location.origin || !ev.data || ev.data.type !== 'nest') return;
    nest = { payload: ev.data.payload as NestPayload, lat0: ev.data.lat0 as number, lon0: ev.data.lon0 as number, size: (ev.data.size as NestSize) ?? 'meso' };
    sel.value = 'nest';
    init();
  });
  if (window.opener) (window.opener as Window).postMessage({ type: 'nest-ready' }, location.origin);
  else log('找不到全球模式視窗；請從全球模式頁面點選地點後開啟 / No global-model window found; open this page from the global model after picking a point');
}
init();
function tick(): void { view.render(Number($<HTMLInputElement>('cloudK').value), Number($<HTMLInputElement>('cloudK').value) * 1.5); requestAnimationFrame(tick); }
requestAnimationFrame(tick);
