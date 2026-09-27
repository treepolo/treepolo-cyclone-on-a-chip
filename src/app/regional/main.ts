// Regional-model page: controls, 3-D volume view, statistics.
import { VolumeView } from './volume.js';
import { sequential, diverging } from '../colormap.js';
import type { FromRegionalWorker, GroundField, RegionalExperiment, ToRegionalWorker } from './protocol.js';

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const log = (s: string): void => { const el = $('log'); el.textContent = `${s}\n${el.textContent ?? ''}`.slice(0, 3000); };
const view = new VolumeView($<HTMLCanvasElement>('view'));
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const send = (m: ToRegionalWorker): void => worker.postMessage(m);
let running = false, dt = 6, aspect = 0.3;

worker.onmessage = (ev: MessageEvent<FromRegionalWorker>): void => {
  const m = ev.data;
  if (m.type === 'ready') {
    dt = m.dt;
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
      else if (m.groundField === 'rain') { const t = Math.sqrt(Math.max(0, v) / hi); c = t < 0.02 ? [0.16, 0.22, 0.16] : sequential(0.15 + 0.85 * t); }
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
    $('legend').textContent = `${lo.toFixed(1)} … ${hi.toFixed(1)} ${m.groundField === 'rain' ? 'mm' : m.groundField === 'wind' ? 'm/s' : 'K'}`;
  } else if (m.type === 'error') { log(`錯誤 / Error: ${m.message}`); running = false; sync(); }
};
function sync(): void { $('run').textContent = running ? '暫停 / Pause' : '執行 / Run'; send({ type: 'run', running }); }
$('run').onclick = (): void => { running = !running; sync(); };
const init = (): void => { running = false; sync(); send({ type: 'init', experiment: $<HTMLSelectElement>('exp').value as RegionalExperiment, backend: $<HTMLSelectElement>('backendSel').value as 'auto' | 'cpu' }); };
$('exp').onchange = init;
$('backendSel').onchange = init;
$('ground').onchange = (): void => send({ type: 'ground', field: $<HTMLSelectElement>('ground').value as GroundField });
$('speed').oninput = (): void => send({ type: 'speed', stepsPerTick: Number($<HTMLInputElement>('speed').value) });
init();
function tick(): void { view.render(Number($<HTMLInputElement>('cloudK').value), Number($<HTMLInputElement>('cloudK').value) * 1.5); requestAnimationFrame(tick); }
requestAnimationFrame(tick);
