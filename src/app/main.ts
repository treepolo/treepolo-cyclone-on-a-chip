// UI thread: controls, globe rendering, wind tracers, zonal-mean sections.

import { Globe } from './globe.js';
import { diverging, sequential } from './colormap.js';
import { drawSection } from './section.js';
import type { FieldId, FrameMessage, FromWorker, ToWorker, ZonalMessage } from './protocol.js';

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('globe');
const log = (s: string): void => { const el = $('log'); el.textContent = `${s}\n${el.textContent ?? ''}`.slice(0, 4000); };

let globe: Globe;
try { globe = new Globe(canvas); } catch (e) { log(String(e)); throw e; }

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const send = (m: ToWorker): void => worker.postMessage(m);

let frame: FrameMessage | null = null;
let zonal: ZonalMessage | null = null;
let running = false;

// ---------------- field colouring
const FIELD_STYLE: Record<FieldId, { div: boolean; range?: [number, number]; sym?: number; unit: string }> = {
  T: { div: false, unit: 'K' },
  u: { div: true, sym: 40, unit: 'm/s' },
  v: { div: true, sym: 20, unit: 'm/s' },
  speed: { div: false, range: [0, 50], unit: 'm/s' },
  vor: { div: true, sym: 1e-4, unit: 's⁻¹' },
  div: { div: true, sym: 1e-5, unit: 's⁻¹' },
  ps: { div: false, range: [970, 1030], unit: 'hPa' },
};

function colourise(f: FrameMessage): void {
  const st = FIELD_STYLE[f.field];
  let lo = Infinity, hi = -Infinity, amax = 0;
  for (let i = 0; i < f.scalar.length; i++) { const v = f.scalar[i]!; lo = Math.min(lo, v); hi = Math.max(hi, v); amax = Math.max(amax, Math.abs(v)); }
  if (st.div) {
    const s = Math.max(st.sym ?? amax, 1e-30) * (f.field === 'vor' || f.field === 'div' ? 1 : 1);
    const scale = Math.max(amax * 0.6, s * 0.25);
    globe.setField(f.lat, f.nlon, f.scalar, (v) => diverging(v / scale));
    $('legend').textContent = `±${scale.toPrecision(2)} ${st.unit}`;
  } else {
    const [a, b] = st.range && f.field !== 'ps' ? st.range : [lo, hi];
    globe.setField(f.lat, f.nlon, f.scalar, (v) => sequential((v - a) / ((b - a) || 1)));
    $('legend').textContent = `${a.toFixed(1)} … ${b.toFixed(1)} ${st.unit}`;
  }
}

// ---------------- tracers (massless Lagrangian particles advected by the model wind at the displayed level)
const NTR = 5000, RADIUS = 6.371e6;
const trLat = new Float64Array(NTR), trLon = new Float64Array(NTR), trAge = new Float64Array(NTR);
const trPos = new Float32Array(NTR * 6), trCol = new Float32Array(NTR * 6);
function respawn(i: number): void {
  trLat[i] = Math.asin(2 * Math.random() - 1);
  trLon[i] = Math.random() * 2 * Math.PI;
  trAge[i] = Math.random() * 200;
}
for (let i = 0; i < NTR; i++) respawn(i);

function sampleWind(f: FrameMessage, lat: number, lon: number): [number, number] {
  const nl = f.nlat, nx = f.nlon, L = f.lat;
  let j = 0;
  while (j < nl - 1 && L[j + 1]! > lat) j++;
  const j1 = Math.min(nl - 1, j + 1);
  const wy = j1 === j ? 0 : Math.max(0, Math.min(1, (L[j]! - lat) / (L[j]! - L[j1]!)));
  const x = ((lon / (2 * Math.PI)) * nx + nx) % nx, i0 = Math.floor(x) % nx, i1 = (i0 + 1) % nx, wx = x - Math.floor(x);
  const at = (a: Float32Array, jj: number): number => a[jj * nx + i0]! * (1 - wx) + a[jj * nx + i1]! * wx;
  return [at(f.u, j) * (1 - wy) + at(f.u, j1) * wy, at(f.v, j) * (1 - wy) + at(f.v, j1) * wy];
}

function updateTracers(dtWall: number): void {
  if (!frame || !($<HTMLInputElement>('tracers')).checked) { globe.setTracers(trPos, trCol, 0); return; }
  const f = frame, dtModel = dtWall * 3 * 3600; // display: 3 model hours per wall second for tracer motion
  let n = 0;
  for (let i = 0; i < NTR; i++) {
    let [u, v] = sampleWind(f, trLat[i]!, trLon[i]!);
    const c = Math.max(0.05, Math.cos(trLat[i]!));
    trLon[i] = trLon[i]! + u * dtModel / (RADIUS * c);
    trLat[i] = Math.max(-1.55, Math.min(1.55, trLat[i]! + v * dtModel / RADIUS));
    trAge[i] = trAge[i]! + 1;
    if (trAge[i]! > 240 || Math.abs(trLat[i]!) > 1.5) respawn(i);
    [u, v] = sampleWind(f, trLat[i]!, trLon[i]!);
    const la = trLat[i]!, lo = trLon[i]!;
    const tail = 2.5 * 3600 / RADIUS; // streak length ~ 2.5 h of motion
    const la2 = la - v * tail, lo2 = lo - u * tail / c;
    const o = n * 3;
    trPos[o] = Math.cos(la) * Math.cos(lo); trPos[o + 1] = Math.sin(la); trPos[o + 2] = -Math.cos(la) * Math.sin(lo);
    trPos[o + 3] = Math.cos(la2) * Math.cos(lo2); trPos[o + 4] = Math.sin(la2); trPos[o + 5] = -Math.cos(la2) * Math.sin(lo2);
    const sp = Math.min(1, Math.hypot(u, v) / 40);
    trCol[o] = 1; trCol[o + 1] = 1; trCol[o + 2] = 1;
    trCol[o + 3] = 0.3 + 0.7 * sp; trCol[o + 4] = 0.3 + 0.7 * sp; trCol[o + 5] = 0.35;
    n += 2;
  }
  globe.setTracers(trPos, trCol, n);
}

// ---------------- worker messages
worker.onmessage = (ev: MessageEvent<FromWorker>): void => {
  const m = ev.data;
  if (m.type === 'ready') {
    log(`模式就緒 / Model ready: T${m.trunc} ${m.nlon}×${m.nlat} L${m.K}, Δt = ${m.dt} s`);
    $('grid').textContent = `T${m.trunc} · ${m.nlon}×${m.nlat} · L${m.K}`;
    const lev = $<HTMLInputElement>('level');
    lev.max = String(m.K - 1);
    if (Number(lev.value) > m.K - 1) lev.value = String(m.K - 1);
    pushView();
  } else if (m.type === 'frame') {
    frame = m;
    colourise(m);
    $('day').textContent = m.day.toFixed(2);
    $('rate').textContent = `${m.stepsPerSecond.toFixed(1)} 步/s steps/s · ${(m.stepsPerSecond * Number($('dt').dataset.dt ?? 0) / 86400 * 60).toFixed(1)} 日/分 days/min`;
    $('maxwind').textContent = `${m.maxWind.toFixed(1)} m/s`;
    $('psdrift').textContent = m.psDrift.toExponential(2);
    $('levelLabel').textContent = `σ = ${m.sigma[m.level]!.toFixed(3)} (≈ ${(m.sigma[m.level]! * 1000).toFixed(0)} hPa)`;
  } else if (m.type === 'zonal') {
    zonal = m;
    drawZonal();
  } else if (m.type === 'error') {
    log(`錯誤 / Error: ${m.message}`);
    running = false; syncRun();
  }
};

function drawZonal(): void {
  if (!zonal) return;
  const z = zonal, K = z.sigma.length, nl = z.lat.length;
  $('zonalInfo').textContent = `自第 ${z.fromDay.toFixed(0)} 日起平均，${z.samples} 筆 6 小時樣本 / Averaged since day ${z.fromDay.toFixed(0)}, ${z.samples} six-hourly samples`;
  drawSection($<HTMLCanvasElement>('secU'), z.lat, z.sigma, z.u, { diverging: true, contour: 5, label: '[u] 緯向平均西風 / zonal-mean zonal wind (m/s)' });
  drawSection($<HTMLCanvasElement>('secT'), z.lat, z.sigma, z.T, { diverging: false, contour: 10, label: '[T] 緯向平均溫度 / zonal-mean temperature (K)' });
  drawSection($<HTMLCanvasElement>('secPsi'), z.lat, z.sigmaHalf.slice(1, K), z.psi.slice(nl, K * nl).map((x) => x / 1e9), { diverging: true, contour: 10, label: 'ψ 經圈質量流函數 / meridional mass streamfunction (10⁹ kg/s)' });
}

// ---------------- controls
function pushView(): void {
  send({ type: 'view', field: $<HTMLSelectElement>('field').value as FieldId, level: Number($<HTMLInputElement>('level').value) });
}
function syncRun(): void {
  $('run').textContent = running ? '暫停 / Pause' : '執行 / Run';
  send({ type: 'run', running });
}
$('run').onclick = (): void => { running = !running; syncRun(); };
$('field').onchange = pushView;
$('level').oninput = pushView;
$('speed').oninput = (): void => send({ type: 'speed', stepsPerTick: Number($<HTMLInputElement>('speed').value) });
$('resetAvg').onclick = (): void => { send({ type: 'resetAverage' }); log('重設緯向平均 / Zonal average reset'); };
$('preset').onchange = (): void => init();
function init(): void {
  running = false; syncRun();
  const p = $<HTMLSelectElement>('preset').value;
  $('dt').dataset.dt = p === 'T21L20' ? '2400' : p === 'T42L20' ? '1200' : '900';
  if (p === 'JW_T42') {
    $<HTMLSelectElement>('field').value = 'ps';
    log('斜壓波：第 6–10 日可見氣旋加深與鋒面 / Baroclinic wave: cyclones deepen and fronts form around days 6–10');
  }
  log(`建立模式中 / Building model: ${p}`);
  send({ type: 'init', preset: p });
  send({ type: 'speed', stepsPerTick: Number($<HTMLInputElement>('speed').value) });
}
init();
window.addEventListener('resize', drawZonal);

let last = performance.now();
function tick(now: number): void {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (running) updateTracers(dt); else if (frame) updateTracers(0);
  globe.render();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
