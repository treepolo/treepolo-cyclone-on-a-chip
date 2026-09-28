// UI thread: controls, globe rendering, wind tracers, zonal-mean sections.

import { Globe } from './globe.js';
import { mountSavesPanel, offerDownload, type SaveMeta } from './saves.js';
import { diverging, sequential } from './colormap.js';
import { drawSection } from './section.js';
import type { FieldId, FrameMessage, FromWorker, ToWorker, ZonalMessage } from './protocol.js';
import { NEST_HALF_WIDTH_KM, type NestSize, type NestPayload, type FromRegionalWorker, type ToRegionalWorker } from './regional/protocol.js';

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('globe');
const log = (s: string): void => { const el = $('log'); el.textContent = `${s}\n${el.textContent ?? ''}`.slice(0, 4000); };

let globe: Globe;
try { globe = new Globe(canvas); } catch (e) { log(String(e)); throw e; }
Object.assign(globalThis, { globe }); // handy for inspecting the view from the console
globe.loadMap(new URL('../../data/earth_map_1024.png', import.meta.url).href).catch((e) => log(`地形圖載入失敗 / Relief map failed: ${String(e)}`));
const bindRange = (id: string, set: (v: number) => void): void => { const el = $<HTMLInputElement>(id); const f = (): void => set(Number(el.value)); el.oninput = f; f(); };
bindRange('exag', (v) => { globe.exaggeration = v; $('exagLabel').textContent = `×${v}`; });
bindRange('fieldAlpha', (v) => { globe.fieldAlpha = v / 100; });
{ const c = $<HTMLInputElement>('clouds3d'); c.onchange = (): void => { globe.cloudsOn = c.checked; }; }

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const send = (m: ToWorker): void => worker.postMessage(m);

let frame: FrameMessage | null = null;
let _landMask: Uint8Array | null = null;
let zonal: ZonalMessage | null = null;
let running = false;

// ---------------- nesting: click the globe to choose a region, then open the regional model there
let pick: { lat: number; lon: number } | null = null;
let nestWin: Window | null = null, nestReady = false;
let pendingNest: import('./regional/protocol.js').NestPayload | null = null;
const nestSize = (): NestSize => $<HTMLSelectElement>('nestSize').value as NestSize;
const nestHalf = (): number => NEST_HALF_WIDTH_KM[nestSize()] * 1e3 / 6.371e6;   // radians of arc
function deliverNest(): void {
  if (!nestWin || !nestReady || !pendingNest || !pick) return;
  // target '*': inside a sandboxed host the origin may be opaque; the receiver checks the message source
  nestWin.postMessage({ type: 'nest', payload: pendingNest, lat0: pick.lat, lon0: pick.lon, size: nestSize() }, '*');
  pendingNest = null;
  log('已傳送全球模式狀態到區域模式 / Global state sent to the regional model');
}
window.addEventListener('message', (ev: MessageEvent) => {
  if (!nestWin || ev.source !== nestWin || !ev.data || ev.data.type !== 'nest-ready') return;
  nestReady = true;
  deliverNest();
});

// ---------------- field colouring
const FIELD_STYLE: Record<FieldId, { div: boolean; range?: [number, number]; sym?: number; unit: string }> = {
  T: { div: false, unit: 'K' },
  u: { div: true, sym: 40, unit: 'm/s' },
  v: { div: true, sym: 20, unit: 'm/s' },
  speed: { div: false, range: [0, 50], unit: 'm/s' },
  vor: { div: true, sym: 1e-4, unit: 's⁻¹' },
  div: { div: true, sym: 1e-5, unit: 's⁻¹' },
  ps: { div: false, range: [970, 1030], unit: 'hPa' },
  precip: { div: false, range: [0, 40], unit: 'mm/day' },
  snow: { div: false, range: [0, 20], unit: 'mm/day (water equivalent)' },
  sst: { div: false, unit: 'K' },
  q: { div: false, unit: 'g/kg' },
  olr: { div: false, range: [100, 320], unit: 'W/m²' },
  ice: { div: false, range: [0, 3], unit: 'm' },
  sat: { div: false, range: [110, 290], unit: 'W/m² (OLR)' },
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
    if (f.field === 'sat') {
      // infrared-satellite style: cold (low-OLR) cloud tops bright over a land / ocean background
      globe.setField(f.lat, f.nlon, f.scalar, (v) => {
        const t = Math.max(0, Math.min(1, (290 - v) / 170)) ** 0.8;
        return [0.97, 0.97, 1, t];
      });
    } else if (f.field === 'snow') {
      globe.setField(f.lat, f.nlon, f.scalar, (v) => v < 0.2 ? [0, 0, 0, 0] : (() => { const t = Math.min(1, Math.log(v / 0.2) / Math.log(100)); return [0.7 + 0.3 * t, 0.72 + 0.28 * t, 0.9 + 0.1 * t, 0.5 + 0.5 * t] as [number, number, number, number]; })());
    } else if (f.field === 'precip') {
      // rain: transparent-to-blue style ramp on a square-root scale
      globe.setField(f.lat, f.nlon, f.scalar, rainColour);
    } else globe.setField(f.lat, f.nlon, f.scalar, (v) => sequential((v - a) / ((b - a) || 1)));
    $('legend').textContent = `${a.toFixed(1)} … ${b.toFixed(1)} ${st.unit}`;
  }
}

/** Radar-style rain colours: transparent below 0.5 mm/day, then log-scaled blue -> green -> yellow -> red -> magenta. */
const RAIN_STOPS: [number, [number, number, number]][] = [
  [0.5, [0.16, 0.3, 0.55]], [2, [0.35, 0.6, 0.95]], [5, [0.2, 0.75, 0.45]], [10, [0.95, 0.9, 0.25]],
  [20, [0.98, 0.55, 0.15]], [40, [0.85, 0.15, 0.15]], [80, [0.8, 0.2, 0.8]],
];
function rainColour(v: number): [number, number, number] | [number, number, number, number] {
  if (!(v >= RAIN_STOPS[0]![0])) return [0, 0, 0, 0];     // also NaN
  for (let i = 1; i < RAIN_STOPS.length; i++) {
    const [x1, c1] = RAIN_STOPS[i]!, [x0, c0] = RAIN_STOPS[i - 1]!;
    if (v <= x1) {
      const t = Math.log(v / x0) / Math.log(x1 / x0);
      return [c0[0] + t * (c1[0] - c0[0]), c0[1] + t * (c1[1] - c0[1]), c0[2] + t * (c1[2] - c0[2])];
    }
  }
  return RAIN_STOPS[RAIN_STOPS.length - 1]![1];
}

/** Calendar date for a model day counted from the northern spring equinox (20 March). */
function dateFromEquinox(day: number): string {
  const d = new Date(Date.UTC(2001, 2, 20) + (day % 365.25) * 86400000);
  return `${d.getUTCMonth() + 1} 月 ${d.getUTCDate()} 日 / ${d.toLocaleString('en', { month: 'short', timeZone: 'UTC' })} ${d.getUTCDate()}`;
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
    log(`模式就緒 / Model ready: T${m.trunc} ${m.nlon}×${m.nlat} L${m.K}, Δt = ${m.dt} s, ${m.backend === 'gpu' ? 'WebGPU f32' : 'CPU Float64'}`);
    if (m.note) log(m.note);
    $('backend').textContent = m.backend === 'gpu' ? 'WebGPU（f32）' : 'CPU（Float64）';
    $('grid').textContent = `T${m.trunc} · ${m.nlon}×${m.nlat} · L${m.K}`;
    globe.setOutline(m.lat, m.nlon, m.land);
    _landMask = m.land;
    const lev = $<HTMLInputElement>('level');
    lev.max = String(m.K - 1);
    if (Number(lev.value) > m.K - 1) lev.value = String(m.K - 1);
    pushView();
  } else if (m.type === 'frame') {
    frame = m;
    colourise(m);
    if (embed) { embed.globalDay = m.day; pace(); }
    globe.setClouds(m.cloud3d, m.nlon, m.nlat, m.cloudNz, m.cloudTop);
    $('day').textContent = m.day.toFixed(2);
    $('rate').textContent = `${m.stepsPerSecond.toFixed(1)} 步/s steps/s · ${(m.stepsPerSecond * Number($('dt').dataset.dt ?? 0) / 86400 * 60).toFixed(1)} 日/分 days/min`;
    $('maxwind').textContent = `${m.maxWind.toFixed(1)} m/s`;
    $('psdrift').textContent = m.psDrift.toExponential(2);
    $('season').textContent = m.declinationDeg === null ? '—' : `${dateFromEquinox(m.day)} · 太陽赤緯 / declination ${m.declinationDeg.toFixed(1)}°`;
    $('levelLabel').textContent = `σ = ${m.sigma[m.level]!.toFixed(3)} (≈ ${(m.sigma[m.level]! * 1000).toFixed(0)} hPa)`;
  } else if (m.type === 'paused') {
    log(m.reason); running = false; $('run').textContent = '執行 / Run';
  } else if (m.type === 'saveData') {
    pendingSave?.({ meta: m.meta, data: m.buffer }); pendingSave = null;
  } else if (m.type === 'snapshot') {
    const purpose = snapQueue.shift() ?? 'window';
    if (purpose === 'window') { pendingNest = m.payload; deliverNest(); }
    else if (purpose === 'embed-init' && embed) startEmbed(m.payload);
    else if (purpose === 'embed-bc' && embed?.reg) embed.reg.postMessage({ type: 'boundary', payload: m.payload } satisfies ToRegionalWorker);
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
  send({ type: 'run', running: running && !embed?.held });
  embed?.reg?.postMessage({ type: 'run', running: running && embed.ready } satisfies ToRegionalWorker);
}

// ---------------- regional nest embedded in the globe (true scale and position)
// The regional model runs in a second worker. One-way coupling: new lateral-boundary targets from the
// global state every model hour; the global model waits whenever it is more than an hour ahead.
type SnapPurpose = 'window' | 'embed-init' | 'embed-bc';
const snapQueue: SnapPurpose[] = [];
const requestSnapshot = (p: SnapPurpose): void => { snapQueue.push(p); send({ type: 'snapshot' }); };
let embed: { reg: Worker | null; lat0: number; lon0: number; day0: number; globalDay: number; tNest: number; lastBc: number; held: boolean; ready: boolean } | null = null;

function startEmbed(payload: NestPayload): void {
  if (!embed) return;
  const e = embed;
  e.day0 = payload.day; e.globalDay = payload.day; e.tNest = 0; e.lastBc = 0;
  const reg = new Worker(new URL('./regional/worker.js', import.meta.url), { type: 'module' });
  e.reg = reg;
  reg.onmessage = (ev: MessageEvent<FromRegionalWorker>): void => {
    const r = ev.data;
    if (embed !== e) return;
    if (r.type === 'ready') {
      e.ready = true;
      globe.setNest({ lat0: e.lat0, lon0: e.lon0, L: r.nx * r.dx, top: r.nz * r.dz });
      globe.setMarker(e.lat0, e.lon0, 0.5 * r.nx * r.dx / 6.371e6, 'nest');
      log(`嵌入區域模式就緒 / Embedded nest ready: ${r.nx}×${r.ny}×${r.nz}, Δx ${r.dx / 1000} km, ${r.backend === 'gpu' ? 'WebGPU' : 'CPU'}${r.note ? ' · ' + r.note : ''}`);
      reg.postMessage({ type: 'speed', stepsPerTick: 6 } satisfies ToRegionalWorker);
      syncRun();
    } else if (r.type === 'frame') {
      globe.setNestData(r.cloud, r.rain, r.nx, r.ny, r.nz);
      e.tNest = r.time;
      $('nestStatus').textContent = `${(r.time / 3600).toFixed(2)} h · ${r.stepsPerSecond.toFixed(1)} 步/s steps/s · w ${r.stats.wmax.toFixed(1)} m/s · ${e.held ? '全球模式等待中 / global waiting' : '同步中 / in step'}`;
      pace();
    } else if (r.type === 'error') log(`嵌入區域模式錯誤 / Embedded nest error: ${r.message}`);
  };
  reg.postMessage({ type: 'initNest', payload, lat0: e.lat0, lon0: e.lon0, size: nestSize(), backend: $<HTMLSelectElement>('backendSel').value as 'auto' | 'cpu' } satisfies ToRegionalWorker);
}

/** Keep the global model at most an hour ahead of the nest and refresh the nest boundaries hourly. */
function pace(): void {
  if (!embed?.ready) return;
  const tg = (embed.globalDay - embed.day0) * 86400;
  const held = tg > embed.tNest + 3600 ? true : tg < embed.tNest + 1800 ? false : embed.held;
  if (held !== embed.held) { embed.held = held; send({ type: 'run', running: running && !held }); }
  if (tg >= embed.lastBc + 3600) { embed.lastBc = Math.floor(tg / 3600) * 3600; requestSnapshot('embed-bc'); }
}

function stopEmbed(): void {
  if (!embed) return;
  embed.reg?.terminate();
  embed = null;
  globe.setNest(null);
  globe.setMarker(null, 0, 0, 'nest');
  $('nestStatus').textContent = '—';
  $<HTMLButtonElement>('unembed').disabled = true;
  syncRun();
}
$('run').onclick = (): void => { running = !running; syncRun(); };
$('field').onchange = pushView;
$('level').oninput = pushView;
$('speed').oninput = (): void => send({ type: 'speed', stepsPerTick: Number($<HTMLInputElement>('speed').value) });
globe.onPick = (lat, lon): void => {
  pick = { lat, lon };
  globe.setMarker(lat, lon, nestHalf());
  const lonD = lon * 180 / Math.PI;
  $('pick').textContent = `${Math.abs(lat * 180 / Math.PI).toFixed(1)}°${lat >= 0 ? 'N' : 'S'}, ${(lonD > 180 ? 360 - lonD : lonD).toFixed(1)}°${lonD > 180 ? 'W' : 'E'}`;
  $<HTMLButtonElement>('zoom').disabled = false;
  $<HTMLButtonElement>('embed').disabled = false;
};
$('zoom').onclick = (): void => {
  if (!pick) return;
  if (Math.abs(pick.lat) > 80 * Math.PI / 180) { log('極區附近無法巢狀（切平面近似）/ Nesting is not available within 10° of the poles (tangent-plane approximation)'); return; }
  nestReady = false;
  // the regional model opens in an in-page overlay (works on phones and inside sandboxed hosts)
  running = false; syncRun();
  const frame = $<HTMLIFrameElement>('nestFrame');
  frame.src = `regional.html#nest-${Date.now()}`;
  $('nestOverlay').hidden = false;
  nestWin = frame.contentWindow;
  requestSnapshot('window');
  log('擷取全球模式狀態中 / Capturing the global model state…');
};
$('embed').onclick = (): void => {
  if (!pick) return;
  if (Math.abs(pick.lat) > 80 * Math.PI / 180) { log('極區附近無法巢狀（切平面近似）/ Nesting is not available within 10° of the poles (tangent-plane approximation)'); return; }
  stopEmbed();
  embed = { reg: null, lat0: pick.lat, lon0: pick.lon, day0: 0, globalDay: 0, tNest: 0, lastBc: 0, held: false, ready: false };
  $<HTMLButtonElement>('unembed').disabled = false;
  globe.setMarker(null);
  requestSnapshot('embed-init');
  log('建立嵌入的區域模式中（以目前全球場為初始與邊界）/ Building the embedded nest from the current global state…');
};
$('unembed').onclick = stopEmbed;
// the regional page opens in the same in-page overlay (it then reaches the artifact capabilities of this page)
$('openRegional').onclick = (ev): void => {
  ev.preventDefault();
  nestReady = false; pendingNest = null;
  const frame = $<HTMLIFrameElement>('nestFrame');
  frame.src = 'regional.html';
  $('nestOverlay').hidden = false;
  nestWin = frame.contentWindow;
};
$('nestClose').onclick = (): void => {
  $('nestOverlay').hidden = true;
  $<HTMLIFrameElement>('nestFrame').src = 'about:blank';
  nestWin = null;
};
$('nestSize').onchange = (): void => { if (pick) globe.setMarker(pick.lat, pick.lon, nestHalf()); };
$('resetAvg').onclick = (): void => { send({ type: 'resetAverage' }); log('重設緯向平均 / Zonal average reset'); };
$('preset').onchange = (): void => init();
$('pace').onchange = (): void => send({ type: 'pace', target: Number($<HTMLSelectElement>('pace').value) });
$('step1').onclick = (): void => { if (!running) send({ type: 'step1' }); };
$('untilGo').onclick = (): void => {
  const d = Number($<HTMLInputElement>('untilD').value);
  if (!(d > 0)) return;
  send({ type: 'runUntil', days: d });
  running = true; syncRun();
  log(`執行 ${d} 模式日後自動暫停 / running for ${d} model days`);
};
// ---------------- saved simulations
// the regional overlay asks this page to offer its exports (the artifact's download capability is here)
addEventListener('message', async (ev: MessageEvent) => {
  if (!nestWin || ev.source !== nestWin || ev.data?.type !== 'export-save') return;
  let result: string | null = null;
  try { result = await offerDownload(String(ev.data.filename), ev.data.data as Blob); } catch (e) { result = String((e as Error).message ?? e); }
  nestWin.postMessage({ type: 'export-save-done', id: ev.data.id, result }, '*');
});
let pendingSave: ((r: { meta: SaveMeta; data: ArrayBuffer }) => void) | null = null;
mountSavesPanel($('saves'), 'global',
  () => new Promise((resolve, reject) => {
    if (pendingSave) { reject(new Error('存檔進行中 / a save is already in progress')); return; }
    pendingSave = resolve; send({ type: 'save' });
    setTimeout(() => { if (pendingSave === resolve) { pendingSave = null; reject(new Error('逾時 / timed out')); } }, 120000);
  }),
  (meta, data) => {
    const sel = $<HTMLSelectElement>('preset'), preset = String(meta.preset);
    if (!Array.from(sel.options).some((o) => o.value === preset)) { log(`這個版本沒有「${preset}」情境 / preset not available: ${preset}`); return; }
    sel.value = preset;
    init(data.slice(0));
  },
  log);
$('backendSel').onchange = (): void => init();
$('spinup').onchange = (): void => init();
function init(state?: ArrayBuffer): void {
  stopEmbed();
  running = false; syncRun();
  const preset = $<HTMLSelectElement>('preset').value, p = preset.replace(/_Q$/, '');
  $('dt').dataset.dt = p === 'T21L20' ? '2400' : p === 'T42L20' ? '1200' : p === 'AQUA_T21' ? '1200' : p === 'AQUA_T42' ? '720' : p === 'EARTH_T21' ? '1200' : p === 'EARTH_T42' ? '720' : p.endsWith('T85') ? '600' : p.endsWith('T170') ? '300' : '900';
  if (p.startsWith('EARTH')) {
    $<HTMLSelectElement>('field').value = 'precip';
    log(preset.endsWith('_Q') && $<HTMLInputElement>('spinup').checked
      ? '地球：從已起轉的 7 月初狀態開始（天氣系統、季風、熱帶擾動已發展）/ Earth: starting from a spun-up early-July state (weather systems, monsoon and tropical disturbances already developed)'
      : '地球：真實海陸與地形、季節日照；模式從 3 月 20 日（春分）靜止開始，天氣系統約需 1–2 週發展 / Earth: real land, orography and seasons; the model starts at rest on 20 March (equinox); weather systems take 1–2 weeks to develop');
  }
  if (p.startsWith('AQUA')) {
    $<HTMLSelectElement>('field').value = 'precip';
    log('濕水球：對流、ITCZ、風暴路徑與降水需約 30–60 模式日發展 / Moist aquaplanet: convection, ITCZ, storm tracks and rain develop over ~30–60 model days');
  }
  if (p === 'JW_T42') {
    $<HTMLSelectElement>('field').value = 'ps';
    log('斜壓波：第 6–10 日可見氣旋加深與鋒面 / Baroclinic wave: cyclones deepen and fronts form around days 6–10');
  }
  log(`建立模式中 / Building model: ${preset}`);
  send({ type: 'init', preset, backend: $<HTMLSelectElement>('backendSel').value as 'auto' | 'cpu', spinup: $<HTMLInputElement>('spinup').checked, ...(state ? { state } : {}) });
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
