// UI thread: controls, globe rendering, wind tracers, zonal-mean sections.

import { Globe } from './globe.js';
import { diverging, sequential } from './colormap.js';
import { drawSection } from './section.js';
import type { FieldId, FrameMessage, FromWorker, ToWorker, ZonalMessage } from './protocol.js';
import { NEST_HALF_WIDTH_KM, type NestSize } from './regional/protocol.js';

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('globe');
const log = (s: string): void => { const el = $('log'); el.textContent = `${s}\n${el.textContent ?? ''}`.slice(0, 4000); };

let globe: Globe;
try { globe = new Globe(canvas); } catch (e) { log(String(e)); throw e; }

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const send = (m: ToWorker): void => worker.postMessage(m);

let frame: FrameMessage | null = null;
let landMask: Uint8Array | null = null;
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
      globe.setField(f.lat, f.nlon, f.scalar, (v, idx) => {
        const t = Math.max(0, Math.min(1, (290 - v) / 170)) ** 0.8;
        const base: [number, number, number] = idx >= 0 && landMask && landMask[idx] ? [0.22, 0.25, 0.16] : [0.03, 0.07, 0.15];
        return [base[0] + t * (0.97 - base[0]), base[1] + t * (0.97 - base[1]), base[2] + t * (1 - base[2])];
      });
    } else if (f.field === 'snow') {
      globe.setField(f.lat, f.nlon, f.scalar, (v) => v < 0.2 ? [0.06, 0.12, 0.24] : (() => { const t = Math.min(1, Math.log(v / 0.2) / Math.log(100)); return [0.7 + 0.3 * t, 0.72 + 0.28 * t, 0.9 + 0.1 * t] as [number, number, number]; })());
    } else if (f.field === 'precip') {
      // rain: transparent-to-blue style ramp on a square-root scale
      globe.setField(f.lat, f.nlon, f.scalar, rainColour);
    } else globe.setField(f.lat, f.nlon, f.scalar, (v) => sequential((v - a) / ((b - a) || 1)));
    $('legend').textContent = `${a.toFixed(1)} … ${b.toFixed(1)} ${st.unit}`;
  }
}

/** Radar-style rain colours: dark ocean below 0.5 mm/day, then log-scaled blue -> green -> yellow -> red -> magenta. */
const RAIN_STOPS: [number, [number, number, number]][] = [
  [0.5, [0.16, 0.3, 0.55]], [2, [0.35, 0.6, 0.95]], [5, [0.2, 0.75, 0.45]], [10, [0.95, 0.9, 0.25]],
  [20, [0.98, 0.55, 0.15]], [40, [0.85, 0.15, 0.15]], [80, [0.8, 0.2, 0.8]],
];
function rainColour(v: number): [number, number, number] {
  if (v < RAIN_STOPS[0]![0]) return [0.06, 0.12, 0.24];
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
    landMask = m.land;
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
    $('season').textContent = m.declinationDeg === null ? '—' : `${dateFromEquinox(m.day)} · 太陽赤緯 / declination ${m.declinationDeg.toFixed(1)}°`;
    $('levelLabel').textContent = `σ = ${m.sigma[m.level]!.toFixed(3)} (≈ ${(m.sigma[m.level]! * 1000).toFixed(0)} hPa)`;
  } else if (m.type === 'snapshot') {
    pendingNest = m.payload;
    deliverNest();
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
globe.onPick = (lat, lon): void => {
  pick = { lat, lon };
  globe.setMarker(lat, lon, nestHalf());
  const lonD = lon * 180 / Math.PI;
  $('pick').textContent = `${Math.abs(lat * 180 / Math.PI).toFixed(1)}°${lat >= 0 ? 'N' : 'S'}, ${(lonD > 180 ? 360 - lonD : lonD).toFixed(1)}°${lonD > 180 ? 'W' : 'E'}`;
  $<HTMLButtonElement>('zoom').disabled = false;
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
  send({ type: 'snapshot' });
  log('擷取全球模式狀態中 / Capturing the global model state…');
};
$('nestClose').onclick = (): void => {
  $('nestOverlay').hidden = true;
  $<HTMLIFrameElement>('nestFrame').src = 'about:blank';
  nestWin = null;
};
$('nestSize').onchange = (): void => { if (pick) globe.setMarker(pick.lat, pick.lon, nestHalf()); };
$('resetAvg').onclick = (): void => { send({ type: 'resetAverage' }); log('重設緯向平均 / Zonal average reset'); };
$('preset').onchange = (): void => init();
$('backendSel').onchange = (): void => init();
function init(): void {
  running = false; syncRun();
  const preset = $<HTMLSelectElement>('preset').value, p = preset.replace(/_Q$/, '');
  $('dt').dataset.dt = p === 'T21L20' ? '2400' : p === 'T42L20' ? '1200' : p === 'AQUA_T21' ? '1200' : p === 'AQUA_T42' ? '720' : p === 'EARTH_T21' ? '1200' : p === 'EARTH_T42' ? '720' : p.endsWith('T85') ? '600' : '900';
  if (p.startsWith('EARTH')) {
    $<HTMLSelectElement>('field').value = 'precip';
    log('地球：真實海陸與地形、季節日照；模式從 3 月 20 日（春分）開始 / Earth: real land, orography and seasons; the model starts on 20 March (equinox)');
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
  send({ type: 'init', preset, backend: $<HTMLSelectElement>('backendSel').value as 'auto' | 'cpu' });
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
