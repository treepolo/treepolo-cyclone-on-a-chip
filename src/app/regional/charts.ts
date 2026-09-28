// Regional-model charts (2-D canvas): horizontal slices, composite maps, cross-sections along a drawn
// line, radius-height azimuthal means, soundings (skew-T log-p, hodograph, parcel indices), time series
// and Hovmoller diagrams. The worker computes the fields (GPU display kernels or the same diagnostics
// on the CPU); this module only draws them and accumulates the time series.

import { parcelAscent, type MapVar, type SliceVar, type SectionVar, type RzVar } from '../../regional/diagnostics.js';
import { isTcExperiment, type ChartRequest, type RegionalFrame, type RegionalExperiment, type TcRain } from './protocol.js';
import { INK, SERIES, FONT, FONT_SMALL, type Rect, type Scale, type ScaleKind, colorOf, niceCeil, ticks, fmt, drawField, drawColorbar, drawAxes, drawContours, drawArrow, drawBarb, haloText, tooltip } from './chartDraw.js';

export type ViewKind = '3d' | 'slice' | 'composite' | 'section' | 'rz' | 'sounding' | 'series' | 'hovmoller';
export const VIEWS: { v: ViewKind; label: string }[] = [
  { v: '3d', label: '3D 雲與降水 / 3-D clouds' },
  { v: 'slice', label: '水平切面 / Horizontal slice' },
  { v: 'composite', label: '合成圖與地面 / Composite & surface maps' },
  { v: 'section', label: '垂直剖面 / Cross-section' },
  { v: 'rz', label: '颱風半徑–高度 / TC radius–height' },
  { v: 'sounding', label: '探空 / Sounding' },
  { v: 'series', label: '時間序列 / Time series' },
  { v: 'hovmoller', label: 'Hovmöller（半徑–時間）/ radius–time' },
];

type SecVar = SectionVar | 'along' | 'normal';
interface VarInfo { label: string; unit: string; scale: ScaleKind; lo?: number; hi?: number; gamma?: number; clear?: number; reverse?: boolean; digits: number }
const VI: Record<string, VarInfo> = {
  dbz: { label: '雷達回波 / Reflectivity', unit: 'dBZ', scale: 'radar', digits: 0 },
  dbzMax: { label: '最大回波（合成）/ Composite reflectivity', unit: 'dBZ', scale: 'radar', digits: 0 },
  w: { label: '垂直速度 / Vertical velocity w', unit: 'm/s', scale: 'div', digits: 1 },
  speed: { label: '風速 / Wind speed', unit: 'm/s', scale: 'seq', lo: 0, digits: 1 },
  u: { label: '東西風 / Zonal wind u', unit: 'm/s', scale: 'div', digits: 1 },
  v: { label: '南北風 / Meridional wind v', unit: 'm/s', scale: 'div', digits: 1 },
  along: { label: '沿剖面風（A→B 為正）/ Along-section wind (A→B positive)', unit: 'm/s', scale: 'div', digits: 1 },
  normal: { label: '穿越剖面風（向左為正）/ Cross-section-normal wind (leftward positive)', unit: 'm/s', scale: 'div', digits: 1 },
  thp: { label: '位溫擾動 / θ′', unit: 'K', scale: 'div', digits: 2 },
  thetaE: { label: '相當位溫 / Equivalent potential temperature θe', unit: 'K', scale: 'seq', digits: 1 },
  rh: { label: '相對濕度（對水）/ Relative humidity (water)', unit: '%', scale: 'seq', lo: 0, hi: 100, digits: 0 },
  zeta: { label: '垂直渦度 / Vertical vorticity ζ', unit: 's⁻¹', scale: 'div', digits: 4 },
  pp: { label: '氣壓擾動 / Pressure perturbation p′', unit: 'hPa', scale: 'div', digits: 2 },
  qv: { label: '水氣混合比 / Water-vapour mixing ratio', unit: 'g/kg', scale: 'seq', lo: 0, digits: 1 },
  cloud: { label: '雲水＋雲冰 / Cloud water + ice', unit: 'g/kg', scale: 'seq', lo: 0, clear: 0.01, gamma: 0.5, digits: 2 },
  precip: { label: '雨＋雪＋霰 / Rain + snow + graupel', unit: 'g/kg', scale: 'seq', lo: 0, clear: 0.01, gamma: 0.5, digits: 2 },
  T: { label: '溫度 / Temperature', unit: '°C', scale: 'seq', digits: 1 },
  ctopT: { label: '紅外雲頂溫度（無雲處為地面）/ Infrared cloud-top temperature (surface where clear)', unit: '°C', scale: 'ir', lo: -90, hi: 40, digits: 1 },
  ctopZ: { label: '雲頂高度 / Cloud-top height', unit: 'km', scale: 'seq', lo: 0, clear: 0.01, digits: 1 },
  uh: { label: '上升氣流螺旋度（2–5 km）/ Updraft helicity (2–5 km)', unit: 'm²/s²', scale: 'div', digits: 0 },
  wMax: { label: '柱內最大上升速度 / Column-max updraft', unit: 'm/s', scale: 'seq', lo: 0, clear: 0.5, digits: 1 },
  rainRate: { label: '即時降水率 / Precipitation rate', unit: 'mm/h', scale: 'seq', lo: 0, clear: 0.1, gamma: 0.5, digits: 1 },
  rain: { label: '累積降水 / Accumulated precipitation', unit: 'mm', scale: 'seq', lo: 0, clear: 0.1, gamma: 0.5, digits: 1 },
  snow: { label: '累積降雪與霰（水當量）/ Accumulated snow + graupel (water equivalent)', unit: 'mm', scale: 'seq', lo: 0, clear: 0.05, gamma: 0.5, digits: 1 },
  slp: { label: '海平面氣壓 / Sea-level pressure', unit: 'hPa', scale: 'seq', digits: 1 },
  sfcWind: { label: '地面風速（最低層）/ Surface wind speed (lowest level)', unit: 'm/s', scale: 'seq', lo: 0, digits: 1 },
  sfcThp: { label: '地面位溫擾動（冷池）/ Surface θ′ (cold pool)', unit: 'K', scale: 'div', digits: 2 },
  sfcThetaE: { label: '地面相當位溫 / Surface θe', unit: 'K', scale: 'seq', digits: 1 },
  cape: { label: '地面氣塊對流可用位能 / Surface-based CAPE', unit: 'J/kg', scale: 'seq', lo: 0, clear: 1, digits: 0 },
  cin: { label: '地面氣塊對流抑制 / Surface-based CIN', unit: 'J/kg', scale: 'seq', hi: 0, reverse: true, clear: 1, digits: 0 },
  vt: { label: '切向風 / Tangential wind', unit: 'm/s', scale: 'div', digits: 1 },
  vr: { label: '徑向風（負：流入）/ Radial wind (negative: inflow)', unit: 'm/s', scale: 'div', digits: 1 },
  cond: { label: '總凝結物 / Total condensate', unit: 'g/kg', scale: 'seq', lo: 0, clear: 0.01, gamma: 0.5, digits: 2 },
};
const SLICE_CHOICES: SliceVar[] = ['dbz', 'w', 'speed', 'zeta', 'thp', 'thetaE', 'rh', 'pp', 'qv', 'cloud', 'precip', 'u', 'v'];
const MAP_CHOICES: MapVar[] = ['dbzMax', 'ctopT', 'rainRate', 'rain', 'snow', 'uh', 'wMax', 'slp', 'sfcWind', 'sfcThp', 'sfcThetaE', 'cape', 'cin', 'ctopZ'];
const SECTION_CHOICES: SecVar[] = ['dbz', 'w', 'along', 'normal', 'thp', 'thetaE', 'rh', 'cloud', 'precip', 'qv', 'pp', 'T'];
const RZ_CHOICES: RzVar[] = ['vt', 'vr', 'w', 'thp', 'cond'];

/** Colour scale for a field (fixed ranges where the variable has them, otherwise nice rounded extremes). */
function scaleFor(name: string, data: ArrayLike<number>): Scale {
  const vi = VI[name]!;
  let lo = Infinity, hi = -Infinity, amax = 0;
  for (let i = 0; i < data.length; i++) { const v = data[i]!; if (!Number.isFinite(v)) continue; lo = Math.min(lo, v); hi = Math.max(hi, v); amax = Math.max(amax, Math.abs(v)); }
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  if (vi.scale === 'div') { const a = niceCeil(Math.max(amax, 1e-6)); return { kind: 'div', lo: -a, hi: a, gamma: 1, clear: null, reverse: false }; }
  if (vi.lo !== undefined) lo = vi.lo;
  if (vi.hi !== undefined) hi = vi.hi;
  if (vi.scale === 'seq' && vi.lo === undefined && vi.hi === undefined) {
    const st = niceCeil(Math.max(hi - lo, 1e-6) / 5); lo = Math.floor(lo / st) * st; hi = Math.ceil(hi / st) * st;
  } else if (vi.scale === 'seq' && vi.hi === undefined) hi = lo + niceCeil(Math.max(hi - lo, vi.clear ?? 1e-6));
  else if (vi.scale === 'seq' && vi.lo === undefined) lo = hi - niceCeil(Math.max(hi - lo, vi.clear ?? 1e-6));
  if (hi <= lo) hi = lo + 1;
  return { kind: vi.scale, lo, hi, gamma: vi.gamma ?? 1, clear: vi.clear ?? null, reverse: !!vi.reverse };
}

interface Sample { t: number; dp: number | null; vmax: number; rmw: number | null; wmax: number; zeta: number; uh: number; dbz: number; vg: number; rain: number; cape: number; storm: { x: number; y: number } | null; ew: { r: number; v: number }[] | null; tcRain: TcRain | null }
interface GridInfo { nx: number; ny: number; nz: number; dx: number; dy: number; dz: number; experiment: RegionalExperiment; land: Uint8Array | null }
type MapFrame = { r: Rect; Lx: number; Ly: number };
/** what a click or drag on a map does */
export type MapTool = 'inspect' | 'warm' | 'cold' | 'warmer' | 'cooler' | 'land' | 'sea';
const TOOLS: { v: MapTool; label: string }[] = [
  { v: 'inspect', label: '滑鼠：剖面線與探空點 / Mouse: section line & sounding' },
  { v: 'warm', label: '點一下放暖泡（+3 K）/ Click: warm bubble (+3 K)' },
  { v: 'cold', label: '點一下放冷池（−6 K）/ Click: cold pool (−6 K)' },
  { v: 'warmer', label: '塗暖海溫 +2 °C / Paint warmer sea' },
  { v: 'cooler', label: '塗冷海溫 −2 °C / Paint cooler sea' },
  { v: 'land', label: '塗陸地（乾地面）/ Paint land (dry ground)' },
  { v: 'sea', label: '塗回海洋 / Paint sea' },
];

export interface ChartsHooks {
  /** chart data wanted in each frame (sent to the worker) */
  request(req: ChartRequest): void;
  volMode(mode: number): void;
  view(v: ViewKind): void;
  /** tracer particles in the 3-D view (count, 0 = off) */
  tracers(n: number): void;
  /** interaction on a map: place a bubble or cold pool, or paint the surface (domain coordinates, m) */
  interact(kind: MapTool, x: number, y: number, radius: number): void;
  /** 3-D camera mode */
  camera(mode: 'orbit' | 'fly'): void;
}

export class RegionalCharts {
  view: ViewKind = '3d';
  private sel = { slice: 'dbz' as SliceVar, composite: 'dbzMax' as MapVar, section: 'dbz' as SecVar, rz: 'vt' as RzVar };
  private level = 3;
  private show = { arrows: true, isobars: true, sfcWind: false, vectors: true, rzOnPoint: false };
  private line: { x0: number; y0: number; x1: number; y1: number } | null = null;
  private point: { x: number; y: number } | null = null;
  private grid: GridInfo | null = null;
  private frame: RegionalFrame | null = null;
  private samples: Sample[] = [];
  private hov: { t: number; dr: number; vt: number[] }[] = [];
  private hover: { x: number; y: number } | null = null;
  private drag: { x0: number; y0: number; px: number; py: number; x1: number; y1: number; moved: boolean } | null = null;
  private mapF: MapFrame | null = null;
  private plotF: { r: Rect; kind: 'section' | 'rz' | 'hov' | 'series'; x0: number; x1: number; y0: number; y1: number; panels?: Rect[] } | null = null;
  private readonly ctx: CanvasRenderingContext2D;
  private pending = 0;
  private lastReq = '';
  private volMode = 0;
  private tracerN = 0;
  private tool: MapTool = 'inspect';
  private cam: 'orbit' | 'fly' = 'orbit';
  private lastPaint: { x: number; y: number } | null = null;
  /** paint brush radius (m): 3 cells or a twentieth of the domain, whichever is larger */
  private get brush(): number { const g = this.grid!; return Math.max(3 * g.dx, Math.min(g.nx * g.dx, g.ny * g.dy) / 20); }
  private noteText = '';
  private noteTimer = 0;
  private set note(t: string) { this.noteText = t; clearTimeout(this.noteTimer); if (t) this.noteTimer = window.setTimeout(() => { this.noteText = ''; this.redraw(); }, 5000); }
  private get note(): string { return this.noteText; }

  constructor(private readonly canvas: HTMLCanvasElement, private readonly bar: HTMLElement, private readonly hooks: ChartsHooks) {
    this.ctx = canvas.getContext('2d')!;
    canvas.addEventListener('pointerdown', (e) => this.down(e));
    canvas.addEventListener('pointermove', (e) => this.move(e));
    canvas.addEventListener('pointerup', (e) => this.up(e));
    canvas.addEventListener('pointerleave', () => { this.hover = null; if (!this.drag) this.redraw(); });
    new ResizeObserver(() => this.redraw()).observe(canvas);
    this.buildBar();
  }

  // ---------------------------------------------------------------- state from the page

  /** A new model is ready; keepSeries: the run continues (refinement), so the time series go on. */
  setGrid(g: GridInfo, keepSeries: boolean): void {
    const same = this.grid && this.grid.nx * this.grid.dx === g.nx * g.dx && this.grid.ny * this.grid.dy === g.ny * g.dy;
    this.grid = g;
    if (!keepSeries) { this.samples = []; this.hov = []; }
    if (!same) { this.line = null; this.point = null; }
    this.level = this.nearestLevel(this.view === 'slice' ? this.levelZ() : 1500);
    this.frame = null;
    this.buildBar();
    this.sendRequest();
  }

  private levelZ(): number { return this.grid ? (this.level + 0.5) * this.grid.dz : 1500; }
  private nearestLevel(z: number): number { const g = this.grid!; return Math.max(0, Math.min(g.nz - 1, Math.round(z / g.dz - 0.5))); }

  onFrame(f: RegionalFrame): void {
    this.frame = f;
    const s = f.stats;
    const smp: Sample = { t: f.time, dp: s.dp, vmax: s.vmax, rmw: s.rmw, wmax: s.wmax, zeta: s.zetaMax, uh: s.uhMax, dbz: s.dbzMax, vg: s.vGround, rain: s.rainmax, cape: s.capeMax, storm: s.storm, ew: s.eyewalls, tcRain: s.tcRain ?? null };
    const last = this.samples[this.samples.length - 1];
    if (last && f.time < last.t - 1e-6) { this.samples = []; this.hov = []; }
    if (last && Math.abs(f.time - last.t) < 1e-6) this.samples[this.samples.length - 1] = smp; else this.samples.push(smp);
    if (s.vtProfile) {
      const h = this.hov[this.hov.length - 1], row = { t: f.time, dr: s.vtProfile.dr, vt: s.vtProfile.vt };
      if (h && Math.abs(h.t - f.time) < 1e-6) this.hov[this.hov.length - 1] = row; else this.hov.push(row);
    }
    // bounded history: drop every other sample (keeping the newest) beyond 4000
    if (this.samples.length > 4000) this.samples = this.samples.filter((_, i) => i % 2 === 1 || i === this.samples.length - 1);
    if (this.hov.length > 3000) this.hov = this.hov.filter((_, i) => i % 2 === 1 || i === this.hov.length - 1);
    if (this.view !== '3d') this.redraw();
  }

  /** Series as CSV (time in hours). */
  seriesCsv(): string {
    const rows = ['t_h,dp_hPa,vmax_ms,rmw_km,wmax_ms,zeta_s-1,uh_m2s2,dbz_max,vground_ms,rainmax_mm,cape_max_Jkg,storm_x_km,storm_y_km,rain_core_mmh,rain_outer_mmh,wet_outer'];
    for (const s of this.samples) rows.push([s.t / 3600, s.dp, s.vmax, s.rmw === null ? null : s.rmw / 1000, s.wmax, s.zeta, s.uh, s.dbz, s.vg, s.rain, s.cape, s.storm ? s.storm.x / 1000 : null, s.storm ? s.storm.y / 1000 : null,
      s.tcRain?.core ?? null, s.tcRain?.outer ?? null, s.tcRain?.wet ?? null]
      .map((v) => (v === null || v === undefined ? '' : +Number(v).toPrecision(6))).join(','));
    return rows.join('\n');
  }

  // ---------------------------------------------------------------- requests

  private request(): ChartRequest {
    const v = this.view, g = this.grid;
    const req: ChartRequest = { maps: [], slice: null, section: null, sounding: null, rz: null };
    if (!g) return req;
    if (v === 'slice') req.slice = { k: this.level, vars: [...new Set<SliceVar>([this.sel.slice, 'u', 'v'])] };
    if (v === 'composite') {
      const maps = new Set<MapVar>([this.sel.composite]);
      if (this.show.isobars) maps.add('slp');
      if (this.show.sfcWind) { maps.add('sfcU'); maps.add('sfcV'); }
      req.maps = [...maps];
    }
    if (v === 'section' || v === 'sounding') req.maps = ['dbzMax'];
    if (v === 'section') req.section = this.line ?? this.defaultLine();
    if (v === 'sounding') req.sounding = this.point ?? this.defaultPoint();
    if (v === 'rz') req.rz = this.show.rzOnPoint && this.point ? this.point : 'auto';
    return req;
  }
  private defaultLine(): { x0: number; y0: number; x1: number; y1: number } {
    const g = this.grid!, Lx = g.nx * g.dx, Ly = g.ny * g.dy, st = this.stormXY();
    const y = st ? st.y : Ly / 2;
    return { x0: 0.05 * Lx, y0: y, x1: 0.95 * Lx, y1: y };
  }
  private defaultPoint(): { x: number; y: number } { const g = this.grid!; return this.stormXY() ?? { x: g.nx * g.dx / 2, y: g.ny * g.dy / 2 }; }
  /** latest storm position in domain coordinates */
  private stormXY(): { x: number; y: number } | null {
    const f = this.frame, s = f?.stats.storm;
    if (!f || !s || !this.grid) return null;
    const x = s.x - f.origin.x, y = s.y - f.origin.y, g = this.grid;
    return x >= 0 && y >= 0 && x <= g.nx * g.dx && y <= g.ny * g.dy ? { x, y } : null;
  }

  private sendRequest(): void {
    const req = this.request(), key = JSON.stringify(req);
    if (key === this.lastReq) return;
    this.lastReq = key;
    clearTimeout(this.pending);
    // throttle (slider drags): the worker re-renders a paused frame for each request
    this.pending = window.setTimeout(() => this.hooks.request(req), 120);
  }

  setView(v: ViewKind): void {
    if (v === this.view) return;
    const was = this.view;
    this.view = v;
    this.note = '';
    if (v === 'slice' && was !== 'slice' && this.grid) this.level = Math.min(this.level, this.grid.nz - 1);
    this.buildBar();
    this.sendRequest();
    this.hooks.view(v);
    this.redraw();
  }

  // ---------------------------------------------------------------- toolbar

  private buildBar(): void {
    const b = this.bar;
    b.textContent = '';
    const sel = (opts: { v: string; label: string }[], value: string, on: (v: string) => void, title: string): HTMLSelectElement => {
      const s = document.createElement('select'); s.title = title;
      for (const o of opts) { const e = document.createElement('option'); e.value = o.v; e.textContent = o.label; s.append(e); }
      s.value = value; s.onchange = (): void => on(s.value); b.append(s); return s;
    };
    const chk = (label: string, on: boolean, set: (v: boolean) => void): void => {
      const l = document.createElement('label'); l.className = 'chk';
      const c = document.createElement('input'); c.type = 'checkbox'; c.checked = on; c.onchange = (): void => { set(c.checked); this.sendRequest(); this.redraw(); };
      l.append(c, document.createTextNode(' ' + label)); b.append(l);
    };
    const hint = (t: string): void => { const s = document.createElement('span'); s.className = 'hint'; s.textContent = t; b.append(s); };
    sel(VIEWS.map((x) => ({ v: x.v, label: x.label })), this.view, (v) => this.setView(v as ViewKind), '主畫面 / Main view');
    const vv = this.view;
    if (vv === '3d') {
      sel([{ v: '0', label: '藍色：降水粒子 / Blue: precipitation' }, { v: '1', label: '藍色：上升氣流 w / Blue: updraft w' }, { v: '2', label: '藍色：氣旋式渦度 / Blue: cyclonic vorticity' }],
        String(this.volMode), (v) => { this.volMode = Number(v); this.hooks.volMode(this.volMode); }, '3D 第二通道 / 3-D second channel');
      sel([{ v: '0', label: '無軌跡粒子 / No trajectory particles' }, { v: '2000', label: '軌跡粒子 2000 / 2,000 particles' }, { v: '6000', label: '軌跡粒子 6000 / 6,000 particles' }, { v: '16000', label: '軌跡粒子 16000 / 16,000 particles' }],
        String(this.tracerN), (v) => { this.tracerN = Number(v); this.hooks.tracers(this.tracerN); this.buildBar(); }, '軌跡粒子 / Trajectory particles');
      sel([{ v: 'orbit', label: '相機：環繞 / Camera: orbit' }, { v: 'fly', label: '相機：自由飛行 / Camera: free flight' }], this.cam, (v) => { this.cam = v as 'orbit' | 'fly'; this.hooks.camera(this.cam); this.buildBar(); }, '相機 / Camera');
      if (this.tracerN) hint('粒子：從低層 2 km 內出發，橘 = 低、淡黃 = 高；尾跡為最近幾個畫面 / particles start in the lowest 2 km; orange low, pale yellow high; trails show recent frames');
      if (this.cam === 'fly') hint('自由飛行：W/S 前後、A/D 左右、Q/E 上下（Shift 加速）、拖曳轉頭、滾輪前進後退 / free flight: W/S forward/back, A/D left/right, Q/E down/up (Shift: faster), drag to look, wheel to move');
    }
    if (vv === 'slice') {
      sel(SLICE_CHOICES.map((v) => ({ v, label: VI[v]!.label })), this.sel.slice, (v) => { this.sel.slice = v as SliceVar; this.sendRequest(); this.redraw(); }, '變數 / Variable');
      if (this.grid) {
        const r = document.createElement('input'); r.type = 'range'; r.min = '0'; r.max = String(this.grid.nz - 1); r.value = String(this.level); r.className = 'lvl';
        r.title = '高度 / Height';
        const lab = document.createElement('span'); lab.className = 'hint';
        const upd = (): void => { lab.textContent = `z = ${(this.levelZ() / 1000).toFixed(2)} km`; };
        r.oninput = (): void => { this.level = Number(r.value); upd(); this.sendRequest(); };
        upd(); b.append(r, lab);
      }
      chk('風向箭頭 / Wind arrows', this.show.arrows, (v) => { this.show.arrows = v; });
      sel(TOOLS, this.tool, (v) => { this.tool = v as MapTool; this.buildBar(); }, '滑鼠工具 / Mouse tool');
      hint(this.toolHint());
    }
    if (vv === 'composite') {
      sel(MAP_CHOICES.map((v) => ({ v, label: VI[v]!.label })), this.sel.composite, (v) => { this.sel.composite = v as MapVar; this.sendRequest(); this.redraw(); }, '變數 / Variable');
      chk('海平面等壓線 / Isobars', this.show.isobars, (v) => { this.show.isobars = v; });
      chk('地面風箭頭 / Surface wind', this.show.sfcWind, (v) => { this.show.sfcWind = v; });
      sel(TOOLS, this.tool, (v) => { this.tool = v as MapTool; this.buildBar(); }, '滑鼠工具 / Mouse tool');
      hint(this.toolHint());
    }
    if (vv === 'section') {
      sel(SECTION_CHOICES.map((v) => ({ v, label: VI[v]!.label })), this.sel.section, (v) => { this.sel.section = v as SecVar; this.redraw(); }, '變數 / Variable');
      chk('剖面內風向量 / In-plane wind', this.show.vectors, (v) => { this.show.vectors = v; });
      hint('在小地圖上拖曳畫新的剖面線。白線：雲邊界 0.1 g/kg；藍虛線：0 °C；向量：沿剖面風與 w，同一比例 / drag on the small map to draw a new line. White: cloud edge 0.1 g/kg; blue dashed: 0 °C; vectors: along-section wind and w on the same scale');
    }
    if (vv === 'rz') {
      sel(RZ_CHOICES.map((v) => ({ v, label: VI[v]!.label })), this.sel.rz, (v) => { this.sel.rz = v as RzVar; this.redraw(); }, '變數 / Variable');
      chk('以選取的探空點為中心 / Centre on the picked point', this.show.rzOnPoint, (v) => { this.show.rzOnPoint = v; });
      hint('預設中心：地面氣壓最低處。白色等值線：切向風（填色為切向風時改畫 w），虛線為負 / default centre: surface-pressure minimum. White contours: tangential wind (w when the fill is the tangential wind), dashed negative');
    }
    if (vv === 'sounding') hint('在小地圖上點一下選位置 / click the small map to pick the column');
    if (vv === 'series' || vv === 'hovmoller') {
      const btn = document.createElement('button'); btn.textContent = '清除紀錄 / Clear';
      btn.onclick = (): void => { this.samples = []; this.hov = []; this.redraw(); };
      const cp = document.createElement('button'); cp.textContent = '複製 CSV / Copy CSV';
      cp.onclick = async (): Promise<void> => {
        try { await navigator.clipboard.writeText(this.seriesCsv()); this.note = '已複製時間序列 / series copied'; }
        catch { this.note = '無法存取剪貼簿 / clipboard unavailable'; }
        this.redraw();
      };
      b.append(btn, cp);
    }
  }

  private toolHint(): string {
    if (this.tool === 'inspect') return '拖曳畫剖面線、點一下選探空點 / drag: cross-section line · click: sounding point';
    if (this.tool === 'warm' || this.tool === 'cold') return '只改變條件：點一下就把暖泡或冷池加進目前的大氣，之後怎麼發展由方程決定 / changes the conditions only: a click adds the perturbation, the equations decide what happens next';
    return `拖曳塗抹（筆刷半徑 ${(this.brush / 1000).toFixed(0)} km），只影響有地面通量的實驗；沒有地形（模式沒有地形座標，放不了山）/ drag to paint (brush radius ${(this.brush / 1000).toFixed(0)} km); affects experiments with surface fluxes; no mountains (the model has no terrain coordinate)`;
  }

  // ---------------------------------------------------------------- pointer

  private isPaint(): boolean { return (this.view === 'slice' || this.view === 'composite') && (this.tool === 'warmer' || this.tool === 'cooler' || this.tool === 'land' || this.tool === 'sea'); }
  private local(e: PointerEvent): { x: number; y: number } { const r = this.canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
  private toDomain(p: { x: number; y: number }): { x: number; y: number } | null {
    const f = this.mapF; if (!f) return null;
    const { r } = f;
    if (p.x < r.x || p.y < r.y || p.x > r.x + r.w || p.y > r.y + r.h) return null;
    return { x: (p.x - r.x) / r.w * f.Lx, y: (r.y + r.h - p.y) / r.h * f.Ly };
  }
  private down(e: PointerEvent): void {
    const p = this.local(e), d = this.toDomain(p);
    if (!d || !['slice', 'composite', 'section', 'sounding'].includes(this.view)) return;
    this.canvas.setPointerCapture(e.pointerId);
    this.drag = { x0: d.x, y0: d.y, px: p.x, py: p.y, x1: d.x, y1: d.y, moved: false };
  }
  private move(e: PointerEvent): void {
    const p = this.local(e);
    this.hover = p;
    if (this.drag) {
      if (Math.hypot(p.x - this.drag.px, p.y - this.drag.py) > 6) this.drag.moved = true;
      const f = this.mapF;
      if (f) { this.drag.x1 = Math.max(0, Math.min(f.Lx, (p.x - f.r.x) / f.r.w * f.Lx)); this.drag.y1 = Math.max(0, Math.min(f.Ly, (f.r.y + f.r.h - p.y) / f.r.h * f.Ly)); }
      // painting: a stroke every half brush radius along the drag
      if (this.isPaint() && (!this.lastPaint || Math.hypot(this.drag.x1 - this.lastPaint.x, this.drag.y1 - this.lastPaint.y) > 0.5 * this.brush)) {
        this.lastPaint = { x: this.drag.x1, y: this.drag.y1 };
        this.hooks.interact(this.tool, this.drag.x1, this.drag.y1, this.brush);
      }
    }
    this.redraw();
  }
  private up(e: PointerEvent): void {
    const d = this.drag; this.drag = null;
    try { this.canvas.releasePointerCapture(e.pointerId); } catch { /* not captured */ }
    if (!d) return;
    if (this.tool !== 'inspect' && (this.view === 'slice' || this.view === 'composite')) {
      if (this.isPaint()) { if (!this.lastPaint) this.hooks.interact(this.tool, d.x0, d.y0, this.brush); this.lastPaint = null; }
      else if (!d.moved) this.hooks.interact(this.tool, d.x0, d.y0, 0);
      return;
    }
    if (d.moved) {
      this.line = { x0: d.x0, y0: d.y0, x1: d.x1, y1: d.y1 };
      if (this.view !== 'section') this.note = '剖面線已設定：到「垂直剖面」查看 / Line set: see Cross-section';
    } else if (this.view !== 'section') {
      this.point = { x: d.x0, y: d.y0 };
      if (this.view !== 'sounding') this.note = '探空點已設定：到「探空」查看 / Point set: see Sounding';
    }
    this.sendRequest();
    this.redraw();
  }

  // ---------------------------------------------------------------- drawing

  redraw(): void {
    if (this.view === '3d' || this.drawQueued) return;
    this.drawQueued = true;
    requestAnimationFrame(() => { this.drawQueued = false; this.draw(); });
  }
  private drawQueued = false;

  private draw(): void {
    const c = this.canvas, dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = Math.max(1, c.clientWidth), H = Math.max(1, c.clientHeight);
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = INK.surface; ctx.fillRect(0, 0, W, H);
    ctx.font = FONT;
    this.mapF = null; this.plotF = null;
    const top = 8, area: Rect = { x: 0, y: top, w: W, h: H - top };
    const f = this.frame, ch = f?.charts;
    const msg = (t: string): void => this.centerText(t, area);
    if (this.view === 'series') this.drawSeries(area);
    else if (this.view === 'hovmoller') this.drawHovmoller(area);
    else if (!f || !ch) msg('等待資料… / Waiting for data…');
    else if (this.view === 'slice') this.drawSlice(area);
    else if (this.view === 'composite') this.drawComposite(area);
    else if (this.view === 'section') this.drawSection(area);
    else if (this.view === 'rz') this.drawRZ(area);
    else if (this.view === 'sounding') this.drawSounding(area);
    if (this.note) {
      ctx.font = FONT_SMALL; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      haloText(ctx, this.note, W - 10, H - 8, INK.secondary);
    }
  }

  private timeLabel(): string {
    const t = this.frame?.time ?? 0;
    return t < 7200 ? `t = ${(t / 60).toFixed(0)} min` : `t = ${(t / 3600).toFixed(2)} h`;
  }

  /** Square-ish map frame (domain aspect) inside rect, leaving room for the colour bar and axes. */
  private mapFrame(r: Rect, colorbar = true): MapFrame {
    const g = this.grid!, Lx = g.nx * g.dx, Ly = g.ny * g.dy;
    const ml = 46, mr = colorbar ? 72 : 12, mt = 26, mb = 36;
    const aw = Math.max(40, r.w - ml - mr), ah = Math.max(40, r.h - mt - mb), s = Math.min(aw / Lx, ah / Ly);
    const w = Lx * s, h = Ly * s;
    return { r: { x: r.x + ml + Math.max(0, (aw - w) / 2), y: r.y + mt + Math.max(0, (ah - h) / 2), w, h }, Lx, Ly };
  }

  private underlay(): (i: number, j: number) => [number, number, number] {
    const g = this.grid!, land = g.land, sea: [number, number, number] = [15, 28, 43], ground: [number, number, number] = [26, 34, 26];
    const allSea = isTcExperiment(g.experiment);
    return (i, j) => (land ? (land[j * g.nx + i] ? ground : sea) : allSea ? sea : ground);
  }

  /** Plan-view map of a [j][i] field with colour bar, axes and title. */
  private drawPlan(r: Rect, name: string, data: Float32Array, title: string, colorbar = true): MapFrame {
    const ctx = this.ctx, g = this.grid!, mf = this.mapFrame(r, colorbar), m = mf.r;
    const sc = scaleFor(name, data);
    drawField(ctx, m, g.nx, g.ny, (i, j) => data[j * g.nx + i]!, sc, this.underlay());
    drawAxes(ctx, m, { lo: 0, hi: mf.Lx / 1000, label: 'x (km)' }, { lo: 0, hi: mf.Ly / 1000, label: 'y (km)' });
    if (colorbar) drawColorbar(ctx, { x: m.x + m.w + 12, y: m.y + 14, w: 12, h: Math.max(40, m.h - 14) }, sc, VI[name]!.unit, VI[name]!.digits);
    ctx.font = FONT; ctx.fillStyle = INK.primary; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText(title, m.x, m.y - 8);
    this.mapF = mf;
    return mf;
  }

  /** Section line, sounding point and storm marker on a map. */
  private drawMarks(mf: MapFrame): void {
    const ctx = this.ctx, X = (x: number): number => mf.r.x + x / mf.Lx * mf.r.w, Y = (y: number): number => mf.r.y + mf.r.h - y / mf.Ly * mf.r.h;
    const ln = this.drag?.moved ? { x0: this.drag.x0, y0: this.drag.y0, x1: this.drag.x1, y1: this.drag.y1 } : this.view === 'section' ? (this.line ?? this.frame?.charts?.section ?? null) : this.line;
    if (ln) {
      ctx.lineCap = 'round';
      for (const [w, c] of [[4, INK.halo], [1.6, '#fafafa']] as const) { ctx.strokeStyle = c; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(X(ln.x0), Y(ln.y0)); ctx.lineTo(X(ln.x1), Y(ln.y1)); ctx.stroke(); }
      ctx.font = FONT; ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
      haloText(ctx, 'A', X(ln.x0) - 9 * Math.sign(ln.x1 - ln.x0 || 1), Y(ln.y0));
      haloText(ctx, 'B', X(ln.x1) + 9 * Math.sign(ln.x1 - ln.x0 || 1), Y(ln.y1));
    }
    const pt = this.view === 'sounding' ? (this.point ?? this.frame?.charts?.sounding ?? null) : this.point;
    if (pt) {
      const px = X(pt.x), py = Y(pt.y);
      for (const [w, c] of [[4, INK.halo], [1.6, '#fafafa']] as const) { ctx.strokeStyle = c; ctx.lineWidth = w; ctx.beginPath(); ctx.arc(px, py, 6, 0, 2 * Math.PI); ctx.moveTo(px - 10, py); ctx.lineTo(px + 10, py); ctx.moveTo(px, py - 10); ctx.lineTo(px, py + 10); ctx.stroke(); }
    }
  }

  /** Hover readout of a map field. */
  private mapReadout(mf: MapFrame, lines: (i: number, j: number) => string[]): void {
    const h = this.hover; if (!h || this.drag) return;
    const d = this.toDomain(h); if (!d) return;
    const g = this.grid!, i = Math.min(g.nx - 1, Math.floor(d.x / g.dx)), j = Math.min(g.ny - 1, Math.floor(d.y / g.dy));
    tooltip(this.ctx, [`x ${(d.x / 1000).toFixed(1)} km, y ${(d.y / 1000).toFixed(1)} km`, ...lines(i, j)], h.x, h.y, this.canvas.clientWidth, this.canvas.clientHeight);
    void mf;
  }

  /** Wind arrows sampled every few cells ([j][i] components in m/s). */
  private drawWind(mf: MapFrame, u: Float32Array, v: Float32Array): void {
    const g = this.grid!, n = Math.max(1, Math.round(Math.max(g.nx, g.ny) / 22));
    let vmax = 0; for (let i = 0; i < u.length; i++) vmax = Math.max(vmax, Math.hypot(u[i]!, v[i]!));
    const ref = niceCeil(Math.max(vmax, 1)), px = mf.r.w / g.nx * n * 0.9 / ref;
    for (let j = Math.floor(n / 2); j < g.ny; j += n) for (let i = Math.floor(n / 2); i < g.nx; i += n) {
      const c = j * g.nx + i, x = mf.r.x + (i + 0.5) / g.nx * mf.r.w, y = mf.r.y + mf.r.h - (j + 0.5) / g.ny * mf.r.h;
      drawArrow(this.ctx, x, y, u[c]! * px, -v[c]! * px);
    }
    // reference arrow
    const ctx = this.ctx, rx = mf.r.x + mf.r.w - ref * px - 60, ry = mf.r.y + mf.r.h + 26;
    drawArrow(ctx, rx, ry, ref * px, 0);
    ctx.font = FONT_SMALL; ctx.fillStyle = INK.secondary; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(`${ref} m/s`, rx + ref * px + 6, ry);
  }

  private drawSlice(area: Rect): void {
    const s = this.frame!.charts!.slice, name = this.sel.slice;
    if (!s || !s.vars[name]) { this.waiting(area); return; }
    const data = s.vars[name]!, info = VI[name]!;
    const mf = this.drawPlan(area, name, data, `${info.label} · z = ${(s.z / 1000).toFixed(2)} km · ${this.timeLabel()}`);
    if (this.show.arrows && s.vars.u && s.vars.v) this.drawWind(mf, s.vars.u, s.vars.v);
    this.drawMarks(mf);
    this.mapReadout(mf, (i, j) => [`${info.label.split(' / ')[0]}: ${fmt(data[j * this.grid!.nx + i]!, info.digits)} ${info.unit}`,
      ...(s.vars.u && s.vars.v ? [`風 / wind: ${fmt(Math.hypot(s.vars.u[j * this.grid!.nx + i]!, s.vars.v[j * this.grid!.nx + i]!), 1)} m/s`] : [])]);
  }

  private drawComposite(area: Rect): void {
    const mp = this.frame!.charts!.maps, name = this.sel.composite, data = mp[name];
    if (!data) { this.waiting(area); return; }
    const info = VI[name]!;
    const mf = this.drawPlan(area, name, data, `${info.label} · ${this.timeLabel()}`);
    const g = this.grid!, X = (i: number): number => mf.r.x + (i + 0.5) / g.nx * mf.r.w, Y = (j: number): number => mf.r.y + mf.r.h - (j + 0.5) / g.ny * mf.r.h;
    const slp = mp.slp;
    if (this.show.isobars && slp) {
      let lo = Infinity, hi = -Infinity; for (const v of slp) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      const step = Math.max(0.5, niceCeil((hi - lo) / 14)), lv: number[] = [];
      for (let p = Math.ceil(lo / step) * step; p <= hi; p += step) lv.push(p);
      drawContours(this.ctx, g.nx, g.ny, (i, j) => slp[j * g.nx + i]!, X, Y, lv, () => ({ color: 'rgba(250,250,250,0.85)', width: 1 }));
      this.ctx.font = FONT_SMALL; this.ctx.textAlign = 'left'; this.ctx.textBaseline = 'middle';
      haloText(this.ctx, `等壓線 / isobars: ${step} hPa`, mf.r.x, mf.r.y + mf.r.h + 30, INK.secondary);
    }
    if (this.show.sfcWind && mp.sfcU && mp.sfcV) this.drawWind(mf, mp.sfcU, mp.sfcV);
    this.drawMarks(mf);
    this.mapReadout(mf, (i, j) => [`${info.label.split(' / ')[0]}: ${fmt(data[j * g.nx + i]!, info.digits)} ${info.unit}`, ...(slp ? [`SLP ${fmt(slp[j * g.nx + i]!, 1)} hPa`] : [])]);
  }

  private waiting(r: Rect, more = ''): void { this.centerText('等待資料… / Waiting for data…' + (more ? ` / ${more}` : ''), r); }
  /** Centred bilingual message: each ' / '-separated part on its own line. */
  private centerText(t: string, r: Rect): void {
    const ctx = this.ctx, parts = t.split(' / ');
    ctx.fillStyle = INK.secondary; ctx.font = FONT; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    parts.forEach((l, i) => ctx.fillText(l, r.x + r.w / 2, r.y + r.h / 2 + (i - (parts.length - 1) / 2) * 18, r.w - 16));
  }

  /** Small reflectivity map (composite) for picking lines and points; returns the map frame. */
  private drawInset(r: Rect): MapFrame | null {
    const d = this.frame!.charts!.maps.dbzMax;
    if (!d) return null;
    const mf = this.drawPlan(r, 'dbzMax', d, '最大回波 / Composite reflectivity', false);
    this.drawMarks(mf);
    return mf;
  }

  private drawSection(area: Rect): void {
    const sec = this.frame!.charts!.section, g = this.grid!, ctx = this.ctx;
    const wide = area.w > 760;
    const inset: Rect = wide ? { x: area.x, y: area.y, w: Math.min(area.w * 0.34, area.h), h: Math.min(area.h, area.w * 0.34) } : { x: area.x, y: area.y, w: area.w, h: area.h * 0.4 };
    const main: Rect = wide ? { x: inset.x + inset.w, y: area.y, w: area.w - inset.w, h: area.h } : { x: area.x, y: area.y + inset.h, w: area.w, h: area.h - inset.h };
    this.drawInset(inset);
    if (!sec) { this.waiting(main); return; }
    const np = sec.np, nz = g.nz, name = this.sel.section, info = VI[name]!;
    const L = Math.hypot(sec.x1 - sec.x0, sec.y1 - sec.y0), cs = (sec.x1 - sec.x0) / (L || 1), sn = (sec.y1 - sec.y0) / (L || 1);
    const val = (vn: SecVar, p: number, k: number): number => {
      const o = k * np + p;
      if (vn === 'along') return sec.vars.u[o]! * cs + sec.vars.v[o]! * sn;
      if (vn === 'normal') return -sec.vars.u[o]! * sn + sec.vars.v[o]! * cs;
      return sec.vars[vn][o]!;
    };
    const data = new Float32Array(np * nz); for (let k = 0; k < nz; k++) for (let p = 0; p < np; p++) data[k * np + p] = val(name, p, k);
    const sc = scaleFor(name, data), ztop = nz * g.dz / 1000;
    const r: Rect = { x: main.x + 56, y: main.y + 44, w: Math.max(40, main.w - 56 - 76), h: Math.max(40, main.h - 44 - 40) };
    drawField(ctx, r, np, nz, (p, k) => data[k * np + p]!, sc, () => [16, 22, 30]);
    const X = (p: number): number => r.x + (p + 0.5) / np * r.w, Y = (k: number): number => r.y + r.h - (k + 0.5) / nz * r.h;
    // cloud outline (0.1 g/kg) and the 0 deg C isotherm for orientation
    if (name !== 'cloud') drawContours(ctx, np, nz, (p, k) => sec.vars.cloud[k * np + p]!, X, Y, [0.1], () => ({ color: 'rgba(245,245,245,0.9)', width: 1 }));
    drawContours(ctx, np, nz, (p, k) => sec.vars.T[k * np + p]!, X, Y, [0], () => ({ color: 'rgba(143,208,248,0.9)', width: 1, dash: [4, 3] }));
    if (this.show.vectors) {
      let vm = 0; for (let k = 0; k < nz; k++) for (let p = 0; p < np; p++) vm = Math.max(vm, Math.hypot(val('along', p, k), sec.vars.w[k * np + p]!));
      const ref = niceCeil(Math.max(vm, 1)), sp = Math.max(1, Math.round(np / 24)), sk = Math.max(1, Math.round(nz / 14)), px = Math.min(r.w / np * sp, r.h / nz * sk) * 0.95 / ref;
      for (let k = Math.floor(sk / 2); k < nz; k += sk) for (let p = Math.floor(sp / 2); p < np; p += sp) drawArrow(ctx, X(p), Y(k), val('along', p, k) * px, -sec.vars.w[k * np + p]! * px);
      ctx.font = FONT_SMALL; ctx.fillStyle = INK.secondary; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      drawArrow(ctx, r.x + r.w - ref * px, r.y - 12, ref * px, 0);
      ctx.fillText(`${ref} m/s`, r.x + r.w - ref * px - 6, r.y - 12);
    }
    drawAxes(ctx, r, { lo: 0, hi: L / 1000, label: '沿剖面距離 A→B / distance along A→B (km)' }, { lo: 0, hi: ztop, label: 'z (km)' });
    drawColorbar(ctx, { x: r.x + r.w + 12, y: r.y + 14, w: 12, h: r.h - 14 }, sc, info.unit, info.digits);
    ctx.font = FONT; ctx.fillStyle = INK.primary; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText(`${info.label} · ${this.timeLabel()}`, r.x, r.y - 26, r.w);
    ctx.font = FONT_SMALL; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = INK.secondary;
    ctx.fillText('A', r.x, r.y + r.h + 18); ctx.textAlign = 'right'; ctx.fillText('B', r.x + r.w, r.y + r.h + 18);
    this.plotF = { r, kind: 'section', x0: 0, x1: L / 1000, y0: 0, y1: ztop };
    const h = this.hover;
    if (h && !this.drag && h.x >= r.x && h.x <= r.x + r.w && h.y >= r.y && h.y <= r.y + r.h) {
      const p = Math.min(np - 1, Math.floor((h.x - r.x) / r.w * np)), k = Math.min(nz - 1, Math.floor((r.y + r.h - h.y) / r.h * nz));
      tooltip(ctx, [`s ${(p / Math.max(1, np - 1) * L / 1000).toFixed(1)} km, z ${((k + 0.5) * g.dz / 1000).toFixed(2)} km`, `${info.label.split(' / ')[0]}: ${fmt(val(name, p, k), info.digits)} ${info.unit}`,
        `T ${fmt(sec.vars.T[k * np + p]!, 1)} °C · w ${fmt(sec.vars.w[k * np + p]!, 1)} m/s`], h.x, h.y, this.canvas.clientWidth, this.canvas.clientHeight);
    }
  }

  private drawRZ(area: Rect): void {
    const rz = this.frame!.charts!.rz, g = this.grid!, ctx = this.ctx;
    if (!rz) { this.waiting(area); return; }
    const { nr, dr } = rz, nz = g.nz, name = this.sel.rz, info = VI[name]!, data = rz.vars[name];
    const R = nr * dr / 1000, ztop = nz * g.dz / 1000;
    const r: Rect = { x: area.x + 56, y: area.y + 26, w: Math.max(40, area.w - 56 - 80), h: Math.max(40, area.h - 26 - 44) };
    const sc = scaleFor(name, data);
    drawField(ctx, r, nr, nz, (i, k) => data[k * nr + i]!, sc, () => [16, 22, 30]);
    const X = (i: number): number => r.x + (i + 0.5) / nr * r.w, Y = (k: number): number => r.y + r.h - (k + 0.5) / nz * r.h;
    const cn = name === 'vt' ? 'w' : 'vt', cd = rz.vars[cn];
    let am = 0; for (const v of cd) am = Math.max(am, Math.abs(v));
    const step = cn === 'vt' ? Math.max(2, niceCeil(am / 7)) : Math.max(0.25, niceCeil(am / 5)), lv: number[] = [];
    for (let v = step; v <= am; v += step) lv.push(v, -v);
    drawContours(ctx, nr, nz, (i, k) => cd[k * nr + i]!, X, Y, lv, (L) => ({ color: L > 0 ? 'rgba(250,250,250,0.9)' : 'rgba(250,250,250,0.6)', width: 1, dash: L < 0 ? [4, 3] : [] }));
    drawAxes(ctx, r, { lo: 0, hi: R, label: '半徑 / radius (km)' }, { lo: 0, hi: ztop, label: 'z (km)' });
    drawColorbar(ctx, { x: r.x + r.w + 12, y: r.y + 14, w: 12, h: r.h - 14 }, sc, info.unit, info.digits);
    ctx.font = FONT; ctx.fillStyle = INK.primary; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText(`${info.label}（方位平均 / azimuthal mean）· ${this.timeLabel()}`, r.x, r.y - 8);
    ctx.font = FONT_SMALL; ctx.fillStyle = INK.secondary; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
    ctx.fillText(`等值線 / contours: ${cn === 'vt' ? 'vt' : 'w'} ${step} m/s · 中心 / centre (${(rz.xc / 1000).toFixed(0)}, ${(rz.yc / 1000).toFixed(0)}) km`, r.x + r.w, area.y + area.h - 2, r.w);
    this.plotF = { r, kind: 'rz', x0: 0, x1: R, y0: 0, y1: ztop };
    const h = this.hover;
    if (h && h.x >= r.x && h.x <= r.x + r.w && h.y >= r.y && h.y <= r.y + r.h) {
      const i = Math.min(nr - 1, Math.floor((h.x - r.x) / r.w * nr)), k = Math.min(nz - 1, Math.floor((r.y + r.h - h.y) / r.h * nz)), o = k * nr + i;
      tooltip(ctx, [`r ${((i + 0.5) * dr / 1000).toFixed(1)} km, z ${((k + 0.5) * g.dz / 1000).toFixed(2)} km`, `vt ${fmt(rz.vars.vt[o]!, 1)} · vr ${fmt(rz.vars.vr[o]!, 1)} · w ${fmt(rz.vars.w[o]!, 2)} m/s`,
        `θ′ ${fmt(rz.vars.thp[o]!, 2)} K · ${fmt(rz.vars.cond[o]!, 2)} g/kg`], h.x, h.y, this.canvas.clientWidth, this.canvas.clientHeight);
    }
  }

  // ---------------------------------------------------------------- sounding

  private drawSounding(area: Rect): void {
    const snd = this.frame!.charts!.sounding, g = this.grid!, ctx = this.ctx;
    const wide = area.w > 820;
    const side = wide ? Math.min(area.w * 0.3, area.h * 0.55) : Math.min(area.w * 0.5, area.h * 0.36);
    const inset: Rect = { x: area.x, y: area.y, w: side, h: side };
    this.drawInset(inset);
    const skew: Rect = wide ? { x: area.x + side + 50, y: area.y + 26, w: area.w - side - 50 - 16, h: area.h - 26 - 40 } : { x: area.x + 46, y: area.y + side + 20, w: area.w - 46 - 12, h: area.h - side - 20 - 36 };
    const text: Rect = wide ? { x: area.x + 14, y: area.y + side + 8, w: side - 14, h: area.h - side - 8 } : { x: area.x + side + 12, y: area.y + 14, w: area.w - side - 12, h: side };
    if (!snd) { this.waiting(skew); return; }
    const nz = g.nz, dz = g.dz, zc = (k: number): number => (k + 0.5) * dz;
    const T = snd.vars.T, Td = snd.vars.Td, p = snd.vars.p, u = snd.vars.u, v = snd.vars.v, qv = snd.vars.qv;
    // parcel
    const Tk = Float64Array.from(T, (x) => x + 273.15), pp = Float64Array.from(p, (x) => x * 100), q = Float64Array.from(qv, (x) => x / 1000), Tp = new Float64Array(nz);
    const pc = parcelAscent(Tk, pp, q, dz, Tp);
    // --- skew-T log-p frame
    const pBot = Math.max(1050, Math.ceil(p[0]! / 50) * 50 + 20), pTop = Math.max(100, Math.floor(p[nz - 1]! / 50) * 50);
    const Tl = Math.round((T[0]! - 45) / 10) * 10, Tr = Tl + 80;
    const Y = (pr: number): number => skew.y + skew.h * (Math.log(pr) - Math.log(pTop)) / (Math.log(pBot) - Math.log(pTop));
    const X = (t: number, pr: number): number => skew.x + (t - Tl) / (Tr - Tl) * skew.w + (skew.y + skew.h - Y(pr)) * 0.9;
    ctx.save(); ctx.beginPath(); ctx.rect(skew.x, skew.y, skew.w, skew.h); ctx.clip();
    const curve = (fT: (pr: number) => number, color: string, width: number, dash: number[] = []): void => {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.beginPath();
      for (let pr = pBot, first = true; pr >= pTop - 1e-9; pr -= 10, first = false) { const x = X(fT(pr), pr), y = Y(pr); if (first) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
      ctx.stroke(); ctx.setLineDash([]);
    };
    for (let t = -120; t <= 50; t += 10) { ctx.strokeStyle = t === 0 ? '#4a5a6c' : INK.grid; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(t, pBot), Y(pBot)); ctx.lineTo(X(t, pTop), Y(pTop)); ctx.stroke(); }
    for (let th = 250; th <= 470; th += 10) curve((pr) => th * Math.pow(pr / 1000, 287.05 / 1004.5) - 273.15, 'rgba(160,110,70,0.35)', 1);
    for (let t0 = -10; t0 <= 36; t0 += 4) {
      const prs: number[] = [], tt: number[] = [];
      for (let pr = 1050; pr >= pTop - 1e-9; pr -= 10) prs.push(pr);
      let th = 0;
      // pseudo-adiabat through (t0, 1000 hPa): saturation adjustment of a saturated parcel in 10 hPa steps
      const start = t0 + 273.15, qs0 = sat(start, 1e5);
      th = start * Math.pow(1000 / 1000, 0.2857); void th;
      const upT = moistAdiabat(start, qs0, 1000, prs);
      for (const x of upT) tt.push(x);
      ctx.strokeStyle = 'rgba(90,150,120,0.35)'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath();
      prs.forEach((pr, i) => { const x = X(tt[i]! - 273.15, pr), y = Y(pr); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      ctx.stroke(); ctx.setLineDash([]);
    }
    for (const pr of [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100]) if (pr <= pBot && pr >= pTop) { ctx.fillStyle = INK.grid; ctx.fillRect(skew.x, Y(pr), skew.w, 1); }
    // CAPE / CIN shading between the parcel and the environment
    if (pc.lfc > 0) {
      const shade = (k0: number, k1: number, color: string): void => {
        ctx.fillStyle = color; ctx.beginPath();
        for (let k = k0; k <= k1; k++) { const x = X(Tp[k]! - 273.15, p[k]!), y = Y(p[k]!); if (k === k0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
        for (let k = k1; k >= k0; k--) ctx.lineTo(X(T[k]!, p[k]!), Y(p[k]!));
        ctx.closePath(); ctx.fill();
      };
      shade(Math.max(0, pc.lfc - 1), Math.max(pc.lfc, pc.el), 'rgba(230,103,103,0.22)');
      shade(0, pc.lfc, 'rgba(57,135,229,0.22)');
    }
    const prof = (a: ArrayLike<number>, color: string, width: number, dash: number[] = []): void => {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.beginPath();
      for (let k = 0; k < nz; k++) { const x = X(a[k]!, p[k]!), y = Y(p[k]!); if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
      ctx.stroke(); ctx.setLineDash([]);
    };
    prof(Array.from(Tp, (x) => x - 273.15), '#c3c2b7', 1.5, [5, 3]);
    prof(Td, '#1baf7a', 2);
    prof(T, '#e66767', 2);
    ctx.restore();
    // axes and labels
    ctx.strokeStyle = INK.axis; ctx.lineWidth = 1; ctx.strokeRect(skew.x + 0.5, skew.y + 0.5, skew.w - 1, skew.h - 1);
    ctx.font = FONT_SMALL; ctx.fillStyle = INK.secondary; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const pr of [1000, 850, 700, 500, 400, 300, 200, 150, 100]) if (pr <= pBot && pr >= pTop) ctx.fillText(String(pr), skew.x - 4, Y(pr));
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let t = Tl; t <= Tr; t += 10) { const x = X(t, pBot); if (x >= skew.x && x <= skew.x + skew.w) ctx.fillText(`${t}`, x, skew.y + skew.h + 4); }
    ctx.fillText('T (°C)，等溫線斜 / skewed isotherms', skew.x + skew.w / 2, skew.y + skew.h + 18);
    ctx.save(); ctx.translate(skew.x - 36, skew.y + skew.h / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle'; ctx.fillText('p (hPa)', 0, 0); ctx.restore();
    // wind barbs on the right edge
    const every = Math.max(1, Math.round(nz / 22));
    ctx.save(); ctx.beginPath(); ctx.rect(skew.x, skew.y, skew.w, skew.h); ctx.clip();
    for (let k = 0; k < nz; k += every) drawBarb(ctx, skew.x + skew.w - 30, Y(p[k]!), u[k]!, v[k]!, 22);
    ctx.restore();
    ctx.font = FONT; ctx.fillStyle = INK.primary; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText(`探空 / Sounding (${(snd.x / 1000).toFixed(1)}, ${(snd.y / 1000).toFixed(1)}) km · ${this.timeLabel()}`, skew.x, skew.y - 8);
    // legend
    const lg: [string, string][] = [['#e66767', '溫度 / T'], ['#1baf7a', '露點 / Td'], ['#c3c2b7', '地面氣塊 / surface parcel']];
    ctx.font = FONT_SMALL; ctx.textBaseline = 'middle';
    lg.forEach(([c, t], i) => { const y = skew.y + 12 + 15 * i; ctx.fillStyle = c; ctx.fillRect(skew.x + 8, y - 1, 16, 2); ctx.fillStyle = INK.secondary; ctx.fillText(t, skew.x + 28, y); });
    // --- hodograph inset (top right of the skew-T) and indices
    const hs = Math.min(skew.w * 0.36, skew.h * 0.42, 230), hr: Rect = { x: skew.x + skew.w - hs - 40, y: skew.y + 8, w: hs, h: hs };
    const ind = indices(u, v, T, p, qv, nz, dz);
    this.drawHodograph(hr, u, v, nz, dz, ind.rm);
    const lines = [
      `CAPE ${pc.cape.toFixed(0)} J/kg · CIN ${pc.cin.toFixed(0)} J/kg`,
      `LCL ${pc.lcl >= 0 ? (zc(pc.lcl) / 1000).toFixed(1) : '—'} km · LFC ${pc.lfc >= 0 ? (zc(pc.lfc) / 1000).toFixed(1) : '—'} km · EL ${pc.el >= 0 ? (zc(pc.el) / 1000).toFixed(1) : '—'} km`,
      `可降水量 / PW ${ind.pw.toFixed(1)} mm`,
      `0–6 km 風切 / shear ${ind.shear6.toFixed(1)} m/s`,
      `SRH 0–1 km ${ind.srh1.toFixed(0)} · 0–3 km ${ind.srh3.toFixed(0)} m²/s²`,
      `右移胞移動 / Bunkers RM (${ind.rm.u.toFixed(1)}, ${ind.rm.v.toFixed(1)}) m/s`,
      `地面 / surface ${T[0]!.toFixed(1)} °C, Td ${Td[0]!.toFixed(1)} °C, ${p[0]!.toFixed(1)} hPa`,
      '風標：短桿 2.5、長桿 5、旗 25 m/s / barbs: half 2.5, full 5, pennant 25 m/s',
    ];
    ctx.font = FONT_SMALL; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    lines.forEach((l, i) => { ctx.fillStyle = i < 2 ? INK.primary : INK.secondary; ctx.fillText(l, text.x, text.y + 4 + 16 * i, text.w); });
  }

  private drawHodograph(r: Rect, u: Float32Array, v: Float32Array, nz: number, dz: number, rm: { u: number; v: number }): void {
    const ctx = this.ctx, kTop = Math.min(nz - 1, Math.round(10000 / dz - 0.5));
    let vm = 10; for (let k = 0; k <= kTop; k++) vm = Math.max(vm, Math.hypot(u[k]!, v[k]!), Math.hypot(rm.u, rm.v));
    const R = niceCeil(vm * 1.05), cx = r.x + r.w / 2, cy = r.y + r.h / 2, s = r.w / 2 / R;
    ctx.fillStyle = 'rgba(11,16,23,0.85)'; ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.strokeStyle = INK.grid; ctx.lineWidth = 1;
    const ring = niceCeil(R / 3);
    for (let q = ring; q <= R + 1e-9; q += ring) { ctx.beginPath(); ctx.arc(cx, cy, q * s, 0, 2 * Math.PI); ctx.stroke(); }
    ctx.beginPath(); ctx.moveTo(r.x, cy); ctx.lineTo(r.x + r.w, cy); ctx.moveTo(cx, r.y); ctx.lineTo(cx, r.y + r.h); ctx.stroke();
    ctx.strokeStyle = INK.axis; ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
    const layers: [number, number][] = [[0, 1000], [1000, 3000], [3000, 6000], [6000, 10000]];
    const wAt = (z: number): [number, number] => { const kf = Math.max(0, Math.min(nz - 1, z / dz - 0.5)), k0 = Math.floor(kf), k1 = Math.min(nz - 1, k0 + 1), f = kf - k0; return [u[k0]! + f * (u[k1]! - u[k0]!), v[k0]! + f * (v[k1]! - v[k0]!)]; };
    layers.forEach(([z0, z1], li) => {
      ctx.strokeStyle = SERIES[li]!; ctx.lineWidth = 2; ctx.beginPath();
      for (let z = z0, first = true; z <= z1 + 1e-6; z += Math.min(dz / 2, 250), first = false) { const [a, b] = wAt(z); if (first) ctx.moveTo(cx + a * s, cy - b * s); else ctx.lineTo(cx + a * s, cy - b * s); }
      ctx.stroke();
    });
    ctx.fillStyle = INK.primary; ctx.strokeStyle = INK.surface; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(cx + rm.u * s, cy - rm.v * s, 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    ctx.font = FONT_SMALL; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; haloText(ctx, 'RM', cx + rm.u * s + 6, cy - rm.v * s);
    ctx.textBaseline = 'top'; ctx.fillStyle = INK.secondary; ctx.fillText(`風徑圖 / Hodograph (${ring} m/s rings)`, r.x + 4, r.y + 3);
    ['0–1', '1–3', '3–6', '6–10 km'].forEach((t, i) => { const y = r.y + r.h - 12 - 13 * (3 - i); ctx.fillStyle = SERIES[i]!; ctx.fillRect(r.x + 6, y + 5, 12, 2); ctx.fillStyle = INK.secondary; ctx.fillText(t, r.x + 22, y); });
  }

  // ---------------------------------------------------------------- time series

  private drawSeries(area: Rect): void {
    const ctx = this.ctx, S = this.samples;
    if (S.length < 2) { this.centerText('等待資料… / Waiting for data… / 時間序列在模式執行時累積 / series accumulate while the model runs', area); return; }
    const tc = this.grid && isTcExperiment(this.grid.experiment);
    type P = { label: string; unit: string; get: (s: Sample) => number | null; digits: number };
    const panels: P[] = tc ? [
      { label: '中心氣壓降 / Central pressure deficit', unit: 'hPa', get: (s) => s.dp, digits: 1 },
      { label: '最大地面風 / Max surface wind', unit: 'm/s', get: (s) => s.vmax, digits: 1 },
      { label: '最大風速半徑 / Radius of max wind', unit: 'km', get: (s) => (s.rmw === null ? null : s.rmw / 1000), digits: 0 },
      { label: '最大上升速度 / Max updraft', unit: 'm/s', get: (s) => s.wmax, digits: 1 },
      { label: '外圍雨量（100–300 km，雨帶）/ Outer rain (rainbands)', unit: 'mm/h', get: (s) => s.tcRain?.outer ?? null, digits: 2 },
    ] : [
      { label: '最大上升速度 / Max updraft', unit: 'm/s', get: (s) => s.wmax, digits: 1 },
      { label: '最大上升氣流螺旋度（2–5 km）/ Max updraft helicity', unit: 'm²/s²', get: (s) => s.uh, digits: 0 },
      { label: '近地面最大垂直渦度 / Max near-surface vorticity', unit: 's⁻¹', get: (s) => s.zeta, digits: 3 },
      { label: '最大對地地面風 / Max ground-relative surface wind', unit: 'm/s', get: (s) => s.vg, digits: 1 },
      { label: '最大回波 / Max reflectivity', unit: 'dBZ', get: (s) => s.dbz, digits: 0 },
    ];
    const hasTrack = S.some((s) => s.storm);
    const wide = area.w > 820;
    const trackR: Rect | null = hasTrack ? (wide ? { x: area.x + area.w * 0.64 + 40, y: area.y + 26, w: area.w * 0.36 - 60, h: Math.min(area.h - 70, area.w * 0.36 - 60) } : null) : null;
    const colW = trackR ? area.w * 0.64 : area.w;
    const t0 = S[0]!.t / 3600, t1 = Math.max(t0 + 1e-3, S[S.length - 1]!.t / 3600);
    const ph = (area.h - 30) / panels.length;
    const rects: Rect[] = [];
    const hover = this.hover;
    let hi = -1;
    panels.forEach((pn, n) => {
      const r: Rect = { x: area.x + 64, y: area.y + 20 + n * ph, w: colW - 64 - 24, h: ph - 44 };
      rects.push(r);
      let lo = Infinity, hiV = -Infinity;
      for (const s of S) { const v = pn.get(s); if (v !== null && Number.isFinite(v)) { lo = Math.min(lo, v); hiV = Math.max(hiV, v); } }
      if (!Number.isFinite(lo)) { lo = 0; hiV = 1; }
      if (hiV - lo < 1e-9) { hiV = lo + 1; }
      const pad = 0.05 * (hiV - lo); lo -= pad; hiV += pad;
      if (lo > 0 && lo < 0.3 * hiV) lo = 0;
      drawAxes(ctx, r, { lo: t0, hi: t1, label: n === panels.length - 1 ? '模式時間 / model time (h)' : '' }, { lo, hi: hiV, label: '' }, true);
      const X = (t: number): number => r.x + (t - t0) / (t1 - t0) * r.w, Y = (v: number): number => r.y + r.h - (v - lo) / (hiV - lo) * r.h;
      ctx.strokeStyle = SERIES[0]!; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.beginPath();
      let pen = false;
      for (const s of S) { const v = pn.get(s); if (v === null || !Number.isFinite(v)) { pen = false; continue; } const x = X(s.t / 3600), y = Y(v); if (pen) ctx.lineTo(x, y); else ctx.moveTo(x, y); pen = true; }
      ctx.stroke();
      const lastV = pn.get(S[S.length - 1]!);
      ctx.font = FONT_SMALL; ctx.fillStyle = INK.primary; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(`${pn.label}（${pn.unit}）${lastV === null ? '' : ' · ' + fmt(lastV, pn.digits)}`, r.x, r.y - 3);
      if (hover && hover.x >= r.x && hover.x <= r.x + r.w && hover.y >= area.y && hover.y <= area.y + area.h) {
        const th = t0 + (hover.x - r.x) / r.w * (t1 - t0);
        if (hi < 0) { let best = 0; for (let i = 1; i < S.length; i++) if (Math.abs(S[i]!.t / 3600 - th) < Math.abs(S[best]!.t / 3600 - th)) best = i; hi = best; }
      }
    });
    if (hi >= 0) {
      const s = S[hi]!;
      rects.forEach((r, n) => {
        const x = r.x + (s.t / 3600 - t0) / (t1 - t0) * r.w;
        ctx.fillStyle = INK.secondary; ctx.fillRect(x, r.y, 1, r.h);
      });
      tooltip(ctx, [`t = ${(s.t / 3600).toFixed(2)} h`, ...panels.map((pn) => { const v = pn.get(s); return `${pn.label.split(' / ')[0]}: ${v === null ? '—' : fmt(v, pn.digits)} ${pn.unit}`; })], hover!.x, hover!.y, this.canvas.clientWidth, this.canvas.clientHeight);
    }
    this.plotF = { r: rects[0]!, kind: 'series', x0: t0, x1: t1, y0: 0, y1: 1, panels: rects };
    if (trackR) this.drawTrack(trackR, hi >= 0 ? S[hi]! : null);
  }

  /** Storm track (ground-relative positions of the strongest storm column, or the TC centre). */
  private drawTrack(r: Rect, mark: Sample | null): void {
    const ctx = this.ctx, pts = this.samples.filter((s) => s.storm);
    if (pts.length === 0) return;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const s of pts) { x0 = Math.min(x0, s.storm!.x); x1 = Math.max(x1, s.storm!.x); y0 = Math.min(y0, s.storm!.y); y1 = Math.max(y1, s.storm!.y); }
    const span = Math.max(x1 - x0, y1 - y0, 10000) * 1.15, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const lo = { x: (cx - span / 2) / 1000, y: (cy - span / 2) / 1000 }, hiK = { x: (cx + span / 2) / 1000, y: (cy + span / 2) / 1000 };
    drawAxes(ctx, r, { lo: lo.x, hi: hiK.x, label: '對地 x / ground-relative x (km)' }, { lo: lo.y, hi: hiK.y, label: 'y (km)' }, true);
    const X = (x: number): number => r.x + (x / 1000 - lo.x) / (hiK.x - lo.x) * r.w, Y = (y: number): number => r.y + r.h - (y / 1000 - lo.y) / (hiK.y - lo.y) * r.h;
    ctx.save(); ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.clip();
    ctx.strokeStyle = SERIES[0]!; ctx.lineWidth = 2; ctx.beginPath();
    pts.forEach((s, i) => { if (i === 0) ctx.moveTo(X(s.storm!.x), Y(s.storm!.y)); else ctx.lineTo(X(s.storm!.x), Y(s.storm!.y)); });
    ctx.stroke();
    const dot = (s: Sample, filled: boolean): void => { ctx.beginPath(); ctx.arc(X(s.storm!.x), Y(s.storm!.y), 5, 0, 2 * Math.PI); ctx.lineWidth = 2; ctx.strokeStyle = INK.surface; if (filled) { ctx.fillStyle = INK.primary; ctx.fill(); } ctx.stroke(); if (!filled) { ctx.strokeStyle = INK.primary; ctx.lineWidth = 1.5; ctx.stroke(); } };
    dot(pts[0]!, false); dot(pts[pts.length - 1]!, true);
    if (mark?.storm) { ctx.beginPath(); ctx.arc(X(mark.storm.x), Y(mark.storm.y), 7, 0, 2 * Math.PI); ctx.strokeStyle = '#fafafa'; ctx.lineWidth = 1.5; ctx.stroke(); }
    ctx.restore();
    ctx.font = FONT_SMALL; ctx.fillStyle = INK.primary; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    const tc = this.grid && isTcExperiment(this.grid.experiment);
    ctx.fillText(tc ? '颱風中心路徑 / TC centre track' : '風暴路徑 / Storm track', r.x, r.y - 4);
    ctx.fillStyle = INK.secondary; ctx.textBaseline = 'top';
    ctx.fillText('○ 起點 / start  ● 目前 / now' + (tc ? '' : ' · 位置：最強 UH（無旋轉時為最強上升氣流）/ position: max UH (max updraft before rotation)'), r.x, r.y + r.h + 32, r.w + 40);
  }

  // ---------------------------------------------------------------- Hovmoller

  private drawHovmoller(area: Rect): void {
    const ctx = this.ctx, Hv = this.hov;
    if (Hv.length < 2) {
      this.centerText('Hovmöller 圖需要熱帶氣旋實驗 / needs a tropical-cyclone experiment / 在模式執行時逐列累積 / rows accumulate while the model runs', area);
      return;
    }
    const nr = Math.max(...Hv.map((h) => h.vt.length)), dr = Hv[Hv.length - 1]!.dr;
    const R = Math.min(nr * dr, 300000) / 1000, nrs = Math.max(1, Math.round(R * 1000 / dr));
    const t0 = Hv[0]!.t / 3600, t1 = Hv[Hv.length - 1]!.t / 3600;
    const r: Rect = { x: area.x + 56, y: area.y + 26, w: Math.max(40, area.w - 56 - 80), h: Math.max(40, area.h - 26 - 44) };
    const rows = Math.max(2, Math.min(600, Math.round(r.h)));
    let vm = 0; for (const h of Hv) for (let i = 0; i < Math.min(nrs, h.vt.length); i++) vm = Math.max(vm, Math.abs(h.vt[i]!));
    const sc: Scale = { kind: 'div', lo: -niceCeil(vm || 1), hi: niceCeil(vm || 1), gamma: 1, clear: null, reverse: false };
    let idx = 0;
    const rowOf = new Int32Array(rows);
    for (let y = 0; y < rows; y++) {
      const t = t0 + (y + 0.5) / rows * (t1 - t0);
      while (idx < Hv.length - 1 && Hv[idx + 1]!.t / 3600 <= t) idx++;
      rowOf[y] = idx;
    }
    drawField(ctx, r, nrs, rows, (i, y) => { const h = Hv[rowOf[y]!]!; return i < h.vt.length ? h.vt[i]! : NaN; }, sc, () => [16, 22, 30]);
    // radius of maximum wind (surface) on top
    const S = this.samples.filter((s) => s.rmw !== null && s.t / 3600 >= t0 - 1e-9);
    if (S.length > 1) {
      ctx.strokeStyle = '#fafafa'; ctx.lineWidth = 1.5; ctx.beginPath();
      S.forEach((s, i) => { const x = r.x + Math.min(1, s.rmw! / 1000 / R) * r.w, y = r.y + r.h - (s.t / 3600 - t0) / ((t1 - t0) || 1) * r.h; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      ctx.stroke();
    }
    // eyewalls: local maxima of the 1.5 km tangential wind (two or more at a time = concentric eyewalls)
    const E = this.samples.filter((s) => s.ew && s.ew.length && s.t / 3600 >= t0 - 1e-9);
    const stride = Math.max(1, Math.ceil(E.length / 400));
    for (let n = 0; n < E.length; n += stride) {
      const s = E[n]!, y = r.y + r.h - (s.t / 3600 - t0) / ((t1 - t0) || 1) * r.h;
      s.ew!.forEach((e, k) => {
        const x = r.x + Math.min(1, e.r / 1000 / R) * r.w;
        ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = k === 0 ? '#fafafa' : '#c98500'; ctx.fill();
      });
    }
    drawAxes(ctx, r, { lo: 0, hi: R, label: '半徑 / radius (km)' }, { lo: t0, hi: t1, label: '模式時間 / model time (h)' });
    drawColorbar(ctx, { x: r.x + r.w + 12, y: r.y + 14, w: 12, h: r.h - 14 }, sc, 'm/s', 1);
    ctx.font = FONT; ctx.fillStyle = INK.primary; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText('1.5 km 方位平均切向風 / 1.5 km azimuthal-mean tangential wind', r.x, r.y - 8);
    ctx.font = FONT_SMALL; ctx.fillStyle = INK.secondary; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
    ctx.fillText('白線：地面最大風半徑；點：眼牆（白：最內圈、橘：外圈）/ line: surface RMW; dots: eyewalls (white: innermost, orange: outer)', r.x + r.w, area.y + area.h - 2, r.w);
    const h = this.hover;
    if (h && h.x >= r.x && h.x <= r.x + r.w && h.y >= r.y && h.y <= r.y + r.h) {
      const i = Math.min(nrs - 1, Math.floor((h.x - r.x) / r.w * nrs)), y = Math.min(rows - 1, Math.floor((r.y + r.h - h.y) / r.h * rows)), row = Hv[rowOf[y]!]!;
      tooltip(ctx, [`t ${(row.t / 3600).toFixed(2)} h, r ${((i + 0.5) * dr / 1000).toFixed(0)} km`, `vt ${fmt(row.vt[i] ?? NaN, 1)} m/s`], h.x, h.y, this.canvas.clientWidth, this.canvas.clientHeight);
    }
    this.plotF = { r, kind: 'hov', x0: 0, x1: R, y0: t0, y1: t1 };
  }
}

// ---------------------------------------------------------------- thermodynamics for the skew-T

function sat(T: number, p: number): number { const es = 611.2 * Math.exp(17.67 * (T - 273.15) / (T - 29.65)); return 0.622 * es / Math.max(p - es, 1); }
/** Pseudo-adiabat through saturated (T0, p0 hPa): temperatures (K) at the given pressures (hPa, descending),
 *  by saturation adjustment after dry steps (as parcelAscent). */
function moistAdiabat(T0: number, q0: number, p0: number, prs: number[]): number[] {
  const out: number[] = [];
  let th = T0 * Math.pow(1000 / p0, 0.2857), q = q0;
  // levels below p0 (higher pressure): follow the same curve downward approximately with the saturated lapse
  for (const pr of prs) {
    const pi = Math.pow(pr / 1000, 287.05 / 1004.5);
    let t = th * pi;
    if (pr > p0) { out.push(t); continue; }
    let d = 0;
    for (let it = 0; it < 4; it++) {
      const tt = t + 2.5e6 * d / 1004.5, es = 611.2 * Math.exp(17.67 * (tt - 273.15) / (tt - 29.65)), qs = 0.622 * es / Math.max(pr * 100 - es, 1);
      const dq = qs * pr * 100 / Math.max(pr * 100 - es, 1) * 17.67 * 243.5 / ((tt - 29.65) ** 2);
      d += (q - d - qs) / (1 + 2.5e6 / 1004.5 * dq);
    }
    d = Math.max(-q0, Math.min(q, d));
    t += 2.5e6 * d / 1004.5; q -= d; th = t / pi;
    out.push(t);
  }
  return out;
}

/** Sounding indices: precipitable water, 0-6 km bulk shear, Bunkers right-mover motion, storm-relative helicity. */
function indices(u: Float32Array, v: Float32Array, T: Float32Array, p: Float32Array, qv: Float32Array, nz: number, dz: number): { pw: number; shear6: number; rm: { u: number; v: number }; srh1: number; srh3: number } {
  let pw = 0;
  for (let k = 0; k < nz; k++) { const rho = p[k]! * 100 / (287.05 * (T[k]! + 273.15) * (1 + 0.61e-3 * qv[k]!)); pw += rho * qv[k]! / 1000 * dz; }
  const at = (z: number): [number, number] => { const kf = Math.max(0, Math.min(nz - 1, z / dz - 0.5)), k0 = Math.floor(kf), k1 = Math.min(nz - 1, k0 + 1), f = kf - k0; return [u[k0]! + f * (u[k1]! - u[k0]!), v[k0]! + f * (v[k1]! - v[k0]!)]; };
  const mean = (z0: number, z1: number): [number, number] => { let a = 0, b = 0; const n = 60; for (let i = 0; i < n; i++) { const w = at(z0 + (i + 0.5) * (z1 - z0) / n); a += w[0]; b += w[1]; } return [a / n, b / n]; };
  const s0 = at(0), s6 = at(6000);
  const m = mean(0, 6000), lo = mean(0, 500), hi = mean(5500, 6000), sx = hi[0] - lo[0], sy = hi[1] - lo[1], sl = Math.hypot(sx, sy) || 1;
  const rm = { u: m[0] + 7.5 * sy / sl, v: m[1] - 7.5 * sx / sl };
  const srh = (ztop: number): number => { let h = 0; const n = Math.max(4, Math.round(ztop / 100)); for (let i = 0; i < n; i++) { const a = at(i * ztop / n), b = at((i + 1) * ztop / n); h += (b[0] - rm.u) * (a[1] - rm.v) - (a[0] - rm.u) * (b[1] - rm.v); } return h; };
  return { pw, shear6: Math.hypot(s6[0] - s0[0], s6[1] - s0[1]), rm, srh1: srh(1000), srh3: srh(3000) };
}
void colorOf; void ticks;
