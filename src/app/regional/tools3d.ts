// Interaction tools of the 3-D view: pick a tool, then tap the ground under the place (the height comes from the
// tool's slider) or drag to paint. A preview ring shows where and how big; lasting winds stay drawn until they end.
import type { VolumeView } from './volume.js';
import type { ForcingInfo, ToRegionalWorker } from './protocol.js';

type Tool = 'view' | 'warm' | 'cold' | 'moist' | 'dry' | 'wind' | 'land' | 'sea' | 'warmer' | 'cooler';
const TOOLS: { t: Tool; label: string; tip: string }[] = [
  { t: 'view', label: '視角', tip: '轉動相機（不互動）/ camera only' },
  { t: 'warm', label: '暖泡', tip: '點地面：在設定高度放暖泡 +3 K / tap: warm bubble at the set height' },
  { t: 'cold', label: '冷池', tip: '點地面：放冷空氣 −6 K / tap: cold air' },
  { t: 'moist', label: '增濕', tip: '點地面：水氣 ×1.3（最多到飽和）/ tap: vapour ×1.3 (up to saturation)' },
  { t: 'dry', label: '變乾', tip: '點地面：水氣 ×0.7 / tap: vapour ×0.7' },
  { t: 'wind', label: '風', tip: '點地面：加一陣風或持續的風（方向、仰角、形式、範圍可選）/ tap: a wind once or lasting' },
  { t: 'land', label: '陸地', tip: '拖曳塗陸地 / drag: paint land' },
  { t: 'sea', label: '海洋', tip: '拖曳塗海洋 / drag: paint sea' },
  { t: 'warmer', label: '海溫+', tip: '拖曳：海溫 +2 °C / drag: sea 2 °C warmer' },
  { t: 'cooler', label: '海溫−', tip: '拖曳：海溫 −2 °C / drag: sea 2 °C cooler' },
];
const PAINT: Tool[] = ['land', 'sea', 'warmer', 'cooler'];
const DIRS = ['北 N', '東北 NE', '東 E', '東南 SE', '南 S', '西南 SW', '西 W', '西北 NW'];

interface Grid { Lx: number; Ly: number; top: number; dx: number; dz: number; /** the run has a surface to paint; takes interactions at all (not the axisymmetric model) */ paint: boolean; interact: boolean }

export class Tools3D {
  private tool: Tool = 'view';
  private grid: Grid | null = null;
  private hover: { x: number; y: number } | null = null;
  private flash: { x: number; y: number; t: number } | null = null;
  private forcings: ForcingInfo[] = [];
  private lastPaint: { x: number; y: number } | null = null;
  private readonly opts: HTMLElement;
  private readonly svg: SVGSVGElement;
  private readonly el: Record<string, HTMLInputElement | HTMLSelectElement> = {};

  constructor(private readonly view: VolumeView, root: HTMLElement, svg: SVGSVGElement, private readonly send: (m: ToRegionalWorker) => void, private readonly log: (s: string) => void) {
    this.svg = svg;
    const bar = document.createElement('div'); bar.className = 'tbtns';
    for (const d of TOOLS) {
      const b = document.createElement('button'); b.textContent = d.label; b.title = d.tip; b.dataset.t = d.t;
      if (d.t === this.tool) b.className = 'on';
      b.onclick = (): void => this.pick(d.t);
      bar.append(b);
    }
    this.opts = document.createElement('div'); this.opts.className = 'topts'; this.opts.hidden = true;
    root.append(bar, this.opts);
    const row = (label: string, input: HTMLInputElement | HTMLSelectElement, key: string, cls = ''): HTMLElement => {
      const l = document.createElement('label'); l.className = 'to ' + cls;
      const s = document.createElement('span'); s.textContent = label; l.append(s, input); this.el[key] = input;
      input.addEventListener('input', () => this.drawOverlay()); return l;
    };
    const num = (v: number, step: number): HTMLInputElement => { const i = document.createElement('input'); i.type = 'number'; i.value = String(v); i.step = String(step); return i; };
    const sel = (opts: [string, string][], v: string): HTMLSelectElement => { const s = document.createElement('select'); for (const [a, b] of opts) s.append(new Option(b, a)); s.value = v; return s; };
    const hgt = document.createElement('input'); hgt.type = 'range'; hgt.min = '0'; hgt.max = '16'; hgt.step = '0.25'; hgt.value = '1.5';
    const hv = document.createElement('span'); hv.className = 'hv';
    hgt.addEventListener('input', () => { hv.textContent = `${hgt.value} km`; });
    const hRow = row('高度 / Height', hgt, 'z', 'place'); hRow.append(hv); hv.textContent = '1.5 km';
    const clear = document.createElement('button'); clear.textContent = '清除持續風 / Clear lasting winds'; clear.className = 'windOnly';
    clear.onclick = (): void => { this.send({ type: 'clearForcing' }); };
    // strength: temperature change (or heat content) of bubbles, factor of the vapour change, sea-temperature change
    const heat = document.createElement('div'); heat.className = 'hv bubbleOnly';
    this.heatEl = heat;
    this.opts.append(
      hRow,
      row('半徑 / Radius (km)', num(10, 1), 'r'),
      row('深度 / Depth (km)', num(3, 0.5), 'depth', 'vol'),
      row('指定 / Set by', sel([['amp', '溫度幅度 / amplitude'], ['heat', '熱含量 / heat content']], 'amp'), 'by', 'bubbleOnly'),
      row('溫度幅度 / Amplitude (K)', num(3, 0.5), 'amp', 'bubbleOnly byAmp'),
      row('熱含量 / Heat (×10¹⁵ J)', num(10, 1), 'heat', 'bubbleOnly byHeat'),
      heat,
      row('水氣倍數 / Vapour ×', num(1.3, 0.05), 'fac', 'moistOnly'),
      row('海溫變化 / Sea change (°C)', num(2, 0.5), 'dsst', 'sstOnly'),
      row('形式 / Form', sel([['push', '推送 / push'], ['ccw', '逆時針旋轉 / counter-clockwise'], ['cw', '順時針旋轉 / clockwise'], ['in', '輻合 / converge'], ['out', '輻散 / diverge']], 'push'), 'form', 'windOnly'),
      row('吹向 / Toward', sel(DIRS.map((d, n) => [String(n * 45), d]), '90'), 'az', 'windOnly pushOnly'),
      row('仰角 / Tilt', sel([['0', '水平 / level'], ['30', '上升 30° / up'], ['60', '上升 60°'], ['90', '垂直上升 / straight up'], ['-30', '下沉 30° / down'], ['-60', '下沉 60°'], ['-90', '垂直下沉 / straight down']], '0'), 'el', 'windOnly pushOnly'),
      row('風速 / Speed (m/s)', num(15, 1), 'speed', 'windOnly'),
      row('持續 / Lasts', sel([['0', '一次 / once'], ['10', '10 分 / min'], ['30', '30 分'], ['60', '1 小時 / h'], ['180', '3 小時'], ['-1', '直到清除 / until cleared']], '30'), 'min', 'windOnly'),
      clear,
    );
    this.el.form!.addEventListener('change', () => this.showOpts());
    this.el.by!.addEventListener('change', () => this.showOpts());
    for (const k of ['amp', 'heat', 'r', 'depth', 'z']) this.el[k]!.addEventListener('input', () => this.showHeat());
    view.onTap = (x, y): boolean => this.tap(x, y);
    view.onToolDrag = (x, y, phase): void => this.drag(x, y, phase);
    view.onHover = (x, y): void => { const p = this.ground(x, y); this.hover = p; };
    view.onLeave = (): void => { this.hover = null; };
  }

  /** New domain: sizes for the defaults (radius about a twelfth of the domain, at least 3 cells). */
  setGrid(g: Grid): void {
    this.grid = g;
    const r = Math.max(3 * g.dx, Math.min(g.Lx / 12, 150000)) / 1000;
    (this.el.r as HTMLInputElement).value = String(+r.toPrecision(2));
    (this.el.z as HTMLInputElement).max = String(Math.floor(g.top / 1000));
    this.forcings = [];
    // painting needs surface fluxes: without them the paint tools are off (and a chosen one falls back to the camera)
    this.opts.parentElement!.querySelectorAll<HTMLButtonElement>('.tbtns button').forEach((b) => {
      const paint = PAINT.includes(b.dataset.t as Tool);
      b.disabled = b.dataset.t !== 'view' && (!g.interact || (paint && !g.paint));
      if (!g.interact && b.dataset.t !== 'view') { b.title = '軸對稱模式請用實驗設定改條件 / the axisymmetric model takes its conditions from the set-up'; return; }
      if (paint) b.title = g.paint ? TOOLS.find((d) => d.t === b.dataset.t)!.tip : '這個實驗沒有地面通量：在實驗設定打開「地面熱量、水氣與摩擦」/ no surface fluxes in this run: turn them on in the set-up';
    });
    if (!g.interact || (!g.paint && PAINT.includes(this.tool))) this.pick('view');
  }
  setForcings(list: ForcingInfo[]): void { this.forcings = list; }

  private heatEl: HTMLElement | null = null;
  /**
   * Bubble amplitude (K, signed) and heat content (J) from the panel. The bubble adds amp cos^2(pi r / 2) inside an
   * ellipsoid of horizontal radius R and vertical radius H (half the depth); its heat content is
   * rho cp amp R^2 H 4 pi (1/6 - 1/pi^2) with the air density at its centre height (standard atmosphere).
   */
  private bubble(): { amp: number; heat: number } {
    const R = 1000 * this.val('r', 10), H = 500 * this.val('depth', 3), z = 1000 * this.val('z', 1.5), rho = 1.2 * Math.exp(-z / 8500);
    const k = rho * 1004.5 * R * R * H * 4 * Math.PI * (1 / 6 - 1 / (Math.PI * Math.PI)), sign = this.tool === 'cold' ? -1 : 1;
    const byHeat = (this.el.by as HTMLSelectElement).value === 'heat';
    const amp = byHeat ? sign * Math.abs(this.val('heat', 10)) * 1e15 / k : sign * Math.abs(this.val('amp', 3));
    return { amp, heat: amp * k };
  }
  private showHeat(): void {
    if (!this.heatEl || (this.tool !== 'warm' && this.tool !== 'cold')) return;
    const b = this.bubble(), e = Math.abs(b.heat), ex = Math.floor(Math.log10(Math.max(e, 1)));
    this.heatEl.textContent = `幅度 ${b.amp >= 0 ? '+' : ''}${b.amp.toFixed(2)} K · 熱含量 ${b.heat < 0 ? '−' : ''}${(e / 10 ** ex).toFixed(2)}×10^${ex} J（約 ${(e / 4.184e15).toFixed(e < 4.184e15 ? 2 : 0)} 百萬噸 TNT / Mt TNT）`;
  }

  private pick(t: Tool): void {
    this.tool = t;
    // defaults of the strength per tool
    if (t === 'warm' || t === 'cold') { (this.el.amp as HTMLInputElement).value = t === 'warm' ? '3' : '6'; }
    if (t === 'moist' || t === 'dry') { (this.el.fac as HTMLInputElement).value = t === 'moist' ? '1.3' : '0.7'; }
    if (t === 'warmer' || t === 'cooler') { (this.el.dsst as HTMLInputElement).value = '2'; }
    this.view.toolActive = t !== 'view';
    this.opts.parentElement!.querySelectorAll<HTMLButtonElement>('.tbtns button').forEach((b) => { b.className = b.dataset.t === t ? 'on' : ''; });
    // sensible heights per tool
    const z = this.el.z as HTMLInputElement, zs: Partial<Record<Tool, number>> = { warm: 1.5, cold: 0, moist: 3, dry: 3, wind: 1 };
    if (zs[t] !== undefined) { z.value = String(zs[t]); z.dispatchEvent(new Event('input')); }
    this.showOpts();
  }
  private showOpts(): void {
    const t = this.tool, paint = PAINT.includes(t), wind = t === 'wind', push = (this.el.form as HTMLSelectElement).value === 'push';
    this.opts.hidden = t === 'view';
    this.opts.querySelectorAll<HTMLElement>('.place').forEach((e) => { e.hidden = paint; });
    this.opts.querySelectorAll<HTMLElement>('.vol').forEach((e) => { e.hidden = !(wind || t === 'moist' || t === 'dry'); });
    this.opts.querySelectorAll<HTMLElement>('.windOnly').forEach((e) => { e.hidden = !wind; });
    this.opts.querySelectorAll<HTMLElement>('.pushOnly').forEach((e) => { e.hidden = !wind || !push; });
    const bubble = t === 'warm' || t === 'cold', byHeat = (this.el.by as HTMLSelectElement).value === 'heat';
    this.opts.querySelectorAll<HTMLElement>('.bubbleOnly').forEach((e) => { e.hidden = !bubble; });
    this.opts.querySelectorAll<HTMLElement>('.byAmp').forEach((e) => { e.hidden = !bubble || byHeat; });
    this.opts.querySelectorAll<HTMLElement>('.byHeat').forEach((e) => { e.hidden = !bubble || !byHeat; });
    this.opts.querySelectorAll<HTMLElement>('.moistOnly').forEach((e) => { e.hidden = t !== 'moist' && t !== 'dry'; });
    this.opts.querySelectorAll<HTMLElement>('.sstOnly').forEach((e) => { e.hidden = t !== 'warmer' && t !== 'cooler'; });
    // bubbles use the depth too (their vertical size)
    if (bubble) this.opts.querySelectorAll<HTMLElement>('.vol').forEach((e) => { e.hidden = false; });
    this.showHeat();
  }

  private val(k: string, d: number): number { const v = Number(this.el[k]!.value); return Number.isFinite(v) ? v : d; }
  /** Ground point (domain m) under a screen point, or null (off the domain / sky). */
  private ground(cx: number, cy: number): { x: number; y: number } | null {
    const g = this.grid, p = this.view.pickPlane(cx, cy, 0); if (!g || !p) return null;
    const [bx, by] = this.view.boxSize, x = p[0] / bx * g.Lx, y = p[1] / by * g.Ly;
    return x >= 0 && y >= 0 && x <= g.Lx && y <= g.Ly ? { x, y } : null;
  }

  private tap(cx: number, cy: number): boolean {
    const t = this.tool; if (t === 'view' || PAINT.includes(t)) return false;
    const p = this.ground(cx, cy); if (!p) return false;
    const R = 1000 * this.val('r', 10), z = 1000 * this.val('z', 1.5), depth = 1000 * this.val('depth', 3);
    if (t === 'warm' || t === 'cold') this.send({ type: 'perturb', kind: t, x: p.x, y: p.y, z, radius: R, depth, amp: this.bubble().amp });
    else if (t === 'moist' || t === 'dry') this.send({ type: 'moisture', x: p.x, y: p.y, z, radius: R, depth, factor: Math.max(0.3, Math.min(2, this.val('fac', t === 'moist' ? 1.3 : 0.7))) });
    else {
      const f = (this.el.form as HTMLSelectElement).value, form = f === 'push' ? 'push' : f === 'in' || f === 'out' ? 'converge' : 'rotate';
      this.send({ type: 'wind', x: p.x, y: p.y, z, radius: R, depth, speed: this.val('speed', 15), az: this.val('az', 90), el: this.val('el', 0), form, sign: f === 'cw' || f === 'out' ? -1 : 1, minutes: this.val('min', 30) });
    }
    this.flash = { ...p, t: performance.now() };
    return true;
  }
  private drag(cx: number, cy: number, phase: 'start' | 'move' | 'end'): void {
    const t = this.tool; if (!PAINT.includes(t)) return;
    if (phase === 'end') { this.lastPaint = null; return; }
    const p = this.ground(cx, cy); if (!p) return;
    const R = 1000 * this.val('r', 10);
    // a stroke every half brush radius along the drag
    if (this.lastPaint && Math.hypot(p.x - this.lastPaint.x, p.y - this.lastPaint.y) < 0.5 * R) return;
    this.lastPaint = p; this.hover = p;
    this.send({ type: 'paint', kind: t as 'land' | 'sea' | 'warmer' | 'cooler', x: p.x, y: p.y, radius: R, amount: Math.abs(this.val('dsst', 2)) });
  }

  /** Redraw the overlay (every animation frame: the camera moves). */
  drawOverlay(): void {
    const svg = this.svg, g = this.grid, cv = this.view;
    const parts: string[] = [];
    if (g) {
      const [bx, by, bz] = cv.boxSize;
      const P = (x: number, y: number, z: number): { x: number; y: number } | null => cv.project([x / g.Lx * bx, y / g.Ly * by, z / g.top * bz]);
      const ring = (x: number, y: number, z: number, R: number, stroke: string, dash = ''): void => {
        const pts: string[] = [];
        for (let a = 0; a <= 32; a++) { const p = P(x + R * Math.cos(a / 32 * 2 * Math.PI), y + R * Math.sin(a / 32 * 2 * Math.PI), z); if (!p) return; pts.push(`${p.x.toFixed(1)},${p.y.toFixed(1)}`); }
        parts.push(`<polyline points="${pts.join(' ')}" fill="none" stroke="${stroke}" stroke-width="1.5"${dash ? ` stroke-dasharray="${dash}"` : ''}/>`);
      };
      const line = (a: { x: number; y: number } | null, b: { x: number; y: number } | null, stroke: string, w = 1.5): void => { if (a && b) parts.push(`<line x1="${a.x.toFixed(1)}" y1="${a.y.toFixed(1)}" x2="${b.x.toFixed(1)}" y2="${b.y.toFixed(1)}" stroke="${stroke}" stroke-width="${w}"/>`); };
      const arrow = (x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, stroke: string): void => {
        const a = P(x0, y0, z0), b = P(x1, y1, z1); if (!a || !b) return;
        line(a, b, stroke, 2);
        const ang = Math.atan2(b.y - a.y, b.x - a.x), h = 9;
        parts.push(`<polygon points="${b.x},${b.y} ${b.x - h * Math.cos(ang - 0.45)},${b.y - h * Math.sin(ang - 0.45)} ${b.x - h * Math.cos(ang + 0.45)},${b.y - h * Math.sin(ang + 0.45)}" fill="${stroke}"/>`);
      };
      const windShape = (f: { x: number; y: number; z: number; radius: number; speed: number; az: number; el: number; form: string; sign: number }, stroke: string): void => {
        ring(f.x, f.y, f.z, f.radius, stroke);
        if (f.form === 'push') {
          const az = f.az * Math.PI / 180, el = f.el * Math.PI / 180, L = 0.8 * f.radius;
          // the tilt drawn as the angle seen in the (vertically exaggerated) box
          arrow(f.x, f.y, f.z, f.x + L * Math.cos(el) * Math.sin(az), f.y + L * Math.cos(el) * Math.cos(az), f.z + L * Math.sin(el) * (bx / g.Lx) / (bz / g.top), stroke);
        } else for (let n = 0; n < 4; n++) {
          const a = n * Math.PI / 2, r0 = 0.5 * f.radius, px = f.x + r0 * Math.cos(a), py = f.y + r0 * Math.sin(a);
          if (f.form === 'rotate') { const s = f.sign * 0.35 * f.radius; arrow(px, py, f.z, px - s * Math.sin(a), py + s * Math.cos(a), f.z, stroke); }
          else { const s = -f.sign * 0.35 * f.radius; arrow(px, py, f.z, px + s * Math.cos(a), py + s * Math.sin(a), f.z, stroke); }
        }
      };
      // lasting winds
      for (const f of this.forcings) windShape(f, '#f2b134');
      // preview of the active tool at the pointer (mouse) or the last tap (brief)
      const at = this.hover ?? (this.flash && performance.now() - this.flash.t < 1500 ? this.flash : null);
      if (at && this.tool !== 'view') {
        const R = 1000 * this.val('r', 10), paint = PAINT.includes(this.tool), z = paint ? 0 : 1000 * this.val('z', 1.5);
        ring(at.x, at.y, 0, R, '#dbe5ef', '4 4');
        if (!paint) {
          line(P(at.x, at.y, 0), P(at.x, at.y, z), '#dbe5ef', 1);
          if (this.tool === 'wind') {
            const f = (this.el.form as HTMLSelectElement).value;
            windShape({ x: at.x, y: at.y, z, radius: R, speed: 0, az: this.val('az', 90), el: this.val('el', 0), form: f === 'push' ? 'push' : f === 'in' || f === 'out' ? 'converge' : 'rotate', sign: f === 'cw' || f === 'out' ? -1 : 1 }, '#8fc1ff');
          } else ring(at.x, at.y, z, R, this.tool === 'warm' ? '#e66767' : this.tool === 'cold' ? '#3987e5' : '#199e70');
        }
      }
    }
    const html = parts.join('');
    if (svg.dataset.last !== html) { svg.innerHTML = html; svg.dataset.last = html; }
  }
}
