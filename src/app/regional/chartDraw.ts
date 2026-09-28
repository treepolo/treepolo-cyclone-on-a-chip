// Drawing helpers for the regional charts: colour scales (dark surface), raster fields, colour bars,
// contours, axes, wind arrows and barbs. All coordinates are CSS pixels.

export interface Rect { x: number; y: number; w: number; h: number }
export type Rgba = [number, number, number, number];

// ---------------------------------------------------------------- tokens (dark page)
export const INK = { primary: '#dbe5ef', secondary: '#8da0b3', grid: '#26303b', axis: '#3a4756', surface: '#0b1017', halo: 'rgba(5,7,10,0.75)' };
/** categorical slots (dark steps of the reference palette), fixed order */
export const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
export const FONT = '12px system-ui, "Noto Sans TC", "Microsoft JhengHei", sans-serif';
export const FONT_SMALL = '11px system-ui, "Noto Sans TC", "Microsoft JhengHei", sans-serif';

const hex = (h: string): [number, number, number] => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
function ramp(stops: string[]): (t: number) => Rgba {
  const c = stops.map(hex);
  return (t: number): Rgba => {
    const x = Math.max(0, Math.min(1, t)) * (c.length - 1), i = Math.min(c.length - 2, Math.floor(x)), f = x - i, a = c[i]!, b = c[i + 1]!;
    return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), 255];
  };
}
/** one hue (blue), near zero receding toward the dark surface, magnitude = lightness */
export const seqRamp = ramp(['#10243d', '#0d366b', '#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4', '#cde2fb', '#eef5fe']);
/** blue <-> red with a neutral grey midpoint; t in [-1, 1] */
const divRampU = ramp(['#dcebfd', '#6da7ec', '#2a78d6', '#184f95', '#383835', '#8f2a2a', '#d03b3b', '#e66767', '#f9d2d2']);
export const divRamp = (t: number): Rgba => divRampU(0.5 + 0.5 * t);
/** NWS radar reflectivity colours, 5 dBZ classes from 5 to 75 dBZ (domain convention) */
export const RADAR = ['#04e9e7', '#019ff4', '#0300f4', '#02fd02', '#01c501', '#008e00', '#fdf802', '#e5bc00', '#fd9500', '#fd0000', '#d40000', '#bc0000', '#f800fd', '#9854c6', '#fdfdfd'].map(hex);
/** enhanced infrared: grey for warm surfaces and low cloud, colour enhancement for cold tops (deg C) */
const IR_STOPS: [number, string][] = [[40, '#0b0b0b'], [20, '#2b2b2b'], [0, '#6e6e6e'], [-20, '#a9a9a9'], [-35, '#dedede'], [-40, '#8fd0f8'], [-50, '#2a78d6'], [-58, '#1baf7a'], [-64, '#eda100'], [-70, '#e34948'], [-76, '#fafafa'], [-90, '#b04fd6']];
function irColor(T: number): Rgba {
  const s = IR_STOPS;
  if (T >= s[0]![0]) return [...hex(s[0]![1]), 255];
  for (let i = 0; i < s.length - 1; i++) {
    const [t0, c0] = s[i]!, [t1, c1] = s[i + 1]!;
    if (T <= t0 && T >= t1) { const f = (t0 - T) / (t0 - t1), a = hex(c0), b = hex(c1); return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), 255]; }
  }
  return [...hex(s[s.length - 1]![1]), 255];
}

export type ScaleKind = 'seq' | 'div' | 'radar' | 'ir';
export interface Scale { kind: ScaleKind; lo: number; hi: number; gamma: number; clear: number | null; reverse: boolean }

/** Colour of a value (alpha 0 = show the underlay). */
export function colorOf(s: Scale, v: number): Rgba {
  if (!Number.isFinite(v)) return [0, 0, 0, 0];
  if (s.kind === 'radar') { if (v < 5) return [0, 0, 0, 0]; const c = RADAR[Math.min(RADAR.length - 1, Math.floor((v - 5) / 5))]!; return [c[0], c[1], c[2], 255]; }
  if (s.kind === 'ir') return irColor(v);
  if (s.kind === 'div') { const a = Math.max(Math.abs(s.lo), Math.abs(s.hi)) || 1; return divRamp(Math.sign(v) * Math.pow(Math.min(1, Math.abs(v) / a), s.gamma)); }
  if (s.clear !== null && (s.reverse ? v > -s.clear : v < s.clear)) return [0, 0, 0, 0];
  let t = (v - s.lo) / ((s.hi - s.lo) || 1);
  if (s.reverse) t = 1 - t;
  return seqRamp(Math.pow(Math.max(0, Math.min(1, t)), s.gamma));
}

/** Smallest "nice" number (1, 2, 2.5, 5 x 10^n) not below x. */
export function niceCeil(x: number): number {
  if (!(x > 0)) return 1;
  const e = Math.pow(10, Math.floor(Math.log10(x)));
  for (const f of [1, 2, 2.5, 5, 10]) if (f * e >= x * (1 - 1e-9)) return f * e;
  return 10 * e;
}
/** Tick spacing used by ticks(). */
export const tickStep = (lo: number, hi: number, n: number): number => niceCeil((hi - lo) / n);
/** Decimals needed to print multiples of a tick step. */
export function stepDigits(s: number): number {
  if (!(s > 0) || s >= 1) return 0;
  const m = s / Math.pow(10, Math.floor(Math.log10(s)));
  return Math.min(6, Math.ceil(-Math.log10(s) - 1e-9) + (Math.abs(m - 2.5) < 1e-6 ? 1 : 0));
}
/** About n nice tick values covering [lo, hi]. */
export function ticks(lo: number, hi: number, n = 5): number[] {
  if (!(hi > lo)) return [lo];
  const step = niceCeil((hi - lo) / n), out: number[] = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9 * step; v += step) out.push(Math.abs(v) < 1e-12 * step ? 0 : v);
  return out;
}
export const fmt = (v: number, digits: number): string => {
  if (!Number.isFinite(v)) return '—';
  if (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0 && digits > 3)) return v.toExponential(1);
  return v.toFixed(digits);
};

// ---------------------------------------------------------------- raster fields

/** Draw a gridded field (ni x nj, row j = 0 at the bottom) into rect, one cell per rectangle
 *  (no smoothing, the grid cells stay visible). underlay: colour behind transparent values. */
export function drawField(ctx: CanvasRenderingContext2D, r: Rect, ni: number, nj: number, value: (i: number, j: number) => number, s: Scale, underlay: (i: number, j: number) => [number, number, number]): void {
  const img = new ImageData(ni, nj), d = img.data;
  for (let j = 0; j < nj; j++) for (let i = 0; i < ni; i++) {
    const c = colorOf(s, value(i, j)), o = 4 * ((nj - 1 - j) * ni + i);
    if (c[3] === 0) { const u = underlay(i, j); d[o] = u[0]; d[o + 1] = u[1]; d[o + 2] = u[2]; }
    else { d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; }
    d[o + 3] = 255;
  }
  const off = document.createElement('canvas'); off.width = ni; off.height = nj;
  off.getContext('2d')!.putImageData(img, 0, 0);
  ctx.save(); ctx.imageSmoothingEnabled = false; ctx.drawImage(off, r.x, r.y, r.w, r.h); ctx.restore();
}

/** Vertical colour bar with tick labels. */
export function drawColorbar(ctx: CanvasRenderingContext2D, r: Rect, s: Scale, unit: string, digits: number): void {
  const n = Math.max(2, Math.round(r.h));
  if (s.kind === 'radar') {
    const nb = RADAR.length, bh = r.h / nb;
    for (let b = 0; b < nb; b++) { const c = RADAR[b]!; ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.fillRect(r.x, r.y + r.h - (b + 1) * bh, r.w, bh + 0.5); }
  } else {
    for (let p = 0; p < n; p++) {
      const t = p / (n - 1), v = s.kind === 'div' ? -Math.max(Math.abs(s.lo), Math.abs(s.hi)) + 2 * t * Math.max(Math.abs(s.lo), Math.abs(s.hi)) : s.lo + t * (s.hi - s.lo);
      const c = colorOf({ ...s, clear: null }, v);
      ctx.fillStyle = `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
      ctx.fillRect(r.x, r.y + r.h - (p + 1) * r.h / n, r.w, r.h / n + 0.5);
    }
  }
  ctx.strokeStyle = INK.axis; ctx.lineWidth = 1; ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
  ctx.fillStyle = INK.secondary; ctx.font = FONT_SMALL; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
  let lo = s.lo, hi = s.hi;
  if (s.kind === 'div') { const a = Math.max(Math.abs(lo), Math.abs(hi)); lo = -a; hi = a; }
  if (s.kind === 'radar') { lo = 5; hi = 80; }
  const tv = s.kind === 'radar' ? [10, 20, 30, 40, 50, 60, 70] : s.kind === 'ir' ? [20, 0, -20, -40, -60, -80] : ticks(lo, hi, 5);
  for (const v of tv) {
    const t = (v - lo) / ((hi - lo) || 1);
    if (t < -1e-6 || t > 1 + 1e-6) continue;
    const y = r.y + r.h - t * r.h;
    ctx.fillRect(r.x + r.w, y, 3, 1);
    ctx.fillText(fmt(v, Math.abs(hi - lo) < 0.1 ? Math.max(digits, 3) : Math.abs(hi - lo) < 5 ? Math.max(1, Math.min(digits, 2)) : 0), r.x + r.w + 5, y);
  }
  ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
  ctx.fillText(unit, r.x + r.w / 2 + 8, r.y - 4);
}

/** Axes frame with tick labels. xTicks / yTicks: value -> pixel. */
export function drawAxes(ctx: CanvasRenderingContext2D, r: Rect, x: { lo: number; hi: number; label: string }, y: { lo: number; hi: number; label: string }, grid = false): void {
  ctx.strokeStyle = INK.axis; ctx.lineWidth = 1;
  ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
  ctx.fillStyle = INK.secondary; ctx.font = FONT_SMALL;
  const X = (v: number): number => r.x + (v - x.lo) / ((x.hi - x.lo) || 1) * r.w, Y = (v: number): number => r.y + r.h - (v - y.lo) / ((y.hi - y.lo) || 1) * r.h;
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  const nxT = Math.max(2, Math.floor(r.w / 70)), nyT = Math.max(3, Math.floor(r.h / 36));
  const xt = ticks(x.lo, x.hi, nxT), yt = ticks(y.lo, y.hi, nyT);
  const xs = tickStep(x.lo, x.hi, nxT), ys = tickStep(y.lo, y.hi, nyT);
  const dg = (t: number[]): number => stepDigits(t === xt ? xs : ys);
  for (const v of xt) { const px = X(v); if (grid) { ctx.fillStyle = INK.grid; ctx.fillRect(px, r.y, 1, r.h); ctx.fillStyle = INK.secondary; } ctx.fillRect(px, r.y + r.h, 1, 4); ctx.fillText(v.toFixed(dg(xt)), px, r.y + r.h + 5); }
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const v of yt) { const py = Y(v); if (grid) { ctx.fillStyle = INK.grid; ctx.fillRect(r.x, py, r.w, 1); ctx.fillStyle = INK.secondary; } ctx.fillRect(r.x - 4, py, 4, 1); ctx.fillText(v.toFixed(dg(yt)), r.x - 6, py); }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  ctx.fillText(x.label, r.x + r.w / 2, r.y + r.h + 19);
  ctx.save(); ctx.translate(r.x - 40, r.y + r.h / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle'; ctx.fillText(y.label, 0, 0); ctx.restore();
}

/** Contour lines at `levels` of a field on cell centres (marching squares); X(i), Y(j) map indices to pixels. */
export function drawContours(ctx: CanvasRenderingContext2D, ni: number, nj: number, value: (i: number, j: number) => number, X: (i: number) => number, Y: (j: number) => number, levels: number[], style: (level: number) => { color: string; width: number; dash?: number[] }): void {
  for (const L of levels) {
    const st = style(L);
    ctx.strokeStyle = st.color; ctx.lineWidth = st.width; ctx.setLineDash(st.dash ?? []);
    ctx.beginPath();
    for (let j = 0; j < nj - 1; j++) for (let i = 0; i < ni - 1; i++) {
      const P = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]] as const, pts: [number, number][] = [];
      for (let e = 0; e < 4; e++) {
        const a = P[e]!, b = P[(e + 1) % 4]!, va = value(a[0], a[1]), vb = value(b[0], b[1]);
        if ((va >= L) !== (vb >= L)) { const t = (L - va) / (vb - va); pts.push([X(a[0]) + t * (X(b[0]) - X(a[0])), Y(a[1]) + t * (Y(b[1]) - Y(a[1]))]); }
      }
      if (pts.length >= 2) { ctx.moveTo(pts[0]![0], pts[0]![1]); ctx.lineTo(pts[1]![0], pts[1]![1]); }
      if (pts.length === 4) { ctx.moveTo(pts[2]![0], pts[2]![1]); ctx.lineTo(pts[3]![0], pts[3]![1]); }
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

/** Arrow from (x, y) by (ax, ay) pixels (screen y down), light with a dark halo. */
export function drawArrow(ctx: CanvasRenderingContext2D, x: number, y: number, ax: number, ay: number): void {
  const l = Math.hypot(ax, ay);
  if (l < 1.5) { ctx.fillStyle = INK.primary; ctx.fillRect(x - 1, y - 1, 2, 2); return; }
  const hx = ax / l, hy = ay / l, hs = Math.min(5, 0.4 * l);
  const path = (): void => {
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + ax, y + ay);
    ctx.moveTo(x + ax - hs * (hx - 0.5 * hy), y + ay - hs * (hy + 0.5 * hx)); ctx.lineTo(x + ax, y + ay);
    ctx.lineTo(x + ax - hs * (hx + 0.5 * hy), y + ay - hs * (hy - 0.5 * hx));
  };
  ctx.lineCap = 'round';
  path(); ctx.strokeStyle = INK.halo; ctx.lineWidth = 3; ctx.stroke();
  path(); ctx.strokeStyle = '#eef3f8'; ctx.lineWidth = 1.2; ctx.stroke();
}

/** Wind barb at (x, y) for a wind (u, v) in m/s: half barb 2.5, full barb 5, pennant 25 m/s. */
export function drawBarb(ctx: CanvasRenderingContext2D, x: number, y: number, u: number, v: number, len = 26): void {
  const sp = Math.hypot(u, v);
  ctx.strokeStyle = INK.primary; ctx.fillStyle = INK.primary; ctx.lineWidth = 1.2;
  if (sp < 1.25) { ctx.beginPath(); ctx.arc(x, y, 3, 0, 2 * Math.PI); ctx.stroke(); return; }
  // the staff points toward where the wind comes from (screen y down)
  const dx = -u / sp, dy = v / sp, px = -dy, py = dx;
  const tx = x + dx * len, ty = y + dy * len;
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(tx, ty); ctx.stroke();
  let rem = Math.round(sp / 2.5) * 2.5, pos = 0;
  const step = 4;
  while (rem >= 25 - 1e-9) {
    const bx = tx - dx * pos, by = ty - dy * pos;
    ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx + px * 10 - dx * 3, by + py * 10 - dy * 3); ctx.lineTo(bx - dx * 6, by - dy * 6); ctx.closePath(); ctx.fill();
    pos += 7; rem -= 25;
  }
  while (rem >= 5 - 1e-9) { const bx = tx - dx * pos, by = ty - dy * pos; ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx + px * 10 + dx * 3, by + py * 10 + dy * 3); ctx.stroke(); pos += step; rem -= 5; }
  if (rem >= 2.5 - 1e-9) { if (pos === 0) pos = step; const bx = tx - dx * pos, by = ty - dy * pos; ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx + px * 5 + dx * 1.5, by + py * 5 + dy * 1.5); ctx.stroke(); }
}

/** Text with a dark halo (readable over any fill). */
export function haloText(ctx: CanvasRenderingContext2D, s: string, x: number, y: number, color: string = INK.primary): void {
  ctx.lineWidth = 3; ctx.strokeStyle = INK.halo; ctx.lineJoin = 'round'; ctx.strokeText(s, x, y);
  ctx.fillStyle = color; ctx.fillText(s, x, y);
}

/** Tooltip box with lines of text near (x, y), kept inside the canvas. */
export function tooltip(ctx: CanvasRenderingContext2D, lines: string[], x: number, y: number, W: number, H: number): void {
  ctx.font = FONT_SMALL;
  const w = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 14, h = lines.length * 15 + 8;
  let bx = x + 14, by = y + 14;
  if (bx + w > W - 4) bx = x - 14 - w;
  if (by + h > H - 4) by = y - 14 - h;
  bx = Math.max(4, bx); by = Math.max(4, by);
  ctx.fillStyle = 'rgba(12,17,24,0.94)'; ctx.strokeStyle = INK.axis; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.roundRect(bx, by, w, h, 4); ctx.fill(); ctx.stroke();
  ctx.fillStyle = INK.primary; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  lines.forEach((l, i) => { ctx.fillStyle = i === 0 ? INK.primary : INK.secondary; ctx.fillText(l, bx + 7, by + 5 + 15 * i); });
}
