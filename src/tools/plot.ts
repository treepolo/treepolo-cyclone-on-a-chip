// Minimal dependency-free SVG contour-style (filled cell) plots for latitude–sigma sections.

export interface SectionPlot {
  title: string;
  lat: number[];          // degrees, north -> south
  sigma: number[];        // vertical coordinate per row (top -> bottom)
  values: number[];       // [row][lat]
  units: string;
  diverging: boolean;
  contourStep: number;
}

function color(t: number, diverging: boolean): string {
  // t in [-1,1] for diverging, [0,1] otherwise
  if (diverging) {
    const x = Math.max(-1, Math.min(1, t));
    const r = x > 0 ? 255 : Math.round(255 * (1 + x));
    const b = x < 0 ? 255 : Math.round(255 * (1 - x));
    const g = Math.round(255 * (1 - Math.abs(x)));
    return `rgb(${r},${g},${b})`;
  }
  const x = Math.max(0, Math.min(1, t));
  const stops = [[48, 18, 59], [70, 107, 227], [41, 187, 236], [49, 242, 153], [163, 253, 61], [237, 208, 58], [251, 128, 34], [208, 47, 5]];
  const f = x * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(f)), w = f - i;
  const a = stops[i]!, b = stops[i + 1]!;
  return `rgb(${Math.round(a[0]! + w * (b[0]! - a[0]!))},${Math.round(a[1]! + w * (b[1]! - a[1]!))},${Math.round(a[2]! + w * (b[2]! - a[2]!))})`;
}

/** Marching-squares contour segments of a rectilinear field. */
function contours(xs: number[], ys: number[], v: (r: number, c: number) => number, level: number): string {
  const segs: string[] = [];
  for (let r = 0; r < ys.length - 1; r++) for (let c = 0; c < xs.length - 1; c++) {
    const pts: [number, number][] = [];
    const corners: [number, number, number][] = [
      [xs[c]!, ys[r]!, v(r, c)], [xs[c + 1]!, ys[r]!, v(r, c + 1)],
      [xs[c + 1]!, ys[r + 1]!, v(r + 1, c + 1)], [xs[c]!, ys[r + 1]!, v(r + 1, c)],
    ];
    for (let e = 0; e < 4; e++) {
      const p = corners[e]!, q = corners[(e + 1) % 4]!;
      if ((p[2] - level) * (q[2] - level) < 0) {
        const t = (level - p[2]) / (q[2] - p[2]);
        pts.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
      }
    }
    if (pts.length >= 2) segs.push(`M${pts[0]![0].toFixed(1)},${pts[0]![1].toFixed(1)}L${pts[1]![0].toFixed(1)},${pts[1]![1].toFixed(1)}`);
    if (pts.length === 4) segs.push(`M${pts[2]![0].toFixed(1)},${pts[2]![1].toFixed(1)}L${pts[3]![0].toFixed(1)},${pts[3]![1].toFixed(1)}`);
  }
  return segs.join('');
}

export function sectionSvg(p: SectionPlot, width = 520, height = 300): string {
  const ml = 48, mr = 16, mt = 28, mb = 36;
  const W = width - ml - mr, H = height - mt - mb;
  const nl = p.lat.length, nr = p.sigma.length;
  const xOf = (lat: number): number => ml + (90 - lat) / 180 * W;       // north on the left
  const yOf = (s: number): number => mt + s * H;                          // sigma 0 at top
  let vmax = 0, vmin = Infinity, vmx = -Infinity;
  for (const x of p.values) { vmax = Math.max(vmax, Math.abs(x)); vmin = Math.min(vmin, x); vmx = Math.max(vmx, x); }
  const out: string[] = [];
  out.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" font-family="sans-serif" font-size="11">`);
  out.push(`<rect width="100%" height="100%" fill="#fff"/>`);
  out.push(`<text x="${ml}" y="16" font-size="13" font-weight="bold">${p.title} (${p.units})</text>`);
  // cell edges
  const latEdge = (j: number): number => j === 0 ? 90 : j === nl ? -90 : 0.5 * (p.lat[j - 1]! + p.lat[j]!);
  const sEdge = (k: number): number => k === 0 ? 0 : k === nr ? 1 : 0.5 * (p.sigma[k - 1]! + p.sigma[k]!);
  for (let k = 0; k < nr; k++) for (let j = 0; j < nl; j++) {
    const val = p.values[k * nl + j]!;
    const t = p.diverging ? val / (vmax || 1) : (val - vmin) / ((vmx - vmin) || 1);
    const x0 = xOf(latEdge(j)), x1 = xOf(latEdge(j + 1)), y0 = yOf(sEdge(k)), y1 = yOf(sEdge(k + 1));
    out.push(`<rect x="${x0.toFixed(1)}" y="${y0.toFixed(1)}" width="${(x1 - x0 + 0.4).toFixed(1)}" height="${(y1 - y0 + 0.4).toFixed(1)}" fill="${color(t, p.diverging)}"/>`);
  }
  const xs = p.lat.map(xOf), ys = p.sigma.map(yOf);
  const val = (r: number, c: number): number => p.values[r * nl + c]!;
  const lo = Math.ceil(vmin / p.contourStep), hi = Math.floor(vmx / p.contourStep);
  for (let i = lo; i <= hi; i++) {
    const level = i * p.contourStep;
    const d = contours(xs, ys, val, level === 0 ? 1e-12 : level);
    if (d) out.push(`<path d="${d}" stroke="${level < 0 ? '#333' : '#000'}" stroke-width="${level === 0 ? 1.6 : 0.8}" ${level < 0 ? 'stroke-dasharray="3,2"' : ''} fill="none"/>`);
  }
  out.push(`<rect x="${ml}" y="${mt}" width="${W}" height="${H}" fill="none" stroke="#000"/>`);
  for (const lat of [90, 60, 30, 0, -30, -60, -90]) {
    out.push(`<text x="${xOf(lat)}" y="${mt + H + 14}" text-anchor="middle">${lat === 0 ? 'EQ' : lat > 0 ? lat + 'N' : -lat + 'S'}</text>`);
  }
  for (const s of [0, 0.2, 0.4, 0.6, 0.8, 1]) {
    out.push(`<text x="${ml - 4}" y="${yOf(s) + 4}" text-anchor="end">${s.toFixed(1)}</text>`);
  }
  out.push(`<text x="12" y="${mt + H / 2}" transform="rotate(-90 12 ${mt + H / 2})" text-anchor="middle">σ</text>`);
  out.push(`<text x="${ml + W}" y="${mt + H + 30}" text-anchor="end" fill="#555">range ${vmin.toPrecision(3)} … ${vmx.toPrecision(3)}, contour ${p.contourStep}</text>`);
  out.push('</svg>');
  return out.join('\n');
}

export interface MapPlot {
  title: string;
  lat: number[];          // degrees north -> south
  lon: number[];          // degrees 0..360
  values: number[];       // [lat][lon]
  units: string;
  contourStep: number;
  latRange: [number, number]; // [south, north]
  diverging?: boolean;
  /** optional 0/1 field whose 0.5 contour is drawn as coastline */
  outline?: number[];
  /** optional fixed colour range */
  range?: [number, number];
  /** optional wind vectors (u, v) drawn every `stride` points */
  vectors?: { u: number[]; v: number[]; stride: number; scale: number };
  /** optional second field drawn as contours over the shading (e.g. sea-level pressure over T850) */
  overlay?: { values: number[]; step: number };
}

/** Equirectangular map of a Gaussian-grid field, restricted to a latitude band. */
export function mapSvg(p: MapPlot, width = 900, height = 300): string {
  const ml = 40, mr = 12, mt = 26, mb = 28;
  const W = width - ml - mr, H = height - mt - mb;
  const [s0, n0] = p.latRange, nl = p.lat.length, nx = p.lon.length;
  const xOf = (lo: number): number => ml + lo / 360 * W;
  const yOf = (la: number): number => mt + (n0 - la) / (n0 - s0) * H;
  const rows: number[] = [];
  for (let j = 0; j < nl; j++) if (p.lat[j]! >= s0 - 3 && p.lat[j]! <= n0 + 3) rows.push(j);
  let vmin = Infinity, vmax = -Infinity, amax = 0;
  for (const j of rows) for (let i = 0; i < nx; i++) { const v = p.values[j * nx + i]!; vmin = Math.min(vmin, v); vmax = Math.max(vmax, v); amax = Math.max(amax, Math.abs(v)); }
  if (p.range) { vmin = p.range[0]; vmax = p.range[1]; amax = Math.max(Math.abs(vmin), Math.abs(vmax)); }
  const out: string[] = [];
  out.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" font-family="sans-serif" font-size="11">`);
  out.push(`<rect width="100%" height="100%" fill="#fff"/><defs><clipPath id="c"><rect x="${ml}" y="${mt}" width="${W}" height="${H}"/></clipPath></defs>`);
  out.push(`<text x="${ml}" y="16" font-size="13" font-weight="bold">${p.title} (${p.units})</text><g clip-path="url(#c)">`);
  const dx = 360 / nx;
  for (let r = 0; r < rows.length; r++) {
    const j = rows[r]!;
    const top = j === 0 ? 90 : 0.5 * (p.lat[j - 1]! + p.lat[j]!), bot = j === nl - 1 ? -90 : 0.5 * (p.lat[j]! + p.lat[j + 1]!);
    for (let i = 0; i < nx; i++) {
      const v = p.values[j * nx + i]!;
      const t = p.diverging ? v / (amax || 1) : (v - vmin) / ((vmax - vmin) || 1);
      out.push(`<rect x="${xOf(p.lon[i]! - dx / 2).toFixed(1)}" y="${yOf(top).toFixed(1)}" width="${(W / nx + 0.5).toFixed(1)}" height="${(yOf(bot) - yOf(top) + 0.5).toFixed(1)}" fill="${color(t, !!p.diverging)}"/>`);
    }
  }
  const xs = [...p.lon, 360].map(xOf), ys = rows.map((j) => yOf(p.lat[j]!));
  const val = (r: number, c: number): number => p.values[rows[r]! * nx + (c % nx)]!;
  if (p.overlay) {
    const ov = p.overlay, oval = (r: number, c: number): number => ov.values[rows[r]! * nx + (c % nx)]!;
    let omin = Infinity, omax = -Infinity;
    for (const j of rows) for (let i = 0; i < nx; i++) { const v = ov.values[j * nx + i]!; omin = Math.min(omin, v); omax = Math.max(omax, v); }
    for (let i = Math.ceil(omin / ov.step); i <= Math.floor(omax / ov.step); i++) {
      const d = contours(xs, ys, oval, i * ov.step + 1e-9);
      if (d) out.push(`<path d="${d}" stroke="#000" stroke-width="0.8" fill="none"/>`);
    }
  } else {
    const lo = Math.ceil(vmin / p.contourStep), hi = Math.floor(vmax / p.contourStep);
    for (let i = lo; i <= hi; i++) {
      const d = contours(xs, ys, val, i * p.contourStep + 1e-9);
      if (d) out.push(`<path d="${d}" stroke="#000" stroke-width="0.7" fill="none"/>`);
    }
  }
  if (p.outline) {
    const ol = p.outline;
    const d = contours(xs, ys, (r: number, c: number): number => ol[rows[r]! * nx + (c % nx)]!, 0.5);
    if (d) out.push(`<path d="${d}" stroke="#222" stroke-width="1.4" fill="none"/>`);
  }
  if (p.vectors) {
    const V = p.vectors;
    for (let r = 0; r < rows.length; r += V.stride) for (let i = 0; i < nx; i += V.stride) {
      const j = rows[r]!, uu = V.u[j * nx + i]!, vv = V.v[j * nx + i]!;
      const x0 = xOf(p.lon[i]!), y0 = yOf(p.lat[j]!), x1 = x0 + uu * V.scale, y1 = y0 - vv * V.scale;
      const ang = Math.atan2(y1 - y0, x1 - x0), hl = Math.min(4, 0.35 * Math.hypot(x1 - x0, y1 - y0));
      out.push(`<path d="M${x0.toFixed(1)},${y0.toFixed(1)}L${x1.toFixed(1)},${y1.toFixed(1)}M${(x1 - hl * Math.cos(ang - 0.5)).toFixed(1)},${(y1 - hl * Math.sin(ang - 0.5)).toFixed(1)}L${x1.toFixed(1)},${y1.toFixed(1)}L${(x1 - hl * Math.cos(ang + 0.5)).toFixed(1)},${(y1 - hl * Math.sin(ang + 0.5)).toFixed(1)}" stroke="#111" stroke-width="0.8" fill="none"/>`);
    }
  }
  out.push('</g>');
  out.push(`<rect x="${ml}" y="${mt}" width="${W}" height="${H}" fill="none" stroke="#000"/>`);
  for (let lo2 = 0; lo2 <= 360; lo2 += 60) out.push(`<text x="${xOf(lo2)}" y="${mt + H + 14}" text-anchor="middle">${lo2}°E</text>`);
  for (let la = Math.ceil(s0 / 30) * 30; la <= n0; la += 30) out.push(`<text x="${ml - 4}" y="${yOf(la) + 4}" text-anchor="end">${la}°</text>`);
  out.push(`<text x="${ml + W}" y="${mt + H + 26}" text-anchor="end" fill="#555">range ${vmin.toPrecision(4)} … ${vmax.toPrecision(4)}, contour ${p.contourStep}</text>`);
  out.push('</svg>');
  return out.join('\n');
}

export interface LinePlot {
  title: string;
  x: number[];              // latitude, degrees (north -> south)
  series: { name: string; values: number[]; color: string; dashed?: boolean }[];
  units: string;
}

/** Line plot against latitude (north on the left). */
export function lineSvg(p: LinePlot, width = 520, height = 280): string {
  const ml = 48, mr = 16, mt = 28, mb = 48;
  const W = width - ml - mr, H = height - mt - mb;
  let lo = Infinity, hi = -Infinity;
  for (const s of p.series) for (const v of s.values) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (lo > 0 && lo < 0.3 * hi) lo = 0;
  const pad = (hi - lo) * 0.05 || 1;
  lo -= pad; hi += pad;
  const xOf = (lat: number): number => ml + (90 - lat) / 180 * W;
  const yOf = (v: number): number => mt + (hi - v) / (hi - lo) * H;
  const out: string[] = [];
  out.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" font-family="sans-serif" font-size="11">`);
  out.push(`<rect width="100%" height="100%" fill="#fff"/>`);
  out.push(`<text x="${ml}" y="16" font-size="13" font-weight="bold">${p.title} (${p.units})</text>`);
  out.push(`<rect x="${ml}" y="${mt}" width="${W}" height="${H}" fill="none" stroke="#000"/>`);
  const step = niceStep((hi - lo) / 5);
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    out.push(`<line x1="${ml}" x2="${ml + W}" y1="${yOf(v)}" y2="${yOf(v)}" stroke="#ddd"/>`);
    out.push(`<text x="${ml - 4}" y="${yOf(v) + 4}" text-anchor="end">${+v.toPrecision(4)}</text>`);
  }
  for (const lat of [90, 60, 30, 0, -30, -60, -90]) out.push(`<text x="${xOf(lat)}" y="${mt + H + 14}" text-anchor="middle">${lat === 0 ? 'EQ' : lat > 0 ? lat + 'N' : -lat + 'S'}</text>`);
  p.series.forEach((s, n) => {
    const d = s.values.map((v, j) => `${j ? 'L' : 'M'}${xOf(p.x[j]!).toFixed(1)},${yOf(v).toFixed(1)}`).join('');
    out.push(`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="1.8" ${s.dashed ? 'stroke-dasharray="4,3"' : ''}/>`);
    const lx = ml + n * 120;
    out.push(`<line x1="${lx}" x2="${lx + 18}" y1="${height - 12}" y2="${height - 12}" stroke="${s.color}" stroke-width="2" ${s.dashed ? 'stroke-dasharray="4,3"' : ''}/><text x="${lx + 22}" y="${height - 8}">${s.name}</text>`);
  });
  out.push('</svg>');
  return out.join('\n');
}

function niceStep(x: number): number {
  const e = Math.pow(10, Math.floor(Math.log10(x)));
  const m = x / e;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * e;
}

export interface XYPlot {
  title: string; nx: number; ny: number; dx: number; values: number[]; units: string; diverging: boolean; contourStep: number;
  outline?: ArrayLike<number> | null;            // 0/1 mask (e.g. land) drawn as cell-edge boundaries
  vectors?: { u: number[]; v: number[]; stride: number; scale: number } | null;   // arrows (m/s -> px per m/s)
  range?: [number, number];
}

/** Horizontal (x-y) section of a regional-model field, y upward. */
export function xySvg(p: XYPlot, size = 420): string {
  const ml = 44, mr = 12, mt = 26, mb = 30, W = size, H = size;
  const nx = p.nx, ny = p.ny;
  let vmin = Infinity, vmax = -Infinity, amax = 0;
  for (const v of p.values) { vmin = Math.min(vmin, v); vmax = Math.max(vmax, v); amax = Math.max(amax, Math.abs(v)); }
  if (p.range) { vmin = p.range[0]; vmax = p.range[1]; amax = Math.max(Math.abs(vmin), Math.abs(vmax)); }
  const cw = W / nx, ch = H / ny;
  const out: string[] = [`<svg xmlns="http://www.w3.org/2000/svg" width="${W + ml + mr}" height="${H + mt + mb}" font-family="sans-serif" font-size="11">`,
    `<rect width="100%" height="100%" fill="#fff"/>`, `<text x="${ml}" y="16" font-size="13" font-weight="bold">${p.title} (${p.units})</text>`];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const v = p.values[j * nx + i]!;
    const t = Math.max(-1, Math.min(1, p.diverging ? v / (amax || 1) : (v - vmin) / ((vmax - vmin) || 1)));
    if (!p.diverging && t < 0.02) continue;
    out.push(`<rect x="${(ml + i * cw).toFixed(1)}" y="${(mt + H - (j + 1) * ch).toFixed(1)}" width="${(cw + 0.5).toFixed(1)}" height="${(ch + 0.5).toFixed(1)}" fill="${color(t, p.diverging)}"/>`);
  }
  const xs = Array.from({ length: nx }, (_, i) => ml + (i + 0.5) * cw), ys = Array.from({ length: ny }, (_, j) => mt + H - (j + 0.5) * ch);
  const val = (r: number, c: number): number => p.values[r * nx + c]!;
  const lo = Math.ceil(vmin / p.contourStep), hi = Math.floor(vmax / p.contourStep);
  for (let i = lo; i <= hi; i++) {
    if (i === 0) continue;
    const d = contours(xs, ys, val, i * p.contourStep);
    if (d) out.push(`<path d="${d}" stroke="${i < 0 ? '#333' : '#000'}" stroke-width="0.7" ${i < 0 ? 'stroke-dasharray="3,2"' : ''} fill="none"/>`);
  }
  if (p.outline) {
    const o = p.outline, seg: string[] = [];
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const a = o[j * nx + i]!;
      const x0 = ml + i * cw, y0 = mt + H - j * ch;
      if (i + 1 < nx && o[j * nx + i + 1]! !== a) seg.push(`M${(x0 + cw).toFixed(1)},${y0.toFixed(1)}v${(-ch).toFixed(1)}`);
      if (j + 1 < ny && o[(j + 1) * nx + i]! !== a) seg.push(`M${x0.toFixed(1)},${(y0 - ch).toFixed(1)}h${cw.toFixed(1)}`);
    }
    if (seg.length) out.push(`<path d="${seg.join('')}" stroke="#7a4a12" stroke-width="1.4" fill="none"/>`);
  }
  if (p.vectors) {
    const { u, v, stride, scale } = p.vectors, arr: string[] = [];
    for (let j = Math.floor(stride / 2); j < ny; j += stride) for (let i = Math.floor(stride / 2); i < nx; i += stride) {
      const x = ml + (i + 0.5) * cw, y = mt + H - (j + 0.5) * ch, du = u[j * nx + i]! * scale, dv = -v[j * nx + i]! * scale;
      const L = Math.hypot(du, dv);
      if (L < 0.5) continue;
      const hx = du / L, hy = dv / L, h = Math.min(4, 0.35 * L);
      arr.push(`M${x.toFixed(1)},${y.toFixed(1)}l${du.toFixed(1)},${dv.toFixed(1)}m${(-h * hx - 0.6 * h * hy).toFixed(1)},${(-h * hy + 0.6 * h * hx).toFixed(1)}l${(h * hx + 0.6 * h * hy).toFixed(1)},${(h * hy - 0.6 * h * hx).toFixed(1)}l${(-h * hx + 0.6 * h * hy).toFixed(1)},${(-h * hy - 0.6 * h * hx).toFixed(1)}`);
    }
    if (arr.length) out.push(`<path d="${arr.join('')}" stroke="#123" stroke-width="0.9" fill="none"/>`);
  }
  out.push(`<rect x="${ml}" y="${mt}" width="${W}" height="${H}" fill="none" stroke="#000"/>`);
  const Lkm = nx * p.dx / 1000, step = Lkm > 600 ? 200 : Lkm > 250 ? 50 : 20;
  for (let km = 0; km <= Lkm; km += step) {
    out.push(`<text x="${ml + km * 1000 / p.dx * cw}" y="${mt + H + 14}" text-anchor="middle">${km}</text>`);
    out.push(`<text x="${ml - 4}" y="${mt + H - km * 1000 / p.dx * ch + 4}" text-anchor="end">${km}</text>`);
  }
  out.push(`<text x="${ml + W}" y="${mt + H + 28}" text-anchor="end" fill="#555">km; range ${vmin.toPrecision(3)} … ${vmax.toPrecision(3)}</text></svg>`);
  return out.join('\n');
}
