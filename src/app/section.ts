// Latitude–sigma cross-section drawn into a 2D canvas (north on the left, top of atmosphere up).
import { diverging, sequential, rgbCss } from './colormap.js';

export function drawSection(canvas: HTMLCanvasElement, lat: number[], sigma: number[], values: number[],
                            opts: { diverging: boolean; contour: number; label: string }): void {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const W = Math.round(canvas.clientWidth * dpr), H = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const ml = 26 * dpr, mr = 6 * dpr, mt = 16 * dpr, mb = 16 * dpr;
  const w = W - ml - mr, h = H - mt - mb, nl = lat.length, nr = sigma.length;
  const xOf = (la: number): number => ml + (90 - la) / 180 * w;
  const yOf = (s: number): number => mt + s * h;
  let vmin = Infinity, vmax = -Infinity, amax = 0;
  for (const v of values) { vmin = Math.min(vmin, v); vmax = Math.max(vmax, v); amax = Math.max(amax, Math.abs(v)); }
  const le = (j: number): number => j === 0 ? 90 : j === nl ? -90 : 0.5 * (lat[j - 1]! + lat[j]!);
  const se = (k: number): number => k === 0 ? 0 : k === nr ? 1 : 0.5 * (sigma[k - 1]! + sigma[k]!);
  for (let k = 0; k < nr; k++) for (let j = 0; j < nl; j++) {
    const v = values[k * nl + j]!;
    ctx.fillStyle = rgbCss(opts.diverging ? diverging(v / (amax || 1)) : sequential((v - vmin) / ((vmax - vmin) || 1)));
    const x0 = xOf(le(j)), x1 = xOf(le(j + 1)), y0 = yOf(se(k)), y1 = yOf(se(k + 1));
    ctx.fillRect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
  }
  // contours via marching squares on cell centres
  ctx.lineWidth = dpr;
  const lo = Math.ceil(vmin / opts.contour), hi = Math.floor(vmax / opts.contour);
  for (let i = lo; i <= hi; i++) {
    const L = i === 0 ? 1e-12 : i * opts.contour;
    ctx.strokeStyle = i < 0 ? 'rgba(0,0,0,0.55)' : 'rgba(0,0,0,0.8)';
    ctx.setLineDash(i < 0 ? [3 * dpr, 2 * dpr] : []);
    ctx.lineWidth = (i === 0 ? 1.8 : 0.9) * dpr;
    ctx.beginPath();
    for (let k = 0; k < nr - 1; k++) for (let j = 0; j < nl - 1; j++) {
      const P = [[j, k], [j + 1, k], [j + 1, k + 1], [j, k + 1]] as const;
      const pts: [number, number][] = [];
      for (let e = 0; e < 4; e++) {
        const a = P[e]!, b = P[(e + 1) % 4]!;
        const va = values[a[1] * nl + a[0]]!, vb = values[b[1] * nl + b[0]]!;
        if ((va - L) * (vb - L) < 0) {
          const t = (L - va) / (vb - va);
          const xa = xOf(lat[a[0]]!), xb = xOf(lat[b[0]]!), ya = yOf(sigma[a[1]]!), yb = yOf(sigma[b[1]]!);
          pts.push([xa + t * (xb - xa), ya + t * (yb - ya)]);
        }
      }
      if (pts.length >= 2) { ctx.moveTo(pts[0]![0], pts[0]![1]); ctx.lineTo(pts[1]![0], pts[1]![1]); }
      if (pts.length === 4) { ctx.moveTo(pts[2]![0], pts[2]![1]); ctx.lineTo(pts[3]![0], pts[3]![1]); }
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.strokeStyle = '#8da0b3'; ctx.lineWidth = dpr;
  ctx.strokeRect(ml, mt, w, h);
  ctx.fillStyle = '#b8c6d4';
  ctx.font = `${10 * dpr}px sans-serif`;
  ctx.textAlign = 'center';
  for (const la of [60, 30, 0, -30, -60]) ctx.fillText(la === 0 ? 'EQ' : la > 0 ? `${la}N` : `${-la}S`, xOf(la), H - 4 * dpr);
  ctx.textAlign = 'right';
  for (const s of [0.2, 0.5, 0.8]) ctx.fillText(s.toFixed(1), ml - 3 * dpr, yOf(s) + 3 * dpr);
  ctx.textAlign = 'left';
  ctx.fillText(`${opts.label}   [${vmin.toPrecision(3)}, ${vmax.toPrecision(3)}]`, ml, 11 * dpr);
}
