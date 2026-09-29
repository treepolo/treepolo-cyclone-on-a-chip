// Storm catalogue of the regional model (measurement only): finds every storm in the domain, gives each an
// identity that lasts while it lives, and keeps its track and key numbers.
//
//   * vortices (tropical cyclones and depressions; domains wider than 200 km): minima of the sea-level pressure
//     smoothed over about 30 km, at least VORTEX_DP hPa below the domain median, with a cyclonic mean tangential
//     wind of at least VORTEX_VT m/s within 150 km, separated by at least 200 km;
//   * convective cells (domains narrower than 400 km): connected areas of column-maximum updraft >= CELL_W m/s.
//
// Tracks are matched between analyses by the nearest detection to each track's predicted position (ground-relative,
// so frame changes and domain rolls do not break them). A track is shown once it has been seen twice; it ends
// when it has not been seen for LOST_* model seconds. Nothing is reported before a storm exists.

import type { RegionalModel } from './core.js';
import { C, COL } from './diagnostics.js';

export const VORTEX_DP = 2, VORTEX_VT = 3, CELL_W = 10;
const LOST_VORTEX = 3 * 3600, LOST_CELL = 900, SAMPLE_VORTEX = 600, SAMPLE_CELL = 60;

export type StormKind = 'vortex' | 'cell';
/** One storm now: ground-relative position (m), motion (m/s) and its key numbers. */
export interface StormNow {
  id: number; kind: StormKind; name: string;
  /** ground-relative position (m) and position in the domain (m) */
  x: number; y: number; xd: number; yd: number;
  /** motion over the ground (m/s), age (s) */
  u: number; v: number; age: number;
  /** vortices: minimum sea-level pressure (hPa), deficit below the domain median (hPa), maximum surface wind within
   *  150 km (m/s), radius of maximum azimuthal-mean tangential wind (m) */
  pmin?: number; dp?: number; vmax?: number; rmw?: number;
  /** cells: maximum updraft (m/s), 2-5 km updraft helicity with the larger magnitude (m^2/s^2), maximum
   *  reflectivity (dBZ), area of updraft above CELL_W (km^2) */
  wmax?: number; uh?: number; dbz?: number; area?: number;
}
export interface Detection { kind: StormKind; xd: number; yd: number; strength: number; data: Partial<StormNow> }
interface Track { id: number; kind: StormKind; born: number; seen: number; hits: number; x: number; y: number; u: number; v: number; now: StormNow; lastSample: number }

/** Sea-level pressure (hPa) from the lowest level: pressure from the Exner function, reduced hydrostatically over the
 *  level's height with its virtual temperature (the base-state moisture; GPU frames do not carry level-0 vapour). */
const slp = (pi: number, th: number, z0: number, qv: number): number => 1e5 * Math.pow(pi, 1004.5 / 287.05) / 100 * Math.exp(9.80665 * z0 / (287.05 * th * pi * (1 + 0.61 * qv)));

/** Vortex detections from the lowest model level (pi', u, v at level 0 of `m`). */
export function findVortices(m: RegionalModel, frame: { u: number; v: number }): Detection[] {
  const { nx, ny, dx, dy } = m.c, sx = m.sx, periodic = m.c.lateral !== 'open';
  const n = nx * ny, ps = new Float64Array(n);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const q = m.idx(i, j, 0); ps[j * nx + i] = slp(m.pi0[0]! + m.pp[q]!, m.th[q]!, m.zc[0]!, m.qv0[0]!); }
  // box smoothing over about 30 km (separable running sums, wrapping on periodic domains, clamped on open ones)
  const r = Math.max(1, Math.round(30000 / dx)), tmp = new Float64Array(n), sm = new Float64Array(n);
  const at = (a: number, len: number): number => (periodic ? ((a % len) + len) % len : Math.max(0, Math.min(len - 1, a)));
  const box = (src: Float64Array, dst: Float64Array, len: number, other: number, idx: (a: number, b: number) => number): void => {
    for (let b = 0; b < other; b++) {
      let s = 0;
      for (let d = -r; d <= r; d++) s += src[idx(at(d, len), b)]!;
      for (let a = 0; a < len; a++) {
        dst[idx(a, b)] = s / (2 * r + 1);
        s += src[idx(at(a + r + 1, len), b)]! - src[idx(at(a - r, len), b)]!;
      }
    }
  };
  box(ps, tmp, nx, ny, (a, b) => b * nx + a);
  box(tmp, sm, ny, nx, (a, b) => a * nx + b);
  const sorted = Float64Array.from(sm).sort(), median = sorted[n >> 1]!;
  // candidate minima: local minima (3 x 3) deep enough, strongest first, none within 200 km of a stronger one
  const sep = Math.max(200000, 6 * dx), all: { i: number; j: number; p: number }[] = [];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const p = sm[j * nx + i]!;
    if (median - p < VORTEX_DP) continue;
    let low = true;
    for (let b = -1; b <= 1 && low; b++) for (let a = -1; a <= 1; a++) {
      if (a === 0 && b === 0) continue;
      const ii = i + a, jj = j + b;
      if (!periodic && (ii < 0 || jj < 0 || ii >= nx || jj >= ny)) continue;
      const pq = sm[at(jj, ny) * nx + at(ii, nx)]!;
      if (pq < p || (pq === p && (b < 0 || (b === 0 && a < 0)))) { low = false; break; }
    }
    if (low) all.push({ i, j, p });
  }
  all.sort((a, b) => a.p - b.p);
  const Lx = nx * dx, Ly = ny * dy, cands: typeof all = [];
  for (const c of all) {
    const far = cands.every((k) => {
      let ex = (c.i - k.i) * dx, ey = (c.j - k.j) * dy;
      if (periodic) { ex -= Math.round(ex / Lx) * Lx; ey -= Math.round(ey / Ly) * Ly; }
      return Math.hypot(ex, ey) >= sep;
    });
    if (far) cands.push(c);
    if (cands.length >= 12) break;
  }
  const out: Detection[] = [], f = m.c.f, sign = f < 0 ? -1 : 1, R = 150000, nb = Math.max(2, Math.ceil(R / dx));
  for (const c of cands) {
    // azimuthal-mean tangential wind in rings around the raw minimum near the candidate; maximum surface wind
    let ic = c.i, jc = c.j, pmin = Infinity;
    for (let b = -r; b <= r; b++) for (let a = -r; a <= r; a++) { const ii = at(c.i + a, nx), jj = at(c.j + b, ny), p = ps[jj * nx + ii]!; if (p < pmin) { pmin = p; ic = ii; jc = jj; } }
    const vt = new Float64Array(nb), cnt = new Float64Array(nb);
    let vmax = 0;
    for (let b = -nb; b <= nb; b++) for (let a = -nb; a <= nb; a++) {
      const rr = Math.hypot(a * dx, b * dy);
      if (rr === 0 || rr >= R) continue;
      const ii = ic + a, jj = jc + b;
      if (!periodic && (ii < 0 || jj < 0 || ii >= nx || jj >= ny)) continue;
      const q = m.idx(at(ii, nx), at(jj, ny), 0);
      const ua = 0.5 * (m.u[q]! + m.u[q + 1]!), va = 0.5 * (m.v[q]! + m.v[q + sx]!);
      const k = Math.floor(rr / dx); vt[k] = vt[k]! + (-ua * b * dy + va * a * dx) / rr; cnt[k] = cnt[k]! + 1;
      vmax = Math.max(vmax, Math.hypot(ua + frame.u, va + frame.v));
    }
    let tot = 0, tn = 0, best = -Infinity, rmw = 0;
    for (let k = 0; k < nb; k++) if (cnt[k]! > 0) { const v = sign * vt[k]! / cnt[k]!; tot += vt[k]!; tn += cnt[k]!; if (v > best) { best = v; rmw = (k + 0.5) * dx; } }
    const mean = tn ? sign * tot / tn : 0;
    if (mean < VORTEX_VT && !(f === 0 && -mean >= VORTEX_VT)) continue;
    out.push({ kind: 'vortex', xd: (ic + 0.5) * dx, yd: (jc + 0.5) * dy, strength: median - pmin, data: { pmin, dp: median - pmin, vmax, rmw } });
  }
  return out;
}

/** Convective-cell detections from the column composites (COL values per column, see diagnostics.ts). */
export function findCells(nx: number, ny: number, dx: number, dy: number, col: Float32Array, periodic: boolean): Detection[] {
  const n = nx * ny, lab = new Int32Array(n).fill(-1), out: Detection[] = [], stack: number[] = [];
  let next = 0;
  for (let s = 0; s < n; s++) {
    if (lab[s] !== -1 || col[COL * s + C.wmax]! < CELL_W) continue;
    // flood fill of the 8-connected updraft area (wrapping on periodic domains)
    lab[s] = next; stack.push(s);
    let sw = 0, sxw = 0, syw = 0, wmax = 0, uh = 0, dbz = -Infinity, cells = 0;
    const i0 = s % nx, j0 = Math.floor(s / nx);
    while (stack.length) {
      const c = stack.pop()!, i = c % nx, j = Math.floor(c / nx), w = col[COL * c + C.wmax]!;
      // position relative to the seed, unwrapped
      let di = i - i0, dj = j - j0;
      if (periodic) { di -= Math.round(di / nx) * nx; dj -= Math.round(dj / ny) * ny; }
      sw += w; sxw += w * di; syw += w * dj; cells++;
      if (w > wmax) wmax = w;
      const h = col[COL * c + C.uh]!; if (Math.abs(h) > Math.abs(uh)) uh = h;
      dbz = Math.max(dbz, col[COL * c + C.dbz]!);
      for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) {
        let ii = i + a, jj = j + b;
        if (periodic) { ii = (ii + nx) % nx; jj = (jj + ny) % ny; } else if (ii < 0 || jj < 0 || ii >= nx || jj >= ny) continue;
        const cc = jj * nx + ii;
        if (lab[cc] === -1 && col[COL * cc + C.wmax]! >= CELL_W) { lab[cc] = next; stack.push(cc); }
      }
    }
    next++;
    let xd = (i0 + 0.5 + sxw / sw) * dx, yd = (j0 + 0.5 + syw / sw) * dy;
    if (periodic) { xd = ((xd % (nx * dx)) + nx * dx) % (nx * dx); yd = ((yd % (ny * dy)) + ny * dy) % (ny * dy); }
    out.push({ kind: 'cell', xd, yd, strength: Math.abs(uh) + 10 * wmax, data: { wmax, uh, dbz, area: cells * dx * dy / 1e6 } });
  }
  return out.sort((a, b) => b.strength - a.strength).slice(0, 24);
}

/** Tracks of every storm found by the analyses (see the file header). */
export class StormCatalog {
  private tracks: Track[] = [];
  private ended: StormNow[] = [];
  private nextId = { vortex: 1, cell: 1 };
  /** samples of each track (ground-relative), for the page's per-storm charts */
  readonly history = new Map<number, StormNow[]>();

  reset(): void { this.tracks = []; this.ended = []; this.nextId = { vortex: 1, cell: 1 }; this.history.clear(); }

  /** The storms shown now (seen at least twice). */
  get active(): StormNow[] { return this.tracks.filter((t) => t.hits >= 2).map((t) => t.now); }

  /** The strongest shown storm: the deepest vortex, else the cell with the largest |UH| + 10 w. */
  main(): StormNow | null {
    const a = this.active;
    const v = a.filter((s) => s.kind === 'vortex').sort((p, q) => (q.dp ?? 0) - (p.dp ?? 0))[0];
    if (v) return v;
    return a.filter((s) => s.kind === 'cell').sort((p, q) => (Math.abs(q.uh ?? 0) + 10 * (q.wmax ?? 0)) - (Math.abs(p.uh ?? 0) + 10 * (p.wmax ?? 0)))[0] ?? null;
  }

  /**
   * One analysis at model time t: detections in domain coordinates, the ground-relative position of the domain
   * origin, and the domain size (periodic domains wrap the displacement between analyses).
   */
  update(t: number, found: Detection[], origin: { x: number; y: number }, L: { x: number; y: number } | null): void {
    const wrap = (d: number, len: number | undefined): number => (len ? d - Math.round(d / len) * len : d);
    const used = new Set<number>();
    // strongest tracks pick first
    const order = [...this.tracks].sort((a, b) => b.hits - a.hits);
    for (const tr of order) {
      const dtt = Math.max(0, t - tr.seen), px = tr.x + tr.u * dtt, py = tr.y + tr.v * dtt;
      const gate = tr.kind === 'vortex' ? 250000 : Math.max(12000, 30 * dtt);
      let best = -1, bd = Infinity;
      found.forEach((d, n) => {
        if (used.has(n) || d.kind !== tr.kind) return;
        const gx = origin.x + d.xd, gy = origin.y + d.yd;
        const e = Math.hypot(wrap(gx - px, L?.x), wrap(gy - py, L?.y));
        if (e < gate && e < bd) { bd = e; best = n; }
      });
      if (best < 0) continue;
      used.add(best);
      const d = found[best]!, gx = origin.x + d.xd, gy = origin.y + d.yd;
      // ground-relative position continues across periodic wraps
      const nx = tr.x + wrap(gx - tr.x, L?.x), ny = tr.y + wrap(gy - tr.y, L?.y);
      if (t > tr.seen + 1e-6) {
        const a = Math.min(1, (t - tr.seen) / (tr.kind === 'vortex' ? 6 * 3600 : 900));
        tr.u += a * ((nx - tr.x) / (t - tr.seen) - tr.u); tr.v += a * ((ny - tr.y) / (t - tr.seen) - tr.v);
      }
      tr.x = nx; tr.y = ny; tr.seen = t; tr.hits++;
      tr.now = { ...tr.now, ...d.data, x: nx, y: ny, xd: d.xd, yd: d.yd, u: tr.u, v: tr.v, age: t - tr.born };
    }
    // new tracks
    found.forEach((d, n) => {
      if (used.has(n)) return;
      const id = this.nextId[d.kind]++, gx = origin.x + d.xd, gy = origin.y + d.yd;
      const name = d.kind === 'vortex' ? `TC${id}` : `C${id}`;
      this.tracks.push({ id: d.kind === 'vortex' ? id : 1000 + id, kind: d.kind, born: t, seen: t, hits: 1, x: gx, y: gy, u: 0, v: 0, lastSample: -Infinity,
        now: { id: d.kind === 'vortex' ? id : 1000 + id, kind: d.kind, name, x: gx, y: gy, xd: d.xd, yd: d.yd, u: 0, v: 0, age: 0, ...d.data } });
    });
    // lost tracks end; the history of shown ones stays
    const keep: Track[] = [];
    for (const tr of this.tracks) {
      const lost = t - tr.seen > (tr.kind === 'vortex' ? LOST_VORTEX : LOST_CELL) || t < tr.born - 1e-6;
      if (!lost) { keep.push(tr); continue; }
      if (tr.hits >= 2) { this.ended.push(tr.now); if (this.ended.length > 60) this.ended.shift(); }
    }
    this.tracks = keep;
    // samples of shown tracks
    for (const tr of this.tracks) {
      if (tr.hits < 2 || tr.seen !== t || t - tr.lastSample < (tr.kind === 'vortex' ? SAMPLE_VORTEX : SAMPLE_CELL) - 1e-6) continue;
      tr.lastSample = t;
      const h = this.history.get(tr.id) ?? [];
      h.push({ ...tr.now });
      if (h.length > 2000) h.splice(0, h.length - 2000);
      this.history.set(tr.id, h);
    }
    // bounded number of remembered storms
    if (this.history.size > 80) for (const id of [...this.history.keys()].slice(0, this.history.size - 80)) if (!this.tracks.some((tr) => tr.id === id)) this.history.delete(id);
  }

  /** Storms that ended (most recent last). */
  get finished(): StormNow[] { return this.ended.slice(); }
}

/** Detections for the model's domain: vortices on domains wider than 200 km, cells on domains narrower than 400 km. */
export function findStorms(m: RegionalModel, col: Float32Array | null, frame: { u: number; v: number }): Detection[] {
  const { nx, ny, dx, dy } = m.c, L = Math.min(nx * dx, ny * dy), out: Detection[] = [];
  if (L > 200000) out.push(...findVortices(m, frame));
  if (L < 400000 && col) out.push(...findCells(nx, ny, dx, dy, col, m.c.lateral !== 'open'));
  return out;
}
