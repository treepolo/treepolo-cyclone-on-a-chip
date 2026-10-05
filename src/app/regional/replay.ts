// Replay: the page keeps the frames the worker marks to be kept (recording interval, in model time): what the 3-D view draws
// (cloud and precipitation bytes, the ground picture, the storms), the numbers of the frame (time series, tracks) and, by the
// recording tier, the packed composite maps and model state with which the charts can be computed again (replayData.ts).
// Memory is bounded by a budget the user sets; when it is full a policy applies: thin the frames (every other one is dropped,
// the run stays covered), drop the oldest, or stop recording. Large grids are kept at half the horizontal resolution in the
// 3-D bytes (the packed state is always whole).
import type { StormNow } from '../../regional/storms.js';
import { recBytes, type ReplayMeta, type ReplayRec } from '../../regional/replayData.js';
import type { NestFrame, RegionalFrame } from './protocol.js';

export interface ReplayFrame {
  t: number; nx: number; ny: number; nz: number;
  /** horizontal and vertical grid spacing (m) of the model (the 3-D bytes may be at half the horizontal resolution) */
  dx: number; dz: number;
  /** domain width, height and top (m) */
  Lx: number; Ly: number; top: number;
  cloud: Uint8Array; rain: Uint8Array; ground: Uint8Array; storms: StormNow[];
  /** the eye nest's volume, if one ran */
  nest?: NestFrame | null;
  /** the frame's numbers and the ground-relative position of the domain origin */
  stats: RegionalFrame['stats'];
  origin: { x: number; y: number };
  /** what the recording tier added: packed maps and state (null: only the 3-D view) */
  rec?: ReplayRec | null;
}

export type FullPolicy = 'thin' | 'oldest' | 'stop';

/** keep the 3-D bytes at full resolution up to this many cells, else at half the horizontal resolution */
const FULL_CELLS = 1.2e6;

export class ReplayStore {
  frames: ReplayFrame[] = [];
  /** the models of the recorded frames (by id) */
  metas = new Map<number, ReplayMeta>();
  budget: number;
  policy: FullPolicy = 'thin';
  /** frames dropped by the policy since the store was cleared */
  dropped = 0;
  /** the 'stop' policy refused a frame: the store is full */
  full = false;
  private bytes = 0;
  constructor(budget: number) { this.budget = budget; }

  /** Forget the frames (the models stay: the worker sends a model's meta only once, and its ids never repeat; they are small). */
  clear(): void { this.frames = []; this.bytes = 0; this.dropped = 0; this.full = false; }
  get length(): number { return this.frames.length; }
  get megabytes(): number { return this.bytes / 1e6; }
  /** mean bytes of a frame (0 without frames) */
  get frameBytes(): number { return this.frames.length ? this.bytes / this.frames.length : 0; }

  /** Keep a frame (the arrays are kept as they are, or halved; they must not be changed afterwards). False: refused (full, policy 'stop'). */
  push(f: ReplayFrame): boolean {
    const last = this.frames[this.frames.length - 1];
    if (last && f.t < last.t - 1e-6) this.clear();                 // an earlier time: a new run or a loaded save
    else if (last && Math.abs(f.t - last.t) < 1e-6) return true;   // the same moment again
    if (f.rec?.meta) { this.metas.set(f.rec.metaId, f.rec.meta); delete f.rec.meta; }
    let k = f.nx * f.ny * f.nz > FULL_CELLS ? halve(f) : f;
    const n = k.nest;
    if (n && n.nx * n.nx * n.nz > FULL_CELLS) { const h = halveVol(n.nx, n.nx, n.nz, n.cloud, n.rain); k = { ...k, nest: { ...n, nx: h.nx, cloud: h.cloud, rain: h.rain } }; }
    const sz = size(k);
    if (this.policy === 'stop' && this.bytes + sz > this.budget && this.frames.length > 0) { this.full = true; return false; }
    this.frames.push(k);
    this.bytes += sz;
    if (this.policy === 'oldest') {
      while (this.bytes > this.budget && this.frames.length > 2) { this.bytes -= size(this.frames.shift()!); this.dropped++; }
    } else if (this.policy === 'thin') {
      while (this.bytes > this.budget && this.frames.length > 3) {
        // drop every other frame (keeping the first and the newest)
        const keep = this.frames.filter((_, i) => i % 2 === 0 || i === this.frames.length - 1);
        this.dropped += this.frames.length - keep.length;
        this.frames = keep;
        this.bytes = keep.reduce((a, x) => a + size(x), 0);
      }
    }
    return true;
  }

  /** The frames inside model-time segments [t0, t1] (s). */
  inSegments(segs: { t0: number; t1: number }[]): ReplayFrame[] {
    return this.frames.filter((f) => segs.some((s) => f.t >= s.t0 - 1e-6 && f.t <= s.t1 + 1e-6));
  }

  /** Replace the content with loaded frames (and their models). */
  load(frames: ReplayFrame[], metas: Map<number, ReplayMeta>): void {
    this.clear();
    this.frames = frames; this.metas = metas;
    this.bytes = frames.reduce((a, x) => a + size(x), 0);
  }
}

/** bytes a frame holds (about: the numbers of the frame count as 4 kB) */
export const size = (f: ReplayFrame): number => f.cloud.length + f.rain.length + f.ground.length + 200 * f.storms.length + 4000 + (f.nest ? f.nest.cloud.length + f.nest.rain.length : 0) + recBytes(f.rec);

/** A volume ([k][j][i] bytes) at half the horizontal resolution (2 x 2 averages). */
function halveVol(fnx: number, fny: number, nz: number, fc: Uint8Array, fr: Uint8Array): { nx: number; ny: number; cloud: Uint8Array; rain: Uint8Array } {
  const nx = Math.max(1, fnx >> 1), ny = Math.max(1, fny >> 1);
  const cloud = new Uint8Array(nx * ny * nz), rain = new Uint8Array(nx * ny * nz);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const a = (k * fny + 2 * j) * fnx + 2 * i, b = a + fnx, o = (k * ny + j) * nx + i;
    cloud[o] = (fc[a]! + fc[a + 1]! + fc[b]! + fc[b + 1]! + 2) >> 2;
    rain[o] = (fr[a]! + fr[a + 1]! + fr[b]! + fr[b + 1]! + 2) >> 2;
  }
  return { nx, ny, cloud, rain };
}

/** Half the horizontal resolution (2 x 2 averages; the storms keep their positions). */
function halve(f: ReplayFrame): ReplayFrame {
  const { nx, ny, cloud, rain } = halveVol(f.nx, f.ny, f.nz, f.cloud, f.rain), ground = new Uint8Array(nx * ny * 4);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) for (let c = 0; c < 4; c++) {
    const a = ((2 * j) * f.nx + 2 * i) * 4 + c, b = a + 4 * f.nx;
    ground[(j * nx + i) * 4 + c] = (f.ground[a]! + f.ground[a + 4]! + f.ground[b]! + f.ground[b + 4]! + 2) >> 2;
  }
  return { ...f, nx, ny, cloud, rain, ground };
}
