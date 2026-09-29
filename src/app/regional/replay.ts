// Replay of the 3-D view: the page keeps the display volumes it has received (cloud and precipitation bytes, the
// ground picture and the storms) and can show them again. Memory is bounded: large grids are kept at half the
// horizontal resolution, and when the budget is full every other frame is dropped and the spacing between kept frames
// doubles, so the replay always covers the whole run.
import type { StormNow } from '../../regional/storms.js';

export interface ReplayFrame {
  t: number; nx: number; ny: number; nz: number;
  /** domain width and height (m) and the vertical exaggeration of the view */
  Lx: number; Ly: number; top: number; aspect: number;
  cloud: Uint8Array; rain: Uint8Array; ground: Uint8Array; storms: StormNow[];
}

/** keep frames at full resolution up to this many cells, else at half the horizontal resolution */
const FULL_CELLS = 1.2e6;

export class ReplayStore {
  frames: ReplayFrame[] = [];
  private bytes = 0;
  /** smallest model time between kept frames (s); grows as the store thins itself */
  private spacing = 0;
  constructor(private readonly budget: number) {}

  clear(): void { this.frames = []; this.bytes = 0; this.spacing = 0; }
  get length(): number { return this.frames.length; }
  get megabytes(): number { return this.bytes / 1e6; }

  /** Keep a frame (the arrays are kept as they are, or halved; they must not be changed afterwards). */
  push(f: ReplayFrame): void {
    const last = this.frames[this.frames.length - 1];
    if (last && f.t < last.t - 1e-6) this.clear();                 // an earlier time: a new run or a loaded save
    else if (last && f.t - last.t < Math.max(this.spacing, 1e-6) - 1e-6) return;
    const k = f.nx * f.ny * f.nz > FULL_CELLS ? halve(f) : f;
    this.frames.push(k);
    this.bytes += size(k);
    while (this.bytes > this.budget && this.frames.length > 8) {
      // drop every other frame (keeping the first and the newest)
      const keep = this.frames.filter((_, i) => i % 2 === 0 || i === this.frames.length - 1);
      this.frames = keep;
      this.bytes = keep.reduce((a, x) => a + size(x), 0);
      this.spacing = Math.max(2 * this.spacing, keep.length > 1 ? (keep[keep.length - 1]!.t - keep[0]!.t) / (keep.length - 1) : 0);
    }
  }
}

const size = (f: ReplayFrame): number => f.cloud.length + f.rain.length + f.ground.length + 200 * f.storms.length;

/** Half the horizontal resolution (2 x 2 averages; the storms keep their positions). */
function halve(f: ReplayFrame): ReplayFrame {
  const nx = Math.max(1, f.nx >> 1), ny = Math.max(1, f.ny >> 1), nz = f.nz;
  const cloud = new Uint8Array(nx * ny * nz), rain = new Uint8Array(nx * ny * nz), ground = new Uint8Array(nx * ny * 4);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const a = (k * f.ny + 2 * j) * f.nx + 2 * i, b = a + f.nx, o = (k * ny + j) * nx + i;
    cloud[o] = (f.cloud[a]! + f.cloud[a + 1]! + f.cloud[b]! + f.cloud[b + 1]! + 2) >> 2;
    rain[o] = (f.rain[a]! + f.rain[a + 1]! + f.rain[b]! + f.rain[b + 1]! + 2) >> 2;
  }
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) for (let c = 0; c < 4; c++) {
    const a = ((2 * j) * f.nx + 2 * i) * 4 + c, b = a + 4 * f.nx;
    ground[(j * nx + i) * 4 + c] = (f.ground[a]! + f.ground[a + 4]! + f.ground[b]! + f.ground[b + 4]! + 2) >> 2;
  }
  return { ...f, nx, ny, cloud, rain, ground };
}
