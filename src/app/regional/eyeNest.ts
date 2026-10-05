// The eye nest of the regional page: a finer grid (across and up) in a cylinder at the centre of the running model,
// coupled both ways (src/regional/twoway.ts; on the GPU src/gpu/nestGpu.ts). The outer domain follows the storm, so the
// cylinder, fixed at its centre, stays on the eye. The worker holds at most one; this module builds it from the running
// (outer) model, steps it with the outer model (CPU) or wires the GPU coupling, and moves it with the outer fields when
// the domain is re-centred.

import type { RegionalModel } from '../../regional/core.js';
import { IceMicrophysics } from '../../regional/ice.js';
import { RegionalPhysics, type RegionalPhysicsConfig } from '../../regional/physics.js';
import { refineInto, type Accumulations } from '../../regional/refine.js';
import { nestGeometry, nestCells, nestRelax, emptyTargets, nestTargets, nestFeedback, nestSurface, parentState, type NestGeom } from '../../regional/twoway.js';
import { applyWind, type WindForcing } from '../../regional/forcing.js';
import { volumeBytes } from '../../regional/display.js';
import { COL } from '../../regional/diagnostics.js';
import { GpuRegional } from '../../gpu/regionalGpu.js';
import { GpuNest } from '../../gpu/nestGpu.js';
import { buildModel } from './build.js';
import type { RegionalSetup } from './setup.js';

/** largest inner grid (cells) */
export const NEST_MAX_CELLS = 30e6;
/** Courant number the adaptive inner step aims at (as the outer model's) */
const CFL_TARGET = 0.8;

/** Copy a GPU model's state (and surface accumulations) into its CPU model. */
export async function syncModel(g: GpuRegional, m: RegionalModel, mp: IceMicrophysics): Promise<void> {
  const st = await g.readState(), size = m.size;
  [m.u, m.v, m.w, m.th, m.pp, ...m.scalars].forEach((a, f) => { if (f < g.nf) for (let i = 0; i < size; i++) a[i] = st[f * size + i]!; });
  const rain = await g.readRain(), snow = await g.readSnow(), { nx, ny } = m.c;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { mp.rainAcc[j * nx + i] = rain[m.idx(i, j, 0)]!; mp.snowAcc[j * nx + i] = snow[m.idx(i, j, 0)]!; }
  m.time = g.time; m.steps = g.steps;
}

type Surface = { tsk: Float64Array; wet: Float64Array } | null;

export class EyeNest {
  gpu: GpuRegional | null = null;
  link: GpuNest | null = null;
  /** inner steps per outer step */
  nsub = 1;
  /** the lasting wind forcings in the inner grid's coordinates */
  private forcings: WindForcing[] = [];

  private constructor(readonly g: NestGeom, readonly m: RegionalModel, readonly mp: IceMicrophysics, readonly cfg: RegionalPhysicsConfig | null,
    readonly phys: RegionalPhysics | null, readonly dt0: number, readonly x0: number, readonly y0: number) {
    const prev = m.preStep;
    m.preStep = (mm): void => { prev?.(mm); if (this.forcings.length) applyWind(mm, this.forcings, mm.c.dt); };
  }

  /**
   * The inner grid of radius R (m) and spacings near dx, dz (m) in the outer model p of set-up `setup` now: the same
   * sounding and top, the outer base wind (its frame and any change of the environment), the outer state interpolated,
   * the outer surface sampled. A message when it cannot be built.
   */
  static build(p: RegionalModel, pAcc: Accumulations, setup: RegionalSetup, frameVel: { u: number; v: number }, surface: Surface,
    R: number, dx: number, dz: number, gpuOk: boolean): EyeNest | string {
    const g = nestGeometry(p.c, R, dx, dz);
    if (typeof g === 'string') return g;
    const cells = nestCells(g);
    if (cells > NEST_MAX_CELLS) return `細化區 ${(cells / 1e6).toFixed(1)} M 格，上限 ${NEST_MAX_CELLS / 1e6} M：請縮小半徑或加大 Δx、Δz / ${(cells / 1e6).toFixed(1)} M inner cells, at most ${NEST_MAX_CELLS / 1e6} M`;
    const b = buildModel({ ...setup, preset: 'custom', L: g.nx * g.dx, dx: g.dx, dz: g.dz, top: p.c.nz * p.c.dz, dt: 0, boundary: 'open', follow: false, initAmp: 0 }, gpuOk);
    const c = b.model;
    if (c.c.nx !== g.nx || c.c.ny !== g.nx || c.c.nz !== g.nz) return `內部錯誤：細化網格不符 / internal error: inner grid ${c.c.nx} x ${c.c.nz}, expected ${g.nx} x ${g.nz}`;
    Object.assign(c.c, nestRelax(g));
    // the outer base wind now, at the inner levels
    const pn = p.c.nz;
    for (let k = 0; k < g.nz; k++) {
      const q = Math.max(0, Math.min(pn - 1, c.zc[k]! / p.c.dz - 0.5)), k0 = Math.min(Math.floor(q), Math.max(0, pn - 2)), k1 = Math.min(k0 + 1, pn - 1), w = q - k0;
      c.ub[k] = (1 - w) * p.ub[k0]! + w * p.ub[k1]!; c.vb[k] = (1 - w) * p.vb[k0]! + w * p.vb[k1]!;
    }
    c.boundary = emptyTargets(c);
    const cfg = b.physics;
    if (cfg) {
      cfg.frameVel = frameVel;
      if (surface && cfg.surface) cfg.surface = { tsk: nestSurface(p, g, surface.tsk), wet: nestSurface(p, g, surface.wet) };
    }
    const mp = new IceMicrophysics(c), x0 = g.i0 * p.c.dx, y0 = g.j0 * p.c.dy;
    refineInto(p, c, x0, y0, pAcc, { rain: mp.rainAcc, snow: mp.snowAcc });
    nestTargets(p, null, c, g, 1, c.boundary);
    c.steps = 0;
    const phys = cfg ? new RegionalPhysics(c, cfg) : null;
    return new EyeNest(g, c, mp, cfg, phys, c.c.dt, x0, y0);
  }

  /** inner cells */
  get cells(): number { return nestCells(this.g); }

  /**
   * Inner steps for an outer step dt: the configured inner step, or (rate: the inner grid's largest Courant rate, 1/s)
   * the adaptive one (target Courant number cfl, 0.8 by default, growing by at most 10 % per check, within the acoustic limit on the GPU).
   */
  fit(dt: number, rate: number | null = null, cfl: number = CFL_TARGET): void {
    let dtc = this.dt0;
    if (rate !== null && Number.isFinite(rate)) {
      const c = this.m.c, cur = dt / this.nsub, ac = c.nsound * 0.45 * Math.min(c.dx, c.dy) / 350;
      const hi = this.gpu ? Math.max(this.dt0, Math.min(3 * this.dt0, ac)) : this.dt0;
      let next = Math.min(hi, cfl / Math.max(rate, 1e-9));
      if (next > cur) next = Math.min(next, 1.1 * cur);
      dtc = Math.max(0.25 * this.dt0, next);
    }
    this.nsub = Math.max(1, Math.min(64, Math.ceil(dt / dtc - 1e-6)));
    this.m.c.dt = dt / this.nsub;
    this.gpu?.setDt(dt / this.nsub);
  }

  /** Largest advective Courant rate of the inner grid (1/s): GPU read-back or CPU loop. */
  async courantRate(): Promise<number> {
    if (this.gpu) return this.gpu.maxCourantRate();
    const m = this.m, { nx, ny, nz, dx, dy, dz } = m.c;
    let r = 0;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const q = m.idx(i, j, k);
      r = Math.max(r, Math.abs(m.u[q]!) / dx + Math.abs(m.v[q]!) / dy + Math.abs(m.w[q]!) / dz);
    }
    return r;
  }

  /** One outer step on the CPU with the inner grid: the outer model, the inner sub-steps (targets interpolated in time
   *  between the outer states), the feedback. */
  cpuStep(p: RegionalModel, pmp: IceMicrophysics): void {
    const old = parentState(p), c = this.m;
    p.step(); pmp.apply(p.c.dt);
    c.c.dt = p.c.dt / this.nsub;
    for (let s = 0; s < this.nsub; s++) {
      nestTargets(p, old, c, this.g, (s + 0.5) / this.nsub, c.boundary!);
      c.step(); this.mp.apply(c.c.dt);
    }
    nestFeedback(c, p, this.g);
  }

  /** Build the inner GPU model and its coupling to the outer one (whenever the outer GPU model is built anew); returns
   *  the reason when the GPU failed ('' on success). */
  async gpuUp(outer: GpuRegional): Promise<string> {
    this.gpuDown();
    const dev = outer.device, scopes = ['validation', 'internal', 'out-of-memory'] as const;
    for (const sc of scopes) dev.pushErrorScope(sc);
    let reason = '';
    try {
      const cg = new GpuRegional(dev, this.m, { moist: true, physics: this.cfg, ice: true, boundary: this.m.boundary });
      this.gpu = cg;
      cg.setForcings(this.forcings);
      cg.uploadFrom(this.m, { rain: this.mp.rainAcc, snow: this.mp.snowAcc });
      this.link = new GpuNest(outer, cg, this.g);
      cg.setDt(outer.dt / this.nsub);
    } catch (e) { reason = String(e); }
    for (let i = 0; i < scopes.length; i++) { const err = await dev.popErrorScope(); if (err && !reason) reason = err.message; }
    if (reason) this.gpuDown();
    return reason;
  }
  gpuDown(): void { this.link?.destroy(); this.gpu?.destroy(); this.link = null; this.gpu = null; }

  /** Copy the inner GPU state into the CPU model. */
  async sync(): Promise<void> { if (this.gpu) await syncModel(this.gpu, this.m, this.mp); }

  /** The CPU state changed (an interaction): hand it to the GPU. */
  upload(): void { this.gpu?.uploadFrom(this.m, { rain: this.mp.rainAcc, snow: this.mp.snowAcc }); }

  /** Galilean change of the frame (the outer model's shiftFrame). */
  shiftFrame(du: number, dv: number): void { this.m.shiftFrame(du, dv); }

  /**
   * The outer fields were rolled by (di, dj) whole cells: the inner fields move with them (r di, r dj inner cells) and
   * the columns coming in are the outer model's, interpolated (they lie in the relaxation ring or beyond). CPU state:
   * sync before, rebuild the GPU models after.
   */
  roll(p: RegionalModel, pAcc: Accumulations, di: number, dj: number, surface: Surface): void {
    const r = this.g.r, n = this.g.nx;
    this.m.roll(di * r, dj * r, [this.mp.rainAcc, this.mp.snowAcc]);
    const enters = (i: number, d: number): boolean => (d > 0 ? i < d * r : d < 0 ? i >= n + d * r : false);
    refineInto(p, this.m, this.x0, this.y0, pAcc, { rain: this.mp.rainAcc, snow: this.mp.snowAcc }, (i, j) => enters(i, di) || enters(j, dj));
    this.surfaceFrom(p, surface);
  }

  /** The inner surface (skin temperature, wetness) sampled from the outer one (painted or moved). */
  surfaceFrom(p: RegionalModel, surface: Surface): void {
    const own = this.phys?.surface;
    if (!own || !surface || !this.cfg) return;
    own.tsk.set(nestSurface(p, this.g, surface.tsk)); own.wet.set(nestSurface(p, this.g, surface.wet));
    this.cfg.surface = { tsk: own.tsk, wet: own.wet };
    this.gpu?.setSurface(own.tsk, own.wet);
  }

  /** The lasting wind forcings (outer coordinates). */
  setForcings(list: readonly WindForcing[]): void {
    this.forcings = list.map((f) => ({ ...f, x: f.x - this.x0, y: f.y - this.y0 }));
    this.gpu?.setForcings(this.forcings);
  }

  /** The 3-D view bytes of the inner grid ([k][j][i]) and its extremes of w (a blow-up check). */
  async volume(mode: number, subgrid: boolean): Promise<{ cloud: Uint8Array; rain: Uint8Array; wmax: number; wmin: number }> {
    const { nx, ny, nz } = this.m.c, n = nx * ny * nz, cloud = new Uint8Array(n), rain = new Uint8Array(n);
    if (this.gpu) {
      const d = await this.gpu.readDisplay([], mode, subgrid);
      for (let i = 0; i < n; i++) { const v = d.packed[i]!; cloud[i] = v & 255; rain[i] = (v >> 8) & 255; }
      let wmax = 0, wmin = 0;
      for (let c = 0; c < nx * ny; c++) { wmax = Math.max(wmax, d.col[COL * c]!); wmin = Math.min(wmin, d.col[COL * c + 1]!); }
      return { cloud, rain, wmax, wmin };
    }
    const r = volumeBytes(this.m, cloud, rain, mode, subgrid);
    return { cloud, rain, wmax: r.wmax, wmin: r.wmin };
  }

  destroy(): void { this.gpuDown(); }
}
