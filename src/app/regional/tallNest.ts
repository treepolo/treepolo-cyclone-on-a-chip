// The tall grid of the regional page ("局部加高", src/regional/tall.ts): a child of the running model over a square part of its domain, reaching higher than the model's lid, with
// its own lid and Rayleigh layer at its own top, coupled both ways (on the GPU: src/gpu/tallGpu.ts). The worker holds at most one. Like the eye nest (eyeNest.ts, which this does not
// touch) it is built from the running model, steps with it, moves with its fields when the domain follows a storm, and has its own 3-D volume for the view.

import type { RegionalModel } from '../../regional/core.js';
import { CFL_DEFAULT, DT_MIN_FRAC } from '../../regional/stepControl.js';
import { IceMicrophysics } from '../../regional/ice.js';
import { RegionalPhysics, type RegionalPhysicsConfig } from '../../regional/physics.js';
import { refineInto, type Accumulations } from '../../regional/refine.js';
import { type parentState, type NestGeom } from '../../regional/twoway.js';
import { tallCells, tallEmptyTargets, tallFeedback, tallGeometry, tallRelaxCells, tallSurface, tallTargets, eyeInTall, eyeToTallFeedback, type EyeInTall, type TallGeom, type TallSpec } from '../../regional/tall.js';
import { applyWind, type WindForcing } from '../../regional/forcing.js';
import { volumeBytes } from '../../regional/display.js';
import { COL } from '../../regional/diagnostics.js';
import { GpuRegional } from '../../gpu/regionalGpu.js';
import { GpuTall, GpuCylFeedback } from '../../gpu/tallGpu.js';
import { TALL_MAX_CELLS } from '../../regional/tall.js';
import { buildModel } from './build.js';
import { syncModel } from './eyeNest.js';
import type { RegionalSetup } from './setup.js';

const CFL_TARGET = CFL_DEFAULT;

type Surface = { tsk: Float64Array; wet: Float64Array } | null;

export class TallNest {
  gpu: GpuRegional | null = null;
  link: GpuTall | null = null;
  /** the eye nest's feedback into this grid (GPU), when the eye nest lies inside it and its ratios fit (see linkEye) */
  eyeLink: GpuCylFeedback | null = null;
  /** the same for the CPU: the eye nest's geometry in this grid */
  eyeIn: EyeInTall | null = null;
  /** child steps per host step */
  nsub = 1;
  /** the lasting wind forcings in this grid's coordinates */
  private forcings: WindForcing[] = [];

  private constructor(readonly g: TallGeom, readonly m: RegionalModel, readonly mp: IceMicrophysics, public cfg: RegionalPhysicsConfig | null,
    public phys: RegionalPhysics | null, readonly dt0: number) {
    this.hook();
  }
  /** The model applies the lasting winds at the start of each step (after the physics' own pre-step work). */
  private hook(): void {
    const prev = this.m.preStep;
    this.m.preStep = (mm): void => { prev?.(mm); if (this.forcings.length) applyWind(mm, this.forcings, mm.c.dt); };
  }
  /** The physics was made anew (a setting changed while running): hook the lasting winds again. */
  rehook(): void { this.hook(); }

  /**
   * The tall grid of `spec` in the host model p of set-up `setup` now: the same sounding (to the new top), the host's base wind below its top (and the environment's above it), the host
   * state interpolated below the host's top (no cloud, rain or departure above it), the host's surface sampled. A message when it cannot be built.
   */
  static build(p: RegionalModel, pAcc: Accumulations, setup: RegionalSetup, frameVel: { u: number; v: number }, surface: Surface, spec: TallSpec, gpuOk: boolean): TallNest | string {
    const g = tallGeometry(p.c, spec);
    if (typeof g === 'string') return g;
    if (tallCells(g) > TALL_MAX_CELLS) return `${(tallCells(g) / 1e6).toFixed(1)} M 格 / too many cells`;
    const b = buildModel({ ...setup, preset: 'custom', L: g.nx * g.dx, dx: g.dx, dz: g.dz, top: g.Z2, dt: 0, boundary: 'open', follow: false, init: 'none', initAmp: 0 }, gpuOk);
    const c = b.model;
    if (c.c.nx !== g.nx || c.c.ny !== g.nx || c.c.nz !== g.nz) return `內部錯誤：局部加高網格不符 / internal error: tall grid ${c.c.nx} x ${c.c.nz}, expected ${g.nx} x ${g.nz}`;
    const cc = c.c as { relaxCells?: number; relaxTau?: number };
    cc.relaxCells = tallRelaxCells(g.dx); cc.relaxTau = 300;
    // the frame of the host
    if (frameVel.u !== b.frame.u || frameVel.v !== b.frame.v) c.shiftFrame(frameVel.u - b.frame.u, frameVel.v - b.frame.v);
    // the host's base wind (a change of the environment by an interaction included) below its top, blended into the environment's above it
    const pn = p.c.nz;
    for (let k = 0; k < g.nz; k++) {
      const z = c.zc[k]!, wh = Math.max(0, Math.min(1, (g.Z1 + 0.5 * p.c.dz - z) / p.c.dz));
      if (wh <= 0) continue;
      const q = Math.max(0, Math.min(pn - 1, z / p.c.dz - 0.5)), k0 = Math.min(Math.floor(q), Math.max(0, pn - 2)), k1 = Math.min(k0 + 1, pn - 1), w = q - k0;
      c.ub[k] = c.ub[k]! + wh * ((1 - w) * p.ub[k0]! + w * p.ub[k1]! - c.ub[k]!); c.vb[k] = c.vb[k]! + wh * ((1 - w) * p.vb[k0]! + w * p.vb[k1]! - c.vb[k]!);
    }
    c.boundary = tallEmptyTargets(c);
    const cfg = b.physics;
    if (cfg) {
      cfg.frameVel = frameVel;
      if (surface && cfg.surface) cfg.surface = { tsk: tallSurface(p, g, surface.tsk), wet: tallSurface(p, g, surface.wet) };
    }
    const mp = new IceMicrophysics(c);
    // the state: the host's in the footprint below its top, the environment above (departures continue the top level's, no water)
    refineInto(p, c, g.x0, g.y0, pAcc, { rain: mp.rainAcc, snow: mp.snowAcc });
    tallTargets(p, null, c, g, 1, c.boundary);
    c.steps = 0;
    const phys = cfg ? new RegionalPhysics(c, cfg) : null;
    return new TallNest(g, c, mp, cfg, phys, c.c.dt);
  }

  /** child cells */
  get cells(): number { return tallCells(this.g); }

  /**
   * Child steps for a host step dt: the configured step, or (rate: the grid's largest Courant rate, 1/s) the adaptive one (target Courant number cfl, growing by at most 10 % per
   * check, within the acoustic limit on the GPU).
   */
  fit(dt: number, rate: number | null = null, cfl: number = CFL_TARGET): void {
    let dtc = this.dt0;
    if (rate !== null && Number.isFinite(rate)) {
      const c = this.m.c, cur = dt / this.nsub, ac = c.nsound * 0.45 * Math.min(c.dx, c.dy) / 350;
      const hi = this.gpu ? Math.max(this.dt0, Math.min(3 * this.dt0, ac)) : this.dt0;
      let next = Math.min(hi, cfl / Math.max(rate, 1e-9));
      if (next > cur) next = Math.min(next, 1.1 * cur);
      dtc = Math.max(DT_MIN_FRAC * this.dt0, next);
    }
    this.nsub = Math.max(1, Math.min(64, Math.ceil(dt / dtc - 1e-6)));
    this.m.c.dt = dt / this.nsub;
    this.gpu?.setDt(dt / this.nsub);
  }

  /** Largest advective Courant rate of the grid (1/s): GPU read-back or CPU loop. */
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

  /**
   * The CPU steps of this grid for one host step that has just been taken (`old`: the host's state before it; `host.c.dt` its step): the child's sub-steps with targets interpolated in time,
   * then (an eye nest inside, `eye`) its feedback into this grid, then this grid's feedback into the host.
   */
  cpuAfter(host: RegionalModel, old: ReturnType<typeof parentState>, eye: RegionalModel | null): void {
    const c = this.m;
    c.c.dt = host.c.dt / this.nsub;
    for (let s = 0; s < this.nsub; s++) {
      tallTargets(host, old, c, this.g, (s + 0.5) / this.nsub, c.boundary!);
      c.step(); this.mp.apply(c.c.dt);
    }
    if (eye && this.eyeIn) eyeToTallFeedback(eye, c, this.eyeIn);
    tallFeedback(c, host, this.g);
  }

  /**
   * Whether the eye nest (geometry `e` in the host's cells) can feed back into this grid too; sets the CPU coupling and returns '' or the reason it cannot (then the eye nest feeds the host
   * only, and this grid's own feedback overwrites it there).
   */
  eyeCoupling(e: NestGeom | null): string {
    this.eyeIn = null;
    if (!e) return '';
    const r = eyeInTall(this.g, e);
    if (typeof r === 'string') return r;
    this.eyeIn = r;
    return '';
  }
  /** The GPU coupling of the eye nest's model into this grid (after eyeCoupling succeeded and both GPU models exist). */
  eyeCouplingGpu(eye: GpuRegional | null): void {
    this.eyeLink?.destroy(); this.eyeLink = null;
    if (eye && this.gpu && this.eyeIn) this.eyeLink = new GpuCylFeedback(this.gpu, eye, this.eyeIn.geom, this.eyeIn.nk);
  }

  /** Build the GPU model and its coupling to the host's (whenever the host's GPU model is built anew); returns the reason when the GPU failed ('' on success). */
  async gpuUp(host: GpuRegional): Promise<string> {
    this.gpuDown();
    const dev = host.device, scopes = ['validation', 'internal', 'out-of-memory'] as const;
    for (const sc of scopes) dev.pushErrorScope(sc);
    let reason = '';
    try {
      const cg = new GpuRegional(dev, this.m, { moist: true, physics: this.cfg, ice: true, boundary: this.m.boundary });
      this.gpu = cg;
      cg.setForcings(this.forcings);
      cg.uploadFrom(this.m, { rain: this.mp.rainAcc, snow: this.mp.snowAcc });
      this.link = new GpuTall(host, cg, this.g);
      cg.setDt(host.dt / this.nsub);
    } catch (e) { reason = String(e); }
    for (let i = 0; i < scopes.length; i++) { const err = await dev.popErrorScope(); if (err && !reason) reason = err.message; }
    if (reason) this.gpuDown();
    return reason;
  }
  gpuDown(): void { this.eyeLink?.destroy(); this.link?.destroy(); this.gpu?.destroy(); this.eyeLink = null; this.link = null; this.gpu = null; }

  /** Copy the GPU state into the CPU model. */
  async sync(): Promise<void> { if (this.gpu) await syncModel(this.gpu, this.m, this.mp); }

  /** The CPU state changed (an interaction): hand it to the GPU. */
  upload(): void { this.gpu?.uploadFrom(this.m, { rain: this.mp.rainAcc, snow: this.mp.snowAcc }); }

  /** Galilean change of the frame (the host's shiftFrame). */
  shiftFrame(du: number, dv: number): void { this.m.shiftFrame(du, dv); }

  /**
   * The host's fields were rolled by (di, dj) whole cells: this grid's fields move with them (r di, r dj cells) and the columns that come in are the host's, interpolated (they lie in the
   * relaxation ring or beyond). CPU state: sync before, rebuild the GPU models after.
   */
  roll(p: RegionalModel, pAcc: Accumulations, di: number, dj: number, surface: Surface): void {
    const r = this.g.r, n = this.g.nx;
    this.m.roll(di * r, dj * r, [this.mp.rainAcc, this.mp.snowAcc]);
    const enters = (i: number, d: number): boolean => (d > 0 ? i < d * r : d < 0 ? i >= n + d * r : false);
    refineInto(p, this.m, this.g.x0, this.g.y0, pAcc, { rain: this.mp.rainAcc, snow: this.mp.snowAcc }, (i, j) => enters(i, di) || enters(j, dj));
    this.surfaceFrom(p, surface);
  }

  /** The surface (skin temperature, wetness) sampled from the host's (painted or moved). */
  surfaceFrom(p: RegionalModel, surface: Surface): void {
    const own = this.phys?.surface;
    if (!own || !surface || !this.cfg) return;
    own.tsk.set(tallSurface(p, this.g, surface.tsk)); own.wet.set(tallSurface(p, this.g, surface.wet));
    this.cfg.surface = { tsk: own.tsk, wet: own.wet };
    this.gpu?.setSurface(own.tsk, own.wet);
  }

  /** The lasting wind forcings (host coordinates). */
  setForcings(list: readonly WindForcing[]): void {
    this.forcings = list.map((f) => ({ ...f, x: f.x - this.g.x0, y: f.y - this.g.y0 }));
    this.gpu?.setForcings(this.forcings);
  }

  /** The 3-D view bytes of the grid ([k][j][i]) and its extremes of w (a blow-up check). */
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
