// Regional fully compressible, non-hydrostatic model on a Cartesian C-grid (CM1 / WRF family).
//
// Prognostic variables: u, v, w, theta, pi' (Exner-function perturbation), passive/moist scalars.
// Base state: horizontally uniform hydrostatic sounding theta0(z), qv0(z), pi0(z), rho0(z).
//   du/dt = -adv(u) + f v - cp thv0 dpi'/dx + D_u
//   dv/dt = -adv(v) - f u - cp thv0 dpi'/dy + D_v
//   dw/dt = -adv(w) - cp thv0 dpi'/dz + B + D_w,      B = g [ theta'/theta0 + ... ]
//   dpi'/dt = -adv(pi') - (cs^2 / (cp rho0 thv0^2)) div(rho0 thv0 V)       (Klemp & Wilhelmson 1978)
//   dtheta/dt = -adv(theta) + Q
// Time integration: Wicker & Skamarock (2002) RK3 with split-explicit acoustic steps; the vertical
// w-pi' coupling is Crank–Nicolson (off-centred) and solved implicitly per column; divergence damping.
// Advection: 5th-order upwind-biased flux form (Wicker & Skamarock 2002), lower order near the lid/ground.
// Lateral boundaries: doubly periodic. Rigid lid with a Rayleigh damping layer; free-slip lower boundary.

import { DRY_AIR, EARTH } from '../core/constants.js';

export const H = 3; // halo width

export interface RegionalConfig {
  nx: number; ny: number; nz: number;
  dx: number; dy: number; dz: number;       // m (uniform vertical spacing)
  dt: number;                               // s, large (advective) step
  nsound: number;                           // acoustic steps per large step (multiple of 6 recommended)
  f: number;                                // Coriolis parameter (s^-1)
  beta: number;                             // vertical off-centring of the acoustic solve (0 = Crank–Nicolson)
  divDamp: number;                          // divergence damping coefficient (non-dimensional, ~0.1)
  dampDepth: number;                        // Rayleigh damping layer depth below the lid (m)
  dampRate: number;                         // Rayleigh damping max rate (s^-1)
  kdiff2: number;                           // constant eddy diffusivity (m^2/s) for tests (0 = off)
  /** lateral boundaries: doubly periodic (default) or open with a Davies relaxation zone */
  lateral?: 'periodic' | 'open';
  relaxCells?: number;                      // relaxation-zone width (cells), open boundaries
  relaxTau?: number;                        // relaxation time scale at the outer boundary (s)
  /** positive-definite flux limiter for the moisture scalars in the final RK3 stage (default on) */
  positiveDefinite?: boolean;
}

/** Boundary targets for open lateral boundaries (same layout as the prognostic arrays). */
export interface BoundaryTargets { u: Float64Array; v: Float64Array; th: Float64Array; qv: Float64Array | null; pp?: Float64Array | null }

export class RegionalModel {
  readonly c: RegionalConfig;
  readonly sx: number; readonly sy: number; readonly plane: number; readonly size: number;
  // base state (per level; "f" arrays at w levels k = 0..nz)
  readonly zc: Float64Array; readonly zf: Float64Array;
  readonly th0: Float64Array; readonly pi0: Float64Array; readonly rho0: Float64Array; readonly qv0: Float64Array;
  readonly th0f: Float64Array; readonly rho0f: Float64Array; readonly pi0f: Float64Array;
  /** base-state wind profile (m/s) the damping layer relaxes toward */
  readonly ub: Float64Array; readonly vb: Float64Array;
  // prognostic state
  u: Float64Array; v: Float64Array; w: Float64Array; th: Float64Array; pp: Float64Array;
  /** additional advected scalars (e.g. moisture species), same layout as th */
  readonly scalars: Float64Array[];
  time = 0;
  steps = 0;
  // work
  private readonly u0: Float64Array; private readonly v0: Float64Array; private readonly w0: Float64Array;
  private readonly th0s: Float64Array; private readonly pp0: Float64Array;
  private readonly fu: Float64Array; private readonly fv: Float64Array; private readonly fw: Float64Array;
  private readonly fth: Float64Array; private readonly fpp: Float64Array;
  private readonly fxF: Float64Array; private readonly fyF: Float64Array; private readonly fzF: Float64Array; private readonly pdRatio: Float64Array;
  private readonly sc0: Float64Array[]; private readonly fsc: Float64Array[];
  private readonly ppOld: Float64Array;
  private readonly flux: Float64Array;
  /** density potential temperature theta_rho = theta (1 + 0.61 qv - qc - qr) of the current stage */
  private readonly thr: Float64Array;
  /** buoyancy / extra slow forcing hook for w and theta (moist physics adds here) */
  buoyancy: ((m: RegionalModel, out: Float64Array) => void) | null = null;
  /** relaxation targets for open lateral boundaries (set by a nesting driver) */
  boundary: BoundaryTargets | null = null;
  /** sub-grid / surface / radiation slow tendencies hook */
  physicsTend: ((m: RegionalModel, t: { fu: Float64Array; fv: Float64Array; fw: Float64Array; fth: Float64Array; fsc: Float64Array[] }) => void) | null = null;

  constructor(cfg: RegionalConfig, sounding: (z: number) => { theta: number; qv: number }, nScalars = 0) {
    this.c = cfg;
    const { nx, ny, nz } = cfg;
    this.sx = nx + 2 * H; this.sy = ny + 2 * H;
    this.plane = this.sx * this.sy;
    this.size = this.plane * (nz + 1);          // nz+1 levels so w fits; scalars use k < nz
    this.zc = new Float64Array(nz); this.zf = new Float64Array(nz + 1);
    for (let k = 0; k <= nz; k++) this.zf[k] = k * cfg.dz;
    for (let k = 0; k < nz; k++) this.zc[k] = (k + 0.5) * cfg.dz;
    this.th0 = new Float64Array(nz); this.qv0 = new Float64Array(nz); this.pi0 = new Float64Array(nz); this.rho0 = new Float64Array(nz);
    this.th0f = new Float64Array(nz + 1); this.rho0f = new Float64Array(nz + 1); this.pi0f = new Float64Array(nz + 1);
    this.ub = new Float64Array(nz); this.vb = new Float64Array(nz);
    this.buildBaseState(sounding);
    const z = (): Float64Array => new Float64Array(this.size);
    this.u = z(); this.v = z(); this.w = z(); this.th = z(); this.pp = z();
    this.u0 = z(); this.v0 = z(); this.w0 = z(); this.th0s = z(); this.pp0 = z();
    this.fu = z(); this.fv = z(); this.fw = z(); this.fth = z(); this.fpp = z();
    this.fxF = z(); this.fyF = z(); this.fzF = z(); this.pdRatio = z();
    this.ppOld = z(); this.flux = z(); this.thr = z();
    this.scalars = Array.from({ length: nScalars }, z);
    this.sc0 = Array.from({ length: nScalars }, z);
    this.fsc = Array.from({ length: nScalars }, z);
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) this.th[this.idx(i, j, k)] = this.th0[k]!;
  }

  idx(i: number, j: number, k: number): number { return k * this.plane + (j + H) * this.sx + (i + H); }

  /** Hydrostatic base state from theta(z), qv(z) with p_surface = 1000 hPa. */
  private buildBaseState(snd: (z: number) => { theta: number; qv: number }): void {
    const { nz } = this.c, g = EARTH.gravity, cp = DRY_AIR.cp, rd = DRY_AIR.rd;
    const thv = (z: number): number => { const s = snd(z); return s.theta * (1 + 0.61 * s.qv); };
    // integrate d pi / dz = -g / (cp thv) with fine substeps
    let pi = 1.0, z = 0;
    const piAt = new Float64Array(2 * nz + 1); piAt[0] = 1;
    const dzh = this.c.dz / 2;
    for (let n = 1; n <= 2 * nz; n++) {
      const sub = 20, h = dzh / sub;
      for (let s = 0; s < sub; s++) { const zm = z + (s + 0.5) * h; pi -= g / (cp * thv(zm)) * h; }
      z += dzh;
      piAt[n] = pi;
    }
    for (let k = 0; k < nz; k++) {
      const s = snd(this.zc[k]!);
      this.th0[k] = s.theta; this.qv0[k] = s.qv; this.pi0[k] = piAt[2 * k + 1]!;
      const p = DRY_AIR.pRef * Math.pow(this.pi0[k]!, cp / rd);
      this.rho0[k] = p / (rd * this.pi0[k]! * s.theta * (1 + 0.61 * s.qv));
    }
    for (let k = 0; k <= nz; k++) {
      const s = snd(this.zf[k]!);
      this.th0f[k] = s.theta * (1 + 0.61 * s.qv); this.pi0f[k] = piAt[2 * k]!;
      const p = DRY_AIR.pRef * Math.pow(this.pi0f[k]!, cp / rd);
      this.rho0f[k] = p / (rd * this.pi0f[k]! * this.th0f[k]!);
    }
  }

  /** Fill periodic halos of a 3-D array for levels [0, nk). */
  fillHalo(a: Float64Array, nk: number): void {
    const { nx, ny } = this.c, sx = this.sx;
    if (this.c.lateral === 'open') {
      // zero-gradient extrapolation into the halo
      for (let k = 0; k < nk; k++) {
        const o = k * this.plane;
        for (let j = 0; j < ny; j++) {
          const r = o + (j + H) * sx;
          for (let h = 0; h < H; h++) { a[r + h] = a[r + H]!; a[r + nx + H + h] = a[r + nx + H - 1]!; }
        }
        for (let h = 0; h < H; h++) {
          a.copyWithin(o + h * sx, o + H * sx, o + (H + 1) * sx);
          a.copyWithin(o + (ny + H + h) * sx, o + (ny + H - 1) * sx, o + (ny + H) * sx);
        }
      }
      return;
    }
    for (let k = 0; k < nk; k++) {
      const o = k * this.plane;
      for (let j = 0; j < ny; j++) {
        const r = o + (j + H) * sx;
        for (let h = 0; h < H; h++) { a[r + h] = a[r + nx + h]!; a[r + nx + H + h] = a[r + H + h]!; }
      }
      for (let h = 0; h < H; h++) {
        a.copyWithin(o + h * sx, o + (ny + h) * sx, o + (ny + h + 1) * sx);
        a.copyWithin(o + (ny + H + h) * sx, o + (H + h) * sx, o + (H + h + 1) * sx);
      }
    }
  }

  /** theta_rho = theta (1 + 0.61 qv - sum of condensates); scalar slot 0 is qv, all further slots are
   *  condensate (qc, qr and, with ice microphysics, qi, qs, qg). */
  private computeThetaRho(): void {
    const n = this.size, th = this.th, qv = this.scalars[0], cond = this.scalars.slice(1), t = this.thr;
    for (let i = 0; i < n; i++) {
      let f = 1;
      if (qv) f += 0.61 * qv[i]!;
      for (const c of cond) f -= c[i]!;
      t[i] = th[i]! * f;
    }
    this.fillHalo(t, this.c.nz);
  }

  // ----------------------------------------------------------------------------------------
  // Advection helpers

  /** 5th-order upwind-biased interface value between a2|a3 given velocity sign (WS2002). */
  private static f5(a0: number, a1: number, a2: number, a3: number, a4: number, a5: number, vel: number): number {
    const c = (37 * (a2 + a3) - 8 * (a1 + a4) + (a0 + a5)) / 60;
    const d = (10 * (a3 - a2) - 5 * (a4 - a1) + (a5 - a0)) / 60;
    return vel >= 0 ? c - d : c + d;
  }
  /** 3rd-order upwind-biased interface value between a1|a2. */
  private static f3(a0: number, a1: number, a2: number, a3: number, vel: number): number {
    const c = (7 * (a1 + a2) - (a0 + a3)) / 12;
    const d = (3 * (a2 - a1) - (a3 - a0)) / 12;
    return vel >= 0 ? c - d : c + d;
  }

  /**
   * Advective tendency of a scalar field phi (cell centres), flux form with rho0 weighting and the
   * divergence correction, accumulated into out: out += -(1/rho0) div(rho0 V phi) + phi (1/rho0) div(rho0 V).
   */
  private advectScalar(phi: Float64Array, out: Float64Array, pd: { phi0: Float64Array; dt: number } | null = null): void {
    const { nx, ny, nz, dx, dy, dz } = this.c, sx = this.sx, pl = this.plane;
    const u = this.u, v = this.v, w = this.w;
    // face fluxes: fxF at the west face of cell q, fyF at the south face, fzF at the bottom face (rho0f w phi)
    const fxF = this.fxF, fyF = this.fyF, fzF = this.fzF;
    for (let k = 0; k < nz; k++) for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) {
      const q = this.idx(i, j, k);
      if (j < ny) { const ul = u[q]!; fxF[q] = ul * RegionalModel.f5(phi[q - 3]!, phi[q - 2]!, phi[q - 1]!, phi[q]!, phi[q + 1]!, phi[q + 2]!, ul); }
      if (i < nx) { const vl = v[q]!; fyF[q] = vl * RegionalModel.f5(phi[q - 3 * sx]!, phi[q - 2 * sx]!, phi[q - sx]!, phi[q]!, phi[q + sx]!, phi[q + 2 * sx]!, vl); }
    }
    for (let k = 0; k <= nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const q = this.idx(i, j, k);
      fzF[q] = k === 0 || k === nz ? 0 : w[q]! * this.rho0f[k]! * this.zface(phi, q, k, w[q]!);
    }
    if (pd) {
      // Positive-definite flux limiter (Skamarock 2006, MWR): scale each cell's outgoing fluxes so the
      // outflow over the step cannot exceed its content at time level n.
      const ratio = this.pdRatio;
      for (let k = 0; k < nz; k++) {
        const r0 = this.rho0[k]!;
        for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
          const q = this.idx(i, j, k);
          const outflow = (Math.max(fxF[q + 1]!, 0) - Math.min(fxF[q]!, 0)) / dx + (Math.max(fyF[q + sx]!, 0) - Math.min(fyF[q]!, 0)) / dy
            + (Math.max(fzF[q + pl]!, 0) - Math.min(fzF[q]!, 0)) / (r0 * dz);
          const avail = Math.max(pd.phi0[q]!, 0);
          ratio[q] = outflow * pd.dt > avail ? avail / (outflow * pd.dt) : 1;
        }
      }
      this.fillHalo(ratio, nz);
      for (let k = 0; k < nz; k++) for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) {
        const q = this.idx(i, j, k);
        if (j < ny) fxF[q] = fxF[q]! * (fxF[q]! > 0 ? ratio[q - 1]! : ratio[q]!);
        if (i < nx) fyF[q] = fyF[q]! * (fyF[q]! > 0 ? ratio[q - sx]! : ratio[q]!);
      }
      for (let k = 1; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const q = this.idx(i, j, k);
        fzF[q] = fzF[q]! * (fzF[q]! > 0 ? ratio[q - pl]! : ratio[q]!);
      }
    }
    for (let k = 0; k < nz; k++) {
      const r0 = this.rho0[k]!;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const q = this.idx(i, j, k);
        const div = (u[q + 1]! - u[q]!) / dx + (v[q + sx]! - v[q]!) / dy + (w[q + pl]! * this.rho0f[k + 1]! - w[q]! * this.rho0f[k]!) / (r0 * dz);
        out[q] = out[q]! - ((fxF[q + 1]! - fxF[q]!) / dx + (fyF[q + sx]! - fyF[q]!) / dy + (fzF[q + pl]! - fzF[q]!) / (r0 * dz)) + phi[q]! * div;
      }
    }
  }

  /** Value of a cell-centred field at w level kf (between kf-1 and kf), index qf = cell kf. */
  private zface(phi: Float64Array, qf: number, kf: number, vel: number): number {
    const nz = this.c.nz, pl = this.plane;
    if (kf >= 3 && kf <= nz - 3) return RegionalModel.f5(phi[qf - 3 * pl]!, phi[qf - 2 * pl]!, phi[qf - pl]!, phi[qf]!, phi[qf + pl]!, phi[qf + 2 * pl]!, vel);
    if (kf >= 2 && kf <= nz - 2) return RegionalModel.f3(phi[qf - 2 * pl]!, phi[qf - pl]!, phi[qf]!, phi[qf + pl]!, vel);
    return 0.5 * (phi[qf - pl]! + phi[qf]!);
  }

  /** Momentum advection (flux form on staggered grids), accumulated into fu, fv, fw. */
  private advectMomentum(): void {
    const { nx, ny, nz, dx, dy, dz } = this.c, sx = this.sx, pl = this.plane;
    const u = this.u, v = this.v, w = this.w, f5 = RegionalModel.f5;
    for (let k = 0; k < nz; k++) {
      const r0 = this.rho0[k]!;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const q = this.idx(i, j, k);
        // ---- u at face i-1/2: x-fluxes at centres i-1 and i
        {
          const ua = 0.5 * (u[q - 1]! + u[q]!), ub = 0.5 * (u[q]! + u[q + 1]!);
          // u values at centres i-1 (between faces i-3/2 and i-1/2) and i: interface of u-cells is the centre
          const fc0 = ua * f5(u[q - 3]!, u[q - 2]!, u[q - 1]!, u[q]!, u[q + 1]!, u[q + 2]!, ua);   // centre i-1 (between u[q-1] and u[q])
          const fc1 = ub * f5(u[q - 2]!, u[q - 1]!, u[q]!, u[q + 1]!, u[q + 2]!, u[q + 3]!, ub);   // centre i
          // y-fluxes at corners (i-1/2, j-1/2) and (i-1/2, j+1/2)
          const va = 0.5 * (v[q - 1]! + v[q]!), vb = 0.5 * (v[q - 1 + sx]! + v[q + sx]!);
          const gc0 = va * f5(u[q - 3 * sx]!, u[q - 2 * sx]!, u[q - sx]!, u[q]!, u[q + sx]!, u[q + 2 * sx]!, va);
          const gc1 = vb * f5(u[q - 2 * sx]!, u[q - sx]!, u[q]!, u[q + sx]!, u[q + 2 * sx]!, u[q + 3 * sx]!, vb);
          // z-fluxes at (i-1/2, k) and (i-1/2, k+1)
          const wa = 0.5 * (w[q - 1]! * this.rho0f[k]! + w[q]! * this.rho0f[k]!);
          const wb = 0.5 * (w[q - 1 + pl]! * this.rho0f[k + 1]! + w[q + pl]! * this.rho0f[k + 1]!);
          const hc0 = k === 0 ? 0 : wa * this.zface(u, q, k, wa);
          const hc1 = k === nz - 1 ? 0 : wb * this.zface(u, q + pl, k + 1, wb);
          const div = (ub - ua) / dx + (vb - va) / dy + (wb - wa) / (r0 * dz);
          this.fu[q] = this.fu[q]! - ((fc1 - fc0) / dx + (gc1 - gc0) / dy + (hc1 - hc0) / (r0 * dz)) + u[q]! * div;
        }
        // ---- v at face j-1/2
        {
          const ua = 0.5 * (u[q - sx]! + u[q]!), ub = 0.5 * (u[q - sx + 1]! + u[q + 1]!);
          const fc0 = ua * f5(v[q - 3]!, v[q - 2]!, v[q - 1]!, v[q]!, v[q + 1]!, v[q + 2]!, ua);
          const fc1 = ub * f5(v[q - 2]!, v[q - 1]!, v[q]!, v[q + 1]!, v[q + 2]!, v[q + 3]!, ub);
          const va = 0.5 * (v[q - sx]! + v[q]!), vb = 0.5 * (v[q]! + v[q + sx]!);
          const gc0 = va * f5(v[q - 3 * sx]!, v[q - 2 * sx]!, v[q - sx]!, v[q]!, v[q + sx]!, v[q + 2 * sx]!, va);
          const gc1 = vb * f5(v[q - 2 * sx]!, v[q - sx]!, v[q]!, v[q + sx]!, v[q + 2 * sx]!, v[q + 3 * sx]!, vb);
          const wa = 0.5 * (w[q - sx]! + w[q]!) * this.rho0f[k]!;
          const wb = 0.5 * (w[q - sx + pl]! + w[q + pl]!) * this.rho0f[k + 1]!;
          const hc0 = k === 0 ? 0 : wa * this.zface(v, q, k, wa);
          const hc1 = k === nz - 1 ? 0 : wb * this.zface(v, q + pl, k + 1, wb);
          const div = (ub - ua) / dx + (vb - va) / dy + (wb - wa) / (r0 * dz);
          this.fv[q] = this.fv[q]! - ((fc1 - fc0) / dx + (gc1 - gc0) / dy + (hc1 - hc0) / (r0 * dz)) + v[q]! * div;
        }
      }
    }
    // ---- w at levels k = 1 .. nz-1 (interior)
    for (let k = 1; k < nz; k++) {
      const rf = this.rho0f[k]!;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const q = this.idx(i, j, k);
        const ua = 0.5 * (u[q - pl]! * this.rho0[k - 1]! + u[q]! * this.rho0[k]!) / rf;
        const ub = 0.5 * (u[q - pl + 1]! * this.rho0[k - 1]! + u[q + 1]! * this.rho0[k]!) / rf;
        const fc0 = ua * f5(w[q - 3]!, w[q - 2]!, w[q - 1]!, w[q]!, w[q + 1]!, w[q + 2]!, ua);
        const fc1 = ub * f5(w[q - 2]!, w[q - 1]!, w[q]!, w[q + 1]!, w[q + 2]!, w[q + 3]!, ub);
        const va = 0.5 * (v[q - pl]! * this.rho0[k - 1]! + v[q]! * this.rho0[k]!) / rf;
        const vb = 0.5 * (v[q - pl + sx]! * this.rho0[k - 1]! + v[q + sx]! * this.rho0[k]!) / rf;
        const gc0 = va * f5(w[q - 3 * sx]!, w[q - 2 * sx]!, w[q - sx]!, w[q]!, w[q + sx]!, w[q + 2 * sx]!, va);
        const gc1 = vb * f5(w[q - 2 * sx]!, w[q - sx]!, w[q]!, w[q + sx]!, w[q + 2 * sx]!, w[q + 3 * sx]!, vb);
        // vertical fluxes at cell centres k-1 and k (rho0 * w_centre)
        const wa = 0.5 * (w[q - pl]! * this.rho0f[k - 1]! + w[q]! * rf);
        const wb = 0.5 * (w[q]! * rf + w[q + pl]! * this.rho0f[k + 1]!);
        let wva: number, wvb: number;
        if (k >= 3 && k <= nz - 3) {
          wva = f5(w[q - 3 * pl]!, w[q - 2 * pl]!, w[q - pl]!, w[q]!, w[q + pl]!, w[q + 2 * pl]!, wa);
          wvb = f5(w[q - 2 * pl]!, w[q - pl]!, w[q]!, w[q + pl]!, w[q + 2 * pl]!, w[q + 3 * pl]!, wb);
        } else {
          wva = 0.5 * (w[q - pl]! + w[q]!); wvb = 0.5 * (w[q]! + w[q + pl]!);
        }
        const div = (ub - ua) / dx + (vb - va) / dy + (wb - wa) / (rf * dz);
        this.fw[q] = this.fw[q]! - ((fc1 - fc0) / dx + (gc1 - gc0) / dy + (wb * wvb - wa * wva) / (rf * dz)) + w[q]! * div;
      }
    }
  }

  // ----------------------------------------------------------------------------------------
  // Slow tendencies (advection, Coriolis, buoyancy, damping, diffusion) from the current state

  private slowTendencies(pdDt = 0): void {
    const { nx, ny, nz, f, dz } = this.c, sx = this.sx, pl = this.plane, g = EARTH.gravity;
    for (const a of [this.fu, this.fv, this.fw, this.fth, this.fpp, ...this.fsc]) a.fill(0);
    for (const a of [this.u, this.v, this.w, this.th, this.pp, ...this.scalars]) this.fillHalo(a, nz + 1);
    this.advectMomentum();
    this.advectScalar(this.th, this.fth);
    this.advectScalar(this.pp, this.fpp);
    for (let s = 0; s < this.scalars.length; s++) this.advectScalar(this.scalars[s]!, this.fsc[s]!, pdDt > 0 && this.c.positiveDefinite !== false ? { phi0: this.sc0[s]!, dt: pdDt } : null);
    const u = this.u, v = this.v;
    // Coriolis (f-plane), buoyancy
    const buoy = this.flux;
    buoy.fill(0);
    this.computeThetaRho();
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const q = this.idx(i, j, k);
      // exact buoyancy of the pi'-form vertical equation with the full theta_rho pressure-gradient term
      const thr0 = this.th0[k]! * (1 + 0.61 * this.qv0[k]!);
      buoy[q] = g * (this.thr[q]! - thr0) / thr0;
      if (f !== 0) {
        const vAtU = 0.25 * (v[q]! + v[q - 1]! + v[q + sx]! + v[q - 1 + sx]!);
        const uAtV = 0.25 * (u[q]! + u[q + 1]! + u[q - sx]! + u[q + 1 - sx]!);
        this.fu[q] = this.fu[q]! + f * vAtU;
        this.fv[q] = this.fv[q]! - f * uAtV;
      }
    }
    if (this.buoyancy) this.buoyancy(this, buoy);
    for (let k = 1; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const q = this.idx(i, j, k);
      this.fw[q] = this.fw[q]! + 0.5 * (buoy[q]! + buoy[q - pl]!);
    }
    // Rayleigh damping layer near the lid
    const zTop = nz * dz, z0 = zTop - this.c.dampDepth;
    if (this.c.dampDepth > 0) for (let k = 0; k <= nz; k++) {
      const zc = k < nz ? this.zc[k]! : zTop;
      const zw = this.zf[k]!;
      const rc = zc > z0 ? this.c.dampRate * Math.sin(0.5 * Math.PI * (zc - z0) / this.c.dampDepth) ** 2 : 0;
      const rw = zw > z0 ? this.c.dampRate * Math.sin(0.5 * Math.PI * (zw - z0) / this.c.dampDepth) ** 2 : 0;
      if (rc === 0 && rw === 0) continue;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const q = this.idx(i, j, k);
        if (k < nz) {
          // nested runs damp toward the (3-D) boundary targets, otherwise toward the base state
          const bd = this.boundary;
          this.fu[q] = this.fu[q]! - rc * (u[q]! - (bd ? bd.u[q]! : this.ub[k]!));
          this.fv[q] = this.fv[q]! - rc * (v[q]! - (bd ? bd.v[q]! : this.vb[k]!));
          this.fth[q] = this.fth[q]! - rc * (this.th[q]! - (bd ? bd.th[q]! : this.th0[k]!));
        }
        this.fw[q] = this.fw[q]! - rw * this.w[q]!;
      }
    }
    if (this.physicsTend) this.physicsTend(this, { fu: this.fu, fv: this.fv, fw: this.fw, fth: this.fth, fsc: this.fsc });
    if (this.c.lateral === 'open' && this.boundary) this.relaxBoundaries();
    // constant eddy diffusion (tests)
    if (this.c.kdiff2 > 0) {
      this.diffuse(this.u, this.fu, nz); this.diffuse(this.v, this.fv, nz); this.diffuse(this.th, this.fth, nz, this.th0);
      this.diffuseW();
    }
  }

  /** Davies-type relaxation toward boundary targets in the outer relaxCells cells (u, v, theta, qv; w -> 0). */
  private relaxBoundaries(): void {
    const { nx, ny, nz } = this.c, nr = this.c.relaxCells ?? 5, tau = this.c.relaxTau ?? 300;
    const b = this.boundary!;
    const qv = this.scalars[0];
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const d = Math.min(i, j, nx - 1 - i, ny - 1 - j);
      if (d >= nr) continue;
      const r = (1 - d / nr) ** 2 / tau;
      for (let k = 0; k < nz; k++) {
        const q = this.idx(i, j, k);
        this.fu[q] = this.fu[q]! - r * (this.u[q]! - b.u[q]!);
        this.fv[q] = this.fv[q]! - r * (this.v[q]! - b.v[q]!);
        this.fth[q] = this.fth[q]! - r * (this.th[q]! - b.th[q]!);
        if (qv && b.qv && this.fsc[0]) this.fsc[0][q] = this.fsc[0][q]! - r * (qv[q]! - b.qv[q]!);
        if (b.pp) this.fpp[q] = this.fpp[q]! - r * (this.pp[q]! - b.pp[q]!);
        this.fw[q] = this.fw[q]! - r * this.w[q]!;
      }
    }
  }

  private diffuse(a: Float64Array, out: Float64Array, nk: number, base?: Float64Array): void {
    const { nx, ny, dx, dy, dz } = this.c, sx = this.sx, pl = this.plane, K = this.c.kdiff2;
    for (let k = 0; k < nk; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const q = this.idx(i, j, k);
      const b = base ? base[k]! : 0;
      const c = a[q]! - b;
      const up = k < nk - 1 ? a[q + pl]! - (base ? base[k + 1]! : 0) : c;
      const dn = k > 0 ? a[q - pl]! - (base ? base[k - 1]! : 0) : c;
      out[q] = out[q]! + K * ((a[q + 1]! - 2 * a[q]! + a[q - 1]!) / (dx * dx) + (a[q + sx]! - 2 * a[q]! + a[q - sx]!) / (dy * dy) + (up - 2 * c + dn) / (dz * dz));
    }
  }
  private diffuseW(): void {
    const { nx, ny, nz, dx, dy, dz } = this.c, sx = this.sx, pl = this.plane, K = this.c.kdiff2, w = this.w;
    for (let k = 1; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const q = this.idx(i, j, k);
      this.fw[q] = this.fw[q]! + K * ((w[q + 1]! - 2 * w[q]! + w[q - 1]!) / (dx * dx) + (w[q + sx]! - 2 * w[q]! + w[q - sx]!) / (dy * dy) + (w[q + pl]! - 2 * w[q]! + w[q - pl]!) / (dz * dz));
    }
  }

  // ----------------------------------------------------------------------------------------
  // Acoustic (fast) integration from the saved time-level-n state over dtStage with nSmall steps

  private acoustic(dtStage: number, nSmall: number): void {
    const { nx, ny, nz, dx, dy, dz, beta, divDamp } = this.c, sx = this.sx, pl = this.plane;
    const cp = DRY_AIR.cp, rd = DRY_AIR.rd, cv = cp - rd;
    const dts = dtStage / nSmall;
    // start from the saved state at time n
    this.u.set(this.u0); this.v.set(this.v0); this.w.set(this.w0); this.pp.set(this.pp0);
    this.ppOld.set(this.pp0);
    const u = this.u, v = this.v, w = this.w, pp = this.pp;
    const bp = 0.5 * (1 + beta), bm = 0.5 * (1 - beta);
    // column work arrays
    const a = new Float64Array(nz + 1), b = new Float64Array(nz + 1), c = new Float64Array(nz + 1), r = new Float64Array(nz + 1);
    const cfac = new Float64Array(nz);   // cs^2 / (cp rho0 thv0^2)
    const E = new Float64Array(nz), wn = new Float64Array(nz + 1);
    this.computeThetaRho();
    const thr = this.thr;
    for (let k = 0; k < nz; k++) {
      const thv = this.th0[k]! * (1 + 0.61 * this.qv0[k]!);
      const cs2 = cp / cv * rd * this.pi0[k]! * thv;
      cfac[k] = cs2 / (cp * this.rho0[k]! * thv * thv);
    }
    for (let s = 0; s < nSmall; s++) {
      this.fillHalo(pp, nz); this.fillHalo(this.ppOld, nz);
      // horizontal momentum with divergence-damped pressure (pi* = pi' + divDamp (pi' - pi'_old))
      for (let k = 0; k < nz; k++) {
        for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
          const q = this.idx(i, j, k);
          const ps = pp[q]! + divDamp * (pp[q]! - this.ppOld[q]!);
          const pw = pp[q - 1]! + divDamp * (pp[q - 1]! - this.ppOld[q - 1]!);
          const pn = pp[q - sx]! + divDamp * (pp[q - sx]! - this.ppOld[q - sx]!);
          u[q] = u[q]! + dts * (this.fu[q]! - cp * 0.5 * (thr[q]! + thr[q - 1]!) * (ps - pw) / dx);
          v[q] = v[q]! + dts * (this.fv[q]! - cp * 0.5 * (thr[q]! + thr[q - sx]!) * (ps - pn) / dy);
        }
      }
      this.fillHalo(u, nz); this.fillHalo(v, nz);
      this.ppOld.set(pp);
      // vertically implicit w - pi' solve per column
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        // explicit parts: horizontal divergence (new u, v), old w contributions
        // pi'_k^{new} = pi'_k - dts cfac_k [Dh_k + (1/(rho0 thv0)_k dz)( rt_{k+1} wbar_{k+1} - rt_k wbar_k )] + dts fpp
        // w_k^{new} = w_k + dts [fw_k - cp thv0f_k (pbar_k - pbar_{k-1})/dz],  pbar = bp pi_new + bm pi_old
        // Substitute pi_new into the w equation -> tridiagonal system in w_new (k = 1..nz-1).
        for (let k = 0; k < nz; k++) {
          const q = this.idx(i, j, k);
          const dh = (u[q + 1]! - u[q]!) / dx + (v[q + sx]! - v[q]!) / dy;
          const rt = (kk: number): number => this.rho0f[kk]! * this.th0f[kk]!;
          const rtc = this.rho0[k]! * this.th0[k]! * (1 + 0.61 * this.qv0[k]!);
          // div(rho0 thv0 V) = rho0 thv0 div_h(V) + d(rho0 thv0 w)/dz
          const oldW = bm * (rt(k + 1) * w[q + pl]! - rt(k) * w[q]!) / dz;
          E[k] = pp[q]! + dts * this.fpp[q]! - dts * cfac[k]! * (rtc * dh + oldW);
        }
        // coefficients for w_k (k=1..nz-1): pi_new_k = E_k - dts cfac_k bp (rt_{k+1} w_{k+1} - rt_k w_k)/(rtc_k dz)
        for (let k = 1; k < nz; k++) {
          const q = this.idx(i, j, k);
          const cth = cp * 0.5 * (thr[q]! + thr[q - pl]!) / dz;
          const rtk = this.rho0f[k]! * this.th0f[k]!;
          const g_k = dts * cfac[k]! * bp / dz;        // pi_new_k = E_k - g_k (rt_{k+1} w_{k+1} - rt_k w_k)
          const g_km = dts * cfac[k - 1]! * bp / dz;
          // w_k - w_k^old = dts [fw - cth ( bp (pi_new_k - pi_new_{k-1}) + bm (pi_old_k - pi_old_{k-1}) )]
          const rt = (kk: number): number => this.rho0f[kk]! * this.th0f[kk]!;
          a[k] = -dts * cth * bp * g_km * rt(k - 1) * (k - 1 >= 1 ? 1 : 0);  // coefficient of w_{k-1}
          c[k] = -dts * cth * bp * g_k * rt(k + 1) * (k + 1 <= nz - 1 ? 1 : 0);  // coefficient of w_{k+1}
          b[k] = 1 + dts * cth * bp * (g_k * rtk + g_km * rtk);
          r[k] = w[q]! + dts * (this.fw[q]! - cth * (bp * (E[k]! - E[k - 1]!) + bm * (pp[q]! - pp[q - pl]!)));
        }
        // Thomas algorithm for k = 1..nz-1 (w_0 = w_nz = 0)
        for (let k = 2; k < nz; k++) { const m = a[k]! / b[k - 1]!; b[k] = b[k]! - m * c[k - 1]!; r[k] = r[k]! - m * r[k - 1]!; }
        wn[0] = 0; wn[nz] = 0;
        if (nz > 1) wn[nz - 1] = r[nz - 1]! / b[nz - 1]!;
        for (let k = nz - 2; k >= 1; k--) wn[k] = (r[k]! - c[k]! * wn[k + 1]!) / b[k]!;
        for (let k = 0; k < nz; k++) {
          const q = this.idx(i, j, k);
          const rt = (kk: number): number => this.rho0f[kk]! * this.th0f[kk]!;
          pp[q] = E[k]! - dts * cfac[k]! * bp * (rt(k + 1) * wn[k + 1]! - rt(k) * wn[k]!) / dz;
        }
        for (let k = 0; k <= nz; k++) w[this.idx(i, j, k)] = wn[k]!;
        void c;
      }
    }
  }

  /** Advance one large step with RK3 + acoustic substeps. */
  step(): void {
    const { dt, nsound, nz } = this.c;
    this.u0.set(this.u); this.v0.set(this.v); this.w0.set(this.w); this.th0s.set(this.th); this.pp0.set(this.pp);
    for (let s = 0; s < this.scalars.length; s++) this.sc0[s]!.set(this.scalars[s]!);
    const stages: [number, number][] = [[dt / 3, Math.max(1, Math.round(nsound / 3))], [dt / 2, Math.max(1, Math.round(nsound / 2))], [dt, nsound]];
    for (const [dts, ns] of stages) {
      this.slowTendencies(dts === dt ? dt : 0);
      // theta and scalars: slow only (from time n)
      for (let i = 0; i < this.size; i++) this.th[i] = this.th0s[i]! + dts * this.fth[i]!;
      for (let s = 0; s < this.scalars.length; s++) {
        const a = this.scalars[s]!, a0 = this.sc0[s]!, fa = this.fsc[s]!;
        for (let i = 0; i < this.size; i++) a[i] = a0[i]! + dts * fa[i]!;
      }
      this.acoustic(dts, ns);
    }
    void nz;
    this.time += dt;
    this.steps++;
  }

  /** Set a horizontally uniform wind profile (also the damping-layer target). */
  setBaseWind(prof: (z: number) => { u: number; v: number }): void {
    const { nx, ny, nz } = this.c;
    for (let k = 0; k < nz; k++) {
      const w = prof(this.zc[k]!);
      this.ub[k] = w.u; this.vb[k] = w.v;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const q = this.idx(i, j, k); this.u[q] = w.u; this.v[q] = w.v; }
    }
  }

  // ----------------------------------------------------------------------------------------
  // Diagnostics
  maxAbs(a: Float64Array, nk = this.c.nz): number {
    let m = 0;
    for (let k = 0; k < nk; k++) for (let j = 0; j < this.c.ny; j++) for (let i = 0; i < this.c.nx; i++) m = Math.max(m, Math.abs(a[this.idx(i, j, k)]!));
    return m;
  }
}
