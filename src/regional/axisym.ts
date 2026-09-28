// Axisymmetric (radius-height) version of the regional model for fast tropical-cyclone experiments
// (Rotunno & Emanuel 1987; Bryan & Rotunno 2009, CM1 axisymmetric mode): the same fully compressible
// non-hydrostatic equations, numerics and physics as src/regional/core.ts + physics.ts, written in
// cylindrical coordinates with no azimuthal variation. It runs days of a hurricane in minutes on the CPU,
// for screening parameters before a 3-D run.
//
// Grid: radial C-grid. Radial wind u at faces r = i dr (i = 0..nr; u = 0 on the axis and at the outer
// wall), tangential wind v, theta, pi' and moisture at cell centres r = (i + 1/2) dr, w at levels k dz.
// Equations (f-plane):
//   du/dt = -adv(u) + v^2/r + f v - cp theta_rho dpi'/dr + D_u
//   dv/dt = -adv(v) - u v/r - f u + D_v            (v advected in r^2-weighted flux form, which gives
//                                                    the -u v / r term and conserves angular momentum)
//   dw/dt = -adv(w) - cp theta_rho dpi'/dz + B + D_w
//   dpi'/dt = -adv(pi') - cs^2/(cp rho0 thv0^2) [(1/r) d(r rho0 thv0 u)/dr + d(rho0 thv0 w)/dz]
// Time integration: RK3 with split acoustic steps and the vertically implicit w-pi' solve of core.ts.
// Advection: 5th-order upwind-biased flux form; positive-definite limiter for the moisture species.
// Physics: Smagorinsky-Lilly mixing with separate radial / vertical lengths (the cylindrical vector
// Laplacian for u and v), bulk sea-surface fluxes with a minimum wind speed, Newtonian radiative
// cooling, six-class ice microphysics (IceMicrophysics, unchanged). An outer sponge relaxes toward the
// environment; a Rayleigh layer damps the top.

import { DRY_AIR, EARTH } from '../core/constants.js';
import { RegionalModel, type RegionalConfig } from './core.js';
import { re87Wind } from './tropical.js';

export const HA = 3;   // radial halo width

export interface AxisymConfig {
  nr: number; nz: number; dr: number; dz: number; dt: number; nsound: number;
  f: number;
  beta?: number; divDamp?: number;
  dampDepth: number; dampRate: number;
  /** outer sponge width (m) and maximum relaxation rate (s^-1) */
  spongeWidth: number; spongeRate: number;
  /** radial and vertical mixing lengths (m) */
  lh: number; lv: number;
  /** sea-surface temperature (K); 0: no surface fluxes */
  sst: number;
  /** enthalpy exchange coefficient */
  ck: number;
  /** minimum wind speed in the surface fluxes (m/s, gustiness) */
  vmin: number;
  /** Newtonian cooling time scale (s, 0 = off) and maximum cooling rate (K/s) */
  radTau: number; radMax: number;
  /** constant clear-sky cooling (K/s) in the troposphere instead of the relaxation (0 = RE87 relaxation);
   *  the stratosphere (base-state T < 210 K) still relaxes toward the base state */
  radConst?: number;
}

const PR = 1 / 3;   // turbulent Prandtl number (scalars mix with K / PR)

export class AxisymModel {
  readonly a: AxisymConfig;
  /** RegionalConfig view (nx = nr, ny = 1): used by the microphysics and the saves */
  readonly c: RegionalConfig;
  readonly sx: number; readonly plane: number; readonly size: number;
  readonly zc: Float64Array; readonly zf: Float64Array;
  readonly th0: Float64Array; readonly pi0: Float64Array; readonly rho0: Float64Array; readonly qv0: Float64Array;
  readonly th0f: Float64Array; readonly rho0f: Float64Array; readonly pi0f: Float64Array;
  /** radii of cell centres and faces, indexed like the fields (i + HA) */
  readonly rc: Float64Array; readonly rf: Float64Array;
  u: Float64Array; v: Float64Array; w: Float64Array; th: Float64Array; pp: Float64Array;
  readonly scalars: Float64Array[];
  time = 0; steps = 0;
  /** surface fluxes of the last physics call (W m^-2), per radius */
  readonly shf: Float64Array; readonly lhf: Float64Array;
  private readonly u0: Float64Array; private readonly v0: Float64Array; private readonly w0: Float64Array; private readonly ths: Float64Array; private readonly pp0: Float64Array;
  private readonly sc0: Float64Array[]; private readonly fsc: Float64Array[];
  private readonly fu: Float64Array; private readonly fv: Float64Array; private readonly fw: Float64Array; private readonly fth: Float64Array; private readonly fpp: Float64Array;
  private readonly thr: Float64Array; private readonly ppOld: Float64Array; private readonly fx: Float64Array; private readonly fz: Float64Array; private readonly ratio: Float64Array;
  private readonly K: Float64Array;
  /** mass divergence at cell centres of the current stage, (1/r) d(r u)/dr + (1/rho0) d(rho0 w)/dz */
  private readonly divc: Float64Array;
  /** per radius: 1 / (r dr) and 1 / (r^2 dr) at cell centres, r and r^2 at faces (indexed i + HA) */
  private readonly irdr: Float64Array; private readonly ir2dr: Float64Array; private readonly rf2: Float64Array;
  /** Rayleigh-layer rates at centres / w levels (per level) and outer-sponge rates at centres / faces (per radius) */
  private readonly dampC: Float64Array; private readonly dampW: Float64Array; private readonly spC: Float64Array; private readonly spF: Float64Array;
  // acoustic work arrays (per column)
  private readonly wk: { a: Float64Array; b: Float64Array; c: Float64Array; r: Float64Array; cfac: Float64Array; E: Float64Array; wn: Float64Array; rt: Float64Array; rtc: Float64Array };
  private readonly pc: { fu: Float64Array; fv: Float64Array; fw: Float64Array; fth: Float64Array; fsc: Float64Array[] };

  constructor(cfg: AxisymConfig, sounding: (z: number) => { theta: number; qv: number }, nScalars = 6) {
    this.a = cfg;
    const { nr, nz, dr, dz } = cfg;
    this.c = { nx: nr, ny: 1, nz, dx: dr, dy: dr, dz, dt: cfg.dt, nsound: cfg.nsound, f: cfg.f, beta: cfg.beta ?? 0.3, divDamp: cfg.divDamp ?? 0.1, dampDepth: cfg.dampDepth, dampRate: cfg.dampRate, kdiff2: 0 };
    // base state from the 3-D model's hydrostatic integration (a one-column instance)
    const b = new RegionalModel({ ...this.c, nx: 1, ny: 1 }, sounding, 0);
    this.zc = b.zc; this.zf = b.zf; this.th0 = b.th0; this.pi0 = b.pi0; this.rho0 = b.rho0; this.qv0 = b.qv0;
    this.th0f = b.th0f; this.rho0f = b.rho0f; this.pi0f = b.pi0f;
    this.sx = nr + 2 * HA; this.plane = this.sx; this.size = this.sx * (nz + 1);
    this.rc = new Float64Array(this.sx); this.rf = new Float64Array(this.sx);
    for (let i = -HA; i < nr + HA; i++) { this.rc[i + HA] = (i + 0.5) * dr; this.rf[i + HA] = i * dr; }
    const z = (): Float64Array => new Float64Array(this.size);
    this.u = z(); this.v = z(); this.w = z(); this.th = z(); this.pp = z();
    this.u0 = z(); this.v0 = z(); this.w0 = z(); this.ths = z(); this.pp0 = z();
    this.fu = z(); this.fv = z(); this.fw = z(); this.fth = z(); this.fpp = z();
    this.thr = z(); this.ppOld = z(); this.fx = z(); this.fz = z(); this.ratio = z(); this.K = z(); this.divc = z();
    this.irdr = new Float64Array(this.sx); this.ir2dr = new Float64Array(this.sx); this.rf2 = new Float64Array(this.sx);
    for (let i = 0; i < this.sx; i++) { const r = this.rc[i]!; this.irdr[i] = 1 / (r * dr); this.ir2dr[i] = 1 / (r * r * dr); this.rf2[i] = this.rf[i]! * this.rf[i]!; }
    const w1 = (): Float64Array => new Float64Array(nz + 1);
    this.wk = { a: w1(), b: w1(), c: w1(), r: w1(), cfac: w1(), E: w1(), wn: w1(), rt: w1(), rtc: w1() };
    for (let k = 0; k <= nz; k++) this.wk.rt[k] = this.rho0f[k]! * this.th0f[k]!;
    for (let k = 0; k < nz; k++) this.wk.rtc[k] = this.rho0[k]! * this.th0[k]! * (1 + 0.61 * this.qv0[k]!);
    const zTop = nz * dz, z0 = zTop - cfg.dampDepth, rs = nr * dr - cfg.spongeWidth;
    const ray = (zz: number): number => (cfg.dampDepth > 0 && zz > z0 ? cfg.dampRate * Math.sin(0.5 * Math.PI * (zz - z0) / cfg.dampDepth) ** 2 : 0);
    const spg = (r: number): number => (cfg.spongeWidth > 0 && r > rs ? cfg.spongeRate * ((r - rs) / cfg.spongeWidth) ** 2 : 0);
    this.dampC = Float64Array.from({ length: nz + 1 }, (_, k) => ray(k < nz ? this.zc[k]! : zTop));
    this.dampW = Float64Array.from({ length: nz + 1 }, (_, k) => ray(this.zf[k]!));
    this.spC = Float64Array.from(this.rc, spg); this.spF = Float64Array.from(this.rf, spg);
    this.scalars = Array.from({ length: nScalars }, z);
    this.sc0 = Array.from({ length: nScalars }, z);
    this.fsc = Array.from({ length: nScalars }, z);
    this.pc = { fu: z(), fv: z(), fw: z(), fth: z(), fsc: this.scalars.map(z) };
    this.shf = new Float64Array(nr); this.lhf = new Float64Array(nr);
    for (let k = 0; k < nz; k++) for (let i = 0; i < nr; i++) {
      const q = this.idx(i, 0, k);
      this.th[q] = this.th0[k]!;
      if (nScalars > 0) this.scalars[0]![q] = this.qv0[k]!;
    }
  }

  idx(i: number, _j: number, k: number): number { return k * this.sx + i + HA; }

  // ---------------------------------------------------------------- boundaries

  /** Halos: mirror across the axis (scalars and w symmetric, v and u antisymmetric), zero gradient
   *  beyond the outer wall (u = 0 on the wall). kind: 's' centred symmetric, 'v' centred antisymmetric,
   *  'u' face-centred antisymmetric. */
  fillHalo(a: Float64Array, kind: 's' | 'v' | 'u', nk = this.a.nz + 1): void {
    const nr = this.a.nr;
    for (let k = 0; k < nk; k++) {
      const o = k * this.sx;
      if (kind === 'u') {
        a[o + HA] = 0; a[o + HA + nr] = 0;
        for (let h = 1; h <= HA; h++) { a[o + HA - h] = -a[o + HA + h]!; if (h < HA) a[o + HA + nr + h] = -a[o + HA + nr - h]!; }
      } else {
        const sg = kind === 'v' ? -1 : 1;
        for (let h = 0; h < HA; h++) { a[o + HA - 1 - h] = sg * a[o + HA + h]!; a[o + HA + nr + h] = a[o + HA + nr - 1]!; }
      }
    }
  }

  private halos(): void {
    const nk = this.a.nz + 1;
    this.fillHalo(this.u, 'u', nk); this.fillHalo(this.v, 'v', nk); this.fillHalo(this.w, 's', nk);
    this.fillHalo(this.th, 's', nk); this.fillHalo(this.pp, 's', nk);
    for (const s of this.scalars) this.fillHalo(s, 's', nk);
  }

  // ---------------------------------------------------------------- advection

  private static f5(a0: number, a1: number, a2: number, a3: number, a4: number, a5: number, vel: number): number {
    const c = (37 * (a2 + a3) - 8 * (a1 + a4) + (a0 + a5)) / 60, d = (10 * (a3 - a2) - 5 * (a4 - a1) + (a5 - a0)) / 60;
    return vel >= 0 ? c - d : c + d;
  }
  private static f3(a0: number, a1: number, a2: number, a3: number, vel: number): number {
    const c = (7 * (a1 + a2) - (a0 + a3)) / 12, d = (3 * (a2 - a1) - (a3 - a0)) / 12;
    return vel >= 0 ? c - d : c + d;
  }
  private zface(phi: Float64Array, qf: number, kf: number, vel: number): number {
    const nz = this.a.nz, pl = this.sx;
    if (kf >= 3 && kf <= nz - 3) return AxisymModel.f5(phi[qf - 3 * pl]!, phi[qf - 2 * pl]!, phi[qf - pl]!, phi[qf]!, phi[qf + pl]!, phi[qf + 2 * pl]!, vel);
    if (kf >= 2 && kf <= nz - 2) return AxisymModel.f3(phi[qf - 2 * pl]!, phi[qf - pl]!, phi[qf]!, phi[qf + pl]!, vel);
    return 0.5 * (phi[qf - pl]! + phi[qf]!);
  }

  /** Mass divergence at cell centres (shared by every centred field of a stage). */
  private divergence(): void {
    const { nr, nz, dz } = this.a, pl = this.sx, u = this.u, w = this.w, rf = this.rf, irdr = this.irdr, d = this.divc;
    for (let k = 0; k < nz; k++) {
      const ir0 = 1 / (this.rho0[k]! * dz), rb = this.rho0f[k]!, rt = this.rho0f[k + 1]!;
      for (let i = 0; i < nr; i++) {
        const q = k * pl + i + HA;
        d[q] = (rf[i + 1 + HA]! * u[q + 1]! - rf[i + HA]! * u[q]!) * irdr[i + HA]! + (w[q + pl]! * rt - w[q]! * rb) * ir0;
      }
    }
  }

  /** Advective tendency of a centred field (flux form, rho0-weighted, divergence-corrected). weight 1:
   *  scalars, (1/r) d(r u phi)/dr; weight 2: the tangential wind, (1/r^2) d(r^2 u v)/dr. */
  private advectCentred(phi: Float64Array, out: Float64Array, weight: 1 | 2, pd: { phi0: Float64Array; dt: number } | null): void {
    const { nr, nz, dz } = this.a, pl = this.sx, u = this.u, w = this.w, fx = this.fx, fz = this.fz;
    const gf = weight === 2 ? this.rf2 : this.rf, gi = weight === 2 ? this.ir2dr : this.irdr;
    for (let k = 0; k < nz; k++) {
      const o = k * pl + HA;
      fx[o] = 0; fx[o + nr] = 0;
      for (let i = 1; i < nr; i++) {
        const q = o + i, ul = u[q]!;
        const c = (37 * (phi[q - 1]! + phi[q]!) - 8 * (phi[q - 2]! + phi[q + 1]!) + (phi[q - 3]! + phi[q + 2]!)) / 60;
        const d = (10 * (phi[q]! - phi[q - 1]!) - 5 * (phi[q + 1]! - phi[q - 2]!) + (phi[q + 2]! - phi[q - 3]!)) / 60;
        fx[q] = gf[i + HA]! * ul * (ul >= 0 ? c - d : c + d);
      }
    }
    for (let i = 0; i < nr; i++) { fz[i + HA] = 0; fz[nz * pl + i + HA] = 0; }
    for (let k = 1; k < nz; k++) {
      const o = k * pl + HA, rf = this.rho0f[k]!, p2 = 2 * pl, p3 = 3 * pl;
      if (k >= 3 && k <= nz - 3) for (let i = 0; i < nr; i++) {
        const q = o + i, wl = w[q]!;
        const c = (37 * (phi[q - pl]! + phi[q]!) - 8 * (phi[q - p2]! + phi[q + pl]!) + (phi[q - p3]! + phi[q + p2]!)) / 60;
        const d = (10 * (phi[q]! - phi[q - pl]!) - 5 * (phi[q + pl]! - phi[q - p2]!) + (phi[q + p2]! - phi[q - p3]!)) / 60;
        fz[q] = wl * rf * (wl >= 0 ? c - d : c + d);
      } else if (k >= 2 && k <= nz - 2) for (let i = 0; i < nr; i++) {
        const q = o + i, wl = w[q]!;
        const c = (7 * (phi[q - pl]! + phi[q]!) - (phi[q - p2]! + phi[q + pl]!)) / 12, d = (3 * (phi[q]! - phi[q - pl]!) - (phi[q + pl]! - phi[q - p2]!)) / 12;
        fz[q] = wl * rf * (wl >= 0 ? c - d : c + d);
      } else for (let i = 0; i < nr; i++) { const q = o + i; fz[q] = w[q]! * rf * 0.5 * (phi[q - pl]! + phi[q]!); }
    }
    if (pd) {
      // Skamarock (2006) positive-definite limiter: scale each cell's outgoing fluxes
      const ratio = this.ratio;
      for (let k = 0; k < nz; k++) {
        const ir0 = 1 / (this.rho0[k]! * dz), o = k * pl + HA;
        for (let i = 0; i < nr; i++) {
          const q = o + i;
          const outflow = (Math.max(fx[q + 1]!, 0) - Math.min(fx[q]!, 0)) * gi[i + HA]! + (Math.max(fz[q + pl]!, 0) - Math.min(fz[q]!, 0)) * ir0;
          const avail = Math.max(pd.phi0[q]!, 0);
          ratio[q] = outflow * pd.dt > avail ? avail / (outflow * pd.dt) : 1;
        }
      }
      this.fillHalo(ratio, 's', nz);
      for (let k = 0; k < nz; k++) for (let i = 0; i <= nr; i++) { const q = k * pl + i + HA; fx[q] = fx[q]! * (fx[q]! > 0 ? ratio[q - 1]! : ratio[q]!); }
      for (let k = 1; k < nz; k++) for (let i = 0; i < nr; i++) { const q = k * pl + i + HA; fz[q] = fz[q]! * (fz[q]! > 0 ? ratio[q - pl]! : ratio[q]!); }
    }
    const dv = this.divc;
    for (let k = 0; k < nz; k++) {
      const ir0 = 1 / (this.rho0[k]! * dz), o = k * pl + HA;
      for (let i = 0; i < nr; i++) {
        const q = o + i;
        out[q] = out[q]! - ((fx[q + 1]! - fx[q]!) * gi[i + HA]! + (fz[q + pl]! - fz[q]!) * ir0) + phi[q]! * dv[q]!;
      }
    }
  }

  /** Radial wind (faces 1..nr-1) and vertical wind (levels 1..nz-1) advection. */
  private advectUW(): void {
    const { nr, nz, dr, dz } = this.a, pl = this.sx, u = this.u, w = this.w, rf = this.rf, rc = this.rc;
    // 5th-order upwind-biased interface value between a2 | a3 (inlined for speed)
    const f5 = (a0: number, a1: number, a2: number, a3: number, a4: number, a5: number, vel: number): number => {
      const c = (37 * (a2 + a3) - 8 * (a1 + a4) + (a0 + a5)) / 60, d = (10 * (a3 - a2) - 5 * (a4 - a1) + (a5 - a0)) / 60;
      return vel >= 0 ? c - d : c + d;
    };
    for (let k = 0; k < nz; k++) {
      const r0 = this.rho0[k]!;
      for (let i = 1; i < nr; i++) {
        const q = this.idx(i, 0, k), r = rf[i + HA]!, ra = rc[i - 1 + HA]!, rb = rc[i + HA]!;
        const ua = 0.5 * (u[q - 1]! + u[q]!), ub = 0.5 * (u[q]! + u[q + 1]!);
        const fc0 = ra * ua * f5(u[q - 3]!, u[q - 2]!, u[q - 1]!, u[q]!, u[q + 1]!, u[q + 2]!, ua);
        const fc1 = rb * ub * f5(u[q - 2]!, u[q - 1]!, u[q]!, u[q + 1]!, u[q + 2]!, u[q + 3]!, ub);
        const wa = 0.5 * (w[q - 1]! + w[q]!) * this.rho0f[k]!, wb = 0.5 * (w[q - 1 + pl]! + w[q + pl]!) * this.rho0f[k + 1]!;
        const hc0 = k === 0 ? 0 : wa * this.zface(u, q, k, wa);
        const hc1 = k === nz - 1 ? 0 : wb * this.zface(u, q + pl, k + 1, wb);
        const div = (rb * ub - ra * ua) / (r * dr) + (wb - wa) / (r0 * dz);
        this.fu[q] = this.fu[q]! - ((fc1 - fc0) / (r * dr) + (hc1 - hc0) / (r0 * dz)) + u[q]! * div;
      }
    }
    for (let k = 1; k < nz; k++) {
      const rfk = this.rho0f[k]!;
      for (let i = 0; i < nr; i++) {
        const q = this.idx(i, 0, k), r = rc[i + HA]!, ra = rf[i + HA]!, rb = rf[i + 1 + HA]!;
        const ua = 0.5 * (u[q - pl]! * this.rho0[k - 1]! + u[q]! * this.rho0[k]!) / rfk;
        const ub = 0.5 * (u[q - pl + 1]! * this.rho0[k - 1]! + u[q + 1]! * this.rho0[k]!) / rfk;
        const fc0 = ra * ua * f5(w[q - 3]!, w[q - 2]!, w[q - 1]!, w[q]!, w[q + 1]!, w[q + 2]!, ua);
        const fc1 = rb * ub * f5(w[q - 2]!, w[q - 1]!, w[q]!, w[q + 1]!, w[q + 2]!, w[q + 3]!, ub);
        const wa = 0.5 * (w[q - pl]! * this.rho0f[k - 1]! + w[q]! * rfk), wb = 0.5 * (w[q]! * rfk + w[q + pl]! * this.rho0f[k + 1]!);
        let wva: number, wvb: number;
        if (k >= 3 && k <= nz - 3) {
          wva = f5(w[q - 3 * pl]!, w[q - 2 * pl]!, w[q - pl]!, w[q]!, w[q + pl]!, w[q + 2 * pl]!, wa);
          wvb = f5(w[q - 2 * pl]!, w[q - pl]!, w[q]!, w[q + pl]!, w[q + 2 * pl]!, w[q + 3 * pl]!, wb);
        } else { wva = 0.5 * (w[q - pl]! + w[q]!); wvb = 0.5 * (w[q]! + w[q + pl]!); }
        const div = (rb * ub - ra * ua) / (r * dr) + (wb - wa) / (rfk * dz);
        this.fw[q] = this.fw[q]! - ((fc1 - fc0) / (r * dr) + (wb * wvb - wa * wva) / (rfk * dz)) + w[q]! * div;
      }
    }
  }

  // ---------------------------------------------------------------- slow tendencies

  private computeThetaRho(): void {
    const n = this.size, th = this.th, S = this.scalars, t = this.thr;
    if (S.length === 6) {
      const qv = S[0]!, a = S[1]!, b = S[2]!, c = S[3]!, d = S[4]!, e = S[5]!;
      for (let i = 0; i < n; i++) t[i] = th[i]! * (1 + 0.61 * qv[i]! - a[i]! - b[i]! - c[i]! - d[i]! - e[i]!);
    } else for (let i = 0; i < n; i++) { let f = 1; if (S[0]) f += 0.61 * S[0][i]!; for (let s = 1; s < S.length; s++) f -= S[s]![i]!; t[i] = th[i]! * f; }
    this.fillHalo(t, 's');
  }

  private slowTendencies(pdDt: number, stage: number): void {
    const { nr, nz, f } = this.a, pl = this.sx, g = EARTH.gravity;
    for (const a of [this.fu, this.fv, this.fw, this.fth, this.fpp, ...this.fsc]) a.fill(0);
    this.halos();
    this.advectUW();
    this.divergence();
    this.advectCentred(this.v, this.fv, 2, null);
    this.advectCentred(this.th, this.fth, 1, null);
    this.advectCentred(this.pp, this.fpp, 1, null);
    for (let s = 0; s < this.scalars.length; s++) this.advectCentred(this.scalars[s]!, this.fsc[s]!, 1, pdDt > 0 ? { phi0: this.sc0[s]!, dt: pdDt } : null);
    const u = this.u, v = this.v, rf = this.rf;
    this.computeThetaRho();
    for (let k = 0; k < nz; k++) {
      const thr0 = this.th0[k]! * (1 + 0.61 * this.qv0[k]!);
      for (let i = 0; i < nr; i++) {
        const q = this.idx(i, 0, k);
        // Coriolis and centrifugal terms (the -u v / r term comes from the r^2 flux form of v)
        this.fv[q] = this.fv[q]! - f * 0.5 * (u[q]! + u[q + 1]!);
        if (i > 0) this.fu[q] = this.fu[q]! + f * 0.5 * (v[q - 1]! + v[q]!) + 0.5 * (v[q - 1]! * v[q - 1]! + v[q]! * v[q]!) / rf[i + HA]!;
        // buoyancy (theta_rho form), onto w levels below
        this.fx[q] = g * (this.thr[q]! - thr0) / thr0;
      }
    }
    for (let k = 1; k < nz; k++) for (let i = 0; i < nr; i++) { const q = this.idx(i, 0, k); this.fw[q] = this.fw[q]! + 0.5 * (this.fx[q]! + this.fx[q - pl]!); }
    // Rayleigh layer near the lid and outer sponge (toward rest and the base state)
    const qv = this.scalars[0], fq = this.fsc[0];
    for (let k = 0; k <= nz; k++) {
      const rcz = this.dampC[k]!, rwz = this.dampW[k]!, o = k * pl + HA, th0 = k < nz ? this.th0[k]! : 0, qv0 = k < nz ? this.qv0[k]! : 0;
      for (let i = 0; i < nr; i++) {
        const q = o + i, sc = this.spC[i + HA]!, su = this.spF[i + HA]!;
        if (k < nz) {
          const rc2 = rcz + sc;
          if (rc2 > 0) { this.fv[q] = this.fv[q]! - rc2 * v[q]!; this.fth[q] = this.fth[q]! - rc2 * (this.th[q]! - th0); }
          if (rcz + su > 0) this.fu[q] = this.fu[q]! - (rcz + su) * u[q]!;
          if (sc > 0 && qv && fq) fq[q] = fq[q]! - sc * (qv[q]! - qv0);
        }
        if (rwz + sc > 0) this.fw[q] = this.fw[q]! - (rwz + sc) * this.w[q]!;
      }
    }
    // sub-grid turbulence and surface fluxes: first RK stage, reused in stages 2 and 3; radiation every stage
    if (stage === 0) this.physics();
    const t = this.pc, add = (o: Float64Array, a: Float64Array): void => { for (let i = 0; i < o.length; i++) o[i] = o[i]! + a[i]!; };
    add(this.fu, t.fu); add(this.fv, t.fv); add(this.fw, t.fw); add(this.fth, t.fth);
    for (let s = 0; s < this.fsc.length; s++) add(this.fsc[s]!, t.fsc[s]!);
    const rc = this.a.radConst ?? 0;
    if (this.a.radTau > 0 || rc > 0) for (let k = 0; k < nz; k++) {
      const trop = rc > 0 && this.th0[k]! * this.pi0[k]! > 210;
      for (let i = 0; i < nr; i++) {
        const q = this.idx(i, 0, k);
        this.fth[q] = this.fth[q]! + (trop ? -rc / this.pi0[k]! : Math.max(-(this.th[q]! - this.th0[k]!) / this.a.radTau, -this.a.radMax / this.pi0[k]!));
      }
    }
  }

  /** Smagorinsky-Lilly mixing (radial length lh, vertical length lv, Richardson-number correction) in
   *  cylindrical form, and bulk surface fluxes; tendencies into the physics cache. */
  private physics(): void {
    const { nr, nz, dr, dz, lh, lv } = this.a, pl = this.sx, u = this.u, v = this.v, w = this.w, th = this.th, rc = this.rc, rf = this.rf, g = EARTH.gravity;
    const t = this.pc;
    for (const a of [t.fu, t.fv, t.fw, t.fth, ...t.fsc]) a.fill(0);
    const K = this.K;
    for (let k = 0; k < nz; k++) for (let i = 0; i < nr; i++) {
      const q = this.idx(i, 0, k), r = rc[i + HA]!;
      const kp = Math.min(k + 1, nz - 1), km = Math.max(k - 1, 0), dzz = (kp - km) * dz || dz, qp = this.idx(i, 0, kp), qm = this.idx(i, 0, km);
      const dudr = (u[q + 1]! - u[q]!) / dr, uc = 0.5 * (u[q]! + u[q + 1]!), dwdz = (w[q + pl]! - w[q]!) / dz;
      const rdvr = r * (v[q + 1]! / rc[i + 1 + HA]! - v[q - 1]! / rc[i - 1 + HA]!) / (2 * dr);
      const dvdz = (v[qp]! - v[qm]!) / dzz;
      const dudz = 0.5 * ((u[qp]! + u[qp + 1]!) - (u[qm]! + u[qm + 1]!)) / dzz;
      const dwdr = 0.25 * ((w[q + 1]! + w[q + 1 + pl]!) - (w[q - 1]! + w[q - 1 + pl]!)) / dr;
      const S2 = 2 * (dudr * dudr + (uc / r) ** 2 + dwdz * dwdz) + (dudz + dwdr) ** 2 + rdvr * rdvr + dvdz * dvdz;
      const N2 = g * (th[qp]! - th[qm]!) / (dzz * th[q]!);
      K[q] = Math.sqrt(S2) * Math.sqrt(Math.max(0, 1 - 3 * N2 / Math.max(S2, 1e-10)));
    }
    this.fillHalo(K, 's', nz);
    const lh2 = lh * lh, lv2 = lv * lv;
    // vertical flux divergence (rho0-weighted) of a centred field
    const vert = (a: Float64Array, q: number, k: number, kf: number, base: Float64Array | null): number => {
      const ac = a[q]! - (base ? base[k]! : 0);
      let ft = 0, fb = 0;
      if (k < nz - 1) ft = 0.5 * (K[q]! + K[q + pl]!) * kf * lv2 * this.rho0f[k + 1]! * (a[q + pl]! - (base ? base[k + 1]! : 0) - ac) / dz;
      if (k > 0) fb = 0.5 * (K[q]! + K[q - pl]!) * kf * lv2 * this.rho0f[k]! * (ac - (a[q - pl]! - (base ? base[k - 1]! : 0))) / dz;
      return (ft - fb) / (this.rho0[k]! * dz);
    };
    // scalars and theta: (1/r) d/dr (r K dphi/dr) / Pr
    const scal = (a: Float64Array, out: Float64Array, base: Float64Array | null): void => {
      for (let k = 0; k < nz; k++) for (let i = 0; i < nr; i++) {
        const q = this.idx(i, 0, k), r = rc[i + HA]!;
        const kr = i < nr - 1 ? 0.5 * (K[q]! + K[q + 1]!) * lh2 / PR * rf[i + 1 + HA]! * (a[q + 1]! - a[q]!) / dr : 0;
        const kl = i > 0 ? 0.5 * (K[q]! + K[q - 1]!) * lh2 / PR * rf[i + HA]! * (a[q]! - a[q - 1]!) / dr : 0;
        out[q] = out[q]! + (kr - kl) / (r * dr) + vert(a, q, k, 1 / PR, base);
      }
    };
    scal(th, t.fth, this.th0);
    for (let s = 0; s < this.scalars.length; s++) scal(this.scalars[s]!, t.fsc[s]!, s === 0 ? this.qv0 : null);
    // tangential wind: (1/r^2) d/dr (r^3 K d(v/r)/dr) (the cylindrical vector Laplacian)
    for (let k = 0; k < nz; k++) for (let i = 0; i < nr; i++) {
      const q = this.idx(i, 0, k), r = rc[i + HA]!;
      const fl = (qq: number, ii: number): number => { const rr = rf[ii + HA]!; return 0.5 * (K[qq]! + K[qq - 1]!) * lh2 * rr * rr * rr * (v[qq]! / rc[ii + HA]! - v[qq - 1]! / rc[ii - 1 + HA]!) / dr; };
      const fr = i < nr - 1 ? fl(q + 1, i + 1) : 0, flf = i > 0 ? fl(q, i) : 0;
      t.fv[q] = t.fv[q]! + (fr - flf) / (r * r * dr) + vert(v, q, k, 1, null);
    }
    // radial wind at faces: (1/r) d/dr (r K du/dr) - K u / r^2, plus vertical mixing
    for (let k = 0; k < nz; k++) for (let i = 1; i < nr; i++) {
      const q = this.idx(i, 0, k), r = rf[i + HA]!, kf = 0.5 * (K[q]! + K[q - 1]!);
      const fr = K[q]! * lh2 * rc[i + HA]! * (u[q + 1]! - u[q]!) / dr, fl = K[q - 1]! * lh2 * rc[i - 1 + HA]! * (u[q]! - u[q - 1]!) / dr;
      let ft = 0, fb = 0;
      if (k < nz - 1) ft = 0.25 * (K[q]! + K[q - 1]! + K[q + pl]! + K[q - 1 + pl]!) * lv2 * this.rho0f[k + 1]! * (u[q + pl]! - u[q]!) / dz;
      if (k > 0) fb = 0.25 * (K[q]! + K[q - 1]! + K[q - pl]! + K[q - 1 - pl]!) * lv2 * this.rho0f[k]! * (u[q]! - u[q - pl]!) / dz;
      t.fu[q] = t.fu[q]! + (fr - fl) / (r * dr) - kf * lh2 * u[q]! / (r * r) + (ft - fb) / (this.rho0[k]! * dz);
    }
    // w at levels: (1/r) d/dr (r K dw/dr) + vertical
    for (let k = 1; k < nz; k++) for (let i = 0; i < nr; i++) {
      const q = this.idx(i, 0, k), r = rc[i + HA]!, kc = 0.5 * (K[q]! + K[q - pl]!);
      const kr = i < nr - 1 ? kc * lh2 * rf[i + 1 + HA]! * (w[q + 1]! - w[q]!) / dr : 0, kl = i > 0 ? kc * lh2 * rf[i + HA]! * (w[q]! - w[q - 1]!) / dr : 0;
      t.fw[q] = t.fw[q]! + (kr - kl) / (r * dr) + lv2 * kc * (w[q + pl]! - 2 * w[q]! + w[q - pl]!) / (dz * dz);
    }
    // bulk sea-surface fluxes at the lowest level (idealised: pi_s = 1, p_s = 1000 hPa, as the 3-D TC)
    if (this.a.sst > 0) {
      const sst = this.a.sst, esS = 611.2 * Math.exp(17.67 * (sst - 273.15) / (sst - 29.65)), qsS = 0.622 * esS / (1e5 - 0.378 * esS);
      const qv = this.scalars[0], rho1 = this.rho0[0]!, pi1 = this.pi0[0]!;
      for (let i = 0; i < nr; i++) {
        const q = this.idx(i, 0, 0), uc = 0.5 * (u[q]! + u[q + 1]!), spd = Math.max(Math.hypot(uc, v[q]!), this.a.vmin, 1);
        const cd = Math.min(2.4e-3, 1e-3 * (1 + 0.07 * spd));
        t.fv[q] = t.fv[q]! - cd * spd * v[q]! / dz;
        if (i > 0) t.fu[q] = t.fu[q]! - 0.5 * cd * spd * uc / dz;
        if (i < nr - 1) t.fu[q + 1] = t.fu[q + 1]! - 0.5 * cd * spd * uc / dz;
        const fth = this.a.ck * spd * (sst - th[q]!);
        t.fth[q] = t.fth[q]! + fth / dz;
        this.shf[i] = rho1 * DRY_AIR.cp * pi1 * fth;
        if (qv) { const fq = this.a.ck * spd * (qsS - qv[q]!); t.fsc[0]![q] = t.fsc[0]![q]! + fq / dz; this.lhf[i] = rho1 * 2.5e6 * fq; }
      }
    }
  }

  // ---------------------------------------------------------------- acoustic steps

  private acoustic(dtStage: number, nSmall: number): void {
    const { nr, nz, dr, dz } = this.a, pl = this.sx, beta = this.c.beta, divDamp = this.c.divDamp;
    const cp = DRY_AIR.cp, rd = DRY_AIR.rd, cv = cp - rd, dts = dtStage / nSmall;
    this.u.set(this.u0); this.w.set(this.w0); this.pp.set(this.pp0); this.ppOld.set(this.pp0);
    const u = this.u, w = this.w, pp = this.pp, po = this.ppOld, thr = this.thr, rf = this.rf, irdr = this.irdr, fu = this.fu, fw = this.fw, fpp = this.fpp;
    const bp = 0.5 * (1 + beta), bm = 0.5 * (1 - beta);
    const { a, b, c, r, cfac, E, wn, rt, rtc } = this.wk;
    this.computeThetaRho();
    for (let k = 0; k < nz; k++) {
      const thv = this.th0[k]! * (1 + 0.61 * this.qv0[k]!), cs2 = cp / cv * rd * this.pi0[k]! * thv;
      cfac[k] = cs2 / (cp * this.rho0[k]! * thv * thv);
    }
    const hcp = 0.5 * cp / dr;
    for (let s = 0; s < nSmall; s++) {
      this.fillHalo(pp, 's', nz); this.fillHalo(po, 's', nz);
      for (let k = 0; k < nz; k++) {
        const o = k * pl + HA;
        for (let i = 1; i < nr; i++) {
          const q = o + i;
          const ps = pp[q]! + divDamp * (pp[q]! - po[q]!), pw = pp[q - 1]! + divDamp * (pp[q - 1]! - po[q - 1]!);
          u[q] = u[q]! + dts * (fu[q]! - hcp * (thr[q]! + thr[q - 1]!) * (ps - pw));
        }
      }
      this.fillHalo(u, 'u', nz);
      po.set(pp);
      for (let i = 0; i < nr; i++) {
        const ra = rf[i + HA]!, rb = rf[i + 1 + HA]!, ird = irdr[i + HA]!;
        for (let k = 0; k < nz; k++) {
          const q = k * pl + i + HA;
          E[k] = pp[q]! + dts * fpp[q]! - dts * cfac[k]! * (rtc[k]! * (rb * u[q + 1]! - ra * u[q]!) * ird + bm * (rt[k + 1]! * w[q + pl]! - rt[k]! * w[q]!) / dz);
        }
        for (let k = 1; k < nz; k++) {
          const q = k * pl + i + HA;
          const cth = cp * 0.5 * (thr[q]! + thr[q - pl]!) / dz, g_k = dts * cfac[k]! * bp / dz, g_km = dts * cfac[k - 1]! * bp / dz, rtk = rt[k]!;
          a[k] = k - 1 >= 1 ? -dts * cth * bp * g_km * rt[k - 1]! : 0;
          c[k] = k + 1 <= nz - 1 ? -dts * cth * bp * g_k * rt[k + 1]! : 0;
          b[k] = 1 + dts * cth * bp * (g_k + g_km) * rtk;
          r[k] = w[q]! + dts * (fw[q]! - cth * (bp * (E[k]! - E[k - 1]!) + bm * (pp[q]! - pp[q - pl]!)));
        }
        for (let k = 2; k < nz; k++) { const m = a[k]! / b[k - 1]!; b[k] = b[k]! - m * c[k - 1]!; r[k] = r[k]! - m * r[k - 1]!; }
        wn[0] = 0; wn[nz] = 0;
        if (nz > 1) wn[nz - 1] = r[nz - 1]! / b[nz - 1]!;
        for (let k = nz - 2; k >= 1; k--) wn[k] = (r[k]! - c[k]! * wn[k + 1]!) / b[k]!;
        for (let k = 0; k < nz; k++) pp[k * pl + i + HA] = E[k]! - dts * cfac[k]! * bp * (rt[k + 1]! * wn[k + 1]! - rt[k]! * wn[k]!) / dz;
        for (let k = 0; k <= nz; k++) w[k * pl + i + HA] = wn[k]!;
      }
    }
  }

  /** Advance one large step (RK3 with acoustic substeps). */
  step(): void {
    const { dt, nsound } = this.a;
    this.halos();
    this.u0.set(this.u); this.v0.set(this.v); this.w0.set(this.w); this.ths.set(this.th); this.pp0.set(this.pp);
    for (let s = 0; s < this.scalars.length; s++) this.sc0[s]!.set(this.scalars[s]!);
    const stages: [number, number][] = [[dt / 3, Math.max(1, Math.round(nsound / 3))], [dt / 2, Math.max(1, Math.round(nsound / 2))], [dt, nsound]];
    for (let st = 0; st < 3; st++) {
      const [dts, ns] = stages[st]!;
      this.slowTendencies(dts === dt ? dt : 0, st);
      // v, theta and scalars have slow tendencies only (no pressure-gradient force on v)
      for (let i = 0; i < this.size; i++) { this.th[i] = this.ths[i]! + dts * this.fth[i]!; }
      for (let s = 0; s < this.scalars.length; s++) { const a = this.scalars[s]!, a0 = this.sc0[s]!, fa = this.fsc[s]!; for (let i = 0; i < this.size; i++) a[i] = a0[i]! + dts * fa[i]!; }
      this.acoustic(dts, ns);
      const v = this.v, v0 = this.v0, fv = this.fv; for (let i = 0; i < this.size; i++) v[i] = v0[i]! + dts * fv[i]!;
    }
    this.time += dt; this.steps++;
  }

  // ---------------------------------------------------------------- set-up and diagnostics

  /** Rotunno & Emanuel (1987) vortex in gradient-wind and hydrostatic balance (as insertVortex in 3-D). */
  insertVortex(vmax = 15, zTop = 20000): void {
    const { nr, nz, dr, dz, f } = this.a, CP = DRY_AIR.cp, G = EARTH.gravity;
    for (let k = 0; k < nz; k++) {
      const fac = Math.max(0, 1 - this.zc[k]! / zTop), thv = this.th0[k]! * (1 + 0.61 * this.qv0[k]!);
      // pi' from gradient-wind balance, integrated inward from the outer wall
      let pi = 0;
      for (let i = nr - 1; i >= 0; i--) {
        const r = this.rc[i + HA]!, vv = re87Wind(r, f, vmax) * fac;
        pi -= (vv * vv / r + f * vv) / (CP * thv) * dr;
        const q = this.idx(i, 0, k);
        this.v[q] = vv; this.pp[q] = pi;
      }
    }
    for (let i = 0; i < nr; i++) for (let k = 0; k < nz; k++) {
      const kp = Math.min(k + 1, nz - 1), km = Math.max(k - 1, 0), q = this.idx(i, 0, k);
      const dpdz = (this.pp[this.idx(i, 0, kp)]! - this.pp[this.idx(i, 0, km)]!) / ((kp - km) * dz);
      const thv = this.th0[k]! * (1 + 0.61 * this.qv0[k]!);
      this.th[q] = this.th0[k]! + CP * thv * thv / G * dpdz;
    }
  }

  /** Surface-pressure deficit at the centre (hPa, lowest level), maximum lowest-level wind speed and its radius. */
  metrics(): { dp: number; vmax: number; rmw: number } {
    const { nr } = this.a, P0 = DRY_AIR.pRef, ex = DRY_AIR.cp / DRY_AIR.rd;
    const q0 = this.idx(0, 0, 0), p0 = P0 * Math.pow(this.pi0[0]!, ex);
    const dp = (P0 * Math.pow(this.pi0[0]! + this.pp[q0]!, ex) - p0) / 100;
    let vmax = 0, rmw = 0;
    for (let i = 0; i < nr; i++) {
      const q = this.idx(i, 0, 0), s = Math.hypot(0.5 * (this.u[q]! + this.u[q + 1]!), this.v[q]!);
      if (s > vmax) { vmax = s; rmw = this.rc[i + HA]!; }
    }
    return { dp, vmax, rmw };
  }

  /** Total absolute angular momentum per unit length of the domain (sum of rho0 (r v + f r^2 / 2) r dr dz). */
  angularMomentum(): number {
    const { nr, nz, dr, dz, f } = this.a;
    let s = 0;
    for (let k = 0; k < nz; k++) for (let i = 0; i < nr; i++) { const r = this.rc[i + HA]!; s += this.rho0[k]! * (r * this.v[this.idx(i, 0, k)]! + 0.5 * f * r * r) * r * dr * dz; }
    return s;
  }

  maxAbs(a: Float64Array, nk = this.a.nz): number {
    let m = 0;
    for (let k = 0; k < nk; k++) for (let i = 0; i < this.a.nr; i++) m = Math.max(m, Math.abs(a[this.idx(i, 0, k)]!));
    return m;
  }
}
