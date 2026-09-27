// Global hydrostatic primitive-equation dynamical core.
//
//   * spherical-harmonic transform method, triangular truncation, Gaussian grid
//   * vorticity / divergence / temperature / ln(ps) prognostics
//   * sigma coordinate, Simmons & Burridge (1981) energy- and AAM-conserving vertical scheme
//   * semi-implicit leapfrog (Hoskins & Simmons 1975) about an isothermal reference state
//   * Robert–Asselin–Williams time filter, implicit del^8 hyperdiffusion
//
// Grid-point physics (forcing) supplies momentum and heating tendencies each step
// through the PhysicsForcing interface; no weather structure is prescribed anywhere.

import { Planet, DryAir } from '../core/constants.js';
import { SpectralTransform, SpectralField } from '../spectral/transform.js';
import { SigmaLevels, buildSigmaLevels } from './vertical.js';
import { SemiLagrangian } from './semiLagrangian.js';

/** Grid-point state handed to physics (all at the current time level). */
export interface GridState {
  nlat: number;
  nlon: number;
  K: number;
  mu: Float64Array;           // sin(lat) per latitude row
  sigma: Float64Array;        // full-level sigma
  sigmaHalf: Float64Array;
  /** u, v (m/s), T (K) per level: [k][lat][lon] flattened with stride ng */
  u: Float64Array;
  v: Float64Array;
  T: Float64Array;
  ps: Float64Array;           // Pa, [lat][lon]
}

/** Tendencies filled by physics: du/dt, dv/dt (m s^-2), dT/dt (K s^-1), same layout as GridState. */
export interface PhysicsTendencies {
  du: Float64Array;
  dv: Float64Array;
  dT: Float64Array;
}

export interface PhysicsForcing {
  compute(state: GridState, tend: PhysicsTendencies, time: number): void;
}

/** Grid-point state including water vapour, handed to sequential column physics. */
export interface MoistGridState extends GridState {
  q: Float64Array;            // specific humidity (kg/kg), [k][lat][lon]
}

/**
 * Column physics applied sequentially after each dynamics step (process splitting):
 * it receives the new state and modifies u, v, T, q in place over dt.
 * Stiff processes (boundary-layer diffusion, convection, condensation) must use this path.
 */
export interface ColumnPhysics {
  apply(state: MoistGridState, dt: number, time: number): void;
}

export interface DycoreOptions {
  trunc: number;
  sigmaHalf: Float64Array;
  dt: number;                  // s
  planet: Planet;
  air: DryAir;
  tRef?: number;               // semi-implicit reference temperature (K)
  hyperdiffTau?: number;       // e-folding time (s) of del^8 diffusion at n = T
  robert?: number;             // RAW filter strength nu
  williams?: number;           // RAW alpha (0.5 = classic Robert–Asselin)
  physics?: PhysicsForcing;
  /** Prognostic water vapour (semi-Lagrangian, grid point) */
  moist?: boolean;
  columnPhysics?: ColumnPhysics;
}

export interface SpectralState {
  vor: SpectralField[];
  div: SpectralField[];
  tmp: SpectralField[];
  lnps: SpectralField;
}

export class Dycore {
  readonly tr: SpectralTransform;
  readonly lev: SigmaLevels;
  readonly planet: Planet;
  readonly air: DryAir;
  readonly dt: number;
  readonly K: number;
  readonly ng: number;
  physics: PhysicsForcing | undefined;
  columnPhysics: ColumnPhysics | undefined;
  readonly moist: boolean;
  /** Specific humidity at the current time level (grid point), [k][lat][lon]. */
  q: Float64Array;
  private qNext: Float64Array;
  private readonly sl: SemiLagrangian | null;
  /** sigma-dot at full levels from the last dynamics evaluation (s^-1). */
  private readonly sdotFull: Float64Array;
  private readonly post: MoistGridState;
  /** accumulated water-fixer correction (kg m^-2, global mean) */
  waterFixer = 0;
  time = 0;
  steps = 0;

  cur: SpectralState;
  old: SpectralState;
  /** Surface geopotential Phi_s (m^2 s^-2), spectral. */
  readonly phis: SpectralField;
  private nxt: SpectralState;
  private tend: SpectralState;

  private readonly hyperdiff: Float64Array; // per spectral index, s^-1
  private readonly robert: number;
  private readonly williams: number;
  private readonly siCache = new Map<number, Float64Array>();

  // grid-point work arrays
  readonly grid: GridState;
  readonly phys: PhysicsTendencies;
  private readonly Ug: Float64Array; private readonly Vg: Float64Array;
  private readonly Zg: Float64Array; private readonly Dg: Float64Array;
  private readonly dpl: Float64Array; private readonly dpm: Float64Array; private readonly lpsg: Float64Array;
  private readonly Ag: Float64Array; private readonly Bg: Float64Array;
  private readonly Eg: Float64Array; private readonly TTg: Float64Array;
  private readonly UTg: Float64Array; private readonly VTg: Float64Array;
  private readonly Npg: Float64Array;
  private readonly fcor: Float64Array;
  private readonly col: { vgp: Float64Array; DG: Float64Array; C: Float64Array; sd: Float64Array };
  private readonly scratch: SpectralField;

  constructor(o: DycoreOptions) {
    this.tr = new SpectralTransform(o.trunc);
    this.planet = o.planet;
    this.air = o.air;
    this.dt = o.dt;
    this.lev = buildSigmaLevels(o.sigmaHalf, o.air, o.tRef ?? 300);
    this.K = this.lev.K;
    this.physics = o.physics;
    this.robert = o.robert ?? 0.04;
    this.williams = o.williams ?? 0.53;
    const tr = this.tr, K = this.K;
    this.ng = tr.gridSize;
    const mk = (): SpectralState => ({
      vor: Array.from({ length: K }, () => tr.newSpec()),
      div: Array.from({ length: K }, () => tr.newSpec()),
      tmp: Array.from({ length: K }, () => tr.newSpec()),
      lnps: tr.newSpec(),
    });
    this.cur = mk(); this.old = mk(); this.nxt = mk(); this.tend = mk();
    this.phis = tr.newSpec();

    const tauH = o.hyperdiffTau ?? 0.1 * 86400;
    const nMax = o.trunc * (o.trunc + 1);
    this.hyperdiff = new Float64Array(tr.nspec);
    for (let s = 0; s < tr.nspec; s++) this.hyperdiff[s] = Math.pow(tr.nn1[s]! / nMax, 4) / tauH;

    const ng = this.ng, nK = ng * K;
    this.grid = {
      nlat: tr.nlat, nlon: tr.nlon, K, mu: tr.mu, sigma: this.lev.sigma, sigmaHalf: this.lev.sigmaHalf,
      u: new Float64Array(nK), v: new Float64Array(nK), T: new Float64Array(nK), ps: new Float64Array(ng),
    };
    this.phys = { du: new Float64Array(nK), dv: new Float64Array(nK), dT: new Float64Array(nK) };
    this.Ug = new Float64Array(nK); this.Vg = new Float64Array(nK);
    this.Zg = new Float64Array(nK); this.Dg = new Float64Array(nK);
    this.Ag = new Float64Array(nK); this.Bg = new Float64Array(nK);
    this.Eg = new Float64Array(nK); this.TTg = new Float64Array(nK);
    this.UTg = new Float64Array(nK); this.VTg = new Float64Array(nK);
    this.dpl = new Float64Array(ng); this.dpm = new Float64Array(ng); this.lpsg = new Float64Array(ng);
    this.Npg = new Float64Array(ng);
    this.fcor = new Float64Array(tr.nlat);
    for (let j = 0; j < tr.nlat; j++) this.fcor[j] = 2 * o.planet.omega * tr.mu[j]!;
    this.scratch = tr.newSpec();
    this.moist = !!o.moist;
    this.columnPhysics = o.columnPhysics;
    this.q = new Float64Array(this.moist ? nK : 0);
    this.qNext = new Float64Array(this.moist ? nK : 0);
    this.sdotFull = new Float64Array(nK);
    this.sl = this.moist ? new SemiLagrangian({ nlat: tr.nlat, nlon: tr.nlon, K, lat: tr.lat, lon: tr.lon, sigma: this.lev.sigma, radius: o.planet.radius }) : null;
    this.post = {
      nlat: tr.nlat, nlon: tr.nlon, K, mu: tr.mu, sigma: this.lev.sigma, sigmaHalf: this.lev.sigmaHalf,
      u: new Float64Array(nK), v: new Float64Array(nK), T: new Float64Array(nK), ps: new Float64Array(ng), q: this.qNext,
    };
    this.col = { vgp: new Float64Array(K), DG: new Float64Array(K), C: new Float64Array(K + 1), sd: new Float64Array(K + 1) };
  }

  // ------------------------------------------------------------------
  // Initialisation helpers

  /** Set the state from grid-point fields T[k][lat][lon], ps[lat][lon] (Pa) and optional u, v. */
  setFromGrid(T: Float64Array, ps: Float64Array, u?: Float64Array, v?: Float64Array): void {
    const tr = this.tr, K = this.K, ng = this.ng, nlon = tr.nlon;
    const g = new Float64Array(ng);
    for (let i = 0; i < ng; i++) g[i] = Math.log(ps[i]!);
    tr.anal(g, this.cur.lnps);
    const A = new Float64Array(ng), B = new Float64Array(ng);
    for (let k = 0; k < K; k++) {
      tr.anal(T.subarray(k * ng, (k + 1) * ng), this.cur.tmp[k]!);
      if (u && v) {
        for (let j = 0; j < tr.nlat; j++) {
          const c = tr.coslat[j]!;
          for (let i = 0; i < nlon; i++) { const p = j * nlon + i; A[p] = u[k * ng + p]! * c; B[p] = v[k * ng + p]! * c; }
        }
        const d = this.cur.div[k]!, z = this.cur.vor[k]!;
        d.re.fill(0); d.im.fill(0); z.re.fill(0); z.im.fill(0);
        tr.analDivCurl(A, B, this.planet.radius, d, z);
      } else {
        this.cur.vor[k]!.re.fill(0); this.cur.vor[k]!.im.fill(0);
        this.cur.div[k]!.re.fill(0); this.cur.div[k]!.im.fill(0);
      }
    }
    copyState(this.cur, this.old);
    this.steps = 0;
    this.time = 0;
  }

  /** Set the surface geopotential Phi_s = g z_s (m^2 s^-2) from a [lat][lon] grid. */
  setSurfaceGeopotential(phis: Float64Array): void {
    this.tr.anal(phis, this.phis);
  }

  // ------------------------------------------------------------------
  // Time stepping

  /** Set specific humidity (kg/kg) from a [k][lat][lon] grid. */
  setMoisture(q: Float64Array): void {
    if (!this.moist) throw new Error('model was built without moisture');
    this.q.set(q);
  }

  step(): void {
    const first = this.steps === 0;
    const leap = first ? this.dt : 2 * this.dt;
    this.computeTendencies();
    this.semiImplicit(leap);
    if (this.moist) this.advectMoisture();
    if (this.columnPhysics) this.applyColumnPhysics();
    if (!first) this.timeFilter();
    if (this.moist) { const t = this.q; this.q = this.qNext; this.qNext = t; this.post.q = this.qNext; }
    // rotate buffers: old <- cur, cur <- nxt
    const o = this.old;
    this.old = this.cur;
    this.cur = this.nxt;
    this.nxt = o;
    this.steps++;
    this.time += this.dt;
  }

  /** Nonlinear (explicit) tendencies at the current time level, excluding the semi-implicit linear terms. */
  private computeTendencies(): void {
    const tr = this.tr, K = this.K, ng = this.ng, nlon = tr.nlon, nlat = tr.nlat;
    const a = this.planet.radius, R = this.air.rd, kap = this.air.kappa, Tr = this.lev.tRef;
    const lev = this.lev, cur = this.cur, grid = this.grid;

    // --- spectral -> grid
    tr.synthGrad(cur.lnps, this.dpl, this.dpm);
    tr.synth(cur.lnps, this.lpsg);
    for (let i = 0; i < ng; i++) grid.ps[i] = Math.exp(this.lpsg[i]!);
    for (let k = 0; k < K; k++) {
      const o = k * ng;
      tr.synthUV(cur.vor[k]!, cur.div[k]!, a, this.Ug.subarray(o, o + ng), this.Vg.subarray(o, o + ng));
      tr.synth(cur.vor[k]!, this.Zg.subarray(o, o + ng));
      tr.synth(cur.div[k]!, this.Dg.subarray(o, o + ng));
      tr.synth(cur.tmp[k]!, grid.T.subarray(o, o + ng));
    }
    for (let j = 0; j < nlat; j++) {
      const ic = 1 / tr.coslat[j]!;
      for (let k = 0; k < K; k++) {
        const o = k * ng + j * nlon;
        for (let i = 0; i < nlon; i++) { grid.u[o + i] = this.Ug[o + i]! * ic; grid.v[o + i] = this.Vg[o + i]! * ic; }
      }
    }

    // --- physics
    const ph = this.phys;
    ph.du.fill(0); ph.dv.fill(0); ph.dT.fill(0);
    if (this.physics) this.physics.compute(grid, ph, this.time);

    // --- column dynamics
    const { vgp, DG, C, sd } = this.col;
    const sh = lev.sigmaHalf, ds = lev.dsigma, lnR = lev.lnRatio, al = lev.alpha;
    const U = this.Ug, V = this.Vg, Z = this.Zg, D = this.Dg, T = grid.T;
    for (let j = 0; j < nlat; j++) {
      const c2 = 1 - tr.mu[j]! ** 2, coef = 1 / (a * c2), f = this.fcor[j]!, cs = tr.coslat[j]!;
      for (let i = 0; i < nlon; i++) {
        const p = j * nlon + i;
        const dl = this.dpl[p]!, dm = this.dpm[p]!;
        let np = 0;
        C[0] = 0;
        for (let k = 0; k < K; k++) {
          const q = k * ng + p;
          vgp[k] = (U[q]! * dl + V[q]! * dm) * coef;
          DG[k] = D[q]! + vgp[k]!;
          C[k + 1] = C[k]! + ds[k]! * DG[k]!;
          np -= ds[k]! * vgp[k]!;
        }
        this.Npg[p] = np;
        const Ctot = C[K]!;
        sd[0] = 0; sd[K] = 0;
        for (let k = 1; k < K; k++) sd[k] = sh[k]! * Ctot - C[k]!;
        for (let k = 0; k < K; k++) this.sdotFull[k * ng + p] = 0.5 * (sd[k]! + sd[k + 1]!);
        for (let k = 0; k < K; k++) {
          const q = k * ng + p;
          const up = k > 0 ? q - ng : q, dn = k < K - 1 ? q + ng : q;
          const i2 = 0.5 / ds[k]!;
          const sTop = sd[k]!, sBot = sd[k + 1]!;
          const vaU = (sBot * (U[dn]! - U[q]!) + sTop * (U[q]! - U[up]!)) * i2;
          const vaV = (sBot * (V[dn]! - V[q]!) + sTop * (V[q]! - V[up]!)) * i2;
          const vaT = (sBot * (T[dn]! - T[q]!) + sTop * (T[q]! - T[up]!)) * i2;
          const omp = vgp[k]! - ((k > 0 ? lnR[k]! * C[k]! : 0) + al[k]! * ds[k]! * DG[k]!) / ds[k]!;
          const Tp = T[q]! - Tr;
          const absv = Z[q]! + f;
          this.Ag[q] = absv * V[q]! - vaU - R * Tp * dl / a + ph.du[q]! * cs;
          this.Bg[q] = -absv * U[q]! - vaV - R * Tp * dm / a + ph.dv[q]! * cs;
          this.Eg[q] = 0.5 * (U[q]! * U[q]! + V[q]! * V[q]!) / c2;
          this.UTg[q] = U[q]! * Tp;
          this.VTg[q] = V[q]! * Tp;
          this.TTg[q] = Tp * D[q]! - vaT + kap * T[q]! * omp + ph.dT[q]!;
        }
      }
    }

    // --- grid -> spectral
    const td = this.tend, a2 = a * a;
    for (let k = 0; k < K; k++) {
      const o = k * ng;
      const vt = td.vor[k]!, dvt = td.div[k]!, tt = td.tmp[k]!;
      vt.re.fill(0); vt.im.fill(0); dvt.re.fill(0); dvt.im.fill(0);
      tr.analDivCurl(this.Ag.subarray(o, o + ng), this.Bg.subarray(o, o + ng), a, dvt, vt);
      tr.anal(this.Eg.subarray(o, o + ng), this.scratch);
      for (let s = 0; s < tr.nspec; s++) {
        const c = tr.nn1[s]! / a2;
        dvt.re[s] = dvt.re[s]! + c * (this.scratch.re[s]! + this.phis.re[s]!);
        dvt.im[s] = dvt.im[s]! + c * (this.scratch.im[s]! + this.phis.im[s]!);
      }
      tr.anal(this.TTg.subarray(o, o + ng), tt);
      tr.analDivCurl(this.UTg.subarray(o, o + ng), this.VTg.subarray(o, o + ng), a, tt, null, -1);
      // remove the linear part (-tau D) evaluated at the current level; it is treated implicitly
      for (let j = 0; j <= k; j++) {
        const tk = lev.tau[k * K + j]!;
        if (tk === 0) continue;
        const dj = cur.div[j]!;
        for (let s = 0; s < tr.nspec; s++) { tt.re[s] = tt.re[s]! + tk * dj.re[s]!; tt.im[s] = tt.im[s]! + tk * dj.im[s]!; }
      }
    }
    tr.anal(this.Npg, td.lnps);
  }

  /** Semi-implicit matrices M(n)^{-1} for half-leap h: M = I + h^2 c (G tau + R Tr 1 nu^T). */
  private siMatrices(h: number): Float64Array {
    const cached = this.siCache.get(h);
    if (cached) return cached;
    const K = this.K, T = this.tr.trunc, lev = this.lev, a2 = this.planet.radius ** 2;
    const RT = this.air.rd * lev.tRef;
    const out = new Float64Array((T + 1) * K * K);
    const M = new Float64Array(K * K);
    for (let n = 0; n <= T; n++) {
      const c = n * (n + 1) / a2;
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
        let gt = 0;
        for (let l = 0; l < K; l++) gt += lev.G[i * K + l]! * lev.tau[l * K + j]!;
        M[i * K + j] = (i === j ? 1 : 0) + h * h * c * (gt + RT * lev.dsigma[j]!);
      }
      out.set(invert(M, K), n * K * K);
    }
    this.siCache.set(h, out);
    return out;
  }

  private semiImplicit(leap: number): void {
    const tr = this.tr, K = this.K, lev = this.lev, a2 = this.planet.radius ** 2;
    const h = leap / 2, RT = this.air.rd * lev.tRef;
    const Minv = this.siMatrices(h);
    const old = this.old, td = this.tend, nx = this.nxt;
    const rhs = new Float64Array(K), rhsI = new Float64Array(K);
    const Tst = new Float64Array(K), TstI = new Float64Array(K);
    const Db = new Float64Array(K), DbI = new Float64Array(K);
    for (let s = 0; s < tr.nspec; s++) {
      const n = tr.nOf[s]!, c = tr.nn1[s]! / a2;
      const Pst = old.lnps.re[s]! + h * td.lnps.re[s]!;
      const PstI = old.lnps.im[s]! + h * td.lnps.im[s]!;
      for (let k = 0; k < K; k++) {
        Tst[k] = old.tmp[k]!.re[s]! + h * td.tmp[k]!.re[s]!;
        TstI[k] = old.tmp[k]!.im[s]! + h * td.tmp[k]!.im[s]!;
      }
      for (let k = 0; k < K; k++) {
        let gT = 0, gTI = 0;
        for (let j = k; j < K; j++) { const g = lev.G[k * K + j]!; gT += g * Tst[j]!; gTI += g * TstI[j]!; }
        rhs[k] = old.div[k]!.re[s]! + h * td.div[k]!.re[s]! + h * c * (gT + RT * Pst);
        rhsI[k] = old.div[k]!.im[s]! + h * td.div[k]!.im[s]! + h * c * (gTI + RT * PstI);
      }
      const mo = n * K * K;
      for (let i = 0; i < K; i++) {
        let x = 0, y = 0;
        for (let j = 0; j < K; j++) { const m = Minv[mo + i * K + j]!; x += m * rhs[j]!; y += m * rhsI[j]!; }
        Db[i] = x; DbI[i] = y;
      }
      let nuD = 0, nuDI = 0;
      for (let k = 0; k < K; k++) { nuD += lev.dsigma[k]! * Db[k]!; nuDI += lev.dsigma[k]! * DbI[k]!; }
      nx.lnps.re[s] = 2 * (Pst - h * nuD) - old.lnps.re[s]!;
      nx.lnps.im[s] = 2 * (PstI - h * nuDI) - old.lnps.im[s]!;
      const kd = this.hyperdiff[s]!, damp = 1 / (1 + leap * kd);
      for (let k = 0; k < K; k++) {
        let tD = 0, tDI = 0;
        for (let j = 0; j <= k; j++) { const t = lev.tau[k * K + j]!; tD += t * Db[j]!; tDI += t * DbI[j]!; }
        nx.tmp[k]!.re[s] = (2 * (Tst[k]! - h * tD) - old.tmp[k]!.re[s]!) * (n === 0 ? 1 : damp);
        nx.tmp[k]!.im[s] = (2 * (TstI[k]! - h * tDI) - old.tmp[k]!.im[s]!) * (n === 0 ? 1 : damp);
        nx.div[k]!.re[s] = (2 * Db[k]! - old.div[k]!.re[s]!) * damp;
        nx.div[k]!.im[s] = (2 * DbI[k]! - old.div[k]!.im[s]!) * damp;
        nx.vor[k]!.re[s] = (old.vor[k]!.re[s]! + leap * td.vor[k]!.re[s]!) * damp;
        nx.vor[k]!.im[s] = (old.vor[k]!.im[s]! + leap * td.vor[k]!.im[s]!) * damp;
      }
    }
  }

  /** Column-integrated water (kg m^-2, global mean) of q with surface pressure field ps. */
  private waterPath(q: Float64Array, ps: Float64Array): number {
    const tr = this.tr, ng = this.ng, nlon = tr.nlon, ds = this.lev.dsigma;
    let W = 0;
    for (let j = 0; j < tr.nlat; j++) {
      let r = 0;
      for (let i = 0; i < nlon; i++) {
        const p = j * nlon + i;
        let col = 0;
        for (let k = 0; k < this.K; k++) col += q[k * ng + p]! * ds[k]!;
        r += col * ps[p]!;
      }
      W += tr.weight[j]! * r / nlon;
    }
    return W / 2 / this.planet.gravity;
  }

  /** Global-mean column water vapour (kg m^-2) of the current state. */
  totalWater(): number {
    if (!this.moist) return 0;
    const g = new Float64Array(this.ng);
    this.tr.synth(this.cur.lnps, g);
    for (let i = 0; i < g.length; i++) g[i] = Math.exp(g[i]!);
    return this.waterPath(this.q, g);
  }

  /** Semi-Lagrangian transport of q from t to t+dt with the winds of the current level, plus mass fixer. */
  private advectMoisture(): void {
    const g = this.grid;
    const W0 = this.waterPath(this.q, g.ps);
    this.sl!.advect(g.u, g.v, this.sdotFull, this.dt, [this.q], [this.qNext]);
    const psNew = this.post.ps;
    this.tr.synth(this.nxt.lnps, psNew);
    for (let i = 0; i < psNew.length; i++) psNew[i] = Math.exp(psNew[i]!);
    const W1 = this.waterPath(this.qNext, psNew);
    if (W1 > 0) {
      const f = W0 / W1;
      for (let i = 0; i < this.qNext.length; i++) this.qNext[i] = this.qNext[i]! * f;
      this.waterFixer += W0 - W1;
    }
  }

  /** Sequential column physics on the new state; increments are transformed back to spectral space. */
  private applyColumnPhysics(): void {
    const tr = this.tr, K = this.K, ng = this.ng, nlon = tr.nlon, a = this.planet.radius;
    const P = this.post, nx = this.nxt;
    tr.synth(nx.lnps, P.ps);
    for (let i = 0; i < ng; i++) P.ps[i] = Math.exp(P.ps[i]!);
    for (let k = 0; k < K; k++) {
      const o = k * ng;
      tr.synthUV(nx.vor[k]!, nx.div[k]!, a, P.u.subarray(o, o + ng), P.v.subarray(o, o + ng));
      tr.synth(nx.tmp[k]!, P.T.subarray(o, o + ng));
      for (let j = 0; j < tr.nlat; j++) {
        const ic = 1 / tr.coslat[j]!;
        for (let i = 0; i < nlon; i++) { const q = o + j * nlon + i; P.u[q] = P.u[q]! * ic; P.v[q] = P.v[q]! * ic; }
      }
    }
    const u0 = Float64Array.from(P.u), v0 = Float64Array.from(P.v), T0 = Float64Array.from(P.T);
    if (!this.moist) P.q = new Float64Array(0);
    this.columnPhysics!.apply(P, this.dt, this.time + this.dt);
    const A = this.Ag, B = this.Bg, dT = this.TTg;
    for (let k = 0; k < K; k++) {
      const o = k * ng;
      for (let j = 0; j < tr.nlat; j++) {
        const c = tr.coslat[j]!;
        for (let i = 0; i < nlon; i++) {
          const q = o + j * nlon + i;
          A[q] = (P.u[q]! - u0[q]!) * c; B[q] = (P.v[q]! - v0[q]!) * c; dT[q] = P.T[q]! - T0[q]!;
        }
      }
      tr.analDivCurl(A.subarray(o, o + ng), B.subarray(o, o + ng), a, nx.div[k]!, nx.vor[k]!);
      tr.anal(dT.subarray(o, o + ng), nx.tmp[k]!, 1, true);
    }
  }

  /** Robert–Asselin–Williams filter on the current level. */
  private timeFilter(): void {
    const nu = this.robert, al = this.williams;
    const f = (o: SpectralField, c: SpectralField, n: SpectralField): void => {
      for (let s = 0; s < c.re.length; s++) {
        const d = 0.5 * nu * (o.re[s]! - 2 * c.re[s]! + n.re[s]!);
        c.re[s] = c.re[s]! + al * d; n.re[s] = n.re[s]! - (1 - al) * d;
        const e = 0.5 * nu * (o.im[s]! - 2 * c.im[s]! + n.im[s]!);
        c.im[s] = c.im[s]! + al * e; n.im[s] = n.im[s]! - (1 - al) * e;
      }
    };
    for (let k = 0; k < this.K; k++) {
      f(this.old.vor[k]!, this.cur.vor[k]!, this.nxt.vor[k]!);
      f(this.old.div[k]!, this.cur.div[k]!, this.nxt.div[k]!);
      f(this.old.tmp[k]!, this.cur.tmp[k]!, this.nxt.tmp[k]!);
    }
    f(this.old.lnps, this.cur.lnps, this.nxt.lnps);
  }

  // ------------------------------------------------------------------
  // Diagnostics

  /** Synthesise u, v, T, ps of the current state into this.grid (no physics call). */
  refreshGrid(): GridState {
    const tr = this.tr, K = this.K, ng = this.ng, nlon = tr.nlon, g = this.grid;
    tr.synth(this.cur.lnps, this.lpsg);
    for (let i = 0; i < ng; i++) g.ps[i] = Math.exp(this.lpsg[i]!);
    for (let k = 0; k < K; k++) {
      const o = k * ng;
      tr.synthUV(this.cur.vor[k]!, this.cur.div[k]!, this.planet.radius, g.u.subarray(o, o + ng), g.v.subarray(o, o + ng));
      tr.synth(this.cur.tmp[k]!, g.T.subarray(o, o + ng));
      for (let j = 0; j < tr.nlat; j++) {
        const ic = 1 / tr.coslat[j]!;
        for (let i = 0; i < nlon; i++) { g.u[o + j * nlon + i] = g.u[o + j * nlon + i]! * ic; g.v[o + j * nlon + i] = g.v[o + j * nlon + i]! * ic; }
      }
    }
    return g;
  }

  /** Relative vorticity (s^-1) of level k on the grid. */
  vorticityGrid(k: number, out: Float64Array): void { this.tr.synth(this.cur.vor[k]!, out); }
  /** Divergence (s^-1) of level k on the grid. */
  divergenceGrid(k: number, out: Float64Array): void { this.tr.synth(this.cur.div[k]!, out); }

  /** Global-mean surface pressure (Pa) = dry air mass * g / area. */
  meanSurfacePressure(): number {
    const tr = this.tr, g = new Float64Array(this.ng);
    tr.synth(this.cur.lnps, g);
    let s = 0;
    for (let j = 0; j < tr.nlat; j++) {
      let r = 0;
      for (let i = 0; i < tr.nlon; i++) r += Math.exp(g[j * tr.nlon + i]!);
      s += tr.weight[j]! * r / tr.nlon;
    }
    return s / 2;
  }

  isFinite(): boolean {
    for (let k = 0; k < this.K; k++) {
      const t = this.cur.tmp[k]!.re, v = this.cur.vor[k]!.re;
      for (let s = 0; s < t.length; s++) if (!Number.isFinite(t[s]!) || !Number.isFinite(v[s]!)) return false;
    }
    return true;
  }

  // ------------------------------------------------------------------
  // Serialisation (for checkpoints / worker transfer)

  exportState(): { time: number; steps: number; data: Float64Array } {
    const parts: Float64Array[] = [];
    for (const st of [this.old, this.cur]) {
      for (const arr of [st.vor, st.div, st.tmp]) for (const f of arr) { parts.push(f.re, f.im); }
      parts.push(st.lnps.re, st.lnps.im);
    }
    if (this.moist) parts.push(this.q);
    const len = parts.reduce((a, p) => a + p.length, 0);
    const data = new Float64Array(len);
    let o = 0;
    for (const p of parts) { data.set(p, o); o += p.length; }
    return { time: this.time, steps: this.steps, data };
  }

  importState(s: { time: number; steps: number; data: Float64Array }): void {
    let o = 0;
    for (const st of [this.old, this.cur]) {
      for (const arr of [st.vor, st.div, st.tmp]) for (const f of arr) {
        f.re.set(s.data.subarray(o, o + f.re.length)); o += f.re.length;
        f.im.set(s.data.subarray(o, o + f.im.length)); o += f.im.length;
      }
      st.lnps.re.set(s.data.subarray(o, o + st.lnps.re.length)); o += st.lnps.re.length;
      st.lnps.im.set(s.data.subarray(o, o + st.lnps.im.length)); o += st.lnps.im.length;
    }
    if (this.moist) { this.q.set(s.data.subarray(o, o + this.q.length)); o += this.q.length; }
    if (o !== s.data.length) throw new Error('checkpoint size mismatch');
    this.time = s.time;
    this.steps = s.steps;
  }
}

function copyState(a: SpectralState, b: SpectralState): void {
  const cp = (x: SpectralField, y: SpectralField): void => { y.re.set(x.re); y.im.set(x.im); };
  for (let k = 0; k < a.vor.length; k++) { cp(a.vor[k]!, b.vor[k]!); cp(a.div[k]!, b.div[k]!); cp(a.tmp[k]!, b.tmp[k]!); }
  cp(a.lnps, b.lnps);
}

/** Gauss–Jordan inverse with partial pivoting (small dense K x K). */
function invert(A: Float64Array, K: number): Float64Array {
  const a = Float64Array.from(A), inv = new Float64Array(K * K);
  for (let i = 0; i < K; i++) inv[i * K + i] = 1;
  for (let c = 0; c < K; c++) {
    let piv = c, best = Math.abs(a[c * K + c]!);
    for (let r = c + 1; r < K; r++) { const v = Math.abs(a[r * K + c]!); if (v > best) { best = v; piv = r; } }
    if (piv !== c) for (let j = 0; j < K; j++) {
      let t = a[c * K + j]!; a[c * K + j] = a[piv * K + j]!; a[piv * K + j] = t;
      t = inv[c * K + j]!; inv[c * K + j] = inv[piv * K + j]!; inv[piv * K + j] = t;
    }
    const d = 1 / a[c * K + c]!;
    for (let j = 0; j < K; j++) { a[c * K + j] = a[c * K + j]! * d; inv[c * K + j] = inv[c * K + j]! * d; }
    for (let r = 0; r < K; r++) {
      if (r === c) continue;
      const m = a[r * K + c]!;
      if (m === 0) continue;
      for (let j = 0; j < K; j++) { a[r * K + j] = a[r * K + j]! - m * a[c * K + j]!; inv[r * K + j] = inv[r * K + j]! - m * inv[c * K + j]!; }
    }
  }
  return inv;
}
