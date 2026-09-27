// Gray-radiation moist aquaplanet physics (Frierson, Held & Zurita-Gotor 2006; Frierson 2007),
// configured as the Isca "frierson" test case. Applied column by column after each dynamics step.
//
//   1. two-stream gray longwave radiation, shortwave absorbed weakly in the atmosphere and at the surface
//   2. bulk surface fluxes (simplified Monin–Obukhov) + implicit K-profile boundary-layer diffusion
//      of u, v, dry static energy and q, with a prognostic boundary-layer depth (bulk Ri = 1)
//   3. slab mixed-layer ocean
//   4. Rayleigh sponge above 50 hPa (energy conserving)
//   5. Simplified Betts–Miller convection
//   6. large-scale condensation with re-evaporation of falling rain
// All fluxes are in SI units and every process conserves water and energy within the column.

import { DAY, DryAir, Planet } from '../../core/constants.js';
import { ColumnPhysics, MoistGridState } from '../dycore.js';
import { MOIST, dqsatdT, qsat } from './thermo.js';
import { sbmColumn, sbmWork, SbmWork } from './sbm.js';

export const AQUA = {
  solarConstant: 1360,     // W m^-2
  delSol: 1.4,             // pole-equator insolation contrast
  albedo: 0.31,
  atmAbs: 0.2,             // shortwave optical depth at the surface
  tauEq: 6.0,              // longwave optical depth at the surface, equator
  tauPole: 1.5,
  linearTau: 0.1,
  mixedLayerDepth: 2.5,    // m
  roughness: 3.21e-5,      // m
  richCrit: 1.0,
  fracInner: 0.1,
  spongePBottom: 5000,     // Pa
  spongeTau: 0.25 * DAY,
  useConvection: true,
};

export interface AquaplanetFields {
  sst: Float64Array;
  /** running accumulations since the last reset (per grid point) */
  precipConv: Float64Array;   // kg m^-2
  precipLS: Float64Array;
  evap: Float64Array;
  shf: Float64Array;          // J m^-2
  olr: Float64Array;          // J m^-2
  accTime: number;            // s
  /** instantaneous */
  precipRate: Float64Array;   // kg m^-2 s^-1 (last step)
  olrNow: Float64Array;       // W m^-2
  blDepth: Float64Array;      // m
}

export class GrayAquaplanet implements ColumnPhysics {
  readonly f: AquaplanetFields;
  private readonly K: number;
  private readonly col: Record<string, Float64Array>;
  private readonly work: SbmWork;
  /** clipped negative water (kg m^-2, global accumulation proxy) */
  negativeWater = 0;

  constructor(private readonly planet: Planet, private readonly air: DryAir, nlat: number, nlon: number, K: number,
              readonly lat: Float64Array) {
    const ng = nlat * nlon;
    this.K = K;
    const z = (): Float64Array => new Float64Array(ng);
    this.f = { sst: z(), precipConv: z(), precipLS: z(), evap: z(), shf: z(), olr: z(), accTime: 0, precipRate: z(), olrNow: z(), blDepth: z() };
    const c = (n: number): Float64Array => new Float64Array(n);
    this.col = {
      T: c(K), q: c(K), u: c(K), v: c(K), pf: c(K), ph: c(K + 1), zf: c(K), zh: c(K + 1),
      lwu: c(K + 1), lwd: c(K + 1), tr: c(K), s: c(K), a: c(K + 1), m: c(K), x: c(K),
      cc: c(K), dd: c(K), kk: c(K + 1), sw: c(K + 1),
    };
    this.work = sbmWork(K);
    for (let j = 0; j < nlat; j++) {
      const s = Math.sin(lat[j]!);
      for (let i = 0; i < nlon; i++) this.f.sst[j * nlon + i] = 270 + 35 * (1 - s * s);
    }
  }

  resetAccumulators(): void {
    for (const a of [this.f.precipConv, this.f.precipLS, this.f.evap, this.f.shf, this.f.olr]) a.fill(0);
    this.f.accTime = 0;
  }

  apply(st: MoistGridState, dt: number): void {
    const { nlat, nlon, K } = st, ng = nlat * nlon;
    const C = this.col;
    for (let j = 0; j < nlat; j++) {
      const sl = Math.sin(this.lat[j]!);
      for (let i = 0; i < nlon; i++) {
        const p = j * nlon + i;
        for (let k = 0; k < K; k++) {
          const q = k * ng + p;
          C.T![k] = st.T[q]!; C.q![k] = st.q[q]!; C.u![k] = st.u[q]!; C.v![k] = st.v[q]!;
        }
        this.column(p, sl, st.ps[p]!, st.sigma, st.sigmaHalf, dt);
        for (let k = 0; k < K; k++) {
          const q = k * ng + p;
          st.T[q] = C.T![k]!; st.q[q] = C.q![k]!; st.u[q] = C.u![k]!; st.v[q] = C.v![k]!;
        }
      }
    }
    this.f.accTime += dt;
  }

  private column(p: number, sinLat: number, ps: number, sigma: Float64Array, sigmaHalf: Float64Array, dt: number): void {
    const K = this.K, C = this.col, g = this.planet.gravity, cp = this.air.cp, R = this.air.rd;
    const T = C.T!, q = C.q!, u = C.u!, v = C.v!, pf = C.pf!, ph = C.ph!, zf = C.zf!, zh = C.zh!;
    const f = this.f;
    for (let k = 0; k <= K; k++) ph[k] = sigmaHalf[k]! * ps;
    for (let k = 0; k < K; k++) pf[k] = sigma[k]! * ps;
    // heights (Simmons–Burridge hydrostatics, consistent with the dynamical core)
    zh[K] = 0;
    for (let k = K - 1; k >= 0; k--) {
      const lnr = k === 0 ? Math.LN2 * 2 : Math.log(ph[k + 1]! / ph[k]!);
      const alpha = k === 0 ? Math.LN2 : 1 - ph[k]! / (ph[k + 1]! - ph[k]!) * lnr;
      zf[k] = zh[k + 1]! + alpha * R * T[k]! / g;
      zh[k] = zh[k + 1]! + R * T[k]! * lnr / g;
    }

    // ---------------- 1. radiation
    const sst = f.sst[p]!;
    const tau0 = AQUA.tauEq + (AQUA.tauPole - AQUA.tauEq) * sinLat * sinLat;
    const lwu = C.lwu!, lwd = C.lwd!, tr = C.tr!;
    const tauAt = (pp: number): number => { const x = pp / 1e5; return tau0 * (AQUA.linearTau * x + (1 - AQUA.linearTau) * x * x * x * x); };
    for (let k = 0; k < K; k++) tr[k] = Math.exp(-(tauAt(ph[k + 1]!) - tauAt(ph[k]!)));
    lwd[0] = 0;
    for (let k = 0; k < K; k++) { const B = MOIST.stefan * T[k]! ** 4; lwd[k + 1] = lwd[k]! * tr[k]! + B * (1 - tr[k]!); }
    lwu[K] = MOIST.stefan * sst ** 4;
    for (let k = K - 1; k >= 0; k--) { const B = MOIST.stefan * T[k]! ** 4; lwu[k] = lwu[k + 1]! * tr[k]! + B * (1 - tr[k]!); }
    const insol = 0.25 * AQUA.solarConstant * (1 + AQUA.delSol * (1 - 3 * sinLat * sinLat) / 4);
    const sw = C.sw!;
    for (let k = 0; k <= K; k++) { const x = ph[k]! / 1e5; sw[k] = insol * Math.exp(-AQUA.atmAbs * x * x * x * x); }
    const swSfc = sw[K]!, swUp = AQUA.albedo * swSfc;
    for (let k = 0; k < K; k++) {
      const Ftop = lwu[k]! - lwd[k]! + swUp - sw[k]!;
      const Fbot = lwu[k + 1]! - lwd[k + 1]! + swUp - sw[k + 1]!;
      T[k] = T[k]! + dt * g * (Fbot - Ftop) / (cp * (ph[k + 1]! - ph[k]!));
    }
    const olr = lwu[0]!;

    // ---------------- 2. surface fluxes + boundary-layer diffusion (implicit)
    const ka = K - 1, za = zf[ka]!;
    const speed = Math.max(Math.hypot(u[ka]!, v[ka]!), 1e-3);
    const rhoA = pf[ka]! / (R * T[ka]!);
    const thetaDiff = T[ka]! + g * za / cp - sst;
    const ri = g * za * thetaDiff / (sst * speed * speed);
    const lnz = Math.log(za / AQUA.roughness);
    const cn = (MOIST.vonKarman / lnz) ** 2;
    const rc = AQUA.richCrit;
    const cd = ri <= 0 ? cn : ri < rc ? cn * (1 - ri / rc) ** 2 : 0;
    const ustar = Math.sqrt(cd) * speed;
    // boundary-layer depth: bulk Richardson number relative to the lowest level reaches Ri_c
    const svBot = T[ka]! + g * za / cp;
    let h = za, rich1 = 0, h1 = za;
    let found = false;
    for (let k = ka - 1; k >= 1; k--) {
      const sv = T[k]! + g * zf[k]! / cp;
      const rich = zf[k]! * g * (sv - svBot) / svBot / (u[k]! ** 2 + v[k]! ** 2 + 1e-10);
      if (rich > rc) { h = zf[k]! + (h1 - zf[k]!) * (rich - rc) / (rich - rich1); found = true; break; }
      rich1 = rich; h1 = zf[k]!;
    }
    if (!found) h = zf[1]!;
    f.blDepth[p] = h;
    const kmo = (z: number): number => {
      if (ri <= 0) return MOIST.vonKarman * ustar * z;
      if (ri >= rc) return 0;
      const x = ri / rc;
      return MOIST.vonKarman * ustar * z / (1 + x * Math.log(z / AQUA.roughness) / (1 - x));
    };
    const hIn = AQUA.fracInner * h, kRef = kmo(hIn);
    const a = C.a!, m = C.m!, kk = C.kk!;
    a[0] = 0; kk[0] = 0;
    for (let i = 1; i < K; i++) {
      const z = zh[i]!;
      let Kd = 0;
      if (z < hIn) Kd = kmo(z);
      else if (z < h) Kd = kRef * (z / hIn) * (1 - (z - hIn) / (h - hIn)) ** 2;
      kk[i] = Kd;
      const Th = 0.5 * (T[i - 1]! + T[i]!);
      const rho = ph[i]! / (R * Th);
      a[i] = rho * Kd / (zf[i - 1]! - zf[i]!);
    }
    a[K] = rhoA * cd * speed;
    for (let k = 0; k < K; k++) m[k] = (ph[k + 1]! - ph[k]!) / g;
    const s = C.s!;
    for (let k = 0; k < K; k++) s[k] = cp * T[k]! + g * zf[k]!;
    const qs = qsat(sst, ps);
    const fluxS = this.diffuse(s, cp * sst, dt);
    const fluxQ = this.diffuse(q, qs, dt);
    this.diffuse(u, 0, dt);
    this.diffuse(v, 0, dt);
    for (let k = 0; k < K; k++) T[k] = (s[k]! - g * zf[k]!) / cp;
    const lh = MOIST.Lv * fluxQ;
    // slab ocean
    const heatCap = MOIST.rhoWater * MOIST.cpWater * AQUA.mixedLayerDepth;
    f.sst[p] = sst + dt * (swSfc * (1 - AQUA.albedo) + lwd[K]! - MOIST.stefan * sst ** 4 - fluxS - lh) / heatCap;

    // ---------------- 4. sponge above 50 hPa (kinetic energy lost is returned as heat)
    for (let k = 0; k < K; k++) {
      if (pf[k]! >= AQUA.spongePBottom) break;
      const x = (AQUA.spongePBottom - pf[k]!) / AQUA.spongePBottom;
      const r = x * x / AQUA.spongeTau;
      const un = u[k]! / (1 + dt * r), vn = v[k]! / (1 + dt * r);
      T[k] = T[k]! + 0.5 * (u[k]! ** 2 + v[k]! ** 2 - un * un - vn * vn) / cp;
      u[k] = un; v[k] = vn;
    }

    // ---------------- 5. convection
    let rainConv = 0;
    if (AQUA.useConvection) rainConv = sbmColumn(T, q, pf, ph, dt, g, this.work).rain;

    // ---------------- 6. large-scale condensation with re-evaporation
    const hlcp = MOIST.Lv / cp;
    let exq = 0, rainLS = 0;
    for (let k = 0; k < K; k++) {
      const mass = (ph[k + 1]! - ph[k]!) / g;
      const qsk = qsat(T[k]!, pf[k]!), dqs = dqsatdT(T[k]!, pf[k]!);
      if (q[k]! > qsk) {
        const dq = (qsk - q[k]!) / (1 + hlcp * dqs);
        q[k] = q[k]! + dq;
        T[k] = T[k]! - hlcp * dq;
        exq -= dq * mass;
      } else if (exq > 0) {
        const def = Math.min(Math.max((qsk - q[k]!) / (1 + hlcp * dqs), 0), exq / mass);
        q[k] = q[k]! + def;
        T[k] = T[k]! - hlcp * def;
        exq -= def * mass;
      }
    }
    rainLS = Math.max(0, exq);
    for (let k = 0; k < K; k++) if (q[k]! < 0) { this.negativeWater -= q[k]! * (ph[k + 1]! - ph[k]!) / g; q[k] = 0; }

    f.precipConv[p] = f.precipConv[p]! + rainConv;
    f.precipLS[p] = f.precipLS[p]! + rainLS;
    f.evap[p] = f.evap[p]! + fluxQ * dt;
    f.shf[p] = f.shf[p]! + fluxS * dt;
    f.olr[p] = f.olr[p]! + olr * dt;
    f.olrNow[p] = olr;
    f.precipRate[p] = (rainConv + rainLS) / dt;
  }

  /**
   * Backward-Euler vertical diffusion of X with conductances a (kg m^-2 s^-1) at half levels,
   * layer masses m, and a surface node of fixed value xs coupled through a[K].
   * Returns the upward surface flux a[K] (xs - X_bottom) (X-units kg m^-2 s^-1).
   */
  private diffuse(X: Float64Array, xs: number, dt: number): number {
    const K = this.K, a = this.col.a!, m = this.col.m!, cc = this.col.cc!, dd = this.col.dd!;
    // tridiagonal: -dt a_k X_{k-1} + (m_k + dt(a_k + a_{k+1})) X_k - dt a_{k+1} X_{k+1} = m_k X_k (+ dt a_K xs)
    let prevC = 0, prevD = 0;
    for (let k = 0; k < K; k++) {
      const lower = -dt * a[k]!;
      const upper = k < K - 1 ? -dt * a[k + 1]! : 0;
      const diag = m[k]! + dt * (a[k]! + a[k + 1]!);
      let rhs = m[k]! * X[k]!;
      if (k === K - 1) rhs += dt * a[K]! * xs;
      const den = diag - lower * prevC;
      cc[k] = upper / den;
      dd[k] = (rhs - lower * prevD) / den;
      prevC = cc[k]!; prevD = dd[k]!;
    }
    X[K - 1] = dd[K - 1]!;
    for (let k = K - 2; k >= 0; k--) X[k] = dd[k]! - cc[k]! * X[k + 1]!;
    return a[K]! * (xs - X[K - 1]!);
  }
}
