// Gray-radiation moist physics for an Earth-like planet, applied column by column after each
// dynamics step. Default configuration = the moist aquaplanet of Frierson, Held & Zurita-Gotor (2006)
// and Frierson (2007), as in the Isca "frierson" test case; options add land, seasons and a
// water-vapour-dependent longwave optical depth (Byrne & O'Gorman 2013).
//
//   1. two-stream gray longwave radiation; shortwave absorbed weakly in the atmosphere and at the surface
//   2. bulk surface fluxes (simplified Monin–Obukhov) + implicit K-profile boundary-layer diffusion
//      of u, v, dry static energy and q, with a prognostic boundary-layer depth (bulk Ri = 1)
//   3. surface: slab mixed-layer ocean (optional prescribed ocean heat transport, simple sea-ice albedo)
//      and land (small heat capacity, Manabe bucket hydrology, larger roughness)
//   4. Rayleigh sponge above 50 hPa (energy conserving)
//   5. Simplified Betts–Miller convection
//   6. large-scale condensation with re-evaporation of falling rain
// All fluxes are in SI units; every process conserves water and energy within the column
// (runoff from full buckets is the only water leaving the system, and it is accounted).

import { DAY, DryAir, Planet } from '../../core/constants.js';
import { ColumnPhysics, MoistGridState } from '../dycore.js';
import { MOIST, dqsatdT, qsat } from './thermo.js';
import { sbmColumn, sbmWork, SbmWork } from './sbm.js';

export interface GrayPhysicsConfig {
  radiation: 'frierson' | 'byrne';
  solarConstant: number;     // W m^-2
  delSol: number;            // perpetual-equinox pole-equator insolation contrast (Frierson p2 profile)
  seasonal: boolean;         // daily-mean insolation with obliquity (otherwise perpetual equinox p2 profile)
  obliquityDeg: number;
  yearLength: number;        // s
  albedo: number;            // surface albedo, ocean
  albedoLand: number;
  albedoIce: number;         // sea ice (or ocean colder than freezing without seaIce)
  /** zero-layer thermodynamic sea ice (Semtner 1976): the mixed layer freezes at 271.35 K, ice grows
   *  and melts from the conductive, surface and ocean (q-flux) heat budgets */
  seaIce: boolean;
  atmAbs: number;            // shortwave optical depth at the surface
  tauEq: number;             // Frierson longwave optical depth at the surface, equator
  tauPole: number;
  linearTau: number;
  byrneA: number;            // Byrne–O'Gorman dtau/dsigma = a + b q
  byrneB: number;
  mixedLayerDepth: number;   // m (ocean)
  landHeatCapacity: number;  // J m^-2 K^-1
  roughness: number;         // m (ocean)
  roughnessLand: number;     // m
  bucketMax: number;         // m of water
  qflux: boolean;            // prescribed ocean heat transport (Merlis et al. 2013 form)
  qfluxAmp: number;          // W m^-2
  qfluxWidthDeg: number;
  richCrit: number;
  fracInner: number;
  spongePBottom: number;     // Pa
  spongeTau: number;         // s
  useConvection: boolean;
}

export const AQUA: GrayPhysicsConfig = {
  radiation: 'frierson', solarConstant: 1360, delSol: 1.4, seasonal: false, obliquityDeg: 23.44, yearLength: 360 * DAY,
  albedo: 0.31, albedoLand: 0.31, albedoIce: 0.31, atmAbs: 0.2, tauEq: 6.0, tauPole: 1.5, linearTau: 0.1,
  byrneA: 0.8678, byrneB: 1997.9,
  mixedLayerDepth: 2.5, landHeatCapacity: 1e6, roughness: 3.21e-5, roughnessLand: 3.21e-5, bucketMax: 0.15,
  qflux: false, qfluxAmp: 30, qfluxWidthDeg: 16,
  seaIce: false,
  richCrit: 1.0, fracInner: 0.1, spongePBottom: 5000, spongeTau: 0.25 * DAY, useConvection: true,
};

/** Surface description per grid point. */
export interface SurfaceMap {
  land: Uint8Array;          // 1 = land
  zsurf: Float64Array;       // m
}

export interface AquaplanetFields {
  /** surface (skin / mixed-layer) temperature, K — SST over ocean, ground temperature over land */
  sst: Float64Array;
  bucket: Float64Array;       // soil water, m (land only)
  ice: Float64Array;          // sea-ice thickness, m (ocean only, seaIce)
  /** running accumulations since the last reset (per grid point) */
  precipConv: Float64Array;   // kg m^-2
  precipLS: Float64Array;
  evap: Float64Array;
  shf: Float64Array;          // J m^-2
  olr: Float64Array;          // J m^-2
  runoff: Float64Array;       // kg m^-2
  tsAcc: Float64Array;        // K s
  accTime: number;            // s
  /** instantaneous */
  precipRate: Float64Array;   // kg m^-2 s^-1 (last step)
  snowRate: Float64Array;     // kg m^-2 s^-1 (last step): precipitation reaching the ground as snow
  snowAcc: Float64Array;      // kg m^-2 accumulated snowfall
  olrNow: Float64Array;       // W m^-2
  blDepth: Float64Array;      // m
}

/** Monthly climatology on the model grid: 12 fields at month-centre days of a `yearDays`-day year
 *  whose day 0 is 1 January. */
export interface MonthlyField { days: number[]; yearDays: number; fields: Float64Array[] }

/** Day of the (climatological) year at model time t; model time 0 is the March equinox, ~20 March. */
export function climDay(t: number, yearLength: number, yearDays: number): number {
  const d = 79 / 360 * yearDays + t / yearLength * yearDays;
  return ((d % yearDays) + yearDays) % yearDays;
}

/** Linear interpolation weights between the two climatological months bracketing day d. */
export function monthWeights(mf: MonthlyField, d: number): [number, number, number] {
  const n = mf.days.length, Y = mf.yearDays;
  for (let m = 0; m < n; m++) {
    const a = mf.days[m]!, b = m + 1 < n ? mf.days[m + 1]! : mf.days[0]! + Y;
    const dd = d < mf.days[0]! ? d + Y : d;
    if (dd >= a && dd < b) return [m, (m + 1) % n, (dd - a) / (b - a)];
  }
  return [n - 1, 0, 0];
}

export class GrayPhysics implements ColumnPhysics {
  readonly f: AquaplanetFields;
  readonly cfg: GrayPhysicsConfig;
  readonly surface: SurfaceMap;
  private readonly K: number;
  private readonly nlon: number;
  private readonly col: Record<string, Float64Array>;
  private readonly work: SbmWork;
  private readonly qfluxLat: Float64Array;
  /** clipped negative water (kg m^-2 summed over points) */
  negativeWater = 0;
  /** current solar declination (rad) */
  declination = 0;
  /** prescribed sea-surface temperature climatology (fixed-SST / AMIP-type runs); open ocean only */
  sstClim: MonthlyField | null = null;
  /** loaded ocean heat-flux convergence climatology (W m^-2), replacing the analytic q-flux */
  qfluxField: MonthlyField | null = null;
  /** fixed-SST runs: monthly accumulations of the net surface energy flux into the ocean (J m^-2)
   *  and of the weights (s), for deriving the implied q-flux */
  readonly netAcc: Float64Array[];
  readonly netW: Float64Array;

  constructor(private readonly planet: Planet, private readonly air: DryAir, nlat: number, nlon: number, K: number,
              readonly lat: Float64Array, cfg: Partial<GrayPhysicsConfig> = {}, surface?: SurfaceMap) {
    const ng = nlat * nlon;
    this.K = K;
    this.nlon = nlon;
    this.cfg = { ...AQUA, ...cfg };
    this.surface = surface ?? { land: new Uint8Array(ng), zsurf: new Float64Array(ng) };
    const z = (): Float64Array => new Float64Array(ng);
    this.f = { sst: z(), bucket: z(), ice: z(), precipConv: z(), precipLS: z(), evap: z(), shf: z(), olr: z(), runoff: z(), tsAcc: z(), accTime: 0, precipRate: z(), snowRate: z(), snowAcc: z(), olrNow: z(), blDepth: z() };
    const c = (n: number): Float64Array => new Float64Array(n);
    this.col = {
      T: c(K), q: c(K), u: c(K), v: c(K), pf: c(K), ph: c(K + 1), zf: c(K), zh: c(K + 1),
      lwu: c(K + 1), lwd: c(K + 1), tr: c(K), s: c(K), a: c(K + 1), m: c(K), x: c(K),
      cc: c(K), dd: c(K), kk: c(K + 1), sw: c(K + 1),
    };
    this.work = sbmWork(K);
    this.netAcc = Array.from({ length: 12 }, () => new Float64Array(ng));
    this.netW = new Float64Array(12);
    this.qfluxLat = new Float64Array(nlat);
    const w = this.cfg.qfluxWidthDeg * Math.PI / 180;
    for (let j = 0; j < nlat; j++) {
      const la = lat[j]!, s = Math.sin(la);
      // Merlis et al. (2013): divergence of ocean heat transport, heats the extratropics, cools the tropics
      this.qfluxLat[j] = this.cfg.qflux ? this.cfg.qfluxAmp * (1 - 2 * la * la / (w * w)) * Math.exp(-la * la / (w * w)) / Math.cos(la) : 0;
      for (let i = 0; i < nlon; i++) {
        const p = j * nlon + i;
        this.f.sst[p] = 270 + 35 * (1 - s * s) - (this.surface.land[p] ? 0.0065 * this.surface.zsurf[p]! : 0);
        this.f.bucket[p] = this.surface.land[p] ? this.cfg.bucketMax : 0;
      }
    }
    // q-flux must integrate to zero over the ocean: remove its ocean-area mean
    if (this.cfg.qflux) {
      let s = 0, a = 0;
      for (let j = 0; j < nlat; j++) {
        const c2 = Math.cos(lat[j]!);
        for (let i = 0; i < nlon; i++) if (!this.surface.land[j * nlon + i]) { s += this.qfluxLat[j]! * c2; a += c2; }
      }
      const mean = a > 0 ? s / a : 0;
      for (let j = 0; j < nlat; j++) this.qfluxLat[j] = this.qfluxLat[j]! - mean;
    }
  }

  resetAccumulators(): void {
    for (const a of [this.f.precipConv, this.f.precipLS, this.f.evap, this.f.shf, this.f.olr, this.f.runoff, this.f.tsAcc, this.f.snowAcc]) a.fill(0);
    this.f.accTime = 0;
  }

  /**
   * Implied ocean heat-flux convergence from a fixed-SST run (Russell et al. 1985): for each month
   * Q = C dT/dt - F_net with T the prescribed climatology and F_net the accumulated net surface flux;
   * the annual ocean-area mean is removed so that the ocean heat transport integrates to zero.
   * Points that were never open water (sea ice, land) get 0.
   */
  impliedQflux(): MonthlyField {
    const sc = this.sstClim;
    if (!sc) throw new Error('impliedQflux needs a fixed-SST run');
    const ng = this.f.sst.length, C = MOIST.rhoWater * MOIST.cpWater * this.cfg.mixedLayerDepth, dtm = this.cfg.yearLength / 12;
    const fields = Array.from({ length: 12 }, (_, m) => {
      const out = new Float64Array(ng), prev = sc.fields[(m + 11) % 12]!, next = sc.fields[(m + 1) % 12]!, w = this.netW[m]!;
      for (let p = 0; p < ng; p++) {
        if (this.surface.land[p] || w === 0) continue;
        const acc = this.netAcc[m]![p]!;
        if (acc === 0) continue;
        out[p] = C * (next[p]! - prev[p]!) / (2 * dtm) - acc / w;
      }
      return out;
    });
    // remove the annual, ocean-area-weighted mean over the points that carry a q-flux
    let s = 0, a = 0;
    const nlon = this.nlon;
    for (let p = 0; p < ng; p++) {
      const c = Math.cos(this.lat[Math.floor(p / nlon)]!);
      let anyv = false, sum = 0;
      for (let m = 0; m < 12; m++) { if (fields[m]![p]! !== 0) anyv = true; sum += fields[m]![p]!; }
      if (anyv) { s += c * sum / 12; a += c; }
    }
    const mean = a > 0 ? s / a : 0;
    for (const f of fields) for (let p = 0; p < ng; p++) if (f[p]! !== 0) f[p] = f[p]! - mean;
    return { days: sc.days, yearDays: sc.yearDays, fields };
  }

  /** Daily-mean top-of-atmosphere insolation (W m^-2) at latitude lat for the current declination. */
  insolation(lat: number): number {
    const c = this.cfg;
    if (!c.seasonal) {
      const s = Math.sin(lat);
      return 0.25 * c.solarConstant * (1 + c.delSol * (1 - 3 * s * s) / 4);
    }
    const d = this.declination;
    const x = Math.max(-1, Math.min(1, -Math.tan(lat) * Math.tan(d)));
    const h0 = Math.acos(x);
    return c.solarConstant / Math.PI * (h0 * Math.sin(lat) * Math.sin(d) + Math.cos(lat) * Math.cos(d) * Math.sin(h0));
  }

  /** Update the solar declination for model time t (s); t = 0 is the northern spring equinox. */
  setTime(t: number): void {
    const c = this.cfg;
    this.declination = c.seasonal ? Math.asin(Math.sin(c.obliquityDeg * Math.PI / 180) * Math.sin(2 * Math.PI * t / c.yearLength)) : 0;
  }

  apply(st: MoistGridState, dt: number, time = 0): void {
    const { nlat, nlon, K } = st, ng = nlat * nlon;
    const C = this.col;
    this.setTime(time);
    const sc = this.sstClim, qc = this.qfluxField;
    const ws = sc ? monthWeights(sc, climDay(time, this.cfg.yearLength, sc.yearDays)) : null;
    const wq = qc ? monthWeights(qc, climDay(time, this.cfg.yearLength, qc.yearDays)) : null;
    if (ws) { this.netW[ws[0]] = this.netW[ws[0]]! + (1 - ws[2]) * dt; this.netW[ws[1]] = this.netW[ws[1]]! + ws[2] * dt; }
    this.fixedW = ws;
    for (let j = 0; j < nlat; j++) {
      const la = this.lat[j]!, sl = Math.sin(la), insol = this.insolation(la);
      for (let i = 0; i < nlon; i++) {
        const p = j * nlon + i;
        const qf = wq ? qc!.fields[wq[0]]![p]! * (1 - wq[2]) + qc!.fields[wq[1]]![p]! * wq[2] : this.qfluxLat[j]!;
        this.sstNow = ws ? sc!.fields[ws[0]]![p]! * (1 - ws[2]) + sc!.fields[ws[1]]![p]! * ws[2] : NaN;
        for (let k = 0; k < K; k++) {
          const q = k * ng + p;
          C.T![k] = st.T[q]!; C.q![k] = st.q[q]!; C.u![k] = st.u[q]!; C.v![k] = st.v[q]!;
        }
        this.column(p, sl, insol, qf, st.ps[p]!, st.sigma, st.sigmaHalf, dt);
        for (let k = 0; k < K; k++) {
          const q = k * ng + p;
          st.T[q] = C.T![k]!; st.q[q] = C.q![k]!; st.u[q] = C.u![k]!; st.v[q] = C.v![k]!;
        }
      }
    }
    this.f.accTime += dt;
  }

  private sstNow = NaN;
  private fixedW: [number, number, number] | null = null;

  private column(p: number, sinLat: number, insol: number, qflux: number, ps: number,
                 sigma: Float64Array, sigmaHalf: Float64Array, dt: number): void {
    const K = this.K, C = this.col, cfg = this.cfg, g = this.planet.gravity, cp = this.air.cp, R = this.air.rd;
    const T = C.T!, q = C.q!, u = C.u!, v = C.v!, pf = C.pf!, ph = C.ph!, zf = C.zf!, zh = C.zh!;
    const f = this.f, isLand = this.surface.land[p] === 1;
    for (let k = 0; k <= K; k++) ph[k] = sigmaHalf[k]! * ps;
    for (let k = 0; k < K; k++) pf[k] = sigma[k]! * ps;
    // heights above the surface (Simmons–Burridge hydrostatics, consistent with the dynamical core)
    zh[K] = 0;
    for (let k = K - 1; k >= 0; k--) {
      const lnr = k === 0 ? Math.LN2 * 2 : Math.log(ph[k + 1]! / ph[k]!);
      const alpha = k === 0 ? Math.LN2 : 1 - ph[k]! / (ph[k + 1]! - ph[k]!) * lnr;
      zf[k] = zh[k + 1]! + alpha * R * T[k]! / g;
      zh[k] = zh[k + 1]! + R * T[k]! * lnr / g;
    }

    // ---------------- 1. radiation
    const ts = f.sst[p]!;
    const lwu = C.lwu!, lwd = C.lwd!, tr = C.tr!;
    if (cfg.radiation === 'frierson') {
      const tau0 = cfg.tauEq + (cfg.tauPole - cfg.tauEq) * sinLat * sinLat;
      const tauAt = (pp: number): number => { const x = pp / 1e5; return tau0 * (cfg.linearTau * x + (1 - cfg.linearTau) * x * x * x * x); };
      for (let k = 0; k < K; k++) tr[k] = Math.exp(-(tauAt(ph[k + 1]!) - tauAt(ph[k]!)));
    } else {
      for (let k = 0; k < K; k++) tr[k] = Math.exp(-(cfg.byrneA + cfg.byrneB * q[k]!) * (ph[k + 1]! - ph[k]!) / 1e5);
    }
    lwd[0] = 0;
    for (let k = 0; k < K; k++) { const B = MOIST.stefan * T[k]! ** 4; lwd[k + 1] = lwd[k]! * tr[k]! + B * (1 - tr[k]!); }
    lwu[K] = MOIST.stefan * ts ** 4;
    for (let k = K - 1; k >= 0; k--) { const B = MOIST.stefan * T[k]! ** 4; lwu[k] = lwu[k + 1]! * tr[k]! + B * (1 - tr[k]!); }
    const sw = C.sw!;
    for (let k = 0; k <= K; k++) { const x = ph[k]! / 1e5; sw[k] = insol * Math.exp(-cfg.atmAbs * x * x * x * x); }
    let albedo = isLand ? cfg.albedoLand : cfg.albedo;
    if (!isLand && cfg.seaIce) {
      // bare-ice albedo reached at 0.5 m thickness
      albedo = cfg.albedo + Math.min(1, f.ice[p]! / 0.5) * (cfg.albedoIce - cfg.albedo);
    } else if (!isLand && cfg.albedoIce !== cfg.albedo) {
      // simple sea-ice albedo: linear ramp from open water at 273.15 K to ice at 263.15 K
      const w = Math.max(0, Math.min(1, (273.15 - ts) / 10));
      albedo = cfg.albedo + w * (cfg.albedoIce - cfg.albedo);
    }
    const swSfc = sw[K]!, swUp = albedo * swSfc;
    for (let k = 0; k < K; k++) {
      const Ftop = lwu[k]! - lwd[k]! + swUp - sw[k]!;
      const Fbot = lwu[k + 1]! - lwd[k + 1]! + swUp - sw[k + 1]!;
      T[k] = T[k]! + dt * g * (Fbot - Ftop) / (cp * (ph[k + 1]! - ph[k]!));
    }
    const olr = lwu[0]!;

    // ---------------- 2. surface fluxes + boundary-layer diffusion (implicit)
    const z0 = isLand ? cfg.roughnessLand : cfg.roughness;
    const ka = K - 1, za = zf[ka]!;
    const speed = Math.max(Math.hypot(u[ka]!, v[ka]!), 1e-3);
    const rhoA = pf[ka]! / (R * T[ka]!);
    const thetaDiff = T[ka]! + g * za / cp - ts;
    const ri = g * za * thetaDiff / (ts * speed * speed);
    const lnz = Math.log(za / z0);
    const cn = (MOIST.vonKarman / lnz) ** 2;
    const rc = cfg.richCrit;
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
      return MOIST.vonKarman * ustar * z / (1 + x * Math.log(z / z0) / (1 - x));
    };
    const hIn = cfg.fracInner * h, kRef = kmo(hIn);
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
    const aSfc = rhoA * cd * speed;
    for (let k = 0; k < K; k++) m[k] = (ph[k + 1]! - ph[k]!) / g;
    const s = C.s!;
    for (let k = 0; k < K; k++) s[k] = cp * T[k]! + g * zf[k]!;
    // evaporation efficiency: 1 over ocean, Manabe bucket over land
    const beta = isLand ? Math.min(1, f.bucket[p]! / (0.75 * cfg.bucketMax)) : 1;
    a[K] = aSfc;
    const fluxS = this.diffuse(s, cp * ts, dt);
    a[K] = aSfc * beta;
    let fluxQ = this.diffuse(q, qsat(ts, ps), dt);
    a[K] = aSfc;
    this.diffuse(u, 0, dt);
    this.diffuse(v, 0, dt);
    for (let k = 0; k < K; k++) T[k] = (s[k]! - g * zf[k]!) / cp;
    // dew onto a dry bucket is allowed (fluxQ < 0); evaporation cannot exceed the available soil water
    if (isLand && fluxQ * dt > f.bucket[p]! * MOIST.rhoWater) {
      const excess = fluxQ * dt - f.bucket[p]! * MOIST.rhoWater;   // kg m^-2 over-evaporated
      q[ka] = q[ka]! - excess / m[ka]!;
      fluxQ -= excess / dt;
    }
    const lh = MOIST.Lv * fluxQ;
    // surface energy budget
    const heatCap = isLand ? cfg.landHeatCapacity : MOIST.rhoWater * MOIST.cpWater * cfg.mixedLayerDepth;
    const net = swSfc * (1 - albedo) + lwd[K]! - MOIST.stefan * ts ** 4 - fluxS - lh + (isLand ? 0 : qflux);
    const sstC = this.sstNow;
    if (!isLand && Number.isFinite(sstC) && sstC > SEA_ICE.TF + 0.2) {
      // fixed SST over open water: record the surface energy flux (without q-flux) for the implied q-flux
      const w = this.fixedW!, e = (net - qflux) * dt;
      this.netAcc[w[0]]![p] = this.netAcc[w[0]]![p]! + (1 - w[2]) * e;
      this.netAcc[w[1]]![p] = this.netAcc[w[1]]![p]! + w[2] * e;
      f.sst[p] = sstC; f.ice[p] = 0;
    } else if (!isLand && cfg.seaIce) seaIceStep(f, p, ts, net, qflux, heatCap, dt);
    else f.sst[p] = ts + dt * net / heatCap;

    // ---------------- 4. sponge above 50 hPa (kinetic energy lost is returned as heat)
    for (let k = 0; k < K; k++) {
      if (pf[k]! >= cfg.spongePBottom) break;
      const x = (cfg.spongePBottom - pf[k]!) / cfg.spongePBottom;
      const r = x * x / cfg.spongeTau;
      const un = u[k]! / (1 + dt * r), vn = v[k]! / (1 + dt * r);
      T[k] = T[k]! + 0.5 * (u[k]! ** 2 + v[k]! ** 2 - un * un - vn * vn) / cp;
      u[k] = un; v[k] = vn;
    }

    // ---------------- 5. convection
    let rainConv = 0;
    if (cfg.useConvection) rainConv = sbmColumn(T, q, pf, ph, dt, g, this.work).rain;

    // ---------------- 6. large-scale condensation with re-evaporation
    const hlcp = MOIST.Lv / cp;
    let exq = 0;
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
    const rainLS = Math.max(0, exq);
    for (let k = 0; k < K; k++) if (q[k]! < 0) { this.negativeWater -= q[k]! * (ph[k + 1]! - ph[k]!) / g; q[k] = 0; }

    // ---------------- land hydrology (bucket): dW/dt = (P - E)/rho_w, runoff above capacity
    if (isLand) {
      let w = f.bucket[p]! + (rainConv + rainLS - fluxQ * dt) / MOIST.rhoWater;
      if (w < 0) w = 0;
      if (w > cfg.bucketMax) { f.runoff[p] = f.runoff[p]! + (w - cfg.bucketMax) * MOIST.rhoWater; w = cfg.bucketMax; }
      f.bucket[p] = w;
    }

    f.precipConv[p] = f.precipConv[p]! + rainConv;
    f.precipLS[p] = f.precipLS[p]! + rainLS;
    f.evap[p] = f.evap[p]! + fluxQ * dt;
    f.shf[p] = f.shf[p]! + fluxS * dt;
    f.olr[p] = f.olr[p]! + olr * dt;
    f.tsAcc[p] = f.tsAcc[p]! + f.sst[p]! * dt;
    f.olrNow[p] = olr;
    f.precipRate[p] = (rainConv + rainLS) / dt;
    // precipitation phase at the ground (diagnostic; the gray benchmark physics has no latent heat of fusion)
    const snow = T[K - 1]! < 273.15 ? rainConv + rainLS : 0;
    f.snowRate[p] = snow / dt;
    f.snowAcc[p] = f.snowAcc[p]! + snow;
  }

  /**
   * Backward-Euler vertical diffusion of X with conductances a (kg m^-2 s^-1) at half levels,
   * layer masses m, and a surface node of fixed value xs coupled through a[K].
   * Returns the upward surface flux a[K] (xs - X_bottom) (X-units kg m^-2 s^-1).
   */
  private diffuse(X: Float64Array, xs: number, dt: number): number {
    const K = this.K, a = this.col.a!, m = this.col.m!, cc = this.col.cc!, dd = this.col.dd!;
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

/** Sea-ice constants: freezing point of sea water, melting point, ice density, latent heat of fusion,
 *  conductivity, heat capacity of the ice surface layer, minimum thickness in the conduction term. */
export const SEA_ICE = { TF: 271.35, TMELT: 273.15, RHOI: 917, LF: 3.34e5, KI: 2.03, CS: 1e6, HMIN: 0.05 };

/**
 * One step of the zero-layer thermodynamic sea-ice / mixed-layer model for ocean point p.
 * Open water: the mixed layer (heat capacity cml) absorbs the net surface flux `net` (including the
 * ocean heat convergence qflux); cooling below TF freezes ice with latent heat LF.
 * Ice: the surface layer (heat capacity CS) takes the atmospheric flux plus conduction k (TF - Ts)/h
 * from the ocean at TF; surface warming above TMELT melts ice; at the base the conductive loss minus
 * qflux freezes (or melts) ice. Ice that melts away returns its leftover energy to the mixed layer.
 * Conserved: E = C_ml (T - TF) (open water) or CS (Ts - TF) (ice), minus rho_i L_f h; dE/dt = net.
 */
export function seaIceStep(f: { sst: Float64Array; ice: Float64Array }, p: number, ts: number, net: number, qflux: number, cml: number, dt: number): void {
  const I = SEA_ICE, h = f.ice[p]!, rl = I.RHOI * I.LF;
  if (h <= 0) {
    let tml = ts + dt * net / cml;
    let hn = 0;
    if (tml < I.TF) { hn = cml * (I.TF - tml) / rl; tml = I.TF; }
    f.sst[p] = tml; f.ice[p] = hn;
    return;
  }
  const cond = I.KI * (I.TF - ts) / Math.max(h, I.HMIN);
  let tsn = ts + dt * (net - qflux + cond) / I.CS;
  let hn = h;
  if (tsn > I.TMELT) { hn -= I.CS * (tsn - I.TMELT) / rl; tsn = I.TMELT; }
  hn += dt * (cond - qflux) / rl;
  // melted out: the leftover latent energy and the surface layer's heat go to the mixed layer
  if (hn <= 0) { tsn = I.TF + (-hn * rl + I.CS * (tsn - I.TF)) / cml; hn = 0; }
  f.sst[p] = tsn; f.ice[p] = hn;
}

/** The Frierson et al. (2006) moist gray aquaplanet (Isca "frierson" test-case settings). */
export class GrayAquaplanet extends GrayPhysics {
  constructor(planet: Planet, air: DryAir, nlat: number, nlon: number, K: number, lat: Float64Array) {
    super(planet, air, nlat, nlon, K, lat, AQUA);
  }
}
