// Drives the axisymmetric tropical-cyclone model inside the regional worker and presents it like a 3-D
// experiment: the radius-height fields are revolved onto a square display grid (3-D volume, maps,
// slices, sections, soundings), while the radius-height chart and the Hovmoller diagram use the native
// fields. Display only: the model itself stays axisymmetric.

import { AxisymModel, HA, type AxisymConfig } from '../../regional/axisym.js';
import { IceMicrophysics, QC, QR, QI, QS, QG } from '../../regional/ice.js';
import { tcSounding, eyewallPeaks, type TcSounding } from '../../regional/tropical.js';
import { cloudExtinction, precipExtinction, iceExtinction, extByte, wByte, iceByte, subgridCloud, subgridRHc, albedo } from '../../regional/display.js';
import { GALE7, GALE10 } from '../../regional/storms.js';
import { SECTION_VARS, WV_PATH, qsatW, sectionValues, parcelAscent, pressure, type MapVar, type SliceVar, type SectionVar, type RzVar } from '../../regional/diagnostics.js';
import type { ChartData, ChartRequest, GroundField, RegionalFrame, TcRain } from './protocol.js';

/** User-adjustable parameters of the axisymmetric experiment. */
export interface AxiParams {
  sst: number;          // K
  dr: number;           // m
  lh: number; lv: number;   // mixing lengths (m)
  ck: number;           // enthalpy exchange coefficient
  vmin: number;         // minimum surface wind in the fluxes (m/s)
  radMax: number;       // maximum radiative cooling (K/day)
  radConst: number;     // constant clear-sky tropospheric cooling (K/day; 0 = Newtonian relaxation, RE87)
  rhTop: number;        // relative humidity of the sounding at 12 km (0.4 standard, 0.6 moist; RE87 sounding only)
  snd: TcSounding;      // environment: conditionally unstable tropical sounding or the neutral RE87 sounding
  blNoise: number;      // stochastic boundary-layer perturbations (K per 10 min, 3-D only; 0 = none)
  vmax0: number;        // initial vortex maximum wind (m/s)
  f: number;            // Coriolis parameter (s^-1)
}
export const AXI_DEFAULTS: AxiParams = { sst: 301.15, dr: 4000, lh: 1000, lv: 100, ck: 1.2e-3, vmin: 1, radMax: 2, radConst: 1.5, rhTop: 0.4, snd: 'unstable', blNoise: 0.1, vmax0: 15, f: 5e-5 };
/** Environment of saves made before the unstable sounding (neutral RE87 sounding, radiative relaxation): merged under saved parameters. */
export const LEGACY_TC: Pick<AxiParams, 'snd' | 'radConst' | 'blNoise'> = { snd: 're87', radConst: 0, blNoise: 0 };
/** Short bilingual description of the TC environment. */
export function tcEnvText(p: AxiParams): { zh: string; en: string } {
  const snd = p.snd === 'unstable' ? { zh: '條件不穩定熱帶探空（CAPE 約 1000 J/kg）', en: 'conditionally unstable tropical sounding (CAPE about 1000 J/kg)' } : { zh: `RE87 中性探空（12 km RH ${Math.round(100 * p.rhTop)}%）`, en: `neutral RE87 sounding (RH ${Math.round(100 * p.rhTop)}% at 12 km)` };
  return {
    zh: `${(p.sst - 273.15).toFixed(1)} °C、${snd.zh}、${p.radConst ? `固定冷卻 ${p.radConst} K/day` : `輻射鬆弛（上限 ${p.radMax} K/day）`}、最小風速 ${p.vmin} m/s`,
    en: `${(p.sst - 273.15).toFixed(1)} °C, ${snd.en}, ${p.radConst ? `constant cooling ${p.radConst} K/day` : `radiative relaxation (cap ${p.radMax} K/day)`}, minimum wind ${p.vmin} m/s`,
  };
}

const NV = SECTION_VARS.length;
const SV = Object.fromEntries(SECTION_VARS.map((v, i) => [v, i])) as Record<SectionVar, number>;

export class AxiDriver {
  readonly ax: AxisymModel;
  readonly mp: IceMicrophysics;
  /** display grid: N x N cells of size dxv over [0, 2 D] (centre at (D, D)) */
  readonly N = 160; readonly D: number; readonly dxv: number;
  private prevRain: { t: number; acc: Float64Array } | null = null;
  /** core / outer precipitation rates over the last completed model hour (as the 3-D experiments) */
  private hourRain: { t: number; acc: Float64Array } | null = null;
  private tcRain: TcRain | null = null;
  private rate: Float64Array | null = null;

  constructor(readonly p: AxiParams) {
    const R = 800000, nr = Math.round(R / p.dr), nz = 50, dz = 500;
    const cfg: AxisymConfig = {
      nr, nz, dr: p.dr, dz, dt: Math.min(20, 7.5 * p.dr / 1000), nsound: 6, f: p.f, dampDepth: 6000, dampRate: 1 / 300,
      spongeWidth: 150000, spongeRate: 1 / 900, lh: p.lh, lv: p.lv, sst: p.sst, ck: p.ck, vmin: p.vmin, radTau: 12 * 3600, radMax: p.radMax / 86400, radConst: (p.radConst ?? 0) / 86400,
    };
    this.ax = new AxisymModel(cfg, tcSounding(p.snd ?? 're87', p.sst, p.rhTop ?? 0.4));
    this.mp = new IceMicrophysics(this.ax);
    this.ax.insertVortex(p.vmax0, 20000);
    this.D = 240000; this.dxv = 2 * this.D / this.N;
  }

  get time(): number { return this.ax.time; }
  get dt(): number { return this.ax.a.dt; }
  description(): string {
    const p = this.p, e = tcEnvText({ ...LEGACY_TC, ...p });
    return `軸對稱颱風快速版 / Axisymmetric TC (fast)：Δr ${p.dr / 1000} km、半徑 800 km、高 25 km；${e.zh}；混合長度 ${p.lh} / ${p.lv} m、Ck ${p.ck.toExponential(1)} ` +
      `/ Δr ${p.dr / 1000} km, 800 km radius, 25 km deep; ${e.en}; mixing lengths ${p.lh} / ${p.lv} m, Ck ${p.ck.toExponential(1)}. 顯示時繞軸旋轉成 3D / revolved about the axis for display`;
  }

  step(n: number): void { for (let s = 0; s < n; s++) { this.ax.step(); this.mp.apply(this.ax.a.dt); } }

  // ---------------------------------------------------------------- per-radius diagnostics

  /** radii needed for display: the revolved square (corner at D sqrt 2) and the radius-height chart (400 km) */
  private get nDisp(): number { return Math.min(this.ax.a.nr, Math.ceil(Math.max(1.45 * this.D, 400000) / this.ax.a.dr) + 2); }

  /** SECTION_VARS per level and radius (layout [var][k][i], radii 0..nDisp-1); u = radial, v = tangential wind. */
  private sectionTable(): Float32Array {
    const ax = this.ax, { nz } = ax.a, nr = this.nDisp, out = new Float32Array(NV * nz * nr), tmp = new Float32Array(NV), q6 = new Float64Array(6);
    for (let k = 0; k < nz; k++) for (let i = 0; i < nr; i++) {
      const q = ax.idx(i, 0, k);
      for (let s = 0; s < 6; s++) q6[s] = ax.scalars[s]![q]!;
      sectionValues({ u: 0.5 * (ax.u[q]! + ax.u[q + 1]!), v: ax.v[q]!, w: 0.5 * (ax.w[q]! + ax.w[q + ax.sx]!), th: ax.th[q]!, pp: ax.pp[q]!, q: q6 }, { th0: ax.th0[k]!, pi0: ax.pi0[k]!, rho0: ax.rho0[k]! }, tmp, 0);
      for (let v = 0; v < NV; v++) out[(v * nz + k) * nr + i] = tmp[v]!;
    }
    return out;
  }

  /** Vertical vorticity (1/r) d(r v)/dr at cell centres, level k. */
  private zeta(k: number): Float64Array {
    const ax = this.ax, { nr, dr } = ax.a, z = new Float64Array(nr);
    for (let i = 0; i < nr; i++) {
      const q = ax.idx(i, 0, k), r = ax.rc[i + HA]!;
      z[i] = (ax.rc[i + 1 + HA]! * ax.v[q + 1]! - ax.rc[i - 1 + HA]! * ax.v[q - 1]!) / (2 * dr * r);
    }
    return z;
  }

  /** draw sub-grid cloud (display.ts) */
  private subgrid = true;
  /** Display extinctions (1/m) of cloud and precipitation at radius index i, level k (display.ts, as the 3-D model). */
  private ext(i: number, k: number): { c: number; p: number; ice: number; T: number } {
    const ax = this.ax, q = ax.idx(i, 0, k), S = ax.scalars, rho = ax.rho0[k]!, pi = ax.pi0[k]! + ax.pp[q]!, T = ax.th[q]! * pi;
    const qsub = this.subgrid && S[QC]![q]! <= 1e-8 ? subgridCloud(S[0]![q]!, qsatW(T, pressure(pi)), subgridRHc(ax.a.dr)) : 0;
    return { c: cloudExtinction(rho, S[QC]![q]!, qsub, S[QI]![q]!, S[QS]![q]!, T), p: precipExtinction(rho, S[QR]![q]!, S[QG]![q]!), ice: iceExtinction(rho, qsub, S[QI]![q]!, S[QS]![q]!, T), T };
  }

  /** Outermost radius (m) of lowest-level wind of at least `v` m/s (the axisymmetric gale radius; 0 where none). */
  private galeRadius(v: number): number {
    const ax = this.ax; let r = 0;
    for (let i = 0; i < ax.a.nr; i++) { const q = ax.idx(i, 0, 0); if (Math.hypot(0.5 * (ax.u[q]! + ax.u[q + 1]!), ax.v[q]!) >= v) r = ax.rc[i + HA]!; }
    return r;
  }

  /** Sea-level pressure at the centre (hPa), reduced from the lowest level as in storms.ts. */
  private centralSlp(): number {
    const ax = this.ax, q = ax.idx(0, 0, 0), pi = ax.pi0[0]! + ax.pp[q]!;
    return 1e5 * Math.pow(pi, 1004.5 / 287.05) / 100 * Math.exp(9.80665 * ax.zc[0]! / (287.05 * ax.th[q]! * pi * (1 + 0.61 * Math.max(0, ax.scalars[0]![q]!))));
  }

  /** Column composites per radius (same meaning as the 3-D column records). */
  private columns(): { dbz: Float64Array; ctopT: Float64Array; ctopZ: Float64Array; uh: Float64Array; wMax: Float64Array; cape: Float64Array; cin: Float64Array; vis: Float64Array; visZ: Float64Array; pw: Float64Array; wvT: Float64Array; wmin: number; cmax: number; pmax: number } {
    const ax = this.ax, { nz, dz } = ax.a, nr = this.nDisp;
    const vis = new Float64Array(nr), visZ = new Float64Array(nr), pw = new Float64Array(nr), wvT = new Float64Array(nr), bet = new Float64Array(nz);
    const dbz = new Float64Array(nr).fill(-30), ctopT = new Float64Array(nr), ctopZ = new Float64Array(nr), uh = new Float64Array(nr), wMax = new Float64Array(nr), cape = new Float64Array(nr), cin = new Float64Array(nr);
    const T = new Float64Array(nz), p = new Float64Array(nz), qv = new Float64Array(nz);
    const zs = Array.from({ length: nz }, (_, k) => (ax.zc[k]! >= 2000 && ax.zc[k]! <= 5000 ? this.zeta(k) : null));
    let wmin = 0, cmax = 0, pmax = 0;
    for (let i = 0; i < nr; i++) {
      let zmax = 0, tau = 0;
      for (let k = 0; k < nz; k++) {
        const q = ax.idx(i, 0, k), rho = ax.rho0[k]!, S = ax.scalars;
        bet[k] = this.ext(i, k).c; tau += bet[k]! * dz; pw[i] = pw[i]! + rho * dz * Math.max(0, S[0]![q]!);
        const qr = Math.max(S[QR]![q]!, 0), qs = Math.max(S[QS]![q]!, 0), qg = Math.max(S[QG]![q]!, 0);
        zmax = Math.max(zmax, 3.63e9 * Math.pow(rho * qr, 1.75) + 9.80e8 * Math.pow(rho * qs, 1.75) + 4.33e10 * Math.pow(rho * qg, 1.75));
        const pi = ax.pi0[k]! + ax.pp[q]!;
        T[k] = ax.th[q]! * pi; p[k] = pressure(pi); qv[k] = S[0]![q]!;
        const cl = Math.max(0, S[QC]![q]! + S[QI]![q]!), pr = Math.max(0, S[QR]![q]! + qs + qg);
        if (cl > 1e-5) { ctopZ[i] = ax.zc[k]!; ctopT[i] = T[k]!; }
        const wc = 0.5 * (ax.w[q]! + ax.w[q + ax.sx]!);
        if (zs[k]) uh[i] = uh[i]! + wc * zs[k]![i]! * dz;
        wMax[i] = Math.max(wMax[i]!, ax.w[q]!); wmin = Math.min(wmin, ax.w[q]!); cmax = Math.max(cmax, cl); pmax = Math.max(pmax, pr);
      }
      if (ctopZ[i] === 0) ctopT[i] = T[0]!;
      dbz[i] = 10 * Math.log10(Math.max(zmax, 1e-3));
      const pc = parcelAscent(T, p, qv, dz); cape[i] = pc.cape; cin[i] = pc.cin;
      // satellite-like values as columnDiagnostics (src/regional/diagnostics.ts)
      let above = pw[i]!, zEmit = ax.zc[nz - 1]!; wvT[i] = T[nz - 1]!;
      for (let k = 0; k < nz; k++) { above -= ax.rho0[k]! * dz * Math.max(0, qv[k]!); if (above < WV_PATH) { wvT[i] = T[k]!; zEmit = ax.zc[k]!; break; } }
      if (ctopZ[i]! > zEmit) wvT[i] = ctopT[i]!;
      vis[i] = albedo(tau);
      let below = 0, zw = 0;
      for (let k = 0; k < nz; k++) { const d = bet[k]! * dz; if (tau - below >= 1) visZ[i] = ax.zc[k]!; zw += d * ax.zc[k]!; below += d; }
      if (tau < 1) visZ[i] = tau > 1e-3 ? zw / tau : 0;
    }
    return { dbz, ctopT, ctopZ, uh, wMax, cape, cin, vis, visZ, pw, wvT, wmin, cmax, pmax };
  }

  // ---------------------------------------------------------------- revolving onto the display grid

  /** Radius (m) and direction of display cell (i, j). */
  private polar(i: number, j: number): { r: number; c: number; s: number } {
    const x = (i + 0.5) * this.dxv - this.D, y = (j + 0.5) * this.dxv - this.D, r = Math.hypot(x, y);
    return { r, c: r > 0 ? x / r : 1, s: r > 0 ? y / r : 0 };
  }
  /** Linear interpolation weights in radius between cell centres. */
  private rw(r: number): { i0: number; w: number } {
    const nr = this.nDisp, x = Math.max(0, Math.min(nr - 1.000001, r / this.ax.a.dr - 0.5)), i0 = Math.floor(x);
    return { i0, w: x - i0 };
  }
  private revolve(prof: ArrayLike<number>): Float32Array {
    const N = this.N, out = new Float32Array(N * N);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { const { r } = this.polar(i, j), { i0, w } = this.rw(r); out[j * N + i] = prof[i0]! * (1 - w) + prof[i0 + 1]! * w; }
    return out;
  }
  /** Revolve a (radial, tangential) wind pair into Cartesian components. */
  private revolveWind(vr: ArrayLike<number>, vt: ArrayLike<number>): { u: Float32Array; v: Float32Array } {
    const N = this.N, u = new Float32Array(N * N), v = new Float32Array(N * N);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const { r, c, s } = this.polar(i, j), { i0, w } = this.rw(r);
      const a = vr[i0]! * (1 - w) + vr[i0 + 1]! * w, b = vt[i0]! * (1 - w) + vt[i0 + 1]! * w;
      u[j * N + i] = a * c - b * s; v[j * N + i] = a * s + b * c;
    }
    return { u, v };
  }

  // ---------------------------------------------------------------- frames

  frame(req: ChartRequest | null, ground: GroundField, volMode: number, stepsPerSecond: number, origin: { x: number; y: number }, subgrid = true): { msg: RegionalFrame; transfer: ArrayBuffer[] } {
    this.subgrid = subgrid;
    const ax = this.ax, { nr, nz, dr, dz } = ax.a, N = this.N, S = ax.scalars;
    // precipitation rate from the accumulation change
    const acc = Float64Array.from(this.mp.rainAcc);
    if (this.prevRain && ax.time > this.prevRain.t + 1e-6) { const f = 3600 / (ax.time - this.prevRain.t); this.rate = acc.map((a, i) => Math.max(0, (a - this.prevRain!.acc[i]!) * f)); }
    if (!this.prevRain || ax.time > this.prevRain.t + 1e-6 || ax.time < this.prevRain.t) this.prevRain = { t: ax.time, acc };
    if (!this.hourRain || ax.time < this.hourRain.t) { this.hourRain = { t: ax.time, acc }; this.tcRain = null; }
    else if (ax.time - this.hourRain.t >= 3600 - 1e-6) {
      // area-weighted (r dr) means over the core (< 60 km) and the outer region (100-300 km); outer area fraction above 1 mm/h
      const f = 3600 / (ax.time - this.hourRain.t);
      let sc = 0, wc = 0, so = 0, wo = 0, wet = 0;
      for (let i = 0; i < nr; i++) {
        const r = ax.rc[i + HA]!, rate = Math.max(0, acc[i]! - this.hourRain.acc[i]!) * f;
        if (r < 60000) { sc += rate * r; wc += r; } else if (r >= 100000 && r < 300000) { so += rate * r; wo += r; if (rate > 1) wet += r; }
      }
      this.tcRain = { core: wc ? sc / wc : 0, outer: wo ? so / wo : 0, wet: wo ? wet / wo : 0 };
      this.hourRain = { t: ax.time, acc };
    }
    // per radius and level: cloud byte and channel-2 byte, revolved into the 3-D volume
    const cb = new Uint8Array(nr * nz), pb = new Uint8Array(nr * nz), wbA = new Uint8Array(nr * nz), ibA = new Uint8Array(nr * nz);
    const zetaL: (Float64Array | null)[] = volMode === 2 ? Array.from({ length: nz }, (_, k) => this.zeta(k)) : [];
    let wmax = 0, qcmax = 0, qrmax = 0;
    for (let k = 0; k < nz; k++) for (let i = 0; i < nr; i++) {
      const q = ax.idx(i, 0, k);
      // extinction bytes as the 3-D experiments (display.ts): cloud with snow and sub-grid cloud, channel 2 rain + graupel
      const cl = Math.max(0, S[QC]![q]! + S[QI]![q]!), pr = Math.max(0, S[QR]![q]! + S[QS]![q]! + S[QG]![q]!), e = this.ext(i, k);
      cb[k * nr + i] = extByte(e.c);
      wbA[k * nr + i] = wByte(0.5 * (ax.w[q]! + ax.w[q + ax.sx]!)); ibA[k * nr + i] = iceByte(e.ice, e.c);
      let v2 = extByte(e.p) / 255;
      if (volMode === 1) v2 = Math.sqrt(Math.max(0.5 * (ax.w[q]! + ax.w[q + ax.sx]!), 0) / 40);
      else if (volMode === 2) v2 = Math.sqrt(Math.max(zetaL[k]![i]!, 0) / 0.05);
      pb[k * nr + i] = Math.min(255, Math.round(v2 * 255));
      qcmax = Math.max(qcmax, cl); qrmax = Math.max(qrmax, pr); wmax = Math.max(wmax, ax.w[q]!);
    }
    const n = N * N * nz, cloud = new Uint8Array(n), rain = new Uint8Array(n), aux = new Uint8Array(2 * n);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const { r } = this.polar(i, j), { i0, w } = this.rw(r);
      for (let k = 0; k < nz; k++) {
        const a = k * nr + i0, o = (k * N + j) * N + i;
        cloud[o] = cb[a]! * (1 - w) + cb[a + 1]! * w; rain[o] = pb[a]! * (1 - w) + pb[a + 1]! * w;
        aux[2 * o] = wbA[a]! * (1 - w) + wbA[a + 1]! * w; aux[2 * o + 1] = ibA[a]! * (1 - w) + ibA[a + 1]! * w;
      }
    }
    // surface profiles
    const vr0 = new Float64Array(nr), vt0 = new Float64Array(nr), spd = new Float64Array(nr), thp0 = new Float64Array(nr);
    for (let i = 0; i < nr; i++) { const q = ax.idx(i, 0, 0); vr0[i] = 0.5 * (ax.u[q]! + ax.u[q + 1]!); vt0[i] = ax.v[q]!; spd[i] = Math.hypot(vr0[i]!, vt0[i]!); thp0[i] = ax.th[q]! - ax.th0[0]!; }
    const gprof = ground === 'rain' ? this.mp.rainAcc : ground === 'snow' ? this.mp.snowAcc : ground === 'wind' ? spd : ground === 'none' ? new Float64Array(spd.length) : thp0;
    const g = this.revolve(gprof);
    let lo = Infinity, hi = -Infinity; for (const v of g) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (ground === 'theta') { const a = Math.max(Math.abs(lo), Math.abs(hi), 0.5); lo = -a; hi = a; } else { lo = 0; hi = Math.max(hi, ground === 'rain' || ground === 'snow' ? 5 : 10); }
    // storm statistics
    const mt = ax.metrics();
    let k15 = 0; for (let k = 0; k < nz; k++) if (Math.abs(ax.zc[k]! - 1500) < Math.abs(ax.zc[k15]! - 1500)) k15 = k;
    const nOut = Math.min(nr, Math.ceil(300000 / dr)), rr: number[] = [], vt: number[] = [];
    for (let i = 0; i < nOut; i++) { rr.push(ax.rc[i + HA]!); vt.push(+ax.v[ax.idx(i, 0, k15)]!.toFixed(2)); }
    const col = this.columns(), z0 = this.zeta(0), nd = this.nDisp;
    let zetaMax = 0, rainmax = 0, dbzMax = 0, uhMax = 0, capeMax = 0;
    for (let i = 0; i < nr; i++) { zetaMax = Math.max(zetaMax, z0[i]!); rainmax = Math.max(rainmax, this.mp.rainAcc[i]!); }
    for (let i = 0; i < nd; i++) { dbzMax = Math.max(dbzMax, col.dbz[i]!); uhMax = Math.max(uhMax, col.uh[i]!); capeMax = Math.max(capeMax, col.cape[i]!); }
    const transfer: ArrayBuffer[] = [cloud.buffer, rain.buffer, aux.buffer, g.buffer];
    const charts = req ? this.charts(req, col, transfer) : null;
    const msg: RegionalFrame = {
      type: 'frame', time: ax.time, nx: N, ny: N, nz, dx: this.dxv, dz, cloud, rain, aux, ground: g, groundField: ground, groundRange: [lo, hi],
      stats: { wmax, wmin: col.wmin, qcmax, qrmax, rainmax, vmax: mt.vmax, dp: mt.dp, rmw: mt.rmw, eyewalls: eyewallPeaks(rr, vt), zetaMax, vGround: mt.vmax, dbzMax, uhMax, uhMin: 0, capeMax,
        storm: { x: origin.x + this.D, y: origin.y + this.D }, vtProfile: { dr, vt }, tornado: null, tcRain: this.tcRain,
        // the one storm of the axisymmetric model: the vortex on the axis
        storms: [{ id: 1, kind: 'vortex', name: 'TC1', x: origin.x + this.D, y: origin.y + this.D, xd: this.D, yd: this.D, u: 0, v: 0, age: ax.time,
          pmin: this.centralSlp(), dp: -mt.dp, vmax: mt.vmax, rmw: mt.rmw, r7: this.galeRadius(GALE7), r10: this.galeRadius(GALE10) }], mainId: 1 },
      origin, anchor: { x: 0, y: 0 }, charts, tracers: null, stepsPerSecond, dt: ax.a.dt,
    };
    return { msg, transfer };
  }

  private charts(req: ChartRequest, col: ReturnType<AxiDriver['columns']>, transfer: ArrayBuffer[]): ChartData {
    const ax = this.ax, { nz, dr } = ax.a, nr = this.nDisp;
    const tab = this.sectionTable(), at = (v: SectionVar, k: number, i: number): number => tab[(SV[v] * nz + k) * nr + i]!;
    const charts: ChartData = { maps: {}, slice: null, section: null, sounding: null, rz: null };
    const keep = <T extends Float32Array>(a: T): T => { transfer.push(a.buffer as ArrayBuffer); return a; };
    // composite and surface maps
    const sfc = (v: SectionVar): Float64Array => Float64Array.from({ length: nr }, (_, i) => at(v, 0, i));
    let wind: { u: Float32Array; v: Float32Array } | null = null;
    const windMaps = (): { u: Float32Array; v: Float32Array } => (wind ??= this.revolveWind(sfc('u'), sfc('v')));
    for (const mv of req.maps as MapVar[]) {
      let prof: ArrayLike<number> | null = null;
      switch (mv) {
        case 'dbzMax': prof = col.dbz; break;
        case 'ctopT': prof = col.ctopT.map((x) => x - 273.15); break;
        case 'ctopZ': prof = col.ctopZ.map((x) => x / 1000); break;
        case 'uh': prof = col.uh; break;
        case 'wMax': prof = col.wMax; break;
        case 'cape': prof = col.cape; break;
        case 'cin': prof = col.cin; break;
        case 'vis': prof = col.vis; break;
        case 'visZ': prof = col.visZ; break;
        case 'pw': prof = col.pw; break;
        case 'wvT': prof = col.wvT.map((x) => x - 273.15); break;
        case 'rain': prof = this.mp.rainAcc; break;
        case 'snow': prof = this.mp.snowAcc; break;
        case 'rainRate': prof = this.rate ?? new Float64Array(nr); break;
        case 'sfcWind': prof = Float64Array.from({ length: nr }, (_, i) => Math.hypot(at('u', 0, i), at('v', 0, i))); break;
        case 'sfcThp': prof = sfc('thp'); break;
        case 'sfcThetaE': prof = sfc('thetaE'); break;
        case 'slp': prof = Float64Array.from({ length: nr }, (_, i) => { const q = ax.idx(i, 0, 0), pi = ax.pi0[0]! + ax.pp[q]!, tv = ax.th[q]! * pi * (1 + 0.61 * ax.scalars[0]![q]!); return pressure(pi) * Math.exp(9.80665 * ax.zc[0]! / (287.05 * tv)) / 100; }); break;
        case 'sfcU': charts.maps.sfcU = keep(windMaps().u); break;
        case 'sfcV': charts.maps.sfcV = keep(windMaps().v); break;
      }
      if (prof) charts.maps[mv] = keep(this.revolve(prof));
    }
    // horizontal slice
    if (req.slice) {
      const k = Math.max(0, Math.min(nz - 1, req.slice.k | 0)), vars: Partial<Record<SliceVar, Float32Array>> = {};
      const prof = (v: SectionVar): Float64Array => Float64Array.from({ length: nr }, (_, i) => at(v, k, i));
      const w2 = new Set(req.slice.vars);
      let wk: { u: Float32Array; v: Float32Array } | null = null;
      const wkf = (): { u: Float32Array; v: Float32Array } => (wk ??= this.revolveWind(prof('u'), prof('v')));
      for (const sv of w2) {
        let a: Float32Array | null = null;
        if (sv === 'u') a = wkf().u; else if (sv === 'v') a = wkf().v;
        else if (sv === 'speed') a = this.revolve(Float64Array.from({ length: nr }, (_, i) => Math.hypot(at('u', k, i), at('v', k, i))));
        else if (sv === 'zeta') a = this.revolve(this.zeta(k));
        else if (sv === 'div') {
          // (1/r) d(r u)/dr at the cell centres (1e-5 s^-1)
          const rf = ax.rf;
          a = this.revolve(Float64Array.from({ length: nr }, (_, i) => { const q = ax.idx(i, 0, k); return 1e5 * (rf[i + 1 + HA]! * ax.u[q + 1]! - rf[i + HA]! * ax.u[q]!) / (ax.rc[i + HA]! * ax.a.dr); }));
        }
        else a = this.revolve(prof(sv as SectionVar));
        vars[sv] = keep(a);
      }
      charts.slice = { k, z: ax.zc[k]!, vars };
    }
    // columns at arbitrary points (sections and soundings): radial interpolation, Cartesian winds
    const column = (x: number, y: number, out: Float32Array, np: number, p: number): void => {
      const dx = x - this.D, dy = y - this.D, r = Math.hypot(dx, dy), c = r > 0 ? dx / r : 1, s = r > 0 ? dy / r : 0, { i0, w } = this.rw(r);
      for (let k = 0; k < nz; k++) {
        const val = (v: SectionVar): number => at(v, k, i0) * (1 - w) + at(v, k, i0 + 1) * w;
        for (const v of SECTION_VARS) out[(SV[v] * nz + k) * np + p] = val(v);
        const a = val('u'), b = val('v');
        out[(SV.u * nz + k) * np + p] = a * c - b * s; out[(SV.v * nz + k) * np + p] = a * s + b * c;
      }
    };
    const split = (all: Float32Array, np: number): Record<SectionVar, Float32Array> => {
      const rec = {} as Record<SectionVar, Float32Array>;
      SECTION_VARS.forEach((v, f) => { rec[v] = keep(all.slice(f * nz * np, (f + 1) * nz * np)); });
      return rec;
    };
    if (req.section) {
      const { x0, y0, x1, y1 } = req.section, np = Math.max(2, Math.min(600, Math.round(Math.hypot(x1 - x0, y1 - y0) / this.dxv) + 1));
      const all = new Float32Array(NV * nz * np);
      for (let p = 0; p < np; p++) column(x0 + (x1 - x0) * p / (np - 1), y0 + (y1 - y0) * p / (np - 1), all, np, p);
      charts.section = { x0, y0, x1, y1, np, vars: split(all, np) };
    }
    if (req.sounding) {
      const all = new Float32Array(NV * nz);
      column(req.sounding.x, req.sounding.y, all, 1, 0);
      charts.sounding = { x: req.sounding.x, y: req.sounding.y, vars: split(all, 1) };
    }
    // radius-height: native fields out to 400 km (the storm is at the axis whatever the requested centre)
    if (req.rz) {
      const nrz = Math.min(nr, Math.ceil(400000 / dr)), vars = {} as Record<RzVar, Float32Array>;
      for (const v of ['vt', 'vr', 'w', 'thp', 'cond'] as RzVar[]) vars[v] = new Float32Array(nrz * nz);
      for (let k = 0; k < nz; k++) for (let i = 0; i < nrz; i++) {
        const o = k * nrz + i;
        vars.vt[o] = at('v', k, i); vars.vr[o] = at('u', k, i); vars.w[o] = at('w', k, i); vars.thp[o] = at('thp', k, i); vars.cond[o] = at('cloud', k, i) + at('precip', k, i);
      }
      for (const v of Object.values(vars)) keep(v);
      charts.rz = { xc: this.D, yc: this.D, dr, nr: nrz, vars };
    }
    return charts;
  }

  // ---------------------------------------------------------------- saves

  arrays(): Record<string, Float32Array> {
    const ax = this.ax, f32 = (a: ArrayLike<number>): Float32Array => Float32Array.from(a);
    const out: Record<string, Float32Array> = { u: f32(ax.u), v: f32(ax.v), w: f32(ax.w), th: f32(ax.th), pp: f32(ax.pp), rain: f32(this.mp.rainAcc), snow: f32(this.mp.snowAcc) };
    ax.scalars.forEach((a, i) => { out[`s${i}`] = f32(a); });
    return out;
  }
  restore(arrays: Record<string, Float32Array | Uint8Array>, time: number, steps: number): void {
    const ax = this.ax;
    const need = (k: string): Float32Array => { const a = arrays[k] as Float32Array | undefined; if (!a || a.length !== (k === 'rain' || k === 'snow' ? ax.a.nr : ax.size)) throw new Error('存檔網格不符 / the saved grid does not match'); return a; };
    for (const [k, a] of [['u', ax.u], ['v', ax.v], ['w', ax.w], ['th', ax.th], ['pp', ax.pp]] as const) a.set(need(k));
    ax.scalars.forEach((a, i) => a.set(need(`s${i}`)));
    this.mp.rainAcc.set(need('rain')); this.mp.snowAcc.set(need('snow'));
    ax.time = time; ax.steps = steps;
    this.prevRain = null; this.hourRain = null; this.tcRain = null;
  }
}
