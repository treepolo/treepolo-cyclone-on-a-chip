// Spherical-harmonic transform on a Gaussian grid, triangular truncation T.
//
// Normalisation: orthonormal associated Legendre functions, int_{-1}^{1} P_n^m(mu)^2 dmu = 1,
// no Condon–Shortley phase. A real field is
//   X(lambda, mu) = sum_{m=0}^{T} sum_{n=m}^{T} X_n^m P_n^m(mu) e^{i m lambda}   (+ c.c. for m > 0)
// so that X_n^m = int int X P_n^m e^{-i m lambda} dmu dlambda / (2 pi).
//
// H_n^m(mu) = (1 - mu^2) dP_n^m/dmu is used for meridional derivatives.
// Grid arrays are [lat][lon] with latitudes north -> south.

import { gaussianLatitudes } from './gauss.js';
import { RealFFT } from './fft.js';

export interface SpectralField { re: Float64Array; im: Float64Array; }

export class SpectralTransform {
  readonly trunc: number;
  readonly nlon: number;
  readonly nlat: number;
  readonly nspec: number;
  readonly mu: Float64Array;
  readonly weight: Float64Array;
  readonly lat: Float64Array;
  readonly lon: Float64Array;
  readonly coslat: Float64Array;
  /** spectral index -> m, n */
  readonly mOf: Int32Array;
  readonly nOf: Int32Array;
  /** start index of each m block */
  readonly mStart: Int32Array;
  /** n(n+1) for each spectral index */
  readonly nn1: Float64Array;
  private readonly half: number;
  // Legendre tables for northern-hemisphere latitudes: [spec][jHalf]
  private readonly P: Float64Array;
  private readonly H: Float64Array;
  private readonly fft: RealFFT;
  // scratch Fourier buffers [lat][m]
  private readonly fr: Float64Array;
  private readonly fi: Float64Array;
  private readonly gr: Float64Array;
  private readonly gi: Float64Array;
  private readonly rowR: Float64Array;
  private readonly rowI: Float64Array;
  private readonly scratchPsi: SpectralField;
  private readonly symR: Float64Array; private readonly symI: Float64Array;
  private readonly antR: Float64Array; private readonly antI: Float64Array;
  private readonly scratchChi: SpectralField;

  constructor(trunc: number, nlon?: number) {
    this.trunc = trunc;
    let n = nlon ?? 0;
    if (!n) { n = 4; while (n < 3 * trunc + 1) n *= 2; }
    this.nlon = n;
    this.nlat = n / 2;
    this.half = this.nlat / 2;
    const g = gaussianLatitudes(this.nlat);
    this.mu = g.mu; this.weight = g.weight; this.lat = g.lat;
    this.coslat = new Float64Array(this.nlat);
    for (let j = 0; j < this.nlat; j++) this.coslat[j] = Math.sqrt(1 - this.mu[j]! ** 2);
    this.lon = new Float64Array(n);
    for (let i = 0; i < n; i++) this.lon[i] = 2 * Math.PI * i / n;

    const T = trunc;
    this.nspec = (T + 1) * (T + 2) / 2;
    this.mOf = new Int32Array(this.nspec);
    this.nOf = new Int32Array(this.nspec);
    this.mStart = new Int32Array(T + 2);
    this.nn1 = new Float64Array(this.nspec);
    let s = 0;
    for (let m = 0; m <= T; m++) {
      this.mStart[m] = s;
      for (let nn = m; nn <= T; nn++) { this.mOf[s] = m; this.nOf[s] = nn; this.nn1[s] = nn * (nn + 1); s++; }
    }
    this.mStart[T + 1] = s;

    const hlf = this.half;
    this.P = new Float64Array(this.nspec * hlf);
    this.H = new Float64Array(this.nspec * hlf);
    const pm = new Float64Array(T + 3);
    for (let j = 0; j < hlf; j++) {
      const x = this.mu[j]!, cx = Math.sqrt(1 - x * x);
      let pmm = Math.SQRT1_2;
      for (let m = 0; m <= T; m++) {
        if (m > 0) pmm *= Math.sqrt((2 * m + 1) / (2 * m)) * cx;
        // P_n^m for n = m .. T+1
        pm[m] = pmm;
        if (m + 1 <= T + 1) pm[m + 1] = Math.sqrt(2 * m + 3) * x * pmm;
        for (let nn = m + 2; nn <= T + 1; nn++) {
          pm[nn] = (x * pm[nn - 1]! - eps(nn - 1, m) * pm[nn - 2]!) / eps(nn, m);
        }
        for (let nn = m; nn <= T; nn++) {
          const idx = this.mStart[m]! + nn - m;
          this.P[idx * hlf + j] = pm[nn]!;
          const hm = (nn + 1) * eps(nn, m) * (nn > m ? pm[nn - 1]! : 0) - nn * eps(nn + 1, m) * pm[nn + 1]!;
          this.H[idx * hlf + j] = hm;
        }
      }
    }
    this.fft = new RealFFT(n);
    this.fr = new Float64Array(this.nlat * (T + 1));
    this.fi = new Float64Array(this.nlat * (T + 1));
    this.gr = new Float64Array(this.nlat * (T + 1));
    this.gi = new Float64Array(this.nlat * (T + 1));
    this.rowR = new Float64Array(T + 1);
    this.rowI = new Float64Array(T + 1);
    this.scratchPsi = this.newSpec();
    this.symR = new Float64Array(hlf); this.symI = new Float64Array(hlf);
    this.antR = new Float64Array(hlf); this.antI = new Float64Array(hlf);
    this.scratchChi = this.newSpec();
  }

  get gridSize(): number { return this.nlat * this.nlon; }

  newSpec(): SpectralField { return { re: new Float64Array(this.nspec), im: new Float64Array(this.nspec) }; }
  newGrid(): Float64Array { return new Float64Array(this.nlat * this.nlon); }

  // ---------- Fourier stage ----------
  private gridToFourier(grid: Float64Array, fr: Float64Array, fi: Float64Array, scaleByInvCos2: boolean): void {
    const T = this.trunc, nl = this.nlon;
    for (let j = 0; j < this.nlat; j++) {
      this.fft.forward(grid, j * nl, this.rowR, this.rowI, T);
      const sc = scaleByInvCos2 ? 1 / (1 - this.mu[j]! ** 2) : 1;
      const o = j * (T + 1);
      for (let m = 0; m <= T; m++) { fr[o + m] = this.rowR[m]! * sc; fi[o + m] = this.rowI[m]! * sc; }
    }
  }

  private fourierToGrid(fr: Float64Array, fi: Float64Array, grid: Float64Array): void {
    const T = this.trunc, nl = this.nlon;
    for (let j = 0; j < this.nlat; j++) {
      const o = j * (T + 1);
      for (let m = 0; m <= T; m++) { this.rowR[m] = fr[o + m]!; this.rowI[m] = fi[o + m]!; }
      this.fft.inverse(this.rowR, this.rowI, T, grid, j * nl);
    }
  }

  // ---------- Legendre stage ----------
  /**
   * Accumulate into Fourier buffers (fr, fi) the Legendre synthesis of spectral coefficients:
   *   F_m(mu) += sum_n c_n^m * L_n^m(mu), with c = (a + i b) * factor, factor = 1 or i*m
   * kind 'P' uses P_n^m, kind 'H' uses H_n^m.
   */
  private legendreSynth(sre: Float64Array, sim: Float64Array, kind: 'P' | 'H', imDeriv: boolean, scale: number,
                        fr: Float64Array, fi: Float64Array): void {
    const T = this.trunc, hlf = this.half, nlat = this.nlat;
    const L = kind === 'P' ? this.P : this.H;
    // parity: for P, P(-mu) = (-1)^{n+m} P(mu); for H, H(-mu) = (-1)^{n+m+1} H(mu)
    const parityShift = kind === 'P' ? 0 : 1;
    for (let m = 0; m <= T; m++) {
      const s0 = this.mStart[m]!;
      for (let j = 0; j < hlf; j++) {
        let er = 0, ei = 0, or = 0, oi = 0; // sums over n+m even / odd
        for (let nn = m; nn <= T; nn++) {
          const idx = s0 + nn - m;
          const l = L[idx * hlf + j]!;
          let cr = sre[idx]!, ci = sim[idx]!;
          if (imDeriv) { const t = cr; cr = -m * ci; ci = m * t; }
          if (((nn - m) & 1) === 0) { er += cr * l; ei += ci * l; } else { or += cr * l; oi += ci * l; }
        }
        const oN = j * (T + 1) + m, oS = (nlat - 1 - j) * (T + 1) + m;
        fr[oN] = fr[oN]! + scale * (er + or);
        fi[oN] = fi[oN]! + scale * (ei + oi);
        if (parityShift === 0) {
          fr[oS] = fr[oS]! + scale * (er - or);
          fi[oS] = fi[oS]! + scale * (ei - oi);
        } else {
          fr[oS] = fr[oS]! + scale * (or - er);
          fi[oS] = fi[oS]! + scale * (oi - ei);
        }
      }
    }
  }

  /**
   * Accumulate into spectral (ore, oim): sum_j w_j F_m(mu_j) L_n^m(mu_j) * factor * scale,
   * factor = 1, or i*m when imDeriv.
   */
  private legendreAnal(fr: Float64Array, fi: Float64Array, kind: 'P' | 'H', imDeriv: boolean, scale: number,
                       ore: Float64Array, oim: Float64Array): void {
    const T = this.trunc, hlf = this.half, nlat = this.nlat;
    const L = kind === 'P' ? this.P : this.H;
    const parityShift = kind === 'P' ? 0 : 1;
    const w = this.weight;
    const sr = this.symR, si = this.symI, ar = this.antR, ai = this.antI;
    for (let m = 0; m <= T; m++) {
      // weighted symmetric / antisymmetric Fourier coefficients for this m
      for (let j = 0; j < hlf; j++) {
        const oN = j * (T + 1) + m, oS = (nlat - 1 - j) * (T + 1) + m, wj = w[j]!;
        sr[j] = (fr[oN]! + fr[oS]!) * wj; si[j] = (fi[oN]! + fi[oS]!) * wj;
        ar[j] = (fr[oN]! - fr[oS]!) * wj; ai[j] = (fi[oN]! - fi[oS]!) * wj;
      }
      const s0 = this.mStart[m]!;
      for (let nn = m; nn <= T; nn++) {
        const idx = s0 + nn - m, base = idx * hlf;
        const useSym = (((nn - m) + parityShift) & 1) === 0;
        const xr = useSym ? sr : ar, xi = useSym ? si : ai;
        let accR = 0, accI = 0;
        for (let j = 0; j < hlf; j++) { const l = L[base + j]!; accR += xr[j]! * l; accI += xi[j]! * l; }
        if (imDeriv) { const t = accR; accR = -m * accI; accI = m * t; }
        ore[idx] = ore[idx]! + scale * accR;
        oim[idx] = oim[idx]! + scale * accI;
      }
    }
  }

  // ---------- public API ----------

  /** Plain synthesis: spectral -> grid. */
  synth(s: SpectralField, grid: Float64Array): void {
    this.fr.fill(0); this.fi.fill(0);
    this.legendreSynth(s.re, s.im, 'P', false, 1, this.fr, this.fi);
    this.fourierToGrid(this.fr, this.fi, grid);
  }

  /** Grid of dX/dlambda and (1-mu^2) dX/dmu. */
  synthGrad(s: SpectralField, dLambda: Float64Array, dMu: Float64Array): void {
    this.fr.fill(0); this.fi.fill(0);
    this.legendreSynth(s.re, s.im, 'P', true, 1, this.fr, this.fi);
    this.fourierToGrid(this.fr, this.fi, dLambda);
    this.fr.fill(0); this.fi.fill(0);
    this.legendreSynth(s.re, s.im, 'H', false, 1, this.fr, this.fi);
    this.fourierToGrid(this.fr, this.fi, dMu);
  }

  /**
   * U = u cos(phi), V = v cos(phi) from vorticity and divergence spectra.
   *   U = (1/a)[ dchi/dlambda - (1-mu^2) dpsi/dmu ],  V = (1/a)[ dpsi/dlambda + (1-mu^2) dchi/dmu ]
   */
  synthUV(vor: SpectralField, div: SpectralField, radius: number, U: Float64Array, V: Float64Array): void {
    const psi = this.scratchPsi, chi = this.scratchChi;
    const a2 = radius * radius;
    for (let s = 0; s < this.nspec; s++) {
      const f = this.nn1[s]! > 0 ? -a2 / this.nn1[s]! : 0;
      psi.re[s] = vor.re[s]! * f; psi.im[s] = vor.im[s]! * f;
      chi.re[s] = div.re[s]! * f; chi.im[s] = div.im[s]! * f;
    }
    const ia = 1 / radius;
    this.fr.fill(0); this.fi.fill(0);
    this.legendreSynth(chi.re, chi.im, 'P', true, ia, this.fr, this.fi);
    this.legendreSynth(psi.re, psi.im, 'H', false, -ia, this.fr, this.fi);
    this.fourierToGrid(this.fr, this.fi, U);
    this.fr.fill(0); this.fi.fill(0);
    this.legendreSynth(psi.re, psi.im, 'P', true, ia, this.fr, this.fi);
    this.legendreSynth(chi.re, chi.im, 'H', false, ia, this.fr, this.fi);
    this.fourierToGrid(this.fr, this.fi, V);
  }

  /** Plain analysis: grid -> spectral (accumulate with scale when add=true). */
  anal(grid: Float64Array, out: SpectralField, scale = 1, add = false): void {
    if (!add) { out.re.fill(0); out.im.fill(0); }
    this.gridToFourier(grid, this.fr, this.fi, false);
    this.legendreAnal(this.fr, this.fi, 'P', false, scale, out.re, out.im);
  }

  /**
   * Spectral divergence and/or curl of a vector (A, B) = (Fx cos phi, Fy cos phi):
   *   div  = [1/(a(1-mu^2))] dA/dlambda + (1/a) dB/dmu
   *   curl = [1/(a(1-mu^2))] dB/dlambda - (1/a) dA/dmu
   * Results are accumulated (scaled) into divOut / curlOut when provided.
   */
  analDivCurl(A: Float64Array, B: Float64Array, radius: number,
              divOut: SpectralField | null, curlOut: SpectralField | null, scale = 1): void {
    const ia = scale / radius;
    this.gridToFourier(A, this.fr, this.fi, true);
    this.gridToFourier(B, this.gr, this.gi, true);
    if (divOut) {
      // int dB/dmu P dmu = - int B dP/dmu = - int B H/(1-mu^2)
      this.legendreAnal(this.fr, this.fi, 'P', true, ia, divOut.re, divOut.im);
      this.legendreAnal(this.gr, this.gi, 'H', false, -ia, divOut.re, divOut.im);
    }
    if (curlOut) {
      this.legendreAnal(this.gr, this.gi, 'P', true, ia, curlOut.re, curlOut.im);
      this.legendreAnal(this.fr, this.fi, 'H', false, ia, curlOut.re, curlOut.im);
    }
  }
}

function eps(n: number, m: number): number {
  return Math.sqrt((n * n - m * m) / (4 * n * n - 1));
}
