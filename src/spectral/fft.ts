// Real FFT of power-of-two length N, via an N/2 complex radix-2 FFT.
//   forward:  c_m = (1/N) sum_i x_i exp(-2 pi i m i / N),  m = 0..mMax
//   inverse:  x_i = Re(c_0) + 2 Re sum_{m=1}^{mMax} c_m exp(+2 pi i m i / N)

export class RealFFT {
  readonly n: number;
  private readonly h: number;
  private readonly rev: Uint32Array;
  private readonly cosT: Float64Array; // twiddles for the N/2 complex FFT
  private readonly sinT: Float64Array;
  private readonly cosN: Float64Array; // twiddles exp(-2 pi i k / N) for real packing
  private readonly sinN: Float64Array;
  private readonly zr: Float64Array;
  private readonly zi: Float64Array;

  constructor(n: number) {
    if (n < 4 || (n & (n - 1)) !== 0) throw new Error('RealFFT length must be a power of two >= 4');
    this.n = n;
    const h = n / 2;
    this.h = h;
    let bits = 0;
    while ((1 << bits) < h) bits++;
    this.rev = new Uint32Array(h);
    for (let i = 0; i < h; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cosT = new Float64Array(h / 2 || 1);
    this.sinT = new Float64Array(h / 2 || 1);
    for (let k = 0; k < h / 2; k++) {
      this.cosT[k] = Math.cos(2 * Math.PI * k / h);
      this.sinT[k] = -Math.sin(2 * Math.PI * k / h);
    }
    this.cosN = new Float64Array(h + 1);
    this.sinN = new Float64Array(h + 1);
    for (let k = 0; k <= h; k++) {
      this.cosN[k] = Math.cos(2 * Math.PI * k / n);
      this.sinN[k] = -Math.sin(2 * Math.PI * k / n);
    }
    this.zr = new Float64Array(h);
    this.zi = new Float64Array(h);
  }

  // In-place complex FFT (forward sign -1 when inverse=false) on zr/zi of length h.
  private complexFFT(inverse: boolean): void {
    const { h, rev, zr, zi, cosT, sinT } = this;
    for (let i = 0; i < h; i++) {
      const r = rev[i]!;
      if (r > i) {
        let t = zr[i]!; zr[i] = zr[r]!; zr[r] = t;
        t = zi[i]!; zi[i] = zi[r]!; zi[r] = t;
      }
    }
    const sgn = inverse ? -1 : 1;
    for (let size = 2; size <= h; size <<= 1) {
      const half = size >> 1;
      const step = h / size;
      for (let start = 0; start < h; start += size) {
        for (let k = 0; k < half; k++) {
          const wr = cosT[k * step]!, wi = sgn * sinT[k * step]!;
          const a = start + k, b = a + half;
          const br = zr[b]! * wr - zi[b]! * wi;
          const bi = zr[b]! * wi + zi[b]! * wr;
          zr[b] = zr[a]! - br; zi[b] = zi[a]! - bi;
          zr[a] = zr[a]! + br; zi[a] = zi[a]! + bi;
        }
      }
    }
  }

  /** x (length n, read at offset xo) -> cr/ci (length >= mMax+1). */
  forward(x: Float64Array, xo: number, cr: Float64Array, ci: Float64Array, mMax: number): void {
    const { h, zr, zi, cosN, sinN, n } = this;
    for (let i = 0; i < h; i++) { zr[i] = x[xo + 2 * i]!; zi[i] = x[xo + 2 * i + 1]!; }
    this.complexFFT(false);
    const inv = 1 / n;
    for (let k = 0; k <= mMax; k++) {
      // X_k = (Z_k + conj Z_{h-k})/2 + W^k (Z_k - conj Z_{h-k})/(2i)
      const k1 = k % h, k2 = (h - k) % h;
      const ar = zr[k1]!, ai = zi[k1]!, br = zr[k2]!, bi = -zi[k2]!;
      const er = 0.5 * (ar + br), ei = 0.5 * (ai + bi);
      const dr = 0.5 * (ar - br), di = 0.5 * (ai - bi);
      // (dr + i di)/i = di - i dr
      const or = di, oi = -dr;
      const wr = cosN[k]!, wi = sinN[k]!;
      cr[k] = (er + wr * or - wi * oi) * inv;
      ci[k] = (ei + wr * oi + wi * or) * inv;
    }
  }

  /** cr/ci (m=0..mMax) -> x (length n, written at offset xo). */
  inverse(cr: Float64Array, ci: Float64Array, mMax: number, x: Float64Array, xo: number): void {
    const { h, zr, zi, cosN, sinN } = this;
    // Build full Hermitian spectrum X_k (k=0..h) scaled so x = sum_k X_k e^{+...}:
    // X_0 = c0, X_k = c_k for 1<=k<=mMax, 0 otherwise; x = X_0 + 2 Re sum X_k e^{ik theta}.
    // Pack: Z_k = E_k + i O_k where E_k = X_k + conj X_{h-k}, O_k = (X_k - conj X_{h-k}) W^{-k}
    for (let k = 0; k < h; k++) {
      const k2 = h - k;
      const xr = k <= mMax ? cr[k]! : 0, xi = k <= mMax ? ci[k]! : 0;
      let yr = k2 <= mMax ? cr[k2]! : 0, yi = k2 <= mMax ? -ci[k2]! : 0;
      if (k === 0) { yr = h <= mMax ? cr[h]! : 0; yi = h <= mMax ? -ci[h]! : 0; }
      // full-spectrum values F_k = X_k (k>0) doubled convention handled below
      const fkr = k === 0 ? xr : xr, fki = k === 0 ? 0 : xi;
      const gkr = yr, gki = yi; // conj(F_{h-k})
      const er = fkr + gkr, ei = fki + gki;
      const dr = fkr - gkr, di = fki - gki;
      const wr = cosN[k]!, wi = -sinN[k]!; // W^{-k} = exp(+2 pi i k/N)
      const or = dr * wr - di * wi, oi = dr * wi + di * wr;
      // Z_k = E_k + i O_k
      zr[k] = er - oi; zi[k] = ei + or;
    }
    this.complexFFT(true);
    for (let i = 0; i < h; i++) { x[xo + 2 * i] = zr[i]!; x[xo + 2 * i + 1] = zi[i]!; }
  }
}
