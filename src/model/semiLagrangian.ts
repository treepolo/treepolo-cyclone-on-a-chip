// Shape-preserving semi-Lagrangian advection of grid-point tracers on the Gaussian grid.
//
//   * three-dimensional trajectories (great-circle midpoint iteration in Cartesian space,
//     sigma-dot in the vertical), valid through the poles and for any Courant number
//   * tricubic Lagrange interpolation (cubic in longitude, cubic on the non-uniform Gaussian
//     latitudes with rows extended across the pole, cubic in sigma, linear next to the lid/surface)
//   * quasi-monotone limiter (Bermejo & Staniforth 1992): the interpolated value is clipped to the
//     range of the 8 grid points surrounding the departure point, so no new extrema and q >= 0.
// Global conservation is restored by a mass fixer in the caller (as in NCAR CCM3 / ECMWF IFS).

export interface SLGrid {
  nlat: number;
  nlon: number;
  K: number;
  lat: Float64Array;     // north -> south
  lon: Float64Array;
  sigma: Float64Array;   // full levels, top -> bottom
  radius: number;
}

export class SemiLagrangian {
  private readonly g: SLGrid;
  private readonly yExt: Float64Array;   // extended latitudes, index r + 2 for r = -2 .. nlat+1
  private readonly vx: Float64Array; private readonly vy: Float64Array; private readonly vz: Float64Array;
  private readonly depLat: Float64Array; private readonly depLon: Float64Array; private readonly depSig: Float64Array;

  constructor(g: SLGrid) {
    this.g = g;
    const n = g.nlat;
    this.yExt = new Float64Array(n + 4);
    for (let r = -2; r <= n + 1; r++) {
      let y: number;
      if (r < 0) y = Math.PI - g.lat[-r - 1]!;
      else if (r >= n) y = -Math.PI - g.lat[2 * n - 1 - r]!;
      else y = g.lat[r]!;
      this.yExt[r + 2] = y;
    }
    const size = g.K * g.nlat * g.nlon;
    this.vx = new Float64Array(size); this.vy = new Float64Array(size); this.vz = new Float64Array(size);
    this.depLat = new Float64Array(size); this.depLon = new Float64Array(size); this.depSig = new Float64Array(size);
  }

  /** Interval index j (extended) such that yExt(j) >= lat > yExt(j+1). */
  private latInterval(lat: number): number {
    const L = this.g.lat, n = this.g.nlat;
    if (lat > L[0]!) return -1;
    if (lat <= L[n - 1]!) return n - 1;
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (L[mid]! >= lat) lo = mid; else hi = mid; }
    return lo;
  }

  private sigInterval(s: number): number {
    const S = this.g.sigma, K = this.g.K;
    if (s <= S[0]!) return 0;
    if (s >= S[K - 1]!) return K - 2;
    let lo = 0, hi = K - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (S[mid]! <= s) lo = mid; else hi = mid; }
    return lo;
  }

  /** Trilinear sample of the Cartesian wind and sigma-dot at one point; results in this.wout. */
  private readonly wout = new Float64Array(4);
  private sampleWind(sdot: Float64Array, lat: number, lon: number, sig: number): void {
    const g = this.g, nl = g.nlon, ng = g.nlat * nl, n = g.nlat, h = nl >> 1;
    const j = this.latInterval(lat);
    const y0 = this.yExt[j + 2]!, y1 = this.yExt[j + 3]!;
    const wy = (y0 - lat) / (y0 - y1);
    let x = (lon / (2 * Math.PI)) * nl;
    x -= Math.floor(x / nl) * nl;
    const i0 = Math.floor(x), wx = x - i0;
    const k = this.sigInterval(sig);
    const s0 = g.sigma[k]!, s1 = g.sigma[k + 1]!;
    let wz = (sig - s0) / (s1 - s0);
    wz = wz < 0 ? 0 : wz > 1 ? 1 : wz;
    let ax = 0, ay = 0, az = 0, as = 0;
    for (let dk = 0; dk < 2; dk++) {
      const wk = dk ? wz : 1 - wz, o = (k + dk) * ng;
      for (let dj = 0; dj < 2; dj++) {
        const r = j + dj;
        let rr = r, sh = 0;
        if (r < 0) { rr = -r - 1; sh = h; } else if (r >= n) { rr = 2 * n - 1 - r; sh = h; }
        const w = wk * (dj ? wy : 1 - wy), ro = o + rr * nl;
        const ia = ro + (i0 + sh) % nl, ib = ro + (i0 + 1 + sh) % nl;
        const wa = w * (1 - wx), wb = w * wx;
        ax += wa * this.vx[ia]! + wb * this.vx[ib]!;
        ay += wa * this.vy[ia]! + wb * this.vy[ib]!;
        az += wa * this.vz[ia]! + wb * this.vz[ib]!;
        as += wa * sdot[ia]! + wb * sdot[ib]!;
      }
    }
    const o = this.wout;
    o[0] = ax; o[1] = ay; o[2] = az; o[3] = as;
  }

  /**
   * Advect fields over dt using winds u, v (m/s) and sigma-dot (s^-1) at full levels, all [k][lat][lon].
   * Each entry of `fields` is read from `src[i]` and written to `dst[i]`.
   */
  advect(u: Float64Array, v: Float64Array, sdot: Float64Array, dt: number,
         src: Float64Array[], dst: Float64Array[]): void {
    const g = this.g, nl = g.nlon, ng = g.nlat * nl, K = g.K, a = g.radius;
    // Cartesian wind
    for (let j = 0; j < g.nlat; j++) {
      const sl = Math.sin(g.lat[j]!), cl = Math.cos(g.lat[j]!);
      for (let i = 0; i < nl; i++) {
        const so = Math.sin(g.lon[i]!), co = Math.cos(g.lon[i]!);
        for (let k = 0; k < K; k++) {
          const q = k * ng + j * nl + i, uu = u[q]!, vv = v[q]!;
          this.vx[q] = -so * uu - sl * co * vv;
          this.vy[q] = co * uu - sl * so * vv;
          this.vz[q] = cl * vv;
        }
      }
    }
    // departure points
    const sMin = g.sigma[0]!, sMax = g.sigma[K - 1]!;
    for (let j = 0; j < g.nlat; j++) {
      const sl = Math.sin(g.lat[j]!), cl = Math.cos(g.lat[j]!);
      for (let i = 0; i < nl; i++) {
        const px = cl * Math.cos(g.lon[i]!), py = cl * Math.sin(g.lon[i]!), pz = sl;
        for (let k = 0; k < K; k++) {
          const q = k * ng + j * nl + i;
          let mx = px, my = py, mz = pz, mlat = g.lat[j]!, mlon = g.lon[i]!, msig = g.sigma[k]!;
          let wx = this.vx[q]!, wy = this.vy[q]!, wz = this.vz[q]!, sd = sdot[q]!;
          for (let it = 0; it < 2; it++) {
            mx = px - 0.5 * dt * wx / a; my = py - 0.5 * dt * wy / a; mz = pz - 0.5 * dt * wz / a;
            const r = 1 / Math.hypot(mx, my, mz);
            mx *= r; my *= r; mz *= r;
            mlat = Math.asin(Math.max(-1, Math.min(1, mz)));
            mlon = Math.atan2(my, mx);
            msig = Math.max(sMin, Math.min(sMax, g.sigma[k]! - 0.5 * dt * sd));
            this.sampleWind(sdot, mlat, mlon, msig);
            wx = this.wout[0]!; wy = this.wout[1]!; wz = this.wout[2]!; sd = this.wout[3]!;
          }
          // reflect the arrival point through the midpoint along the great circle
          const d = px * mx + py * my + pz * mz;
          const dx = 2 * d * mx - px, dy = 2 * d * my - py, dz = 2 * d * mz - pz;
          this.depLat[q] = Math.asin(Math.max(-1, Math.min(1, dz)));
          this.depLon[q] = Math.atan2(dy, dx);
          this.depSig[q] = Math.max(sMin, Math.min(sMax, g.sigma[k]! - dt * sd));
        }
      }
    }
    for (let f = 0; f < src.length; f++) this.interpolate(src[f]!, dst[f]!);
  }

  /** Tricubic, quasi-monotone interpolation of f at the stored departure points. */
  private interpolate(f: Float64Array, out: Float64Array): void {
    const g = this.g, nl = g.nlon, ng = g.nlat * nl, K = g.K, S = g.sigma;
    const wl = new Float64Array(4), wy = new Float64Array(4), wz = new Float64Array(4);
    const rows = new Int32Array(4), shifts = new Int32Array(4);
    for (let q = 0; q < out.length; q++) {
      const lat = this.depLat[q]!, lon = this.depLon[q]!, sig = this.depSig[q]!;
      // longitude
      let x = (lon / (2 * Math.PI)) * nl;
      x -= Math.floor(x / nl) * nl;
      const i0 = Math.floor(x), t = x - i0;
      wl[0] = -t * (t - 1) * (t - 2) / 6; wl[1] = (t + 1) * (t - 1) * (t - 2) / 2;
      wl[2] = -(t + 1) * t * (t - 2) / 2; wl[3] = (t + 1) * t * (t - 1) / 6;
      // latitude (non-uniform nodes, extended across poles)
      const j = this.latInterval(lat);
      for (let a = 0; a < 4; a++) {
        const r = j - 1 + a;
        const ya = this.yExt[r + 2]!;
        let w = 1;
        for (let b = 0; b < 4; b++) if (b !== a) { const yb = this.yExt[j - 1 + b + 2]!; w *= (lat - yb) / (ya - yb); }
        wy[a] = w;
        const n = g.nlat;
        if (r < 0) { rows[a] = -r - 1; shifts[a] = nl >> 1; }
        else if (r >= n) { rows[a] = 2 * n - 1 - r; shifts[a] = nl >> 1; }
        else { rows[a] = r; shifts[a] = 0; }
      }
      // sigma
      const k = this.sigInterval(sig);
      let k0: number, nk: number;
      if (k >= 1 && k + 2 <= K - 1) {
        k0 = k - 1; nk = 4;
        for (let a = 0; a < 4; a++) {
          let w = 1;
          for (let b = 0; b < 4; b++) if (b !== a) w *= (sig - S[k0 + b]!) / (S[k0 + a]! - S[k0 + b]!);
          wz[a] = w;
        }
      } else {
        k0 = k; nk = 2;
        const w1 = Math.max(0, Math.min(1, (sig - S[k]!) / (S[k + 1]! - S[k]!)));
        wz[0] = 1 - w1; wz[1] = w1;
      }
      let acc = 0, lo = Infinity, hi = -Infinity;
      const inner = nk === 4 ? 1 : 0;
      for (let c = 0; c < nk; c++) {
        const o = (k0 + c) * ng;
        let accK = 0;
        for (let a = 0; a < 4; a++) {
          const ro = o + rows[a]! * nl, sh = shifts[a]!;
          let accR = 0;
          for (let b = 0; b < 4; b++) {
            const val = f[ro + (i0 - 1 + b + sh + nl) % nl]!;
            accR += wl[b]! * val;
            if ((a === 1 || a === 2) && (b === 1 || b === 2) && (c === inner || c === inner + 1)) {
              if (val < lo) lo = val;
              if (val > hi) hi = val;
            }
          }
          accK += wy[a]! * accR;
        }
        acc += wz[c]! * accK;
      }
      out[q] = acc < lo ? lo : acc > hi ? hi : acc;
    }
  }
}
