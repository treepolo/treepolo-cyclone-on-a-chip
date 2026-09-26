// Zonal-mean climate diagnostics: [u], [T], mean meridional mass streamfunction,
// eddy momentum / heat fluxes, time-averaged by an accumulator.

import { Planet } from '../core/constants.js';
import { GridState } from './dycore.js';

export interface ZonalMeanClimate {
  nlat: number;
  K: number;
  lat: number[];            // degrees, north -> south
  sigma: number[];
  sigmaHalf: number[];
  samples: number;
  u: number[];              // [k][lat]
  T: number[];
  v: number[];
  psi: number[];            // [khalf][lat] kg/s, khalf = 0..K
  uvEddy: number[];         // [u'v'] m^2/s^2
  vTEddy: number[];         // [v'T'] K m/s
  eke: number[];            // 0.5[u'^2+v'^2]
  ps: number[];             // [lat]
}

export class ZonalMeanAccumulator {
  private n = 0;
  private readonly u: Float64Array; private readonly T: Float64Array; private readonly v: Float64Array;
  private readonly vps: Float64Array; private readonly uv: Float64Array; private readonly vT: Float64Array;
  private readonly eke: Float64Array; private readonly ps: Float64Array;

  constructor(private readonly nlat: number, private readonly nlon: number, private readonly K: number) {
    const s = nlat * K;
    this.u = new Float64Array(s); this.T = new Float64Array(s); this.v = new Float64Array(s);
    this.vps = new Float64Array(s); this.uv = new Float64Array(s); this.vT = new Float64Array(s);
    this.eke = new Float64Array(s); this.ps = new Float64Array(nlat);
  }

  add(g: GridState): void {
    const { nlat, nlon, K } = this, ng = nlat * nlon, inv = 1 / nlon;
    for (let j = 0; j < nlat; j++) {
      let psm = 0;
      for (let i = 0; i < nlon; i++) psm += g.ps[j * nlon + i]!;
      this.ps[j] = this.ps[j]! + psm * inv;
      for (let k = 0; k < K; k++) {
        const o = k * ng + j * nlon;
        let um = 0, vm = 0, tm = 0, vp = 0;
        for (let i = 0; i < nlon; i++) { um += g.u[o + i]!; vm += g.v[o + i]!; tm += g.T[o + i]!; vp += g.v[o + i]! * g.ps[j * nlon + i]!; }
        um *= inv; vm *= inv; tm *= inv; vp *= inv;
        let uv = 0, vt = 0, ke = 0;
        for (let i = 0; i < nlon; i++) {
          const du = g.u[o + i]! - um, dv = g.v[o + i]! - vm, dt = g.T[o + i]! - tm;
          uv += du * dv; vt += dv * dt; ke += 0.5 * (du * du + dv * dv);
        }
        const q = k * nlat + j;
        this.u[q] = this.u[q]! + um; this.v[q] = this.v[q]! + vm; this.T[q] = this.T[q]! + tm;
        this.vps[q] = this.vps[q]! + vp;
        this.uv[q] = this.uv[q]! + uv * inv; this.vT[q] = this.vT[q]! + vt * inv; this.eke[q] = this.eke[q]! + ke * inv;
      }
    }
    this.n++;
  }

  get samples(): number { return this.n; }

  reset(): void {
    this.n = 0;
    for (const a of [this.u, this.T, this.v, this.vps, this.uv, this.vT, this.eke, this.ps]) a.fill(0);
  }

  result(lat: Float64Array, sigma: Float64Array, sigmaHalf: Float64Array, planet: Planet): ZonalMeanClimate {
    const { nlat, K } = this, n = Math.max(1, this.n);
    const avg = (a: Float64Array): number[] => Array.from(a, (x) => x / n);
    const psi = new Array<number>((K + 1) * nlat).fill(0);
    for (let j = 0; j < nlat; j++) {
      const c = 2 * Math.PI * planet.radius * Math.cos(lat[j]!) / planet.gravity;
      let acc = 0;
      for (let k = 0; k < K; k++) {
        acc += this.vps[k * nlat + j]! / n * (sigmaHalf[k + 1]! - sigmaHalf[k]!);
        psi[(k + 1) * nlat + j] = c * acc;
      }
    }
    return {
      nlat, K, samples: this.n,
      lat: Array.from(lat, (x) => x * 180 / Math.PI),
      sigma: Array.from(sigma), sigmaHalf: Array.from(sigmaHalf),
      u: avg(this.u), T: avg(this.T), v: avg(this.v), psi,
      uvEddy: avg(this.uv), vTEddy: avg(this.vT), eke: avg(this.eke), ps: avg(this.ps),
    };
  }
}

export interface CirculationSummary {
  jetMaxNH: { u: number; lat: number; sigma: number };
  jetMaxSH: { u: number; lat: number; sigma: number };
  surfaceU: { lat: number; u: number }[];
  /** Overturning cells along the streamfunction at the level of max |psi|, walking north -> south. */
  cells: { fromLat: number; toLat: number; peak: number; sigma: number }[];
  maxPsi: number;
  minPsi: number;
  eqPoleDeltaT: number;
}

/** Summarise the zonal-mean climate: jets, surface winds and overturning cells. */
export function summarize(c: ZonalMeanClimate): CirculationSummary {
  const { nlat, K } = c;
  let nh = { u: -Infinity, lat: 0, sigma: 0 }, sh = { u: -Infinity, lat: 0, sigma: 0 };
  for (let k = 0; k < K; k++) for (let j = 0; j < nlat; j++) {
    const u = c.u[k * nlat + j]!, lat = c.lat[j]!;
    if (lat > 0 && u > nh.u) nh = { u, lat, sigma: c.sigma[k]! };
    if (lat < 0 && u > sh.u) sh = { u, lat, sigma: c.sigma[k]! };
  }
  const surfaceU = c.lat.map((lat, j) => ({ lat, u: c.u[(K - 1) * nlat + j]! }));
  // Cells: sign segments of psi, taken at each latitude as the value of largest magnitude in the column.
  const colPsi: { v: number; sigma: number }[] = [];
  let maxPsi = -Infinity, minPsi = Infinity;
  for (let j = 0; j < nlat; j++) {
    let best = 0, bs = 0;
    for (let k = 1; k < K; k++) {
      const v = c.psi[k * nlat + j]!;
      if (Math.abs(v) > Math.abs(best)) { best = v; bs = c.sigmaHalf[k]!; }
      maxPsi = Math.max(maxPsi, v); minPsi = Math.min(minPsi, v);
    }
    colPsi.push({ v: best, sigma: bs });
  }
  const cells: CirculationSummary['cells'] = [];
  let start = 0;
  for (let j = 1; j <= nlat; j++) {
    if (j === nlat || Math.sign(colPsi[j]!.v) !== Math.sign(colPsi[start]!.v)) {
      let peak = 0, ps = 0;
      for (let q = start; q < j; q++) if (Math.abs(colPsi[q]!.v) > Math.abs(peak)) { peak = colPsi[q]!.v; ps = colPsi[q]!.sigma; }
      cells.push({ fromLat: c.lat[start]!, toLat: c.lat[j - 1]!, peak, sigma: ps });
      start = j;
    }
  }
  // equator-to-pole surface temperature difference (lowest level)
  const tEq = (c.T[(K - 1) * nlat + nlat / 2 - 1]! + c.T[(K - 1) * nlat + nlat / 2]!) / 2;
  const tPole = (c.T[(K - 1) * nlat]! + c.T[(K - 1) * nlat + nlat - 1]!) / 2;
  return { jetMaxNH: nh, jetMaxSH: sh, surfaceU, cells, maxPsi, minPsi, eqPoleDeltaT: tEq - tPole };
}
