// Spun-up Earth states: save one model state and start other runs from it, at the same or a different
// spectral truncation, so that a session begins with developed weather (cyclones, fronts, monsoon,
// tropical disturbances) instead of the zonally symmetric initial state.
//
// File (little-endian float32): 8-value header [1 (version), trunc, nlat, nlon, K, time (s), steps, 0],
// Gaussian latitudes (nlat), the current spectral state (vor, div, T per level, then ln ps; re then im),
// q [k][lat][lon], surface geopotential, SST / skin temperature, bucket water and sea ice on the grid.
//
// Loading at another truncation: spectral coefficients are copied for n <= min(T_file, T_model) and
// zero above (the synoptic scales carry over; the finer scales develop within a day or two); grid
// fields are interpolated bilinearly; ln ps is corrected hydrostatically for the different orography,
// d ln ps = -d phis / (R T_lowest). The leapfrog restarts with old = current (Robert filter damps the
// computational mode).

import type { Dycore } from './dycore.js';
import type { GrayPhysics } from './moist/aquaplanet.js';
import type { SpectralField } from '../spectral/transform.js';
import { DRY_AIR } from '../core/constants.js';

export function encodeSpinup(model: Dycore, physics: GrayPhysics): ArrayBuffer {
  const tr = model.tr, K = model.K, ng = model.ng, ns = tr.nspec;
  const s = model.exportState(), half = (s.data.length - (model.moist ? K * ng : 0)) / 2;
  const n = 8 + tr.nlat + half + K * ng + 4 * ng;
  const out = new Float32Array(n);
  out.set([1, tr.trunc, tr.nlat, tr.nlon, K, model.time, model.steps, 0]);
  let o = 8;
  out.set(tr.lat, o); o += tr.nlat;
  out.set(s.data.subarray(half, 2 * half), o); o += half;               // current time level
  if (half !== (3 * K + 1) * 2 * ns) throw new Error('unexpected state layout');
  out.set(s.data.subarray(2 * half), o); o += K * ng;                   // q
  const phis = new Float64Array(ng); model.surfaceGeopotentialGrid(phis);
  out.set(phis, o); o += ng;
  out.set(physics.f.sst, o); o += ng;
  out.set(physics.f.bucket, o); o += ng;
  out.set(physics.f.ice, o);
  return out.buffer;
}

/** Bilinear interpolation between Gaussian grids (lat north -> south, lon from 0, periodic). */
function regrid(src: ArrayLike<number>, lat: ArrayLike<number>, nlon: number, dstLat: ArrayLike<number>, dstNlon: number, out: Float64Array, off = 0, dOff = 0): void {
  const nlat = lat.length;
  for (let j = 0; j < dstLat.length; j++) {
    const la = dstLat[j]!;
    let a = 0;
    while (a < nlat - 2 && lat[a + 1]! > la) a++;
    const wy = Math.max(0, Math.min(1, (lat[a]! - la) / (lat[a]! - lat[a + 1]!)));
    for (let i = 0; i < dstNlon; i++) {
      const x = i / dstNlon * nlon, i0 = Math.floor(x) % nlon, i1 = (i0 + 1) % nlon, wx = x - Math.floor(x);
      const r0 = off + a * nlon, r1 = off + (a + 1) * nlon;
      out[dOff + j * dstNlon + i] = (1 - wy) * ((1 - wx) * src[r0 + i0]! + wx * src[r0 + i1]!) + wy * ((1 - wx) * src[r1 + i0]! + wx * src[r1 + i1]!);
    }
  }
}

export function applySpinup(model: Dycore, physics: GrayPhysics, buf: ArrayBuffer): void {
  const f = new Float32Array(buf);
  if (f[0] !== 1) throw new Error('unknown spin-up file version');
  const T0 = f[1]!, nlat0 = f[2]!, nlon0 = f[3]!, K = f[4]!, time = f[5]!, steps = f[6]!;
  if (K !== model.K) throw new Error(`spin-up file has ${K} levels, model has ${model.K}`);
  const tr = model.tr, ng = model.ng, ng0 = nlat0 * nlon0, ns0 = (T0 + 1) * (T0 + 2) / 2, ns = tr.nspec;
  let o = 8;
  const lat0 = f.subarray(o, o + nlat0); o += nlat0;
  const spec0 = f.subarray(o, o + (3 * K + 1) * 2 * ns0); o += (3 * K + 1) * 2 * ns0;
  const q0 = f.subarray(o, o + K * ng0); o += K * ng0;
  const phis0 = f.subarray(o, o + ng0); o += ng0;
  const sst0 = f.subarray(o, o + ng0); o += ng0;
  const bucket0 = f.subarray(o, o + ng0); o += ng0;
  const ice0 = f.subarray(o, o + ng0);
  // spectral copy by (m, n)
  const mStart0 = (m: number): number => m * (T0 + 1) - m * (m - 1) / 2;
  const fields: SpectralField[] = [];
  for (let r = 0; r < 3 * K + 1; r++) {
    const fld: SpectralField = tr.newSpec();
    const base = r * 2 * ns0;
    for (let s = 0; s < ns; s++) {
      const m = tr.mOf[s]!, n = tr.nOf[s]!;
      if (n > T0) continue;
      const s0 = mStart0(m) + n - m;
      fld.re[s] = spec0[base + s0]!; fld.im[s] = spec0[base + ns0 + s0]!;
    }
    fields.push(fld);
  }
  // ln ps: hydrostatic correction for the model's own orography
  const lnps = fields[3 * K]!, g = new Float64Array(ng), phisM = new Float64Array(ng), phisI = new Float64Array(ng), tLow = new Float64Array(ng);
  tr.synth(lnps, g);
  model.surfaceGeopotentialGrid(phisM);
  regrid(phis0, lat0, nlon0, tr.lat, tr.nlon, phisI);
  tr.synth(fields[3 * K - 1]!, tLow);                                    // lowest-level temperature
  for (let p = 0; p < ng; p++) g[p] = g[p]! - (phisM[p]! - phisI[p]!) / (DRY_AIR.rd * Math.max(200, tLow[p]!));
  tr.anal(g, lnps);
  // assemble the model's state layout: (old, current) x (vor[K], div[K], T[K], ln ps), then q
  const cur = new Float64Array((3 * K + 1) * 2 * ns);
  fields.forEach((fld, r) => { cur.set(fld.re, r * 2 * ns); cur.set(fld.im, r * 2 * ns + ns); });
  const q = new Float64Array(K * ng);
  for (let k = 0; k < K; k++) regrid(q0, lat0, nlon0, tr.lat, tr.nlon, q, k * ng0, k * ng);
  for (let i = 0; i < q.length; i++) q[i] = Math.max(0, q[i]!);
  const data = new Float64Array(2 * cur.length + (model.moist ? q.length : 0));
  data.set(cur, 0); data.set(cur, cur.length);
  if (model.moist) data.set(q, 2 * cur.length);
  model.importState({ time, steps, data });
  model.massTarget = model.meanSurfacePressure();
  // surface: regrid, then keep land / ocean values consistent with the model's own mask
  const tmp = new Float64Array(ng);
  regrid(sst0, lat0, nlon0, tr.lat, tr.nlon, tmp); physics.f.sst.set(tmp);
  regrid(bucket0, lat0, nlon0, tr.lat, tr.nlon, tmp);
  for (let p = 0; p < ng; p++) physics.f.bucket[p] = physics.surface.land[p] ? Math.min(physics.cfg.bucketMax, Math.max(0, tmp[p]!)) : 0;
  regrid(ice0, lat0, nlon0, tr.lat, tr.nlon, tmp);
  for (let p = 0; p < ng; p++) physics.f.ice[p] = physics.surface.land[p] ? 0 : Math.max(0, tmp[p]!);
}
