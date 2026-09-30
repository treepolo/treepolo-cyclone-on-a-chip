// Light and clear-air volumes for the pictures of the regional model (3-D view, satellite images), computed on the CPU
// from the display bytes (display.ts: extinction = EXT_MAX * byte^3): per voxel the transmittance toward the sun and
// straight up, and an occupancy grid of blocks that hold no cloud or precipitation at all. Box units: any unit in which
// the box is (bx, by, bz); the 3-D view exaggerates bz, the satellite pictures use the true shape.
import { EXT_MAX } from '../../regional/display.js';

export type V3 = [number, number, number];
/** a cut plane in box units: nothing where dot(n, p) > d */
export interface Plane { n: V3; d: number }
/** the eye nest's cloud bytes ([k][j][i], nx x nx x nz) and its box (lower-left corner and width, box units) */
export interface NestLightGeom { nx: number; nz: number; cloud: Uint8Array; x0: number; y0: number; L: number }

/**
 * Sun and sky light of every voxel (bytes: transmittance to the sun, transmittance straight up) from the cloud
 * extinction times k: one sweep down the levels toward the sun (the light of the level above, shifted toward the sun,
 * times this layer's transmittance) and one straight down. Optical depths in the metres of a domain `top` high.
 * With keepOd, also the optical depth toward the sun of every voxel ([k][j][i]; the nest's sweep leaves its box into it).
 */
export function sunSkyLight(nx: number, ny: number, nz: number, cloudIn: Uint8Array, box: V3, top: number, sun: V3, k: number, cut: Plane | null, keepOd: boolean): { out: Uint8Array; od: Float32Array | null } {
  const np = nx * ny, [bx, by, bz] = box;
  // a cutaway: the removed part casts no shadow
  let cloud = cloudIn;
  if (cut) {
    cloud = Uint8Array.from(cloud);
    for (let kk = 0; kk < nz; kk++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      if (cut.n[0] * (i + 0.5) / nx * bx + cut.n[1] * (j + 0.5) / ny * by + cut.n[2] * (kk + 0.5) / nz * bz > cut.d) cloud[(kk * ny + j) * nx + i] = 0;
    }
  }
  const lut = new Float32Array(256); for (let v = 0; v < 256; v++) lut[v] = EXT_MAX * (v / 255) ** 3 * k;
  const dzm = top / nz, s = sun;
  // horizontal shift toward the sun per level (cells) and the slant path per level (m)
  const di = (bz / nz) * s[0] / s[2] / (bx / nx), dj = (bz / nz) * s[1] / s[2] / (by / ny), slant = dzm / s[2];
  let od = new Float32Array(np), prev = new Float32Array(np);
  const sky = new Float32Array(np), out = new Uint8Array(2 * nz * np), all = keepOd ? new Float32Array(nz * np) : null;
  const put = (kk: number): void => {
    const o = kk * np;
    for (let c = 0; c < np; c++) { out[2 * (o + c)] = Math.round(255 * Math.exp(-od[c]!)); out[2 * (o + c) + 1] = Math.round(255 * Math.exp(-sky[c]!)); }
    all?.set(od, o);
  };
  { const o = (nz - 1) * np; for (let c = 0; c < np; c++) { const b = lut[cloud[o + c]!]!; od[c] = 0.5 * b * slant; sky[c] = 0.5 * b * dzm; } put(nz - 1); }
  for (let kk = nz - 2; kk >= 0; kk--) {
    [od, prev] = [prev, od];
    const o = kk * np, oa = (kk + 1) * np;
    for (let j = 0; j < ny; j++) {
      const y = Math.max(0, Math.min(ny - 1, j + dj)), j0 = Math.min(ny - 2, Math.floor(y)), fy = ny > 1 ? y - j0 : 0;
      for (let i = 0; i < nx; i++) {
        const x = Math.max(0, Math.min(nx - 1, i + di)), i0 = Math.min(nx - 2, Math.floor(x)), fx = nx > 1 ? x - i0 : 0;
        const a = j0 * nx + i0, above = (prev[a]! * (1 - fx) + prev[a + 1]! * fx) * (1 - fy) + (prev[a + nx]! * (1 - fx) + prev[a + nx + 1]! * fx) * fy;
        const c = j * nx + i, b = 0.5 * (lut[cloud[o + c]!]! + lut[cloud[oa + c]!]!);
        od[c] = above + b * slant; sky[c] = sky[c]! + b * dzm;
      }
    }
    put(kk);
  }
  return { out, od: all };
}

/**
 * The eye nest's sun and sky light, as sunSkyLight on its finer grid: the sweep toward the sun runs through the nest's
 * own cloud while the path stays in its box and takes the outer grid's optical depth (odAll of the outer sweep) where it
 * leaves the box; the sky light is its own column (same top).
 */
export function nestSunSkyLight(n: NestLightGeom, outer: { nx: number; ny: number; nz: number; top: number; od: Float32Array }, box: V3, sun: V3, k: number, cut: Plane | null): Uint8Array {
  const { nx: cn, nz: cz } = n, np = cn * cn, [bx, by, bz] = box, s = sun, all = outer.od;
  let cloud = n.cloud;
  if (cut) {
    cloud = Uint8Array.from(cloud);
    for (let kk = 0; kk < cz; kk++) for (let j = 0; j < cn; j++) for (let i = 0; i < cn; i++) {
      if (cut.n[0] * (n.x0 + (i + 0.5) / cn * n.L) + cut.n[1] * (n.y0 + (j + 0.5) / cn * n.L) + cut.n[2] * (kk + 0.5) / cz * bz > cut.d) cloud[(kk * cn + j) * cn + i] = 0;
    }
  }
  const lut = new Float32Array(256); for (let v = 0; v < 256; v++) lut[v] = EXT_MAX * (v / 255) ** 3 * k;
  const dzm = outer.top / cz, cell = n.L / cn, di = (bz / cz) * s[0] / s[2] / cell, dj = (bz / cz) * s[1] / s[2] / cell, slant = dzm / s[2];
  // the outer grid's optical depth at box point (x, y) and height fraction zf (bilinear across, linear between levels)
  const onx = outer.nx, ony = outer.ny, onz = outer.nz, onp = onx * ony;
  const outerAt = (x: number, y: number, zf: number): number => {
    const fx = Math.max(0, Math.min(onx - 1, x / bx * onx - 0.5)), fy = Math.max(0, Math.min(ony - 1, y / by * ony - 0.5)), fz = Math.max(0, Math.min(onz - 1, zf * onz - 0.5));
    const i0 = Math.min(onx - 2, Math.floor(fx)), j0 = Math.min(ony - 2, Math.floor(fy)), k0 = Math.min(onz - 2, Math.floor(fz));
    const wx = onx > 1 ? fx - i0 : 0, wy = ony > 1 ? fy - j0 : 0, wz = onz > 1 ? fz - k0 : 0;
    const at = (kk: number): number => {
      const o = Math.max(0, kk) * onp + Math.max(0, j0) * onx + Math.max(0, i0);
      return (all[o]! * (1 - wx) + all[o + (onx > 1 ? 1 : 0)]! * wx) * (1 - wy) + (all[o + (ony > 1 ? onx : 0)]! * (1 - wx) + all[o + (ony > 1 ? onx : 0) + (onx > 1 ? 1 : 0)]! * wx) * wy;
    };
    return onz > 1 ? at(k0) * (1 - wz) + at(k0 + 1) * wz : at(0);
  };
  let od = new Float32Array(np), prev = new Float32Array(np);
  const sky = new Float32Array(np), out = new Uint8Array(2 * cz * np);
  const put = (kk: number): void => { const o = kk * np; for (let c = 0; c < np; c++) { out[2 * (o + c)] = Math.round(255 * Math.exp(-od[c]!)); out[2 * (o + c) + 1] = Math.round(255 * Math.exp(-sky[c]!)); } };
  { const o = (cz - 1) * np; for (let c = 0; c < np; c++) { const b = lut[cloud[o + c]!]!; od[c] = 0.5 * b * slant; sky[c] = 0.5 * b * dzm; } put(cz - 1); }
  for (let kk = cz - 2; kk >= 0; kk--) {
    [od, prev] = [prev, od];
    const o = kk * np, oa = (kk + 1) * np, zfa = (kk + 1.5) / cz;
    for (let j = 0; j < cn; j++) {
      const y = j + dj, inY = y >= 0 && y <= cn - 1, j0 = Math.min(cn - 2, Math.max(0, Math.floor(y))), fy = y - j0;
      for (let i = 0; i < cn; i++) {
        const x = i + di;
        let above: number;
        if (inY && x >= 0 && x <= cn - 1) {
          const i0 = Math.min(cn - 2, Math.max(0, Math.floor(x))), fx = x - i0, a = j0 * cn + i0;
          above = (prev[a]! * (1 - fx) + prev[a + 1]! * fx) * (1 - fy) + (prev[a + cn]! * (1 - fx) + prev[a + cn + 1]! * fx) * fy;
        } else above = outerAt(n.x0 + (x + 0.5) * cell, n.y0 + (y + 0.5) * cell, zfa);
        const c = j * cn + i, b = 0.5 * (lut[cloud[o + c]!]! + lut[cloud[oa + c]!]!);
        od[c] = above + b * slant; sky[c] = sky[c]! + b * dzm;
      }
    }
    put(kk);
  }
  return out;
}

/**
 * Occupancy grid for skipping clear air: blocks of B x B cells across and Bz levels, 255 where the block or any of its
 * 26 neighbours holds any cloud or second-channel value (outer grid or nest), 0 elsewhere. A ray in an empty block may
 * step up to a block length without passing any cloud (the trilinear sampling reaches one cell into a neighbour, which
 * is empty as well). The nest's box is given in metres of a domain Lx x Ly.
 */
export function occupancyGrid(nx: number, ny: number, nz: number, cloud: Uint8Array, rain: Uint8Array, B: number, Bz: number,
  nest: { nx: number; nz: number; cloud: Uint8Array; rain: Uint8Array; x0: number; y0: number; L: number; Lx: number; Ly: number } | null): { occ: Uint8Array; mx: number; my: number; mz: number } {
  const mx = Math.ceil(nx / B), my = Math.ceil(ny / B), mz = Math.ceil(nz / Bz), raw = new Uint8Array(mx * my * mz);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) {
    const o = (k * ny + j) * nx, ob = (Math.floor(k / Bz) * my + Math.floor(j / B)) * mx;
    for (let i = 0; i < nx; i++) if (cloud[o + i] || rain[o + i]) raw[ob + Math.floor(i / B)] = 1;
  }
  if (nest) {
    const cn = nest.nx, cz = nest.nz;
    for (let k = 0; k < cz; k++) {
      const kb = Math.min(mz - 1, Math.floor(((k + 0.5) / cz * nz) / Bz));
      for (let j = 0; j < cn; j++) {
        const jb = Math.max(0, Math.min(my - 1, Math.floor(((nest.y0 + (j + 0.5) / cn * nest.L) / nest.Ly * ny) / B))), o = (k * cn + j) * cn;
        for (let i = 0; i < cn; i++) if (nest.cloud[o + i] || nest.rain[o + i]) {
          const ib = Math.max(0, Math.min(mx - 1, Math.floor(((nest.x0 + (i + 0.5) / cn * nest.L) / nest.Lx * nx) / B)));
          raw[(kb * my + jb) * mx + ib] = 1;
        }
      }
    }
  }
  const occ = new Uint8Array(mx * my * mz);
  for (let k = 0; k < mz; k++) for (let j = 0; j < my; j++) for (let i = 0; i < mx; i++) {
    let any = 0;
    for (let c = Math.max(0, k - 1); c <= Math.min(mz - 1, k + 1) && !any; c++) for (let b = Math.max(0, j - 1); b <= Math.min(my - 1, j + 1) && !any; b++)
      for (let a = Math.max(0, i - 1); a <= Math.min(mx - 1, i + 1); a++) if (raw[(c * my + b) * mx + a]) { any = 1; break; }
    occ[(k * my + j) * mx + i] = any ? 255 : 0;
  }
  return { occ, mx, my, mz };
}
