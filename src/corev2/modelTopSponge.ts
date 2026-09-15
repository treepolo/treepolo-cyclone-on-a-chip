import { cellCountOf, type ConservativeFields } from './fields.js';
import type { SphericalShellGeometry } from './sphericalShellGeometry.js';

const PACKED = 8;

export interface ModelTopSpongeConfig {
  startFraction: number;
  maxRate: number;
}

/** Thin radial-velocity absorber used only near the rigid model lid. */
export const CORE_V2_MODEL_TOP_SPONGE: ModelTopSpongeConfig = {
  startFraction: 0.75,
  maxRate: 0.2,
};

export function coreV2ModelTopSpongeRate(
  height: number,
  topHeight: number,
  config: ModelTopSpongeConfig = CORE_V2_MODEL_TOP_SPONGE,
): number {
  if (!(topHeight > 0)) return 0;
  const start = config.startFraction * topHeight;
  if (height <= start) return 0;
  const s = Math.max(0, Math.min(1, (height - start) / (topHeight - start)));
  const ramp = Math.sin(0.5 * Math.PI * s);
  return config.maxRate * ramp * ramp;
}

function dampCell(
  rho: number,
  mx: number,
  my: number,
  mz: number,
  rhoE: number,
  rx: number,
  ry: number,
  rz: number,
  rate: number,
  dt: number,
): [number, number, number, number] {
  if (rate <= 0 || dt === 0) return [mx, my, mz, rhoE];
  const radialMomentum = mx * rx + my * ry + mz * rz;
  const nextRadial = radialMomentum / (1 + rate * dt);
  const delta = nextRadial - radialMomentum;
  const nextMx = mx + delta * rx;
  const nextMy = my + delta * ry;
  const nextMz = mz + delta * rz;
  const oldKinetic = 0.5 * (mx * mx + my * my + mz * mz) / rho;
  const newKinetic = 0.5 * (
    nextMx * nextMx + nextMy * nextMy + nextMz * nextMz
  ) / rho;
  return [nextMx, nextMy, nextMz, rhoE + newKinetic - oldKinetic];
}

/**
 * Absorb upward/downward wave reflection at the rigid model lid by damping only
 * radial momentum in the upper quarter of the domain. Density and thermodynamic
 * internal energy are unchanged; removed radial kinetic energy leaves rhoE.
 */
export function applyCoreV2ModelTopSponge(
  fields: ConservativeFields,
  dt: number,
  geometry: SphericalShellGeometry,
  config: ModelTopSpongeConfig = CORE_V2_MODEL_TOP_SPONGE,
): void {
  const expected = geometry.horizontal.cellCount * geometry.nz;
  if (cellCountOf(fields) !== expected) {
    throw new Error('Core v2 model-top sponge state/geometry size mismatch');
  }
  if (!(dt >= 0) || !Number.isFinite(dt)) {
    throw new Error(`invalid Core v2 model-top sponge dt: ${dt}`);
  }
  const surface = geometry.radiusInterface[0]!;
  const top = geometry.radiusInterface[geometry.nz]! - surface;
  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    const rx = geometry.horizontal.cellCenters[c * 3]!;
    const ry = geometry.horizontal.cellCenters[c * 3 + 1]!;
    const rz = geometry.horizontal.cellCenters[c * 3 + 2]!;
    for (let k = 0; k < geometry.nz; k++) {
      const center = 0.5 * (
        geometry.radiusInterface[k]! + geometry.radiusInterface[k + 1]!
      ) - surface;
      const rate = coreV2ModelTopSpongeRate(center, top, config);
      if (rate <= 0) continue;
      const q = c * geometry.nz + k;
      const next = dampCell(
        fields.rho[q]!, fields.momX[q]!, fields.momY[q]!, fields.momZ[q]!, fields.rhoE[q]!,
        rx, ry, rz, rate, dt,
      );
      fields.momX[q] = next[0];
      fields.momY[q] = next[1];
      fields.momZ[q] = next[2];
      fields.rhoE[q] = next[3];
    }
  }
}

export function applyCoreV2ModelTopSpongePackedF32(
  packedState: Float32Array,
  dt: number,
  geometry: SphericalShellGeometry,
  config: ModelTopSpongeConfig = CORE_V2_MODEL_TOP_SPONGE,
): void {
  const cells = packedState.length / PACKED;
  const expected = geometry.horizontal.cellCount * geometry.nz;
  if (!Number.isInteger(cells) || cells !== expected) {
    throw new Error('Core v2 packed model-top sponge state/geometry size mismatch');
  }
  if (!(dt >= 0) || !Number.isFinite(dt)) {
    throw new Error(`invalid Core v2 packed model-top sponge dt: ${dt}`);
  }
  const surface = geometry.radiusInterface[0]!;
  const top = geometry.radiusInterface[geometry.nz]! - surface;
  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    const rx = geometry.horizontal.cellCenters[c * 3]!;
    const ry = geometry.horizontal.cellCenters[c * 3 + 1]!;
    const rz = geometry.horizontal.cellCenters[c * 3 + 2]!;
    for (let k = 0; k < geometry.nz; k++) {
      const center = 0.5 * (
        geometry.radiusInterface[k]! + geometry.radiusInterface[k + 1]!
      ) - surface;
      const rate = coreV2ModelTopSpongeRate(center, top, config);
      if (rate <= 0) continue;
      const i = (c * geometry.nz + k) * PACKED;
      const next = dampCell(
        packedState[i]!, packedState[i + 1]!, packedState[i + 2]!, packedState[i + 3]!, packedState[i + 4]!,
        rx, ry, rz, rate, dt,
      );
      packedState[i + 1] = Math.fround(next[0]);
      packedState[i + 2] = Math.fround(next[1]);
      packedState[i + 3] = Math.fround(next[2]);
      packedState[i + 4] = Math.fround(next[3]);
    }
  }
}
