import type { Vec3 } from '../core/math.js';
import {
  cellCountOf,
  createConservativeFields,
  type ConservativeFields,
} from './fields.js';

export interface FaceFlux {
  /** Mass flux density through the oriented face [kg m^-2 s^-1]. */
  mass: number;
  /** Cartesian momentum flux density [N m^-2]. */
  momentum: Vec3;
  /** Total-energy flux density [W m^-2]. */
  totalEnergy: number;
}

export interface IntegratedFaceFlux {
  /** Face-integrated mass rate [kg s^-1]. */
  mass: number;
  /** Face-integrated Cartesian momentum rate [N]. */
  momentum: Vec3;
  /** Face-integrated total-energy rate [W]. */
  totalEnergy: number;
}

/**
 * Stores cell-integrated rates, not density tendencies.
 * Each internal face is accumulated once and with exactly opposite signs into
 * its two adjacent cells. This is the non-negotiable conservation invariant
 * for Core v2.
 */
export interface IntegratedConservativeRate extends ConservativeFields {}

export function createIntegratedRate(cellCount: number): IntegratedConservativeRate {
  return createConservativeFields(cellCount, 'f64');
}

export function zeroIntegratedRate(rate: IntegratedConservativeRate): void {
  rate.rho.fill(0);
  rate.momX.fill(0);
  rate.momY.fill(0);
  rate.momZ.fill(0);
  rate.rhoE.fill(0);
}

export function accumulateInternalIntegratedFaceFlux(
  rate: IntegratedConservativeRate,
  leftCell: number,
  rightCell: number,
  flux: IntegratedFaceFlux,
): void {
  const n = cellCountOf(rate);
  if (
    leftCell < 0 ||
    leftCell >= n ||
    rightCell < 0 ||
    rightCell >= n ||
    leftCell === rightCell
  ) {
    throw new Error(`invalid internal face cells: ${leftCell}, ${rightCell}`);
  }

  rate.rho[leftCell] = rate.rho[leftCell]! - flux.mass;
  rate.rho[rightCell] = rate.rho[rightCell]! + flux.mass;

  rate.momX[leftCell] = rate.momX[leftCell]! - flux.momentum[0];
  rate.momX[rightCell] = rate.momX[rightCell]! + flux.momentum[0];
  rate.momY[leftCell] = rate.momY[leftCell]! - flux.momentum[1];
  rate.momY[rightCell] = rate.momY[rightCell]! + flux.momentum[1];
  rate.momZ[leftCell] = rate.momZ[leftCell]! - flux.momentum[2];
  rate.momZ[rightCell] = rate.momZ[rightCell]! + flux.momentum[2];

  rate.rhoE[leftCell] = rate.rhoE[leftCell]! - flux.totalEnergy;
  rate.rhoE[rightCell] = rate.rhoE[rightCell]! + flux.totalEnergy;
}

export function accumulateInternalFaceFlux(
  rate: IntegratedConservativeRate,
  leftCell: number,
  rightCell: number,
  area: number,
  flux: FaceFlux,
): void {
  if (!(area > 0) || !Number.isFinite(area)) {
    throw new Error(`invalid face area: ${area}`);
  }
  accumulateInternalIntegratedFaceFlux(rate, leftCell, rightCell, {
    mass: area * flux.mass,
    momentum: [
      area * flux.momentum[0],
      area * flux.momentum[1],
      area * flux.momentum[2],
    ],
    totalEnergy: area * flux.totalEnergy,
  });
}

export function advanceFromIntegratedRate(
  fields: ConservativeFields,
  rate: IntegratedConservativeRate,
  cellVolumes: ArrayLike<number>,
  dt: number,
): void {
  const n = cellCountOf(fields);
  if (cellCountOf(rate) !== n || cellVolumes.length !== n) {
    throw new Error('Core v2 update arrays have inconsistent lengths');
  }
  if (!(dt >= 0) || !Number.isFinite(dt)) {
    throw new Error(`invalid timestep: ${dt}`);
  }

  for (let c = 0; c < n; c++) {
    const volume = cellVolumes[c]!;
    if (!(volume > 0) || !Number.isFinite(volume)) {
      throw new Error(`invalid cell volume at ${c}: ${volume}`);
    }
    const scale = dt / volume;
    fields.rho[c] = fields.rho[c]! + scale * rate.rho[c]!;
    fields.momX[c] = fields.momX[c]! + scale * rate.momX[c]!;
    fields.momY[c] = fields.momY[c]! + scale * rate.momY[c]!;
    fields.momZ[c] = fields.momZ[c]! + scale * rate.momZ[c]!;
    fields.rhoE[c] = fields.rhoE[c]! + scale * rate.rhoE[c]!;
  }
}

export interface IntegratedTotals {
  mass: number;
  momentum: Vec3;
  totalEnergy: number;
}

function kahanWeightedSum(values: ArrayLike<number>, weights: ArrayLike<number>): number {
  let sum = 0;
  let compensation = 0;
  for (let i = 0; i < values.length; i++) {
    const y = values[i]! * weights[i]! - compensation;
    const t = sum + y;
    compensation = (t - sum) - y;
    sum = t;
  }
  return sum;
}

export function integratedTotals(
  fields: ConservativeFields,
  cellVolumes: ArrayLike<number>,
): IntegratedTotals {
  const n = cellCountOf(fields);
  if (cellVolumes.length !== n) {
    throw new Error('Core v2 diagnostic arrays have inconsistent lengths');
  }
  for (let c = 0; c < n; c++) {
    if (!(cellVolumes[c]! > 0) || !Number.isFinite(cellVolumes[c]!)) {
      throw new Error(`invalid cell volume at ${c}: ${cellVolumes[c]}`);
    }
  }
  return {
    mass: kahanWeightedSum(fields.rho, cellVolumes),
    momentum: [
      kahanWeightedSum(fields.momX, cellVolumes),
      kahanWeightedSum(fields.momY, cellVolumes),
      kahanWeightedSum(fields.momZ, cellVolumes),
    ],
    totalEnergy: kahanWeightedSum(fields.rhoE, cellVolumes),
  };
}
