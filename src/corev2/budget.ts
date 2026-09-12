import { DRY_AIR, type AtmosphereConfig } from '../core/constants.js';
import type { ConservativeFields } from './fields.js';
import type { HydrostaticReference1D } from './hydrostaticReference.js';
import {
  kineticEnergyDensity,
  pressureFromConservedRelativeToReference,
  type ConservativeCell,
} from './state.js';
import {
  shellCellIndex,
  type SphericalShellGeometry,
} from './sphericalShellGeometry.js';

export interface ClosedSystemBudget {
  mass: number;
  internalEnergy: number;
  kineticEnergy: number;
  potentialEnergy: number;
  totalEnergy: number;
  /** totalEnergy - (internal + kinetic + potential) [J]. */
  energyClosureResidual: number;
  minDensity: number;
  minPressure: number;
  maxSpeed: number;
}

class KahanAccumulator {
  private sumValue = 0;
  private compensation = 0;

  add(value: number): void {
    const y = value - this.compensation;
    const t = this.sumValue + y;
    this.compensation = (t - this.sumValue) - y;
    this.sumValue = t;
  }

  value(): number {
    return this.sumValue;
  }
}

/**
 * Diagnose the complete energy ledger of the closed dry Core v2 state.
 *
 * rhoE is prognostic and already contains internal + 3-D kinetic + gravitational
 * potential energy. The three components are recomputed independently here so
 * every closed-system test can check both conservation of rhoE and closure of
 * its physical decomposition. No residual is hidden in an unnamed numerical
 * energy bucket.
 */
export function closedSystemBudget(
  fields: ConservativeFields,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  gas: AtmosphereConfig = DRY_AIR,
): ClosedSystemBudget {
  const cellCount = geometry.horizontal.cellCount * geometry.nz;
  if (
    fields.rho.length !== cellCount ||
    fields.momX.length !== cellCount ||
    fields.momY.length !== cellCount ||
    fields.momZ.length !== cellCount ||
    fields.rhoE.length !== cellCount
  ) {
    throw new Error('Core v2 budget field/geometry size mismatch');
  }
  if (
    reference.cellDensity.length !== geometry.nz ||
    reference.cellPressure.length !== geometry.nz ||
    reference.cellGeopotential.length !== geometry.nz
  ) {
    throw new Error('Core v2 budget reference/geometry size mismatch');
  }

  const mass = new KahanAccumulator();
  const internal = new KahanAccumulator();
  const kinetic = new KahanAccumulator();
  const potential = new KahanAccumulator();
  const total = new KahanAccumulator();
  let minDensity = Number.POSITIVE_INFINITY;
  let minPressure = Number.POSITIVE_INFINITY;
  let maxSpeed = 0;

  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const volume = geometry.cellVolume[q]!;
      if (!(volume > 0) || !Number.isFinite(volume)) {
        throw new Error(`invalid Core v2 budget cell volume at ${q}: ${volume}`);
      }
      const state: ConservativeCell = {
        rho: fields.rho[q]!,
        momentum: [fields.momX[q]!, fields.momY[q]!, fields.momZ[q]!],
        rhoE: fields.rhoE[q]!,
      };
      if (!(state.rho > 0) || !Number.isFinite(state.rho)) {
        throw new Error(`invalid Core v2 budget density at ${q}: ${state.rho}`);
      }
      const pressure = pressureFromConservedRelativeToReference(
        state,
        reference.cellDensity[k]!,
        reference.cellPressure[k]!,
        reference.cellGeopotential[k]!,
        gas,
      );
      const kineticDensity = kineticEnergyDensity(state);
      const internalDensity = pressure / (gas.gamma - 1);
      const potentialDensity = state.rho * reference.cellGeopotential[k]!;
      const speed = Math.hypot(
        state.momentum[0],
        state.momentum[1],
        state.momentum[2],
      ) / state.rho;

      mass.add(state.rho * volume);
      internal.add(internalDensity * volume);
      kinetic.add(kineticDensity * volume);
      potential.add(potentialDensity * volume);
      total.add(state.rhoE * volume);
      minDensity = Math.min(minDensity, state.rho);
      minPressure = Math.min(minPressure, pressure);
      maxSpeed = Math.max(maxSpeed, speed);
    }
  }

  const internalEnergy = internal.value();
  const kineticEnergy = kinetic.value();
  const potentialEnergy = potential.value();
  const totalEnergy = total.value();
  return {
    mass: mass.value(),
    internalEnergy,
    kineticEnergy,
    potentialEnergy,
    totalEnergy,
    energyClosureResidual:
      totalEnergy - (internalEnergy + kineticEnergy + potentialEnergy),
    minDensity,
    minPressure,
    maxSpeed,
  };
}
