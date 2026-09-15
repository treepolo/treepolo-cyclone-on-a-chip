import { DRY_AIR, type AtmosphereConfig } from '../core/constants.js';
import {
  cellCountOf,
  type ConservativeFields,
} from './fields.js';
import type { HydrostaticReference1D } from './hydrostaticReference.js';
import { pressureFromConservedRelativeToReference } from './state.js';
import type { SphericalShellGeometry } from './sphericalShellGeometry.js';

const DAY = 86400;
const PACKED_FLOATS_PER_CELL = 8;

export const CORE_V2_HELD_SUAREZ = {
  T0: 315,
  deltaTy: 60,
  deltaThetaZ: 10,
  Tmin: 200,
  sigmaB: 0.7,
  ka: 1 / (40 * DAY),
  ks: 1 / (4 * DAY),
  kf: 1 / DAY,
} as const;

export function heldSuarezEquilibriumTemperature(
  latitude: number,
  pressure: number,
  gas: AtmosphereConfig = DRY_AIR,
): number {
  const sigma = Math.max(1e-6, pressure / gas.pRef);
  const sinLat = Math.sin(latitude);
  const cosLat = Math.cos(latitude);
  const raw = (
    CORE_V2_HELD_SUAREZ.T0 -
    CORE_V2_HELD_SUAREZ.deltaTy * sinLat * sinLat -
    CORE_V2_HELD_SUAREZ.deltaThetaZ * Math.log(sigma) * cosLat * cosLat
  ) * Math.pow(sigma, gas.kappa);
  return Math.max(CORE_V2_HELD_SUAREZ.Tmin, raw);
}

export function heldSuarezThermalRate(latitude: number, sigma: number): number {
  const surfaceWeight = Math.max(
    0,
    (sigma - CORE_V2_HELD_SUAREZ.sigmaB) /
      (1 - CORE_V2_HELD_SUAREZ.sigmaB),
  );
  return CORE_V2_HELD_SUAREZ.ka +
    (CORE_V2_HELD_SUAREZ.ks - CORE_V2_HELD_SUAREZ.ka) *
      surfaceWeight * Math.pow(Math.cos(latitude), 4);
}

export function heldSuarezDragRate(sigma: number): number {
  return CORE_V2_HELD_SUAREZ.kf * Math.max(
    0,
    (sigma - CORE_V2_HELD_SUAREZ.sigmaB) /
      (1 - CORE_V2_HELD_SUAREZ.sigmaB),
  );
}

interface AdvancedCell {
  mx: number;
  my: number;
  mz: number;
  rhoE: number;
}

function advanceHeldSuarezCell(
  rho: number,
  mx: number,
  my: number,
  mz: number,
  rhoE: number,
  radialX: number,
  radialY: number,
  radialZ: number,
  latitude: number,
  k: number,
  dt: number,
  reference: HydrostaticReference1D,
  gas: AtmosphereConfig,
): AdvancedCell {
  const geopotential = reference.cellGeopotential[k]!;
  const pressure = pressureFromConservedRelativeToReference({
    rho,
    momentum: [mx, my, mz],
    rhoE,
  }, reference.cellDensity[k]!, reference.cellPressure[k]!, geopotential, gas);
  const temperature = pressure / (rho * gas.rd);
  const sigma = pressure / gas.pRef;

  const equilibriumTemperature = heldSuarezEquilibriumTemperature(latitude, pressure, gas);
  const thermalDecay = Math.exp(-heldSuarezThermalRate(latitude, sigma) * dt);
  const nextTemperature = equilibriumTemperature +
    (temperature - equilibriumTemperature) * thermalDecay;

  const radialMomentum = mx * radialX + my * radialY + mz * radialZ;
  const tangentX = mx - radialMomentum * radialX;
  const tangentY = my - radialMomentum * radialY;
  const tangentZ = mz - radialMomentum * radialZ;
  const dragDecay = Math.exp(-heldSuarezDragRate(sigma) * dt);
  const nextMx = radialMomentum * radialX + tangentX * dragDecay;
  const nextMy = radialMomentum * radialY + tangentY * dragDecay;
  const nextMz = radialMomentum * radialZ + tangentZ * dragDecay;

  // Held-Suarez drag is an explicit kinetic-energy sink, not frictional heating.
  // Rebuild rhoE so the thermal relaxation controls internal energy while the
  // damped horizontal/tangential kinetic energy is removed from the system.
  const nextInternal = rho * gas.cvd * nextTemperature;
  const nextKinetic = 0.5 * (
    nextMx * nextMx + nextMy * nextMy + nextMz * nextMz
  ) / rho;
  const nextRhoE = nextInternal + nextKinetic + rho * geopotential;

  return { mx: nextMx, my: nextMy, mz: nextMz, rhoE: nextRhoE };
}

function validateInputs(
  cellCount: number,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  dt: number,
): void {
  const expected = geometry.horizontal.cellCount * geometry.nz;
  if (cellCount !== expected) {
    throw new Error('Core v2 Held-Suarez state/geometry size mismatch');
  }
  if (
    reference.cellDensity.length !== geometry.nz ||
    reference.cellPressure.length !== geometry.nz ||
    reference.cellGeopotential.length !== geometry.nz
  ) {
    throw new Error('Core v2 Held-Suarez reference/geometry size mismatch');
  }
  if (!(dt >= 0) || !Number.isFinite(dt)) {
    throw new Error(`invalid Core v2 Held-Suarez dt: ${dt}`);
  }
}

/**
 * Apply the standard Held-Suarez dry-atmosphere forcing to the conservative
 * Core v2 state in place.
 *
 * Density and radial momentum are unchanged. Newtonian temperature relaxation
 * changes internal energy; near-surface Rayleigh drag damps only tangential
 * momentum and removes the corresponding kinetic energy from rhoE.
 */
export function applyHeldSuarezForcing(
  fields: ConservativeFields,
  dt: number,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  gas: AtmosphereConfig = DRY_AIR,
): void {
  const cells = cellCountOf(fields);
  validateInputs(cells, geometry, reference, dt);
  if (dt === 0) return;

  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    const radialX = geometry.horizontal.cellCenters[c * 3]!;
    const radialY = geometry.horizontal.cellCenters[c * 3 + 1]!;
    const radialZ = geometry.horizontal.cellCenters[c * 3 + 2]!;
    const latitude = Math.asin(Math.max(-1, Math.min(1, radialZ)));
    for (let k = 0; k < geometry.nz; k++) {
      const q = c * geometry.nz + k;
      const rho = fields.rho[q]!;
      const next = advanceHeldSuarezCell(
        rho,
        fields.momX[q]!,
        fields.momY[q]!,
        fields.momZ[q]!,
        fields.rhoE[q]!,
        radialX,
        radialY,
        radialZ,
        latitude,
        k,
        dt,
        reference,
        gas,
      );
      fields.momX[q] = next.mx;
      fields.momY[q] = next.my;
      fields.momZ[q] = next.mz;
      fields.rhoE[q] = next.rhoE;
    }
  }
}

/** Same forcing for the existing 8-float-per-cell GPU host-visible state. */
export function applyHeldSuarezForcingPackedF32(
  packedState: Float32Array,
  dt: number,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  gas: AtmosphereConfig = DRY_AIR,
): void {
  const cells = packedState.length / PACKED_FLOATS_PER_CELL;
  if (!Number.isInteger(cells)) {
    throw new Error('Core v2 Held-Suarez packed state must contain 8 floats per cell');
  }
  validateInputs(cells, geometry, reference, dt);
  if (dt === 0) return;

  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    const radialX = geometry.horizontal.cellCenters[c * 3]!;
    const radialY = geometry.horizontal.cellCenters[c * 3 + 1]!;
    const radialZ = geometry.horizontal.cellCenters[c * 3 + 2]!;
    const latitude = Math.asin(Math.max(-1, Math.min(1, radialZ)));
    for (let k = 0; k < geometry.nz; k++) {
      const q = c * geometry.nz + k;
      const i = q * PACKED_FLOATS_PER_CELL;
      const rho = packedState[i]!;
      const next = advanceHeldSuarezCell(
        rho,
        packedState[i + 1]!,
        packedState[i + 2]!,
        packedState[i + 3]!,
        packedState[i + 4]!,
        radialX,
        radialY,
        radialZ,
        latitude,
        k,
        dt,
        reference,
        gas,
      );
      packedState[i + 1] = Math.fround(next.mx);
      packedState[i + 2] = Math.fround(next.my);
      packedState[i + 3] = Math.fround(next.mz);
      packedState[i + 4] = Math.fround(next.rhoE);
    }
  }
}
