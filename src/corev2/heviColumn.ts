import { EARTH, type PlanetConfig } from '../core/constants.js';
import { scale3, type Vec3 } from '../core/math.js';
import type { ConservativeFields } from './fields.js';
import type { HydrostaticReference1D } from './hydrostaticReference.js';
import { integratedSlau2FluxFromPrimitive } from './slau2Flux.js';
import {
  pressurePerturbationFromReference,
  type ConservativeCell,
  type PrimitiveCell,
} from './state.js';
import {
  radialFaceVectorArea,
  shellCellIndex,
  type SphericalShellGeometry,
} from './sphericalShellGeometry.js';

export const CONSERVED_COMPONENTS = 5;

export function columnStateIndex(k: number, component: number): number {
  return k * CONSERVED_COMPONENTS + component;
}

export function extractColumnState(
  fields: ConservativeFields,
  horizontalCell: number,
  nz: number,
): Float64Array {
  const out = new Float64Array(nz * CONSERVED_COMPONENTS);
  for (let k = 0; k < nz; k++) {
    const q = shellCellIndex(horizontalCell, k, nz);
    out[columnStateIndex(k, 0)] = fields.rho[q]!;
    out[columnStateIndex(k, 1)] = fields.momX[q]!;
    out[columnStateIndex(k, 2)] = fields.momY[q]!;
    out[columnStateIndex(k, 3)] = fields.momZ[q]!;
    out[columnStateIndex(k, 4)] = fields.rhoE[q]!;
  }
  return out;
}

export function writeColumnState(
  fields: ConservativeFields,
  horizontalCell: number,
  nz: number,
  column: ArrayLike<number>,
): void {
  if (column.length !== nz * CONSERVED_COMPONENTS) {
    throw new Error('Core v2 HEVI column write length mismatch');
  }
  for (let k = 0; k < nz; k++) {
    const q = shellCellIndex(horizontalCell, k, nz);
    fields.rho[q] = column[columnStateIndex(k, 0)]!;
    fields.momX[q] = column[columnStateIndex(k, 1)]!;
    fields.momY[q] = column[columnStateIndex(k, 2)]!;
    fields.momZ[q] = column[columnStateIndex(k, 3)]!;
    fields.rhoE[q] = column[columnStateIndex(k, 4)]!;
  }
}

function readColumnState(column: ArrayLike<number>, k: number): ConservativeCell {
  return {
    rho: column[columnStateIndex(k, 0)]!,
    momentum: [
      column[columnStateIndex(k, 1)]!,
      column[columnStateIndex(k, 2)]!,
      column[columnStateIndex(k, 3)]!,
    ],
    rhoE: column[columnStateIndex(k, 4)]!,
  };
}

function hydrostaticPrimitive(
  column: ArrayLike<number>,
  k: number,
  reference: HydrostaticReference1D,
  referenceFacePressure: number,
): PrimitiveCell {
  const state = readColumnState(column, k);
  if (!(state.rho > 0) || !Number.isFinite(state.rho)) {
    throw new Error(`invalid HEVI column density at level ${k}: ${state.rho}`);
  }
  const invRho = 1 / state.rho;
  const pressurePerturbation = pressurePerturbationFromReference(
    state,
    reference.cellDensity[k]!,
    reference.cellPressure[k]!,
    reference.cellGeopotential[k]!,
  );
  const pressure = referenceFacePressure + pressurePerturbation;
  if (!(pressure > 0) || !Number.isFinite(pressure)) {
    throw new Error(`invalid HEVI column pressure at level ${k}: ${pressure}`);
  }
  return {
    rho: state.rho,
    velocity: [
      state.momentum[0] * invRho,
      state.momentum[1] * invRho,
      state.momentum[2] * invRho,
    ],
    pressure,
  };
}

function addFluxLeft(rate: Float64Array, k: number, flux: {
  mass: number;
  momentum: Vec3;
  totalEnergy: number;
}): void {
  rate[columnStateIndex(k, 0)] -= flux.mass;
  rate[columnStateIndex(k, 1)] -= flux.momentum[0];
  rate[columnStateIndex(k, 2)] -= flux.momentum[1];
  rate[columnStateIndex(k, 3)] -= flux.momentum[2];
  rate[columnStateIndex(k, 4)] -= flux.totalEnergy;
}

function addFluxRight(rate: Float64Array, k: number, flux: {
  mass: number;
  momentum: Vec3;
  totalEnergy: number;
}): void {
  rate[columnStateIndex(k, 0)] += flux.mass;
  rate[columnStateIndex(k, 1)] += flux.momentum[0];
  rate[columnStateIndex(k, 2)] += flux.momentum[1];
  rate[columnStateIndex(k, 3)] += flux.momentum[2];
  rate[columnStateIndex(k, 4)] += flux.totalEnergy;
}

function faceGeopotential(
  geometry: SphericalShellGeometry,
  kInterface: number,
  planet: PlanetConfig,
): number {
  return planet.gravity * (
    geometry.radiusInterface[kInterface]! - geometry.radiusInterface[0]!
  );
}

function referenceSidePressureForce(
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  horizontalCell: number,
  k: number,
): Vec3 {
  const r0 = geometry.radiusInterface[k]!;
  const r1 = geometry.radiusInterface[k + 1]!;
  const coefficient = reference.sideFacePressure[k]! * (r1 * r1 - r0 * r0);
  return [
    coefficient * geometry.cellVectorAreaUnit[horizontalCell * 3]!,
    coefficient * geometry.cellVectorAreaUnit[horizontalCell * 3 + 1]!,
    coefficient * geometry.cellVectorAreaUnit[horizontalCell * 3 + 2]!,
  ];
}

function gravityForce(
  geometry: SphericalShellGeometry,
  horizontalCell: number,
  k: number,
  density: number,
  planet: PlanetConfig,
): Vec3 {
  const r0 = geometry.radiusInterface[k]!;
  const r1 = geometry.radiusInterface[k + 1]!;
  const scale = -density * planet.gravity * (r1 ** 3 - r0 ** 3) / 3;
  return [
    scale * geometry.cellVectorAreaUnit[horizontalCell * 3]!,
    scale * geometry.cellVectorAreaUnit[horizontalCell * 3 + 1]!,
    scale * geometry.cellVectorAreaUnit[horizontalCell * 3 + 2]!,
  ];
}

/**
 * Integrated vertical stiff rate for exactly one horizontal atmospheric column.
 * The output ordering is [rho,mx,my,mz,rhoE] repeated by vertical level.
 */
export function verticalStiffColumnIntegratedRate(
  column: ArrayLike<number>,
  horizontalCell: number,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  planet: PlanetConfig = EARTH,
): Float64Array {
  const nz = geometry.nz;
  if (column.length !== nz * CONSERVED_COMPONENTS) {
    throw new Error('Core v2 HEVI column state length mismatch');
  }
  if (horizontalCell < 0 || horizontalCell >= geometry.horizontal.cellCount) {
    throw new Error(`invalid Core v2 HEVI horizontal column ${horizontalCell}`);
  }
  const rate = new Float64Array(column.length);

  for (let ki = 1; ki < nz; ki++) {
    const pRef = reference.radialFacePressure[ki]!;
    const lower = hydrostaticPrimitive(column, ki - 1, reference, pRef);
    const upper = hydrostaticPrimitive(column, ki, reference, pRef);
    const phi = faceGeopotential(geometry, ki, planet);
    const flux = integratedSlau2FluxFromPrimitive(
      lower,
      upper,
      radialFaceVectorArea(geometry, horizontalCell, ki),
      phi,
      phi,
    );
    addFluxLeft(rate, ki - 1, flux);
    addFluxRight(rate, ki, flux);
  }

  const bottom = hydrostaticPrimitive(
    column,
    0,
    reference,
    reference.radialFacePressure[0]!,
  );
  const bottomArea = scale3(
    radialFaceVectorArea(geometry, horizontalCell, 0),
    -1,
  );
  rate[columnStateIndex(0, 1)] -= bottom.pressure * bottomArea[0];
  rate[columnStateIndex(0, 2)] -= bottom.pressure * bottomArea[1];
  rate[columnStateIndex(0, 3)] -= bottom.pressure * bottomArea[2];

  const topK = nz - 1;
  const top = hydrostaticPrimitive(
    column,
    topK,
    reference,
    reference.radialFacePressure[nz]!,
  );
  const topArea = radialFaceVectorArea(geometry, horizontalCell, nz);
  rate[columnStateIndex(topK, 1)] -= top.pressure * topArea[0];
  rate[columnStateIndex(topK, 2)] -= top.pressure * topArea[1];
  rate[columnStateIndex(topK, 3)] -= top.pressure * topArea[2];

  for (let k = 0; k < nz; k++) {
    const density = column[columnStateIndex(k, 0)]!;
    const gravity = gravityForce(geometry, horizontalCell, k, density, planet);
    const side = referenceSidePressureForce(geometry, reference, horizontalCell, k);
    rate[columnStateIndex(k, 1)] += gravity[0] + side[0];
    rate[columnStateIndex(k, 2)] += gravity[1] + side[1];
    rate[columnStateIndex(k, 3)] += gravity[2] + side[2];
  }

  return rate;
}

/** Convert integrated column rates to conserved-density tendencies. */
export function verticalStiffColumnTendency(
  column: ArrayLike<number>,
  horizontalCell: number,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  planet: PlanetConfig = EARTH,
): Float64Array {
  const integrated = verticalStiffColumnIntegratedRate(
    column,
    horizontalCell,
    geometry,
    reference,
    planet,
  );
  const out = new Float64Array(integrated.length);
  for (let k = 0; k < geometry.nz; k++) {
    const volume = geometry.cellVolume[shellCellIndex(horizontalCell, k, geometry.nz)]!;
    for (let v = 0; v < CONSERVED_COMPONENTS; v++) {
      out[columnStateIndex(k, v)] = integrated[columnStateIndex(k, v)]! / volume;
    }
  }
  return out;
}
