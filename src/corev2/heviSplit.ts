import { EARTH, type PlanetConfig } from '../core/constants.js';
import { scale3, type Vec3 } from '../core/math.js';
import type { ConservativeFields } from './fields.js';
import {
  accumulateInternalIntegratedFaceFlux,
  createIntegratedRate,
  type IntegratedConservativeRate,
  type IntegratedFaceFlux,
} from './finiteVolume.js';
import {
  integratedCellGravityForce,
  type HydrostaticReference1D,
} from './hydrostaticReference.js';
import type { LinearReconstructionStencil } from './reconstruction.js';
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
import { closedShellWellBalancedEulerGravityRate } from './wellBalancedEulerGravity.js';

function readState(fields: ConservativeFields, cell: number): ConservativeCell {
  return {
    rho: fields.rho[cell]!,
    momentum: [fields.momX[cell]!, fields.momY[cell]!, fields.momZ[cell]!],
    rhoE: fields.rhoE[cell]!,
  };
}

function firstOrderHydrostaticPrimitive(
  fields: ConservativeFields,
  cell: number,
  k: number,
  reference: HydrostaticReference1D,
  referenceFacePressure: number,
): PrimitiveCell {
  const state = readState(fields, cell);
  if (!(state.rho > 0) || !Number.isFinite(state.rho)) {
    throw new Error(`invalid HEVI split density at cell ${cell}`);
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
    throw new Error(`invalid HEVI split face pressure at cell ${cell}: ${pressure}`);
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

function subtractBoundaryFlux(
  rate: IntegratedConservativeRate,
  cell: number,
  outwardFlux: IntegratedFaceFlux,
): void {
  rate.rho[cell] = rate.rho[cell]! - outwardFlux.mass;
  rate.momX[cell] = rate.momX[cell]! - outwardFlux.momentum[0];
  rate.momY[cell] = rate.momY[cell]! - outwardFlux.momentum[1];
  rate.momZ[cell] = rate.momZ[cell]! - outwardFlux.momentum[2];
  rate.rhoE[cell] = rate.rhoE[cell]! - outwardFlux.totalEnergy;
}

function slipWallPressureFlux(pressure: number, outwardVectorArea: Vec3): IntegratedFaceFlux {
  return {
    mass: 0,
    momentum: [
      pressure * outwardVectorArea[0],
      pressure * outwardVectorArea[1],
      pressure * outwardVectorArea[2],
    ],
    totalEnergy: 0,
  };
}

function addMomentum(rate: IntegratedConservativeRate, cell: number, force: Vec3): void {
  rate.momX[cell] = rate.momX[cell]! + force[0];
  rate.momY[cell] = rate.momY[cell]! + force[1];
  rate.momZ[cell] = rate.momZ[cell]! + force[2];
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

function faceGeopotential(
  geometry: SphericalShellGeometry,
  kInterface: number,
  planet: PlanetConfig,
): number {
  return planet.gravity * (
    geometry.radiusInterface[kInterface]! - geometry.radiusInterface[0]!
  );
}

/**
 * Column-local low-order stiff operator used by Core v2 HEVI.
 *
 * It contains only radial shared-face Euler fluxes, closed radial boundaries,
 * gravity, and the exact local side-pressure force of the hydrostatic reference.
 * It has no horizontal neighbor coupling. The reference side-force term is the
 * geometric piece required because Cartesian momentum in a curved shell needs
 * the conical side-face pressure contribution even for a purely radial
 * hydrostatic pressure field.
 *
 * This operator is not added on top of the physical equations. The production
 * explicit remainder is defined as FULL_SECOND_ORDER - THIS_OPERATOR, so their
 * sum is algebraically exactly the already validated full Euler+gravity rate.
 */
export function closedShellVerticalStiffRateFirstOrder(
  fields: ConservativeFields,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  planet: PlanetConfig = EARTH,
): IntegratedConservativeRate {
  const horizontal = geometry.horizontal;
  const cellCount = horizontal.cellCount * geometry.nz;
  if (fields.rho.length !== cellCount) {
    throw new Error('Core v2 HEVI vertical split field/geometry size mismatch');
  }
  const rate = createIntegratedRate(cellCount);

  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let ki = 1; ki < geometry.nz; ki++) {
      const lower = shellCellIndex(c, ki - 1, geometry.nz);
      const upper = shellCellIndex(c, ki, geometry.nz);
      const pRef = reference.radialFacePressure[ki]!;
      const lowerPrimitive = firstOrderHydrostaticPrimitive(
        fields, lower, ki - 1, reference, pRef,
      );
      const upperPrimitive = firstOrderHydrostaticPrimitive(
        fields, upper, ki, reference, pRef,
      );
      const phi = faceGeopotential(geometry, ki, planet);
      const flux = integratedSlau2FluxFromPrimitive(
        lowerPrimitive,
        upperPrimitive,
        radialFaceVectorArea(geometry, c, ki),
        phi,
        phi,
      );
      accumulateInternalIntegratedFaceFlux(rate, lower, upper, flux);
    }

    const bottom = shellCellIndex(c, 0, geometry.nz);
    const bottomPrimitive = firstOrderHydrostaticPrimitive(
      fields,
      bottom,
      0,
      reference,
      reference.radialFacePressure[0]!,
    );
    subtractBoundaryFlux(
      rate,
      bottom,
      slipWallPressureFlux(
        bottomPrimitive.pressure,
        scale3(radialFaceVectorArea(geometry, c, 0), -1),
      ),
    );

    const topK = geometry.nz - 1;
    const top = shellCellIndex(c, topK, geometry.nz);
    const topPrimitive = firstOrderHydrostaticPrimitive(
      fields,
      top,
      topK,
      reference,
      reference.radialFacePressure[geometry.nz]!,
    );
    subtractBoundaryFlux(
      rate,
      top,
      slipWallPressureFlux(
        topPrimitive.pressure,
        radialFaceVectorArea(geometry, c, geometry.nz),
      ),
    );

    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      addMomentum(
        rate,
        q,
        integratedCellGravityForce(geometry, c, k, fields.rho[q]!, planet),
      );
      addMomentum(rate, q, referenceSidePressureForce(geometry, reference, c, k));
    }
  }

  return rate;
}

function subtractRates(
  full: IntegratedConservativeRate,
  stiff: IntegratedConservativeRate,
): IntegratedConservativeRate {
  const n = full.rho.length;
  if (stiff.rho.length !== n) throw new Error('Core v2 HEVI split rate size mismatch');
  const out = createIntegratedRate(n);
  for (let q = 0; q < n; q++) {
    out.rho[q] = full.rho[q]! - stiff.rho[q]!;
    out.momX[q] = full.momX[q]! - stiff.momX[q]!;
    out.momY[q] = full.momY[q]! - stiff.momY[q]!;
    out.momZ[q] = full.momZ[q]! - stiff.momZ[q]!;
    out.rhoE[q] = full.rhoE[q]! - stiff.rhoE[q]!;
  }
  return out;
}

/**
 * Explicit HEVI remainder. By construction
 *
 *   explicitRemainder(U) + verticalStiff(U) = fullSecondOrder(U)
 *
 * component-by-component, so the split cannot silently add or remove a force,
 * flux, gravity term, or stabilizer.
 */
export function closedShellHeviExplicitRemainderRate(
  fields: ConservativeFields,
  geometry: SphericalShellGeometry,
  stencil: LinearReconstructionStencil,
  reference: HydrostaticReference1D,
  planet: PlanetConfig = EARTH,
): IntegratedConservativeRate {
  const full = closedShellWellBalancedEulerGravityRate(
    fields,
    geometry,
    stencil,
    reference,
    planet,
  );
  const stiff = closedShellVerticalStiffRateFirstOrder(
    fields,
    geometry,
    reference,
    planet,
  );
  return subtractRates(full, stiff);
}
