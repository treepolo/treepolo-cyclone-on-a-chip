import { EARTH, type PlanetConfig } from '../core/constants.js';
import { norm3, scale3, type Vec3 } from '../core/math.js';
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
import {
  radialFaceCentroid,
  sideFaceCentroid,
  type LinearReconstructionStencil,
} from './reconstruction.js';
import { integratedSlau2FluxFromPrimitive } from './slau2Flux.js';
import {
  radialFaceVectorArea,
  shellCellIndex,
  sideFaceVectorArea,
  type SphericalShellGeometry,
} from './sphericalShellGeometry.js';
import {
  buildHydrostaticReconstruction,
  reconstructHydrostaticPrimitiveAt,
} from './wellBalancedReconstruction.js';

function sideFaceRadialMean(geometry: SphericalShellGeometry, k: number): number {
  const r0 = geometry.radiusInterface[k]!;
  const r1 = geometry.radiusInterface[k + 1]!;
  return (2 / 3) * (r1 ** 3 - r0 ** 3) / (r1 * r1 - r0 * r0);
}

function geopotentialAtRadius(
  geometry: SphericalShellGeometry,
  radius: number,
  planet: PlanetConfig,
): number {
  return planet.gravity * (radius - geometry.radiusInterface[0]!);
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

function addGravityForce(
  rate: IntegratedConservativeRate,
  cell: number,
  force: Vec3,
): void {
  rate.momX[cell] = rate.momX[cell]! + force[0];
  rate.momY[cell] = rate.momY[cell]! + force[1];
  rate.momZ[cell] = rate.momZ[cell]! + force[2];
  // rhoE already contains gravitational potential energy, so gravity has no
  // separate total-energy source in this conservative formulation.
}

/**
 * Second-order conservative Euler + constant-radial-gravity operator with an
 * exactly well-balanced isothermal hydrostatic reference.
 *
 * The reference is only an algebraic split. It never changes the prognostic
 * state by itself. At every face we reconstruct p' = p-p_ref and then add the
 * exact reference face pressure. The direct primitive-state SLAU2 path is used
 * so this pressure is not converted through rhoE-rho*Phi again before the
 * Riemann solve.
 */
export function closedShellWellBalancedEulerGravityRate(
  fields: ConservativeFields,
  geometry: SphericalShellGeometry,
  stencil: LinearReconstructionStencil,
  reference: HydrostaticReference1D,
  planet: PlanetConfig = EARTH,
): IntegratedConservativeRate {
  const cellCount = geometry.horizontal.cellCount * geometry.nz;
  if (
    fields.rho.length !== cellCount ||
    fields.momX.length !== cellCount ||
    fields.momY.length !== cellCount ||
    fields.momZ.length !== cellCount ||
    fields.rhoE.length !== cellCount
  ) {
    throw new Error('Core v2 well-balanced operator field/geometry size mismatch');
  }
  if (stencil.geometry !== geometry) {
    throw new Error('Core v2 well-balanced operator received a foreign reconstruction stencil');
  }
  if (
    reference.cellDensity.length !== geometry.nz ||
    reference.cellPressure.length !== geometry.nz ||
    reference.radialFacePressure.length !== geometry.nz + 1 ||
    reference.sideFacePressure.length !== geometry.nz
  ) {
    throw new Error('Core v2 well-balanced operator reference/geometry size mismatch');
  }

  const rate = createIntegratedRate(cellCount);
  const reconstruction = buildHydrostaticReconstruction(fields, stencil, reference);
  const horizontal = geometry.horizontal;

  for (let e = 0; e < horizontal.edgeCount; e++) {
    const edge = horizontal.edges[e]!;
    for (let k = 0; k < geometry.nz; k++) {
      const left = shellCellIndex(edge.leftCell, k, geometry.nz);
      const right = shellCellIndex(edge.rightCell, k, geometry.nz);
      const position = sideFaceCentroid(geometry, e, k);
      const pRef = reference.sideFacePressure[k]!;
      const leftPrimitive = reconstructHydrostaticPrimitiveAt(
        fields, reconstruction, stencil, reference, left, position, pRef,
      );
      const rightPrimitive = reconstructHydrostaticPrimitiveAt(
        fields, reconstruction, stencil, reference, right, position, pRef,
      );
      const phiFace = geopotentialAtRadius(
        geometry,
        sideFaceRadialMean(geometry, k),
        planet,
      );
      const flux = integratedSlau2FluxFromPrimitive(
        leftPrimitive,
        rightPrimitive,
        sideFaceVectorArea(geometry, e, k),
        phiFace,
        phiFace,
      );
      accumulateInternalIntegratedFaceFlux(rate, left, right, flux);
    }
  }

  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let ki = 1; ki < geometry.nz; ki++) {
      const lower = shellCellIndex(c, ki - 1, geometry.nz);
      const upper = shellCellIndex(c, ki, geometry.nz);
      const position = radialFaceCentroid(geometry, c, ki);
      const pRef = reference.radialFacePressure[ki]!;
      const lowerPrimitive = reconstructHydrostaticPrimitiveAt(
        fields, reconstruction, stencil, reference, lower, position, pRef,
      );
      const upperPrimitive = reconstructHydrostaticPrimitiveAt(
        fields, reconstruction, stencil, reference, upper, position, pRef,
      );
      const phiFace = geopotentialAtRadius(geometry, geometry.radiusInterface[ki]!, planet);
      const flux = integratedSlau2FluxFromPrimitive(
        lowerPrimitive,
        upperPrimitive,
        radialFaceVectorArea(geometry, c, ki),
        phiFace,
        phiFace,
      );
      accumulateInternalIntegratedFaceFlux(rate, lower, upper, flux);
    }
  }

  for (let c = 0; c < horizontal.cellCount; c++) {
    const bottom = shellCellIndex(c, 0, geometry.nz);
    const bottomPosition = radialFaceCentroid(geometry, c, 0);
    const bottomPrimitive = reconstructHydrostaticPrimitiveAt(
      fields,
      reconstruction,
      stencil,
      reference,
      bottom,
      bottomPosition,
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

    const top = shellCellIndex(c, geometry.nz - 1, geometry.nz);
    const topPosition = radialFaceCentroid(geometry, c, geometry.nz);
    const topPrimitive = reconstructHydrostaticPrimitiveAt(
      fields,
      reconstruction,
      stencil,
      reference,
      top,
      topPosition,
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
  }

  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      addGravityForce(
        rate,
        q,
        integratedCellGravityForce(geometry, c, k, fields.rho[q]!, planet),
      );
    }
  }

  for (let q = 0; q < cellCount; q++) {
    const magnitude = Math.max(
      Math.abs(rate.rho[q]!),
      Math.abs(rate.momX[q]!),
      Math.abs(rate.momY[q]!),
      Math.abs(rate.momZ[q]!),
      Math.abs(rate.rhoE[q]!),
    );
    if (!Number.isFinite(magnitude)) {
      throw new Error(`non-finite Core v2 well-balanced rate at cell ${q}`);
    }
  }

  return rate;
}

export function integratedMomentumRateNorm(
  rate: IntegratedConservativeRate,
  cell: number,
): number {
  return norm3([rate.momX[cell]!, rate.momY[cell]!, rate.momZ[cell]!]);
}
