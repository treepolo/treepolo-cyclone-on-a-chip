import { scale3, type Vec3 } from '../core/math.js';
import { cellCountOf, type ConservativeFields } from './fields.js';
import {
  accumulateInternalIntegratedFaceFlux,
  createIntegratedRate,
  type IntegratedConservativeRate,
  type IntegratedFaceFlux,
} from './finiteVolume.js';
import {
  computeLimitedPrimitiveGradients,
  radialFaceCentroid,
  reconstructPrimitiveAt,
  sideFaceCentroid,
  type LinearReconstructionStencil,
} from './reconstruction.js';
import { integratedSlau2Flux } from './slau2Flux.js';
import {
  conservedFromPrimitive,
  primitiveFromConserved,
  type ConservativeCell,
} from './state.js';
import {
  radialFaceVectorArea,
  shellCellIndex,
  sideFaceVectorArea,
  type SphericalShellGeometry,
} from './sphericalShellGeometry.js';

function readCell(fields: ConservativeFields, cell: number): ConservativeCell {
  return {
    rho: fields.rho[cell]!,
    momentum: [fields.momX[cell]!, fields.momY[cell]!, fields.momZ[cell]!],
    rhoE: fields.rhoE[cell]!,
  };
}

function phiAt(geopotential: ArrayLike<number> | undefined, cell: number): number {
  return geopotential === undefined ? 0 : geopotential[cell]!;
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

function rigidSlipWallFluxFromPressure(
  pressure: number,
  outwardVectorArea: Vec3,
): IntegratedFaceFlux {
  if (!(pressure > 0) || !Number.isFinite(pressure)) {
    throw new Error(`invalid slip-wall pressure: ${pressure}`);
  }
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

function rigidSlipWallFlux(
  state: ConservativeCell,
  outwardVectorArea: Vec3,
  geopotential: number,
): IntegratedFaceFlux {
  const pressure = primitiveFromConserved(state, geopotential).pressure;
  return rigidSlipWallFluxFromPressure(pressure, outwardVectorArea);
}

/**
 * Conservative first-order spatial Euler operator on the closed spherical shell.
 *
 * This function deliberately contains no gravity, Coriolis, forcing, filter or
 * sponge term. Every interior face is evaluated exactly once and the resulting
 * integrated flux is applied with opposite signs to the two neighboring cells.
 * The bottom and top are stationary impermeable slip walls, so they exchange no
 * mass or energy with the domain and exert only their pressure reaction.
 *
 * This piecewise-constant path remains as the monotone floor and regression gate.
 * The production spatial path is `closedShellEulerRateSecondOrder` below.
 */
export function closedShellEulerRateFirstOrder(
  fields: ConservativeFields,
  geometry: SphericalShellGeometry,
  geopotential?: ArrayLike<number>,
): IntegratedConservativeRate {
  const cellCount = geometry.horizontal.cellCount * geometry.nz;
  if (cellCountOf(fields) !== cellCount) {
    throw new Error('Core v2 shell operator field/geometry cell counts differ');
  }
  if (geopotential !== undefined && geopotential.length !== cellCount) {
    throw new Error('Core v2 shell operator geopotential length differs from cells');
  }

  const rate = createIntegratedRate(cellCount);
  const horizontal = geometry.horizontal;

  for (let e = 0; e < horizontal.edgeCount; e++) {
    const edge = horizontal.edges[e]!;
    for (let k = 0; k < geometry.nz; k++) {
      const leftCell = shellCellIndex(edge.leftCell, k, geometry.nz);
      const rightCell = shellCellIndex(edge.rightCell, k, geometry.nz);
      const vectorArea = sideFaceVectorArea(geometry, e, k);
      const flux = integratedSlau2Flux(
        readCell(fields, leftCell),
        readCell(fields, rightCell),
        vectorArea,
        phiAt(geopotential, leftCell),
        phiAt(geopotential, rightCell),
      );
      accumulateInternalIntegratedFaceFlux(rate, leftCell, rightCell, flux);
    }
  }

  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let ki = 1; ki < geometry.nz; ki++) {
      const lowerCell = shellCellIndex(c, ki - 1, geometry.nz);
      const upperCell = shellCellIndex(c, ki, geometry.nz);
      const vectorArea = radialFaceVectorArea(geometry, c, ki);
      const flux = integratedSlau2Flux(
        readCell(fields, lowerCell),
        readCell(fields, upperCell),
        vectorArea,
        phiAt(geopotential, lowerCell),
        phiAt(geopotential, upperCell),
      );
      accumulateInternalIntegratedFaceFlux(rate, lowerCell, upperCell, flux);
    }
  }

  for (let c = 0; c < horizontal.cellCount; c++) {
    const bottomCell = shellCellIndex(c, 0, geometry.nz);
    const bottomArea = scale3(radialFaceVectorArea(geometry, c, 0), -1);
    subtractBoundaryFlux(
      rate,
      bottomCell,
      rigidSlipWallFlux(
        readCell(fields, bottomCell),
        bottomArea,
        phiAt(geopotential, bottomCell),
      ),
    );

    const topCell = shellCellIndex(c, geometry.nz - 1, geometry.nz);
    const topArea = radialFaceVectorArea(geometry, c, geometry.nz);
    subtractBoundaryFlux(
      rate,
      topCell,
      rigidSlipWallFlux(
        readCell(fields, topCell),
        topArea,
        phiAt(geopotential, topCell),
      ),
    );
  }

  return rate;
}

/**
 * Conservative limited-linear Euler operator used by Core v2 production spatial
 * transport. Primitive variables are reconstructed from the exact volume
 * centroids to the exact face centroids with a parameter-free Barth-Jespersen
 * limiter. The numerical flux is still evaluated only once per shared face.
 *
 * Gravity is intentionally absent here. The later gravity layer must be derived
 * together with hydrostatic well-balancing and total-energy consistency rather
 * than being smuggled into the reconstruction as a source-term patch.
 */
export function closedShellEulerRateSecondOrder(
  fields: ConservativeFields,
  geometry: SphericalShellGeometry,
  stencil: LinearReconstructionStencil,
): IntegratedConservativeRate {
  const cellCount = geometry.horizontal.cellCount * geometry.nz;
  if (cellCountOf(fields) !== cellCount) {
    throw new Error('Core v2 second-order shell field/geometry cell counts differ');
  }
  if (stencil.geometry !== geometry || stencil.neighborCount.length !== cellCount) {
    throw new Error('Core v2 second-order shell reconstruction stencil belongs to another geometry');
  }

  const rate = createIntegratedRate(cellCount);
  const horizontal = geometry.horizontal;
  const gradients = computeLimitedPrimitiveGradients(fields, stencil);

  for (let e = 0; e < horizontal.edgeCount; e++) {
    const edge = horizontal.edges[e]!;
    for (let k = 0; k < geometry.nz; k++) {
      const leftCell = shellCellIndex(edge.leftCell, k, geometry.nz);
      const rightCell = shellCellIndex(edge.rightCell, k, geometry.nz);
      const position = sideFaceCentroid(geometry, e, k);
      const leftFace = conservedFromPrimitive(
        reconstructPrimitiveAt(fields, gradients, stencil, leftCell, position),
      );
      const rightFace = conservedFromPrimitive(
        reconstructPrimitiveAt(fields, gradients, stencil, rightCell, position),
      );
      const flux = integratedSlau2Flux(
        leftFace,
        rightFace,
        sideFaceVectorArea(geometry, e, k),
      );
      accumulateInternalIntegratedFaceFlux(rate, leftCell, rightCell, flux);
    }
  }

  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let ki = 1; ki < geometry.nz; ki++) {
      const lowerCell = shellCellIndex(c, ki - 1, geometry.nz);
      const upperCell = shellCellIndex(c, ki, geometry.nz);
      const position = radialFaceCentroid(geometry, c, ki);
      const lowerFace = conservedFromPrimitive(
        reconstructPrimitiveAt(fields, gradients, stencil, lowerCell, position),
      );
      const upperFace = conservedFromPrimitive(
        reconstructPrimitiveAt(fields, gradients, stencil, upperCell, position),
      );
      const flux = integratedSlau2Flux(
        lowerFace,
        upperFace,
        radialFaceVectorArea(geometry, c, ki),
      );
      accumulateInternalIntegratedFaceFlux(rate, lowerCell, upperCell, flux);
    }
  }

  for (let c = 0; c < horizontal.cellCount; c++) {
    const bottomCell = shellCellIndex(c, 0, geometry.nz);
    const bottomPosition = radialFaceCentroid(geometry, c, 0);
    const bottomPrimitive = reconstructPrimitiveAt(
      fields,
      gradients,
      stencil,
      bottomCell,
      bottomPosition,
    );
    subtractBoundaryFlux(
      rate,
      bottomCell,
      rigidSlipWallFluxFromPressure(
        bottomPrimitive.pressure,
        scale3(radialFaceVectorArea(geometry, c, 0), -1),
      ),
    );

    const topCell = shellCellIndex(c, geometry.nz - 1, geometry.nz);
    const topPosition = radialFaceCentroid(geometry, c, geometry.nz);
    const topPrimitive = reconstructPrimitiveAt(
      fields,
      gradients,
      stencil,
      topCell,
      topPosition,
    );
    subtractBoundaryFlux(
      rate,
      topCell,
      rigidSlipWallFluxFromPressure(
        topPrimitive.pressure,
        radialFaceVectorArea(geometry, c, geometry.nz),
      ),
    );
  }

  return rate;
}
