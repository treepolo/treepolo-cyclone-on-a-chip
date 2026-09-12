import type { Vec3 } from '../core/math.js';
import type { ConservativeFields } from './fields.js';
import type { HydrostaticReference1D } from './hydrostaticReference.js';
import {
  MAX_RECONSTRUCTION_NEIGHBORS,
  computeLimitedPrimitiveGradients,
  radialFaceCentroid,
  sideFaceCentroid,
  type LinearReconstructionStencil,
  type PrimitiveGradients,
} from './reconstruction.js';
import { primitiveFromConserved, type PrimitiveCell } from './state.js';

export interface HydrostaticReconstruction {
  primitive: PrimitiveGradients;
  pressurePerturbX: Float64Array;
  pressurePerturbY: Float64Array;
  pressurePerturbZ: Float64Array;
  geopotential: Float64Array;
}

function primitiveAtCell(
  fields: ConservativeFields,
  cell: number,
  geopotential: ArrayLike<number>,
): PrimitiveCell {
  return primitiveFromConserved({
    rho: fields.rho[cell]!,
    momentum: [fields.momX[cell]!, fields.momY[cell]!, fields.momZ[cell]!],
    rhoE: fields.rhoE[cell]!,
  }, geopotential[cell]!);
}

function packedCellPosition(stencil: LinearReconstructionStencil, cell: number): Vec3 {
  return [
    stencil.cellCentroid[cell * 3]!,
    stencil.cellCentroid[cell * 3 + 1]!,
    stencil.cellCentroid[cell * 3 + 2]!,
  ];
}

function facePositions(stencil: LinearReconstructionStencil, cell: number): Vec3[] {
  const geometry = stencil.geometry;
  const horizontalCell = Math.floor(cell / geometry.nz);
  const k = cell % geometry.nz;
  const out: Vec3[] = [];
  for (let s = 0; s < 4; s++) {
    const edge = geometry.horizontal.cellEdges[horizontalCell * 4 + s]!;
    out.push(sideFaceCentroid(geometry, edge, k));
  }
  out.push(radialFaceCentroid(geometry, horizontalCell, k));
  out.push(radialFaceCentroid(geometry, horizontalCell, k + 1));
  return out;
}

function cellReferencePressure(
  reference: HydrostaticReference1D,
  cell: number,
  nz: number,
): number {
  return reference.cellPressure[cell % nz]!;
}

/**
 * Build the same limited primitive reconstruction used by the production Euler
 * operator, but replace the pressure slope by the slope of
 *
 *     p' = p - p_ref.
 *
 * This is the key well-balanced change: an exact hydrostatic reference has
 * p'=0 in every cell, so both sides of every face reconstruct the same exact
 * reference face pressure. The Riemann solver therefore sees no spurious
 * hydrostatic pressure jump and cannot manufacture vertical mass flux from it.
 */
export function buildHydrostaticReconstruction(
  fields: ConservativeFields,
  stencil: LinearReconstructionStencil,
  reference: HydrostaticReference1D,
): HydrostaticReconstruction {
  const geometry = stencil.geometry;
  const cellCount = stencil.neighborCount.length;
  if (
    fields.rho.length !== cellCount ||
    fields.momX.length !== cellCount ||
    fields.momY.length !== cellCount ||
    fields.momZ.length !== cellCount ||
    fields.rhoE.length !== cellCount
  ) {
    throw new Error('Core v2 hydrostatic reconstruction field/stencil size mismatch');
  }
  if (
    reference.cellPressure.length !== geometry.nz ||
    reference.cellGeopotential.length !== geometry.nz
  ) {
    throw new Error('Core v2 hydrostatic reference/stencil vertical size mismatch');
  }

  const geopotential = new Float64Array(cellCount);
  for (let q = 0; q < cellCount; q++) {
    geopotential[q] = reference.cellGeopotential[q % geometry.nz]!;
  }
  const primitive = computeLimitedPrimitiveGradients(fields, stencil, geopotential);
  const px = new Float64Array(cellCount);
  const py = new Float64Array(cellCount);
  const pz = new Float64Array(cellCount);
  const pPerturb = new Float64Array(cellCount);
  for (let q = 0; q < cellCount; q++) {
    pPerturb[q] = primitiveAtCell(fields, q, geopotential).pressure -
      cellReferencePressure(reference, q, geometry.nz);
  }

  for (let q = 0; q < cellCount; q++) {
    const q0 = pPerturb[q]!;
    const count = stencil.neighborCount[q]!;
    let gx = 0;
    let gy = 0;
    let gz = 0;
    let qMin = q0;
    let qMax = q0;
    for (let j = 0; j < count; j++) {
      const slot = q * MAX_RECONSTRUCTION_NEIGHBORS + j;
      const neighbor = stencil.neighborCell[slot]!;
      const qn = pPerturb[neighbor]!;
      const delta = qn - q0;
      gx += stencil.coeffX[slot]! * delta;
      gy += stencil.coeffY[slot]! * delta;
      gz += stencil.coeffZ[slot]! * delta;
      qMin = Math.min(qMin, qn);
      qMax = Math.max(qMax, qn);
    }

    const x0 = packedCellPosition(stencil, q);
    let limiter = 1;
    for (const xf of facePositions(stencil, q)) {
      const delta = gx * (xf[0] - x0[0]) + gy * (xf[1] - x0[1]) + gz * (xf[2] - x0[2]);
      if (delta > 0) limiter = Math.min(limiter, (qMax - q0) / delta);
      else if (delta < 0) limiter = Math.min(limiter, (qMin - q0) / delta);
    }
    limiter = Math.max(0, Math.min(1, limiter));
    px[q] = limiter * gx;
    py[q] = limiter * gy;
    pz[q] = limiter * gz;
  }

  return {
    primitive,
    pressurePerturbX: px,
    pressurePerturbY: py,
    pressurePerturbZ: pz,
    geopotential,
  };
}

/** Reconstruct one primitive face state around an exact reference face pressure. */
export function reconstructHydrostaticPrimitiveAt(
  fields: ConservativeFields,
  reconstruction: HydrostaticReconstruction,
  stencil: LinearReconstructionStencil,
  reference: HydrostaticReference1D,
  cell: number,
  position: Vec3,
  referenceFacePressure: number,
): PrimitiveCell {
  const geometry = stencil.geometry;
  const base = primitiveAtCell(fields, cell, reconstruction.geopotential);
  const x0 = packedCellPosition(stencil, cell);
  const dx = position[0] - x0[0];
  const dy = position[1] - x0[1];
  const dz = position[2] - x0[2];
  const apply = (
    value: number,
    gx: Float64Array,
    gy: Float64Array,
    gz: Float64Array,
  ): number => value + gx[cell]! * dx + gy[cell]! * dy + gz[cell]! * dz;

  const rho = apply(
    base.rho,
    reconstruction.primitive.rhoX,
    reconstruction.primitive.rhoY,
    reconstruction.primitive.rhoZ,
  );
  const basePressurePerturbation =
    base.pressure - cellReferencePressure(reference, cell, geometry.nz);
  const pressure = referenceFacePressure + apply(
    basePressurePerturbation,
    reconstruction.pressurePerturbX,
    reconstruction.pressurePerturbY,
    reconstruction.pressurePerturbZ,
  );
  if (!(rho > 0) || !(pressure > 0) || !Number.isFinite(rho) || !Number.isFinite(pressure)) {
    throw new Error(`Core v2 hydrostatic reconstruction produced invalid rho/p at cell ${cell}`);
  }

  return {
    rho,
    velocity: [
      apply(base.velocity[0], reconstruction.primitive.uxX, reconstruction.primitive.uxY, reconstruction.primitive.uxZ),
      apply(base.velocity[1], reconstruction.primitive.uyX, reconstruction.primitive.uyY, reconstruction.primitive.uyZ),
      apply(base.velocity[2], reconstruction.primitive.uzX, reconstruction.primitive.uzY, reconstruction.primitive.uzZ),
    ],
    pressure,
  };
}
