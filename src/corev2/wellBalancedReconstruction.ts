import type { Vec3 } from '../core/math.js';
import type { ConservativeFields } from './fields.js';
import type { HydrostaticReference1D } from './hydrostaticReference.js';
import {
  MAX_RECONSTRUCTION_NEIGHBORS,
  radialFaceCentroid,
  sideFaceCentroid,
  type LinearReconstructionStencil,
  type PrimitiveGradients,
} from './reconstruction.js';
import {
  pressurePerturbationFromReference,
  type ConservativeCell,
  type PrimitiveCell,
} from './state.js';

export interface HydrostaticReconstruction {
  primitive: PrimitiveGradients;
  pressurePerturbX: Float64Array;
  pressurePerturbY: Float64Array;
  pressurePerturbZ: Float64Array;
  pressurePerturb: Float64Array;
  geopotential: Float64Array;
}

interface StableCellPrimitive {
  rho: number;
  velocity: Vec3;
  pressurePerturbation: number;
}

function readState(fields: ConservativeFields, cell: number): ConservativeCell {
  return {
    rho: fields.rho[cell]!,
    momentum: [fields.momX[cell]!, fields.momY[cell]!, fields.momZ[cell]!],
    rhoE: fields.rhoE[cell]!,
  };
}

function stableCellPrimitive(
  fields: ConservativeFields,
  cell: number,
  reference: HydrostaticReference1D,
  geopotential: ArrayLike<number>,
  nz: number,
): StableCellPrimitive {
  const state = readState(fields, cell);
  if (!(state.rho > 0) || !Number.isFinite(state.rho)) {
    throw new Error(`invalid hydrostatic-reconstruction density at cell ${cell}`);
  }
  const k = cell % nz;
  const invRho = 1 / state.rho;
  const pressurePerturbation = pressurePerturbationFromReference(
    state,
    reference.cellDensity[k]!,
    reference.cellPressure[k]!,
    geopotential[cell]!,
  );
  const pressure = reference.cellPressure[k]! + pressurePerturbation;
  if (!(pressure > 0) || !Number.isFinite(pressure)) {
    throw new Error(`invalid hydrostatic-reconstruction pressure at cell ${cell}: ${pressure}`);
  }
  return {
    rho: state.rho,
    velocity: [
      state.momentum[0] * invRho,
      state.momentum[1] * invRho,
      state.momentum[2] * invRho,
    ],
    pressurePerturbation,
  };
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

function allocatePrimitiveGradients(cellCount: number): PrimitiveGradients {
  const a = (): Float64Array => new Float64Array(cellCount);
  return {
    rhoX:a(),rhoY:a(),rhoZ:a(),
    uxX:a(),uxY:a(),uxZ:a(),
    uyX:a(),uyY:a(),uyZ:a(),
    uzX:a(),uzY:a(),uzZ:a(),
    pX:a(),pY:a(),pZ:a(),
  };
}

function gradientArrays(
  out: PrimitiveGradients,
  variable: number,
): readonly [Float64Array, Float64Array, Float64Array] {
  if (variable === 0) return [out.rhoX, out.rhoY, out.rhoZ];
  if (variable === 1) return [out.uxX, out.uxY, out.uxZ];
  if (variable === 2) return [out.uyX, out.uyY, out.uyZ];
  if (variable === 3) return [out.uzX, out.uzY, out.uzZ];
  return [out.pX, out.pY, out.pZ];
}

function variableOf(cell: StableCellPrimitive, variable: number): number {
  if (variable === 0) return cell.rho;
  if (variable === 1) return cell.velocity[0];
  if (variable === 2) return cell.velocity[1];
  if (variable === 3) return cell.velocity[2];
  return cell.pressurePerturbation;
}

/**
 * Build a cancellation-free hydrostatic perturbation reconstruction.
 *
 * All five reconstructed quantities are evaluated without first forming the
 * absolute pressure as rhoE-K-rho*Phi. The pressure variable seen by the WLS
 * stencil is p'=p-p_ref, diagnosed with the reference-relative energy identity.
 * Therefore an exactly initialized hydrostatic reference has p'=0 bit-for-bit
 * and the Riemann solver cannot turn energy/geopotential roundoff into mass flux.
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
    reference.cellDensity.length !== geometry.nz ||
    reference.cellPressure.length !== geometry.nz ||
    reference.cellGeopotential.length !== geometry.nz
  ) {
    throw new Error('Core v2 hydrostatic reference/stencil vertical size mismatch');
  }

  const geopotential = new Float64Array(cellCount);
  for (let q = 0; q < cellCount; q++) {
    geopotential[q] = reference.cellGeopotential[q % geometry.nz]!;
  }

  const cells: StableCellPrimitive[] = new Array(cellCount);
  const pressurePerturb = new Float64Array(cellCount);
  for (let q = 0; q < cellCount; q++) {
    const cell = stableCellPrimitive(fields, q, reference, geopotential, geometry.nz);
    cells[q] = cell;
    pressurePerturb[q] = cell.pressurePerturbation;
  }

  const primitive = allocatePrimitiveGradients(cellCount);
  for (let q = 0; q < cellCount; q++) {
    const count = stencil.neighborCount[q]!;
    const x0 = packedCellPosition(stencil, q);
    const faces = facePositions(stencil, q);
    for (let variable = 0; variable < 5; variable++) {
      const q0 = variableOf(cells[q]!, variable);
      let gx = 0;
      let gy = 0;
      let gz = 0;
      let qMin = q0;
      let qMax = q0;
      for (let j = 0; j < count; j++) {
        const slot = q * MAX_RECONSTRUCTION_NEIGHBORS + j;
        const neighbor = stencil.neighborCell[slot]!;
        const qn = variableOf(cells[neighbor]!, variable);
        const delta = qn - q0;
        gx += stencil.coeffX[slot]! * delta;
        gy += stencil.coeffY[slot]! * delta;
        gz += stencil.coeffZ[slot]! * delta;
        qMin = Math.min(qMin, qn);
        qMax = Math.max(qMax, qn);
      }

      let limiter = 1;
      for (const xf of faces) {
        const delta = gx * (xf[0] - x0[0]) + gy * (xf[1] - x0[1]) + gz * (xf[2] - x0[2]);
        if (delta > 0) limiter = Math.min(limiter, (qMax - q0) / delta);
        else if (delta < 0) limiter = Math.min(limiter, (qMin - q0) / delta);
      }
      limiter = Math.max(0, Math.min(1, limiter));
      const [ox, oy, oz] = gradientArrays(primitive, variable);
      ox[q] = limiter * gx;
      oy[q] = limiter * gy;
      oz[q] = limiter * gz;
    }
  }

  return {
    primitive,
    pressurePerturbX: primitive.pX,
    pressurePerturbY: primitive.pY,
    pressurePerturbZ: primitive.pZ,
    pressurePerturb,
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
  const state = readState(fields, cell);
  const invRho = 1 / state.rho;
  const baseVelocity: Vec3 = [
    state.momentum[0] * invRho,
    state.momentum[1] * invRho,
    state.momentum[2] * invRho,
  ];
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
    state.rho,
    reconstruction.primitive.rhoX,
    reconstruction.primitive.rhoY,
    reconstruction.primitive.rhoZ,
  );
  const pressure = referenceFacePressure + apply(
    reconstruction.pressurePerturb[cell]!,
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
      apply(baseVelocity[0], reconstruction.primitive.uxX, reconstruction.primitive.uxY, reconstruction.primitive.uxZ),
      apply(baseVelocity[1], reconstruction.primitive.uyX, reconstruction.primitive.uyY, reconstruction.primitive.uyZ),
      apply(baseVelocity[2], reconstruction.primitive.uzX, reconstruction.primitive.uzY, reconstruction.primitive.uzZ),
    ],
    pressure,
  };
}
