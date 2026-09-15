import { EARTH, type PlanetConfig } from '../core/constants.js';
import {
  cellCountOf,
  createConservativeFields,
  type ConservativeFields,
} from './fields.js';
import type { IntegratedConservativeRate } from './finiteVolume.js';
import {
  extractColumnState,
  writeColumnState,
} from './heviColumn.js';
import { solveVerticalImplicitColumnStage } from './heviImplicitColumn.js';
import {
  closedShellVerticalStiffRateFirstOrder,
} from './heviSplit.js';
import type { HydrostaticReference1D } from './hydrostaticReference.js';
import type { LinearReconstructionStencil } from './reconstruction.js';
import {
  pressureFromConservedRelativeToReference,
} from './state.js';
import {
  shellCellIndex,
  type SphericalShellGeometry,
} from './sphericalShellGeometry.js';
import { closedShellWellBalancedEulerGravityRate } from './wellBalancedEulerGravity.js';

/** Pareschi-Russo IMEX-SSP2(2,2,2) diagonal coefficient. */
export const IMEX_SSP2_GAMMA = 1 - 1 / Math.sqrt(2);

export interface HeviImexStepDiagnostics {
  maxNewtonIterationsStage1: number;
  maxNewtonIterationsStage2: number;
  maxImplicitResidualStage1: number;
  maxImplicitResidualStage2: number;
}

export interface HeviImexStepResult {
  fields: ConservativeFields;
  diagnostics: HeviImexStepDiagnostics;
}

function cloneFields(fields: ConservativeFields): ConservativeFields {
  const n = cellCountOf(fields);
  const out = createConservativeFields(n, 'f64');
  out.rho.set(fields.rho);
  out.momX.set(fields.momX);
  out.momY.set(fields.momY);
  out.momZ.set(fields.momZ);
  out.rhoE.set(fields.rhoE);
  return out;
}

function rateComponent(
  rate: IntegratedConservativeRate,
  cell: number,
  component: number,
): number {
  if (component === 0) return rate.rho[cell]!;
  if (component === 1) return rate.momX[cell]!;
  if (component === 2) return rate.momY[cell]!;
  if (component === 3) return rate.momZ[cell]!;
  return rate.rhoE[cell]!;
}

function fieldComponent(
  fields: ConservativeFields,
  cell: number,
  component: number,
): number {
  if (component === 0) return fields.rho[cell]!;
  if (component === 1) return fields.momX[cell]!;
  if (component === 2) return fields.momY[cell]!;
  if (component === 3) return fields.momZ[cell]!;
  return fields.rhoE[cell]!;
}

function setFieldComponent(
  fields: ConservativeFields,
  cell: number,
  component: number,
  value: number,
): void {
  if (component === 0) fields.rho[cell] = value;
  else if (component === 1) fields.momX[cell] = value;
  else if (component === 2) fields.momY[cell] = value;
  else if (component === 3) fields.momZ[cell] = value;
  else fields.rhoE[cell] = value;
}

function validateState(
  fields: ConservativeFields,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
): void {
  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const rho = fields.rho[q]!;
      if (!(rho > 0) || !Number.isFinite(rho)) {
        throw new Error(`Core v2 IMEX produced invalid density at cell ${q}: ${rho}`);
      }
      pressureFromConservedRelativeToReference({
        rho,
        momentum: [fields.momX[q]!, fields.momY[q]!, fields.momZ[q]!],
        rhoE: fields.rhoE[q]!,
      }, reference.cellDensity[k]!, reference.cellPressure[k]!, reference.cellGeopotential[k]!);
    }
  }
}

function solveImplicitAllColumns(
  baseFields: ConservativeFields,
  alphaDt: number,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  planet: PlanetConfig,
): {
  fields: ConservativeFields;
  maxIterations: number;
  maxResidual: number;
} {
  const out = cloneFields(baseFields);
  let maxIterations = 0;
  let maxResidual = 0;
  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    const baseColumn = extractColumnState(baseFields, c, geometry.nz);
    const solved = solveVerticalImplicitColumnStage(
      baseColumn,
      alphaDt,
      c,
      geometry,
      reference,
      planet,
    );
    writeColumnState(out, c, geometry.nz, solved.state);
    maxIterations = Math.max(maxIterations, solved.diagnostics.iterations);
    maxResidual = Math.max(maxResidual, solved.diagnostics.finalResidual);
  }
  validateState(out, geometry, reference);
  return { fields: out, maxIterations, maxResidual };
}

function buildSecondStageBase(
  original: ConservativeFields,
  fullStage1: IntegratedConservativeRate,
  stiffStage1: IntegratedConservativeRate,
  dt: number,
  geometry: SphericalShellGeometry,
): ConservativeFields {
  const out = cloneFields(original);
  const stiffCoefficient = 2 * IMEX_SSP2_GAMMA;
  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const volume = geometry.cellVolume[q]!;
      const factor = dt / volume;
      for (let v = 0; v < 5; v++) {
        const increment = factor * (
          rateComponent(fullStage1, q, v) -
          stiffCoefficient * rateComponent(stiffStage1, q, v)
        );
        setFieldComponent(out, q, v, fieldComponent(original, q, v) + increment);
      }
    }
  }
  return out;
}

function finalizeSecondOrderStep(
  original: ConservativeFields,
  fullStage1: IntegratedConservativeRate,
  fullStage2: IntegratedConservativeRate,
  dt: number,
  geometry: SphericalShellGeometry,
): ConservativeFields {
  const out = cloneFields(original);
  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const factor = 0.5 * dt / geometry.cellVolume[q]!;
      for (let v = 0; v < 5; v++) {
        const next = fieldComponent(original, q, v) + factor * (
          rateComponent(fullStage1, q, v) + rateComponent(fullStage2, q, v)
        );
        setFieldComponent(out, q, v, next);
      }
    }
  }
  return out;
}

/**
 * One second-order conservative HEVI timestep for Euler + gravity.
 *
 * This is IMEX-SSP2(2,2,2), gamma=1-1/sqrt(2):
 *
 *   Y1 = U^n + gamma dt V(Y1)
 *   Y2 = U^n + dt E(Y1) + (1-2gamma)dt V(Y1) + gamma dt V(Y2)
 *   U+ = U^n + dt/2 [E(Y1)+V(Y1)+E(Y2)+V(Y2)]
 *
 * Because FULL=E+V exactly, stage 2 is assembled as
 * U^n + dt[FULL(Y1)-2gamma V(Y1)], and the final update uses only the two
 * conservative FULL shared-face rates. No acoustic substeps, off-centering,
 * divergence damping, sponge, or global fixer is present.
 */
export function heviImexSsp2Step(
  fields: ConservativeFields,
  dt: number,
  geometry: SphericalShellGeometry,
  stencil: LinearReconstructionStencil,
  reference: HydrostaticReference1D,
  planet: PlanetConfig = EARTH,
): HeviImexStepResult {
  const expectedCells = geometry.horizontal.cellCount * geometry.nz;
  if (cellCountOf(fields) !== expectedCells) {
    throw new Error('Core v2 IMEX field/geometry cell count mismatch');
  }
  if (stencil.geometry !== geometry) {
    throw new Error('Core v2 IMEX reconstruction stencil belongs to another geometry');
  }
  if (!(dt > 0) || !Number.isFinite(dt)) {
    throw new Error(`invalid Core v2 IMEX timestep: ${dt}`);
  }
  validateState(fields, geometry, reference);
  const original = cloneFields(fields);

  const stage1 = solveImplicitAllColumns(
    original,
    IMEX_SSP2_GAMMA * dt,
    geometry,
    reference,
    planet,
  );
  const full1 = closedShellWellBalancedEulerGravityRate(
    stage1.fields,
    geometry,
    stencil,
    reference,
    planet,
  );
  const stiff1 = closedShellVerticalStiffRateFirstOrder(
    stage1.fields,
    geometry,
    reference,
    planet,
  );

  const stage2Base = buildSecondStageBase(
    original,
    full1,
    stiff1,
    dt,
    geometry,
  );
  validateState(stage2Base, geometry, reference);
  const stage2 = solveImplicitAllColumns(
    stage2Base,
    IMEX_SSP2_GAMMA * dt,
    geometry,
    reference,
    planet,
  );
  const full2 = closedShellWellBalancedEulerGravityRate(
    stage2.fields,
    geometry,
    stencil,
    reference,
    planet,
  );

  const next = finalizeSecondOrderStep(original, full1, full2, dt, geometry);
  validateState(next, geometry, reference);
  return {
    fields: next,
    diagnostics: {
      maxNewtonIterationsStage1: stage1.maxIterations,
      maxNewtonIterationsStage2: stage2.maxIterations,
      maxImplicitResidualStage1: stage1.maxResidual,
      maxImplicitResidualStage2: stage2.maxResidual,
    },
  };
}
