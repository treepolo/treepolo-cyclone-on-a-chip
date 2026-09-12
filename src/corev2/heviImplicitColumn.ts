import { DRY_AIR, EARTH, type PlanetConfig } from '../core/constants.js';
import { solveBlockTridiagonal5 } from './blockTridiagonal5.js';
import {
  CONSERVED_COMPONENTS,
  columnStateIndex,
  verticalStiffColumnTendency,
} from './heviColumn.js';
import type { HydrostaticReference1D } from './hydrostaticReference.js';
import type { SphericalShellGeometry } from './sphericalShellGeometry.js';

const BLOCK = CONSERVED_COMPONENTS;
const BLOCK2 = BLOCK * BLOCK;
const SQRT_EPSILON = Math.sqrt(Number.EPSILON);
const NEWTON_TOLERANCE = 2e-10;
const NEWTON_MAX_ITERATIONS = 10;
const LINE_SEARCH_MAX_HALVINGS = 10;

export interface VerticalImplicitSolveDiagnostics {
  iterations: number;
  initialResidual: number;
  finalResidual: number;
}

export interface VerticalImplicitSolveResult {
  state: Float64Array;
  diagnostics: VerticalImplicitSolveDiagnostics;
}

function componentScale(
  reference: HydrostaticReference1D,
  k: number,
  component: number,
): number {
  const rho = reference.cellDensity[k]!;
  const pressure = reference.cellPressure[k]!;
  if (component === 0) return Math.max(rho, 1e-8);
  if (component >= 1 && component <= 3) {
    const soundSpeed = Math.sqrt(DRY_AIR.gamma * pressure / rho);
    return Math.max(rho * soundSpeed, 1e-6);
  }
  const internal = pressure / (DRY_AIR.gamma - 1);
  const potential = Math.abs(rho * reference.cellGeopotential[k]!);
  return Math.max(internal + potential, 1);
}

function residual(
  state: Float64Array,
  base: Float64Array,
  alphaDt: number,
  horizontalCell: number,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  planet: PlanetConfig,
): { value: Float64Array; stiff: Float64Array; norm: number } {
  const stiff = verticalStiffColumnTendency(
    state,
    horizontalCell,
    geometry,
    reference,
    planet,
  );
  const value = new Float64Array(state.length);
  let norm = 0;
  for (let k = 0; k < geometry.nz; k++) {
    for (let v = 0; v < BLOCK; v++) {
      const i = columnStateIndex(k, v);
      const r = state[i]! - base[i]! - alphaDt * stiff[i]!;
      value[i] = r;
      const scale = Math.max(
        componentScale(reference, k, v),
        Math.abs(base[i]!),
        Math.abs(state[i]!),
      );
      norm = Math.max(norm, Math.abs(r) / scale);
    }
  }
  return { value, stiff, norm };
}

function finiteDifferenceStep(
  state: Float64Array,
  reference: HydrostaticReference1D,
  k: number,
  component: number,
): number {
  const value = state[columnStateIndex(k, component)]!;
  const scale = Math.max(Math.abs(value), componentScale(reference, k, component));
  return SQRT_EPSILON * scale;
}

function buildNewtonSystem(
  state: Float64Array,
  baseResidual: Float64Array,
  baseStiff: Float64Array,
  alphaDt: number,
  horizontalCell: number,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  planet: PlanetConfig,
) {
  const nz = geometry.nz;
  const lower = new Float64Array(nz * BLOCK2);
  const diagonal = new Float64Array(nz * BLOCK2);
  const upper = new Float64Array(nz * BLOCK2);
  const rhs = new Float64Array(nz * BLOCK);

  for (let k = 0; k < nz; k++) {
    for (let v = 0; v < BLOCK; v++) {
      diagonal[k * BLOCK2 + v * BLOCK + v] = 1;
      rhs[columnStateIndex(k, v)] = -baseResidual[columnStateIndex(k, v)]!;
    }
  }

  for (let sourceK = 0; sourceK < nz; sourceK++) {
    for (let sourceVar = 0; sourceVar < BLOCK; sourceVar++) {
      const sourceIndex = columnStateIndex(sourceK, sourceVar);
      const h = finiteDifferenceStep(state, reference, sourceK, sourceVar);
      const perturbed = state.slice();
      perturbed[sourceIndex] = perturbed[sourceIndex]! + h;
      const perturbedStiff = verticalStiffColumnTendency(
        perturbed,
        horizontalCell,
        geometry,
        reference,
        planet,
      );

      const rowBegin = Math.max(0, sourceK - 1);
      const rowEnd = Math.min(nz - 1, sourceK + 1);
      for (let rowK = rowBegin; rowK <= rowEnd; rowK++) {
        let block: Float64Array;
        let blockOffset: number;
        if (sourceK === rowK - 1) {
          block = lower;
          blockOffset = rowK * BLOCK2;
        } else if (sourceK === rowK) {
          block = diagonal;
          blockOffset = rowK * BLOCK2;
        } else if (sourceK === rowK + 1) {
          block = upper;
          blockOffset = rowK * BLOCK2;
        } else {
          continue;
        }
        for (let rowVar = 0; rowVar < BLOCK; rowVar++) {
          const rowIndex = columnStateIndex(rowK, rowVar);
          const derivative = (perturbedStiff[rowIndex]! - baseStiff[rowIndex]!) / h;
          block[blockOffset + rowVar * BLOCK + sourceVar] =
            block[blockOffset + rowVar * BLOCK + sourceVar]! - alphaDt * derivative;
        }
      }
    }
  }

  return { lower, diagonal, upper, rhs };
}

/**
 * Solve one diagonally-implicit HEVI stage
 *
 *     Y = B + alphaDt * V(Y)
 *
 * for a single atmospheric column. V is exactly the column-local stiff rate in
 * `heviColumn.ts`. Newton's Jacobian is block-tridiagonal because V only couples
 * adjacent vertical cells; the linear solve is therefore O(nz) and does not
 * introduce a global pressure solve.
 *
 * The finite-difference Jacobian is currently the CPU implementation of the
 * same nonlinear stage equation. It is not a second/reference physics core;
 * the GPU path can replace only this Jacobian-evaluation detail with an analytic
 * block Jacobian while solving the identical stage equation.
 */
export function solveVerticalImplicitColumnStage(
  base: ArrayLike<number>,
  alphaDt: number,
  horizontalCell: number,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  planet: PlanetConfig = EARTH,
): VerticalImplicitSolveResult {
  const expectedLength = geometry.nz * BLOCK;
  if (base.length !== expectedLength) {
    throw new Error('Core v2 implicit HEVI base column length mismatch');
  }
  if (!(alphaDt >= 0) || !Number.isFinite(alphaDt)) {
    throw new Error(`invalid Core v2 implicit stage alphaDt=${alphaDt}`);
  }
  const baseState = new Float64Array(base);
  if (alphaDt === 0) {
    return {
      state: baseState,
      diagnostics: { iterations: 0, initialResidual: 0, finalResidual: 0 },
    };
  }

  let state = baseState.slice();
  let current = residual(
    state,
    baseState,
    alphaDt,
    horizontalCell,
    geometry,
    reference,
    planet,
  );
  const initialResidual = current.norm;
  if (current.norm <= NEWTON_TOLERANCE) {
    return {
      state,
      diagnostics: { iterations: 0, initialResidual, finalResidual: current.norm },
    };
  }

  for (let iteration = 1; iteration <= NEWTON_MAX_ITERATIONS; iteration++) {
    const system = buildNewtonSystem(
      state,
      current.value,
      current.stiff,
      alphaDt,
      horizontalCell,
      geometry,
      reference,
      planet,
    );
    const delta = solveBlockTridiagonal5(system);

    let accepted = false;
    let trialResidual = current;
    let trialState = state;
    let lambda = 1;
    for (let half = 0; half <= LINE_SEARCH_MAX_HALVINGS; half++) {
      const candidate = state.slice();
      for (let i = 0; i < candidate.length; i++) {
        candidate[i] = candidate[i]! + lambda * delta[i]!;
      }
      try {
        const candidateResidual = residual(
          candidate,
          baseState,
          alphaDt,
          horizontalCell,
          geometry,
          reference,
          planet,
        );
        if (candidateResidual.norm < current.norm) {
          trialState = candidate;
          trialResidual = candidateResidual;
          accepted = true;
          break;
        }
      } catch {
        // Candidate crossed a positive-density/pressure boundary. Reduce the
        // Newton step; this changes only the nonlinear solve path, not V(Y).
      }
      lambda *= 0.5;
    }

    if (!accepted) {
      throw new Error(
        `Core v2 HEVI Newton line search failed at iteration ${iteration}; residual=${current.norm}`,
      );
    }
    state = trialState;
    current = trialResidual;
    if (current.norm <= NEWTON_TOLERANCE) {
      return {
        state,
        diagnostics: {
          iterations: iteration,
          initialResidual,
          finalResidual: current.norm,
        },
      };
    }
  }

  throw new Error(
    `Core v2 HEVI Newton did not converge: initial=${initialResidual}, final=${current.norm}`,
  );
}

export function verticalImplicitStageResidualNorm(
  state: Float64Array,
  base: Float64Array,
  alphaDt: number,
  horizontalCell: number,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  planet: PlanetConfig = EARTH,
): number {
  return residual(
    state,
    base,
    alphaDt,
    horizontalCell,
    geometry,
    reference,
    planet,
  ).norm;
}
