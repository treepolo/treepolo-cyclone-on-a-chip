import { DRY_AIR, EARTH, type PlanetConfig } from '../core/constants.js';
import type { HydrostaticReference1D } from '../corev2/hydrostaticReference.js';
import type { SphericalShellGeometry } from '../corev2/sphericalShellGeometry.js';
import {
  CoreV2GpuBlockTridiagonal5,
  type CoreV2GpuBlockTridiagonalBatch,
} from './coreV2BlockTridiagonalGpu.js';
import {
  buildCoreV2GpuHeviJacobianStaticData,
  CORE_V2_GPU_HEVI_JACOBIAN_FLOATS_PER_CELL,
  CoreV2GpuHeviJacobian,
} from './coreV2HeviJacobianGpu.js';
import {
  buildCoreV2GpuHeviVerticalStaticData,
  CoreV2GpuHeviVertical,
} from './coreV2HeviVerticalGpu.js';

const BLOCK = 5;
const BLOCK2 = BLOCK * BLOCK;
const PACKED = 8;

/**
 * Float32 cannot converge to the CPU f64 Newton tolerance of 2e-10. This is a
 * scaled residual tolerance for the correctness-first GPU stage. The hardware
 * gate also compares the converged state against the f64 CPU stage, so this
 * tolerance is not the only accuracy criterion.
 */
export const CORE_V2_GPU_HEVI_NEWTON_TOLERANCE = 2e-5;
export const CORE_V2_GPU_HEVI_NEWTON_MAX_ITERATIONS = 10;
export const CORE_V2_GPU_HEVI_LINE_SEARCH_MAX_HALVINGS = 10;

export interface CoreV2GpuHeviImplicitDiagnostics {
  iterations: number;
  initialResidual: number;
  finalResidual: number;
  totalLineSearchHalvings: number;
}

export interface CoreV2GpuHeviImplicitResult {
  packedState: Float32Array;
  diagnostics: CoreV2GpuHeviImplicitDiagnostics;
}

type GPUAny = any;

function f32(value: number): number {
  return Math.fround(value);
}

function componentScale(
  reference: HydrostaticReference1D,
  k: number,
  component: number,
): number {
  const rho = f32(reference.cellDensity[k]!);
  const pressure = f32(reference.cellPressure[k]!);
  if (component === 0) return Math.max(rho, 1e-8);
  if (component >= 1 && component <= 3) {
    const sound = f32(Math.sqrt(f32(f32(DRY_AIR.gamma) * f32(pressure / rho))));
    return Math.max(f32(rho * sound), 1e-6);
  }
  const gm1 = f32(f32(DRY_AIR.gamma) - 1);
  const internal = f32(pressure / gm1);
  const potential = f32(Math.abs(f32(rho * f32(reference.cellGeopotential[k]!))));
  return Math.max(f32(internal + potential), 1);
}

interface ResidualEvaluation {
  value: Float32Array;
  norm: number;
}

/**
 * Correctness-first GPU implementation of one nonlinear diagonally implicit
 * Core v2 HEVI stage
 *
 *   Y = B + alphaDt * V(Y).
 *
 * V(Y), dV/dY and the 5x5 block-tridiagonal linear solve all execute through
 * the already hardware-gated GPU operators. Residual reduction, Newton-system
 * assembly and line-search orchestration currently happen on the host between
 * those GPU calls. This deliberately establishes numerical agreement before a
 * later buffer-resident fusion removes the round trips; it does not introduce
 * a second physics formulation.
 */
export class CoreV2GpuHeviImplicitStage {
  private readonly geometry: SphericalShellGeometry;
  private readonly reference: HydrostaticReference1D;
  private readonly inverseVolume: Float32Array;
  private readonly vertical: CoreV2GpuHeviVertical;
  private readonly jacobian: CoreV2GpuHeviJacobian;
  private readonly blockSolver: CoreV2GpuBlockTridiagonal5;

  constructor(
    device: GPUAny,
    geometry: SphericalShellGeometry,
    reference: HydrostaticReference1D,
    planet: PlanetConfig = EARTH,
  ) {
    const cellCount = geometry.horizontal.cellCount * geometry.nz;
    if (reference.cellDensity.length !== geometry.nz) {
      throw new Error('Core v2 GPU implicit HEVI reference/geometry size mismatch');
    }
    this.geometry = geometry;
    this.reference = reference;
    this.inverseVolume = new Float32Array(cellCount);
    for (let q = 0; q < cellCount; q++) {
      this.inverseVolume[q] = f32(1 / geometry.cellVolume[q]!);
    }
    this.vertical = new CoreV2GpuHeviVertical(
      device,
      buildCoreV2GpuHeviVerticalStaticData(geometry, reference, planet),
    );
    this.jacobian = new CoreV2GpuHeviJacobian(
      device,
      buildCoreV2GpuHeviJacobianStaticData(geometry, reference, planet),
    );
    this.blockSolver = new CoreV2GpuBlockTridiagonal5(device);
  }

  private validateState(packed: Float32Array): void {
    const cellCount = this.geometry.horizontal.cellCount * this.geometry.nz;
    if (packed.length !== cellCount * PACKED) {
      throw new Error('Core v2 GPU implicit HEVI packed-state size mismatch');
    }
  }

  private async residual(
    state: Float32Array,
    base: Float32Array,
    alphaDt: number,
  ): Promise<ResidualEvaluation> {
    const integrated = await this.vertical.computeIntegratedRate(state);
    const cellCount = this.geometry.horizontal.cellCount * this.geometry.nz;
    const out = new Float32Array(cellCount * BLOCK);
    let norm = 0;
    for (let q = 0; q < cellCount; q++) {
      const k = q % this.geometry.nz;
      const invVolume = this.inverseVolume[q]!;
      for (let v = 0; v < BLOCK; v++) {
        const packedIndex = q * PACKED + v;
        const residualIndex = q * BLOCK + v;
        const tendency = f32(integrated[packedIndex]! * invVolume);
        const difference = f32(state[packedIndex]! - base[packedIndex]!);
        const r = f32(difference - f32(alphaDt * tendency));
        if (!Number.isFinite(r)) {
          return { value: out, norm: Number.POSITIVE_INFINITY };
        }
        out[residualIndex] = r;
        const scale = Math.max(
          componentScale(this.reference, k, v),
          Math.abs(base[packedIndex]!),
          Math.abs(state[packedIndex]!),
        );
        norm = Math.max(norm, Math.abs(r) / Math.max(scale, 1e-30));
      }
    }
    return { value: out, norm };
  }

  private assembleNewtonSystem(
    jacobian: Float32Array,
    residual: Float32Array,
    alphaDt: number,
  ): CoreV2GpuBlockTridiagonalBatch {
    const columnCount = this.geometry.horizontal.cellCount;
    const nz = this.geometry.nz;
    const blocks = columnCount * nz;
    if (jacobian.length !== blocks * CORE_V2_GPU_HEVI_JACOBIAN_FLOATS_PER_CELL) {
      throw new Error('Core v2 GPU implicit HEVI Jacobian size mismatch');
    }
    if (residual.length !== blocks * BLOCK) {
      throw new Error('Core v2 GPU implicit HEVI residual size mismatch');
    }

    const lower = new Float32Array(blocks * BLOCK2);
    const diagonal = new Float32Array(blocks * BLOCK2);
    const upper = new Float32Array(blocks * BLOCK2);
    const rhs = new Float32Array(blocks * BLOCK);
    const alpha = f32(alphaDt);

    for (let q = 0; q < blocks; q++) {
      const jBase = q * CORE_V2_GPU_HEVI_JACOBIAN_FLOATS_PER_CELL;
      const blockBase = q * BLOCK2;
      for (let row = 0; row < BLOCK; row++) {
        rhs[q * BLOCK + row] = f32(-residual[q * BLOCK + row]!);
        for (let col = 0; col < BLOCK; col++) {
          const i = row * BLOCK + col;
          lower[blockBase + i] = f32(-f32(alpha * jacobian[jBase + i]!));
          diagonal[blockBase + i] = f32(
            (row === col ? 1 : 0) - f32(alpha * jacobian[jBase + BLOCK2 + i]!),
          );
          upper[blockBase + i] = f32(
            -f32(alpha * jacobian[jBase + 2 * BLOCK2 + i]!),
          );
        }
      }
    }
    return { columnCount, nz, lower, diagonal, upper, rhs };
  }

  private candidate(
    state: Float32Array,
    delta: Float32Array,
    lambda: number,
  ): Float32Array | null {
    const cellCount = this.geometry.horizontal.cellCount * this.geometry.nz;
    if (delta.length !== cellCount * BLOCK) {
      throw new Error('Core v2 GPU implicit HEVI Newton correction size mismatch');
    }
    const out = state.slice();
    const l = f32(lambda);
    for (let q = 0; q < cellCount; q++) {
      for (let v = 0; v < BLOCK; v++) {
        const packedIndex = q * PACKED + v;
        const value = f32(state[packedIndex]! + f32(l * delta[q * BLOCK + v]!));
        if (!Number.isFinite(value)) return null;
        out[packedIndex] = value;
      }
    }
    return out;
  }

  async solve(
    packedBase: Float32Array,
    alphaDt: number,
  ): Promise<CoreV2GpuHeviImplicitResult> {
    this.validateState(packedBase);
    if (!(alphaDt >= 0) || !Number.isFinite(alphaDt)) {
      throw new Error(`invalid Core v2 GPU implicit stage alphaDt=${alphaDt}`);
    }
    const base = packedBase.slice();
    if (alphaDt === 0) {
      return {
        packedState: base,
        diagnostics: {
          iterations: 0,
          initialResidual: 0,
          finalResidual: 0,
          totalLineSearchHalvings: 0,
        },
      };
    }

    let state = base.slice();
    let current = await this.residual(state, base, alphaDt);
    const initialResidual = current.norm;
    let totalLineSearchHalvings = 0;
    if (current.norm <= CORE_V2_GPU_HEVI_NEWTON_TOLERANCE) {
      return {
        packedState: state,
        diagnostics: {
          iterations: 0,
          initialResidual,
          finalResidual: current.norm,
          totalLineSearchHalvings,
        },
      };
    }

    for (let iteration = 1; iteration <= CORE_V2_GPU_HEVI_NEWTON_MAX_ITERATIONS; iteration++) {
      const j = await this.jacobian.computeTendencyJacobian(state);
      const system = this.assembleNewtonSystem(j, current.value, alphaDt);
      const linear = await this.blockSolver.solve(system);
      let failedColumns = 0;
      for (const status of linear.status) if (status !== 0) failedColumns++;
      if (failedColumns > 0) {
        throw new Error(
          `Core v2 GPU implicit HEVI block solve failed in ${failedColumns} columns`,
        );
      }
      for (const value of linear.solution) {
        if (!Number.isFinite(value)) {
          throw new Error('Core v2 GPU implicit HEVI block solve returned non-finite correction');
        }
      }

      let acceptedState: Float32Array | null = null;
      let acceptedResidual: ResidualEvaluation | null = null;
      let lambda = 1;
      for (let half = 0; half <= CORE_V2_GPU_HEVI_LINE_SEARCH_MAX_HALVINGS; half++) {
        const trial = this.candidate(state, linear.solution, lambda);
        if (trial) {
          const trialResidual = await this.residual(trial, base, alphaDt);
          if (Number.isFinite(trialResidual.norm) && trialResidual.norm < current.norm) {
            acceptedState = trial;
            acceptedResidual = trialResidual;
            totalLineSearchHalvings += half;
            break;
          }
        }
        lambda *= 0.5;
      }
      if (!acceptedState || !acceptedResidual) {
        throw new Error(
          `Core v2 GPU HEVI Newton line search failed at iteration ${iteration}; residual=${current.norm}`,
        );
      }
      state = acceptedState;
      current = acceptedResidual;
      if (current.norm <= CORE_V2_GPU_HEVI_NEWTON_TOLERANCE) {
        return {
          packedState: state,
          diagnostics: {
            iterations: iteration,
            initialResidual,
            finalResidual: current.norm,
            totalLineSearchHalvings,
          },
        };
      }
    }

    throw new Error(
      `Core v2 GPU HEVI Newton did not converge: initial=${initialResidual}, final=${current.norm}`,
    );
  }

  destroy(): void {
    this.vertical.destroy();
    this.jacobian.destroy();
    this.blockSolver.destroy();
  }
}
