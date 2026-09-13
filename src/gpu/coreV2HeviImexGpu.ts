import { EARTH, type PlanetConfig } from '../core/constants.js';
import type { HydrostaticReference1D } from '../corev2/hydrostaticReference.js';
import type { LinearReconstructionStencil } from '../corev2/reconstruction.js';
import { pressureFromConservedRelativeToReference } from '../corev2/state.js';
import type { SphericalShellGeometry } from '../corev2/sphericalShellGeometry.js';
import {
  buildCoreV2GpuInternalFaceStaticData,
  CoreV2GpuInternalFaceFlux,
} from './coreV2FaceFluxGpu.js';
import { CoreV2GpuHeviImplicitStage } from './coreV2HeviImplicitGpu.js';
import {
  buildCoreV2GpuHeviVerticalStaticData,
  CoreV2GpuHeviVertical,
} from './coreV2HeviVerticalGpu.js';
import {
  buildCoreV2GpuRateStaticData,
  CoreV2GpuRateGather,
} from './coreV2RateGatherGpu.js';
import {
  buildCoreV2GpuReconstructionStaticData,
  CoreV2GpuReconstruction,
} from './coreV2ReconstructionGpu.js';

const PACKED = 8;
const PHYSICAL = 5;
const GAMMA = 1 - 1 / Math.sqrt(2);
const f32 = Math.fround;
type GPUAny = any;

export interface CoreV2GpuHeviImexDiagnostics {
  stage1Iterations: number;
  stage2Iterations: number;
  stage1Residual: number;
  stage2Residual: number;
  stage1LineSearchHalvings: number;
  stage2LineSearchHalvings: number;
}

export interface CoreV2GpuHeviImexResult {
  packedState: Float32Array;
  diagnostics: CoreV2GpuHeviImexDiagnostics;
}

/**
 * Correctness-first GPU composition of the same Pareschi-Russo
 * IMEX-SSP2(2,2,2) HEVI step used by `corev2/heviImex.ts`.
 *
 * This class deliberately composes the already hardware-gated GPU spatial
 * operator, vertical stiff operator and nonlinear implicit stage before any
 * buffer-resident fusion. It therefore still performs host-visible readbacks
 * between sub-operators, but the numerical method and conservative update are
 * the production Core v2 equations rather than a separate GPU approximation.
 */
export class CoreV2GpuHeviImexSsp2 {
  private readonly geometry: SphericalShellGeometry;
  private readonly reference: HydrostaticReference1D;
  private readonly inverseVolume: Float32Array;
  private readonly reconstruction: CoreV2GpuReconstruction;
  private readonly faces: CoreV2GpuInternalFaceFlux;
  private readonly gather: CoreV2GpuRateGather;
  private readonly stiff: CoreV2GpuHeviVertical;
  private readonly implicit: CoreV2GpuHeviImplicitStage;

  constructor(
    device: GPUAny,
    geometry: SphericalShellGeometry,
    stencil: LinearReconstructionStencil,
    reference: HydrostaticReference1D,
    planet: PlanetConfig = EARTH,
  ) {
    if (stencil.geometry !== geometry) {
      throw new Error('Core v2 GPU IMEX reconstruction stencil belongs to another geometry');
    }
    if (reference.cellDensity.length !== geometry.nz) {
      throw new Error('Core v2 GPU IMEX reference/geometry size mismatch');
    }
    this.geometry = geometry;
    this.reference = reference;
    const cells = geometry.horizontal.cellCount * geometry.nz;
    this.inverseVolume = new Float32Array(cells);
    for (let q = 0; q < cells; q++) this.inverseVolume[q] = f32(1 / geometry.cellVolume[q]!);

    const reconstructionData = buildCoreV2GpuReconstructionStaticData(geometry, stencil, reference);
    const faceData = buildCoreV2GpuInternalFaceStaticData(geometry, stencil, reference);
    const rateData = buildCoreV2GpuRateStaticData(geometry, stencil, reference, faceData, planet);
    this.reconstruction = new CoreV2GpuReconstruction(device, reconstructionData);
    this.faces = new CoreV2GpuInternalFaceFlux(device, faceData);
    this.gather = new CoreV2GpuRateGather(device, rateData);
    this.stiff = new CoreV2GpuHeviVertical(
      device,
      buildCoreV2GpuHeviVerticalStaticData(geometry, reference, planet),
    );
    this.implicit = new CoreV2GpuHeviImplicitStage(device, geometry, reference, planet);
  }

  private validatePacked(state: Float32Array): void {
    const cells = this.geometry.horizontal.cellCount * this.geometry.nz;
    if (state.length !== cells * PACKED) {
      throw new Error('Core v2 GPU IMEX packed-state size mismatch');
    }
    for (let q = 0; q < cells; q++) {
      const k = q % this.geometry.nz;
      const i = q * PACKED;
      const rho = state[i]!;
      if (!(rho > 0) || !Number.isFinite(rho)) {
        throw new Error(`Core v2 GPU IMEX invalid density at cell ${q}: ${rho}`);
      }
      pressureFromConservedRelativeToReference({
        rho,
        momentum: [state[i + 1]!, state[i + 2]!, state[i + 3]!],
        rhoE: state[i + 4]!,
      }, this.reference.cellDensity[k]!, this.reference.cellPressure[k]!, this.reference.cellGeopotential[k]!);
    }
  }

  private async fullIntegratedRate(state: Float32Array): Promise<Float32Array> {
    const gradients = await this.reconstruction.computeGradients(state);
    const fluxes = await this.faces.compute(state, gradients);
    return await this.gather.compute(state, gradients, fluxes);
  }

  private buildStage2Base(
    original: Float32Array,
    full1: Float32Array,
    stiff1: Float32Array,
    dt: number,
  ): Float32Array {
    const cells = this.geometry.horizontal.cellCount * this.geometry.nz;
    const out = original.slice();
    const stiffCoefficient = f32(2 * GAMMA);
    const step = f32(dt);
    for (let q = 0; q < cells; q++) {
      const factor = f32(step * this.inverseVolume[q]!);
      const i = q * PACKED;
      for (let v = 0; v < PHYSICAL; v++) {
        const combined = f32(full1[i + v]! - f32(stiffCoefficient * stiff1[i + v]!));
        out[i + v] = f32(original[i + v]! + f32(factor * combined));
      }
    }
    return out;
  }

  private finalize(
    original: Float32Array,
    full1: Float32Array,
    full2: Float32Array,
    dt: number,
  ): Float32Array {
    const cells = this.geometry.horizontal.cellCount * this.geometry.nz;
    const out = original.slice();
    const halfDt = f32(0.5 * dt);
    for (let q = 0; q < cells; q++) {
      const factor = f32(halfDt * this.inverseVolume[q]!);
      const i = q * PACKED;
      for (let v = 0; v < PHYSICAL; v++) {
        const sum = f32(full1[i + v]! + full2[i + v]!);
        out[i + v] = f32(original[i + v]! + f32(factor * sum));
      }
    }
    return out;
  }

  async step(packedState: Float32Array, dt: number): Promise<CoreV2GpuHeviImexResult> {
    this.validatePacked(packedState);
    if (!(dt > 0) || !Number.isFinite(dt)) {
      throw new Error(`invalid Core v2 GPU IMEX timestep: ${dt}`);
    }
    const original = packedState.slice();
    const alphaDt = GAMMA * dt;

    const stage1 = await this.implicit.solve(original, alphaDt);
    const full1 = await this.fullIntegratedRate(stage1.packedState);
    const stiff1 = await this.stiff.computeIntegratedRate(stage1.packedState);

    const stage2Base = this.buildStage2Base(original, full1, stiff1, dt);
    this.validatePacked(stage2Base);
    const stage2 = await this.implicit.solve(stage2Base, alphaDt);
    const full2 = await this.fullIntegratedRate(stage2.packedState);

    const next = this.finalize(original, full1, full2, dt);
    this.validatePacked(next);
    return {
      packedState: next,
      diagnostics: {
        stage1Iterations: stage1.diagnostics.iterations,
        stage2Iterations: stage2.diagnostics.iterations,
        stage1Residual: stage1.diagnostics.finalResidual,
        stage2Residual: stage2.diagnostics.finalResidual,
        stage1LineSearchHalvings: stage1.diagnostics.totalLineSearchHalvings,
        stage2LineSearchHalvings: stage2.diagnostics.totalLineSearchHalvings,
      },
    };
  }

  destroy(): void {
    this.reconstruction.destroy();
    this.faces.destroy();
    this.gather.destroy();
    this.stiff.destroy();
    this.implicit.destroy();
  }
}

export const CORE_V2_GPU_IMEX_SSP2_GAMMA = GAMMA;
