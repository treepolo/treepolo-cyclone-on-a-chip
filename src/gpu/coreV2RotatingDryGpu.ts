import { EARTH, type PlanetConfig } from '../core/constants.js';
import { applyHeldSuarezForcingPackedF32 } from '../corev2/heldSuarezForcing.js';
import type { HydrostaticReference1D } from '../corev2/hydrostaticReference.js';
import type { LinearReconstructionStencil } from '../corev2/reconstruction.js';
import type { SphericalShellGeometry } from '../corev2/sphericalShellGeometry.js';
import { CoreV2GpuFoundation } from './coreV2GpuFoundation.js';
import {
  CoreV2GpuHeviImexSsp2,
  type CoreV2GpuHeviImexDiagnostics,
} from './coreV2HeviImexGpu.js';

type GPUAny = any;

export interface CoreV2GpuRotatingDryResult {
  packedState: Float32Array;
  diagnostics: CoreV2GpuHeviImexDiagnostics;
}

/**
 * Correctness-first Stage-4 GPU dry-core composition.
 *
 * The current Core v2 GPU path is intentionally still host-visible between
 * sub-operators, so this wrapper reuses the already hardware-gated exact
 * Coriolis kernel and IMEX-HEVI timestep instead of introducing another shader
 * architecture. Held-Suarez forcing is applied to the host-visible packed f32
 * state until a later performance pass makes the whole timestep buffer-resident.
 */
export class CoreV2GpuRotatingDryCore {
  private readonly geometry: SphericalShellGeometry;
  private readonly reference: HydrostaticReference1D;
  private readonly planet: PlanetConfig;
  private readonly foundation: CoreV2GpuFoundation;
  private readonly dynamics: CoreV2GpuHeviImexSsp2;

  constructor(
    device: GPUAny,
    geometry: SphericalShellGeometry,
    stencil: LinearReconstructionStencil,
    reference: HydrostaticReference1D,
    planet: PlanetConfig = EARTH,
  ) {
    this.geometry = geometry;
    this.reference = reference;
    this.planet = planet;
    this.foundation = new CoreV2GpuFoundation(device);
    this.dynamics = new CoreV2GpuHeviImexSsp2(
      device,
      geometry,
      stencil,
      reference,
      planet,
    );
  }

  async step(
    packedState: Float32Array,
    dt: number,
  ): Promise<CoreV2GpuRotatingDryResult> {
    if (!(dt > 0) || !Number.isFinite(dt)) {
      throw new Error(`invalid Core v2 GPU rotating dry-core timestep: ${dt}`);
    }

    const forcedInput = packedState.slice();
    applyHeldSuarezForcingPackedF32(
      forcedInput,
      0.5 * dt,
      this.geometry,
      this.reference,
    );

    const omega: readonly [number, number, number] = [0, 0, this.planet.omega];
    const rotatedInput = await this.foundation.applyExactCoriolis(
      forcedInput,
      omega,
      0.5 * dt,
    );
    const dynamics = await this.dynamics.step(rotatedInput, dt);
    const rotatedOutput = await this.foundation.applyExactCoriolis(
      dynamics.packedState,
      omega,
      0.5 * dt,
    );

    applyHeldSuarezForcingPackedF32(
      rotatedOutput,
      0.5 * dt,
      this.geometry,
      this.reference,
    );

    return {
      packedState: rotatedOutput,
      diagnostics: dynamics.diagnostics,
    };
  }

  destroy(): void {
    this.dynamics.destroy();
  }
}
