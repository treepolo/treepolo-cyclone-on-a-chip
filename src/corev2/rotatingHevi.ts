import { EARTH, type PlanetConfig } from '../core/constants.js';
import {
  cellCountOf,
  createConservativeFields,
  type ConservativeFields,
} from './fields.js';
import {
  heviImexSsp2Step,
  type HeviImexStepDiagnostics,
} from './heviImex.js';
import type { HydrostaticReference1D } from './hydrostaticReference.js';
import type { LinearReconstructionStencil } from './reconstruction.js';
import { applyExactCoriolis } from './rotation.js';
import type { SphericalShellGeometry } from './sphericalShellGeometry.js';

export interface RotatingHeviStepResult {
  fields: ConservativeFields;
  diagnostics: HeviImexStepDiagnostics;
}

function cloneFields(fields: ConservativeFields): ConservativeFields {
  const out = createConservativeFields(cellCountOf(fields), 'f64');
  out.rho.set(fields.rho);
  out.momX.set(fields.momX);
  out.momY.set(fields.momY);
  out.momZ.set(fields.momZ);
  out.rhoE.set(fields.rhoE);
  return out;
}

/**
 * Second-order closed rotating dry-core timestep.
 *
 * The non-rotating Euler+gravity advance is the conservative IMEX-HEVI step.
 * Coriolis is integrated exactly as a momentum-space rotation and composed by
 * Strang splitting:
 *
 *   U(1)   = C(dt/2) U^n
 *   U(2)   = H(dt)   U(1)
 *   U^{n+1}= C(dt/2) U(2)
 *
 * where C is the exact flow of dm/dt=-2 Omega x m and H is the second-order
 * conservative HEVI flow for Euler+gravity. C changes neither rho nor rhoE and
 * preserves |rho u| exactly up to roundoff, so rotation cannot create or remove
 * total energy. No drag, sponge, divergence damping, or global fixer is present.
 *
 * The planet rotation axis is the global +z axis by convention.
 */
export function rotatingHeviImexSsp2Step(
  fields: ConservativeFields,
  dt: number,
  geometry: SphericalShellGeometry,
  stencil: LinearReconstructionStencil,
  reference: HydrostaticReference1D,
  planet: PlanetConfig = EARTH,
): RotatingHeviStepResult {
  if (!(dt > 0) || !Number.isFinite(dt)) {
    throw new Error(`invalid Core v2 rotating timestep: ${dt}`);
  }
  const expectedCells = geometry.horizontal.cellCount * geometry.nz;
  if (cellCountOf(fields) !== expectedCells) {
    throw new Error('Core v2 rotating HEVI field/geometry cell count mismatch');
  }

  const rotatedInput = cloneFields(fields);
  applyExactCoriolis(rotatedInput, [0, 0, planet.omega], 0.5 * dt);

  const hevi = heviImexSsp2Step(
    rotatedInput,
    dt,
    geometry,
    stencil,
    reference,
    planet,
  );

  applyExactCoriolis(hevi.fields, [0, 0, planet.omega], 0.5 * dt);
  return {
    fields: hevi.fields,
    diagnostics: hevi.diagnostics,
  };
}
