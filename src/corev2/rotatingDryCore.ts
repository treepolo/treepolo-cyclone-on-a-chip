import { EARTH, type PlanetConfig } from '../core/constants.js';
import {
  cellCountOf,
  createConservativeFields,
  type ConservativeFields,
} from './fields.js';
import { applyHeldSuarezForcing } from './heldSuarezForcing.js';
import type { HydrostaticReference1D } from './hydrostaticReference.js';
import type { LinearReconstructionStencil } from './reconstruction.js';
import {
  rotatingHeviImexSsp2Step,
  type RotatingHeviStepResult,
} from './rotatingHevi.js';
import type { SphericalShellGeometry } from './sphericalShellGeometry.js';

function cloneFields(fields: ConservativeFields): ConservativeFields {
  const precision = fields.rho instanceof Float32Array ? 'f32' : 'f64';
  const out = createConservativeFields(cellCountOf(fields), precision);
  out.rho.set(fields.rho);
  out.momX.set(fields.momX);
  out.momY.set(fields.momY);
  out.momZ.set(fields.momZ);
  out.rhoE.set(fields.rhoE);
  return out;
}

/**
 * Stage-4 dry global timestep.
 *
 * Held-Suarez forcing is Strang-split around the already second-order rotating
 * conservative HEVI dynamics:
 *
 *   F(dt/2) -> C(dt/2) -> H(dt) -> C(dt/2) -> F(dt/2)
 *
 * F is Newtonian thermal relaxation plus near-surface tangential Rayleigh drag,
 * C is exact Coriolis rotation, and H is the conservative Euler+gravity HEVI
 * advance. No numerical damping, sponge, or global fixer is introduced here.
 */
export function rotatingHeldSuarezStep(
  fields: ConservativeFields,
  dt: number,
  geometry: SphericalShellGeometry,
  stencil: LinearReconstructionStencil,
  reference: HydrostaticReference1D,
  planet: PlanetConfig = EARTH,
): RotatingHeviStepResult {
  if (!(dt > 0) || !Number.isFinite(dt)) {
    throw new Error(`invalid Core v2 rotating dry-core timestep: ${dt}`);
  }
  const work = cloneFields(fields);
  applyHeldSuarezForcing(work, 0.5 * dt, geometry, reference);
  const result = rotatingHeviImexSsp2Step(
    work,
    dt,
    geometry,
    stencil,
    reference,
    planet,
  );
  applyHeldSuarezForcing(result.fields, 0.5 * dt, geometry, reference);
  return result;
}
