import { dot3, norm3, type Vec3 } from '../core/math.js';
import { primitiveFromConserved, type ConservativeCell } from './state.js';
import type { FaceFlux } from './finiteVolume.js';

/**
 * Exact physical Euler flux for one state through an oriented unit face.
 *
 * This is not a Riemann solver and does not choose the time integrator.
 * It is the common physical flux definition that every Core v2 spatial
 * discretization must reduce to when both sides of a face are identical.
 */
export function physicalEulerFlux(
  state: ConservativeCell,
  unitNormal: Vec3,
  geopotential = 0,
): FaceFlux {
  const normalLength = norm3(unitNormal);
  if (Math.abs(normalLength - 1) > 1e-12) {
    throw new Error(`face normal must be unit length; got ${normalLength}`);
  }

  const primitive = primitiveFromConserved(state, geopotential);
  const un = dot3(primitive.velocity, unitNormal);
  const mass = state.rho * un;

  return {
    mass,
    momentum: [
      state.momentum[0] * un + primitive.pressure * unitNormal[0],
      state.momentum[1] * un + primitive.pressure * unitNormal[1],
      state.momentum[2] * un + primitive.pressure * unitNormal[2],
    ],
    totalEnergy: (state.rhoE + primitive.pressure) * un,
  };
}
