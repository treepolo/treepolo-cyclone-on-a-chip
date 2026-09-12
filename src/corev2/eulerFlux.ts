import { dot3, norm3, scale3, type Vec3 } from '../core/math.js';
import { primitiveFromConserved, type ConservativeCell } from './state.js';
import type { FaceFlux, IntegratedFaceFlux } from './finiteVolume.js';

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

/**
 * Face-integrated physical Euler flux using the geometric vector area
 * A = integral_face n dA.
 *
 * For a constant state this is the exact surface integral even on a curved
 * spherical radial face. This is the geometric form used by Core v2 to make
 * uniform states independent of cubed-sphere non-orthogonality.
 */
export function integratedPhysicalEulerFlux(
  state: ConservativeCell,
  vectorArea: Vec3,
  geopotential = 0,
): IntegratedFaceFlux {
  const area = norm3(vectorArea);
  if (!(area > 0) || !Number.isFinite(area)) {
    throw new Error(`invalid face vector area magnitude: ${area}`);
  }
  const flux = physicalEulerFlux(state, scale3(vectorArea, 1 / area), geopotential);
  return {
    mass: area * flux.mass,
    momentum: [
      area * flux.momentum[0],
      area * flux.momentum[1],
      area * flux.momentum[2],
    ],
    totalEnergy: area * flux.totalEnergy,
  };
}
