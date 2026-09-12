import { DRY_AIR, type AtmosphereConfig } from '../core/constants.js';
import { dot3, norm3, type Vec3 } from '../core/math.js';
import type { FaceFlux, IntegratedFaceFlux } from './finiteVolume.js';
import {
  primitiveFromConserved,
  type ConservativeCell,
  type PrimitiveCell,
} from './state.js';

function soundSpeed(primitive: PrimitiveCell, gas: AtmosphereConfig): number {
  const a2 = gas.gamma * primitive.pressure / primitive.rho;
  if (!(a2 > 0) || !Number.isFinite(a2)) {
    throw new Error(`invalid sound-speed square: ${a2}`);
  }
  return Math.sqrt(a2);
}

function betaPlus(mach: number): number {
  if (Math.abs(mach) < 1) {
    return 0.25 * (2 - mach) * (mach + 1) * (mach + 1);
  }
  return mach >= 0 ? 1 : 0;
}

function betaMinus(mach: number): number {
  if (Math.abs(mach) < 1) {
    return 0.25 * (2 + mach) * (mach - 1) * (mach - 1);
  }
  return mach >= 0 ? 0 : 1;
}

/**
 * Parameter-free SLAU2 numerical flux for the compressible Euler equations.
 *
 * The scheme is used here because Core v2 has to cover very-low-Mach weather
 * flows and strongly compressible local flows with one spatial operator. It
 * does not require a freestream/reference Mach-number tuning parameter.
 *
 * `unitNormal` is oriented from the left state toward the right state.
 */
export function slau2Flux(
  left: ConservativeCell,
  right: ConservativeCell,
  unitNormal: Vec3,
  leftGeopotential = 0,
  rightGeopotential = 0,
  gas: AtmosphereConfig = DRY_AIR,
): FaceFlux {
  const normalLength = norm3(unitNormal);
  if (Math.abs(normalLength - 1) > 1e-12) {
    throw new Error(`face normal must be unit length; got ${normalLength}`);
  }

  const pl = primitiveFromConserved(left, leftGeopotential, gas);
  const pr = primitiveFromConserved(right, rightGeopotential, gas);
  const unl = dot3(pl.velocity, unitNormal);
  const unr = dot3(pr.velocity, unitNormal);
  const speed2l = dot3(pl.velocity, pl.velocity);
  const speed2r = dot3(pr.velocity, pr.velocity);

  const al = soundSpeed(pl, gas);
  const ar = soundSpeed(pr, gas);
  const aFace = 0.5 * (al + ar);
  if (!(aFace > 0) || !Number.isFinite(aFace)) {
    throw new Error(`invalid interface sound speed: ${aFace}`);
  }

  const ml = unl / aFace;
  const mr = unr / aFace;
  const velocityRms = Math.sqrt(0.5 * (speed2l + speed2r));
  const machHat = Math.min(1, velocityRms / aFace);
  const chi = (1 - machHat) * (1 - machHat);

  const expansionSwitch =
    -Math.max(Math.min(ml, 0), -1) * Math.min(Math.max(mr, 0), 1);
  const weightedAbsNormalVelocity =
    (pl.rho * Math.abs(unl) + pr.rho * Math.abs(unr)) / (pl.rho + pr.rho);
  const absNormalLeft =
    (1 - expansionSwitch) * weightedAbsNormalVelocity +
    expansionSwitch * Math.abs(unl);
  const absNormalRight =
    (1 - expansionSwitch) * weightedAbsNormalVelocity +
    expansionSwitch * Math.abs(unr);

  const massFlux = 0.5 * (
    pl.rho * (unl + absNormalLeft) +
    pr.rho * (unr - absNormalRight) -
    (chi / aFace) * (pr.pressure - pl.pressure)
  );

  const bp = betaPlus(ml);
  const bm = betaMinus(mr);
  const rhoMean = 0.5 * (pl.rho + pr.rho);
  const pressureFlux =
    0.5 * (pl.pressure + pr.pressure) +
    0.5 * (bp - bm) * (pl.pressure - pr.pressure) +
    velocityRms * (bp + bm - 1) * aFace * rhoMean;

  if (!Number.isFinite(massFlux) || !Number.isFinite(pressureFlux)) {
    throw new Error('non-finite SLAU2 interface flux');
  }

  const upwindState = massFlux >= 0 ? left : right;
  const upwindPrimitive = massFlux >= 0 ? pl : pr;
  const enthalpy = (upwindState.rhoE + upwindPrimitive.pressure) / upwindState.rho;

  return {
    mass: massFlux,
    momentum: [
      massFlux * upwindPrimitive.velocity[0] + pressureFlux * unitNormal[0],
      massFlux * upwindPrimitive.velocity[1] + pressureFlux * unitNormal[1],
      massFlux * upwindPrimitive.velocity[2] + pressureFlux * unitNormal[2],
    ],
    totalEnergy: massFlux * enthalpy,
  };
}

/** Face-integrated SLAU2 flux for a geometric vector area. */
export function integratedSlau2Flux(
  left: ConservativeCell,
  right: ConservativeCell,
  vectorArea: Vec3,
  leftGeopotential = 0,
  rightGeopotential = 0,
  gas: AtmosphereConfig = DRY_AIR,
): IntegratedFaceFlux {
  const area = norm3(vectorArea);
  if (!(area > 0) || !Number.isFinite(area)) {
    throw new Error(`invalid face vector area magnitude: ${area}`);
  }
  const unitNormal: Vec3 = [
    vectorArea[0] / area,
    vectorArea[1] / area,
    vectorArea[2] / area,
  ];
  const flux = slau2Flux(
    left,
    right,
    unitNormal,
    leftGeopotential,
    rightGeopotential,
    gas,
  );
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
