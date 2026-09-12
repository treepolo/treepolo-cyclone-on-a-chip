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

interface Slau2ScalarFluxes {
  massFlux: number;
  pressureFlux: number;
  pressurePerturbationFlux: number;
  upwind: PrimitiveCell;
  upwindSpeed2: number;
}

function slau2ScalarFluxes(
  pl: PrimitiveCell,
  pr: PrimitiveCell,
  unitNormal: Vec3,
  pressureReference: number,
  gas: AtmosphereConfig,
): Slau2ScalarFluxes {
  const normalLength = norm3(unitNormal);
  if (Math.abs(normalLength - 1) > 1e-12) {
    throw new Error(`face normal must be unit length; got ${normalLength}`);
  }
  if (!(pl.rho > 0) || !(pr.rho > 0) || !(pl.pressure > 0) || !(pr.pressure > 0)) {
    throw new Error('SLAU2 primitive states require positive density and pressure');
  }
  if (!Number.isFinite(pressureReference)) {
    throw new Error(`invalid SLAU2 pressure reference: ${pressureReference}`);
  }

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

  // Use pressure perturbations for differences so an exact common reference
  // produces a bitwise-zero pressure jump instead of subtracting two ~1e5 Pa
  // numbers after unrelated arithmetic.
  const pPrimeLeft = pl.pressure - pressureReference;
  const pPrimeRight = pr.pressure - pressureReference;
  const pressureDifference = pPrimeRight - pPrimeLeft;
  const massFlux = 0.5 * (
    pl.rho * (unl + absNormalLeft) +
    pr.rho * (unr - absNormalRight) -
    (chi / aFace) * pressureDifference
  );

  const bp = betaPlus(ml);
  const bm = betaMinus(mr);
  const rhoMean = 0.5 * (pl.rho + pr.rho);
  const velocityPressureTerm = velocityRms * (bp + bm - 1) * aFace * rhoMean;
  const pressurePerturbationFlux =
    0.5 * (pPrimeLeft + pPrimeRight) +
    0.5 * (bp - bm) * (pPrimeLeft - pPrimeRight) +
    velocityPressureTerm;
  const pressureFlux = pressureReference + pressurePerturbationFlux;

  if (
    !Number.isFinite(massFlux) ||
    !Number.isFinite(pressureFlux) ||
    !Number.isFinite(pressurePerturbationFlux)
  ) {
    throw new Error('non-finite SLAU2 interface flux');
  }

  const useLeft = massFlux >= 0;
  return {
    massFlux,
    pressureFlux,
    pressurePerturbationFlux,
    upwind: useLeft ? pl : pr,
    upwindSpeed2: useLeft ? speed2l : speed2r,
  };
}

function totalEnthalpy(
  primitive: PrimitiveCell,
  speed2: number,
  geopotential: number,
  gas: AtmosphereConfig,
): number {
  return gas.gamma / (gas.gamma - 1) * primitive.pressure / primitive.rho +
    0.5 * speed2 + geopotential;
}

/** Parameter-free physical SLAU2 numerical flux from primitive face states. */
export function slau2FluxFromPrimitive(
  pl: PrimitiveCell,
  pr: PrimitiveCell,
  unitNormal: Vec3,
  leftGeopotential = 0,
  rightGeopotential = 0,
  gas: AtmosphereConfig = DRY_AIR,
): FaceFlux {
  const scalar = slau2ScalarFluxes(pl, pr, unitNormal, 0, gas);
  const phi = scalar.upwind === pl ? leftGeopotential : rightGeopotential;
  const hTotal = totalEnthalpy(scalar.upwind, scalar.upwindSpeed2, phi, gas);
  return {
    mass: scalar.massFlux,
    momentum: [
      scalar.massFlux * scalar.upwind.velocity[0] + scalar.pressureFlux * unitNormal[0],
      scalar.massFlux * scalar.upwind.velocity[1] + scalar.pressureFlux * unitNormal[1],
      scalar.massFlux * scalar.upwind.velocity[2] + scalar.pressureFlux * unitNormal[2],
    ],
    totalEnergy: scalar.massFlux * hTotal,
  };
}

/**
 * SLAU2 flux with only the common hydrostatic reference-pressure momentum flux
 * removed algebraically. Mass and total-energy fluxes remain the full physical
 * flux. This is exactly F - [0,p_ref n,0], but computes the pressure part from
 * p'=p-p_ref directly so an exact reference state gives zero momentum flux
 * without subtracting two huge face forces.
 */
export function slau2ReferenceSubtractedFluxFromPrimitive(
  pl: PrimitiveCell,
  pr: PrimitiveCell,
  unitNormal: Vec3,
  pressureReference: number,
  leftGeopotential = 0,
  rightGeopotential = 0,
  gas: AtmosphereConfig = DRY_AIR,
): FaceFlux {
  const scalar = slau2ScalarFluxes(
    pl,
    pr,
    unitNormal,
    pressureReference,
    gas,
  );
  const phi = scalar.upwind === pl ? leftGeopotential : rightGeopotential;
  const hTotal = totalEnthalpy(scalar.upwind, scalar.upwindSpeed2, phi, gas);
  return {
    mass: scalar.massFlux,
    momentum: [
      scalar.massFlux * scalar.upwind.velocity[0] + scalar.pressurePerturbationFlux * unitNormal[0],
      scalar.massFlux * scalar.upwind.velocity[1] + scalar.pressurePerturbationFlux * unitNormal[1],
      scalar.massFlux * scalar.upwind.velocity[2] + scalar.pressurePerturbationFlux * unitNormal[2],
    ],
    totalEnergy: scalar.massFlux * hTotal,
  };
}

export function slau2Flux(
  left: ConservativeCell,
  right: ConservativeCell,
  unitNormal: Vec3,
  leftGeopotential = 0,
  rightGeopotential = 0,
  gas: AtmosphereConfig = DRY_AIR,
): FaceFlux {
  return slau2FluxFromPrimitive(
    primitiveFromConserved(left, leftGeopotential, gas),
    primitiveFromConserved(right, rightGeopotential, gas),
    unitNormal,
    leftGeopotential,
    rightGeopotential,
    gas,
  );
}

function integrateFlux(flux: FaceFlux, area: number): IntegratedFaceFlux {
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

function vectorAreaGeometry(vectorArea: Vec3): { area: number; unitNormal: Vec3 } {
  const area = norm3(vectorArea);
  if (!(area > 0) || !Number.isFinite(area)) {
    throw new Error(`invalid face vector area magnitude: ${area}`);
  }
  return {
    area,
    unitNormal: [
      vectorArea[0] / area,
      vectorArea[1] / area,
      vectorArea[2] / area,
    ],
  };
}

export function integratedSlau2FluxFromPrimitive(
  left: PrimitiveCell,
  right: PrimitiveCell,
  vectorArea: Vec3,
  leftGeopotential = 0,
  rightGeopotential = 0,
  gas: AtmosphereConfig = DRY_AIR,
): IntegratedFaceFlux {
  const geometry = vectorAreaGeometry(vectorArea);
  return integrateFlux(
    slau2FluxFromPrimitive(
      left,
      right,
      geometry.unitNormal,
      leftGeopotential,
      rightGeopotential,
      gas,
    ),
    geometry.area,
  );
}

export function integratedSlau2ReferenceSubtractedFluxFromPrimitive(
  left: PrimitiveCell,
  right: PrimitiveCell,
  vectorArea: Vec3,
  pressureReference: number,
  leftGeopotential = 0,
  rightGeopotential = 0,
  gas: AtmosphereConfig = DRY_AIR,
): IntegratedFaceFlux {
  const geometry = vectorAreaGeometry(vectorArea);
  return integrateFlux(
    slau2ReferenceSubtractedFluxFromPrimitive(
      left,
      right,
      geometry.unitNormal,
      pressureReference,
      leftGeopotential,
      rightGeopotential,
      gas,
    ),
    geometry.area,
  );
}

export function integratedSlau2Flux(
  left: ConservativeCell,
  right: ConservativeCell,
  vectorArea: Vec3,
  leftGeopotential = 0,
  rightGeopotential = 0,
  gas: AtmosphereConfig = DRY_AIR,
): IntegratedFaceFlux {
  const geometry = vectorAreaGeometry(vectorArea);
  return integrateFlux(
    slau2Flux(
      left,
      right,
      geometry.unitNormal,
      leftGeopotential,
      rightGeopotential,
      gas,
    ),
    geometry.area,
  );
}
