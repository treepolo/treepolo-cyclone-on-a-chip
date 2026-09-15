import { DRY_AIR, type AtmosphereConfig } from '../core/constants.js';
import type { Vec3 } from '../core/math.js';

export interface ConservativeCell {
  /** Dry-air mass density [kg m^-3]. */
  rho: number;
  /** Cartesian momentum density [kg m^-2 s^-1]. */
  momentum: Vec3;
  /**
   * Total energy density [J m^-3].
   * By convention this includes internal + kinetic + rho * geopotential.
   */
  rhoE: number;
}

export interface PrimitiveCell {
  rho: number;
  velocity: Vec3;
  pressure: number;
}

export function kineticEnergyDensity(state: ConservativeCell): number {
  if (!(state.rho > 0) || !Number.isFinite(state.rho)) {
    throw new Error(`non-positive or invalid density: ${state.rho}`);
  }
  const [mx, my, mz] = state.momentum;
  return 0.5 * (mx * mx + my * my + mz * mz) / state.rho;
}

export function internalEnergyDensity(
  state: ConservativeCell,
  geopotential = 0,
): number {
  return state.rhoE - kineticEnergyDensity(state) - state.rho * geopotential;
}

export function pressureFromConserved(
  state: ConservativeCell,
  geopotential = 0,
  gas: AtmosphereConfig = DRY_AIR,
): number {
  const internal = internalEnergyDensity(state, geopotential);
  const pressure = (gas.gamma - 1) * internal;
  if (!(pressure > 0) || !Number.isFinite(pressure)) {
    throw new Error(`non-positive or invalid pressure: ${pressure}`);
  }
  return pressure;
}

/**
 * Algebraically identical pressure perturbation evaluated relative to a fixed
 * hydrostatic reference:
 *
 * p-p_ref = (gamma-1) [(rhoE-rhoE_ref) - K - (rho-rho_ref) Phi].
 *
 * This form avoids subtracting the full rho*Phi from rhoE before subtracting
 * two nearly equal hydrostatic pressures. In particular an exactly constructed
 * reference state gives a bitwise-zero pressure perturbation, which prevents
 * roundoff-sized hydrostatic pressure jumps from being amplified by an
 * all-speed Riemann mass flux on Earth-sized faces.
 */
export function pressurePerturbationFromReference(
  state: ConservativeCell,
  referenceDensity: number,
  referencePressure: number,
  geopotential: number,
  gas: AtmosphereConfig = DRY_AIR,
): number {
  if (!(referenceDensity > 0) || !Number.isFinite(referenceDensity)) {
    throw new Error(`invalid reference density: ${referenceDensity}`);
  }
  if (!(referencePressure > 0) || !Number.isFinite(referencePressure)) {
    throw new Error(`invalid reference pressure: ${referencePressure}`);
  }
  const kinetic = kineticEnergyDensity(state);
  const referenceEnergy =
    referencePressure / (gas.gamma - 1) + referenceDensity * geopotential;
  const internalPerturbation =
    (state.rhoE - referenceEnergy) -
    kinetic -
    (state.rho - referenceDensity) * geopotential;
  const pressurePerturbation = (gas.gamma - 1) * internalPerturbation;
  if (!Number.isFinite(pressurePerturbation)) {
    throw new Error(`invalid pressure perturbation: ${pressurePerturbation}`);
  }
  return pressurePerturbation;
}

export function pressureFromConservedRelativeToReference(
  state: ConservativeCell,
  referenceDensity: number,
  referencePressure: number,
  geopotential: number,
  gas: AtmosphereConfig = DRY_AIR,
): number {
  const pressure = referencePressure + pressurePerturbationFromReference(
    state,
    referenceDensity,
    referencePressure,
    geopotential,
    gas,
  );
  if (!(pressure > 0) || !Number.isFinite(pressure)) {
    throw new Error(`non-positive or invalid reference-relative pressure: ${pressure}`);
  }
  return pressure;
}

export function primitiveFromConserved(
  state: ConservativeCell,
  geopotential = 0,
  gas: AtmosphereConfig = DRY_AIR,
): PrimitiveCell {
  const invRho = 1 / state.rho;
  return {
    rho: state.rho,
    velocity: [
      state.momentum[0] * invRho,
      state.momentum[1] * invRho,
      state.momentum[2] * invRho,
    ],
    pressure: pressureFromConserved(state, geopotential, gas),
  };
}

export function conservedFromPrimitive(
  primitive: PrimitiveCell,
  geopotential = 0,
  gas: AtmosphereConfig = DRY_AIR,
): ConservativeCell {
  const { rho, velocity, pressure } = primitive;
  if (!(rho > 0) || !Number.isFinite(rho)) {
    throw new Error(`non-positive or invalid density: ${rho}`);
  }
  if (!(pressure > 0) || !Number.isFinite(pressure)) {
    throw new Error(`non-positive or invalid pressure: ${pressure}`);
  }

  const [ux, uy, uz] = velocity;
  const momentum: Vec3 = [rho * ux, rho * uy, rho * uz];
  const kinetic = 0.5 * rho * (ux * ux + uy * uy + uz * uz);
  const internal = pressure / (gas.gamma - 1);
  const rhoE = internal + kinetic + rho * geopotential;
  return { rho, momentum, rhoE };
}
