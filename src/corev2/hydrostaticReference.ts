import { DRY_AIR, EARTH, type AtmosphereConfig, type PlanetConfig } from '../core/constants.js';
import type { Vec3 } from '../core/math.js';
import type { SphericalShellGeometry } from './sphericalShellGeometry.js';

export interface HydrostaticReference1D {
  /** Reference temperature [K]. This is a numerical balance reference, not a forcing target. */
  temperature: number;
  /** Pressure at the lower shell boundary [Pa]. */
  surfacePressure: number;
  /** Cell-volume-mean reference density per vertical layer [kg m^-3]. */
  cellDensity: Float64Array;
  /** Cell-volume-mean reference pressure per layer [Pa]. */
  cellPressure: Float64Array;
  /** Reference pressure at radial interfaces [Pa]. */
  radialFacePressure: Float64Array;
  /** Area-mean reference pressure on conical side faces in each layer [Pa]. */
  sideFacePressure: Float64Array;
  /** Volume-mean geopotential assigned to each cell layer [J kg^-1]. */
  cellGeopotential: Float64Array;
}

function oneMinusExpTimesOnePlusX(x: number): number {
  if (Math.abs(x) < 1e-3) {
    const x2 = x * x;
    return x2 * (0.5 + x * (-1 / 3 + x * (1 / 8 - x / 30)));
  }
  return -Math.expm1(-x) - x * Math.exp(-x);
}

function volumeMeanRadius(r0: number, r1: number): number {
  return 0.75 * (r1 ** 4 - r0 ** 4) / (r1 ** 3 - r0 ** 3);
}

/**
 * Build an exactly balanced discrete isothermal reference for constant radial
 * gravity. The reference is used only to algebraically split pressure and
 * gravity into perturbation terms; it does not relax the simulated atmosphere
 * toward this profile and therefore is not a climate forcing.
 *
 * For p(r)=p_s exp[-g(r-R)/(R_d T)], the conical side-face pressure is the
 * exact area mean  int(p r dr) / int(r dr), while the cell density is the
 * exact volume mean int(rho r^2 dr) / int(r^2 dr). With these definitions the
 * finite-volume pressure surface integral and gravity volume integral cancel
 * to roundoff in every spherical-shell cell.
 */
export function buildIsothermalHydrostaticReference(
  geometry: SphericalShellGeometry,
  temperature = 288,
  surfacePressure = 100000,
  planet: PlanetConfig = EARTH,
  gas: AtmosphereConfig = DRY_AIR,
): HydrostaticReference1D {
  if (!(temperature > 0) || !Number.isFinite(temperature)) {
    throw new Error(`invalid hydrostatic reference temperature: ${temperature}`);
  }
  if (!(surfacePressure > 0) || !Number.isFinite(surfacePressure)) {
    throw new Error(`invalid hydrostatic reference pressure: ${surfacePressure}`);
  }
  const radii = geometry.radiusInterface;
  const nz = geometry.nz;
  const surfaceRadius = radii[0]!;
  const inverseScaleHeight = planet.gravity / (gas.rd * temperature);
  const radialFacePressure = new Float64Array(nz + 1);
  const sideFacePressure = new Float64Array(nz);
  const cellDensity = new Float64Array(nz);
  const cellPressure = new Float64Array(nz);
  const cellGeopotential = new Float64Array(nz);

  for (let ki = 0; ki <= nz; ki++) {
    const z = radii[ki]! - surfaceRadius;
    radialFacePressure[ki] = surfacePressure * Math.exp(-inverseScaleHeight * z);
  }

  for (let k = 0; k < nz; k++) {
    const r0 = radii[k]!;
    const r1 = radii[k + 1]!;
    const h = r1 - r0;
    const x = inverseScaleHeight * h;
    const p0 = radialFacePressure[k]!;
    const oneMinusExp = -Math.expm1(-x);
    const secondMoment = oneMinusExpTimesOnePlusX(x);
    const integralRP = p0 * (
      r0 * oneMinusExp / inverseScaleHeight +
      secondMoment / (inverseScaleHeight * inverseScaleHeight)
    );
    const sideAreaRadialIntegral = 0.5 * (r1 * r1 - r0 * r0);
    sideFacePressure[k] = integralRP / sideAreaRadialIntegral;

    // Hydrostatic integration by parts:
    // g int(rho r^2 dr) = r0^2 p0 - r1^2 p1 + 2 int(r p dr).
    const rhoR2Integral = (
      r0 * r0 * radialFacePressure[k]! -
      r1 * r1 * radialFacePressure[k + 1]! +
      2 * integralRP
    ) / planet.gravity;
    const volumeRadialIntegral = (r1 ** 3 - r0 ** 3) / 3;
    const rhoMean = rhoR2Integral / volumeRadialIntegral;
    if (!(rhoMean > 0) || !Number.isFinite(rhoMean)) {
      throw new Error(`invalid hydrostatic reference mean density at layer ${k}: ${rhoMean}`);
    }
    cellDensity[k] = rhoMean;
    cellPressure[k] = rhoMean * gas.rd * temperature;
    cellGeopotential[k] = planet.gravity * (volumeMeanRadius(r0, r1) - surfaceRadius);
  }

  return {
    temperature,
    surfacePressure,
    cellDensity,
    cellPressure,
    radialFacePressure,
    sideFacePressure,
    cellGeopotential,
  };
}

/**
 * Exact integrated gravity force for a piecewise-constant cell density under
 * constant radial gravity. Returned vector points inward.
 */
export function integratedCellGravityForce(
  geometry: SphericalShellGeometry,
  horizontalCell: number,
  k: number,
  density: number,
  planet: PlanetConfig = EARTH,
): Vec3 {
  if (!(density >= 0) || !Number.isFinite(density)) {
    throw new Error(`invalid density for gravity force: ${density}`);
  }
  const r0 = geometry.radiusInterface[k]!;
  const r1 = geometry.radiusInterface[k + 1]!;
  const radialIntegral = (r1 ** 3 - r0 ** 3) / 3;
  const scale = -density * planet.gravity * radialIntegral;
  return [
    scale * geometry.cellVectorAreaUnit[horizontalCell * 3]!,
    scale * geometry.cellVectorAreaUnit[horizontalCell * 3 + 1]!,
    scale * geometry.cellVectorAreaUnit[horizontalCell * 3 + 2]!,
  ];
}

/**
 * Reference pressure force on one closed 3-D shell cell, using exact pressure
 * averages on its radial and conical faces. Returned vector is the force rate
 * contribution - integral(p n dA).
 */
export function integratedReferencePressureForce(
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  horizontalCell: number,
  k: number,
): Vec3 {
  const r0 = geometry.radiusInterface[k]!;
  const r1 = geometry.radiusInterface[k + 1]!;
  const pBottom = reference.radialFacePressure[k]!;
  const pTop = reference.radialFacePressure[k + 1]!;
  const pSide = reference.sideFacePressure[k]!;
  // Sum outward pressure-area vectors is proportional to the exact angular
  // vector area. The side vector sum equals -(r1^2-r0^2) A_omega.
  const surfaceCoefficient =
    pTop * r1 * r1 -
    pBottom * r0 * r0 -
    pSide * (r1 * r1 - r0 * r0);
  const scale = -surfaceCoefficient;
  return [
    scale * geometry.cellVectorAreaUnit[horizontalCell * 3]!,
    scale * geometry.cellVectorAreaUnit[horizontalCell * 3 + 1]!,
    scale * geometry.cellVectorAreaUnit[horizontalCell * 3 + 2]!,
  ];
}
