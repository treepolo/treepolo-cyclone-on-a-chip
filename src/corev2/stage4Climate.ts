import { DRY_AIR } from '../core/constants.js';
import type { Vec3 } from '../core/math.js';
import { createConservativeFields, type ConservativeFields } from './fields.js';
import type { HydrostaticReference1D } from './hydrostaticReference.js';
import {
  conservedFromPrimitive,
  pressureFromConservedRelativeToReference,
} from './state.js';
import {
  shellCellIndex,
  type SphericalShellGeometry,
} from './sphericalShellGeometry.js';

export interface Stage4CirculationDiagnostics {
  meanTropicalLowLevelZonal: number;
  maxUpperMidlatitudeWesterly: number;
  meanTropicalMeridionalSpeed: number;
  midlatitudeEddyKineticEnergy: number;
  midlatitudePolewardHeatFlux: number;
  lowLevelPressureStd: number;
  maxLowLevelTemperatureGradientKPer1000km: number;
  maxAbsRadialVelocity: number;
  maxSpeed: number;
  minDensity: number;
  minPressure: number;
}

function localBasis(x: number, y: number, z: number): {
  radial: Vec3;
  east: Vec3;
  north: Vec3;
  latitude: number;
} {
  const latitude = Math.asin(Math.max(-1, Math.min(1, z)));
  const horizontal = Math.hypot(x, y);
  const east: Vec3 = horizontal > 1e-12
    ? [-y / horizontal, x / horizontal, 0]
    : [0, 1, 0];
  const radial: Vec3 = [x, y, z];
  const north: Vec3 = [
    radial[1] * east[2] - radial[2] * east[1],
    radial[2] * east[0] - radial[0] * east[2],
    radial[0] * east[1] - radial[1] * east[0],
  ];
  return { radial, east, north, latitude };
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cellState(
  fields: ConservativeFields,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  c: number,
  k: number,
): {
  velocity: Vec3;
  pressure: number;
  temperature: number;
  density: number;
} {
  const q = shellCellIndex(c, k, geometry.nz);
  const density = fields.rho[q]!;
  const momentum: Vec3 = [fields.momX[q]!, fields.momY[q]!, fields.momZ[q]!];
  const pressure = pressureFromConservedRelativeToReference({
    rho: density,
    momentum,
    rhoE: fields.rhoE[q]!,
  }, reference.cellDensity[k]!, reference.cellPressure[k]!, reference.cellGeopotential[k]!);
  const invRho = 1 / density;
  return {
    velocity: [momentum[0] * invRho, momentum[1] * invRho, momentum[2] * invRho],
    pressure,
    temperature: pressure / (density * DRY_AIR.rd),
    density,
  };
}

/**
 * Hydrostatic-rest initial state with a tiny zonal thermal wave as a symmetry
 * breaker. The pressure profile remains the discrete hydrostatic reference;
 * only density is adjusted by the small temperature perturbation.
 */
export function buildStage4HeldSuarezInitialState(
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  perturbationK = 0.05,
): ConservativeFields {
  if (!(perturbationK >= 0) || !Number.isFinite(perturbationK)) {
    throw new Error(`invalid Stage 4 perturbation amplitude: ${perturbationK}`);
  }
  const fields = createConservativeFields(geometry.horizontal.cellCount * geometry.nz);
  const surfaceRadius = geometry.radiusInterface[0]!;
  const topHeight = geometry.radiusInterface[geometry.nz]! - surfaceRadius;

  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    const x = geometry.horizontal.cellCenters[c * 3]!;
    const y = geometry.horizontal.cellCenters[c * 3 + 1]!;
    const z = geometry.horizontal.cellCenters[c * 3 + 2]!;
    const latitude = Math.asin(Math.max(-1, Math.min(1, z)));
    const longitude = Math.atan2(y, x);
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const centerRadius = 0.5 * (
        geometry.radiusInterface[k]! + geometry.radiusInterface[k + 1]!
      );
      const normalizedHeight = topHeight > 0
        ? (centerRadius - surfaceRadius) / topHeight
        : 0;
      const wave = perturbationK *
        Math.cos(latitude) * Math.cos(latitude) *
        Math.cos(4 * longitude) *
        Math.sin(Math.PI * normalizedHeight);
      const temperature = reference.temperature + wave;
      const pressure = reference.cellPressure[k]!;
      const density = pressure / (DRY_AIR.rd * temperature);
      const state = conservedFromPrimitive({
        rho: density,
        pressure,
        velocity: [0, 0, 0],
      }, reference.cellGeopotential[k]!);
      fields.rho[q] = state.rho;
      fields.momX[q] = state.momentum[0];
      fields.momY[q] = state.momentum[1];
      fields.momZ[q] = state.momentum[2];
      fields.rhoE[q] = state.rhoE;
    }
  }
  return fields;
}

/**
 * Compact Stage-4 diagnostics aimed at the actual dry-global-atmosphere goal:
 * tropical easterlies/overturning, midlatitude westerlies and baroclinic eddy
 * activity. They are diagnostics only and never feed back into the model.
 */
export function diagnoseStage4Circulation(
  fields: ConservativeFields,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  latitudeBins = 18,
): Stage4CirculationDiagnostics {
  const expected = geometry.horizontal.cellCount * geometry.nz;
  if (fields.rho.length !== expected) {
    throw new Error('Stage 4 diagnostic state/geometry size mismatch');
  }
  if (!Number.isInteger(latitudeBins) || latitudeBins < 6) {
    throw new Error(`invalid Stage 4 latitude-bin count: ${latitudeBins}`);
  }

  const binLayerCount = latitudeBins * geometry.nz;
  const meanU = new Float64Array(binLayerCount);
  const meanV = new Float64Array(binLayerCount);
  const meanT = new Float64Array(binLayerCount);
  const weights = new Float64Array(binLayerCount);
  let maxAbsRadialVelocity = 0;
  let maxSpeed = 0;
  let minDensity = Number.POSITIVE_INFINITY;
  let minPressure = Number.POSITIVE_INFINITY;

  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    const x = geometry.horizontal.cellCenters[c * 3]!;
    const y = geometry.horizontal.cellCenters[c * 3 + 1]!;
    const z = geometry.horizontal.cellCenters[c * 3 + 2]!;
    const basis = localBasis(x, y, z);
    const bin = Math.max(0, Math.min(
      latitudeBins - 1,
      Math.floor((basis.latitude + Math.PI / 2) / Math.PI * latitudeBins),
    ));
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const state = cellState(fields, geometry, reference, c, k);
      const u = dot(state.velocity, basis.east);
      const v = dot(state.velocity, basis.north);
      const w = dot(state.velocity, basis.radial);
      const speed = Math.hypot(state.velocity[0], state.velocity[1], state.velocity[2]);
      const weight = state.density * geometry.cellVolume[q]!;
      const id = bin * geometry.nz + k;
      meanU[id] = meanU[id]! + weight * u;
      meanV[id] = meanV[id]! + weight * v;
      meanT[id] = meanT[id]! + weight * state.temperature;
      weights[id] = weights[id]! + weight;
      maxAbsRadialVelocity = Math.max(maxAbsRadialVelocity, Math.abs(w));
      maxSpeed = Math.max(maxSpeed, speed);
      minDensity = Math.min(minDensity, state.density);
      minPressure = Math.min(minPressure, state.pressure);
    }
  }
  for (let i = 0; i < binLayerCount; i++) {
    const weight = weights[i]!;
    if (weight > 0) {
      meanU[i] = meanU[i]! / weight;
      meanV[i] = meanV[i]! / weight;
      meanT[i] = meanT[i]! / weight;
    }
  }

  let tropicalLowU = 0;
  let tropicalLowWeight = 0;
  let tropicalMeridional = 0;
  let tropicalMeridionalWeight = 0;
  let maxUpperMidlatitudeWesterly = Number.NEGATIVE_INFINITY;
  let eddyKinetic = 0;
  let eddyHeat = 0;
  let eddyWeight = 0;
  let lowPWeight = 0;
  let lowP = 0;
  let lowP2 = 0;

  const surfaceRadius = geometry.radiusInterface[0]!;
  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    const x = geometry.horizontal.cellCenters[c * 3]!;
    const y = geometry.horizontal.cellCenters[c * 3 + 1]!;
    const z = geometry.horizontal.cellCenters[c * 3 + 2]!;
    const basis = localBasis(x, y, z);
    const latitudeDeg = basis.latitude * 180 / Math.PI;
    const absLatitude = Math.abs(latitudeDeg);
    const polewardSign = latitudeDeg >= 0 ? 1 : -1;
    const bin = Math.max(0, Math.min(
      latitudeBins - 1,
      Math.floor((basis.latitude + Math.PI / 2) / Math.PI * latitudeBins),
    ));
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const state = cellState(fields, geometry, reference, c, k);
      const u = dot(state.velocity, basis.east);
      const v = dot(state.velocity, basis.north);
      const id = bin * geometry.nz + k;
      const weight = state.density * geometry.cellVolume[q]!;
      const centerHeight = 0.5 * (
        geometry.radiusInterface[k]! + geometry.radiusInterface[k + 1]!
      ) - surfaceRadius;
      const sigma = state.pressure / DRY_AIR.pRef;

      if (absLatitude <= 25 && sigma >= 0.7) {
        tropicalLowU += weight * u;
        tropicalLowWeight += weight;
      }
      if (absLatitude <= 30) {
        tropicalMeridional += weight * Math.abs(v);
        tropicalMeridionalWeight += weight;
      }
      if (absLatitude >= 30 && absLatitude <= 70 && centerHeight >= 4000 && centerHeight <= 18000) {
        maxUpperMidlatitudeWesterly = Math.max(maxUpperMidlatitudeWesterly, meanU[id]!);
      }
      if (absLatitude >= 30 && absLatitude <= 70 && centerHeight <= 16000) {
        const up = u - meanU[id]!;
        const vp = v - meanV[id]!;
        const tp = state.temperature - meanT[id]!;
        eddyKinetic += weight * 0.5 * (up * up + vp * vp);
        eddyHeat += weight * polewardSign * vp * tp;
        eddyWeight += weight;
      }
      if (k === 0 && absLatitude >= 20 && absLatitude <= 70) {
        const areaWeight = geometry.horizontal.cellAreaUnit[c]!;
        lowP += areaWeight * state.pressure;
        lowP2 += areaWeight * state.pressure * state.pressure;
        lowPWeight += areaWeight;
      }
    }
  }

  let maxLowLevelTemperatureGradientKPer1000km = 0;
  for (const edge of geometry.horizontal.edges) {
    const left = cellState(fields, geometry, reference, edge.leftCell, 0);
    const right = cellState(fields, geometry, reference, edge.rightCell, 0);
    const distance = Math.max(
      1,
      edge.centerDistanceAngle * geometry.radiusInterface[0]!,
    );
    const gradient = Math.abs(right.temperature - left.temperature) / distance * 1e6;
    maxLowLevelTemperatureGradientKPer1000km = Math.max(
      maxLowLevelTemperatureGradientKPer1000km,
      gradient,
    );
  }

  const pressureMean = lowPWeight > 0 ? lowP / lowPWeight : Number.NaN;
  const pressureVariance = lowPWeight > 0
    ? Math.max(0, lowP2 / lowPWeight - pressureMean * pressureMean)
    : Number.NaN;

  return {
    meanTropicalLowLevelZonal: tropicalLowWeight > 0
      ? tropicalLowU / tropicalLowWeight
      : Number.NaN,
    maxUpperMidlatitudeWesterly: Number.isFinite(maxUpperMidlatitudeWesterly)
      ? maxUpperMidlatitudeWesterly
      : Number.NaN,
    meanTropicalMeridionalSpeed: tropicalMeridionalWeight > 0
      ? tropicalMeridional / tropicalMeridionalWeight
      : Number.NaN,
    midlatitudeEddyKineticEnergy: eddyWeight > 0 ? eddyKinetic / eddyWeight : Number.NaN,
    midlatitudePolewardHeatFlux: eddyWeight > 0 ? eddyHeat / eddyWeight : Number.NaN,
    lowLevelPressureStd: Math.sqrt(pressureVariance),
    maxLowLevelTemperatureGradientKPer1000km,
    maxAbsRadialVelocity,
    maxSpeed,
    minDensity,
    minPressure,
  };
}
