declare const process: { exitCode?: number };

import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { closedSystemBudget } from '../corev2/budget.js';
import {
  buildStage4HeldSuarezInitialState,
  diagnoseStage4Circulation,
} from '../corev2/stage4Climate.js';
import { buildIsothermalHydrostaticReference } from '../corev2/hydrostaticReference.js';
import { buildLinearReconstructionStencil } from '../corev2/reconstruction.js';
import { rotatingHeldSuarezStep } from '../corev2/rotatingDryCore.js';
import { buildSphericalShellGeometry } from '../corev2/sphericalShellGeometry.js';
import { assert } from './assert.js';

const DAY = 86400;
const DT = 180;
const DAYS = 2;

function finite(value: number, label: string): void {
  assert(Number.isFinite(value), `${label} is not finite: ${value}`);
}

const horizontal = buildCubedSphere(3);
const heights = [0, 1000, 3000, 6000, 10000, 15000, 22000];
const radii = new Float64Array(heights.map(height => EARTH.radius + height));
const geometry = buildSphericalShellGeometry(horizontal, radii);
const reference = buildIsothermalHydrostaticReference(geometry, 288, 100000);
const stencil = buildLinearReconstructionStencil(geometry);
let fields = buildStage4HeldSuarezInitialState(geometry, reference, 0.05);

const initialBudget = closedSystemBudget(fields, geometry, reference);
const initial = diagnoseStage4Circulation(fields, geometry, reference, 18);
const steps = Math.round(DAYS * DAY / DT);
let maxNewton1 = 0;
let maxNewton2 = 0;

for (let step = 0; step < steps; step++) {
  const result = rotatingHeldSuarezStep(
    fields,
    DT,
    geometry,
    stencil,
    reference,
    EARTH,
  );
  fields = result.fields;
  maxNewton1 = Math.max(maxNewton1, result.diagnostics.maxNewtonIterationsStage1);
  maxNewton2 = Math.max(maxNewton2, result.diagnostics.maxNewtonIterationsStage2);
}

const finalBudget = closedSystemBudget(fields, geometry, reference);
const final = diagnoseStage4Circulation(fields, geometry, reference, 18);
const massDrift = Math.abs(finalBudget.mass - initialBudget.mass) / initialBudget.mass;
const diagnostics = {
  days: DAYS,
  dt: DT,
  horizontalN: horizontal.n,
  columns: horizontal.cellCount,
  nz: geometry.nz,
  massDrift,
  maxNewton1,
  maxNewton2,
  initial,
  final,
};
console.log(`STAGE4_CIRCULATION ${JSON.stringify(diagnostics)}`);

finite(final.meanTropicalLowLevelZonal, 'tropical low-level zonal wind');
finite(final.maxUpperMidlatitudeWesterly, 'upper-midlatitude westerly');
finite(final.meanTropicalMeridionalSpeed, 'tropical meridional speed');
finite(final.midlatitudeEddyKineticEnergy, 'midlatitude EKE');
finite(final.midlatitudePolewardHeatFlux, 'midlatitude poleward heat flux');
finite(final.lowLevelPressureStd, 'low-level pressure std');
finite(final.maxLowLevelTemperatureGradientKPer1000km, 'low-level temperature gradient');

assert(massDrift < 2e-11, `Stage 4 dry mass drift too large: ${massDrift}`);
assert(finalBudget.minDensity > 0, `Stage 4 non-positive density: ${finalBudget.minDensity}`);
assert(finalBudget.minPressure > 0, `Stage 4 non-positive pressure: ${finalBudget.minPressure}`);
assert(final.maxSpeed < 250, `Stage 4 wind runaway: ${final.maxSpeed} m/s`);
assert(final.maxAbsRadialVelocity < 25, `Stage 4 radial-wind runaway: ${final.maxAbsRadialVelocity} m/s`);
assert(
  final.meanTropicalMeridionalSpeed > 0.005,
  `Stage 4 tropical overturning did not develop: ${final.meanTropicalMeridionalSpeed} m/s`,
);
assert(
  final.maxUpperMidlatitudeWesterly > 0.05,
  `Stage 4 midlatitude westerlies did not develop: ${final.maxUpperMidlatitudeWesterly} m/s`,
);
assert(
  final.midlatitudeEddyKineticEnergy > initial.midlatitudeEddyKineticEnergy + 1e-6,
  `Stage 4 eddy kinetic energy did not grow: initial=${initial.midlatitudeEddyKineticEnergy}, final=${final.midlatitudeEddyKineticEnergy}`,
);
assert(
  final.lowLevelPressureStd > initial.lowLevelPressureStd + 0.01,
  `Stage 4 synoptic pressure structure did not grow: initial=${initial.lowLevelPressureStd}, final=${final.lowLevelPressureStd}`,
);
assert(
  final.maxLowLevelTemperatureGradientKPer1000km > 0.02,
  `Stage 4 meridional/front-like thermal gradient too weak: ${final.maxLowLevelTemperatureGradientKPer1000km} K/1000km`,
);

console.log('PASS Core v2 Stage 4 rotating Held-Suarez circulation closure');
