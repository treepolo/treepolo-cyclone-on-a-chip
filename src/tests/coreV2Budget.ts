declare const process: { exitCode?: number };

import { DRY_AIR, EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { closedSystemBudget } from '../corev2/budget.js';
import { createConservativeFields } from '../corev2/fields.js';
import { buildIsothermalHydrostaticReference } from '../corev2/hydrostaticReference.js';
import { conservedFromPrimitive } from '../corev2/state.js';
import {
  buildSphericalShellGeometry,
  shellCellIndex,
} from '../corev2/sphericalShellGeometry.js';
import { assert } from './assert.js';

interface Test { name: string; fn: () => void }
const tests: Test[] = [];
const test = (name: string, fn: () => void): void => { tests.push({ name, fn }); };

test('Core v2 energy ledger closes internal + kinetic + potential to prognostic total energy', () => {
  const horizontal = buildCubedSphere(3);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 400,
    EARTH.radius + 1600,
    EARTH.radius + 5000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 286, 100300);
  const fields = createConservativeFields(horizontal.cellCount * geometry.nz);

  for (let c = 0; c < horizontal.cellCount; c++) {
    const x = horizontal.cellCenters[c * 3]!;
    const y = horizontal.cellCenters[c * 3 + 1]!;
    const z = horizontal.cellCenters[c * 3 + 2]!;
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const state = conservedFromPrimitive({
        rho: reference.cellDensity[k]! * (1 + 0.01 * x - 0.004 * y),
        pressure: reference.cellPressure[k]! * (1 + 0.008 * z + 0.002 * k),
        velocity: [25 * y, -17 * x, 4 * z],
      }, reference.cellGeopotential[k]!);
      fields.rho[q] = state.rho;
      fields.momX[q] = state.momentum[0];
      fields.momY[q] = state.momentum[1];
      fields.momZ[q] = state.momentum[2];
      fields.rhoE[q] = state.rhoE;
    }
  }

  const budget = closedSystemBudget(fields, geometry, reference);
  const scale = Math.max(1, Math.abs(budget.totalEnergy));
  assert(
    Math.abs(budget.energyClosureResidual) / scale < 8e-15,
    `energy ledger closure relative residual=${budget.energyClosureResidual / scale}`,
  );
  assert(budget.mass > 0, 'budget mass must be positive');
  assert(budget.internalEnergy > 0, 'budget internal energy must be positive');
  assert(budget.kineticEnergy > 0, 'budget kinetic energy must be positive');
  assert(budget.potentialEnergy >= 0, 'budget potential energy must be non-negative');
  assert(budget.minDensity > 0 && budget.minPressure > 0, 'budget state must stay physical');
});

test('Core v2 energy ledger reports the analytic kinetic contribution of a uniform velocity shift', () => {
  const horizontal = buildCubedSphere(2);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 1000,
    EARTH.radius + 3000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 290, 100000);
  const rest = createConservativeFields(horizontal.cellCount * geometry.nz);
  const moving = createConservativeFields(horizontal.cellCount * geometry.nz);
  const velocity: readonly [number, number, number] = [12, -5, 3];
  const speed2 = velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2;

  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const restState = conservedFromPrimitive({
        rho: reference.cellDensity[k]!,
        pressure: reference.cellPressure[k]!,
        velocity: [0, 0, 0],
      }, reference.cellGeopotential[k]!);
      const movingState = conservedFromPrimitive({
        rho: reference.cellDensity[k]!,
        pressure: reference.cellPressure[k]!,
        velocity: [velocity[0], velocity[1], velocity[2]],
      }, reference.cellGeopotential[k]!);
      rest.rho[q] = restState.rho;
      rest.momX[q] = restState.momentum[0];
      rest.momY[q] = restState.momentum[1];
      rest.momZ[q] = restState.momentum[2];
      rest.rhoE[q] = restState.rhoE;
      moving.rho[q] = movingState.rho;
      moving.momX[q] = movingState.momentum[0];
      moving.momY[q] = movingState.momentum[1];
      moving.momZ[q] = movingState.momentum[2];
      moving.rhoE[q] = movingState.rhoE;
    }
  }

  const restBudget = closedSystemBudget(rest, geometry, reference);
  const movingBudget = closedSystemBudget(moving, geometry, reference);
  const expected = 0.5 * restBudget.mass * speed2;
  const actual = movingBudget.kineticEnergy - restBudget.kineticEnergy;
  assert(
    Math.abs(actual - expected) / expected < 5e-15,
    `uniform-shift kinetic energy relative error=${(actual - expected) / expected}`,
  );
  const internalRelative = Math.abs(movingBudget.internalEnergy - restBudget.internalEnergy) /
    Math.max(1, restBudget.internalEnergy);
  assert(internalRelative < 5e-15, `uniform shift changed internal energy=${internalRelative}`);

  const expectedPressure = reference.cellPressure[0]!;
  assert(expectedPressure > 0 && DRY_AIR.gamma > 1, 'test gas/reference setup invalid');
});

let passed = 0;
for (const t of tests) {
  try {
    t.fn();
    console.log(`PASS ${t.name}`);
    passed++;
  } catch (error) {
    console.error(`FAIL ${t.name}`);
    console.error(error);
    process.exitCode = 1;
  }
}
console.log(`${passed}/${tests.length} Core v2 budget tests passed`);
