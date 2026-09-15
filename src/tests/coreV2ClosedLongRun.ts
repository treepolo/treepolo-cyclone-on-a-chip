declare const process: { exitCode?: number };

import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { closedSystemBudget } from '../corev2/budget.js';
import { createConservativeFields, type ConservativeFields } from '../corev2/fields.js';
import { buildIsothermalHydrostaticReference } from '../corev2/hydrostaticReference.js';
import { buildLinearReconstructionStencil } from '../corev2/reconstruction.js';
import { rotatingHeviImexSsp2Step } from '../corev2/rotatingHevi.js';
import { conservedFromPrimitive } from '../corev2/state.js';
import {
  buildSphericalShellGeometry,
  shellCellIndex,
} from '../corev2/sphericalShellGeometry.js';
import { assert } from './assert.js';

interface Test { name: string; fn: () => void }
const tests: Test[] = [];
const test = (name: string, fn: () => void): void => { tests.push({ name, fn }); };

type Geometry = ReturnType<typeof buildSphericalShellGeometry>;
type Reference = ReturnType<typeof buildIsothermalHydrostaticReference>;

function fillReference(
  geometry: Geometry,
  reference: Reference,
): ConservativeFields {
  const fields = createConservativeFields(geometry.horizontal.cellCount * geometry.nz);
  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const state = conservedFromPrimitive({
        rho: reference.cellDensity[k]!,
        pressure: reference.cellPressure[k]!,
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

function relativeDrift(after: number, before: number): number {
  return Math.abs(after - before) / Math.max(1, Math.abs(before));
}

test('Core v2 closed rotating hydrostatic atmosphere stays at rest for one simulated hour', () => {
  const horizontal = buildCubedSphere(2);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 180,
    EARTH.radius + 650,
    EARTH.radius + 1900,
    EARTH.radius + 6500,
    EARTH.radius + 20000,
    EARTH.radius + 40000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 282, 100400);
  const stencil = buildLinearReconstructionStencil(geometry);
  let fields = fillReference(geometry, reference);
  const initial = closedSystemBudget(fields, geometry, reference);

  const dt = 30;
  const steps = 120;
  let worstSpeed = 0;
  let worstClosureRelative = 0;
  for (let step = 0; step < steps; step++) {
    fields = rotatingHeviImexSsp2Step(
      fields,
      dt,
      geometry,
      stencil,
      reference,
      EARTH,
    ).fields;
    if ((step + 1) % 10 === 0 || step + 1 === steps) {
      const budget = closedSystemBudget(fields, geometry, reference);
      worstSpeed = Math.max(worstSpeed, budget.maxSpeed);
      worstClosureRelative = Math.max(
        worstClosureRelative,
        Math.abs(budget.energyClosureResidual) / Math.max(1, Math.abs(budget.totalEnergy)),
      );
      assert(budget.minDensity > 0 && budget.minPressure > 0,
        `hydrostatic long-run state became nonphysical at step ${step + 1}`);
    }
  }

  const final = closedSystemBudget(fields, geometry, reference);
  assert(relativeDrift(final.mass, initial.mass) < 2e-12,
    `hydrostatic one-hour mass drift=${relativeDrift(final.mass, initial.mass)}`);
  assert(relativeDrift(final.totalEnergy, initial.totalEnergy) < 2e-12,
    `hydrostatic one-hour energy drift=${relativeDrift(final.totalEnergy, initial.totalEnergy)}`);
  assert(worstClosureRelative < 2e-14,
    `hydrostatic one-hour energy-ledger closure=${worstClosureRelative}`);
  assert(worstSpeed < 2e-8, `hydrostatic one-hour spurious speed=${worstSpeed}`);
});

test('Core v2 closed rotating perturbation cannot create mass or total energy over many HEVI steps', () => {
  const horizontal = buildCubedSphere(2);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 160,
    EARTH.radius + 500,
    EARTH.radius + 1100,
    EARTH.radius + 2300,
    EARTH.radius + 5000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 288, 100000);
  const stencil = buildLinearReconstructionStencil(geometry);
  let fields = fillReference(geometry, reference);

  for (let c = 0; c < horizontal.cellCount; c++) {
    const x = horizontal.cellCenters[c * 3]!;
    const y = horizontal.cellCenters[c * 3 + 1]!;
    const z = horizontal.cellCenters[c * 3 + 2]!;
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const verticalShape = Math.sin(Math.PI * (k + 0.5) / geometry.nz);
      const state = conservedFromPrimitive({
        rho: reference.cellDensity[k]! *
          (1 + 1.5e-4 * verticalShape * (x - 0.4 * y)),
        pressure: reference.cellPressure[k]! *
          (1 + 2.5e-4 * verticalShape * (0.6 * x + z)),
        velocity: [
          -2.0 * y * verticalShape,
          2.0 * x * verticalShape,
          0.15 * z * verticalShape,
        ],
      }, reference.cellGeopotential[k]!);
      fields.rho[q] = state.rho;
      fields.momX[q] = state.momentum[0];
      fields.momY[q] = state.momentum[1];
      fields.momZ[q] = state.momentum[2];
      fields.rhoE[q] = state.rhoE;
    }
  }

  const initial = closedSystemBudget(fields, geometry, reference);
  const dt = 10;
  const steps = 100;
  let worstMassDrift = 0;
  let worstEnergyDrift = 0;
  let worstClosure = 0;
  for (let step = 0; step < steps; step++) {
    fields = rotatingHeviImexSsp2Step(
      fields,
      dt,
      geometry,
      stencil,
      reference,
      EARTH,
    ).fields;
    if ((step + 1) % 10 === 0 || step + 1 === steps) {
      const budget = closedSystemBudget(fields, geometry, reference);
      worstMassDrift = Math.max(worstMassDrift, relativeDrift(budget.mass, initial.mass));
      worstEnergyDrift = Math.max(
        worstEnergyDrift,
        relativeDrift(budget.totalEnergy, initial.totalEnergy),
      );
      worstClosure = Math.max(
        worstClosure,
        Math.abs(budget.energyClosureResidual) / Math.max(1, Math.abs(budget.totalEnergy)),
      );
      assert(budget.minDensity > 0, `long-run density became non-positive at step ${step + 1}`);
      assert(budget.minPressure > 0, `long-run pressure became non-positive at step ${step + 1}`);
      assert(Number.isFinite(budget.maxSpeed), `long-run speed became non-finite at step ${step + 1}`);
      assert(
        budget.kineticEnergy + budget.potentialEnergy <= budget.totalEnergy * (1 + 2e-13),
        `long-run kinetic+potential exceeded conserved total energy at step ${step + 1}`,
      );
    }
  }

  assert(worstMassDrift < 8e-12, `closed perturbation worst mass drift=${worstMassDrift}`);
  assert(worstEnergyDrift < 8e-12, `closed perturbation worst total-energy drift=${worstEnergyDrift}`);
  assert(worstClosure < 2e-14, `closed perturbation worst energy-ledger closure=${worstClosure}`);
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
console.log(`${passed}/${tests.length} Core v2 closed-long-run tests passed`);
