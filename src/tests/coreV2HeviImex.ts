declare const process: { exitCode?: number };

import { DRY_AIR, EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { createConservativeFields, type ConservativeFields } from '../corev2/fields.js';
import { integratedTotals } from '../corev2/finiteVolume.js';
import { heviImexSsp2Step } from '../corev2/heviImex.js';
import { buildIsothermalHydrostaticReference } from '../corev2/hydrostaticReference.js';
import { buildLinearReconstructionStencil } from '../corev2/reconstruction.js';
import {
  conservedFromPrimitive,
  pressureFromConservedRelativeToReference,
} from '../corev2/state.js';
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

function fillReference(horizontalCells: number, geometry: Geometry, reference: Reference): ConservativeFields {
  const fields = createConservativeFields(horizontalCells * geometry.nz);
  for (let c = 0; c < horizontalCells; c++) {
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

function assertPositive(fields: ConservativeFields, geometry: Geometry, reference: Reference): void {
  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const rho = fields.rho[q]!;
      assert(rho > 0 && Number.isFinite(rho), `positive density cell=${q}`);
      const pressure = pressureFromConservedRelativeToReference({
        rho,
        momentum: [fields.momX[q]!, fields.momY[q]!, fields.momZ[q]!],
        rhoE: fields.rhoE[q]!,
      }, reference.cellDensity[k]!, reference.cellPressure[k]!, reference.cellGeopotential[k]!);
      assert(pressure > 0 && Number.isFinite(pressure), `positive pressure cell=${q}`);
    }
  }
}

function cloneFields(fields: ConservativeFields): ConservativeFields {
  const out = createConservativeFields(fields.rho.length);
  out.rho.set(fields.rho);
  out.momX.set(fields.momX);
  out.momY.set(fields.momY);
  out.momZ.set(fields.momZ);
  out.rhoE.set(fields.rhoE);
  return out;
}

function runToTime(
  initial: ConservativeFields,
  dt: number,
  finalTime: number,
  geometry: Geometry,
  reference: Reference,
  stencil: ReturnType<typeof buildLinearReconstructionStencil>,
): ConservativeFields {
  let fields = cloneFields(initial);
  const steps = Math.round(finalTime / dt);
  assert(Math.abs(steps * dt - finalTime) < 1e-12, 'test final time must be an integer number of steps');
  for (let step = 0; step < steps; step++) {
    fields = heviImexSsp2Step(fields, dt, geometry, stencil, reference).fields;
  }
  return fields;
}

function normalizedStateDifference(a: ConservativeFields, b: ConservativeFields, reference: Reference, geometry: Geometry): number {
  let numerator = 0;
  let denominator = 0;
  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const scales = [
        reference.cellDensity[k]!,
        reference.cellDensity[k]! * 350,
        reference.cellDensity[k]! * 350,
        reference.cellDensity[k]! * 350,
        reference.cellPressure[k]! / (DRY_AIR.gamma - 1) +
          reference.cellDensity[k]! * reference.cellGeopotential[k]!,
      ];
      const av = [a.rho[q]!, a.momX[q]!, a.momY[q]!, a.momZ[q]!, a.rhoE[q]!];
      const bv = [b.rho[q]!, b.momX[q]!, b.momY[q]!, b.momZ[q]!, b.rhoE[q]!];
      for (let v = 0; v < 5; v++) {
        const d = (av[v]! - bv[v]!) / Math.max(scales[v]!, 1e-12);
        numerator += d * d;
        denominator += 1;
      }
    }
  }
  return Math.sqrt(numerator / denominator);
}

test('Core v2 IMEX-HEVI timestep preserves exact hydrostatic rest without damping', () => {
  const horizontal = buildCubedSphere(3);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 100,
    EARTH.radius + 450,
    EARTH.radius + 1600,
    EARTH.radius + 5200,
    EARTH.radius + 15000,
    EARTH.radius + 40000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 281, 100400);
  const stencil = buildLinearReconstructionStencil(geometry);
  const initial = fillReference(horizontal.cellCount, geometry, reference);
  const result = heviImexSsp2Step(initial, 30, geometry, stencil, reference);
  let worstDensityRelative = 0;
  let worstEnergyRelative = 0;
  let worstSpeed = 0;
  for (let q = 0; q < initial.rho.length; q++) {
    worstDensityRelative = Math.max(
      worstDensityRelative,
      Math.abs(result.fields.rho[q]! - initial.rho[q]!) / initial.rho[q]!,
    );
    worstEnergyRelative = Math.max(
      worstEnergyRelative,
      Math.abs(result.fields.rhoE[q]! - initial.rhoE[q]!) / Math.max(Math.abs(initial.rhoE[q]!), 1),
    );
    const rho = result.fields.rho[q]!;
    worstSpeed = Math.max(
      worstSpeed,
      Math.hypot(result.fields.momX[q]!, result.fields.momY[q]!, result.fields.momZ[q]!) / rho,
    );
  }
  assert(worstDensityRelative < 2e-14, `hydrostatic density drift=${worstDensityRelative}`);
  assert(worstEnergyRelative < 2e-14, `hydrostatic total-energy drift=${worstEnergyRelative}`);
  assert(worstSpeed < 2e-10, `hydrostatic spurious speed=${worstSpeed}`);
});

test('Core v2 IMEX-HEVI full step conserves closed-domain mass and total energy', () => {
  const horizontal = buildCubedSphere(3);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 250,
    EARTH.radius + 1000,
    EARTH.radius + 3000,
    EARTH.radius + 7500,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 286, 100100);
  const stencil = buildLinearReconstructionStencil(geometry);
  const fields = fillReference(horizontal.cellCount, geometry, reference);
  for (let c = 0; c < horizontal.cellCount; c++) {
    const x = horizontal.cellCenters[c * 3]!;
    const y = horizontal.cellCenters[c * 3 + 1]!;
    const z = horizontal.cellCenters[c * 3 + 2]!;
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const state = conservedFromPrimitive({
        rho: reference.cellDensity[k]! * (1 + 0.002 * x - 0.001 * y),
        pressure: reference.cellPressure[k]! * (1 + 0.003 * z + 0.0005 * k),
        velocity: [7 * y, -6 * x, 1.5 * z],
      }, reference.cellGeopotential[k]!);
      fields.rho[q] = state.rho;
      fields.momX[q] = state.momentum[0];
      fields.momY[q] = state.momentum[1];
      fields.momZ[q] = state.momentum[2];
      fields.rhoE[q] = state.rhoE;
    }
  }
  const before = integratedTotals(fields, geometry.cellVolume);
  const next = heviImexSsp2Step(fields, 2, geometry, stencil, reference).fields;
  const after = integratedTotals(next, geometry.cellVolume);
  const massRelative = Math.abs(after.mass - before.mass) / Math.max(Math.abs(before.mass), 1);
  const energyRelative = Math.abs(after.totalEnergy - before.totalEnergy) / Math.max(Math.abs(before.totalEnergy), 1);
  assert(massRelative < 3e-13, `IMEX global mass drift=${massRelative}`);
  assert(energyRelative < 3e-13, `IMEX global total-energy drift=${energyRelative}`);
  assertPositive(next, geometry, reference);
});

test('Core v2 IMEX-HEVI advances a vertical acoustic disturbance at CFL above 50 without acoustic substeps', () => {
  const horizontal = buildCubedSphere(2);
  const radii = [EARTH.radius];
  for (let k = 1; k <= 12; k++) radii.push(EARTH.radius + 120 * k);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array(radii));
  const reference = buildIsothermalHydrostaticReference(geometry, 288, 100000);
  const stencil = buildLinearReconstructionStencil(geometry);
  let fields = fillReference(horizontal.cellCount, geometry, reference);
  const middle = Math.floor(geometry.nz / 2);
  for (let c = 0; c < horizontal.cellCount; c++) {
    const q = shellCellIndex(c, middle, geometry.nz);
    const state = conservedFromPrimitive({
      rho: reference.cellDensity[middle]! * 1.001,
      pressure: reference.cellPressure[middle]! * 1.01,
      velocity: [0, 0, 0],
    }, reference.cellGeopotential[middle]!);
    fields.rho[q] = state.rho;
    fields.rhoE[q] = state.rhoE;
  }
  const dt = 20;
  const acousticCfl = Math.sqrt(DRY_AIR.gamma * DRY_AIR.rd * 288) * dt / 120;
  assert(acousticCfl > 50, `large-CFL test setup failed: ${acousticCfl}`);
  const before = integratedTotals(fields, geometry.cellVolume);
  for (let step = 0; step < 4; step++) {
    const result = heviImexSsp2Step(fields, dt, geometry, stencil, reference);
    fields = result.fields;
    assert(result.diagnostics.maxImplicitResidualStage1 < 2e-10,
      `stage1 implicit residual step=${step}: ${result.diagnostics.maxImplicitResidualStage1}`);
    assert(result.diagnostics.maxImplicitResidualStage2 < 2e-10,
      `stage2 implicit residual step=${step}: ${result.diagnostics.maxImplicitResidualStage2}`);
    assertPositive(fields, geometry, reference);
  }
  const after = integratedTotals(fields, geometry.cellVolume);
  const massRelative = Math.abs(after.mass - before.mass) / Math.max(Math.abs(before.mass), 1);
  const energyRelative = Math.abs(after.totalEnergy - before.totalEnergy) / Math.max(Math.abs(before.totalEnergy), 1);
  assert(massRelative < 2e-12, `large-CFL mass drift=${massRelative}`);
  assert(energyRelative < 2e-12, `large-CFL energy drift=${energyRelative}`);
});

test('Core v2 IMEX-HEVI shows second-order temporal self-convergence on a smooth perturbation', () => {
  const horizontal = buildCubedSphere(2);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 600,
    EARTH.radius + 1400,
    EARTH.radius + 2500,
    EARTH.radius + 4000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 288, 100000);
  const stencil = buildLinearReconstructionStencil(geometry);
  const initial = fillReference(horizontal.cellCount, geometry, reference);
  for (let c = 0; c < horizontal.cellCount; c++) {
    const x = horizontal.cellCenters[c * 3]!;
    const y = horizontal.cellCenters[c * 3 + 1]!;
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const verticalShape = Math.sin(Math.PI * (k + 0.5) / geometry.nz);
      const state = conservedFromPrimitive({
        rho: reference.cellDensity[k]! * (1 + 2e-4 * verticalShape * x),
        pressure: reference.cellPressure[k]! * (1 + 3e-4 * verticalShape * y),
        velocity: [0.4 * y * verticalShape, -0.3 * x * verticalShape, 0],
      }, reference.cellGeopotential[k]!);
      initial.rho[q] = state.rho;
      initial.momX[q] = state.momentum[0];
      initial.momY[q] = state.momentum[1];
      initial.momZ[q] = state.momentum[2];
      initial.rhoE[q] = state.rhoE;
    }
  }

  const finalTime = 2;
  const coarse = runToTime(initial, 1, finalTime, geometry, reference, stencil);
  const fine = runToTime(initial, 0.5, finalTime, geometry, reference, stencil);
  const referenceTime = runToTime(initial, 0.125, finalTime, geometry, reference, stencil);
  const coarseError = normalizedStateDifference(coarse, referenceTime, reference, geometry);
  const fineError = normalizedStateDifference(fine, referenceTime, reference, geometry);
  assert(fineError > 0, 'temporal convergence fine error must be nonzero');
  const ratio = coarseError / fineError;
  assert(ratio > 3.0, `IMEX temporal self-convergence ratio=${ratio}`);
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
console.log(`${passed}/${tests.length} Core v2 IMEX-HEVI timestep tests passed`);
