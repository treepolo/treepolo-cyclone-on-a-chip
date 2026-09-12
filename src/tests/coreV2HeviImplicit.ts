declare const process: { exitCode?: number };

import { DRY_AIR, EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import {
  CONSERVED_COMPONENTS,
  columnStateIndex,
} from '../corev2/heviColumn.js';
import {
  solveVerticalImplicitColumnStage,
  verticalImplicitStageResidualNorm,
} from '../corev2/heviImplicitColumn.js';
import { buildIsothermalHydrostaticReference } from '../corev2/hydrostaticReference.js';
import {
  conservedFromPrimitive,
  pressureFromConservedRelativeToReference,
} from '../corev2/state.js';
import { buildSphericalShellGeometry, shellCellIndex } from '../corev2/sphericalShellGeometry.js';
import { assert } from './assert.js';

interface Test { name: string; fn: () => void }
const tests: Test[] = [];
const test = (name: string, fn: () => void): void => { tests.push({ name, fn }); };

function referenceColumn(
  reference: ReturnType<typeof buildIsothermalHydrostaticReference>,
): Float64Array {
  const nz = reference.cellDensity.length;
  const out = new Float64Array(nz * CONSERVED_COMPONENTS);
  for (let k = 0; k < nz; k++) {
    const state = conservedFromPrimitive({
      rho: reference.cellDensity[k]!,
      pressure: reference.cellPressure[k]!,
      velocity: [0, 0, 0],
    }, reference.cellGeopotential[k]!);
    out[columnStateIndex(k, 0)] = state.rho;
    out[columnStateIndex(k, 1)] = state.momentum[0];
    out[columnStateIndex(k, 2)] = state.momentum[1];
    out[columnStateIndex(k, 3)] = state.momentum[2];
    out[columnStateIndex(k, 4)] = state.rhoE;
  }
  return out;
}

function columnIntegratedTotals(
  column: Float64Array,
  horizontalCell: number,
  geometry: ReturnType<typeof buildSphericalShellGeometry>,
): { mass: number; energy: number } {
  let mass = 0;
  let energy = 0;
  for (let k = 0; k < geometry.nz; k++) {
    const volume = geometry.cellVolume[shellCellIndex(horizontalCell, k, geometry.nz)]!;
    mass += column[columnStateIndex(k, 0)]! * volume;
    energy += column[columnStateIndex(k, 4)]! * volume;
  }
  return { mass, energy };
}

function assertPositiveColumn(
  column: Float64Array,
  reference: ReturnType<typeof buildIsothermalHydrostaticReference>,
): void {
  for (let k = 0; k < reference.cellDensity.length; k++) {
    const state = {
      rho: column[columnStateIndex(k, 0)]!,
      momentum: [
        column[columnStateIndex(k, 1)]!,
        column[columnStateIndex(k, 2)]!,
        column[columnStateIndex(k, 3)]!,
      ] as const,
      rhoE: column[columnStateIndex(k, 4)]!,
    };
    assert(state.rho > 0 && Number.isFinite(state.rho), `implicit density positive level=${k}`);
    const pressure = pressureFromConservedRelativeToReference(
      state,
      reference.cellDensity[k]!,
      reference.cellPressure[k]!,
      reference.cellGeopotential[k]!,
    );
    assert(pressure > 0 && Number.isFinite(pressure), `implicit pressure positive level=${k}`);
  }
}

test('Core v2 implicit HEVI stage leaves exact hydrostatic reference unchanged', () => {
  const horizontal = buildCubedSphere(3);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 80,
    EARTH.radius + 320,
    EARTH.radius + 1100,
    EARTH.radius + 3600,
    EARTH.radius + 10000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 280, 100000);
  const base = referenceColumn(reference);
  const result = solveVerticalImplicitColumnStage(
    base,
    30,
    4,
    geometry,
    reference,
  );
  assert(result.diagnostics.iterations === 0,
    `exact hydrostatic column should require zero Newton iterations; got ${result.diagnostics.iterations}`);
  for (let i = 0; i < base.length; i++) {
    assert(result.state[i] === base[i], `exact hydrostatic state changed at component ${i}`);
  }
});

test('Core v2 implicit HEVI stage converges with vertical acoustic CFL far above one', () => {
  const horizontal = buildCubedSphere(2);
  const radii = [EARTH.radius];
  for (let k = 1; k <= 16; k++) radii.push(EARTH.radius + 120 * k);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array(radii));
  const reference = buildIsothermalHydrostaticReference(geometry, 288, 100000);
  const base = referenceColumn(reference);
  const middle = Math.floor(geometry.nz / 2);
  const perturbed = conservedFromPrimitive({
    rho: reference.cellDensity[middle]! * 1.002,
    pressure: reference.cellPressure[middle]! * 1.02,
    velocity: [18, -11, 7],
  }, reference.cellGeopotential[middle]!);
  base[columnStateIndex(middle, 0)] = perturbed.rho;
  base[columnStateIndex(middle, 1)] = perturbed.momentum[0];
  base[columnStateIndex(middle, 2)] = perturbed.momentum[1];
  base[columnStateIndex(middle, 3)] = perturbed.momentum[2];
  base[columnStateIndex(middle, 4)] = perturbed.rhoE;

  const dt = 20;
  const soundSpeed = Math.sqrt(DRY_AIR.gamma * DRY_AIR.rd * 288);
  const acousticCfl = soundSpeed * dt / 120;
  assert(acousticCfl > 50, `test must exercise large acoustic CFL, got ${acousticCfl}`);

  const result = solveVerticalImplicitColumnStage(base, dt, 1, geometry, reference);
  assert(result.diagnostics.iterations > 0 && result.diagnostics.iterations <= 10,
    `unexpected Newton iteration count=${result.diagnostics.iterations}`);
  assert(result.diagnostics.finalResidual < 2e-10,
    `implicit final residual=${result.diagnostics.finalResidual}`);
  const checkedResidual = verticalImplicitStageResidualNorm(
    result.state,
    base,
    dt,
    1,
    geometry,
    reference,
  );
  assert(checkedResidual < 2e-10, `rechecked implicit residual=${checkedResidual}`);
  assertPositiveColumn(result.state, reference);
});

test('Core v2 implicit HEVI stage preserves closed-column mass and total energy', () => {
  const horizontal = buildCubedSphere(2);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 150,
    EARTH.radius + 450,
    EARTH.radius + 1000,
    EARTH.radius + 2100,
    EARTH.radius + 4300,
    EARTH.radius + 8000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 284, 100300);
  const horizontalCell = 3;
  const base = referenceColumn(reference);
  for (let k = 1; k < geometry.nz - 1; k += 2) {
    const factor = 1 + 0.004 * Math.sin(1.3 * k);
    const state = conservedFromPrimitive({
      rho: reference.cellDensity[k]! * factor,
      pressure: reference.cellPressure[k]! * (2 - factor),
      velocity: [4 * k, -2 * k, 1.5 * k],
    }, reference.cellGeopotential[k]!);
    base[columnStateIndex(k, 0)] = state.rho;
    base[columnStateIndex(k, 1)] = state.momentum[0];
    base[columnStateIndex(k, 2)] = state.momentum[1];
    base[columnStateIndex(k, 3)] = state.momentum[2];
    base[columnStateIndex(k, 4)] = state.rhoE;
  }

  const before = columnIntegratedTotals(base, horizontalCell, geometry);
  const result = solveVerticalImplicitColumnStage(base, 12, horizontalCell, geometry, reference);
  const after = columnIntegratedTotals(result.state, horizontalCell, geometry);
  const massRelative = Math.abs(after.mass - before.mass) / Math.max(Math.abs(before.mass), 1);
  const energyRelative = Math.abs(after.energy - before.energy) / Math.max(Math.abs(before.energy), 1);
  assert(massRelative < 3e-10, `implicit closed-column mass drift=${massRelative}`);
  assert(energyRelative < 3e-10, `implicit closed-column total-energy drift=${energyRelative}`);
  assertPositiveColumn(result.state, reference);
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
console.log(`${passed}/${tests.length} Core v2 implicit-HEVI tests passed`);
