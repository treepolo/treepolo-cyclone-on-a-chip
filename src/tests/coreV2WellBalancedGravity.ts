declare const process: { exitCode?: number };

import { EARTH } from '../core/constants.js';
import { norm3 } from '../core/math.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { createConservativeFields } from '../corev2/fields.js';
import {
  buildIsothermalHydrostaticReference,
  integratedCellGravityForce,
} from '../corev2/hydrostaticReference.js';
import { buildLinearReconstructionStencil } from '../corev2/reconstruction.js';
import { conservedFromPrimitive } from '../corev2/state.js';
import {
  buildSphericalShellGeometry,
  shellCellIndex,
} from '../corev2/sphericalShellGeometry.js';
import {
  closedShellWellBalancedEulerGravityRate,
  integratedMomentumRateNorm,
} from '../corev2/wellBalancedEulerGravity.js';
import { assert } from './assert.js';

interface Test { name: string; fn: () => void }
const tests: Test[] = [];
const test = (name: string, fn: () => void): void => { tests.push({ name, fn }); };

function fillExactReferenceState(
  n: number,
  nz: number,
  reference: ReturnType<typeof buildIsothermalHydrostaticReference>,
) {
  const fields = createConservativeFields(n * nz);
  for (let c = 0; c < n; c++) {
    for (let k = 0; k < nz; k++) {
      const q = shellCellIndex(c, k, nz);
      const state = conservedFromPrimitive({
        rho: reference.cellDensity[k]!,
        velocity: [0, 0, 0],
        pressure: reference.cellPressure[k]!,
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

test('Core v2 well-balanced operator leaves the exact hydrostatic atmosphere motionless cell by cell', () => {
  const horizontal = buildCubedSphere(6);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 120,
    EARTH.radius + 650,
    EARTH.radius + 1900,
    EARTH.radius + 4800,
    EARTH.radius + 11000,
    EARTH.radius + 24000,
    EARTH.radius + 40000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 288, 100000);
  const stencil = buildLinearReconstructionStencil(geometry);
  const fields = fillExactReferenceState(horizontal.cellCount, geometry.nz, reference);
  const rate = closedShellWellBalancedEulerGravityRate(fields, geometry, stencil, reference);

  let worstMass = 0;
  let worstEnergy = 0;
  let worstMomentumRelative = 0;
  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      worstMass = Math.max(worstMass, Math.abs(rate.rho[q]!));
      worstEnergy = Math.max(worstEnergy, Math.abs(rate.rhoE[q]!));
      const gravity = integratedCellGravityForce(geometry, c, k, fields.rho[q]!);
      const scale = Math.max(norm3(gravity), 1);
      worstMomentumRelative = Math.max(
        worstMomentumRelative,
        integratedMomentumRateNorm(rate, q) / scale,
      );
    }
  }
  assert(worstMass < 1e-8, `hydrostatic max mass rate=${worstMass}`);
  assert(worstEnergy < 1e-2, `hydrostatic max energy rate=${worstEnergy}`);
  assert(worstMomentumRelative < 5e-10,
    `hydrostatic max relative momentum residual=${worstMomentumRelative}`);
});

test('Core v2 well-balanced operator keeps closed-domain mass and total energy conservative away from reference', () => {
  const horizontal = buildCubedSphere(4);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 300,
    EARTH.radius + 1300,
    EARTH.radius + 4100,
    EARTH.radius + 12000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 285, 100500);
  const stencil = buildLinearReconstructionStencil(geometry);
  const fields = createConservativeFields(horizontal.cellCount * geometry.nz);

  for (let c = 0; c < horizontal.cellCount; c++) {
    const x = horizontal.cellCenters[c * 3]!;
    const y = horizontal.cellCenters[c * 3 + 1]!;
    const z = horizontal.cellCenters[c * 3 + 2]!;
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const rho = reference.cellDensity[k]! * (1 + 0.015 * x - 0.009 * y);
      const pressure = reference.cellPressure[k]! * (1 + 0.02 * z + 0.006 * x);
      const state = conservedFromPrimitive({
        rho,
        pressure,
        velocity: [27 * y - 2 * k, -19 * x + k, 4 * z],
      }, reference.cellGeopotential[k]!);
      fields.rho[q] = state.rho;
      fields.momX[q] = state.momentum[0];
      fields.momY[q] = state.momentum[1];
      fields.momZ[q] = state.momentum[2];
      fields.rhoE[q] = state.rhoE;
    }
  }

  const rate = closedShellWellBalancedEulerGravityRate(fields, geometry, stencil, reference);
  let mass = 0;
  let energy = 0;
  let massScale = 0;
  let energyScale = 0;
  for (let q = 0; q < fields.rho.length; q++) {
    mass += rate.rho[q]!;
    energy += rate.rhoE[q]!;
    massScale += Math.abs(rate.rho[q]!);
    energyScale += Math.abs(rate.rhoE[q]!);
  }
  assert(Math.abs(mass) <= 5e-13 * Math.max(massScale, 1),
    `well-balanced global mass-rate residual=${mass}`);
  assert(Math.abs(energy) <= 5e-13 * Math.max(energyScale, 1),
    `well-balanced global total-energy-rate residual=${energy}`);
});

test('Core v2 hydrostatic balance survives strong vertical grid stretching without damping', () => {
  const horizontal = buildCubedSphere(3);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 40,
    EARTH.radius + 140,
    EARTH.radius + 420,
    EARTH.radius + 1200,
    EARTH.radius + 3400,
    EARTH.radius + 9000,
    EARTH.radius + 20000,
    EARTH.radius + 40000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 270, 101000);
  const stencil = buildLinearReconstructionStencil(geometry);
  const fields = fillExactReferenceState(horizontal.cellCount, geometry.nz, reference);
  const rate = closedShellWellBalancedEulerGravityRate(fields, geometry, stencil, reference);

  let worst = 0;
  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const gravity = integratedCellGravityForce(geometry, c, k, fields.rho[q]!);
      worst = Math.max(worst, integratedMomentumRateNorm(rate, q) / Math.max(norm3(gravity), 1));
    }
  }
  assert(worst < 8e-10, `stretched-grid hydrostatic relative residual=${worst}`);
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
console.log(`${passed}/${tests.length} Core v2 well-balanced gravity tests passed`);
