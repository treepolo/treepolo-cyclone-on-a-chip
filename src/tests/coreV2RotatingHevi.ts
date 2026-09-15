declare const process: { exitCode?: number };

import { EARTH, type PlanetConfig } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { createConservativeFields, type ConservativeFields } from '../corev2/fields.js';
import { integratedTotals } from '../corev2/finiteVolume.js';
import { heviImexSsp2Step } from '../corev2/heviImex.js';
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
  horizontalCells: number,
  geometry: Geometry,
  reference: Reference,
): ConservativeFields {
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

function maxRelativeStateDifference(a: ConservativeFields, b: ConservativeFields): number {
  let worst = 0;
  for (let q = 0; q < a.rho.length; q++) {
    const av = [a.rho[q]!, a.momX[q]!, a.momY[q]!, a.momZ[q]!, a.rhoE[q]!];
    const bv = [b.rho[q]!, b.momX[q]!, b.momY[q]!, b.momZ[q]!, b.rhoE[q]!];
    for (let v = 0; v < 5; v++) {
      worst = Math.max(
        worst,
        Math.abs(av[v]! - bv[v]!) / Math.max(1, Math.abs(av[v]!), Math.abs(bv[v]!)),
      );
    }
  }
  return worst;
}

test('Core v2 rotating HEVI preserves exact hydrostatic rest', () => {
  const horizontal = buildCubedSphere(3);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 120,
    EARTH.radius + 500,
    EARTH.radius + 1800,
    EARTH.radius + 6000,
    EARTH.radius + 18000,
    EARTH.radius + 40000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 283, 100500);
  const stencil = buildLinearReconstructionStencil(geometry);
  const initial = fillReference(horizontal.cellCount, geometry, reference);
  const next = rotatingHeviImexSsp2Step(
    initial,
    30,
    geometry,
    stencil,
    reference,
    EARTH,
  ).fields;

  let worstSpeed = 0;
  let worstDensityRelative = 0;
  let worstEnergyRelative = 0;
  for (let q = 0; q < initial.rho.length; q++) {
    worstDensityRelative = Math.max(
      worstDensityRelative,
      Math.abs(next.rho[q]! - initial.rho[q]!) / initial.rho[q]!,
    );
    worstEnergyRelative = Math.max(
      worstEnergyRelative,
      Math.abs(next.rhoE[q]! - initial.rhoE[q]!) / Math.max(1, Math.abs(initial.rhoE[q]!)),
    );
    worstSpeed = Math.max(
      worstSpeed,
      Math.hypot(next.momX[q]!, next.momY[q]!, next.momZ[q]!) / next.rho[q]!,
    );
  }
  assert(worstDensityRelative < 2e-14, `rotating hydrostatic density drift=${worstDensityRelative}`);
  assert(worstEnergyRelative < 2e-14, `rotating hydrostatic energy drift=${worstEnergyRelative}`);
  assert(worstSpeed < 2e-10, `rotating hydrostatic spurious speed=${worstSpeed}`);
});

test('Core v2 rotating HEVI conserves closed-domain mass and total energy over repeated steps', () => {
  const horizontal = buildCubedSphere(3);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 300,
    EARTH.radius + 1200,
    EARTH.radius + 3500,
    EARTH.radius + 9000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 287, 100000);
  const stencil = buildLinearReconstructionStencil(geometry);
  let fields = fillReference(horizontal.cellCount, geometry, reference);

  for (let c = 0; c < horizontal.cellCount; c++) {
    const x = horizontal.cellCenters[c * 3]!;
    const y = horizontal.cellCenters[c * 3 + 1]!;
    const z = horizontal.cellCenters[c * 3 + 2]!;
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const state = conservedFromPrimitive({
        rho: reference.cellDensity[k]! * (1 + 0.0015 * x - 0.0008 * y),
        pressure: reference.cellPressure[k]! * (1 + 0.002 * z + 0.0003 * k),
        velocity: [12 * y + 2 * z, -9 * x, 1.2 * z],
      }, reference.cellGeopotential[k]!);
      fields.rho[q] = state.rho;
      fields.momX[q] = state.momentum[0];
      fields.momY[q] = state.momentum[1];
      fields.momZ[q] = state.momentum[2];
      fields.rhoE[q] = state.rhoE;
    }
  }

  const before = integratedTotals(fields, geometry.cellVolume);
  for (let step = 0; step < 12; step++) {
    fields = rotatingHeviImexSsp2Step(
      fields,
      1,
      geometry,
      stencil,
      reference,
      EARTH,
    ).fields;
  }
  const after = integratedTotals(fields, geometry.cellVolume);
  const massRelative = Math.abs(after.mass - before.mass) / Math.max(1, Math.abs(before.mass));
  const energyRelative = Math.abs(after.totalEnergy - before.totalEnergy) /
    Math.max(1, Math.abs(before.totalEnergy));
  assert(massRelative < 2e-12, `rotating repeated-step mass drift=${massRelative}`);
  assert(energyRelative < 2e-12, `rotating repeated-step total-energy drift=${energyRelative}`);
});

test('Core v2 rotating HEVI reduces to the non-rotating IMEX method when omega is zero', () => {
  const horizontal = buildCubedSphere(2);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 500,
    EARTH.radius + 1600,
    EARTH.radius + 4200,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 285, 100200);
  const stencil = buildLinearReconstructionStencil(geometry);
  const fields = fillReference(horizontal.cellCount, geometry, reference);
  for (let c = 0; c < horizontal.cellCount; c++) {
    const x = horizontal.cellCenters[c * 3]!;
    const y = horizontal.cellCenters[c * 3 + 1]!;
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const state = conservedFromPrimitive({
        rho: reference.cellDensity[k]! * (1 + 2e-4 * x),
        pressure: reference.cellPressure[k]! * (1 + 3e-4 * y),
        velocity: [0.8 * y, -0.6 * x, 0.1 * (k - 1)],
      }, reference.cellGeopotential[k]!);
      fields.rho[q] = state.rho;
      fields.momX[q] = state.momentum[0];
      fields.momY[q] = state.momentum[1];
      fields.momZ[q] = state.momentum[2];
      fields.rhoE[q] = state.rhoE;
    }
  }

  const zeroOmega: PlanetConfig = { ...EARTH, omega: 0 };
  const direct = heviImexSsp2Step(fields, 0.5, geometry, stencil, reference, zeroOmega).fields;
  const rotating = rotatingHeviImexSsp2Step(
    fields,
    0.5,
    geometry,
    stencil,
    reference,
    zeroOmega,
  ).fields;
  const difference = maxRelativeStateDifference(direct, rotating);
  assert(difference === 0, `zero-omega rotating wrapper must be identical; difference=${difference}`);
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
console.log(`${passed}/${tests.length} Core v2 rotating-HEVI tests passed`);
