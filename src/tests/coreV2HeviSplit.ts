declare const process: { exitCode?: number };

import { EARTH } from '../core/constants.js';
import { norm3 } from '../core/math.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { createConservativeFields } from '../corev2/fields.js';
import {
  closedShellHeviExplicitRemainderRate,
  closedShellVerticalStiffRateFirstOrder,
} from '../corev2/heviSplit.js';
import {
  buildIsothermalHydrostaticReference,
} from '../corev2/hydrostaticReference.js';
import { buildLinearReconstructionStencil } from '../corev2/reconstruction.js';
import { conservedFromPrimitive } from '../corev2/state.js';
import {
  buildSphericalShellGeometry,
  radialFaceVectorArea,
  shellCellIndex,
} from '../corev2/sphericalShellGeometry.js';
import { closedShellWellBalancedEulerGravityRate } from '../corev2/wellBalancedEulerGravity.js';
import { assert } from './assert.js';

interface Test { name: string; fn: () => void }
const tests: Test[] = [];
const test = (name: string, fn: () => void): void => { tests.push({ name, fn }); };

function fillReference(
  horizontalCells: number,
  nz: number,
  reference: ReturnType<typeof buildIsothermalHydrostaticReference>,
) {
  const fields = createConservativeFields(horizontalCells * nz);
  for (let c = 0; c < horizontalCells; c++) {
    for (let k = 0; k < nz; k++) {
      const q = shellCellIndex(c, k, nz);
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

function rateMagnitude(rate: ReturnType<typeof closedShellVerticalStiffRateFirstOrder>, q: number): number {
  return Math.max(
    Math.abs(rate.rho[q]!),
    Math.abs(rate.momX[q]!),
    Math.abs(rate.momY[q]!),
    Math.abs(rate.momZ[q]!),
    Math.abs(rate.rhoE[q]!),
  );
}

function pressureAreaScale(
  geometry: ReturnType<typeof buildSphericalShellGeometry>,
  reference: ReturnType<typeof buildIsothermalHydrostaticReference>,
  c: number,
  k: number,
): number {
  const bottom = reference.radialFacePressure[k]! * norm3(radialFaceVectorArea(geometry, c, k));
  const top = reference.radialFacePressure[k + 1]! * norm3(radialFaceVectorArea(geometry, c, k + 1));
  return Math.max(bottom, top, 1);
}

test('Core v2 HEVI vertical stiff operator is hydrostatic well-balanced without large-force cancellation', () => {
  const horizontal = buildCubedSphere(5);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 70,
    EARTH.radius + 350,
    EARTH.radius + 1300,
    EARTH.radius + 4200,
    EARTH.radius + 12000,
    EARTH.radius + 40000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 282, 100700);
  const fields = fillReference(horizontal.cellCount, geometry.nz, reference);
  const rate = closedShellVerticalStiffRateFirstOrder(fields, geometry, reference);

  let worstMass = 0;
  let worstEnergy = 0;
  let worstMomentumRelative = 0;
  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      worstMass = Math.max(worstMass, Math.abs(rate.rho[q]!));
      worstEnergy = Math.max(worstEnergy, Math.abs(rate.rhoE[q]!));
      const momentum = norm3([rate.momX[q]!, rate.momY[q]!, rate.momZ[q]!]);
      worstMomentumRelative = Math.max(
        worstMomentumRelative,
        momentum / pressureAreaScale(geometry, reference, c, k),
      );
    }
  }
  assert(worstMass === 0, `HEVI hydrostatic mass rate must be exact zero; got ${worstMass}`);
  assert(worstEnergy === 0, `HEVI hydrostatic energy rate must be exact zero; got ${worstEnergy}`);
  assert(worstMomentumRelative < 3e-15,
    `HEVI hydrostatic reference-subtracted momentum residual=${worstMomentumRelative}`);
});

test('Core v2 HEVI split recombines to the full second-order Euler+gravity operator', () => {
  const horizontal = buildCubedSphere(4);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 250,
    EARTH.radius + 1100,
    EARTH.radius + 3600,
    EARTH.radius + 10000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 286, 100300);
  const stencil = buildLinearReconstructionStencil(geometry);
  const fields = createConservativeFields(horizontal.cellCount * geometry.nz);

  for (let c = 0; c < horizontal.cellCount; c++) {
    const x = horizontal.cellCenters[c * 3]!;
    const y = horizontal.cellCenters[c * 3 + 1]!;
    const z = horizontal.cellCenters[c * 3 + 2]!;
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const state = conservedFromPrimitive({
        rho: reference.cellDensity[k]! * (1 + 0.012 * x - 0.007 * z),
        pressure: reference.cellPressure[k]! * (1 + 0.018 * y + 0.004 * k),
        velocity: [31 * y - 1.5 * k, -24 * x + 0.5 * k, 5 * z],
      }, reference.cellGeopotential[k]!);
      fields.rho[q] = state.rho;
      fields.momX[q] = state.momentum[0];
      fields.momY[q] = state.momentum[1];
      fields.momZ[q] = state.momentum[2];
      fields.rhoE[q] = state.rhoE;
    }
  }

  const full = closedShellWellBalancedEulerGravityRate(fields, geometry, stencil, reference);
  const stiff = closedShellVerticalStiffRateFirstOrder(fields, geometry, reference);
  const explicit = closedShellHeviExplicitRemainderRate(fields, geometry, stencil, reference);
  let worstRelative = 0;
  for (let q = 0; q < fields.rho.length; q++) {
    const fullValues = [full.rho[q]!, full.momX[q]!, full.momY[q]!, full.momZ[q]!, full.rhoE[q]!];
    const recombined = [
      stiff.rho[q]! + explicit.rho[q]!,
      stiff.momX[q]! + explicit.momX[q]!,
      stiff.momY[q]! + explicit.momY[q]!,
      stiff.momZ[q]! + explicit.momZ[q]!,
      stiff.rhoE[q]! + explicit.rhoE[q]!,
    ];
    for (let v = 0; v < 5; v++) {
      worstRelative = Math.max(
        worstRelative,
        Math.abs(recombined[v]! - fullValues[v]!) / Math.max(1, Math.abs(fullValues[v]!)),
      );
    }
  }
  assert(worstRelative < 3e-15, `HEVI split recombination relative error=${worstRelative}`);
});

test('Core v2 HEVI vertical stiff operator is strictly column-local', () => {
  const horizontal = buildCubedSphere(4);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 300,
    EARTH.radius + 1500,
    EARTH.radius + 5000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 280, 100000);
  const fields = fillReference(horizontal.cellCount, geometry.nz, reference);
  const perturbedColumn = 7;
  const k = 1;
  const q = shellCellIndex(perturbedColumn, k, geometry.nz);
  const state = conservedFromPrimitive({
    rho: reference.cellDensity[k]! * 1.02,
    pressure: reference.cellPressure[k]! * 1.03,
    velocity: [12, -7, 4],
  }, reference.cellGeopotential[k]!);
  fields.rho[q] = state.rho;
  fields.momX[q] = state.momentum[0];
  fields.momY[q] = state.momentum[1];
  fields.momZ[q] = state.momentum[2];
  fields.rhoE[q] = state.rhoE;

  const rate = closedShellVerticalStiffRateFirstOrder(fields, geometry, reference);
  let outsideRelative = 0;
  let insideWorst = 0;
  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let level = 0; level < geometry.nz; level++) {
      const cell = shellCellIndex(c, level, geometry.nz);
      if (c === perturbedColumn) {
        insideWorst = Math.max(insideWorst, rateMagnitude(rate, cell));
      } else {
        outsideRelative = Math.max(
          outsideRelative,
          rateMagnitude(rate, cell) / pressureAreaScale(geometry, reference, c, level),
        );
      }
    }
  }
  assert(insideWorst > 0, 'perturbed column must produce a nonzero stiff rate');
  assert(outsideRelative < 3e-15,
    `HEVI vertical residual leaked into untouched columns: relative=${outsideRelative}`);
});

test('Core v2 HEVI vertical stiff operator conserves global mass and total energy in a closed column set', () => {
  const horizontal = buildCubedSphere(3);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 200,
    EARTH.radius + 900,
    EARTH.radius + 2800,
    EARTH.radius + 8000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 284, 100200);
  const fields = fillReference(horizontal.cellCount, geometry.nz, reference);

  for (let c = 0; c < horizontal.cellCount; c += 5) {
    const k = (c / 5) % geometry.nz;
    const q = shellCellIndex(c, k, geometry.nz);
    const x = horizontal.cellCenters[c * 3]!;
    const state = conservedFromPrimitive({
      rho: reference.cellDensity[k]! * (1.01 + 0.003 * x),
      pressure: reference.cellPressure[k]! * (0.99 + 0.006 * x),
      velocity: [8 + 2 * x, -5, 3],
    }, reference.cellGeopotential[k]!);
    fields.rho[q] = state.rho;
    fields.momX[q] = state.momentum[0];
    fields.momY[q] = state.momentum[1];
    fields.momZ[q] = state.momentum[2];
    fields.rhoE[q] = state.rhoE;
  }

  const rate = closedShellVerticalStiffRateFirstOrder(fields, geometry, reference);
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
  assert(Math.abs(mass) <= 3e-13 * Math.max(1, massScale), `HEVI stiff global mass rate=${mass}`);
  assert(Math.abs(energy) <= 3e-13 * Math.max(1, energyScale), `HEVI stiff global energy rate=${energy}`);
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
console.log(`${passed}/${tests.length} Core v2 HEVI-split tests passed`);
