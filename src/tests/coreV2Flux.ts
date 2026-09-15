declare const process: { exitCode?: number };

import { EARTH } from '../core/constants.js';
import { norm3 } from '../core/math.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { createConservativeFields } from '../corev2/fields.js';
import { physicalEulerFlux } from '../corev2/eulerFlux.js';
import { slau2Flux } from '../corev2/slau2Flux.js';
import { conservedFromPrimitive } from '../corev2/state.js';
import {
  buildSphericalShellGeometry,
  shellCellIndex,
} from '../corev2/sphericalShellGeometry.js';
import { closedShellEulerRateFirstOrder } from '../corev2/shellEulerOperator.js';
import { assert, near, relative } from './assert.js';

interface Test { name: string; fn: () => void }
const tests: Test[] = [];
const test = (name: string, fn: () => void): void => { tests.push({ name, fn }); };

function nearFlux(
  actual: ReturnType<typeof slau2Flux>,
  expected: ReturnType<typeof physicalEulerFlux>,
  scale: number,
  label: string,
): void {
  near(actual.mass, expected.mass, 2e-13 * scale, `${label} mass`);
  near(actual.momentum[0], expected.momentum[0], 2e-13 * scale, `${label} mx`);
  near(actual.momentum[1], expected.momentum[1], 2e-13 * scale, `${label} my`);
  near(actual.momentum[2], expected.momentum[2], 2e-13 * scale, `${label} mz`);
  near(actual.totalEnergy, expected.totalEnergy, 2e-13 * scale, `${label} energy`);
}

test('Core v2 SLAU2 is exactly consistent with physical Euler flux for identical states', () => {
  const normals = [
    [1, 0, 0],
    [0, 1, 0],
    [0.6, -0.8, 0],
    [0.36, 0.48, 0.8],
  ] as const;
  const states = [
    conservedFromPrimitive({ rho: 1.2, velocity: [0.03, -0.02, 0.01], pressure: 101000 }),
    conservedFromPrimitive({ rho: 0.7, velocity: [75, -31, 14], pressure: 62000 }),
    conservedFromPrimitive({ rho: 0.25, velocity: [410, -20, 30], pressure: 12000 }),
  ];

  for (const state of states) {
    for (const n of normals) {
      relative(norm3(n), 1, 2e-15, 'test normal');
      const numerical = slau2Flux(state, state, n);
      const physical = physicalEulerFlux(state, n);
      const scale = Math.max(
        1,
        Math.abs(physical.mass),
        Math.abs(physical.momentum[0]),
        Math.abs(physical.momentum[1]),
        Math.abs(physical.momentum[2]),
        Math.abs(physical.totalEnergy),
      );
      nearFlux(numerical, physical, scale, 'identical-state consistency');
    }
  }
});

test('Core v2 SLAU2 has left-right/orientation antisymmetry', () => {
  const left = conservedFromPrimitive({
    rho: 1.08,
    velocity: [82, -17, 9],
    pressure: 87000,
  });
  const right = conservedFromPrimitive({
    rho: 0.63,
    velocity: [-34, 26, -11],
    pressure: 44000,
  });
  const n = [0.48, 0.64, 0.6] as const;
  relative(norm3(n), 1, 2e-15, 'orientation test normal');

  const forward = slau2Flux(left, right, n);
  const reverse = slau2Flux(right, left, [-n[0], -n[1], -n[2]]);
  const scale = Math.max(
    1,
    Math.abs(forward.mass),
    Math.abs(forward.momentum[0]),
    Math.abs(forward.momentum[1]),
    Math.abs(forward.momentum[2]),
    Math.abs(forward.totalEnergy),
  );
  near(forward.mass + reverse.mass, 0, 5e-13 * scale, 'orientation mass');
  near(forward.momentum[0] + reverse.momentum[0], 0, 5e-13 * scale, 'orientation mx');
  near(forward.momentum[1] + reverse.momentum[1], 0, 5e-13 * scale, 'orientation my');
  near(forward.momentum[2] + reverse.momentum[2], 0, 5e-13 * scale, 'orientation mz');
  near(forward.totalEnergy + reverse.totalEnergy, 0, 5e-13 * scale, 'orientation energy');
});

test('Core v2 SLAU2 preserves a stationary pressure-equilibrium contact', () => {
  const left = conservedFromPrimitive({ rho: 1.4, velocity: [0, 0, 0], pressure: 80000 });
  const right = conservedFromPrimitive({ rho: 0.4, velocity: [0, 0, 0], pressure: 80000 });
  const n = [0, 0, 1] as const;
  const flux = slau2Flux(left, right, n);
  near(flux.mass, 0, 0, 'contact mass');
  near(flux.totalEnergy, 0, 0, 'contact energy');
  near(flux.momentum[0], 0, 0, 'contact mx');
  near(flux.momentum[1], 0, 0, 'contact my');
  near(flux.momentum[2], 80000, 1e-11, 'contact pressure');
});

test('Core v2 closed shell uniform resting gas has zero cellwise Euler rate', () => {
  const horizontal = buildCubedSphere(5);
  const radii = new Float64Array([
    EARTH.radius,
    EARTH.radius + 1200,
    EARTH.radius + 4200,
  ]);
  const geometry = buildSphericalShellGeometry(horizontal, radii);
  const fields = createConservativeFields(horizontal.cellCount * geometry.nz);
  const state = conservedFromPrimitive({
    rho: 1,
    velocity: [0, 0, 0],
    pressure: 90000,
  });
  fields.rho.fill(state.rho);
  fields.momX.fill(state.momentum[0]);
  fields.momY.fill(state.momentum[1]);
  fields.momZ.fill(state.momentum[2]);
  fields.rhoE.fill(state.rhoE);

  const rate = closedShellEulerRateFirstOrder(fields, geometry);
  let worst = 0;
  let scale = 0;
  for (let q = 0; q < fields.rho.length; q++) {
    worst = Math.max(
      worst,
      Math.abs(rate.rho[q]!),
      Math.abs(rate.momX[q]!),
      Math.abs(rate.momY[q]!),
      Math.abs(rate.momZ[q]!),
      Math.abs(rate.rhoE[q]!),
    );
    scale = Math.max(scale, geometry.cellVolume[q]! * 90000 / 1000);
  }
  assert(worst / Math.max(scale, 1) < 2e-12, `uniform closed-shell residual=${worst / Math.max(scale, 1)}`);
});

test('Core v2 closed shell shared-face operator cannot create global mass or energy', () => {
  const horizontal = buildCubedSphere(4);
  const radii = new Float64Array([
    EARTH.radius,
    EARTH.radius + 800,
    EARTH.radius + 2200,
    EARTH.radius + 6000,
  ]);
  const geometry = buildSphericalShellGeometry(horizontal, radii);
  const fields = createConservativeFields(horizontal.cellCount * geometry.nz);

  for (let c = 0; c < horizontal.cellCount; c++) {
    const center = [
      horizontal.cellCenters[c * 3]!,
      horizontal.cellCenters[c * 3 + 1]!,
      horizontal.cellCenters[c * 3 + 2]!,
    ] as const;
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const rho = 0.7 + 0.15 * (center[0] + 1) + 0.03 * k;
      const pressure = 55000 + 9000 * center[2] + 1200 * k;
      const velocity = [
        35 * center[1] - 4 * k,
        -28 * center[0] + 3 * k,
        7 * center[2],
      ] as const;
      const state = conservedFromPrimitive({ rho, pressure, velocity });
      fields.rho[q] = state.rho;
      fields.momX[q] = state.momentum[0];
      fields.momY[q] = state.momentum[1];
      fields.momZ[q] = state.momentum[2];
      fields.rhoE[q] = state.rhoE;
    }
  }

  const rate = closedShellEulerRateFirstOrder(fields, geometry);
  let massRate = 0;
  let energyRate = 0;
  let massScale = 0;
  let energyScale = 0;
  for (let q = 0; q < fields.rho.length; q++) {
    massRate += rate.rho[q]!;
    energyRate += rate.rhoE[q]!;
    massScale += Math.abs(rate.rho[q]!);
    energyScale += Math.abs(rate.rhoE[q]!);
  }
  assert(Math.abs(massRate) <= 2e-13 * Math.max(massScale, 1), `global mass-rate residual=${massRate}`);
  assert(Math.abs(energyRate) <= 2e-13 * Math.max(energyScale, 1), `global energy-rate residual=${energyRate}`);
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
console.log(`${passed}/${tests.length} Core v2 flux tests passed`);
