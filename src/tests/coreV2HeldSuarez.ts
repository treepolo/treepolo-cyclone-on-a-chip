declare const process: { exitCode?: number };

import { DRY_AIR, EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { createConservativeFields } from '../corev2/fields.js';
import {
  applyHeldSuarezForcing,
  heldSuarezEquilibriumTemperature,
} from '../corev2/heldSuarezForcing.js';
import { buildIsothermalHydrostaticReference } from '../corev2/hydrostaticReference.js';
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

function tangentAt(x: number, y: number, z: number): [number, number, number] {
  let tx = -y;
  let ty = x;
  let tz = 0;
  let norm = Math.hypot(tx, ty, tz);
  if (norm < 1e-8) {
    tx = 0;
    ty = -z;
    tz = y;
    norm = Math.hypot(tx, ty, tz);
  }
  return [tx / norm, ty / norm, tz / norm];
}

test('Core v2 Held-Suarez forcing relaxes temperature and damps only tangential momentum', () => {
  const horizontal = buildCubedSphere(2);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 1000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 288, 100000);
  const fields = createConservativeFields(horizontal.cellCount);

  for (let c = 0; c < horizontal.cellCount; c++) {
    const x = horizontal.cellCenters[c * 3]!;
    const y = horizontal.cellCenters[c * 3 + 1]!;
    const z = horizontal.cellCenters[c * 3 + 2]!;
    const tangent = tangentAt(x, y, z);
    const rho = reference.cellDensity[0]!;
    const temperature = 330;
    const pressure = rho * DRY_AIR.rd * temperature;
    const velocity: [number, number, number] = [
      40 * tangent[0] + 2 * x,
      40 * tangent[1] + 2 * y,
      40 * tangent[2] + 2 * z,
    ];
    const state = conservedFromPrimitive(
      { rho, pressure, velocity },
      reference.cellGeopotential[0]!,
    );
    const q = shellCellIndex(c, 0, geometry.nz);
    fields.rho[q] = state.rho;
    fields.momX[q] = state.momentum[0];
    fields.momY[q] = state.momentum[1];
    fields.momZ[q] = state.momentum[2];
    fields.rhoE[q] = state.rhoE;
  }

  const c = 0;
  const q = shellCellIndex(c, 0, geometry.nz);
  const x = horizontal.cellCenters[c * 3]!;
  const y = horizontal.cellCenters[c * 3 + 1]!;
  const z = horizontal.cellCenters[c * 3 + 2]!;
  const tangent = tangentAt(x, y, z);
  const rhoBefore = fields.rho[q]!;
  const mxBefore = fields.momX[q]!;
  const myBefore = fields.momY[q]!;
  const mzBefore = fields.momZ[q]!;
  const radialBefore = (mxBefore * x + myBefore * y + mzBefore * z) / rhoBefore;
  const tangentBefore = (
    mxBefore * tangent[0] + myBefore * tangent[1] + mzBefore * tangent[2]
  ) / rhoBefore;
  const pressureBefore = pressureFromConservedRelativeToReference({
    rho: rhoBefore,
    momentum: [mxBefore, myBefore, mzBefore],
    rhoE: fields.rhoE[q]!,
  }, reference.cellDensity[0]!, reference.cellPressure[0]!, reference.cellGeopotential[0]!);
  const temperatureBefore = pressureBefore / (rhoBefore * DRY_AIR.rd);
  const latitude = Math.asin(z);
  const equilibrium = heldSuarezEquilibriumTemperature(latitude, pressureBefore);

  applyHeldSuarezForcing(fields, 86400, geometry, reference);

  const rhoAfter = fields.rho[q]!;
  const mxAfter = fields.momX[q]!;
  const myAfter = fields.momY[q]!;
  const mzAfter = fields.momZ[q]!;
  const radialAfter = (mxAfter * x + myAfter * y + mzAfter * z) / rhoAfter;
  const tangentAfter = (
    mxAfter * tangent[0] + myAfter * tangent[1] + mzAfter * tangent[2]
  ) / rhoAfter;
  const pressureAfter = pressureFromConservedRelativeToReference({
    rho: rhoAfter,
    momentum: [mxAfter, myAfter, mzAfter],
    rhoE: fields.rhoE[q]!,
  }, reference.cellDensity[0]!, reference.cellPressure[0]!, reference.cellGeopotential[0]!);
  const temperatureAfter = pressureAfter / (rhoAfter * DRY_AIR.rd);

  assert(rhoAfter === rhoBefore, 'Held-Suarez forcing must not change density');
  assert(
    Math.abs(radialAfter - radialBefore) < 1e-12,
    `Held-Suarez drag changed radial velocity: before=${radialBefore}, after=${radialAfter}`,
  );
  assert(
    Math.abs(tangentAfter) < Math.abs(tangentBefore),
    `Held-Suarez drag did not damp tangential velocity: before=${tangentBefore}, after=${tangentAfter}`,
  );
  assert(
    Math.abs(temperatureAfter - equilibrium) < Math.abs(temperatureBefore - equilibrium),
    `Held-Suarez thermal forcing did not relax toward Teq: before=${temperatureBefore}, after=${temperatureAfter}, Teq=${equilibrium}`,
  );
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
console.log(`${passed}/${tests.length} Core v2 Held-Suarez tests passed`);
