declare const process: { exitCode?: number };

import { DRY_AIR, EARTH } from '../core/constants.js';
import { norm3 } from '../core/math.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import {
  buildIsothermalHydrostaticReference,
  integratedCellGravityForce,
  integratedReferencePressureForce,
} from '../corev2/hydrostaticReference.js';
import { buildSphericalShellGeometry } from '../corev2/sphericalShellGeometry.js';
import { assert, relative } from './assert.js';

interface Test { name: string; fn: () => void }
const tests: Test[] = [];
const test = (name: string, fn: () => void): void => { tests.push({ name, fn }); };

test('Core v2 isothermal reference obeys the ideal-gas relation in cell averages', () => {
  const horizontal = buildCubedSphere(4);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 80,
    EARTH.radius + 650,
    EARTH.radius + 3100,
    EARTH.radius + 12000,
    EARTH.radius + 40000,
  ]));
  const temperature = 287.3;
  const reference = buildIsothermalHydrostaticReference(geometry, temperature, 100800);
  for (let k = 0; k < geometry.nz; k++) {
    relative(
      reference.cellPressure[k]!,
      reference.cellDensity[k]! * DRY_AIR.rd * temperature,
      2e-15,
      `reference EOS layer=${k}`,
    );
    assert(reference.radialFacePressure[k + 1]! < reference.radialFacePressure[k]!, 'pressure must decrease upward');
    assert(reference.cellDensity[k]! > 0, 'reference density must remain positive');
  }
});

test('Core v2 reference pressure and gravity forces cancel cell by cell to roundoff', () => {
  const horizontal = buildCubedSphere(7);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 120,
    EARTH.radius + 900,
    EARTH.radius + 3200,
    EARTH.radius + 8100,
    EARTH.radius + 17000,
    EARTH.radius + 40000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 288, 100000);

  let worst = 0;
  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const pressure = integratedReferencePressureForce(geometry, reference, c, k);
      const gravity = integratedCellGravityForce(
        geometry,
        c,
        k,
        reference.cellDensity[k]!,
      );
      const residual = [
        pressure[0] + gravity[0],
        pressure[1] + gravity[1],
        pressure[2] + gravity[2],
      ] as const;
      const scale = Math.max(norm3(pressure), norm3(gravity), 1);
      worst = Math.max(worst, norm3(residual) / scale);
    }
  }
  assert(worst < 2e-10, `hydrostatic force relative residual=${worst}`);
});

test('Core v2 exact side-face pressure average converges to local analytic pressure in thin layers', () => {
  const horizontal = buildCubedSphere(3);
  const r0 = EARTH.radius;
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    r0,
    r0 + 10,
    r0 + 20,
  ]));
  const temperature = 290;
  const pSurface = 101325;
  const reference = buildIsothermalHydrostaticReference(geometry, temperature, pSurface);
  const a = EARTH.gravity / (DRY_AIR.rd * temperature);
  for (let k = 0; k < geometry.nz; k++) {
    const rm = 0.5 * (geometry.radiusInterface[k]! + geometry.radiusInterface[k + 1]!);
    const expected = pSurface * Math.exp(-a * (rm - r0));
    relative(reference.sideFacePressure[k]!, expected, 2e-7, `thin-layer side pressure k=${k}`);
  }
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
console.log(`${passed}/${tests.length} Core v2 hydrostatic-reference tests passed`);
