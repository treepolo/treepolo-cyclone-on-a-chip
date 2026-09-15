declare const process: { exitCode?: number };

import { EARTH } from '../core/constants.js';
import { norm3 } from '../core/math.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { createConservativeFields, type ConservativeFields } from '../corev2/fields.js';
import {
  MAX_RECONSTRUCTION_NEIGHBORS,
  buildLinearReconstructionStencil,
  computeLimitedPrimitiveGradients,
  radialFaceCentroid,
  reconstructPrimitiveAt,
  shellCellCentroid,
  sideFaceCentroid,
} from '../corev2/reconstruction.js';
import {
  closedShellEulerRateSecondOrder,
} from '../corev2/shellEulerOperator.js';
import {
  conservedFromPrimitive,
  primitiveFromConserved,
  type PrimitiveCell,
} from '../corev2/state.js';
import {
  buildSphericalShellGeometry,
  shellCellIndex,
} from '../corev2/sphericalShellGeometry.js';
import { assert } from './assert.js';

interface Test { name: string; fn: () => void }
const tests: Test[] = [];
const test = (name: string, fn: () => void): void => { tests.push({ name, fn }); };

function writePrimitive(fields: ConservativeFields, q: number, primitive: PrimitiveCell): void {
  const s = conservedFromPrimitive(primitive);
  fields.rho[q] = s.rho;
  fields.momX[q] = s.momentum[0];
  fields.momY[q] = s.momentum[1];
  fields.momZ[q] = s.momentum[2];
  fields.rhoE[q] = s.rhoE;
}

function readPrimitive(fields: ConservativeFields, q: number): PrimitiveCell {
  return primitiveFromConserved({
    rho: fields.rho[q]!,
    momentum: [fields.momX[q]!, fields.momY[q]!, fields.momZ[q]!],
    rhoE: fields.rhoE[q]!,
  });
}

test('Core v2 least-squares stencil reproduces any Cartesian linear scalar gradient', () => {
  const horizontal = buildCubedSphere(5);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 600,
    EARTH.radius + 1800,
    EARTH.radius + 4200,
  ]));
  const stencil = buildLinearReconstructionStencil(geometry);
  const expected = [2.3e-7, -1.7e-7, 0.9e-7] as const;
  const expectedNorm = norm3(expected);

  for (let q = 0; q < stencil.neighborCount.length; q++) {
    const x0 = [
      stencil.cellCentroid[q * 3]!,
      stencil.cellCentroid[q * 3 + 1]!,
      stencil.cellCentroid[q * 3 + 2]!,
    ] as const;
    const q0 = expected[0] * x0[0] + expected[1] * x0[1] + expected[2] * x0[2];
    let gx = 0;
    let gy = 0;
    let gz = 0;
    for (let j = 0; j < stencil.neighborCount[q]!; j++) {
      const slot = q * MAX_RECONSTRUCTION_NEIGHBORS + j;
      const nq = stencil.neighborCell[slot]!;
      const xn = [
        stencil.cellCentroid[nq * 3]!,
        stencil.cellCentroid[nq * 3 + 1]!,
        stencil.cellCentroid[nq * 3 + 2]!,
      ] as const;
      const qn = expected[0] * xn[0] + expected[1] * xn[1] + expected[2] * xn[2];
      const delta = qn - q0;
      gx += stencil.coeffX[slot]! * delta;
      gy += stencil.coeffY[slot]! * delta;
      gz += stencil.coeffZ[slot]! * delta;
    }
    const error = Math.hypot(gx - expected[0], gy - expected[1], gz - expected[2]) / expectedNorm;
    assert(error < 2e-9, `linear-gradient relative error cell=${q}: ${error}`);
  }
});

test('Core v2 Barth-Jespersen reconstruction keeps face rho/p inside local center bounds', () => {
  const horizontal = buildCubedSphere(5);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 700,
    EARTH.radius + 2100,
    EARTH.radius + 5000,
  ]));
  const stencil = buildLinearReconstructionStencil(geometry);
  const cellCount = horizontal.cellCount * geometry.nz;
  const fields = createConservativeFields(cellCount);

  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const x = shellCellCentroid(geometry, c, k);
      const xr = x[0] / EARTH.radius;
      const yr = x[1] / EARTH.radius;
      const zr = x[2] / EARTH.radius;
      writePrimitive(fields, q, {
        rho: 0.9 + 0.10 * xr - 0.06 * yr + 0.04 * zr + 0.015 * k,
        pressure: 72000 + 6000 * xr + 3500 * yr - 2500 * zr + 900 * k,
        velocity: [45 * yr, -37 * xr, 8 * zr + k],
      });
    }
  }

  const gradients = computeLimitedPrimitiveGradients(fields, stencil);
  for (let q = 0; q < cellCount; q++) {
    const c = Math.floor(q / geometry.nz);
    const k = q % geometry.nz;
    const values = [readPrimitive(fields, q)];
    for (let j = 0; j < stencil.neighborCount[q]!; j++) {
      const nq = stencil.neighborCell[q * MAX_RECONSTRUCTION_NEIGHBORS + j]!;
      values.push(readPrimitive(fields, nq));
    }
    const rhoMin = Math.min(...values.map(v => v.rho));
    const rhoMax = Math.max(...values.map(v => v.rho));
    const pMin = Math.min(...values.map(v => v.pressure));
    const pMax = Math.max(...values.map(v => v.pressure));
    const positions = [] as Array<readonly [number, number, number]>;
    for (let s = 0; s < 4; s++) {
      const edgeId = horizontal.cellEdges[c * 4 + s]!;
      positions.push(sideFaceCentroid(geometry, edgeId, k));
    }
    positions.push(radialFaceCentroid(geometry, c, k));
    positions.push(radialFaceCentroid(geometry, c, k + 1));

    for (const position of positions) {
      const face = reconstructPrimitiveAt(fields, gradients, stencil, q, position);
      const rhoTol = 3e-13 * Math.max(1, Math.abs(rhoMin), Math.abs(rhoMax));
      const pTol = 3e-13 * Math.max(1, Math.abs(pMin), Math.abs(pMax));
      assert(face.rho > 0 && face.pressure > 0, `face state must remain positive cell=${q}`);
      assert(face.rho >= rhoMin - rhoTol && face.rho <= rhoMax + rhoTol,
        `rho limiter bound cell=${q}: ${face.rho} not in [${rhoMin},${rhoMax}]`);
      assert(face.pressure >= pMin - pTol && face.pressure <= pMax + pTol,
        `p limiter bound cell=${q}: ${face.pressure} not in [${pMin},${pMax}]`);
    }
  }
});

test('Core v2 second-order closed shell preserves a uniform resting state cell by cell', () => {
  const horizontal = buildCubedSphere(6);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 900,
    EARTH.radius + 2600,
  ]));
  const stencil = buildLinearReconstructionStencil(geometry);
  const fields = createConservativeFields(horizontal.cellCount * geometry.nz);
  const state = conservedFromPrimitive({ rho: 0.95, velocity: [0, 0, 0], pressure: 76000 });
  fields.rho.fill(state.rho);
  fields.momX.fill(0);
  fields.momY.fill(0);
  fields.momZ.fill(0);
  fields.rhoE.fill(state.rhoE);

  const rate = closedShellEulerRateSecondOrder(fields, geometry, stencil);
  let worst = 0;
  let momentumScale = 0;
  for (let q = 0; q < fields.rho.length; q++) {
    worst = Math.max(
      worst,
      Math.abs(rate.rho[q]!),
      Math.abs(rate.momX[q]!),
      Math.abs(rate.momY[q]!),
      Math.abs(rate.momZ[q]!),
      Math.abs(rate.rhoE[q]!),
    );
    momentumScale = Math.max(momentumScale, 76000 * Math.cbrt(geometry.cellVolume[q]! ** 2));
  }
  assert(worst / Math.max(momentumScale, 1) < 2e-11,
    `second-order uniform-state normalized residual=${worst / Math.max(momentumScale, 1)}`);
});

test('Core v2 second-order shared-face operator cannot create global mass or total energy', () => {
  const horizontal = buildCubedSphere(4);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 800,
    EARTH.radius + 2200,
    EARTH.radius + 6000,
  ]));
  const stencil = buildLinearReconstructionStencil(geometry);
  const cellCount = horizontal.cellCount * geometry.nz;
  const fields = createConservativeFields(cellCount);

  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const x = shellCellCentroid(geometry, c, k);
      const xr = x[0] / EARTH.radius;
      const yr = x[1] / EARTH.radius;
      const zr = x[2] / EARTH.radius;
      writePrimitive(fields, q, {
        rho: 0.75 + 0.12 * (xr + 1) + 0.02 * k,
        pressure: 60000 + 7000 * zr - 3500 * yr + 1000 * k,
        velocity: [42 * yr - 3 * k, -31 * xr + 2 * k, 6 * zr],
      });
    }
  }

  const rate = closedShellEulerRateSecondOrder(fields, geometry, stencil);
  let mass = 0;
  let energy = 0;
  let massScale = 0;
  let energyScale = 0;
  for (let q = 0; q < cellCount; q++) {
    mass += rate.rho[q]!;
    energy += rate.rhoE[q]!;
    massScale += Math.abs(rate.rho[q]!);
    energyScale += Math.abs(rate.rhoE[q]!);
  }
  assert(Math.abs(mass) <= 3e-13 * Math.max(massScale, 1), `second-order global mass-rate=${mass}`);
  assert(Math.abs(energy) <= 3e-13 * Math.max(energyScale, 1), `second-order global energy-rate=${energy}`);
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
console.log(`${passed}/${tests.length} Core v2 reconstruction tests passed`);
