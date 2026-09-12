declare const process: { exitCode?: number };

import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildCoreV2GpuInternalFaceStaticData } from '../gpu/coreV2FaceFluxGpu.js';
import { buildCoreV2GpuRateStaticData } from '../gpu/coreV2RateGatherGpu.js';
import { buildIsothermalHydrostaticReference } from '../corev2/hydrostaticReference.js';
import { buildLinearReconstructionStencil } from '../corev2/reconstruction.js';
import {
  buildSphericalShellGeometry,
  shellCellIndex,
} from '../corev2/sphericalShellGeometry.js';
import { assert } from './assert.js';

interface Test { name: string; fn: () => void }
const tests: Test[] = [];
const test = (name: string, fn: () => void): void => { tests.push({ name, fn }); };

test('Core v2 GPU internal-face list contains every shared 3-D face exactly once', () => {
  const horizontal = buildCubedSphere(4);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 100,
    EARTH.radius + 600,
    EARTH.radius + 2500,
    EARTH.radius + 9000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 284, 100100);
  const stencil = buildLinearReconstructionStencil(geometry);
  const faces = buildCoreV2GpuInternalFaceStaticData(geometry, stencil, reference);
  const expected = horizontal.edgeCount * geometry.nz +
    horizontal.cellCount * (geometry.nz - 1);
  assert(faces.faceCount === expected, `GPU shared face count ${faces.faceCount} != ${expected}`);

  const seen = new Set<string>();
  for (let f = 0; f < faces.faceCount; f++) {
    const left = faces.faceCells[f * 2]!;
    const right = faces.faceCells[f * 2 + 1]!;
    assert(left !== right, `GPU shared face ${f} has identical cells`);
    const key = left < right ? `${left}:${right}` : `${right}:${left}`;
    assert(!seen.has(key), `duplicate GPU shared face pair ${key}`);
    seen.add(key);
    const a = f * 4;
    const area = Math.hypot(
      faces.vectorArea[a]!,
      faces.vectorArea[a + 1]!,
      faces.vectorArea[a + 2]!,
    );
    assert(area > 0 && Number.isFinite(area), `GPU shared face ${f} invalid area=${area}`);
    assert(faces.leftDxReferencePressure[a + 3]! > 0,
      `GPU shared face ${f} non-positive reference pressure`);
    assert(Number.isFinite(faces.rightDxGeopotential[a + 3]!),
      `GPU shared face ${f} invalid geopotential`);
  }
});

test('Core v2 GPU cell gather gives every shared face one minus and one plus incidence', () => {
  const horizontal = buildCubedSphere(3);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 150,
    EARTH.radius + 800,
    EARTH.radius + 3000,
    EARTH.radius + 10000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 287, 100000);
  const stencil = buildLinearReconstructionStencil(geometry);
  const faces = buildCoreV2GpuInternalFaceStaticData(geometry, stencil, reference);
  const gather = buildCoreV2GpuRateStaticData(geometry, stencil, reference, faces);

  const minus = new Uint8Array(faces.faceCount);
  const plus = new Uint8Array(faces.faceCount);
  for (let q = 0; q < gather.cellCount; q++) {
    let localCount = 0;
    for (let slot = 0; slot < 6; slot++) {
      const face = gather.cellFacePairs[q * 12 + slot * 2]!;
      const sign = gather.cellFacePairs[q * 12 + slot * 2 + 1]!;
      if (face < 0) continue;
      localCount++;
      assert(face < faces.faceCount, `cell ${q} invalid face id ${face}`);
      assert(sign === -1 || sign === 1, `cell ${q} face ${face} invalid sign ${sign}`);
      if (sign < 0) minus[face]++;
      else plus[face]++;
      const left = faces.faceCells[face * 2]!;
      const right = faces.faceCells[face * 2 + 1]!;
      assert(
        (sign === -1 && q === left) || (sign === 1 && q === right),
        `cell ${q} face ${face} incidence disagrees with left/right orientation`,
      );
    }
    const k = q % geometry.nz;
    const expectedLocal = 4 + (k > 0 ? 1 : 0) + (k + 1 < geometry.nz ? 1 : 0);
    assert(localCount === expectedLocal,
      `cell ${q} has ${localCount} internal faces, expected ${expectedLocal}`);
  }
  for (let f = 0; f < faces.faceCount; f++) {
    assert(minus[f] === 1 && plus[f] === 1,
      `shared face ${f} incidence minus/plus=${minus[f]}/${plus[f]}`);
  }
});

test('Core v2 GPU shared-face gather is globally conservative for arbitrary face fluxes', () => {
  const horizontal = buildCubedSphere(3);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 200,
    EARTH.radius + 900,
    EARTH.radius + 3500,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 285, 100400);
  const stencil = buildLinearReconstructionStencil(geometry);
  const faces = buildCoreV2GpuInternalFaceStaticData(geometry, stencil, reference);
  const gather = buildCoreV2GpuRateStaticData(geometry, stencil, reference, faces);

  const global = new Float64Array(5);
  let absolute = 0;
  for (let q = 0; q < gather.cellCount; q++) {
    for (let slot = 0; slot < 6; slot++) {
      const face = gather.cellFacePairs[q * 12 + slot * 2]!;
      if (face < 0) continue;
      const sign = gather.cellFacePairs[q * 12 + slot * 2 + 1]!;
      const values = [
        Math.sin(face * 0.37) * 1e7,
        Math.cos(face * 0.13) * 2e10,
        Math.sin(face * 0.17 + 0.2) * 3e10,
        Math.cos(face * 0.11 - 0.7) * 4e10,
        Math.sin(face * 0.07 + 0.9) * 5e14,
      ];
      for (let v = 0; v < 5; v++) {
        global[v] += sign * values[v]!;
        absolute += Math.abs(values[v]!);
      }
    }
  }
  const worst = Math.max(...Array.from(global, Math.abs));
  assert(worst <= 5e-15 * Math.max(1, absolute),
    `GPU shared-face topology global conservation residual=${worst}`);
});

test('Core v2 GPU gravity coefficients are exactly the f32 finite-volume rho-prime force coefficients', () => {
  const horizontal = buildCubedSphere(2);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 80,
    EARTH.radius + 500,
    EARTH.radius + 4000,
  ]));
  const reference = buildIsothermalHydrostaticReference(geometry, 288, 100000);
  const stencil = buildLinearReconstructionStencil(geometry);
  const faces = buildCoreV2GpuInternalFaceStaticData(geometry, stencil, reference);
  const gather = buildCoreV2GpuRateStaticData(geometry, stencil, reference, faces);

  let worst = 0;
  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const r0 = geometry.radiusInterface[k]!;
      const r1 = geometry.radiusInterface[k + 1]!;
      const scale = -EARTH.gravity * (r1 ** 3 - r0 ** 3) / 3;
      for (let d = 0; d < 3; d++) {
        const expected = Math.fround(scale * geometry.cellVectorAreaUnit[c * 3 + d]!);
        worst = Math.max(worst,
          Math.abs(gather.gravityCoefficient[q * 4 + d]! - expected));
      }
    }
  }
  assert(worst === 0, `GPU gravity coefficient packing mismatch=${worst}`);
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
console.log(`${passed}/${tests.length} Core v2 GPU-operator-static tests passed`);
