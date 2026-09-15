declare const process: { exitCode?: number };

import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildIsothermalHydrostaticReference } from '../corev2/hydrostaticReference.js';
import {
  MAX_RECONSTRUCTION_NEIGHBORS,
  buildLinearReconstructionStencil,
  radialFaceCentroid,
  sideFaceCentroid,
} from '../corev2/reconstruction.js';
import { buildSphericalShellGeometry } from '../corev2/sphericalShellGeometry.js';
import { buildCoreV2GpuReconstructionStaticData } from '../gpu/coreV2ReconstructionGpu.js';
import { assert } from './assert.js';

interface Test { name: string; fn: () => void }
const tests: Test[] = [];
const test = (name: string, fn: () => void): void => { tests.push({ name, fn }); };

test('Core v2 GPU reconstruction static layout is the f32 image of the CPU stencil', () => {
  const horizontal = buildCubedSphere(4);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 90,
    EARTH.radius + 500,
    EARTH.radius + 2400,
    EARTH.radius + 9000,
  ]));
  const stencil = buildLinearReconstructionStencil(geometry);
  const reference = buildIsothermalHydrostaticReference(geometry, 284, 100200);
  const gpu = buildCoreV2GpuReconstructionStaticData(geometry, stencil, reference);
  const cellCount = horizontal.cellCount * geometry.nz;

  assert(gpu.cellCount === cellCount, 'GPU reconstruction cell count');
  assert(gpu.neighborMeta.length === cellCount * 8, 'GPU neighbor meta length');
  assert(gpu.neighborCoeff.length === cellCount * 6 * 4, 'GPU neighbor coeff length');
  assert(gpu.faceDisplacement.length === cellCount * 6 * 4, 'GPU face displacement length');

  let worstCoeff = 0;
  let worstDx = 0;
  for (let q = 0; q < cellCount; q++) {
    const count = stencil.neighborCount[q]!;
    assert(gpu.neighborMeta[q * 8 + 6] === count, `GPU neighbor count q=${q}`);
    for (let j = 0; j < MAX_RECONSTRUCTION_NEIGHBORS; j++) {
      const source = q * MAX_RECONSTRUCTION_NEIGHBORS + j;
      const expectedNeighbor = j < count ? stencil.neighborCell[source]! : q;
      assert(gpu.neighborMeta[q * 8 + j] === expectedNeighbor,
        `GPU neighbor id q=${q} j=${j}`);
      const base = (q * 6 + j) * 4;
      const expected = [
        Math.fround(stencil.coeffX[source] ?? 0),
        Math.fround(stencil.coeffY[source] ?? 0),
        Math.fround(stencil.coeffZ[source] ?? 0),
      ];
      for (let d = 0; d < 3; d++) {
        worstCoeff = Math.max(worstCoeff, Math.abs(gpu.neighborCoeff[base + d]! - expected[d]!));
      }
    }

    const c = Math.floor(q / geometry.nz);
    const k = q % geometry.nz;
    const x0 = [
      stencil.cellCentroid[q * 3]!,
      stencil.cellCentroid[q * 3 + 1]!,
      stencil.cellCentroid[q * 3 + 2]!,
    ];
    const positions = [] as Array<readonly [number, number, number]>;
    for (let s = 0; s < 4; s++) {
      positions.push(sideFaceCentroid(geometry, horizontal.cellEdges[c * 4 + s]!, k));
    }
    positions.push(radialFaceCentroid(geometry, c, k));
    positions.push(radialFaceCentroid(geometry, c, k + 1));
    for (let s = 0; s < 6; s++) {
      const position = positions[s]!;
      const base = (q * 6 + s) * 4;
      for (let d = 0; d < 3; d++) {
        const expected = Math.fround(position[d]! - x0[d]!);
        worstDx = Math.max(worstDx, Math.abs(gpu.faceDisplacement[base + d]! - expected));
      }
    }
  }
  assert(worstCoeff === 0, `GPU coefficient packing mismatch=${worstCoeff}`);
  assert(worstDx === 0, `GPU face displacement packing mismatch=${worstDx}`);
});

test('Core v2 GPU hydrostatic reference packing preserves exact shared layer values in f32', () => {
  const horizontal = buildCubedSphere(2);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 200,
    EARTH.radius + 1000,
    EARTH.radius + 5000,
  ]));
  const stencil = buildLinearReconstructionStencil(geometry);
  const reference = buildIsothermalHydrostaticReference(geometry, 288, 100000);
  const gpu = buildCoreV2GpuReconstructionStaticData(geometry, stencil, reference);
  for (let k = 0; k < geometry.nz; k++) {
    assert(gpu.referenceLayer[k * 4] === Math.fround(reference.cellDensity[k]!), `rho_ref k=${k}`);
    assert(gpu.referenceLayer[k * 4 + 1] === Math.fround(reference.cellPressure[k]!), `p_ref k=${k}`);
    assert(gpu.referenceLayer[k * 4 + 2] === Math.fround(reference.cellGeopotential[k]!), `phi_ref k=${k}`);
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
console.log(`${passed}/${tests.length} Core v2 GPU-static tests passed`);
