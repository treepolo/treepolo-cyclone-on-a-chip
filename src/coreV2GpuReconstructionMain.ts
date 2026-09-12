import { EARTH } from './core/constants.js';
import { buildCubedSphere } from './grid/cubedSphere.js';
import { createConservativeFields } from './corev2/fields.js';
import { buildIsothermalHydrostaticReference, type HydrostaticReference1D } from './corev2/hydrostaticReference.js';
import { buildLinearReconstructionStencil } from './corev2/reconstruction.js';
import { conservedFromPrimitive } from './corev2/state.js';
import { buildSphericalShellGeometry, shellCellIndex } from './corev2/sphericalShellGeometry.js';
import { buildHydrostaticReconstruction } from './corev2/wellBalancedReconstruction.js';
import {
  CoreV2GpuReconstruction,
  buildCoreV2GpuReconstructionStaticData,
} from './gpu/coreV2ReconstructionGpu.js';

const $ = (id: string): HTMLElement => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing #${id}`);
  return node;
};
const f32 = Math.fround;

function f32Reference(reference: HydrostaticReference1D): HydrostaticReference1D {
  const cast = (a: Float64Array): Float64Array => Float64Array.from(a, f32);
  return {
    temperature: f32(reference.temperature),
    surfacePressure: f32(reference.surfacePressure),
    cellDensity: cast(reference.cellDensity),
    cellPressure: cast(reference.cellPressure),
    radialFacePressure: cast(reference.radialFacePressure),
    sideFacePressure: cast(reference.sideFacePressure),
    cellGeopotential: cast(reference.cellGeopotential),
  };
}

function packFields(fields: ReturnType<typeof createConservativeFields>): Float32Array {
  const out = new Float32Array(fields.rho.length * 8);
  for (let q = 0; q < fields.rho.length; q++) {
    const i = q * 8;
    out[i] = f32(fields.rho[q]!);
    out[i + 1] = f32(fields.momX[q]!);
    out[i + 2] = f32(fields.momY[q]!);
    out[i + 3] = f32(fields.momZ[q]!);
    out[i + 4] = f32(fields.rhoE[q]!);
  }
  return out;
}

function fieldsFromPacked(packed: Float32Array) {
  const fields = createConservativeFields(packed.length / 8);
  for (let q = 0; q < fields.rho.length; q++) {
    const i = q * 8;
    fields.rho[q] = packed[i]!;
    fields.momX[q] = packed[i + 1]!;
    fields.momY[q] = packed[i + 2]!;
    fields.momZ[q] = packed[i + 3]!;
    fields.rhoE[q] = packed[i + 4]!;
  }
  return fields;
}

function relativeGradientError(
  gpu: Float32Array,
  variable: number,
  expected: readonly [Float64Array, Float64Array, Float64Array],
): number {
  let num = 0;
  let den = 0;
  for (let q = 0; q < expected[0].length; q++) {
    const base = q * 20 + variable * 4;
    for (let d = 0; d < 3; d++) {
      const e = expected[d]![q]!;
      const delta = gpu[base + d]! - e;
      num += delta * delta;
      den += e * e;
    }
  }
  return Math.sqrt(num / Math.max(den, 1e-30));
}

async function run(): Promise<void> {
  $('status').textContent = 'RUNNING';
  const webgpu = (navigator as Navigator & { gpu?: any }).gpu;
  if (!webgpu) throw new Error('navigator.gpu unavailable');
  const adapter = await webgpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  const device = await adapter.requestDevice();

  const horizontal = buildCubedSphere(4);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 120,
    EARTH.radius + 600,
    EARTH.radius + 2300,
    EARTH.radius + 8000,
  ]));
  const reference = f32Reference(buildIsothermalHydrostaticReference(geometry, 285, 100300));
  const stencil = buildLinearReconstructionStencil(geometry);
  const fields = createConservativeFields(horizontal.cellCount * geometry.nz);

  for (let c = 0; c < horizontal.cellCount; c++) {
    const x = horizontal.cellCenters[c * 3]!;
    const y = horizontal.cellCenters[c * 3 + 1]!;
    const z = horizontal.cellCenters[c * 3 + 2]!;
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const vertical = (k + 0.5) / geometry.nz;
      const state = conservedFromPrimitive({
        rho: reference.cellDensity[k]! * (1 + 0.018 * x - 0.011 * y + 0.006 * vertical),
        pressure: reference.cellPressure[k]! * (1 + 0.013 * y + 0.008 * z - 0.004 * vertical),
        velocity: [
          35 * y + 3 * vertical,
          -27 * x + 2 * z,
          8 * z - 1.5 * vertical,
        ],
      }, reference.cellGeopotential[k]!);
      fields.rho[q] = state.rho;
      fields.momX[q] = state.momentum[0];
      fields.momY[q] = state.momentum[1];
      fields.momZ[q] = state.momentum[2];
      fields.rhoE[q] = state.rhoE;
    }
  }

  const packed = packFields(fields);
  const roundedFields = fieldsFromPacked(packed);
  const expected = buildHydrostaticReconstruction(roundedFields, stencil, reference);
  const staticData = buildCoreV2GpuReconstructionStaticData(geometry, stencil, reference);
  const runtime = new CoreV2GpuReconstruction(device, staticData);
  const gradients = await runtime.computeGradients(packed);
  runtime.destroy();

  const errors = [
    relativeGradientError(gradients, 0, [expected.primitive.rhoX, expected.primitive.rhoY, expected.primitive.rhoZ]),
    relativeGradientError(gradients, 1, [expected.primitive.uxX, expected.primitive.uxY, expected.primitive.uxZ]),
    relativeGradientError(gradients, 2, [expected.primitive.uyX, expected.primitive.uyY, expected.primitive.uyZ]),
    relativeGradientError(gradients, 3, [expected.primitive.uzX, expected.primitive.uzY, expected.primitive.uzZ]),
    relativeGradientError(gradients, 4, [expected.pressurePerturbX, expected.pressurePerturbY, expected.pressurePerturbZ]),
  ];
  const names = ['rhoErr', 'uxErr', 'uyErr', 'uzErr', 'pErr'];
  for (let i = 0; i < errors.length; i++) $(names[i]!).textContent = errors[i]!.toExponential(6);
  $('adapter').textContent = JSON.stringify(adapter.info ?? {}, null, 2);

  const worst = Math.max(...errors);
  const pass = worst < 2.5e-3;
  $('status').textContent = pass ? 'GATE PASS' : 'GATE FAIL';
  $('status').className = pass ? 'ok' : 'bad';
  if (!pass) throw new Error(`Core v2 GPU reconstruction relative gradient error=${worst}`);
}

run().catch((error) => {
  console.error(error);
  $('status').textContent = 'GATE FAIL';
  $('status').className = 'bad';
  $('log').textContent = error instanceof Error ? `${error.stack ?? error.message}` : String(error);
});
