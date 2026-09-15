import { DRY_AIR, EARTH } from './core/constants.js';
import { buildCubedSphere } from './grid/cubedSphere.js';
import { createConservativeFields } from './corev2/fields.js';
import { buildIsothermalHydrostaticReference, type HydrostaticReference1D } from './corev2/hydrostaticReference.js';
import { buildLinearReconstructionStencil } from './corev2/reconstruction.js';
import { rotatingHeldSuarezStep } from './corev2/rotatingDryCore.js';
import { buildStage4HeldSuarezInitialState } from './corev2/stage4Climate.js';
import { pressureFromConservedRelativeToReference } from './corev2/state.js';
import { buildSphericalShellGeometry, type SphericalShellGeometry } from './corev2/sphericalShellGeometry.js';
import { CoreV2GpuRotatingDryCore } from './gpu/coreV2RotatingDryGpu.js';

const PACKED = 8;
const f32 = Math.fround;
const $ = (id: string): HTMLElement => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing #${id}`);
  return node;
};

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

function pack(fields: ReturnType<typeof createConservativeFields>): Float32Array {
  const out = new Float32Array(fields.rho.length * PACKED);
  for (let q = 0; q < fields.rho.length; q++) {
    const i = q * PACKED;
    out[i] = f32(fields.rho[q]!);
    out[i + 1] = f32(fields.momX[q]!);
    out[i + 2] = f32(fields.momY[q]!);
    out[i + 3] = f32(fields.momZ[q]!);
    out[i + 4] = f32(fields.rhoE[q]!);
  }
  return out;
}

function unpack(state: Float32Array) {
  const fields = createConservativeFields(state.length / PACKED);
  for (let q = 0; q < fields.rho.length; q++) {
    const i = q * PACKED;
    fields.rho[q] = state[i]!;
    fields.momX[q] = state[i + 1]!;
    fields.momY[q] = state[i + 2]!;
    fields.momZ[q] = state[i + 3]!;
    fields.rhoE[q] = state[i + 4]!;
  }
  return fields;
}

function componentScale(reference: HydrostaticReference1D, k: number, v: number): number {
  const rho = reference.cellDensity[k]!;
  const pressure = reference.cellPressure[k]!;
  if (v === 0) return Math.max(rho, 1e-8);
  if (v >= 1 && v <= 3) return Math.max(rho * Math.sqrt(DRY_AIR.gamma * pressure / rho), 1e-6);
  return Math.max(
    pressure / (DRY_AIR.gamma - 1) + Math.abs(rho * reference.cellGeopotential[k]!),
    1,
  );
}

function compare(
  gpu: Float32Array,
  cpu: ReturnType<typeof createConservativeFields>,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
): { relative: number; maxScaled: number; nonFinite: number } {
  let numerator = 0;
  let denominator = 0;
  let maxScaled = 0;
  let nonFinite = 0;
  for (let q = 0; q < cpu.rho.length; q++) {
    const k = q % geometry.nz;
    const expected = [cpu.rho[q]!, cpu.momX[q]!, cpu.momY[q]!, cpu.momZ[q]!, cpu.rhoE[q]!];
    for (let v = 0; v < 5; v++) {
      const actual = gpu[q * PACKED + v]!;
      if (!Number.isFinite(actual)) nonFinite++;
      const delta = actual - expected[v]!;
      numerator += delta * delta;
      denominator += expected[v]! * expected[v]!;
      maxScaled = Math.max(
        maxScaled,
        Math.abs(delta) / Math.max(componentScale(reference, k, v), Math.abs(expected[v]!)),
      );
    }
  }
  return {
    relative: Math.sqrt(numerator / Math.max(denominator, 1e-30)),
    maxScaled,
    nonFinite,
  };
}

function positive(
  state: Float32Array,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
): boolean {
  for (let q = 0; q < state.length / PACKED; q++) {
    const k = q % geometry.nz;
    const i = q * PACKED;
    const rho = state[i]!;
    if (!(rho > 0) || !Number.isFinite(rho)) return false;
    const pressure = pressureFromConservedRelativeToReference({
      rho,
      momentum: [state[i + 1]!, state[i + 2]!, state[i + 3]!],
      rhoE: state[i + 4]!,
    }, reference.cellDensity[k]!, reference.cellPressure[k]!, reference.cellGeopotential[k]!);
    if (!(pressure > 0) || !Number.isFinite(pressure)) return false;
  }
  return true;
}

async function run(): Promise<void> {
  $('status').textContent = 'RUNNING';
  const webgpu = (navigator as Navigator & { gpu?: any }).gpu;
  if (!webgpu) throw new Error('navigator.gpu unavailable');
  const adapter = await webgpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  const device = await adapter.requestDevice();

  const horizontal = buildCubedSphere(2);
  const radii = new Float64Array([
    EARTH.radius,
    EARTH.radius + 500,
    EARTH.radius + 1200,
    EARTH.radius + 2200,
    EARTH.radius + 3600,
    EARTH.radius + 5500,
    EARTH.radius + 8000,
    EARTH.radius + 11000,
    EARTH.radius + 15000,
  ]);
  const geometry = buildSphericalShellGeometry(horizontal, radii);
  const reference = f32Reference(buildIsothermalHydrostaticReference(geometry, 288, 100000));
  const stencil = buildLinearReconstructionStencil(geometry);
  const initialFields = buildStage4HeldSuarezInitialState(geometry, reference, 0.05);
  const initial = pack(initialFields);
  const dt = 20;

  const cpu = rotatingHeldSuarezStep(
    unpack(initial),
    dt,
    geometry,
    stencil,
    reference,
    EARTH,
  ).fields;
  const gpuCore = new CoreV2GpuRotatingDryCore(device, geometry, stencil, reference, EARTH);
  const t0 = performance.now();
  const gpu = await gpuCore.step(initial, dt);
  const elapsedMs = performance.now() - t0;
  const error = compare(gpu.packedState, cpu, geometry, reference);
  const isPositive = positive(gpu.packedState, geometry, reference);
  gpuCore.destroy();

  const passed =
    error.nonFinite === 0 &&
    error.relative < 8e-6 &&
    error.maxScaled < 3e-5 &&
    isPositive;

  const result = {
    relative: error.relative,
    maxScaled: error.maxScaled,
    nonFinite: error.nonFinite,
    positive: isPositive,
    stage1Iterations: gpu.diagnostics.stage1Iterations,
    stage2Iterations: gpu.diagnostics.stage2Iterations,
    elapsedMs,
  };
  $('details').textContent = JSON.stringify(result, null, 2);
  $('status').textContent = passed ? 'GATE PASS' : 'GATE FAIL';
  if (!passed) throw new Error(`Stage 4 GPU gate failed: ${JSON.stringify(result)}`);
}

run().catch(error => {
  console.error(error);
  $('details').textContent = String(error instanceof Error ? error.stack ?? error.message : error);
  $('status').textContent = 'GATE FAIL';
});
