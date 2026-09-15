import { DRY_AIR, EARTH } from './core/constants.js';
import { buildCubedSphere } from './grid/cubedSphere.js';
import { createConservativeFields } from './corev2/fields.js';
import { closedShellVerticalStiffRateFirstOrder } from './corev2/heviSplit.js';
import { buildIsothermalHydrostaticReference, type HydrostaticReference1D } from './corev2/hydrostaticReference.js';
import { conservedFromPrimitive } from './corev2/state.js';
import { buildSphericalShellGeometry, shellCellIndex } from './corev2/sphericalShellGeometry.js';
import { buildCoreV2GpuHeviVerticalStaticData, CoreV2GpuHeviVertical } from './gpu/coreV2HeviVerticalGpu.js';
import { coreV2ReferenceTotalEnergyF32 } from './gpu/coreV2ReferenceF32.js';

const $ = (id: string): HTMLElement => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing #${id}`);
  return node;
};
const f32 = Math.fround;

function f32Reference(r: HydrostaticReference1D): HydrostaticReference1D {
  const cast = (a: Float64Array): Float64Array => Float64Array.from(a, f32);
  return {
    temperature: f32(r.temperature),
    surfacePressure: f32(r.surfacePressure),
    cellDensity: cast(r.cellDensity),
    cellPressure: cast(r.cellPressure),
    radialFacePressure: cast(r.radialFacePressure),
    sideFacePressure: cast(r.sideFacePressure),
    cellGeopotential: cast(r.cellGeopotential),
  };
}

function pack(fields: ReturnType<typeof createConservativeFields>): Float32Array {
  const out = new Float32Array(fields.rho.length * 8);
  for (let q = 0; q < fields.rho.length; q++) {
    const i = q * 8;
    out[i] = fields.rho[q]!;
    out[i + 1] = fields.momX[q]!;
    out[i + 2] = fields.momY[q]!;
    out[i + 3] = fields.momZ[q]!;
    out[i + 4] = fields.rhoE[q]!;
  }
  return out;
}

function unpack(a: Float32Array) {
  const fields = createConservativeFields(a.length / 8);
  for (let q = 0; q < fields.rho.length; q++) {
    const i = q * 8;
    fields.rho[q] = a[i]!;
    fields.momX[q] = a[i + 1]!;
    fields.momY[q] = a[i + 2]!;
    fields.momZ[q] = a[i + 3]!;
    fields.rhoE[q] = a[i + 4]!;
  }
  return fields;
}

function relative(actual: ArrayLike<number>, expected: ArrayLike<number>): number {
  let numerator = 0;
  let denominator = 0;
  for (let i = 0; i < actual.length; i++) {
    const d = actual[i]! - expected[i]!;
    numerator += d * d;
    denominator += expected[i]! * expected[i]!;
  }
  return Math.sqrt(numerator / Math.max(denominator, 1e-30));
}

async function gpuRate(
  device: any,
  packed: Float32Array,
  geometry: ReturnType<typeof buildSphericalShellGeometry>,
  reference: HydrostaticReference1D,
): Promise<Float32Array> {
  const gpu = new CoreV2GpuHeviVertical(
    device,
    buildCoreV2GpuHeviVerticalStaticData(geometry, reference),
  );
  try {
    return await gpu.computeIntegratedRate(packed);
  } finally {
    gpu.destroy();
  }
}

async function run(): Promise<void> {
  $('status').textContent = 'RUNNING';
  const webgpu = (navigator as Navigator & { gpu?: any }).gpu;
  if (!webgpu) throw new Error('navigator.gpu unavailable');
  const adapter = await webgpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  const device = await adapter.requestDevice();

  const horizontal = buildCubedSphere(3);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 120,
    EARTH.radius + 420,
    EARTH.radius + 1300,
    EARTH.radius + 3900,
    EARTH.radius + 10000,
  ]));
  const reference = f32Reference(buildIsothermalHydrostaticReference(geometry, 284, 100100));
  const fields = createConservativeFields(horizontal.cellCount * geometry.nz);

  for (let c = 0; c < horizontal.cellCount; c++) {
    const x = horizontal.cellCenters[c * 3]!;
    const y = horizontal.cellCenters[c * 3 + 1]!;
    const z = horizontal.cellCenters[c * 3 + 2]!;
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const vertical = Math.sin(Math.PI * (k + 0.5) / geometry.nz);
      const state = conservedFromPrimitive({
        rho: reference.cellDensity[k]! * (1 + 0.0025 * vertical * (0.7 * x - 0.3 * y)),
        pressure: reference.cellPressure[k]! * (1 + 0.004 * vertical * (0.5 * y + 0.2 * z)),
        velocity: [
          7 * y * vertical,
          -5 * x * vertical,
          2.2 * z * vertical + 0.4 * (k - 2),
        ],
      }, reference.cellGeopotential[k]!);
      fields.rho[q] = f32(state.rho);
      fields.momX[q] = f32(state.momentum[0]);
      fields.momY[q] = f32(state.momentum[1]);
      fields.momZ[q] = f32(state.momentum[2]);
      fields.rhoE[q] = f32(state.rhoE);
    }
  }

  const packed = pack(fields);
  const cpu = closedShellVerticalStiffRateFirstOrder(unpack(packed), geometry, reference);
  const gpu = await gpuRate(device, packed, geometry, reference);
  const n = fields.rho.length;
  const massGpu = new Float64Array(n);
  const massCpu = new Float64Array(n);
  const momGpu = new Float64Array(n * 3);
  const momCpu = new Float64Array(n * 3);
  const energyGpu = new Float64Array(n);
  const energyCpu = new Float64Array(n);
  for (let q = 0; q < n; q++) {
    massGpu[q] = gpu[q * 8]!;
    massCpu[q] = cpu.rho[q]!;
    momGpu[q * 3] = gpu[q * 8 + 1]!;
    momGpu[q * 3 + 1] = gpu[q * 8 + 2]!;
    momGpu[q * 3 + 2] = gpu[q * 8 + 3]!;
    momCpu[q * 3] = cpu.momX[q]!;
    momCpu[q * 3 + 1] = cpu.momY[q]!;
    momCpu[q * 3 + 2] = cpu.momZ[q]!;
    energyGpu[q] = gpu[q * 8 + 4]!;
    energyCpu[q] = cpu.rhoE[q]!;
  }

  const massErr = relative(massGpu, massCpu);
  const momErr = relative(momGpu, momCpu);
  const energyErr = relative(energyGpu, energyCpu);
  $('massErr').textContent = massErr.toExponential(6);
  $('momErr').textContent = momErr.toExponential(6);
  $('energyErr').textContent = energyErr.toExponential(6);

  const hydro = new Float32Array(n * 8);
  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const i = q * 8;
      const rho = f32(reference.cellDensity[k]!);
      const p = f32(reference.cellPressure[k]!);
      const phi = f32(reference.cellGeopotential[k]!);
      hydro[i] = rho;
      hydro[i + 4] = coreV2ReferenceTotalEnergyF32(rho, p, phi);
    }
  }
  const hydroRate = await gpuRate(device, hydro, geometry, reference);
  let hydroMax = 0;
  for (const value of hydroRate) hydroMax = Math.max(hydroMax, Math.abs(value));
  $('hydroMax').textContent = hydroMax.toExponential(6);

  const soundSpeed = Math.sqrt(DRY_AIR.gamma * DRY_AIR.rd * reference.temperature);
  const nominalAcousticCfl = soundSpeed * 20 / 120;
  $('cfl').textContent = nominalAcousticCfl.toFixed(2);
  $('adapter').textContent = JSON.stringify(adapter.info ?? {}, null, 2);

  const pass = massErr < 2e-3 && momErr < 2e-3 && energyErr < 2e-3 && hydroMax === 0;
  $('status').textContent = pass ? 'GATE PASS' : 'GATE FAIL';
  $('status').className = pass ? 'ok' : 'bad';
  if (!pass) {
    throw new Error(`Core v2 GPU vertical HEVI operator gate: mass=${massErr} mom=${momErr} energy=${energyErr} hydro=${hydroMax}`);
  }
}

run().catch((error) => {
  console.error(error);
  $('status').textContent = 'GATE FAIL';
  $('status').className = 'bad';
  $('log').textContent = error instanceof Error ? (error.stack ?? error.message) : String(error);
});
