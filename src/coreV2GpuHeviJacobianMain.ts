import { DRY_AIR, EARTH } from './core/constants.js';
import { buildCubedSphere } from './grid/cubedSphere.js';
import { createConservativeFields } from './corev2/fields.js';
import { buildIsothermalHydrostaticReference, type HydrostaticReference1D } from './corev2/hydrostaticReference.js';
import { conservedFromPrimitive } from './corev2/state.js';
import { buildSphericalShellGeometry, shellCellIndex } from './corev2/sphericalShellGeometry.js';
import {
  buildCoreV2GpuHeviJacobianStaticData,
  CORE_V2_GPU_FD_SQRT_EPSILON,
  CORE_V2_GPU_HEVI_JACOBIAN_FLOATS_PER_CELL,
  CoreV2GpuHeviJacobian,
} from './gpu/coreV2HeviJacobianGpu.js';
import {
  buildCoreV2GpuHeviVerticalStaticData,
  CoreV2GpuHeviVertical,
} from './gpu/coreV2HeviVerticalGpu.js';

const $ = (id: string): HTMLElement => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing #${id}`);
  return node;
};
const f32 = Math.fround;

type Geometry = ReturnType<typeof buildSphericalShellGeometry>;

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

function packedComponent(packed: Float32Array, q: number, v: number): number {
  return packed[q * 8 + v]!;
}

function referenceComponentScale(
  reference: HydrostaticReference1D,
  k: number,
  component: number,
): number {
  const rho = f32(reference.cellDensity[k]!);
  const pressure = f32(reference.cellPressure[k]!);
  if (component === 0) return f32(Math.max(rho, 1e-8));
  if (component >= 1 && component <= 3) {
    const sound = f32(Math.sqrt(f32(f32(DRY_AIR.gamma) * f32(pressure / rho))));
    return f32(Math.max(f32(rho * sound), 1e-6));
  }
  const gm1 = f32(f32(DRY_AIR.gamma) - f32(1));
  const internal = f32(pressure / gm1);
  const potential = f32(Math.abs(f32(rho * f32(reference.cellGeopotential[k]!))));
  return f32(Math.max(f32(internal + potential), 1));
}

function perturbationStep(
  packed: Float32Array,
  q: number,
  v: number,
  reference: HydrostaticReference1D,
  nz: number,
): number {
  const k = q % nz;
  const value = packedComponent(packed, q, v);
  const scale = f32(Math.max(Math.abs(value), referenceComponentScale(reference, k, v)));
  return f32(f32(CORE_V2_GPU_FD_SQRT_EPSILON) * scale);
}

function rateComponent(rate: Float32Array, q: number, v: number): number {
  return rate[q * 8 + v]!;
}

function expectedSlot(rowQ: number, block: number, rowVar: number, sourceVar: number): number {
  return rowQ * CORE_V2_GPU_HEVI_JACOBIAN_FLOATS_PER_CELL + block * 25 + rowVar * 5 + sourceVar;
}

async function buildExpectedFromGpuVerticalOperator(
  vertical: CoreV2GpuHeviVertical,
  packed: Float32Array,
  geometry: Geometry,
  reference: HydrostaticReference1D,
): Promise<Float32Array> {
  const cellCount = geometry.horizontal.cellCount * geometry.nz;
  const expected = new Float32Array(cellCount * CORE_V2_GPU_HEVI_JACOBIAN_FLOATS_PER_CELL);
  const base = await vertical.computeIntegratedRate(packed);
  const invVolume = new Float32Array(cellCount);
  for (let q = 0; q < cellCount; q++) invVolume[q] = 1 / geometry.cellVolume[q]!;

  for (let sourceQ = 0; sourceQ < cellCount; sourceQ++) {
    const sourceK = sourceQ % geometry.nz;
    const column = Math.floor(sourceQ / geometry.nz);
    for (let sourceVar = 0; sourceVar < 5; sourceVar++) {
      const h = perturbationStep(packed, sourceQ, sourceVar, reference, geometry.nz);
      const perturbed = packed.slice();
      perturbed[sourceQ * 8 + sourceVar] = f32(perturbed[sourceQ * 8 + sourceVar]! + h);
      const rate = await vertical.computeIntegratedRate(perturbed);
      const rowBegin = Math.max(0, sourceK - 1);
      const rowEnd = Math.min(geometry.nz - 1, sourceK + 1);
      for (let rowK = rowBegin; rowK <= rowEnd; rowK++) {
        const rowQ = column * geometry.nz + rowK;
        const block = rowK > sourceK ? 0 : rowK < sourceK ? 2 : 1;
        for (let rowVar = 0; rowVar < 5; rowVar++) {
          const baseTendency = f32(rateComponent(base, rowQ, rowVar) * invVolume[rowQ]!);
          const perturbedTendency = f32(rateComponent(rate, rowQ, rowVar) * invVolume[rowQ]!);
          expected[expectedSlot(rowQ, block, rowVar, sourceVar)] =
            f32(f32(perturbedTendency - baseTendency) / h);
        }
      }
    }
  }
  return expected;
}

function compare(actual: Float32Array, expected: Float32Array): {
  relativeL2: number;
  maxScaled: number;
  nonFinite: number;
} {
  let numerator = 0;
  let denominator = 0;
  let maxScaled = 0;
  let nonFinite = 0;
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i]!;
    const e = expected[i]!;
    if (!Number.isFinite(a)) nonFinite++;
    const d = a - e;
    numerator += d * d;
    denominator += e * e;
    maxScaled = Math.max(maxScaled, Math.abs(d) / Math.max(1, Math.abs(e)));
  }
  return {
    relativeL2: Math.sqrt(numerator / Math.max(denominator, 1e-30)),
    maxScaled,
    nonFinite,
  };
}

async function run(): Promise<void> {
  $('status').textContent = 'RUNNING';
  const webgpu = (navigator as Navigator & { gpu?: any }).gpu;
  if (!webgpu) throw new Error('navigator.gpu unavailable');
  const adapter = await webgpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  const device = await adapter.requestDevice();

  const horizontal = buildCubedSphere(2);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array([
    EARTH.radius,
    EARTH.radius + 180,
    EARTH.radius + 720,
    EARTH.radius + 2400,
  ]));
  const reference = f32Reference(buildIsothermalHydrostaticReference(geometry, 286, 100300));
  const fields = createConservativeFields(horizontal.cellCount * geometry.nz);
  for (let c = 0; c < horizontal.cellCount; c++) {
    const x = horizontal.cellCenters[c * 3]!;
    const y = horizontal.cellCenters[c * 3 + 1]!;
    const z = horizontal.cellCenters[c * 3 + 2]!;
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const vertical = (k + 0.65) / geometry.nz;
      const state = conservedFromPrimitive({
        rho: reference.cellDensity[k]! * (1 + 0.0018 * (0.6 * x - 0.3 * y + 0.2 * vertical)),
        pressure: reference.cellPressure[k]! * (1 + 0.0027 * (0.4 * y + 0.25 * z - 0.15 * vertical)),
        velocity: [
          3.2 + 0.8 * y,
          -2.4 + 0.6 * x,
          1.1 + 0.35 * z + 0.12 * k,
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
  const vertical = new CoreV2GpuHeviVertical(
    device,
    buildCoreV2GpuHeviVerticalStaticData(geometry, reference),
  );
  const jacobian = new CoreV2GpuHeviJacobian(
    device,
    buildCoreV2GpuHeviJacobianStaticData(geometry, reference),
  );
  try {
    const expected = await buildExpectedFromGpuVerticalOperator(
      vertical,
      packed,
      geometry,
      reference,
    );
    const actual = await jacobian.computeTendencyJacobian(packed);
    const result = compare(actual, expected);
    $('relative').textContent = result.relativeL2.toExponential(6);
    $('maxScaled').textContent = result.maxScaled.toExponential(6);
    $('nonFinite').textContent = String(result.nonFinite);
    $('fdScale').textContent = f32(CORE_V2_GPU_FD_SQRT_EPSILON).toExponential(6);
    $('adapter').textContent = JSON.stringify(adapter.info ?? {}, null, 2);

    const pass = result.nonFinite === 0 && result.relativeL2 < 5e-3 && result.maxScaled < 2e-2;
    $('status').textContent = pass ? 'GATE PASS' : 'GATE FAIL';
    $('status').className = pass ? 'ok' : 'bad';
    if (!pass) {
      throw new Error(
        `Core v2 GPU HEVI Jacobian gate: rel=${result.relativeL2} maxScaled=${result.maxScaled} nonFinite=${result.nonFinite}`,
      );
    }
  } finally {
    vertical.destroy();
    jacobian.destroy();
  }
}

run().catch((error) => {
  console.error(error);
  $('status').textContent = 'GATE FAIL';
  $('status').className = 'bad';
  $('log').textContent = error instanceof Error ? (error.stack ?? error.message) : String(error);
});
