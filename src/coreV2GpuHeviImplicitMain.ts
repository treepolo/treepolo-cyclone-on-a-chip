import { DRY_AIR, EARTH } from './core/constants.js';
import { buildCubedSphere } from './grid/cubedSphere.js';
import {
  CONSERVED_COMPONENTS,
  columnStateIndex,
} from './corev2/heviColumn.js';
import {
  solveVerticalImplicitColumnStage,
  verticalImplicitStageResidualNorm,
} from './corev2/heviImplicitColumn.js';
import {
  buildIsothermalHydrostaticReference,
  type HydrostaticReference1D,
} from './corev2/hydrostaticReference.js';
import {
  conservedFromPrimitive,
  pressureFromConservedRelativeToReference,
} from './corev2/state.js';
import {
  buildSphericalShellGeometry,
  shellCellIndex,
  type SphericalShellGeometry,
} from './corev2/sphericalShellGeometry.js';
import {
  CORE_V2_GPU_HEVI_NEWTON_TOLERANCE,
  CoreV2GpuHeviImplicitStage,
} from './gpu/coreV2HeviImplicitGpu.js';

const BLOCK = 5;
const PACKED = 8;
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

function componentScale(reference: HydrostaticReference1D, k: number, component: number): number {
  const rho = reference.cellDensity[k]!;
  const pressure = reference.cellPressure[k]!;
  if (component === 0) return Math.max(rho, 1e-8);
  if (component >= 1 && component <= 3) {
    const sound = Math.sqrt(DRY_AIR.gamma * pressure / rho);
    return Math.max(rho * sound, 1e-6);
  }
  return Math.max(
    pressure / (DRY_AIR.gamma - 1) + Math.abs(rho * reference.cellGeopotential[k]!),
    1,
  );
}

function buildPackedBase(
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
): Float32Array {
  const cellCount = geometry.horizontal.cellCount * geometry.nz;
  const packed = new Float32Array(cellCount * PACKED);
  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    const x = geometry.horizontal.cellCenters[c * 3]!;
    const y = geometry.horizontal.cellCenters[c * 3 + 1]!;
    const z = geometry.horizontal.cellCenters[c * 3 + 2]!;
    const pulse = 2 + ((c * 5 + 1) % Math.max(3, geometry.nz - 3));
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const shape = Math.exp(-0.5 * ((k - pulse) / 1.1) ** 2);
      const densityFactor = 1 + 0.0012 * shape * (0.7 + 0.3 * x);
      const pressureFactor = 1 + 0.012 * shape * (0.65 + 0.2 * y - 0.1 * z);
      const state = conservedFromPrimitive({
        rho: reference.cellDensity[k]! * densityFactor,
        pressure: reference.cellPressure[k]! * pressureFactor,
        velocity: [
          9 + 2.5 * y + 1.5 * shape,
          -6 + 1.7 * x - 0.8 * shape,
          3.5 + 0.9 * z + 2.2 * shape,
        ],
      }, reference.cellGeopotential[k]!);
      const i = q * PACKED;
      packed[i] = f32(state.rho);
      packed[i + 1] = f32(state.momentum[0]);
      packed[i + 2] = f32(state.momentum[1]);
      packed[i + 3] = f32(state.momentum[2]);
      packed[i + 4] = f32(state.rhoE);
    }
  }
  return packed;
}

function extractColumn(packed: Float32Array, c: number, nz: number): Float64Array {
  const out = new Float64Array(nz * CONSERVED_COMPONENTS);
  for (let k = 0; k < nz; k++) {
    const q = c * nz + k;
    for (let v = 0; v < BLOCK; v++) {
      out[columnStateIndex(k, v)] = packed[q * PACKED + v]!;
    }
  }
  return out;
}

function insertColumn(packed: Float64Array, c: number, nz: number, column: Float64Array): void {
  for (let k = 0; k < nz; k++) {
    const q = c * nz + k;
    for (let v = 0; v < BLOCK; v++) {
      packed[q * BLOCK + v] = column[columnStateIndex(k, v)]!;
    }
  }
}

function compareStage(
  gpu: Float32Array,
  cpu: Float64Array,
  base: Float32Array,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
): { stateRelativeL2: number; incrementRelativeL2: number; maxScaled: number; nonFinite: number } {
  let stateNum = 0;
  let stateDen = 0;
  let updateNum = 0;
  let updateDen = 0;
  let maxScaled = 0;
  let nonFinite = 0;
  for (let q = 0; q < geometry.horizontal.cellCount * geometry.nz; q++) {
    const k = q % geometry.nz;
    for (let v = 0; v < BLOCK; v++) {
      const gi = q * PACKED + v;
      const ci = q * BLOCK + v;
      const g = gpu[gi]!;
      const e = cpu[ci]!;
      const b = base[gi]!;
      if (!Number.isFinite(g)) nonFinite++;
      const d = g - e;
      stateNum += d * d;
      stateDen += e * e;
      const du = (g - b) - (e - b);
      updateNum += du * du;
      updateDen += (e - b) ** 2;
      const scale = Math.max(componentScale(reference, k, v), Math.abs(e), Math.abs(b));
      maxScaled = Math.max(maxScaled, Math.abs(d) / Math.max(scale, 1e-30));
    }
  }
  return {
    stateRelativeL2: Math.sqrt(stateNum / Math.max(stateDen, 1e-30)),
    incrementRelativeL2: Math.sqrt(updateNum / Math.max(updateDen, 1e-30)),
    maxScaled,
    nonFinite,
  };
}

function maxCpuResidual(
  gpu: Float32Array,
  base: Float32Array,
  alphaDt: number,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
): number {
  let maxResidual = 0;
  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    const stateColumn = extractColumn(gpu, c, geometry.nz);
    const baseColumn = extractColumn(base, c, geometry.nz);
    maxResidual = Math.max(
      maxResidual,
      verticalImplicitStageResidualNorm(
        stateColumn,
        baseColumn,
        alphaDt,
        c,
        geometry,
        reference,
      ),
    );
  }
  return maxResidual;
}

function assertPositive(
  gpu: Float32Array,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
): boolean {
  for (let q = 0; q < geometry.horizontal.cellCount * geometry.nz; q++) {
    const k = q % geometry.nz;
    const i = q * PACKED;
    const rho = gpu[i]!;
    if (!(rho > 0) || !Number.isFinite(rho)) return false;
    const pressure = pressureFromConservedRelativeToReference({
      rho,
      momentum: [gpu[i + 1]!, gpu[i + 2]!, gpu[i + 3]!],
      rhoE: gpu[i + 4]!,
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
  const radii = [EARTH.radius];
  for (let k = 1; k <= 8; k++) radii.push(EARTH.radius + 120 * k);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array(radii));
  const reference = f32Reference(buildIsothermalHydrostaticReference(geometry, 288, 100000));
  const base = buildPackedBase(geometry, reference);
  const alphaDt = 20;
  const soundSpeed = Math.sqrt(DRY_AIR.gamma * DRY_AIR.rd * 288);
  const acousticCfl = soundSpeed * alphaDt / 120;

  const cpuExpected = new Float64Array(horizontal.cellCount * geometry.nz * BLOCK);
  let cpuMaxIterations = 0;
  for (let c = 0; c < horizontal.cellCount; c++) {
    const baseColumn = extractColumn(base, c, geometry.nz);
    const solved = solveVerticalImplicitColumnStage(baseColumn, alphaDt, c, geometry, reference);
    cpuMaxIterations = Math.max(cpuMaxIterations, solved.diagnostics.iterations);
    insertColumn(cpuExpected, c, geometry.nz, solved.state);
  }

  const stage = new CoreV2GpuHeviImplicitStage(device, geometry, reference);
  try {
    const gpuResult = await stage.solve(base, alphaDt);
    const comparison = compareStage(gpuResult.packedState, cpuExpected, base, geometry, reference);
    const cpuResidual = maxCpuResidual(gpuResult.packedState, base, alphaDt, geometry, reference);
    const positive = assertPositive(gpuResult.packedState, geometry, reference);

    $('stateRelative').textContent = comparison.stateRelativeL2.toExponential(6);
    $('incrementRelative').textContent = comparison.incrementRelativeL2.toExponential(6);
    $('maxScaled').textContent = comparison.maxScaled.toExponential(6);
    $('nonFinite').textContent = String(comparison.nonFinite);
    $('gpuResidual').textContent = gpuResult.diagnostics.finalResidual.toExponential(6);
    $('cpuResidual').textContent = cpuResidual.toExponential(6);
    $('gpuIterations').textContent = String(gpuResult.diagnostics.iterations);
    $('cpuIterations').textContent = String(cpuMaxIterations);
    $('halvings').textContent = String(gpuResult.diagnostics.totalLineSearchHalvings);
    $('cfl').textContent = acousticCfl.toFixed(3);
    $('positive').textContent = String(positive);
    $('adapter').textContent = JSON.stringify(adapter.info ?? {}, null, 2);

    const pass =
      acousticCfl > 50 &&
      comparison.nonFinite === 0 &&
      positive &&
      gpuResult.diagnostics.finalResidual <= CORE_V2_GPU_HEVI_NEWTON_TOLERANCE &&
      cpuResidual < 2e-4 &&
      comparison.stateRelativeL2 < 2e-4 &&
      comparison.incrementRelativeL2 < 8e-2 &&
      comparison.maxScaled < 2e-3;
    $('status').textContent = pass ? 'GATE PASS' : 'GATE FAIL';
    $('status').className = pass ? 'ok' : 'bad';
    const summary = [
      `acousticCfl=${acousticCfl}`,
      `stateRel=${comparison.stateRelativeL2}`,
      `incrementRel=${comparison.incrementRelativeL2}`,
      `maxScaled=${comparison.maxScaled}`,
      `gpuResidual=${gpuResult.diagnostics.finalResidual}`,
      `cpuResidual=${cpuResidual}`,
      `gpuIterations=${gpuResult.diagnostics.iterations}`,
      `cpuMaxIterations=${cpuMaxIterations}`,
      `halvings=${gpuResult.diagnostics.totalLineSearchHalvings}`,
      `positive=${positive}`,
    ].join(' ');
    console.log(`CORE_V2_GPU_HEVI_IMPLICIT ${summary}`);
    $('log').textContent = summary;
    if (!pass) throw new Error(`Core v2 GPU implicit HEVI gate failed: ${summary}`);
  } finally {
    stage.destroy();
  }
}

run().catch((error) => {
  console.error(error);
  $('status').textContent = 'GATE FAIL';
  $('status').className = 'bad';
  $('log').textContent = error instanceof Error ? (error.stack ?? error.message) : String(error);
});
