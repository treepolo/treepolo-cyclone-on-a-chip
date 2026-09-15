import { DRY_AIR, EARTH } from './core/constants.js';
import { buildCubedSphere } from './grid/cubedSphere.js';
import { createConservativeFields } from './corev2/fields.js';
import { heviImexSsp2Step } from './corev2/heviImex.js';
import {
  buildIsothermalHydrostaticReference,
  type HydrostaticReference1D,
} from './corev2/hydrostaticReference.js';
import { buildLinearReconstructionStencil } from './corev2/reconstruction.js';
import {
  conservedFromPrimitive,
  pressureFromConservedRelativeToReference,
} from './corev2/state.js';
import {
  buildSphericalShellGeometry,
  shellCellIndex,
  type SphericalShellGeometry,
} from './corev2/sphericalShellGeometry.js';
import { CoreV2GpuHeviImexSsp2 } from './gpu/coreV2HeviImexGpu.js';
import { coreV2ReferenceTotalEnergyF32 } from './gpu/coreV2ReferenceF32.js';

const PACKED = 8;
const PHYSICAL = 5;
const STEPS = 4;
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

function hydrostaticPacked(
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
): Float32Array {
  const out = new Float32Array(geometry.horizontal.cellCount * geometry.nz * PACKED);
  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const i = q * PACKED;
      const rho = f32(reference.cellDensity[k]!);
      const pressure = f32(reference.cellPressure[k]!);
      const phi = f32(reference.cellGeopotential[k]!);
      out[i] = rho;
      out[i + 4] = coreV2ReferenceTotalEnergyF32(rho, pressure, phi);
    }
  }
  return out;
}

function disturbedPacked(
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
): Float32Array {
  const out = hydrostaticPacked(geometry, reference);
  const middle = Math.floor(geometry.nz / 2);
  for (let c = 0; c < geometry.horizontal.cellCount; c++) {
    const x = geometry.horizontal.cellCenters[c * 3]!;
    const y = geometry.horizontal.cellCenters[c * 3 + 1]!;
    const z = geometry.horizontal.cellCenters[c * 3 + 2]!;
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const vertical = Math.exp(-0.5 * ((k - middle) / 1.25) ** 2);
      const state = conservedFromPrimitive({
        rho: reference.cellDensity[k]! * (1 + 0.001 * vertical * (1 + 0.12 * x) + 2e-4 * y),
        pressure: reference.cellPressure[k]! * (1 + 0.01 * vertical * (1 - 0.08 * y) + 3e-4 * z),
        velocity: [0.7 * y * vertical, -0.55 * x * vertical, 0.22 * z * vertical],
      }, reference.cellGeopotential[k]!);
      const i = q * PACKED;
      out[i] = f32(state.rho);
      out[i + 1] = f32(state.momentum[0]);
      out[i + 2] = f32(state.momentum[1]);
      out[i + 3] = f32(state.momentum[2]);
      out[i + 4] = f32(state.rhoE);
    }
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

type Comparison = {
  stateRelative: number;
  incrementRelative: number;
  scaledRms: number;
  maxScaled: number;
  nonFinite: number;
};

function compare(
  gpu: Float32Array,
  cpuFields: ReturnType<typeof createConservativeFields>,
  initial: Float32Array,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
): Comparison {
  let stateNum = 0;
  let stateDen = 0;
  let incrementNum = 0;
  let incrementDen = 0;
  let scaledSum = 0;
  let scaledCount = 0;
  let maxScaled = 0;
  let nonFinite = 0;
  for (let q = 0; q < geometry.horizontal.cellCount * geometry.nz; q++) {
    const k = q % geometry.nz;
    const cpu = [
      cpuFields.rho[q]!, cpuFields.momX[q]!, cpuFields.momY[q]!, cpuFields.momZ[q]!, cpuFields.rhoE[q]!,
    ];
    for (let v = 0; v < PHYSICAL; v++) {
      const i = q * PACKED + v;
      const g = gpu[i]!;
      const e = cpu[v]!;
      const b = initial[i]!;
      if (!Number.isFinite(g)) nonFinite++;
      const d = g - e;
      stateNum += d * d;
      stateDen += e * e;
      const deltaD = (g - b) - (e - b);
      incrementNum += deltaD * deltaD;
      incrementDen += (e - b) * (e - b);
      const scaled = Math.abs(d) / Math.max(componentScale(reference, k, v), Math.abs(e), Math.abs(b));
      scaledSum += scaled * scaled;
      scaledCount++;
      maxScaled = Math.max(maxScaled, scaled);
    }
  }
  return {
    stateRelative: Math.sqrt(stateNum / Math.max(stateDen, 1e-30)),
    incrementRelative: Math.sqrt(incrementNum / Math.max(incrementDen, 1e-30)),
    scaledRms: Math.sqrt(scaledSum / Math.max(scaledCount, 1)),
    maxScaled,
    nonFinite,
  };
}

function integratedMassEnergy(
  state: Float32Array,
  geometry: SphericalShellGeometry,
): { mass: number; energy: number } {
  let mass = 0;
  let energy = 0;
  for (let q = 0; q < geometry.horizontal.cellCount * geometry.nz; q++) {
    const volume = geometry.cellVolume[q]!;
    mass += state[q * PACKED]! * volume;
    energy += state[q * PACKED + 4]! * volume;
  }
  return { mass, energy };
}

function positive(
  state: Float32Array,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
): boolean {
  for (let q = 0; q < geometry.horizontal.cellCount * geometry.nz; q++) {
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
  const radii = [EARTH.radius];
  for (let k = 1; k <= 12; k++) radii.push(EARTH.radius + 120 * k);
  const geometry = buildSphericalShellGeometry(horizontal, new Float64Array(radii));
  const reference = f32Reference(buildIsothermalHydrostaticReference(geometry, 288, 100000));
  const stencil = buildLinearReconstructionStencil(geometry);
  const dt = 20;
  const acousticCfl = Math.sqrt(DRY_AIR.gamma * DRY_AIR.rd * 288) * dt / 120;

  const gpuStepper = new CoreV2GpuHeviImexSsp2(device, geometry, stencil, reference);
  try {
    const hydro = hydrostaticPacked(geometry, reference);
    const hydroNext = await gpuStepper.step(hydro, dt);
    let hydroMaxScaled = 0;
    for (let q = 0; q < horizontal.cellCount * geometry.nz; q++) {
      const k = q % geometry.nz;
      for (let v = 0; v < PHYSICAL; v++) {
        const i = q * PACKED + v;
        hydroMaxScaled = Math.max(
          hydroMaxScaled,
          Math.abs(hydroNext.packedState[i]! - hydro[i]!) / componentScale(reference, k, v),
        );
      }
    }

    const initial = disturbedPacked(geometry, reference);
    const before = integratedMassEnergy(initial, geometry);
    let cpuFields = unpack(initial);
    let gpuState = initial.slice();
    let firstComparison: Comparison | null = null;
    let worstStage1Iterations = 0;
    let worstStage2Iterations = 0;
    let worstStage1Residual = 0;
    let worstStage2Residual = 0;
    let totalStage1Halvings = 0;
    let totalStage2Halvings = 0;
    let allPositive = true;

    for (let step = 0; step < STEPS; step++) {
      cpuFields = heviImexSsp2Step(cpuFields, dt, geometry, stencil, reference).fields;
      const gpu = await gpuStepper.step(gpuState, dt);
      gpuState = gpu.packedState;
      if (step === 0) firstComparison = compare(gpuState, cpuFields, initial, geometry, reference);
      worstStage1Iterations = Math.max(worstStage1Iterations, gpu.diagnostics.stage1Iterations);
      worstStage2Iterations = Math.max(worstStage2Iterations, gpu.diagnostics.stage2Iterations);
      worstStage1Residual = Math.max(worstStage1Residual, gpu.diagnostics.stage1Residual);
      worstStage2Residual = Math.max(worstStage2Residual, gpu.diagnostics.stage2Residual);
      totalStage1Halvings += gpu.diagnostics.stage1LineSearchHalvings;
      totalStage2Halvings += gpu.diagnostics.stage2LineSearchHalvings;
      allPositive = allPositive && positive(gpuState, geometry, reference);
    }
    if (!firstComparison) throw new Error('Core v2 GPU IMEX multistep gate did not execute');

    const finalComparison = compare(gpuState, cpuFields, initial, geometry, reference);
    const after = integratedMassEnergy(gpuState, geometry);
    const massDrift = Math.abs(after.mass - before.mass) / Math.max(Math.abs(before.mass), 1);
    const energyDrift = Math.abs(after.energy - before.energy) / Math.max(Math.abs(before.energy), 1);

    $('cfl').textContent = acousticCfl.toFixed(3);
    $('hydro').textContent = hydroMaxScaled.toExponential(6);
    $('stateRelative').textContent = finalComparison.stateRelative.toExponential(6);
    $('incrementRelative').textContent = finalComparison.incrementRelative.toExponential(6);
    $('scaledRms').textContent = finalComparison.scaledRms.toExponential(6);
    $('maxScaled').textContent = finalComparison.maxScaled.toExponential(6);
    $('massDrift').textContent = massDrift.toExponential(6);
    $('energyDrift').textContent = energyDrift.toExponential(6);
    $('stage1').textContent = `${worstStage1Iterations} / ${worstStage1Residual.toExponential(3)}`;
    $('stage2').textContent = `${worstStage2Iterations} / ${worstStage2Residual.toExponential(3)}`;
    $('halvings').textContent = `${totalStage1Halvings} / ${totalStage2Halvings}`;
    $('positive').textContent = String(allPositive);
    $('adapter').textContent = JSON.stringify(adapter.info ?? {}, null, 2);

    const pass =
      acousticCfl > 50 &&
      hydroMaxScaled < 2e-6 &&
      firstComparison.nonFinite === 0 &&
      finalComparison.nonFinite === 0 &&
      allPositive &&
      firstComparison.stateRelative < 2e-5 &&
      firstComparison.incrementRelative < 2e-2 &&
      finalComparison.stateRelative < 5e-5 &&
      finalComparison.incrementRelative < 3e-2 &&
      finalComparison.scaledRms < 4e-4 &&
      finalComparison.maxScaled < 2e-3 &&
      worstStage1Residual <= 2e-5 &&
      worstStage2Residual <= 2e-5 &&
      massDrift < 3e-6 &&
      energyDrift < 3e-6;

    const summary = [
      `steps=${STEPS}`,
      `acousticCfl=${acousticCfl}`,
      `hydroMaxScaled=${hydroMaxScaled}`,
      `oneStateRel=${firstComparison.stateRelative}`,
      `oneIncrementRel=${firstComparison.incrementRelative}`,
      `fourStateRel=${finalComparison.stateRelative}`,
      `fourIncrementRel=${finalComparison.incrementRelative}`,
      `fourScaledRms=${finalComparison.scaledRms}`,
      `fourMaxScaled=${finalComparison.maxScaled}`,
      `massDrift=${massDrift}`,
      `energyDrift=${energyDrift}`,
      `worstStage1Iter=${worstStage1Iterations}`,
      `worstStage1Residual=${worstStage1Residual}`,
      `worstStage2Iter=${worstStage2Iterations}`,
      `worstStage2Residual=${worstStage2Residual}`,
      `halvings=${totalStage1Halvings}/${totalStage2Halvings}`,
      `positive=${allPositive}`,
    ].join(' ');
    console.log(`CORE_V2_GPU_HEVI_IMEX ${summary}`);
    $('log').textContent = summary;
    $('status').textContent = pass ? 'GATE PASS' : 'GATE FAIL';
    $('status').className = pass ? 'ok' : 'bad';
    if (!pass) throw new Error(`Core v2 GPU IMEX-HEVI multistep gate failed: ${summary}`);
  } finally {
    gpuStepper.destroy();
  }
}

run().catch((error) => {
  console.error(error);
  $('status').textContent = 'GATE FAIL';
  $('status').className = 'bad';
  $('log').textContent = error instanceof Error ? (error.stack ?? error.message) : String(error);
});
