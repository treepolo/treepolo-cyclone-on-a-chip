import { EARTH } from './core/constants.js';
import type { Vec3 } from './core/math.js';
import { rotateVectorAroundAxis } from './corev2/rotation.js';
import { integratedSlau2ReferenceSubtractedFluxFromPrimitive } from './corev2/slau2Flux.js';
import type { PrimitiveCell } from './corev2/state.js';
import { CoreV2GpuFoundation } from './gpu/coreV2GpuFoundation.js';

const $ = (id: string): HTMLElement => {
  const element = document.getElementById(id);
  if (!element) throw new Error(`missing element #${id}`);
  return element;
};

function relativeL2(actual: ArrayLike<number>, expected: ArrayLike<number>): number {
  if (actual.length !== expected.length) throw new Error('relativeL2 length mismatch');
  let num = 0;
  let den = 0;
  for (let i = 0; i < actual.length; i++) {
    const d = actual[i]! - expected[i]!;
    num += d * d;
    den += expected[i]! * expected[i]!;
  }
  return Math.sqrt(num / Math.max(den, 1e-30));
}

function maxAbs(actual: ArrayLike<number>, expected: ArrayLike<number>): number {
  let worst = 0;
  for (let i = 0; i < actual.length; i++) {
    worst = Math.max(worst, Math.abs(actual[i]! - expected[i]!));
  }
  return worst;
}

function f32(value: number): number {
  return Math.fround(value);
}

function cpuCoriolisExpected(
  input: Float32Array,
  omega: Vec3,
  dt: number,
): Float32Array {
  const out = input.slice();
  const omegaMagnitude = Math.hypot(...omega);
  const axis: Vec3 = omegaMagnitude === 0
    ? [0, 0, 1]
    : [omega[0] / omegaMagnitude, omega[1] / omegaMagnitude, omega[2] / omegaMagnitude];
  const angle = -2 * omegaMagnitude * dt;
  for (let q = 0; q < input.length / 8; q++) {
    const i = q * 8;
    const rotated = rotateVectorAroundAxis(
      [input[i + 1]!, input[i + 2]!, input[i + 3]!],
      axis,
      angle,
    );
    out[i + 1] = f32(rotated[0]);
    out[i + 2] = f32(rotated[1]);
    out[i + 3] = f32(rotated[2]);
  }
  return out;
}

function packPrimitive(primitive: PrimitiveCell, geopotential: number): number[] {
  return [
    f32(primitive.rho),
    f32(primitive.velocity[0]),
    f32(primitive.velocity[1]),
    f32(primitive.velocity[2]),
    f32(primitive.pressure),
    f32(geopotential),
    0,
    0,
  ];
}

function unpackPrimitive(packed: Float32Array, face: number): { primitive: PrimitiveCell; phi: number } {
  const i = face * 8;
  return {
    primitive: {
      rho: packed[i]!,
      velocity: [packed[i + 1]!, packed[i + 2]!, packed[i + 3]!],
      pressure: packed[i + 4]!,
    },
    phi: packed[i + 5]!,
  };
}

async function run(): Promise<void> {
  $('status').textContent = 'RUNNING';
  const gpu = (navigator as Navigator & { gpu?: any }).gpu;
  if (!gpu) throw new Error('navigator.gpu unavailable');
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  const device = await adapter.requestDevice();
  const runtime = new CoreV2GpuFoundation(device);

  const state = new Float32Array(6 * 8);
  for (let q = 0; q < 6; q++) {
    const i = q * 8;
    state[i] = f32(0.4 + 0.13 * q);
    state[i + 1] = f32(11 - 2.7 * q);
    state[i + 2] = f32(-7 + 1.9 * q);
    state[i + 3] = f32(3.5 - 0.6 * q);
    state[i + 4] = f32(2.4e5 + 1234 * q);
  }
  const omega: Vec3 = [f32(1.2e-5), f32(-2.5e-5), f32(EARTH.omega)];
  const dt = f32(137.5);
  const gpuCoriolis = await runtime.applyExactCoriolis(state, omega, dt);
  const cpuCoriolis = cpuCoriolisExpected(state, omega, dt);
  const coriolisRel = relativeL2(gpuCoriolis, cpuCoriolis);
  const coriolisMax = maxAbs(gpuCoriolis, cpuCoriolis);

  let normBefore = 0;
  let normAfter = 0;
  for (let q = 0; q < 6; q++) {
    const i = q * 8;
    normBefore += state[i + 1]! ** 2 + state[i + 2]! ** 2 + state[i + 3]! ** 2;
    normAfter += gpuCoriolis[i + 1]! ** 2 + gpuCoriolis[i + 2]! ** 2 + gpuCoriolis[i + 3]! ** 2;
  }
  const coriolisNormRel = Math.abs(normAfter - normBefore) / normBefore;

  const leftValues: number[] = [];
  const rightValues: number[] = [];
  const geometryValues: number[] = [];
  const cases = [
    {
      left: { rho: 1.15, velocity: [0, 0, 0] as Vec3, pressure: 90000 },
      right: { rho: 1.15, velocity: [0, 0, 0] as Vec3, pressure: 90000 },
      phiL: 5000,
      phiR: 5000,
      area: [2.5e7, -1.2e7, 0.8e7] as Vec3,
      pRef: 90000,
    },
    {
      left: { rho: 1.08, velocity: [18, -4, 2] as Vec3, pressure: 92030 },
      right: { rho: 1.02, velocity: [13, 3, -1] as Vec3, pressure: 91970 },
      phiL: 3100,
      phiR: 3300,
      area: [1.6e7, 2.1e7, -0.7e7] as Vec3,
      pRef: 92000,
    },
    {
      left: { rho: 0.52, velocity: [-62, 17, 8] as Vec3, pressure: 42080 },
      right: { rho: 0.61, velocity: [-25, -9, 4] as Vec3, pressure: 42140 },
      phiL: 42000,
      phiR: 42500,
      area: [-1.2e7, 0.6e7, 1.9e7] as Vec3,
      pRef: 42100,
    },
    {
      left: { rho: 0.18, velocity: [210, -35, 20] as Vec3, pressure: 11000 },
      right: { rho: 0.14, velocity: [160, 15, -12] as Vec3, pressure: 8600 },
      phiL: 150000,
      phiR: 160000,
      area: [0.3e7, -0.4e7, 0.5e7] as Vec3,
      pRef: 9800,
    },
  ];
  for (const item of cases) {
    leftValues.push(...packPrimitive(item.left, item.phiL));
    rightValues.push(...packPrimitive(item.right, item.phiR));
    geometryValues.push(
      f32(item.area[0]),
      f32(item.area[1]),
      f32(item.area[2]),
      f32(item.pRef),
    );
  }
  const left = new Float32Array(leftValues);
  const right = new Float32Array(rightValues);
  const geometry = new Float32Array(geometryValues);
  const gpuFlux = await runtime.referenceSubtractedSlau2Faces(left, right, geometry);
  const cpuFlux = new Float32Array(cases.length * 8);
  for (let face = 0; face < cases.length; face++) {
    const l = unpackPrimitive(left, face);
    const r = unpackPrimitive(right, face);
    const gi = face * 4;
    const vectorArea: Vec3 = [geometry[gi]!, geometry[gi + 1]!, geometry[gi + 2]!];
    const flux = integratedSlau2ReferenceSubtractedFluxFromPrimitive(
      l.primitive,
      r.primitive,
      vectorArea,
      geometry[gi + 3]!,
      l.phi,
      r.phi,
    );
    const oi = face * 8;
    cpuFlux[oi] = f32(flux.mass);
    cpuFlux[oi + 1] = f32(flux.momentum[0]);
    cpuFlux[oi + 2] = f32(flux.momentum[1]);
    cpuFlux[oi + 3] = f32(flux.momentum[2]);
    cpuFlux[oi + 4] = f32(flux.totalEnergy);
  }
  const fluxRel = relativeL2(gpuFlux, cpuFlux);
  const fluxMax = maxAbs(gpuFlux, cpuFlux);
  const hydrostaticFluxMax = Math.max(
    Math.abs(gpuFlux[0]!),
    Math.abs(gpuFlux[1]!),
    Math.abs(gpuFlux[2]!),
    Math.abs(gpuFlux[3]!),
    Math.abs(gpuFlux[4]!),
  );

  $('coriolisRel').textContent = coriolisRel.toExponential(6);
  $('coriolisMax').textContent = coriolisMax.toExponential(6);
  $('coriolisNorm').textContent = coriolisNormRel.toExponential(6);
  $('fluxRel').textContent = fluxRel.toExponential(6);
  $('fluxMax').textContent = fluxMax.toExponential(6);
  $('hydroFlux').textContent = hydrostaticFluxMax.toExponential(6);
  $('adapter').textContent = JSON.stringify(adapter.info ?? {}, null, 2);

  const pass =
    coriolisRel < 2e-6 &&
    coriolisNormRel < 3e-6 &&
    fluxRel < 3e-5 &&
    hydrostaticFluxMax === 0;
  $('status').textContent = pass ? 'GATE PASS' : 'GATE FAIL';
  $('status').className = pass ? 'ok' : 'bad';
  if (!pass) {
    throw new Error(
      `Core v2 GPU foundation gate failed: coriolisRel=${coriolisRel}, ` +
      `coriolisNorm=${coriolisNormRel}, fluxRel=${fluxRel}, hydroFlux=${hydrostaticFluxMax}`,
    );
  }
}

run().catch((error) => {
  console.error(error);
  $('status').textContent = 'GATE FAIL';
  $('status').className = 'bad';
  $('log').textContent = error instanceof Error ? `${error.stack ?? error.message}` : String(error);
});
