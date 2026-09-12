import { solveBlockTridiagonal5 } from './corev2/blockTridiagonal5.js';
import { CoreV2GpuBlockTridiagonal5, type CoreV2GpuBlockTridiagonalBatch } from './gpu/coreV2BlockTridiagonalGpu.js';

const $ = (id: string): HTMLElement => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing #${id}`);
  return node;
};

function buildBatch(columnCount: number, nz: number): CoreV2GpuBlockTridiagonalBatch {
  const blocks = columnCount * nz;
  const lower = new Float32Array(blocks * 25);
  const diagonal = new Float32Array(blocks * 25);
  const upper = new Float32Array(blocks * 25);
  const rhs = new Float32Array(blocks * 5);
  for (let c = 0; c < columnCount; c++) {
    for (let k = 0; k < nz; k++) {
      const block = c * nz + k;
      for (let r = 0; r < 5; r++) {
        rhs[block * 5 + r] = Math.fround(
          Math.sin(0.31 * (c + 1) * (r + 1) + 0.17 * (k + 1)) + 0.03 * k,
        );
        for (let j = 0; j < 5; j++) {
          const ij = block * 25 + r * 5 + j;
          const off = r === j ? 0 : 0.018 * Math.sin((r + 1) * (j + 2) + 0.2 * k);
          diagonal[ij] = Math.fround((r === j ? 3.8 + 0.07 * r + 0.01 * k : 0) + off);
          lower[ij] = Math.fround(k === 0 ? 0 : 0.025 * Math.cos(0.4 * (r + 1) * (j + 1) + c));
          upper[ij] = Math.fround(k + 1 === nz ? 0 : 0.022 * Math.sin(0.3 * (r + 2) * (j + 1) + k));
        }
      }
      if ((c + k) % 5 === 0) {
        const base = block * 25;
        diagonal[base] = 0;
        diagonal[base + 5] = Math.fround(3.6);
      }
    }
  }
  return { columnCount, nz, lower, diagonal, upper, rhs };
}

function cpuSolution(batch: CoreV2GpuBlockTridiagonalBatch): Float64Array {
  const out = new Float64Array(batch.rhs.length);
  for (let c = 0; c < batch.columnCount; c++) {
    const block0 = c * batch.nz;
    const block1 = block0 + batch.nz;
    const rhs0 = block0 * 5;
    const rhs1 = block1 * 5;
    const solution = solveBlockTridiagonal5({
      lower: Float64Array.from(batch.lower.subarray(block0 * 25, block1 * 25)),
      diagonal: Float64Array.from(batch.diagonal.subarray(block0 * 25, block1 * 25)),
      upper: Float64Array.from(batch.upper.subarray(block0 * 25, block1 * 25)),
      rhs: Float64Array.from(batch.rhs.subarray(rhs0, rhs1)),
    });
    out.set(solution, rhs0);
  }
  return out;
}

function relative(actual: ArrayLike<number>, expected: ArrayLike<number>): number {
  let n = 0;
  let d = 0;
  for (let i = 0; i < actual.length; i++) {
    const e = expected[i]!;
    const dx = actual[i]! - e;
    n += dx * dx;
    d += e * e;
  }
  return Math.sqrt(n / Math.max(d, 1e-30));
}

function maxResidual(batch: CoreV2GpuBlockTridiagonalBatch, x: Float32Array): number {
  let worst = 0;
  for (let c = 0; c < batch.columnCount; c++) {
    for (let k = 0; k < batch.nz; k++) {
      const block = c * batch.nz + k;
      for (let r = 0; r < 5; r++) {
        let value = 0;
        for (let j = 0; j < 5; j++) {
          value += batch.diagonal[block * 25 + r * 5 + j]! * x[block * 5 + j]!;
          if (k > 0) value += batch.lower[block * 25 + r * 5 + j]! * x[(block - 1) * 5 + j]!;
          if (k + 1 < batch.nz) value += batch.upper[block * 25 + r * 5 + j]! * x[(block + 1) * 5 + j]!;
        }
        const residual = Math.abs(value - batch.rhs[block * 5 + r]!);
        worst = Math.max(worst, residual);
      }
    }
  }
  return worst;
}

async function run(): Promise<void> {
  $('status').textContent = 'RUNNING';
  const webgpu = (navigator as Navigator & { gpu?: any }).gpu;
  if (!webgpu) throw new Error('navigator.gpu unavailable');
  const adapter = await webgpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  const device = await adapter.requestDevice();
  const solver = new CoreV2GpuBlockTridiagonal5(device);

  const batch = buildBatch(24, 24);
  const expected = cpuSolution(batch);
  const result = await solver.solve(batch);
  let badStatus = 0;
  for (const status of result.status) if (status !== 0) badStatus++;
  const error = relative(result.solution, expected);
  const residual = maxResidual(batch, result.solution);
  $('relative').textContent = error.toExponential(6);
  $('residual').textContent = residual.toExponential(6);
  $('statusCount').textContent = String(badStatus);

  const singular: CoreV2GpuBlockTridiagonalBatch = {
    columnCount: 1,
    nz: 1,
    lower: new Float32Array(25),
    diagonal: new Float32Array(25),
    upper: new Float32Array(25),
    rhs: new Float32Array([1, 2, 3, 4, 5]),
  };
  const singularResult = await solver.solve(singular);
  const singularDetected = singularResult.status[0]! !== 0;
  $('singular').textContent = singularDetected ? 'YES' : 'NO';
  $('adapter').textContent = JSON.stringify(adapter.info ?? {}, null, 2);

  const pass = badStatus === 0 && error < 2e-4 && residual < 2e-4 && singularDetected;
  $('status').textContent = pass ? 'GATE PASS' : 'GATE FAIL';
  $('status').className = pass ? 'ok' : 'bad';
  if (!pass) {
    throw new Error(`Core v2 GPU block5 gate: status=${badStatus} rel=${error} residual=${residual} singular=${singularDetected}`);
  }
}

run().catch((error) => {
  console.error(error);
  $('status').textContent = 'GATE FAIL';
  $('status').className = 'bad';
  $('log').textContent = error instanceof Error ? (error.stack ?? error.message) : String(error);
});
