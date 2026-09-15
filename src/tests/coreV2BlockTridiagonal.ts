declare const process: { exitCode?: number };

import {
  solveBlockTridiagonal5,
  type BlockTridiagonal5System,
} from '../corev2/blockTridiagonal5.js';
import { assert } from './assert.js';

const BLOCK = 5;
const BLOCK2 = BLOCK * BLOCK;

interface Test { name: string; fn: () => void }
const tests: Test[] = [];
const test = (name: string, fn: () => void): void => { tests.push({ name, fn }); };

function setBlock(array: Float64Array, block: number, values: readonly number[]): void {
  if (values.length !== BLOCK2) throw new Error('test block must contain 25 entries');
  array.set(values, block * BLOCK2);
}

function matVec(
  matrix: Float64Array,
  block: number,
  vector: Float64Array,
  vectorBlock: number,
): Float64Array {
  const out = new Float64Array(BLOCK);
  const m0 = block * BLOCK2;
  const v0 = vectorBlock * BLOCK;
  for (let r = 0; r < BLOCK; r++) {
    let sum = 0;
    for (let c = 0; c < BLOCK; c++) {
      sum += matrix[m0 + r * BLOCK + c]! * vector[v0 + c]!;
    }
    out[r] = sum;
  }
  return out;
}

function rhsFromKnownSolution(
  lower: Float64Array,
  diagonal: Float64Array,
  upper: Float64Array,
  solution: Float64Array,
): Float64Array {
  const n = solution.length / BLOCK;
  const rhs = new Float64Array(solution.length);
  for (let k = 0; k < n; k++) {
    const center = matVec(diagonal, k, solution, k);
    const left = k > 0 ? matVec(lower, k, solution, k - 1) : new Float64Array(BLOCK);
    const right = k + 1 < n ? matVec(upper, k, solution, k + 1) : new Float64Array(BLOCK);
    for (let i = 0; i < BLOCK; i++) rhs[k * BLOCK + i] = left[i]! + center[i]! + right[i]!;
  }
  return rhs;
}

function maxRelativeError(actual: Float64Array, expected: Float64Array): number {
  let error = 0;
  for (let i = 0; i < actual.length; i++) {
    error = Math.max(error, Math.abs(actual[i]! - expected[i]!) / Math.max(1, Math.abs(expected[i]!)));
  }
  return error;
}

function residualNorm(system: BlockTridiagonal5System, solution: Float64Array): number {
  const reconstructed = rhsFromKnownSolution(system.lower, system.diagonal, system.upper, solution);
  let numerator = 0;
  let denominator = 0;
  for (let i = 0; i < reconstructed.length; i++) {
    const d = reconstructed[i]! - system.rhs[i]!;
    numerator += d * d;
    denominator += system.rhs[i]! * system.rhs[i]!;
  }
  return Math.sqrt(numerator / Math.max(denominator, 1));
}

function deterministicSystem(n: number): { system: BlockTridiagonal5System; solution: Float64Array } {
  const lower = new Float64Array(n * BLOCK2);
  const diagonal = new Float64Array(n * BLOCK2);
  const upper = new Float64Array(n * BLOCK2);
  const solution = new Float64Array(n * BLOCK);

  for (let k = 0; k < n; k++) {
    for (let r = 0; r < BLOCK; r++) {
      solution[k * BLOCK + r] =
        Math.sin(0.37 * (k + 1) * (r + 2)) + 0.13 * k - 0.07 * r;
      for (let c = 0; c < BLOCK; c++) {
        const index = k * BLOCK2 + r * BLOCK + c;
        const offDiag = r === c ? 0 : 0.035 * Math.sin((k + 1) * (r + 2) * (c + 3));
        diagonal[index] = (r === c ? 3.0 + 0.27 * r + 0.11 * k : offDiag);
        if (k > 0) lower[index] =
          (r === c ? -0.24 - 0.015 * r : 0.018 * Math.cos((k + 2) * (r + 1) * (c + 2)));
        if (k + 1 < n) upper[index] =
          (r === c ? -0.19 - 0.012 * c : 0.014 * Math.sin((k + 3) * (r + 2) * (c + 1)));
      }
    }
  }

  return {
    system: {
      lower,
      diagonal,
      upper,
      rhs: rhsFromKnownSolution(lower, diagonal, upper, solution),
    },
    solution,
  };
}

test('Core v2 5x5 column solver handles a single pivoting block', () => {
  const lower = new Float64Array(BLOCK2);
  const upper = new Float64Array(BLOCK2);
  const diagonal = new Float64Array(BLOCK2);
  setBlock(diagonal, 0, [
    0, 2, -1, 0.5, 0,
    3, 0.2, 0, -0.4, 0.1,
    0.5, -0.3, 4, 0.2, -0.7,
    0, 0.6, -0.2, 3.5, 0.9,
    -0.4, 0, 0.8, -0.5, 2.8,
  ]);
  const expected = new Float64Array([1.2, -0.7, 2.1, 0.4, -1.3]);
  const rhs = rhsFromKnownSolution(lower, diagonal, upper, expected);
  const system = { lower, diagonal, upper, rhs };
  const actual = solveBlockTridiagonal5(system);
  const error = maxRelativeError(actual, expected);
  assert(error < 2e-13, `single-block solution relative error=${error}`);
  const residual = residualNorm(system, actual);
  assert(residual < 2e-14, `single-block residual=${residual}`);
});

test('Core v2 5x5 block Thomas solver recovers a nine-level coupled column', () => {
  const { system, solution } = deterministicSystem(9);
  const actual = solveBlockTridiagonal5(system);
  const error = maxRelativeError(actual, solution);
  assert(error < 2e-12, `nine-level solution relative error=${error}`);
  const residual = residualNorm(system, actual);
  assert(residual < 2e-13, `nine-level residual=${residual}`);
});

test('Core v2 5x5 block Thomas solver scales to a deep atmospheric column', () => {
  const { system, solution } = deterministicSystem(96);
  const actual = solveBlockTridiagonal5(system);
  const error = maxRelativeError(actual, solution);
  assert(error < 4e-12, `96-level solution relative error=${error}`);
  const residual = residualNorm(system, actual);
  assert(residual < 4e-13, `96-level residual=${residual}`);
});

test('Core v2 column solver rejects a singular diagonal block', () => {
  const lower = new Float64Array(BLOCK2);
  const diagonal = new Float64Array(BLOCK2);
  const upper = new Float64Array(BLOCK2);
  const rhs = new Float64Array(BLOCK);
  let threw = false;
  try {
    solveBlockTridiagonal5({ lower, diagonal, upper, rhs });
  } catch {
    threw = true;
  }
  assert(threw, 'singular block system must be rejected');
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
console.log(`${passed}/${tests.length} Core v2 block-tridiagonal tests passed`);
