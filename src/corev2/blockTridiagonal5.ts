const BLOCK = 5;
const BLOCK2 = BLOCK * BLOCK;

export interface BlockTridiagonal5System {
  /** Lower block A_k; block 0 is ignored. Row-major 5x5 blocks. */
  lower: Float64Array;
  /** Diagonal block B_k. Row-major 5x5 blocks. */
  diagonal: Float64Array;
  /** Upper block C_k; final block is ignored. Row-major 5x5 blocks. */
  upper: Float64Array;
  /** Right-hand side, five components per vertical cell. */
  rhs: Float64Array;
}

function validate(system: BlockTridiagonal5System): number {
  if (system.rhs.length === 0 || system.rhs.length % BLOCK !== 0) {
    throw new Error('Core v2 block-tridiagonal RHS length must be a positive multiple of 5');
  }
  const n = system.rhs.length / BLOCK;
  const expected = n * BLOCK2;
  if (
    system.lower.length !== expected ||
    system.diagonal.length !== expected ||
    system.upper.length !== expected
  ) {
    throw new Error('Core v2 block-tridiagonal matrix array lengths do not match RHS');
  }
  return n;
}

function copyBlock(source: Float64Array, block: number, out: Float64Array): void {
  const start = block * BLOCK2;
  for (let i = 0; i < BLOCK2; i++) out[i] = source[start + i]!;
}

function copyVector(source: Float64Array, block: number, out: Float64Array): void {
  const start = block * BLOCK;
  for (let i = 0; i < BLOCK; i++) out[i] = source[start + i]!;
}

function multiplyBlockVector(
  matrix: Float64Array,
  vector: Float64Array,
  out: Float64Array,
): void {
  for (let r = 0; r < BLOCK; r++) {
    let value = 0;
    for (let c = 0; c < BLOCK; c++) value += matrix[r * BLOCK + c]! * vector[c]!;
    out[r] = value;
  }
}

function multiplyBlocks(
  left: Float64Array,
  right: Float64Array,
  out: Float64Array,
): void {
  for (let r = 0; r < BLOCK; r++) {
    for (let c = 0; c < BLOCK; c++) {
      let value = 0;
      for (let k = 0; k < BLOCK; k++) value += left[r * BLOCK + k]! * right[k * BLOCK + c]!;
      out[r * BLOCK + c] = value;
    }
  }
}

/**
 * Solve one 5x5 matrix against five or fewer RHS columns with partial pivoting.
 * `rhsColumns` is row-major [row][column]. The matrix and RHS are overwritten.
 */
function solveDense5InPlace(
  matrix: Float64Array,
  rhsColumns: Float64Array,
  columnCount: number,
): void {
  if (columnCount < 1 || columnCount > BLOCK) throw new Error('invalid dense5 RHS column count');

  let matrixScale = 0;
  for (let i = 0; i < BLOCK2; i++) matrixScale = Math.max(matrixScale, Math.abs(matrix[i]!));
  if (!(matrixScale > 0) || !Number.isFinite(matrixScale)) {
    throw new Error('singular/non-finite Core v2 5x5 block');
  }

  for (let pivotColumn = 0; pivotColumn < BLOCK; pivotColumn++) {
    let pivotRow = pivotColumn;
    let pivotMagnitude = Math.abs(matrix[pivotColumn * BLOCK + pivotColumn]!);
    for (let row = pivotColumn + 1; row < BLOCK; row++) {
      const magnitude = Math.abs(matrix[row * BLOCK + pivotColumn]!);
      if (magnitude > pivotMagnitude) {
        pivotMagnitude = magnitude;
        pivotRow = row;
      }
    }
    if (!(pivotMagnitude > 1e-14 * matrixScale) || !Number.isFinite(pivotMagnitude)) {
      throw new Error(`singular Core v2 5x5 pivot column=${pivotColumn} magnitude=${pivotMagnitude}`);
    }

    if (pivotRow !== pivotColumn) {
      for (let c = 0; c < BLOCK; c++) {
        const a = pivotColumn * BLOCK + c;
        const b = pivotRow * BLOCK + c;
        const temp = matrix[a]!;
        matrix[a] = matrix[b]!;
        matrix[b] = temp;
      }
      for (let c = 0; c < columnCount; c++) {
        const a = pivotColumn * columnCount + c;
        const b = pivotRow * columnCount + c;
        const temp = rhsColumns[a]!;
        rhsColumns[a] = rhsColumns[b]!;
        rhsColumns[b] = temp;
      }
    }

    const pivot = matrix[pivotColumn * BLOCK + pivotColumn]!;
    for (let row = pivotColumn + 1; row < BLOCK; row++) {
      const factor = matrix[row * BLOCK + pivotColumn]! / pivot;
      matrix[row * BLOCK + pivotColumn] = 0;
      for (let c = pivotColumn + 1; c < BLOCK; c++) {
        matrix[row * BLOCK + c] = matrix[row * BLOCK + c]! - factor * matrix[pivotColumn * BLOCK + c]!;
      }
      for (let c = 0; c < columnCount; c++) {
        rhsColumns[row * columnCount + c] =
          rhsColumns[row * columnCount + c]! - factor * rhsColumns[pivotColumn * columnCount + c]!;
      }
    }
  }

  for (let row = BLOCK - 1; row >= 0; row--) {
    const pivot = matrix[row * BLOCK + row]!;
    for (let c = 0; c < columnCount; c++) {
      let value = rhsColumns[row * columnCount + c]!;
      for (let k = row + 1; k < BLOCK; k++) {
        value -= matrix[row * BLOCK + k]! * rhsColumns[k * columnCount + c]!;
      }
      rhsColumns[row * columnCount + c] = value / pivot;
    }
  }
}

/**
 * Solve a block-tridiagonal system with fixed 5x5 blocks using block Thomas
 * elimination. This is the column-local linear algebra primitive required by
 * the Core v2 HEVI update: one atmospheric column can be solved independently,
 * with O(nz) work and no global pressure solve.
 */
export function solveBlockTridiagonal5(system: BlockTridiagonal5System): Float64Array {
  const n = validate(system);
  const modifiedUpper = new Float64Array(n * BLOCK2);
  const modifiedRhs = new Float64Array(n * BLOCK);

  const b = new Float64Array(BLOCK2);
  const c = new Float64Array(BLOCK2);
  const a = new Float64Array(BLOCK2);
  const rhs = new Float64Array(BLOCK);
  const productBlock = new Float64Array(BLOCK2);
  const productVector = new Float64Array(BLOCK);

  for (let k = 0; k < n; k++) {
    copyBlock(system.diagonal, k, b);
    copyVector(system.rhs, k, rhs);

    if (k > 0) {
      copyBlock(system.lower, k, a);
      const previousUpper = modifiedUpper.subarray((k - 1) * BLOCK2, k * BLOCK2);
      const previousRhs = modifiedRhs.subarray((k - 1) * BLOCK, k * BLOCK);
      multiplyBlocks(a, previousUpper, productBlock);
      multiplyBlockVector(a, previousRhs, productVector);
      for (let i = 0; i < BLOCK2; i++) b[i] = b[i]! - productBlock[i]!;
      for (let i = 0; i < BLOCK; i++) rhs[i] = rhs[i]! - productVector[i]!;
    }

    const hasUpper = k + 1 < n;
    const columnCount = hasUpper ? BLOCK : 1;
    const combined = new Float64Array(BLOCK * columnCount);
    if (hasUpper) {
      copyBlock(system.upper, k, c);
      for (let row = 0; row < BLOCK; row++) {
        for (let col = 0; col < BLOCK; col++) combined[row * BLOCK + col] = c[row * BLOCK + col]!;
      }
      // Solve B_k * C'_k = C_k.
      solveDense5InPlace(b.slice(), combined, BLOCK);
      modifiedUpper.set(combined, k * BLOCK2);

      // Solve B_k * d'_k = d_k with the same matrix. Re-factorization is tiny
      // (5x5) and avoids storing LU factors per level; the GPU kernel can fuse
      // these RHS columns when it is implemented.
      const rhsColumn = new Float64Array(BLOCK);
      rhsColumn.set(rhs);
      solveDense5InPlace(b, rhsColumn, 1);
      modifiedRhs.set(rhsColumn, k * BLOCK);
    } else {
      const rhsColumn = new Float64Array(BLOCK);
      rhsColumn.set(rhs);
      solveDense5InPlace(b, rhsColumn, columnCount);
      modifiedRhs.set(rhsColumn, k * BLOCK);
    }
  }

  const solution = new Float64Array(n * BLOCK);
  solution.set(modifiedRhs.subarray((n - 1) * BLOCK, n * BLOCK), (n - 1) * BLOCK);
  for (let k = n - 2; k >= 0; k--) {
    const upper = modifiedUpper.subarray(k * BLOCK2, (k + 1) * BLOCK2);
    const next = solution.subarray((k + 1) * BLOCK, (k + 2) * BLOCK);
    multiplyBlockVector(upper, next, productVector);
    for (let i = 0; i < BLOCK; i++) {
      solution[k * BLOCK + i] = modifiedRhs[k * BLOCK + i]! - productVector[i]!;
    }
  }
  return solution;
}
