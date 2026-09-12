export type NumericArray = Float32Array | Float64Array;

export interface ConservativeFields {
  rho: NumericArray;
  momX: NumericArray;
  momY: NumericArray;
  momZ: NumericArray;
  rhoE: NumericArray;
}

export type FieldPrecision = 'f32' | 'f64';

function allocateArray(length: number, precision: FieldPrecision): NumericArray {
  return precision === 'f32' ? new Float32Array(length) : new Float64Array(length);
}

export function createConservativeFields(
  cellCount: number,
  precision: FieldPrecision = 'f64',
): ConservativeFields {
  if (!Number.isInteger(cellCount) || cellCount <= 0) {
    throw new Error(`invalid cell count: ${cellCount}`);
  }
  return {
    rho: allocateArray(cellCount, precision),
    momX: allocateArray(cellCount, precision),
    momY: allocateArray(cellCount, precision),
    momZ: allocateArray(cellCount, precision),
    rhoE: allocateArray(cellCount, precision),
  };
}

export function cellCountOf(fields: ConservativeFields): number {
  const n = fields.rho.length;
  if (
    fields.momX.length !== n ||
    fields.momY.length !== n ||
    fields.momZ.length !== n ||
    fields.rhoE.length !== n
  ) {
    throw new Error('Core v2 conservative field arrays have inconsistent lengths');
  }
  return n;
}
