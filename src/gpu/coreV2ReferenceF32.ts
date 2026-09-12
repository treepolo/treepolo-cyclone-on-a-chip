import { DRY_AIR } from '../core/constants.js';

/**
 * Canonical float32 total-energy density of one hydrostatic reference layer.
 *
 * The GPU dry core stores the prognostic state in f32. Recomputing the large
 * reference internal + gravitational energy expression independently inside
 * WGSL can differ by a float32 rounding unit because shader contraction and
 * evaluation order are not required to match JavaScript Math.fround exactly.
 * Storing this canonical value with the reference state makes the discrete
 * hydrostatic reference an exact algebraic zero instead of subtracting two
 * independently rounded large numbers.
 */
export function coreV2ReferenceTotalEnergyF32(
  density: number,
  pressure: number,
  geopotential: number,
): number {
  const f32 = Math.fround;
  const gamma = f32(DRY_AIR.gamma);
  const gammaMinusOne = f32(gamma - f32(1));
  const internal = f32(f32(pressure) / gammaMinusOne);
  const potential = f32(f32(density) * f32(geopotential));
  return f32(internal + potential);
}
