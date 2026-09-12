import { EARTH, type PlanetConfig } from '../core/constants.js';
import type { ConservativeFields } from './fields.js';
import {
  createIntegratedRate,
  type IntegratedConservativeRate,
} from './finiteVolume.js';
import {
  CONSERVED_COMPONENTS,
  columnStateIndex,
  extractColumnState,
  verticalStiffColumnIntegratedRate,
} from './heviColumn.js';
import type { HydrostaticReference1D } from './hydrostaticReference.js';
import type { LinearReconstructionStencil } from './reconstruction.js';
import {
  shellCellIndex,
  type SphericalShellGeometry,
} from './sphericalShellGeometry.js';
import { closedShellWellBalancedEulerGravityRate } from './wellBalancedEulerGravity.js';

/**
 * Column-local low-order stiff operator used by Core v2 HEVI.
 *
 * Each horizontal column is evaluated by exactly the same column residual used
 * by the implicit solver. No horizontal neighbor is read by this path.
 */
export function closedShellVerticalStiffRateFirstOrder(
  fields: ConservativeFields,
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  planet: PlanetConfig = EARTH,
): IntegratedConservativeRate {
  const horizontal = geometry.horizontal;
  const cellCount = horizontal.cellCount * geometry.nz;
  if (
    fields.rho.length !== cellCount ||
    fields.momX.length !== cellCount ||
    fields.momY.length !== cellCount ||
    fields.momZ.length !== cellCount ||
    fields.rhoE.length !== cellCount
  ) {
    throw new Error('Core v2 HEVI vertical split field/geometry size mismatch');
  }
  const rate = createIntegratedRate(cellCount);

  for (let c = 0; c < horizontal.cellCount; c++) {
    const column = extractColumnState(fields, c, geometry.nz);
    const columnRate = verticalStiffColumnIntegratedRate(
      column,
      c,
      geometry,
      reference,
      planet,
    );
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      rate.rho[q] = columnRate[columnStateIndex(k, 0)]!;
      rate.momX[q] = columnRate[columnStateIndex(k, 1)]!;
      rate.momY[q] = columnRate[columnStateIndex(k, 2)]!;
      rate.momZ[q] = columnRate[columnStateIndex(k, 3)]!;
      rate.rhoE[q] = columnRate[columnStateIndex(k, 4)]!;
    }
  }

  return rate;
}

function subtractRates(
  full: IntegratedConservativeRate,
  stiff: IntegratedConservativeRate,
): IntegratedConservativeRate {
  const n = full.rho.length;
  if (stiff.rho.length !== n) throw new Error('Core v2 HEVI split rate size mismatch');
  const out = createIntegratedRate(n);
  for (let q = 0; q < n; q++) {
    out.rho[q] = full.rho[q]! - stiff.rho[q]!;
    out.momX[q] = full.momX[q]! - stiff.momX[q]!;
    out.momY[q] = full.momY[q]! - stiff.momY[q]!;
    out.momZ[q] = full.momZ[q]! - stiff.momZ[q]!;
    out.rhoE[q] = full.rhoE[q]! - stiff.rhoE[q]!;
  }
  return out;
}

/**
 * Explicit HEVI remainder. By construction
 *
 *   explicitRemainder(U) + verticalStiff(U) = fullSecondOrder(U)
 *
 * component-by-component, so the split cannot silently add or remove a force,
 * flux, gravity term, or stabilizer.
 */
export function closedShellHeviExplicitRemainderRate(
  fields: ConservativeFields,
  geometry: SphericalShellGeometry,
  stencil: LinearReconstructionStencil,
  reference: HydrostaticReference1D,
  planet: PlanetConfig = EARTH,
): IntegratedConservativeRate {
  const full = closedShellWellBalancedEulerGravityRate(
    fields,
    geometry,
    stencil,
    reference,
    planet,
  );
  const stiff = closedShellVerticalStiffRateFirstOrder(
    fields,
    geometry,
    reference,
    planet,
  );
  return subtractRates(full, stiff);
}

export { CONSERVED_COMPONENTS };
