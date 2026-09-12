import { norm3, type Vec3 } from '../core/math.js';
import { cellCountOf, type ConservativeFields } from './fields.js';

export function rotateVectorAroundAxis(
  vector: Vec3,
  unitAxis: Vec3,
  angle: number,
): Vec3 {
  const [x, y, z] = vector;
  const [ax, ay, az] = unitAxis;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const dot = ax * x + ay * y + az * z;
  return [
    x * c + (ay * z - az * y) * s + ax * dot * (1 - c),
    y * c + (az * x - ax * z) * s + ay * dot * (1 - c),
    z * c + (ax * y - ay * x) * s + az * dot * (1 - c),
  ];
}

/**
 * Exact update of dm/dt = -2 Omega x m for a constant planetary rotation
 * vector. Density and total energy are unchanged, so the discrete Coriolis
 * operator performs zero work up to roundoff instead of relying on a small
 * explicit timestep to approximate that property.
 */
export function applyExactCoriolis(
  fields: ConservativeFields,
  omega: Vec3,
  dt: number,
): void {
  if (!(dt >= 0) || !Number.isFinite(dt)) {
    throw new Error(`invalid timestep: ${dt}`);
  }
  const omegaMagnitude = norm3(omega);
  if (omegaMagnitude === 0 || dt === 0) return;

  const axis: Vec3 = [
    omega[0] / omegaMagnitude,
    omega[1] / omegaMagnitude,
    omega[2] / omegaMagnitude,
  ];
  const angle = -2 * omegaMagnitude * dt;
  const n = cellCountOf(fields);

  for (let c = 0; c < n; c++) {
    const rotated = rotateVectorAroundAxis(
      [fields.momX[c]!, fields.momY[c]!, fields.momZ[c]!],
      axis,
      angle,
    );
    fields.momX[c] = rotated[0];
    fields.momY[c] = rotated[1];
    fields.momZ[c] = rotated[2];
  }
}
