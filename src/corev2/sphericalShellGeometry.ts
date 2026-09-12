import type { Vec3 } from '../core/math.js';
import type { CubedSphereGrid } from '../grid/cubedSphere.js';

export interface SphericalShellGeometry {
  horizontal: CubedSphereGrid;
  nz: number;
  /** Absolute radius of each radial interface [m], length nz + 1. */
  radiusInterface: Float64Array;
  /** 3-D cell volume [m^3], indexed horizontalCell * nz + k. */
  cellVolume: Float64Array;
  /**
   * Area of spherical radial faces [m^2], indexed horizontalCell * (nz + 1) + kInterface.
   * The scalar area is exact for the cubed-sphere spherical polygon solid angle.
   */
  radialFaceArea: Float64Array;
  /**
   * Area of vertical side faces [m^2], indexed horizontalEdge * nz + k.
   * Because each cubed-sphere edge is a great-circle arc, this conical side-face
   * area is exact: 0.5 * (rTop^2-rBottom^2) * angularLength.
   */
  sideFaceArea: Float64Array;
}

export function shellCellIndex(horizontalCell: number, k: number, nz: number): number {
  return horizontalCell * nz + k;
}

export function shellRadialFaceIndex(
  horizontalCell: number,
  kInterface: number,
  nz: number,
): number {
  return horizontalCell * (nz + 1) + kInterface;
}

export function shellSideFaceIndex(horizontalEdge: number, k: number, nz: number): number {
  return horizontalEdge * nz + k;
}

/**
 * Build exact control-volume sizes for an extrusion of the existing
 * great-circle cubed-sphere cells through concentric spherical shells.
 *
 * This function deliberately does not invent a radial-face normal approximation;
 * radial vector-area geometry is a separate operator contract and will be added
 * only when its discrete pressure/gravity coupling is fixed.
 */
export function buildSphericalShellGeometry(
  horizontal: CubedSphereGrid,
  radii: ArrayLike<number>,
): SphericalShellGeometry {
  if (radii.length < 2) {
    throw new Error('Core v2 spherical shell needs at least two radial interfaces');
  }
  const nz = radii.length - 1;
  const radiusInterface = new Float64Array(radii.length);
  for (let k = 0; k < radii.length; k++) {
    const r = radii[k]!;
    if (!(r > 0) || !Number.isFinite(r)) {
      throw new Error(`invalid radius at interface ${k}: ${r}`);
    }
    if (k > 0 && !(r > radii[k - 1]!)) {
      throw new Error('Core v2 radial interfaces must be strictly increasing');
    }
    radiusInterface[k] = r;
  }

  const cellVolume = new Float64Array(horizontal.cellCount * nz);
  const radialFaceArea = new Float64Array(horizontal.cellCount * (nz + 1));
  const sideFaceArea = new Float64Array(horizontal.edgeCount * nz);

  for (let c = 0; c < horizontal.cellCount; c++) {
    const solidAngle = horizontal.cellAreaUnit[c]!;
    for (let ki = 0; ki <= nz; ki++) {
      const r = radiusInterface[ki]!;
      radialFaceArea[shellRadialFaceIndex(c, ki, nz)] = solidAngle * r * r;
    }
    for (let k = 0; k < nz; k++) {
      const r0 = radiusInterface[k]!;
      const r1 = radiusInterface[k + 1]!;
      cellVolume[shellCellIndex(c, k, nz)] =
        solidAngle * (r1 * r1 * r1 - r0 * r0 * r0) / 3;
    }
  }

  for (let e = 0; e < horizontal.edgeCount; e++) {
    const angularLength = horizontal.edges[e]!.angularLength;
    for (let k = 0; k < nz; k++) {
      const r0 = radiusInterface[k]!;
      const r1 = radiusInterface[k + 1]!;
      sideFaceArea[shellSideFaceIndex(e, k, nz)] =
        0.5 * (r1 * r1 - r0 * r0) * angularLength;
    }
  }

  return {
    horizontal,
    nz,
    radiusInterface,
    cellVolume,
    radialFaceArea,
    sideFaceArea,
  };
}

/** Exact constant unit normal of a great-circle side face, oriented left -> right. */
export function sideFaceUnitNormal(
  geometry: SphericalShellGeometry,
  horizontalEdge: number,
): Vec3 {
  return geometry.horizontal.edges[horizontalEdge]!.normal;
}
