import { dot3, scale3, type Vec3 } from '../core/math.js';
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
  /**
   * Exact vector area of each horizontal cell on the unit sphere:
   * integral_cell rHat dOmega. This is enough to reconstruct the vector area
   * of every spherical radial face as r^2 * cellVectorAreaUnit, without storing
   * three numbers per 3-D radial face.
   */
  cellVectorAreaUnit: Float64Array;
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

function cellCenter(horizontal: CubedSphereGrid, c: number): Vec3 {
  return [
    horizontal.cellCenters[c * 3]!,
    horizontal.cellCenters[c * 3 + 1]!,
    horizontal.cellCenters[c * 3 + 2]!,
  ];
}

/**
 * Exact vector area for a spherical polygon bounded by great-circle arcs.
 *
 * For a unit sphere and a boundary edge with outward spherical conormal q and
 * arc length L, the vector-area theorem gives a contribution -0.5 q L.
 */
function buildUnitCellVectorAreas(horizontal: CubedSphereGrid): Float64Array {
  const out = new Float64Array(horizontal.cellCount * 3);
  for (let c = 0; c < horizontal.cellCount; c++) {
    let ax = 0;
    let ay = 0;
    let az = 0;
    for (let s = 0; s < 4; s++) {
      const edgeId = horizontal.cellEdges[c * 4 + s]!;
      const sign = horizontal.cellEdgeSigns[c * 4 + s]!;
      const edge = horizontal.edges[edgeId]!;
      const qx = sign * edge.normal[0];
      const qy = sign * edge.normal[1];
      const qz = sign * edge.normal[2];
      const factor = -0.5 * edge.angularLength;
      ax += factor * qx;
      ay += factor * qy;
      az += factor * qz;
    }

    // Keep the vector orientation explicitly outward even if a future grid
    // builder changes its boundary-edge orientation convention.
    const center = cellCenter(horizontal, c);
    if (dot3([ax, ay, az], center) < 0) {
      ax = -ax;
      ay = -ay;
      az = -az;
    }
    out[c * 3] = ax;
    out[c * 3 + 1] = ay;
    out[c * 3 + 2] = az;
  }
  return out;
}

/**
 * Build exact control-volume sizes for an extrusion of the existing
 * great-circle cubed-sphere cells through concentric spherical shells.
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
  const cellVectorAreaUnit = buildUnitCellVectorAreas(horizontal);

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
    cellVectorAreaUnit,
  };
}

/** Exact constant unit normal of a great-circle side face, oriented left -> right. */
export function sideFaceUnitNormal(
  geometry: SphericalShellGeometry,
  horizontalEdge: number,
): Vec3 {
  return geometry.horizontal.edges[horizontalEdge]!.normal;
}

/** Exact side-face vector area, oriented from the edge's left cell to right cell. */
export function sideFaceVectorArea(
  geometry: SphericalShellGeometry,
  horizontalEdge: number,
  k: number,
): Vec3 {
  const area = geometry.sideFaceArea[shellSideFaceIndex(horizontalEdge, k, geometry.nz)]!;
  return scale3(sideFaceUnitNormal(geometry, horizontalEdge), area);
}

/**
 * Exact spherical radial-face vector area oriented in the +radial sense.
 * A lower cell uses +this vector on its top face; an upper cell uses -this
 * vector on its bottom face.
 */
export function radialFaceVectorArea(
  geometry: SphericalShellGeometry,
  horizontalCell: number,
  kInterface: number,
): Vec3 {
  const r = geometry.radiusInterface[kInterface]!;
  const scale = r * r;
  return [
    geometry.cellVectorAreaUnit[horizontalCell * 3]! * scale,
    geometry.cellVectorAreaUnit[horizontalCell * 3 + 1]! * scale,
    geometry.cellVectorAreaUnit[horizontalCell * 3 + 2]! * scale,
  ];
}
