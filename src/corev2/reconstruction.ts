import { norm3, type Vec3 } from '../core/math.js';
import { cellCountOf, type ConservativeFields } from './fields.js';
import { primitiveFromConserved, type PrimitiveCell } from './state.js';
import {
  shellCellIndex,
  type SphericalShellGeometry,
} from './sphericalShellGeometry.js';

export const MAX_RECONSTRUCTION_NEIGHBORS = 6;

export interface LinearReconstructionStencil {
  geometry: SphericalShellGeometry;
  /** Exact volume centroid of each 3-D control volume [m], xyz packed. */
  cellCentroid: Float64Array;
  neighborCount: Uint8Array;
  /** Fixed six-slot neighbor table; unused slots contain -1. */
  neighborCell: Int32Array;
  /** Coefficients such that grad(q)=sum_j coeff_j*(q_j-q_i). */
  coeffX: Float64Array;
  coeffY: Float64Array;
  coeffZ: Float64Array;
}

export interface PrimitiveGradients {
  /** Packed as [variable][xyz][cell] through one array per component. */
  rhoX: Float64Array;
  rhoY: Float64Array;
  rhoZ: Float64Array;
  uxX: Float64Array;
  uxY: Float64Array;
  uxZ: Float64Array;
  uyX: Float64Array;
  uyY: Float64Array;
  uyZ: Float64Array;
  uzX: Float64Array;
  uzY: Float64Array;
  uzZ: Float64Array;
  pX: Float64Array;
  pY: Float64Array;
  pZ: Float64Array;
}

function horizontalCellOf(cell: number, nz: number): number {
  return Math.floor(cell / nz);
}

function layerOf(cell: number, nz: number): number {
  return cell % nz;
}

/** Exact centroid of an extruded cubed-sphere volume cell. */
export function shellCellCentroid(
  geometry: SphericalShellGeometry,
  horizontalCell: number,
  k: number,
): Vec3 {
  const omega = geometry.horizontal.cellAreaUnit[horizontalCell]!;
  const r0 = geometry.radiusInterface[k]!;
  const r1 = geometry.radiusInterface[k + 1]!;
  const radialMean =
    0.75 * (r1 ** 4 - r0 ** 4) / (r1 ** 3 - r0 ** 3);
  const scale = radialMean / omega;
  return [
    geometry.cellVectorAreaUnit[horizontalCell * 3]! * scale,
    geometry.cellVectorAreaUnit[horizontalCell * 3 + 1]! * scale,
    geometry.cellVectorAreaUnit[horizontalCell * 3 + 2]! * scale,
  ];
}

/** Exact area centroid of a spherical radial face. */
export function radialFaceCentroid(
  geometry: SphericalShellGeometry,
  horizontalCell: number,
  kInterface: number,
): Vec3 {
  const omega = geometry.horizontal.cellAreaUnit[horizontalCell]!;
  const r = geometry.radiusInterface[kInterface]!;
  const scale = r / omega;
  return [
    geometry.cellVectorAreaUnit[horizontalCell * 3]! * scale,
    geometry.cellVectorAreaUnit[horizontalCell * 3 + 1]! * scale,
    geometry.cellVectorAreaUnit[horizontalCell * 3 + 2]! * scale,
  ];
}

/** Exact area centroid of a conical side face. */
export function sideFaceCentroid(
  geometry: SphericalShellGeometry,
  horizontalEdge: number,
  k: number,
): Vec3 {
  const edge = geometry.horizontal.edges[horizontalEdge]!;
  const r0 = geometry.radiusInterface[k]!;
  const r1 = geometry.radiusInterface[k + 1]!;
  const radialMean =
    (2 / 3) * (r1 ** 3 - r0 ** 3) / (r1 * r1 - r0 * r0);
  const half = 0.5 * edge.angularLength;
  const directionalMean = 2 * Math.sin(half) / edge.angularLength;
  const scale = radialMean * directionalMean;
  return [
    edge.midpoint[0] * scale,
    edge.midpoint[1] * scale,
    edge.midpoint[2] * scale,
  ];
}

function invertSymmetric3(
  a: number,
  b: number,
  c: number,
  d: number,
  e: number,
  f: number,
): readonly [number, number, number, number, number, number] {
  // Matrix is [a d e; d b f; e f c]. Return the same six packed entries.
  const aa = b * c - f * f;
  const bb = a * c - e * e;
  const cc = a * b - d * d;
  const dd = e * f - d * c;
  const ee = d * f - b * e;
  const ff = d * e - a * f;
  const det = a * aa + d * dd + e * ee;
  if (!(Math.abs(det) > 1e-12) || !Number.isFinite(det)) {
    throw new Error(`singular Core v2 reconstruction stencil determinant=${det}`);
  }
  const inv = 1 / det;
  return [aa * inv, bb * inv, cc * inv, dd * inv, ee * inv, ff * inv];
}

function packedCentroid(centroid: Float64Array, cell: number): Vec3 {
  return [centroid[cell * 3]!, centroid[cell * 3 + 1]!, centroid[cell * 3 + 2]!];
}

/**
 * Precompute a compact six-neighbor weighted least-squares stencil.
 * Weight 1/|dx|^2 makes the normal matrix depend on direction rather than the
 * extreme horizontal/vertical aspect ratio. No tunable coefficient is used.
 */
export function buildLinearReconstructionStencil(
  geometry: SphericalShellGeometry,
): LinearReconstructionStencil {
  const horizontal = geometry.horizontal;
  const cellCount = horizontal.cellCount * geometry.nz;
  const cellCentroid = new Float64Array(cellCount * 3);
  const neighborCount = new Uint8Array(cellCount);
  const slotCount = cellCount * MAX_RECONSTRUCTION_NEIGHBORS;
  const neighborCell = new Int32Array(slotCount);
  neighborCell.fill(-1);
  const coeffX = new Float64Array(slotCount);
  const coeffY = new Float64Array(slotCount);
  const coeffZ = new Float64Array(slotCount);

  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const x = shellCellCentroid(geometry, c, k);
      cellCentroid.set(x, q * 3);
    }
  }

  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const q = shellCellIndex(c, k, geometry.nz);
      const neighbors: number[] = [];
      for (let s = 0; s < 4; s++) {
        const edgeId = horizontal.cellEdges[c * 4 + s]!;
        const edge = horizontal.edges[edgeId]!;
        const other = edge.leftCell === c ? edge.rightCell : edge.leftCell;
        neighbors.push(shellCellIndex(other, k, geometry.nz));
      }
      if (k > 0) neighbors.push(shellCellIndex(c, k - 1, geometry.nz));
      if (k + 1 < geometry.nz) neighbors.push(shellCellIndex(c, k + 1, geometry.nz));
      if (neighbors.length > MAX_RECONSTRUCTION_NEIGHBORS) {
        throw new Error('Core v2 reconstruction neighbor count exceeds fixed storage');
      }
      neighborCount[q] = neighbors.length;

      const x0 = packedCentroid(cellCentroid, q);
      let a = 0, b = 0, cc = 0, d = 0, e = 0, f = 0;
      const dirs: Array<readonly [number, number, number, number]> = [];
      for (let j = 0; j < neighbors.length; j++) {
        const nq = neighbors[j]!;
        neighborCell[q * MAX_RECONSTRUCTION_NEIGHBORS + j] = nq;
        const xn = packedCentroid(cellCentroid, nq);
        const dx = xn[0] - x0[0];
        const dy = xn[1] - x0[1];
        const dz = xn[2] - x0[2];
        const distance = Math.hypot(dx, dy, dz);
        if (!(distance > 0) || !Number.isFinite(distance)) {
          throw new Error('invalid Core v2 reconstruction neighbor distance');
        }
        const sx = dx / distance;
        const sy = dy / distance;
        const sz = dz / distance;
        dirs.push([sx, sy, sz, distance]);
        a += sx * sx;
        b += sy * sy;
        cc += sz * sz;
        d += sx * sy;
        e += sx * sz;
        f += sy * sz;
      }

      const inv = invertSymmetric3(a, b, cc, d, e, f);
      const [ia, ib, ic, id, ie, iff] = inv;
      for (let j = 0; j < dirs.length; j++) {
        const [sx, sy, sz, distance] = dirs[j]!;
        const bx = sx / distance;
        const by = sy / distance;
        const bz = sz / distance;
        const slot = q * MAX_RECONSTRUCTION_NEIGHBORS + j;
        coeffX[slot] = ia * bx + id * by + ie * bz;
        coeffY[slot] = id * bx + ib * by + iff * bz;
        coeffZ[slot] = ie * bx + iff * by + ic * bz;
      }
    }
  }

  return {
    geometry,
    cellCentroid,
    neighborCount,
    neighborCell,
    coeffX,
    coeffY,
    coeffZ,
  };
}

function cellPrimitive(
  fields: ConservativeFields,
  cell: number,
  geopotential: ArrayLike<number> | undefined,
): PrimitiveCell {
  return primitiveFromConserved({
    rho: fields.rho[cell]!,
    momentum: [fields.momX[cell]!, fields.momY[cell]!, fields.momZ[cell]!],
    rhoE: fields.rhoE[cell]!,
  }, geopotential === undefined ? 0 : geopotential[cell]!);
}

function variableOf(p: PrimitiveCell, variable: number): number {
  if (variable === 0) return p.rho;
  if (variable === 1) return p.velocity[0];
  if (variable === 2) return p.velocity[1];
  if (variable === 3) return p.velocity[2];
  return p.pressure;
}

function gradientArrays(out: PrimitiveGradients, variable: number): readonly [Float64Array, Float64Array, Float64Array] {
  if (variable === 0) return [out.rhoX, out.rhoY, out.rhoZ];
  if (variable === 1) return [out.uxX, out.uxY, out.uxZ];
  if (variable === 2) return [out.uyX, out.uyY, out.uyZ];
  if (variable === 3) return [out.uzX, out.uzY, out.uzZ];
  return [out.pX, out.pY, out.pZ];
}

function allocateGradients(cellCount: number): PrimitiveGradients {
  const a = (): Float64Array => new Float64Array(cellCount);
  return {
    rhoX:a(),rhoY:a(),rhoZ:a(),
    uxX:a(),uxY:a(),uxZ:a(),
    uyX:a(),uyY:a(),uyZ:a(),
    uzX:a(),uzY:a(),uzZ:a(),
    pX:a(),pY:a(),pZ:a(),
  };
}

function faceCentroidsOfCell(
  stencil: LinearReconstructionStencil,
  cell: number,
): Vec3[] {
  const geometry = stencil.geometry;
  const c = horizontalCellOf(cell, geometry.nz);
  const k = layerOf(cell, geometry.nz);
  const out: Vec3[] = [];
  for (let s = 0; s < 4; s++) {
    const edgeId = geometry.horizontal.cellEdges[c * 4 + s]!;
    out.push(sideFaceCentroid(geometry, edgeId, k));
  }
  out.push(radialFaceCentroid(geometry, c, k));
  out.push(radialFaceCentroid(geometry, c, k + 1));
  return out;
}

/**
 * Build parameter-free Barth-Jespersen-limited primitive gradients.
 * Reconstructed density and pressure remain within neighboring center extrema,
 * so positive cell-center rho/p cannot be made negative at a face by the linear
 * reconstruction itself.
 */
export function computeLimitedPrimitiveGradients(
  fields: ConservativeFields,
  stencil: LinearReconstructionStencil,
  geopotential?: ArrayLike<number>,
): PrimitiveGradients {
  const cellCount = cellCountOf(fields);
  if (cellCount !== stencil.neighborCount.length) {
    throw new Error('Core v2 reconstruction field/stencil cell counts differ');
  }
  if (geopotential !== undefined && geopotential.length !== cellCount) {
    throw new Error('Core v2 reconstruction geopotential length differs from cells');
  }

  const primitive: PrimitiveCell[] = new Array(cellCount);
  for (let q = 0; q < cellCount; q++) primitive[q] = cellPrimitive(fields, q, geopotential);
  const out = allocateGradients(cellCount);

  for (let q = 0; q < cellCount; q++) {
    const count = stencil.neighborCount[q]!;
    for (let variable = 0; variable < 5; variable++) {
      const q0 = variableOf(primitive[q]!, variable);
      let gx = 0, gy = 0, gz = 0;
      let qMin = q0, qMax = q0;
      for (let j = 0; j < count; j++) {
        const slot = q * MAX_RECONSTRUCTION_NEIGHBORS + j;
        const nq = stencil.neighborCell[slot]!;
        const qn = variableOf(primitive[nq]!, variable);
        const delta = qn - q0;
        gx += stencil.coeffX[slot]! * delta;
        gy += stencil.coeffY[slot]! * delta;
        gz += stencil.coeffZ[slot]! * delta;
        qMin = Math.min(qMin, qn);
        qMax = Math.max(qMax, qn);
      }

      const x0 = packedCentroid(stencil.cellCentroid, q);
      let limiter = 1;
      for (const xf of faceCentroidsOfCell(stencil, q)) {
        const delta = gx * (xf[0] - x0[0]) + gy * (xf[1] - x0[1]) + gz * (xf[2] - x0[2]);
        if (delta > 0) limiter = Math.min(limiter, (qMax - q0) / delta);
        else if (delta < 0) limiter = Math.min(limiter, (qMin - q0) / delta);
      }
      limiter = Math.max(0, Math.min(1, limiter));
      const [ox, oy, oz] = gradientArrays(out, variable);
      ox[q] = limiter * gx;
      oy[q] = limiter * gy;
      oz[q] = limiter * gz;
    }
  }

  return out;
}

export function reconstructPrimitiveAt(
  fields: ConservativeFields,
  gradients: PrimitiveGradients,
  stencil: LinearReconstructionStencil,
  cell: number,
  position: Vec3,
  geopotential?: ArrayLike<number>,
): PrimitiveCell {
  const p0 = cellPrimitive(fields, cell, geopotential);
  const x0 = packedCentroid(stencil.cellCentroid, cell);
  const dx = position[0] - x0[0];
  const dy = position[1] - x0[1];
  const dz = position[2] - x0[2];
  const apply = (base: number, gx: Float64Array, gy: Float64Array, gz: Float64Array): number =>
    base + gx[cell]! * dx + gy[cell]! * dy + gz[cell]! * dz;
  const rho = apply(p0.rho, gradients.rhoX, gradients.rhoY, gradients.rhoZ);
  const pressure = apply(p0.pressure, gradients.pX, gradients.pY, gradients.pZ);
  if (!(rho > 0) || !(pressure > 0) || !Number.isFinite(rho) || !Number.isFinite(pressure)) {
    throw new Error(`Core v2 limited reconstruction produced invalid rho/p at cell ${cell}`);
  }
  return {
    rho,
    velocity: [
      apply(p0.velocity[0], gradients.uxX, gradients.uxY, gradients.uxZ),
      apply(p0.velocity[1], gradients.uyX, gradients.uyY, gradients.uyZ),
      apply(p0.velocity[2], gradients.uzX, gradients.uzY, gradients.uzZ),
    ],
    pressure,
  };
}
