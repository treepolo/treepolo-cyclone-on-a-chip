declare const process: { exitCode?: number };

import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import {
  buildSphericalShellGeometry,
  shellCellIndex,
  shellRadialFaceIndex,
  shellSideFaceIndex,
  radialFaceVectorArea,
  sideFaceVectorArea,
} from '../corev2/sphericalShellGeometry.js';
import { norm3 } from '../core/math.js';
import { assert, near, relative } from './assert.js';
import {
  createConservativeFields,
} from '../corev2/fields.js';
import {
  conservedFromPrimitive,
  primitiveFromConserved,
} from '../corev2/state.js';
import {
  accumulateInternalFaceFlux,
  advanceFromIntegratedRate,
  createIntegratedRate,
  integratedTotals,
} from '../corev2/finiteVolume.js';
import { integratedPhysicalEulerFlux, physicalEulerFlux } from '../corev2/eulerFlux.js';
import { applyExactCoriolis } from '../corev2/rotation.js';

interface Test {
  name: string;
  fn: () => void;
}
const tests: Test[] = [];
const test = (name: string, fn: () => void): void => {
  tests.push({ name, fn });
};

test('Core v2 primitive/conservative round trip includes geopotential consistently', () => {
  for (const rho of [0.15, 0.8, 1.2]) {
    for (const pressure of [15000, 60000, 101325]) {
      const primitive = {
        rho,
        velocity: [123, -47, 9] as const,
        pressure,
      };
      const phi = 9.80665 * 12000;
      const conserved = conservedFromPrimitive(primitive, phi);
      const recovered = primitiveFromConserved(conserved, phi);
      relative(recovered.rho, rho, 2e-15, 'rho round trip');
      relative(recovered.pressure, pressure, 2e-14, 'pressure round trip');
      near(recovered.velocity[0], primitive.velocity[0], 2e-14, 'ux round trip');
      near(recovered.velocity[1], primitive.velocity[1], 2e-14, 'uy round trip');
      near(recovered.velocity[2], primitive.velocity[2], 2e-14, 'uz round trip');
    }
  }
});

test('Core v2 physical Euler face flux has exact orientation symmetry', () => {
  const state = conservedFromPrimitive({
    rho: 0.9,
    velocity: [80, -20, 15],
    pressure: 70000,
  });
  const n = [0.6, 0.8, 0] as const;
  const f = physicalEulerFlux(state, n);
  const r = physicalEulerFlux(state, [-n[0], -n[1], -n[2]]);
  near(f.mass + r.mass, 0, 1e-13, 'mass flux orientation');
  near(f.momentum[0] + r.momentum[0], 0, 2e-11, 'mx flux orientation');
  near(f.momentum[1] + r.momentum[1], 0, 2e-11, 'my flux orientation');
  near(f.momentum[2] + r.momentum[2], 0, 2e-11, 'mz flux orientation');
  near(f.totalEnergy + r.totalEnergy, 0, 2e-8, 'energy flux orientation');
});

test('Core v2 resting gas face flux is pressure only', () => {
  const state = conservedFromPrimitive({
    rho: 1.1,
    velocity: [0, 0, 0],
    pressure: 90000,
  });
  const n = [0, 1, 0] as const;
  const f = physicalEulerFlux(state, n);
  near(f.mass, 0, 0, 'rest mass flux');
  near(f.totalEnergy, 0, 0, 'rest energy flux');
  near(f.momentum[0], 0, 0, 'rest mx flux');
  near(f.momentum[1], 90000, 1e-12, 'rest normal pressure flux');
  near(f.momentum[2], 0, 0, 'rest mz flux');
});

test('Core v2 one internal face conserves all five integrated quantities', () => {
  const fields = createConservativeFields(2);
  const volumes = new Float64Array([3.5, 8.25]);

  fields.rho.set([1.0, 0.7]);
  fields.momX.set([2.0, -3.0]);
  fields.momY.set([0.5, 4.0]);
  fields.momZ.set([-1.0, 2.5]);
  fields.rhoE.set([2.5e5, 1.8e5]);

  const before = integratedTotals(fields, volumes);
  const rate = createIntegratedRate(2);
  accumulateInternalFaceFlux(rate, 0, 1, 2.75, {
    mass: 0.31,
    momentum: [120, -45, 9],
    totalEnergy: 1800,
  });
  advanceFromIntegratedRate(fields, rate, volumes, 0.2);
  const after = integratedTotals(fields, volumes);

  near(after.mass, before.mass, 2e-15, 'internal face mass conservation');
  near(after.momentum[0], before.momentum[0], 2e-14, 'internal face mx conservation');
  near(after.momentum[1], before.momentum[1], 2e-14, 'internal face my conservation');
  near(after.momentum[2], before.momentum[2], 2e-14, 'internal face mz conservation');
  near(after.totalEnergy, before.totalEnergy, 2e-10, 'internal face energy conservation');
});

test('Core v2 closed internal-face network cannot change global conserved totals', () => {
  const fields = createConservativeFields(3);
  const volumes = new Float64Array([1.25, 2.5, 5]);
  fields.rho.set([1.0, 0.9, 1.1]);
  fields.momX.set([10, -2, 4]);
  fields.momY.set([3, 8, -6]);
  fields.momZ.set([-1, 2, 7]);
  fields.rhoE.set([3e5, 2e5, 4e5]);

  const before = integratedTotals(fields, volumes);
  const rate = createIntegratedRate(3);
  accumulateInternalFaceFlux(rate, 0, 1, 2, {
    mass: 0.3,
    momentum: [7, 11, -5],
    totalEnergy: 120,
  });
  accumulateInternalFaceFlux(rate, 1, 2, 4, {
    mass: -0.1,
    momentum: [3, -17, 2],
    totalEnergy: -90,
  });
  accumulateInternalFaceFlux(rate, 2, 0, 1.5, {
    mass: 0.07,
    momentum: [-13, 5, 8],
    totalEnergy: 33,
  });
  advanceFromIntegratedRate(fields, rate, volumes, 0.4);
  const after = integratedTotals(fields, volumes);

  near(after.mass, before.mass, 3e-15, 'closed-network mass');
  near(after.momentum[0], before.momentum[0], 3e-14, 'closed-network mx');
  near(after.momentum[1], before.momentum[1], 3e-14, 'closed-network my');
  near(after.momentum[2], before.momentum[2], 3e-14, 'closed-network mz');
  near(after.totalEnergy, before.totalEnergy, 3e-10, 'closed-network energy');
});

test('Core v2 exact Coriolis operator performs zero work', () => {
  const fields = createConservativeFields(1);
  const primitive = {
    rho: 0.82,
    velocity: [145, -88, 37] as const,
    pressure: 51000,
  };
  const state = conservedFromPrimitive(primitive);
  fields.rho[0] = state.rho;
  fields.momX[0] = state.momentum[0];
  fields.momY[0] = state.momentum[1];
  fields.momZ[0] = state.momentum[2];
  fields.rhoE[0] = state.rhoE;

  const p0 = primitiveFromConserved(state).pressure;
  const momentum0 = norm3(state.momentum);
  const energy0 = fields.rhoE[0]!;

  const omega = [0, 0, EARTH.omega] as const;
  const dt = 137;
  for (let i = 0; i < 1000; i++) {
    applyExactCoriolis(fields, omega, dt);
  }

  const momentum1 = norm3([fields.momX[0]!, fields.momY[0]!, fields.momZ[0]!]);
  relative(momentum1, momentum0, 2e-12, 'Coriolis momentum norm');
  near(fields.rhoE[0]!, energy0, 0, 'Coriolis total energy');
  const p1 = primitiveFromConserved({
    rho: fields.rho[0]!,
    momentum: [fields.momX[0]!, fields.momY[0]!, fields.momZ[0]!],
    rhoE: fields.rhoE[0]!,
  }).pressure;
  relative(p1, p0, 2e-12, 'Coriolis pressure');
});

test('Core v2 rejects a conservative state with non-positive internal energy', () => {
  let threw = false;
  try {
    primitiveFromConserved({
      rho: 1,
      momentum: [100, 0, 0],
      rhoE: 1,
    });
  } catch {
    threw = true;
  }
  assert(threw, 'invalid total energy must not silently produce a state');
});

test('Core v2 spherical-shell control volumes close exactly to the analytic shell volume', () => {
  const horizontal = buildCubedSphere(8);
  const r0 = EARTH.radius;
  const radii = new Float64Array([r0, r0 + 12000, r0 + 40000]);
  const geometry = buildSphericalShellGeometry(horizontal, radii);

  let volume = 0;
  for (const v of geometry.cellVolume) volume += v;
  const expected = 4 * Math.PI * (radii[2]! ** 3 - radii[0]! ** 3) / 3;
  relative(volume, expected, 3e-13, 'global spherical-shell volume');

  for (let ki = 0; ki < radii.length; ki++) {
    let area = 0;
    for (let c = 0; c < horizontal.cellCount; c++) {
      area += geometry.radialFaceArea[shellRadialFaceIndex(c, ki, geometry.nz)]!;
    }
    relative(area, 4 * Math.PI * radii[ki]! ** 2, 3e-13, `global radial area k=${ki}`);
  }
});

test('Core v2 spherical-shell side faces are shared geometric objects', () => {
  const horizontal = buildCubedSphere(5);
  const r0 = EARTH.radius;
  const radii = new Float64Array([r0, r0 + 1000, r0 + 3000]);
  const geometry = buildSphericalShellGeometry(horizontal, radii);

  for (let e = 0; e < horizontal.edgeCount; e++) {
    const edge = horizontal.edges[e]!;
    assert(edge.leftCell !== edge.rightCell, 'side face needs two distinct cells');
    const n = edge.normal;
    relative(norm3(n), 1, 3e-15, 'side-face unit normal');
    for (let k = 0; k < geometry.nz; k++) {
      const area = geometry.sideFaceArea[shellSideFaceIndex(e, k, geometry.nz)]!;
      assert(area > 0 && Number.isFinite(area), 'side-face area must be positive finite');
      const left = shellCellIndex(edge.leftCell, k, geometry.nz);
      const right = shellCellIndex(edge.rightCell, k, geometry.nz);
      assert(left !== right, 'extruded side face must separate two 3-D cells');
    }
  }
});

test('Core v2 spherical-shell vector areas close every 3-D control volume', () => {
  const horizontal = buildCubedSphere(7);
  const r0 = EARTH.radius;
  const radii = new Float64Array([r0, r0 + 700, r0 + 2100, r0 + 5000]);
  const geometry = buildSphericalShellGeometry(horizontal, radii);

  let globalAx = 0;
  let globalAy = 0;
  let globalAz = 0;
  for (let c = 0; c < horizontal.cellCount; c++) {
    globalAx += geometry.cellVectorAreaUnit[c * 3]!;
    globalAy += geometry.cellVectorAreaUnit[c * 3 + 1]!;
    globalAz += geometry.cellVectorAreaUnit[c * 3 + 2]!;
  }
  const globalVectorArea = Math.hypot(globalAx, globalAy, globalAz);
  assert(globalVectorArea < 2e-14, `closed unit sphere vector area=${globalVectorArea}`);

  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      const top = radialFaceVectorArea(geometry, c, k + 1);
      const bottom = radialFaceVectorArea(geometry, c, k);
      let sx = top[0] - bottom[0];
      let sy = top[1] - bottom[1];
      let sz = top[2] - bottom[2];
      let scale = norm3(top) + norm3(bottom);

      for (let s = 0; s < 4; s++) {
        const edgeId = horizontal.cellEdges[c * 4 + s]!;
        const sign = horizontal.cellEdgeSigns[c * 4 + s]!;
        const side = sideFaceVectorArea(geometry, edgeId, k);
        sx += sign * side[0];
        sy += sign * side[1];
        sz += sign * side[2];
        scale += norm3(side);
      }

      const residual = Math.hypot(sx, sy, sz) / Math.max(scale, 1);
      assert(residual < 3e-14, `cell ${c} layer ${k} vector-area closure=${residual}`);
    }
  }
});

test('Core v2 exact vector-area geometry preserves any uniform Euler state cell by cell', () => {
  const horizontal = buildCubedSphere(6);
  const r0 = EARTH.radius;
  const radii = new Float64Array([r0, r0 + 900, r0 + 2600]);
  const geometry = buildSphericalShellGeometry(horizontal, radii);
  const state = conservedFromPrimitive({
    rho: 0.95,
    velocity: [73, -21, 14],
    pressure: 76000,
  });

  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let k = 0; k < geometry.nz; k++) {
      let mass = 0;
      let mx = 0;
      let my = 0;
      let mz = 0;
      let energy = 0;
      let scale = 0;

      const addFlux = (vectorArea: readonly [number, number, number]): void => {
        const flux = integratedPhysicalEulerFlux(state, vectorArea);
        mass += flux.mass;
        mx += flux.momentum[0];
        my += flux.momentum[1];
        mz += flux.momentum[2];
        energy += flux.totalEnergy;
        scale +=
          Math.abs(flux.mass) +
          Math.abs(flux.momentum[0]) +
          Math.abs(flux.momentum[1]) +
          Math.abs(flux.momentum[2]) +
          Math.abs(flux.totalEnergy);
      };

      addFlux(radialFaceVectorArea(geometry, c, k + 1));
      const bottom = radialFaceVectorArea(geometry, c, k);
      addFlux([-bottom[0], -bottom[1], -bottom[2]]);

      for (let s = 0; s < 4; s++) {
        const edgeId = horizontal.cellEdges[c * 4 + s]!;
        const sign = horizontal.cellEdgeSigns[c * 4 + s]!;
        const side = sideFaceVectorArea(geometry, edgeId, k);
        addFlux([sign * side[0], sign * side[1], sign * side[2]]);
      }

      const residual =
        (Math.abs(mass) + Math.abs(mx) + Math.abs(my) + Math.abs(mz) + Math.abs(energy)) /
        Math.max(scale, 1);
      assert(residual < 6e-14, `uniform-state finite-volume residual c=${c} k=${k}: ${residual}`);
    }
  }
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
console.log(`${passed}/${tests.length} Core v2 tests passed`);
