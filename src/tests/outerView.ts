// Tests without time steps: what the 3-D view gets of an outer grid (outerView.ts): the reduction of the volume and the place of the outer grid around the inner domain.
import { buildModel } from '../app/regional/build.js';
import { buildOuter } from '../app/regional/extension.js';
import { decimateVolume, decimationFor, outerFrameOf } from '../app/regional/outerView.js';
import { setupOf, type RegionalSetup } from '../app/regional/setup.js';
import { volumeBytes } from '../regional/display.js';
import { check, summary } from './assert.js';

// the reduction: a uniform volume stays as it is, a lone bright cell keeps its extinction (mean over the block), f = 1 is the volume itself, the last blocks are narrower
{
  const nx = 10, nz = 3, a = new Uint8Array(nx * nx * nz).fill(128);
  const d = decimateVolume(a, nx, nz, 3);
  check('outer view: a uniform volume is reduced to the same bytes (10 -> 4 columns of 3)', d.nx === 4 && d.data.every((v) => v === 128));
  const b = new Uint8Array(nx * nx * nz); b[(1 * nx + 4) * nx + 4] = 255;
  const e = decimateVolume(b, nx, nz, 2);
  const expect = Math.round(255 * Math.cbrt(1 / 4));
  check('outer view: a lone bright cell becomes the cube-root of its extinction shared by the 2 x 2 block', e.nx === 5 && e.data[(1 * 5 + 2) * 5 + 2] === expect && e.data[(0 * 5 + 2) * 5 + 2] === 0, `${e.data[(1 * 5 + 2) * 5 + 2]} vs ${expect}`);
  check('outer view: f = 1 gives the volume itself', decimateVolume(b, nx, nz, 1).data === b);
  check('outer view: the reduction brings a volume under the cap (2000 x 2000 x 60 -> at most 4 M)', (() => { const f = decimationFor(2000, 60, 4e6); return f > 1 && Math.ceil(2000 / f) ** 2 * 60 <= 4e6 && Math.ceil(2000 / (f - 1)) ** 2 * 60 > 4e6; })());
}

// the frame of a real outer grid (built, not stepped): its corner is where the inner box is not, its size and top, the blend band
{
  const S: RegionalSetup = { ...setupOf('tc'), preset: 'custom', L: 150000, dx: 5000, top: 15000, dz: 1000, dt: 0, init: 'vortex', initAmp: 20 };
  const b = buildModel(S, true), inner = b.model, n = 30;
  const acc = { rain: new Float64Array(n * n), snow: new Float64Array(n * n) };
  const out = buildOuter(inner, S, b.frame, 300000, 15000, true, acc, null);
  if (typeof out === 'string') check('outer view: an outer grid can be built', false, out);
  else {
    const o = out.model, { nx: onx, nz: onz } = o.c, g = out.geom;
    // a cloud in the outer model, away from the inner box
    const qc = o.scalars[1]!, ci = 3, cj = 4, ck = 5;
    qc[o.idx(ci, cj, ck)] = 1e-3;
    const cloud = new Uint8Array(onx * onx * onz), rain = new Uint8Array(onx * onx * onz);
    volumeBytes(o, cloud, rain, 0, false);
    const f = outerFrameOf(cloud, rain, onx, onz, o.c.dx, o.c.dz, g.x0!, g.y0!, g.ring ?? 0, inner.c.dx);
    check('outer view: the corner is minus the inner box\'s place, the width and the top are the outer grid\'s', f.x0 === -g.x0! && f.y0 === -g.y0! && f.L === onx * o.c.dx && f.top === onz * o.c.dz && f.f === 1 && f.nx === onx, `${f.x0}, ${f.L}`);
    check('outer view: the blend band is the inner relaxation ring', f.taper === (g.ring ?? 0) * inner.c.dx && f.taper > 0, f.taper);
    const q = (ck * onx + cj) * onx + ci;
    check('outer view: the outer cloud is at its own cell (and the bytes are cloud there only)', f.cloud[q]! > 0 && f.cloud.reduce((s, v) => s + (v > 0 ? 1 : 0), 0) === 1);
    // reduced: the same cloud, in the reduced cell
    const r = outerFrameOf(cloud, rain, onx, onz, o.c.dx, o.c.dz, g.x0!, g.y0!, g.ring ?? 0, inner.c.dx, 20 * 20 * onz / 4);
    check('outer view: reduced by 2 the grid spacing doubles and the place and the width stay', r.f === 2 && r.dx === 2 * o.c.dx && r.nx === 10 && r.L === f.L && r.x0 === f.x0 && r.cloud[(ck * 10 + Math.floor(cj / 2)) * 10 + Math.floor(ci / 2)]! > 0);
    // the corner in the middle: the inner box sits in the middle of the outer grid
    check('outer view: the inner box is in the middle of the outer grid (corner = minus half the difference of the widths)', Math.abs(f.x0 + (f.L - n * inner.c.dx) / 2) < 1e-6, f.x0);
  }
}
summary('outerView');
