// WebGPU coupling of a two-way nest (src/regional/twoway.ts): the child's relaxation targets interpolated from the
// parent (in space and in time between the parent's states at the start and the end of its step) and the feedback of
// the child's cell means into the parent, both on the GPU (no readback), mirroring nestTargets and nestFeedback.

import { H } from '../regional/core.js';
import type { NestGeom } from '../regional/twoway.js';
import { GpuRegional, WG, GX, linearGid } from './regionalGpu.js';

export class GpuNest {
  private readonly device: GPUDevice;
  private readonly tgt: { pipe: GPUComputePipeline; bind: GPUBindGroup; n: number };
  private readonly fb: { pipe: GPUComputePipeline; bind: GPUBindGroup; n: number };
  private readonly prm: GPUBuffer;
  private readonly bases: GPUBuffer;

  constructor(readonly parent: GpuRegional, readonly child: GpuRegional, readonly g: NestGeom) {
    const dev = parent.device, p = parent.cpu, c = child.cpu, pc = p.c, cc = c.c;
    this.device = dev;
    if (!child.B) throw new Error('the child model has no relaxation targets');
    if (parent.nf !== child.nf) throw new Error('parent and child must carry the same fields');
    // base profiles: parent th0, qv0, ub, vb (nz each), then the child's
    const bp = new Float32Array(4 * pc.nz + 4 * cc.nz);
    [p.th0, p.qv0, p.ub, p.vb].forEach((a, f) => bp.set(Float32Array.from(a), f * pc.nz));
    [c.th0, c.qv0, c.ub, c.vb].forEach((a, f) => bp.set(Float32Array.from(a), 4 * pc.nz + f * cc.nz));
    this.bases = dev.createBuffer({ size: bp.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(this.bases, 0, bp);
    this.prm = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const zSponge = cc.nz * cc.dz - cc.dampDepth;
    const consts = `
const HH: u32 = ${H}u;
const PNX: u32 = ${pc.nx}u; const PNY: u32 = ${pc.ny}u; const PNZ: u32 = ${pc.nz}u; const PSX: u32 = ${p.sx}u; const PPL: u32 = ${p.plane}u; const PSIZE: u32 = ${p.size}u;
const PDX: f32 = ${pc.dx}; const PDY: f32 = ${pc.dy}; const PDZ: f32 = ${pc.dz};
const CNX: u32 = ${cc.nx}u; const CNY: u32 = ${cc.ny}u; const CNZ: u32 = ${cc.nz}u; const CSX: u32 = ${c.sx}u; const CPL: u32 = ${c.plane}u; const CSIZE: u32 = ${c.size}u;
const CDX: f32 = ${cc.dx}; const CDZ: f32 = ${cc.dz};
const X0: f32 = ${this.g.i0 * pc.dx}; const Y0: f32 = ${this.g.j0 * pc.dy};
const I0: u32 = ${this.g.i0}u; const J0: u32 = ${this.g.j0}u; const NP: u32 = ${this.g.np}u; const R: u32 = ${this.g.r}u; const RZ: u32 = ${this.g.rz}u;
const CX: f32 = ${this.g.cx}; const CY: f32 = ${this.g.cy}; const RC: f32 = ${this.g.R}; const WF: f32 = ${this.g.Wf};
const CYLR: f32 = ${cc.relaxCyl?.r ?? 0}; const CYL: bool = ${!!cc.relaxCyl}; const ZSP: f32 = ${zSponge}; const SPONGE: bool = ${cc.dampDepth > 0};
const NF: u32 = ${parent.nf}u; const MOIST: bool = ${parent.nq > 0};
fn pix(i: u32, j: u32, k: u32) -> u32 { return k * PPL + (j + HH) * PSX + (i + HH); }
fn cix(i: u32, j: u32, k: u32) -> u32 { return k * CPL + (j + HH) * CSX + (i + HH); }
// base profiles: parent 0 th0, 1 qv0, 2 ub, 3 vb; child 4 .. 7
fn pbase(f: u32, k: u32) -> f32 { return BS[f * PNZ + k]; }
fn cbase(f: u32, k: u32) -> f32 { return BS[4u * PNZ + f * CNZ + k]; }
`;
    const targetsCode = consts + `
@group(0) @binding(0) var<storage, read> PS0: array<f32>;
@group(0) @binding(1) var<storage, read> PS: array<f32>;
@group(0) @binding(2) var<storage, read_write> B: array<f32>;
@group(0) @binding(3) var<storage, read> BS: array<f32>;
@group(0) @binding(4) var<uniform> P: vec4<f32>;
// fractional index along an axis of n points -> lower index and weight (clamped as makeSampler's open axes)
fn ax(p: f32, n: u32) -> vec2<f32> {
  let q = clamp(p, 0.0, f32(n - 1u));
  let i0 = min(floor(q), f32(max(n, 2u) - 2u));
  return vec2<f32>(i0, q - i0);
}
// the parent field f, departure from base profile bf (4: none), at (x, y, z) m with staggering sx, sy, mixed in time
fn samp(f: u32, bf: u32, x: f32, y: f32, z: f32, sx: f32, sy: f32) -> f32 {
  let a = ax(x / PDX - sx, PNX); let b = ax(y / PDY - sy, PNY); let c = ax(z / PDZ - 0.5, PNZ);
  let i0 = u32(a.x); let j0 = u32(b.x); let k0 = u32(c.x);
  var v = 0.0;
  for (var dk = 0u; dk < 2u; dk++) {
    let wk = select(1.0 - c.y, c.y, dk == 1u);
    let k = min(k0 + dk, PNZ - 1u);
    var bs = 0.0;
    if (bf < 4u) { bs = pbase(bf, k); }
    for (var dj = 0u; dj < 2u; dj++) {
      let wj = select(1.0 - b.y, b.y, dj == 1u);
      for (var di = 0u; di < 2u; di++) {
        let wi = select(1.0 - a.y, a.y, di == 1u);
        let q = f * PSIZE + pix(i0 + di, j0 + dj, k);
        let now = PS[q] - bs; let old = PS0[q] - bs;
        v += wk * wj * wi * mix(old, now, P.x);
      }
    }
  }
  return v;
}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= (CNX + 1u) * (CNY + 1u) * CNZ) { return; }
  let k = t / ((CNX + 1u) * (CNY + 1u)); let r = t % ((CNX + 1u) * (CNY + 1u)); let j = r / (CNX + 1u); let i = r % (CNX + 1u);
  let ic = min(i, CNX - 1u); let jc = min(j, CNY - 1u);
  let z = (f32(k) + 0.5) * CDZ;
  // only where the child uses the targets: its relaxation ring and beyond, its sponge layer
  var used = !CYL || (SPONGE && z > ZSP);
  if (!used) { used = length(vec2<f32>(f32(ic) + 0.5 - 0.5 * f32(CNX), f32(jc) + 0.5 - 0.5 * f32(CNY))) > CYLR; }
  if (!used) { return; }
  let q = cix(i, j, k);
  let xf = X0 + f32(i) * CDX; let yf = Y0 + f32(j) * CDX; let xc = X0 + (f32(i) + 0.5) * CDX; let yc = Y0 + (f32(j) + 0.5) * CDX;
  if (j < CNY) { B[q] = cbase(2u, k) + samp(0u, 2u, xf, yc, z, 0.0, 0.5); }
  if (i < CNX) { B[CSIZE + q] = cbase(3u, k) + samp(1u, 3u, xc, yf, z, 0.5, 0.0); }
  if (i < CNX && j < CNY) {
    B[2u * CSIZE + q] = cbase(0u, k) + samp(3u, 0u, xc, yc, z, 0.5, 0.5);
    if (MOIST) { B[3u * CSIZE + q] = max(0.0, cbase(1u, k) + samp(5u, 1u, xc, yc, z, 0.5, 0.5)); }
    B[4u * CSIZE + q] = samp(4u, 4u, xc, yc, z, 0.5, 0.5);
  }
}`;
    const feedbackCode = consts + `
@group(0) @binding(0) var<storage, read> CS: array<f32>;
@group(0) @binding(1) var<storage, read_write> PS: array<f32>;
@group(0) @binding(2) var<storage, read> BS: array<f32>;
const PI: f32 = 3.14159265358979;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NP * NP * PNZ) { return; }
  let k = t / (NP * NP); let r = t % (NP * NP); let b = r / NP; let a = r % NP;
  let ip = I0 + a; let jp = J0 + b;
  let d = length(vec2<f32>((f32(ip) + 0.5) * PDX - CX, (f32(jp) + 0.5) * PDY - CY));
  var wt = 1.0;
  if (d >= RC) { return; }
  if (d > RC - WF) { let c = cos(0.5 * PI * (d - (RC - WF)) / WF); wt = c * c; }
  let ic = a * R; let jc = b * R; let kc = k * RZ; let q = pix(ip, jp, k);
  // u, v over the coincident faces
  var su = 0.0; var sv = 0.0;
  for (var dk = 0u; dk < RZ; dk++) {
    for (var e = 0u; e < R; e++) {
      su += CS[cix(ic, jc + e, kc + dk)] - cbase(2u, kc + dk);
      sv += CS[CSIZE + cix(ic + e, jc, kc + dk)] - cbase(3u, kc + dk);
    }
  }
  PS[q] = mix(PS[q], pbase(2u, k) + su / f32(R * RZ), wt);
  PS[PSIZE + q] = mix(PS[PSIZE + q], pbase(3u, k) + sv / f32(R * RZ), wt);
  // w over the coincident level
  if (k > 0u) {
    var sw = 0.0;
    for (var dj = 0u; dj < R; dj++) { for (var di = 0u; di < R; di++) { sw += CS[2u * CSIZE + cix(ic + di, jc + dj, kc)]; } }
    PS[2u * PSIZE + q] = mix(PS[2u * PSIZE + q], sw / f32(R * R), wt);
  }
  // theta, pi' and the moisture species over the r x r x rz cells
  for (var f = 3u; f < NF; f++) {
    var s = 0.0;
    for (var dk = 0u; dk < RZ; dk++) {
      var bs = 0.0;
      if (f == 3u) { bs = cbase(0u, kc + dk); } else if (f == 5u) { bs = cbase(1u, kc + dk); }
      for (var dj = 0u; dj < R; dj++) { for (var di = 0u; di < R; di++) { s += CS[f * CSIZE + cix(ic + di, jc + dj, kc + dk)] - bs; } }
    }
    var pb = 0.0;
    if (f == 3u) { pb = pbase(0u, k); } else if (f == 5u) { pb = pbase(1u, k); }
    var v = mix(PS[f * PSIZE + q], pb + s / f32(R * R * RZ), wt);
    if (f >= 5u) { v = max(v, 0.0); }
    PS[f * PSIZE + q] = v;
  }
}`;
    const mk = (code: string, bufs: GPUBuffer[], n: number): { pipe: GPUComputePipeline; bind: GPUBindGroup; n: number } => {
      const pipe = dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code: linearGid(code) }), entryPoint: 'main' } });
      const bind = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
      return { pipe, bind, n };
    };
    this.tgt = mk(targetsCode, [parent.S0, parent.S, child.B, this.bases, this.prm], (cc.nx + 1) * (cc.ny + 1) * cc.nz);
    this.fb = mk(feedbackCode, [child.S, parent.S, this.bases], this.g.np * this.g.np * pc.nz);
  }

  private run(k: { pipe: GPUComputePipeline; bind: GPUBindGroup; n: number }): void {
    const enc = this.device.createCommandEncoder(), pass = enc.beginComputePass(), groups = Math.ceil(k.n / WG);
    pass.setPipeline(k.pipe); pass.setBindGroup(0, k.bind); pass.dispatchWorkgroups(Math.min(groups, GX), Math.ceil(groups / GX)); pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /** The child's relaxation targets at fraction alpha of the parent's last step (after parent.step(1)). */
  targets(alpha: number): void {
    this.device.queue.writeBuffer(this.prm, 0, new Float32Array([alpha, 0, 0, 0]));
    this.run(this.tgt);
  }

  /** Feed the child back into the parent (after the child's sub-steps). */
  feedback(): void { this.run(this.fb); }

  /**
   * One parent step with the nest: the parent, then n child sub-steps with targets at the middle of each, then the
   * feedback. The child's time step must be the parent's / n (setDt).
   */
  step(n: number): void {
    this.parent.step(1);
    for (let s = 0; s < n; s++) { this.targets((s + 0.5) / n); this.child.step(1); }
    this.feedback();
  }

  destroy(): void { this.prm.destroy(); this.bases.destroy(); }
}
