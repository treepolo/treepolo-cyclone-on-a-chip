// WebGPU coupling of a tall grid (src/regional/tall.ts) to the model it stands in, mirroring tallTargets and tallFeedback: the child's relaxation targets (host fields interpolated in space
// and in time below the host's top, the child's own environment above it) in its edge ring and Rayleigh layer, and the feedback of the child's cell means into the host over the footprint.
// Also the feedback of an eye nest (a cylinder, twoway.ts) into the tall grid it lies in (GpuCylFeedback), which needs a loop over only the tall grid's lower levels. Both are written
// after nestGpu.ts (which the eye nest uses, and which stays as it is).

import { H } from '../regional/core.js';
import type { NestGeom } from '../regional/twoway.js';
import type { TallGeom } from '../regional/tall.js';
import { GpuRegional, WG, GX, linearGid } from './regionalGpu.js';

type Kernel = { pipe: GPUComputePipeline; bind: GPUBindGroup; n: number };

/** the base profiles of two models in one buffer: parent th0, qv0, ub, vb (nz each), then the child's */
function basesBuffer(dev: GPUDevice, p: GpuRegional, c: GpuRegional): GPUBuffer {
  const pm = p.cpu, cm = c.cpu, pn = pm.c.nz, cn = cm.c.nz, bp = new Float32Array(4 * pn + 4 * cn);
  [pm.th0, pm.qv0, pm.ub, pm.vb].forEach((a, f) => bp.set(Float32Array.from(a), f * pn));
  [cm.th0, cm.qv0, cm.ub, cm.vb].forEach((a, f) => bp.set(Float32Array.from(a), 4 * pn + f * cn));
  const b = dev.createBuffer({ size: bp.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(b, 0, bp);
  return b;
}

/** WGSL constants of a parent / child pair */
function pairConsts(p: GpuRegional, c: GpuRegional): string {
  const pm = p.cpu, cm = c.cpu, pc = pm.c, cc = cm.c;
  return `
const HH: u32 = ${H}u;
const PNX: u32 = ${pc.nx}u; const PNY: u32 = ${pc.ny}u; const PNZ: u32 = ${pc.nz}u; const PSX: u32 = ${pm.sx}u; const PPL: u32 = ${pm.plane}u; const PSIZE: u32 = ${pm.size}u;
const PDX: f32 = ${pc.dx}; const PDY: f32 = ${pc.dy}; const PDZ: f32 = ${pc.dz};
const CNX: u32 = ${cc.nx}u; const CNY: u32 = ${cc.ny}u; const CNZ: u32 = ${cc.nz}u; const CSX: u32 = ${cm.sx}u; const CPL: u32 = ${cm.plane}u; const CSIZE: u32 = ${cm.size}u;
const CDX: f32 = ${cc.dx}; const CDZ: f32 = ${cc.dz};
const NF: u32 = ${p.nf}u; const MOIST: bool = ${p.nq > 0};
fn pix(i: u32, j: u32, k: u32) -> u32 { return k * PPL + (j + HH) * PSX + (i + HH); }
fn cix(i: u32, j: u32, k: u32) -> u32 { return k * CPL + (j + HH) * CSX + (i + HH); }
// base profiles: parent 0 th0, 1 qv0, 2 ub, 3 vb; child 4 .. 7
fn pbase(f: u32, k: u32) -> f32 { return BS[f * PNZ + k]; }
fn cbase(f: u32, k: u32) -> f32 { return BS[4u * PNZ + f * CNZ + k]; }
`;
}

/** WGSL of the feedback: the parent takes the child's cell means over a box of whole parent cells, weighted by `weight` (WGSL statements setting `wt`, or returning), for the parent's lowest NKP levels */
function feedbackCode(consts: string, box: string, weight: string): string {
  return consts + box + `
const NKP: u32 = ${'${NKP}'}u;
@group(0) @binding(0) var<storage, read> CS: array<f32>;
@group(0) @binding(1) var<storage, read_write> PS: array<f32>;
@group(0) @binding(2) var<storage, read> BS: array<f32>;
const PI: f32 = 3.14159265358979;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NP * NP * NKP) { return; }
  let k = t / (NP * NP); let r = t % (NP * NP); let b = r / NP; let a = r % NP;
  let ip = I0 + a; let jp = J0 + b;
  var wt = 1.0;
  ${weight}
  let ic = a * R; let jc = b * R; let kc = k * RZ; let q = pix(ip, jp, k);
  var su = 0.0; var sv = 0.0;
  for (var dk = 0u; dk < RZ; dk++) {
    for (var e = 0u; e < R; e++) {
      su += CS[cix(ic, jc + e, kc + dk)] - cbase(2u, kc + dk);
      sv += CS[CSIZE + cix(ic + e, jc, kc + dk)] - cbase(3u, kc + dk);
    }
  }
  PS[q] = mix(PS[q], pbase(2u, k) + su / f32(R * RZ), wt);
  PS[PSIZE + q] = mix(PS[PSIZE + q], pbase(3u, k) + sv / f32(R * RZ), wt);
  if (k > 0u) {
    var sw = 0.0;
    for (var dj = 0u; dj < R; dj++) { for (var di = 0u; di < R; di++) { sw += CS[2u * CSIZE + cix(ic + di, jc + dj, kc)]; } }
    PS[2u * PSIZE + q] = mix(PS[2u * PSIZE + q], sw / f32(R * R), wt);
  }
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
}

function kernel(dev: GPUDevice, code: string, bufs: GPUBuffer[], n: number): Kernel {
  const pipe = dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code: linearGid(code) }), entryPoint: 'main' } });
  const bind = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  return { pipe, bind, n };
}
function run(dev: GPUDevice, k: Kernel): void {
  const enc = dev.createCommandEncoder(), pass = enc.beginComputePass(), groups = Math.ceil(k.n / WG);
  pass.setPipeline(k.pipe); pass.setBindGroup(0, k.bind); pass.dispatchWorkgroups(Math.min(groups, GX), Math.ceil(groups / GX)); pass.end();
  dev.queue.submit([enc.finish()]);
}

export class GpuTall {
  private readonly device: GPUDevice;
  private readonly tgt: Kernel;
  private readonly fb: Kernel;
  private readonly prm: GPUBuffer;
  private readonly bases: GPUBuffer;
  /** the footprint's lower-left corner in the host's coordinates (m): a footprint that moves (a domain that follows a storm) */
  x0: number; y0: number;

  constructor(readonly parent: GpuRegional, readonly child: GpuRegional, readonly g: TallGeom) {
    const dev = parent.device, c = child.cpu, cc = c.c, pc = parent.cpu.c;
    this.device = dev;
    this.x0 = g.x0; this.y0 = g.y0;
    if (!child.B) throw new Error('the child model has no relaxation targets');
    if (parent.nf !== child.nf) throw new Error('parent and child must carry the same fields');
    this.bases = basesBuffer(dev, parent, child);
    this.prm = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const consts = pairConsts(parent, child);
    const zSponge = cc.nz * cc.dz - cc.dampDepth;
    const targetsCode = consts + `
const RING: i32 = ${g.ring};
const Z1: f32 = ${g.Z1};
const ZSP: f32 = ${zSponge}; const SPONGE: bool = ${cc.dampDepth > 0};
@group(0) @binding(0) var<storage, read> PS0: array<f32>;
@group(0) @binding(1) var<storage, read> PS: array<f32>;
@group(0) @binding(2) var<storage, read_write> B: array<f32>;
@group(0) @binding(3) var<storage, read> BS: array<f32>;
@group(0) @binding(4) var<uniform> P: vec4<f32>;
fn ax(p: f32, n: u32) -> vec2<f32> {
  let q = clamp(p, 0.0, f32(n - 1u));
  let i0 = min(floor(q), f32(max(n, 2u) - 2u));
  return vec2<f32>(i0, q - i0);
}
fn axz(p: f32, n: u32) -> vec2<f32> {
  let q = clamp(p, -0.5, f32(n) - 0.5);
  let i0 = clamp(floor(q), 0.0, f32(max(n, 2u) - 2u));
  return vec2<f32>(i0, q - i0);
}
fn samp(f: u32, bf: u32, x: f32, y: f32, z: f32, sx: f32, sy: f32) -> f32 {
  let a = ax(x / PDX - sx, PNX); let b = ax(y / PDY - sy, PNY); let c = axz(z / PDZ - 0.5, PNZ);
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
  var used = SPONGE && z > ZSP;
  if (!used) { used = min(min(i32(ic), i32(jc)), min(i32(CNX) - 1 - i32(ic), i32(CNY) - 1 - i32(jc))) < RING; }
  if (!used) { return; }
  // the host's departure counts up to half a host level below its top, nothing from half a level above it
  let wh = clamp((Z1 + 0.5 * PDZ - z) / PDZ, 0.0, 1.0);
  let q = cix(i, j, k);
  let xf = P.y + f32(i) * CDX; let yf = P.z + f32(j) * CDX; let xc = P.y + (f32(i) + 0.5) * CDX; let yc = P.z + (f32(j) + 0.5) * CDX;
  if (j < CNY) { var d = 0.0; if (wh > 0.0) { d = wh * samp(0u, 2u, xf, yc, z, 0.0, 0.5); } B[q] = cbase(2u, k) + d; }
  if (i < CNX) { var d = 0.0; if (wh > 0.0) { d = wh * samp(1u, 3u, xc, yf, z, 0.5, 0.0); } B[CSIZE + q] = cbase(3u, k) + d; }
  if (i < CNX && j < CNY) {
    var dt = 0.0; var dq = 0.0; var dp = 0.0;
    if (wh > 0.0) { dt = wh * samp(3u, 0u, xc, yc, z, 0.5, 0.5); dp = wh * samp(4u, 4u, xc, yc, z, 0.5, 0.5); if (MOIST) { dq = wh * samp(5u, 1u, xc, yc, z, 0.5, 0.5); } }
    B[2u * CSIZE + q] = cbase(0u, k) + dt;
    if (MOIST) { B[3u * CSIZE + q] = max(0.0, cbase(1u, k) + dq); }
    B[4u * CSIZE + q] = dp;
  }
}`;
    const box = `
const I0: u32 = ${g.i0}u; const J0: u32 = ${g.j0}u; const NP: u32 = ${g.np}u; const R: u32 = ${g.r}u; const RZ: u32 = ${g.rz}u; const WF: f32 = ${g.Wf};`;
    const weight = `
  let dxe = min(f32(I0 + NP - ip) - 0.5, f32(ip - I0) + 0.5) * PDX; let dye = min(f32(J0 + NP - jp) - 0.5, f32(jp - J0) + 0.5) * PDY;
  let dist = min(dxe, dye);
  if (dist <= 0.0) { return; }
  if (dist < WF) { let s = sin(0.5 * PI * dist / WF); wt = s * s; }`;
    const fbCode = feedbackCode(consts, box, weight).replace('${NKP}', String(pc.nz));
    this.tgt = kernel(dev, targetsCode, [parent.S0, parent.S, child.B, this.bases, this.prm], (cc.nx + 1) * (cc.ny + 1) * cc.nz);
    this.fb = kernel(dev, fbCode, [child.S, parent.S, this.bases], g.np * g.np * pc.nz);
  }

  /** The child's relaxation targets at fraction alpha of the host's last step (after parent.step(1)). */
  targets(alpha: number): void {
    this.device.queue.writeBuffer(this.prm, 0, new Float32Array([alpha, this.x0, this.y0, 0]));
    run(this.device, this.tgt);
  }
  /** Feed the child back into the host. */
  feedback(): void { run(this.device, this.fb); }
  destroy(): void { this.prm.destroy(); this.bases.destroy(); }
}

/** The feedback of an eye nest (child) into a tall grid (parent) it lies in: a cylinder's weights (as feedbackWeight), over the parent's lowest `nk` levels. `g`: the eye's geometry in the tall grid's cells. */
export class GpuCylFeedback {
  private readonly device: GPUDevice;
  private readonly fb: Kernel;
  private readonly bases: GPUBuffer;
  constructor(readonly parent: GpuRegional, readonly child: GpuRegional, readonly g: NestGeom, nk: number) {
    const dev = parent.device;
    this.device = dev;
    if (parent.nf !== child.nf) throw new Error('parent and child must carry the same fields');
    this.bases = basesBuffer(dev, parent, child);
    const box = `
const I0: u32 = ${g.i0}u; const J0: u32 = ${g.j0}u; const NP: u32 = ${g.np}u; const R: u32 = ${g.r}u; const RZ: u32 = ${g.rz}u;
const CX: f32 = ${g.cx}; const CY: f32 = ${g.cy}; const RC: f32 = ${g.R}; const WF: f32 = ${g.Wf};`;
    const weight = `
  let d = length(vec2<f32>((f32(ip) + 0.5) * PDX - CX, (f32(jp) + 0.5) * PDY - CY));
  if (d >= RC) { return; }
  if (d > RC - WF) { let c = cos(0.5 * PI * (d - (RC - WF)) / WF); wt = c * c; }`;
    this.fb = kernel(dev, feedbackCode(pairConsts(parent, child), box, weight).replace('${NKP}', String(nk)), [child.S, parent.S, this.bases], g.np * g.np * nk);
  }
  feedback(): void { run(this.device, this.fb); }
  destroy(): void { this.bases.destroy(); }
}
