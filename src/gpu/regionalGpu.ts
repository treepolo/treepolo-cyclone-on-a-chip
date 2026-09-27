// WebGPU (f32) version of the regional compressible non-hydrostatic model, mirroring
// src/regional/{core,kessler,physics}.ts. All prognostic fields live in one buffer (field-major,
// same halo layout as the CPU model), the RK3 time-level-n copy in a second and tendencies in a third.
//
// Field slots: 0 u, 1 v, 2 w, 3 theta, 4 pi', 5 qv, 6 qc, 7 qr.

import { RegionalModel, H } from '../regional/core.js';
import { RegionalPhysicsConfig } from '../regional/physics.js';

const WG = 64;
const NF = 8;

/** Base-state accessors (only for shaders that bind `base`). */
const BASE_FNS = `
fn bth0(k: u32) -> f32 { return base[k]; }
fn bpi0(k: u32) -> f32 { return base[L + k]; }
fn brho0(k: u32) -> f32 { return base[2u * L + k]; }
fn bqv0(k: u32) -> f32 { return base[3u * L + k]; }
fn brho0f(k: u32) -> f32 { return base[4u * L + k]; }
fn bth0f(k: u32) -> f32 { return base[5u * L + k]; }
fn bcfac(k: u32) -> f32 { return base[6u * L + k]; }
fn bthv(k: u32) -> f32 { return base[7u * L + k]; }
fn brc(k: u32) -> f32 { return base[8u * L + k]; }
fn brw(k: u32) -> f32 { return base[9u * L + k]; }
fn bub(k: u32) -> f32 { return base[10u * L + k]; }
fn bvb(k: u32) -> f32 { return base[11u * L + k]; }
fn brtc(k: u32) -> f32 { return base[12u * L + k]; }
fn brtf(k: u32) -> f32 { return base[13u * L + k]; }
`;

export interface GpuRegionalOptions { moist: boolean; physics: RegionalPhysicsConfig | null }

export class GpuRegional {
  readonly device: GPUDevice;
  readonly cpu: RegionalModel;
  readonly S: GPUBuffer; readonly S0: GPUBuffer; readonly F: GPUBuffer;
  readonly aux: GPUBuffer;       // [0] ppOld, [1] eddy-viscosity deformation, [2] rain accumulation (2-D, level 0)
  time = 0;
  steps = 0;
  private readonly params: GPUBuffer[];
  private readonly passes: ((p: GPUComputePassEncoder) => void)[];

  constructor(device: GPUDevice, m: RegionalModel, opts: GpuRegionalOptions) {
    this.device = device;
    this.cpu = m;
    const { nx, ny, nz, dx, dy, dz, f, beta, divDamp, nsound, dt } = m.c;
    const size = m.size;
    const buf = (bytes: number): GPUBuffer => device.createBuffer({ size: Math.max(16, bytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.S = buf(NF * size * 4); this.S0 = buf(NF * size * 4); this.F = buf(NF * size * 4);
    this.aux = buf(3 * size * 4);
    // base-state table: per level (nz+1 entries each)
    const L = nz + 1, cp = 1004.5, rd = 287.05, cv = cp - rd;
    const base = new Float32Array(14 * L);
    const zTop = nz * dz, z0 = zTop - m.c.dampDepth;
    for (let k = 0; k <= nz; k++) {
      const kc = Math.min(k, nz - 1);
      base[0 * L + k] = m.th0[kc]!; base[1 * L + k] = m.pi0[kc]!; base[2 * L + k] = m.rho0[kc]!; base[3 * L + k] = m.qv0[kc]!;
      base[4 * L + k] = m.rho0f[k]!; base[5 * L + k] = m.th0f[k]!;
      const thv = m.th0[kc]! * (1 + 0.61 * m.qv0[kc]!);
      base[6 * L + k] = cp / cv * rd * m.pi0[kc]! * thv / (cp * m.rho0[kc]! * thv * thv);   // cfac
      base[7 * L + k] = thv;                                                               // thv0 (cells)
      const zc = k < nz ? m.zc[k]! : zTop, zw = m.zf[k]!;
      base[8 * L + k] = m.c.dampDepth > 0 && zc > z0 ? m.c.dampRate * Math.sin(0.5 * Math.PI * (zc - z0) / m.c.dampDepth) ** 2 : 0;
      base[9 * L + k] = m.c.dampDepth > 0 && zw > z0 ? m.c.dampRate * Math.sin(0.5 * Math.PI * (zw - z0) / m.c.dampDepth) ** 2 : 0;
      base[10 * L + k] = m.ub[kc]!; base[11 * L + k] = m.vb[kc]!;
      base[12 * L + k] = m.rho0[kc]! * thv;                                                // rho0 thv0 (cells)
      base[13 * L + k] = m.rho0f[k]! * m.th0f[k]!;                                         // rho0 thv0 at w levels
    }
    const baseBuf = device.createBuffer({ size: base.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(baseBuf, 0, base);
    const ph = opts.physics;
    const consts = `
const NX: u32 = ${nx}u; const NY: u32 = ${ny}u; const NZ: u32 = ${nz}u; const HH: u32 = ${H}u;
const SX: u32 = ${m.sx}u; const SY: u32 = ${m.sy}u; const PL: u32 = ${m.plane}u; const SIZE: u32 = ${size}u; const L: u32 = ${L}u;
const DX: f32 = ${dx}; const DY: f32 = ${dy}; const DZ: f32 = ${dz}; const FCOR: f32 = ${f};
const CP: f32 = ${cp}; const RD: f32 = ${rd}; const G: f32 = 9.80665; const XLV: f32 = 2.5e6;
const BETA: f32 = ${beta}; const DIVD: f32 = ${divDamp};
const MOIST: bool = ${opts.moist}; const PHYS: bool = ${!!ph};
const LH2: f32 = ${ph ? ph.lh * ph.lh : 0}; const LV2: f32 = ${ph ? ph.lv * ph.lv : 0};
const SST: f32 = ${ph ? ph.sst : 0}; const CK: f32 = ${ph ? ph.ck : 0}; const RADTAU: f32 = ${ph ? ph.radTau : 0}; const RADMAX: f32 = ${ph ? ph.radMax : 0};
struct P { dts: f32, dtStage: f32, dtBig: f32, pad: f32 };
fn ix(i: u32, j: u32, k: u32) -> u32 { return k * PL + (j + HH) * SX + (i + HH); }
fn f5(a0: f32, a1: f32, a2: f32, a3: f32, a4: f32, a5: f32, vel: f32) -> f32 {
  let c = (37.0 * (a2 + a3) - 8.0 * (a1 + a4) + (a0 + a5)) / 60.0;
  let d = (10.0 * (a3 - a2) - 5.0 * (a4 - a1) + (a5 - a0)) / 60.0;
  return select(c + d, c - d, vel >= 0.0);
}
fn f3(a0: f32, a1: f32, a2: f32, a3: f32, vel: f32) -> f32 {
  let c = (7.0 * (a1 + a2) - (a0 + a3)) / 12.0;
  let d = (3.0 * (a2 - a1) - (a3 - a0)) / 12.0;
  return select(c + d, c - d, vel >= 0.0);
}
`;
    const pipe = (code: string): GPUComputePipeline => device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: consts + code }), entryPoint: 'main' } });
    const bg = (p: GPUComputePipeline, bufs: GPUBuffer[]): GPUBindGroup => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const disp = (p: GPUComputePipeline, g: GPUBindGroup, n: number) => (pass: GPUComputePassEncoder): void => { pass.setPipeline(p); pass.setBindGroup(0, g); pass.dispatchWorkgroups(Math.ceil(n / WG)); };

    const pHalo = pipe(HALO_WGSL), pMom = pipe(MOM_WGSL), pSca = pipe(SCALAR_WGSL), pStage = pipe(STAGE_WGSL);
    const pAh = pipe(ACOUSTIC_H_WGSL), pAv = pipe(ACOUSTIC_V_WGSL), pCopyPP = pipe(COPYPP_WGSL), pSave = pipe(SAVE_WGSL);
    const pKes = pipe(KESSLER_WGSL);
    const pTurbK = ph ? pipe(TURBK_WGSL) : null, pTurb = ph ? pipe(TURB_WGSL) : null, pSfc = ph && ph.sst > 0 ? pipe(SURFACE_WGSL) : null;
    // uniform params per stage: [dts, dtStage, dtBig]
    const stages: [number, number][] = [[dt / 3, Math.max(1, Math.round(nsound / 3))], [dt / 2, Math.max(1, Math.round(nsound / 2))], [dt, nsound]];
    this.params = stages.map(([dts, ns]) => {
      const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, new Float32Array([dts / ns, dts, dt, 0]));
      return b;
    });
    const haloParams = (f0: number, nf: number, nk: number): GPUBuffer => {
      const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, new Uint32Array([f0, nf, nk, 0]));
      return b;
    };
    const haloN = (nf: number, nk: number): number => m.sx * m.sy * nk * nf;
    const haloAll = disp(pHalo, bg(pHalo, [this.S, haloParams(0, NF, nz + 1)]), haloN(NF, nz + 1));
    const haloUV = disp(pHalo, bg(pHalo, [this.S, haloParams(0, 2, nz)]), haloN(2, nz));
    const haloPP = disp(pHalo, bg(pHalo, [this.S, haloParams(4, 1, nz)]), haloN(1, nz));
    const haloThQ = disp(pHalo, bg(pHalo, [this.S, haloParams(3, 5, nz)]), haloN(5, nz));
    const haloPPold = disp(pHalo, bg(pHalo, [this.aux, haloParams(0, 1, nz)]), haloN(1, nz));
    const haloK = disp(pHalo, bg(pHalo, [this.aux, haloParams(1, 1, nz)]), haloN(1, nz));
    const nInt = nx * ny * nz, nInt1 = nx * ny * (nz + 1), nCol = nx * ny;
    const save = disp(pSave, bg(pSave, [this.S, this.S0]), NF * size);
    const kes = disp(pKes, bg(pKes, [this.S, baseBuf, this.aux, this.params[2]!]), nCol);
    this.passes = [];
    const seq: ((p: GPUComputePassEncoder) => void)[] = [save];
    stages.forEach(([, ns], s) => {
      const prm = this.params[s]!;
      seq.push(haloAll);
      seq.push(disp(pMom, bg(pMom, [this.S, this.F, baseBuf]), nInt1));
      seq.push(disp(pSca, bg(pSca, [this.S, this.F, baseBuf]), nInt));
      if (ph) {
        seq.push(disp(pTurbK!, bg(pTurbK!, [this.S, this.aux]), nInt));
        seq.push(haloK);
        seq.push(disp(pTurb!, bg(pTurb!, [this.S, this.F, this.aux, baseBuf]), nInt1));
        if (pSfc) seq.push(disp(pSfc, bg(pSfc, [this.S, this.F]), nCol));
      }
      seq.push(disp(pStage, bg(pStage, [this.S, this.S0, this.F, this.aux, prm]), size));
      seq.push(haloThQ);
      const ah = disp(pAh, bg(pAh, [this.S, this.F, this.aux, prm]), nInt);
      const av = disp(pAv, bg(pAv, [this.S, this.F, baseBuf, prm]), nCol);
      const cp2 = disp(pCopyPP, bg(pCopyPP, [this.S, this.aux]), size);
      for (let i = 0; i < ns; i++) seq.push(haloPP, haloPPold, ah, haloUV, cp2, av);
    });
    if (opts.moist) seq.push(kes);
    this.passes = seq;
  }

  /** Copy the CPU model state to the GPU. */
  uploadFrom(m: RegionalModel): void {
    const size = m.size, d = new Float32Array(NF * size);
    const fields = [m.u, m.v, m.w, m.th, m.pp, m.scalars[0], m.scalars[1], m.scalars[2]];
    fields.forEach((a, f) => { if (a) d.set(Float32Array.from(a), f * size); });
    this.device.queue.writeBuffer(this.S, 0, d);
    this.device.queue.writeBuffer(this.aux, 0, new Float32Array(3 * size));
    this.time = m.time; this.steps = m.steps;
  }

  step(n = 1): void {
    for (let s = 0; s < n; s++) {
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (const p of this.passes) p(pass);
      pass.end();
      this.device.queue.submit([enc.finish()]);
      this.time += this.cpu.c.dt;
      this.steps++;
    }
  }

  async readState(): Promise<Float32Array> {
    const bytes = NF * this.cpu.size * 4;
    const st = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.S, 0, st, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(st.getMappedRange().slice(0));
    st.unmap(); st.destroy();
    return out;
  }

  async readBuffer(b: GPUBuffer, bytes: number): Promise<Float32Array> {
    const st = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(b, 0, st, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(st.getMappedRange().slice(0));
    st.unmap(); st.destroy();
    return out;
  }

  /** Run only the first-stage slow tendencies (debug). */
  debugSlow(n: number): void {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (let i = 0; i < n; i++) this.passes[i]!(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  async readRain(): Promise<Float32Array> {
    const size = this.cpu.size, bytes = size * 4;
    const st = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.aux, 2 * bytes, st, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(st.getMappedRange().slice(0));
    st.unmap(); st.destroy();
    return out;
  }
}

// ------------------------------------------------------------------------------------------------

// periodic halo fill for fields [f0, f0+nf) of buffer A (field stride SIZE), levels [0, nk)
const HALO_WGSL = /* wgsl */`
struct HP { f0: u32, nf: u32, nk: u32, pad: u32 };
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<uniform> hp: HP;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let per = SX * SY * hp.nk;
  if (t >= per * hp.nf) { return; }
  let f = hp.f0 + t / per;
  let r = t % per;
  let k = r / (SX * SY);
  let e = r % (SX * SY);
  let je = e / SX; let ie = e % SX;
  let inI = ie >= HH && ie < NX + HH;
  let inJ = je >= HH && je < NY + HH;
  if (inI && inJ) { return; }
  let si = (ie + NX - HH) % NX + HH;
  let sj = (je + NY - HH) % NY + HH;
  A[f * SIZE + k * PL + je * SX + ie] = A[f * SIZE + k * PL + sj * SX + si];
}
`;

const SAVE_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> S0: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { if (gid.x < 8u * SIZE) { S0[gid.x] = S[gid.x]; } }
`;

// momentum slow tendencies: advection, Coriolis, buoyancy, Rayleigh damping (overwrites F[u,v,w])
const MOM_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> F: array<f32>;
@group(0) @binding(2) var<storage, read> base: array<f32>;
${BASE_FNS}
fn U(q: u32) -> f32 { return S[q]; }
fn V(q: u32) -> f32 { return S[SIZE + q]; }
fn W(q: u32) -> f32 { return S[2u * SIZE + q]; }
fn thr(q: u32) -> f32 {
  var f = 1.0;
  if (MOIST) { f += 0.61 * S[5u * SIZE + q] - S[6u * SIZE + q] - S[7u * SIZE + q]; }
  return S[3u * SIZE + q] * f;
}
fn buoy(q: u32, k: u32) -> f32 { return G * (thr(q) - bthv(k)) / bthv(k); }
fn zfaceU(q: u32, kf: u32, vel: f32, off: u32) -> f32 {
  if (kf >= 3u && kf <= NZ - 3u) { return f5(S[off + q - 3u * PL], S[off + q - 2u * PL], S[off + q - PL], S[off + q], S[off + q + PL], S[off + q + 2u * PL], vel); }
  if (kf >= 2u && kf <= NZ - 2u) { return f3(S[off + q - 2u * PL], S[off + q - PL], S[off + q], S[off + q + PL], vel); }
  return 0.5 * (S[off + q - PL] + S[off + q]);
}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY * (NZ + 1u)) { return; }
  let k = t / (NX * NY); let r = t % (NX * NY); let j = r / NX; let i = r % NX;
  let q = ix(i, j, k);
  if (k < NZ) {
    let r0 = brho0(k);
    // ---- u
    {
      let ua = 0.5 * (U(q - 1u) + U(q)); let ub = 0.5 * (U(q) + U(q + 1u));
      let fc0 = ua * f5(U(q - 3u), U(q - 2u), U(q - 1u), U(q), U(q + 1u), U(q + 2u), ua);
      let fc1 = ub * f5(U(q - 2u), U(q - 1u), U(q), U(q + 1u), U(q + 2u), U(q + 3u), ub);
      let va = 0.5 * (V(q - 1u) + V(q)); let vb = 0.5 * (V(q - 1u + SX) + V(q + SX));
      let gc0 = va * f5(U(q - 3u * SX), U(q - 2u * SX), U(q - SX), U(q), U(q + SX), U(q + 2u * SX), va);
      let gc1 = vb * f5(U(q - 2u * SX), U(q - SX), U(q), U(q + SX), U(q + 2u * SX), U(q + 3u * SX), vb);
      let wa = 0.5 * (W(q - 1u) + W(q)) * brho0f(k);
      let wb = 0.5 * (W(q - 1u + PL) + W(q + PL)) * brho0f(k + 1u);
      var hc0 = 0.0; var hc1 = 0.0;
      if (k > 0u) { hc0 = wa * zfaceU(q, k, wa, 0u); }
      if (k < NZ - 1u) { hc1 = wb * zfaceU(q + PL, k + 1u, wb, 0u); }
      let dv = (ub - ua) / DX + (vb - va) / DY + (wb - wa) / (r0 * DZ);
      var fu = -((fc1 - fc0) / DX + (gc1 - gc0) / DY + (hc1 - hc0) / (r0 * DZ)) + U(q) * dv;
      fu += FCOR * 0.25 * (V(q) + V(q - 1u) + V(q + SX) + V(q - 1u + SX));
      fu -= brc(k) * (U(q) - bub(k));
      F[q] = fu;
    }
    // ---- v
    {
      let ua = 0.5 * (U(q - SX) + U(q)); let ub = 0.5 * (U(q - SX + 1u) + U(q + 1u));
      let fc0 = ua * f5(V(q - 3u), V(q - 2u), V(q - 1u), V(q), V(q + 1u), V(q + 2u), ua);
      let fc1 = ub * f5(V(q - 2u), V(q - 1u), V(q), V(q + 1u), V(q + 2u), V(q + 3u), ub);
      let va = 0.5 * (V(q - SX) + V(q)); let vb = 0.5 * (V(q) + V(q + SX));
      let gc0 = va * f5(V(q - 3u * SX), V(q - 2u * SX), V(q - SX), V(q), V(q + SX), V(q + 2u * SX), va);
      let gc1 = vb * f5(V(q - 2u * SX), V(q - SX), V(q), V(q + SX), V(q + 2u * SX), V(q + 3u * SX), vb);
      let wa = 0.5 * (W(q - SX) + W(q)) * brho0f(k);
      let wb = 0.5 * (W(q - SX + PL) + W(q + PL)) * brho0f(k + 1u);
      var hc0 = 0.0; var hc1 = 0.0;
      if (k > 0u) { hc0 = wa * zfaceU(q, k, wa, SIZE); }
      if (k < NZ - 1u) { hc1 = wb * zfaceU(q + PL, k + 1u, wb, SIZE); }
      let dv = (ub - ua) / DX + (vb - va) / DY + (wb - wa) / (r0 * DZ);
      var fv = -((fc1 - fc0) / DX + (gc1 - gc0) / DY + (hc1 - hc0) / (r0 * DZ)) + V(q) * dv;
      fv -= FCOR * 0.25 * (U(q) + U(q + 1u) + U(q - SX) + U(q + 1u - SX));
      fv -= brc(k) * (V(q) - bvb(k));
      F[SIZE + q] = fv;
    }
  }
  // ---- w (interior levels)
  if (k >= 1u && k < NZ) {
    let rf = brho0f(k);
    let ua = 0.5 * (U(q - PL) * brho0(k - 1u) + U(q) * brho0(k)) / rf;
    let ub = 0.5 * (U(q - PL + 1u) * brho0(k - 1u) + U(q + 1u) * brho0(k)) / rf;
    let fc0 = ua * f5(W(q - 3u), W(q - 2u), W(q - 1u), W(q), W(q + 1u), W(q + 2u), ua);
    let fc1 = ub * f5(W(q - 2u), W(q - 1u), W(q), W(q + 1u), W(q + 2u), W(q + 3u), ub);
    let va = 0.5 * (V(q - PL) * brho0(k - 1u) + V(q) * brho0(k)) / rf;
    let vb = 0.5 * (V(q - PL + SX) * brho0(k - 1u) + V(q + SX) * brho0(k)) / rf;
    let gc0 = va * f5(W(q - 3u * SX), W(q - 2u * SX), W(q - SX), W(q), W(q + SX), W(q + 2u * SX), va);
    let gc1 = vb * f5(W(q - 2u * SX), W(q - SX), W(q), W(q + SX), W(q + 2u * SX), W(q + 3u * SX), vb);
    let wa = 0.5 * (W(q - PL) * brho0f(k - 1u) + W(q) * rf);
    let wb = 0.5 * (W(q) * rf + W(q + PL) * brho0f(k + 1u));
    var wva: f32; var wvb: f32;
    if (k >= 3u && k <= NZ - 3u) {
      wva = f5(W(q - 3u * PL), W(q - 2u * PL), W(q - PL), W(q), W(q + PL), W(q + 2u * PL), wa);
      wvb = f5(W(q - 2u * PL), W(q - PL), W(q), W(q + PL), W(q + 2u * PL), W(q + 3u * PL), wb);
    } else { wva = 0.5 * (W(q - PL) + W(q)); wvb = 0.5 * (W(q) + W(q + PL)); }
    let dv = (ub - ua) / DX + (vb - va) / DY + (wb - wa) / (rf * DZ);
    var fw = -((fc1 - fc0) / DX + (gc1 - gc0) / DY + (wb * wvb - wa * wva) / (rf * DZ)) + W(q) * dv;
    fw += 0.5 * (buoy(q, k) + buoy(q - PL, k - 1u));
    fw -= brw(k) * W(q);
    F[2u * SIZE + q] = fw;
  } else if (k == 0u || k == NZ) {
    F[2u * SIZE + q] = 0.0;
  }
}
`;

// scalar slow tendencies (theta, pi', qv, qc, qr): flux-form advection + theta damping / radiation (overwrite)
const SCALAR_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> F: array<f32>;
@group(0) @binding(2) var<storage, read> base: array<f32>;
${BASE_FNS}
fn zface(off: u32, qf: u32, kf: u32, vel: f32, c0: f32) -> f32 {
  if (kf >= 3u && kf <= NZ - 3u) { return f5(S[off + qf - 3u * PL] - c0, S[off + qf - 2u * PL] - c0, S[off + qf - PL] - c0, S[off + qf] - c0, S[off + qf + PL] - c0, S[off + qf + 2u * PL] - c0, vel); }
  if (kf >= 2u && kf <= NZ - 2u) { return f3(S[off + qf - 2u * PL] - c0, S[off + qf - PL] - c0, S[off + qf] - c0, S[off + qf + PL] - c0, vel); }
  return 0.5 * (S[off + qf - PL] + S[off + qf]) - c0;
}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY * NZ) { return; }
  let k = t / (NX * NY); let r = t % (NX * NY); let j = r / NX; let i = r % NX;
  let q = ix(i, j, k);
  let r0 = brho0(k);
  let ul = S[q]; let ur = S[q + 1u];
  let vl = S[SIZE + q]; let vr = S[SIZE + q + SX];
  let wb = S[2u * SIZE + q] * brho0f(k); let wt = S[2u * SIZE + q + PL] * brho0f(k + 1u);
  let dv = (ur - ul) / DX + (vr - vl) / DY + (wt - wb) / (r0 * DZ);
  var nf = 2u;
  if (MOIST) { nf = 5u; }
  for (var s = 0u; s < nf; s++) {
    let off = (3u + s) * SIZE;
    // advect the deviation from the cell's own value: exact for this flux form with the divergence
    // correction (constants are advected exactly) and free of f32 cancellation for theta ~ 300 K
    let c0 = S[off + q];
    let fl = ul * f5(S[off + q - 3u] - c0, S[off + q - 2u] - c0, S[off + q - 1u] - c0, 0.0, S[off + q + 1u] - c0, S[off + q + 2u] - c0, ul);
    let fr = ur * f5(S[off + q - 2u] - c0, S[off + q - 1u] - c0, 0.0, S[off + q + 1u] - c0, S[off + q + 2u] - c0, S[off + q + 3u] - c0, ur);
    let gl = vl * f5(S[off + q - 3u * SX] - c0, S[off + q - 2u * SX] - c0, S[off + q - SX] - c0, 0.0, S[off + q + SX] - c0, S[off + q + 2u * SX] - c0, vl);
    let gr = vr * f5(S[off + q - 2u * SX] - c0, S[off + q - SX] - c0, 0.0, S[off + q + SX] - c0, S[off + q + 2u * SX] - c0, S[off + q + 3u * SX] - c0, vr);
    var hb = 0.0; var ht = 0.0;
    if (k > 0u) { hb = wb * zface(off, q, k, S[2u * SIZE + q], c0); }
    if (k < NZ - 1u) { ht = wt * zface(off, q + PL, k + 1u, S[2u * SIZE + q + PL], c0); }
    var tend = -((fr - fl) / DX + (gr - gl) / DY + (ht - hb) / (r0 * DZ));
    if (s == 0u) {
      tend -= brc(k) * (S[off + q] - bth0(k));
      if (PHYS && RADTAU > 0.0) { tend += max(-(S[off + q] - bth0(k)) / RADTAU, -RADMAX / bpi0(k)); }
    }
    F[off + q] = tend;
  }
}
`;

// eddy viscosity deformation |S| sqrt(1 - 3 Ri)  -> aux[1]
const TURBK_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> A: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY * NZ) { return; }
  let k = t / (NX * NY); let r = t % (NX * NY); let j = r / NX; let i = r % NX;
  let q = ix(i, j, k);
  let dudx = (S[q + 1u] - S[q]) / DX; let dvdy = (S[SIZE + q + SX] - S[SIZE + q]) / DY; let dwdz = (S[2u * SIZE + q + PL] - S[2u * SIZE + q]) / DZ;
  let dudy = 0.25 * ((S[q + SX] + S[q + SX + 1u]) - (S[q - SX] + S[q - SX + 1u])) / DY;
  let dvdx = 0.25 * ((S[SIZE + q + 1u] + S[SIZE + q + 1u + SX]) - (S[SIZE + q - 1u] + S[SIZE + q - 1u + SX])) / DX;
  let kp = min(k + 1u, NZ - 1u); let km = select(k - 1u, 0u, k == 0u);
  var dzz = f32(kp - km) * DZ; if (dzz == 0.0) { dzz = DZ; }
  let qp = ix(i, j, kp); let qm = ix(i, j, km);
  let dudz = 0.5 * ((S[qp] + S[qp + 1u]) - (S[qm] + S[qm + 1u])) / dzz;
  let dvdz = 0.5 * ((S[SIZE + qp] + S[SIZE + qp + SX]) - (S[SIZE + qm] + S[SIZE + qm + SX])) / dzz;
  let S2 = 2.0 * (dudx * dudx + dvdy * dvdy + dwdz * dwdz) + (dudy + dvdx) * (dudy + dvdx) + dudz * dudz + dvdz * dvdz;
  let N2 = G * (S[3u * SIZE + qp] - S[3u * SIZE + qm]) / (dzz * S[3u * SIZE + q]);
  let ri = N2 / max(S2, 1e-10);
  A[SIZE + q] = sqrt(S2) * sqrt(max(0.0, 1.0 - 3.0 * ri));
}
`;

// eddy diffusion of u, v, w, theta, moisture (adds to F)
const TURB_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> F: array<f32>;
@group(0) @binding(2) var<storage, read> A: array<f32>;
@group(0) @binding(3) var<storage, read> base: array<f32>;
${BASE_FNS}
fn Kd(q: u32) -> f32 { return A[SIZE + q]; }
fn bval(f: u32, k: u32) -> f32 { if (f == 3u) { return bth0(k); } if (f == 5u) { return bqv0(k); } return 0.0; }
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY * (NZ + 1u)) { return; }
  let k = t / (NX * NY); let r = t % (NX * NY); let j = r / NX; let i = r % NX;
  let q = ix(i, j, k);
  if (k < NZ) {
    for (var f = 0u; f < 8u; f++) {
      if (f == 2u || f == 4u) { continue; }
      if (!MOIST && f >= 5u) { continue; }
      let pr = select(1.0 / 3.0, 1.0, f < 2u);
      let off = f * SIZE;
      let b0 = bval(f, k);
      let ac = S[off + q] - b0;
      let khq = LH2 * Kd(q) / pr;
      let fxr = 0.5 * (khq + LH2 * Kd(q + 1u) / pr) * (S[off + q + 1u] - S[off + q]) / DX;
      let fxl = 0.5 * (khq + LH2 * Kd(q - 1u) / pr) * (S[off + q] - S[off + q - 1u]) / DX;
      let fyr = 0.5 * (khq + LH2 * Kd(q + SX) / pr) * (S[off + q + SX] - S[off + q]) / DY;
      let fyl = 0.5 * (khq + LH2 * Kd(q - SX) / pr) * (S[off + q] - S[off + q - SX]) / DY;
      var fzt = 0.0; var fzb = 0.0;
      if (k < NZ - 1u) { let kv = 0.5 * (Kd(q) + Kd(q + PL)) * LV2 / pr * brho0f(k + 1u); fzt = kv * ((S[off + q + PL] - bval(f, k + 1u)) - ac) / DZ; }
      if (k > 0u) { let kv = 0.5 * (Kd(q) + Kd(q - PL)) * LV2 / pr * brho0f(k); fzb = kv * (ac - (S[off + q - PL] - bval(f, k - 1u))) / DZ; }
      F[off + q] += (fxr - fxl) / DX + (fyr - fyl) / DY + (fzt - fzb) / (brho0(k) * DZ);
    }
  }
  if (k >= 1u && k < NZ) {
    let kc = 0.5 * (Kd(q) + Kd(q - PL));
    let W = 2u * SIZE;
    F[W + q] += LH2 * kc * ((S[W + q + 1u] - 2.0 * S[W + q] + S[W + q - 1u]) / (DX * DX) + (S[W + q + SX] - 2.0 * S[W + q] + S[W + q - SX]) / (DY * DY))
      + LV2 * kc * (S[W + q + PL] - 2.0 * S[W + q] + S[W + q - PL]) / (DZ * DZ);
  }
}
`;

// bulk sea-surface fluxes at the lowest level (adds to F); one thread per column
const SURFACE_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> F: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY || SST <= 0.0) { return; }
  let j = t / NX; let i = t % NX;
  let q = ix(i, j, 0u);
  let ua = 0.5 * (S[q] + S[q + 1u]); let va = 0.5 * (S[SIZE + q] + S[SIZE + q + SX]);
  let spd = max(sqrt(ua * ua + va * va), 1.0);
  let cd = min(2.4e-3, 1.0e-3 * (1.0 + 0.07 * spd));
  let taux = cd * spd * ua; let tauy = cd * spd * va;
  // each u/v face is shared by two columns: apply half of each column's stress to its two faces
  // (atomic-free: every thread only writes its own west/south face with its own and neighbour's share)
  let qw = ix((i + NX - 1u) % NX, j, 0u); let qs = ix(i, (j + NY - 1u) % NY, 0u);
  let uw = 0.5 * (S[qw] + S[qw + 1u]); let vw = 0.5 * (S[SIZE + qw] + S[SIZE + qw + SX]);
  let spw = max(sqrt(uw * uw + vw * vw), 1.0); let cdw = min(2.4e-3, 1.0e-3 * (1.0 + 0.07 * spw));
  let us = 0.5 * (S[qs] + S[qs + 1u]); let vs = 0.5 * (S[SIZE + qs] + S[SIZE + qs + SX]);
  let sps = max(sqrt(us * us + vs * vs), 1.0); let cds = min(2.4e-3, 1.0e-3 * (1.0 + 0.07 * sps));
  F[q] -= 0.5 * (taux + cdw * spw * uw) / DZ;
  F[SIZE + q] -= 0.5 * (tauy + cds * sps * vs) / DZ;
  let esS = 611.2 * exp(17.67 * (SST - 273.15) / (SST - 29.65));
  let qsS = 0.622 * esS / (1.0e5 - 0.378 * esS);
  F[3u * SIZE + q] += CK * spd * (SST - S[3u * SIZE + q]) / DZ;
  if (MOIST) { F[5u * SIZE + q] += CK * spd * (qsS - S[5u * SIZE + q]) / DZ; }
}
`;

// RK3 stage: theta and moisture from time n + dtStage * F; restore u, v, w, pi' to time n; ppOld = pi'(n)
const STAGE_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> S: array<f32>;
@group(0) @binding(1) var<storage, read> S0: array<f32>;
@group(0) @binding(2) var<storage, read> F: array<f32>;
@group(0) @binding(3) var<storage, read_write> A: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x;
  if (q >= SIZE) { return; }
  S[3u * SIZE + q] = S0[3u * SIZE + q] + p.dtStage * F[3u * SIZE + q];
  if (MOIST) { for (var f = 5u; f < 8u; f++) { S[f * SIZE + q] = S0[f * SIZE + q] + p.dtStage * F[f * SIZE + q]; } }
  S[q] = S0[q]; S[SIZE + q] = S0[SIZE + q]; S[2u * SIZE + q] = S0[2u * SIZE + q]; S[4u * SIZE + q] = S0[4u * SIZE + q];
  A[q] = S0[4u * SIZE + q];
}
`;

// acoustic step: horizontal momentum with divergence-damped pressure gradient
const ACOUSTIC_H_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> S: array<f32>;
@group(0) @binding(1) var<storage, read> F: array<f32>;
@group(0) @binding(2) var<storage, read> A: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
fn thr(q: u32) -> f32 {
  var f = 1.0;
  if (MOIST) { f += 0.61 * S[5u * SIZE + q] - S[6u * SIZE + q] - S[7u * SIZE + q]; }
  return S[3u * SIZE + q] * f;
}
fn pstar(q: u32) -> f32 { let pp = S[4u * SIZE + q]; return pp + DIVD * (pp - A[q]); }
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY * NZ) { return; }
  let k = t / (NX * NY); let r = t % (NX * NY); let j = r / NX; let i = r % NX;
  let q = ix(i, j, k);
  let tq = thr(q);
  let ps = pstar(q);
  S[q] += p.dts * (F[q] - CP * 0.5 * (tq + thr(q - 1u)) * (ps - pstar(q - 1u)) / DX);
  S[SIZE + q] += p.dts * (F[SIZE + q] - CP * 0.5 * (tq + thr(q - SX)) * (ps - pstar(q - SX)) / DY);
}
`;

const COPYPP_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> A: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { if (gid.x < SIZE) { A[gid.x] = S[4u * SIZE + gid.x]; } }
`;

// acoustic step: vertically implicit w - pi' per column
const ACOUSTIC_V_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> S: array<f32>;
@group(0) @binding(1) var<storage, read> F: array<f32>;
@group(0) @binding(2) var<storage, read> base: array<f32>;
${BASE_FNS}
@group(0) @binding(3) var<uniform> p: P;
fn thr(q: u32) -> f32 {
  var f = 1.0;
  if (MOIST) { f += 0.61 * S[5u * SIZE + q] - S[6u * SIZE + q] - S[7u * SIZE + q]; }
  return S[3u * SIZE + q] * f;
}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY) { return; }
  let j = t / NX; let i = t % NX;
  let dts = p.dts;
  let bp = 0.5 * (1.0 + BETA); let bm = 0.5 * (1.0 - BETA);
  var a: array<f32, NZ + 1u>; var b: array<f32, NZ + 1u>; var c: array<f32, NZ + 1u>; var rr: array<f32, NZ + 1u>; var wn: array<f32, NZ + 1u>;
  var Ek: array<f32, NZ>;
  for (var k = 0u; k < NZ; k++) {
    let q = ix(i, j, k);
    let dh = (S[q + 1u] - S[q]) / DX + (S[SIZE + q + SX] - S[SIZE + q]) / DY;
    let oldW = bm * (brtf(k + 1u) * S[2u * SIZE + q + PL] - brtf(k) * S[2u * SIZE + q]) / DZ;
    Ek[k] = S[4u * SIZE + q] + dts * F[4u * SIZE + q] - dts * bcfac(k) * (brtc(k) * dh + oldW);
  }
  for (var k = 1u; k < NZ; k++) {
    let q = ix(i, j, k);
    let cth = CP * 0.5 * (thr(q) + thr(q - PL)) / DZ;
    let gk = dts * bcfac(k) * bp / DZ; let gkm = dts * bcfac(k - 1u) * bp / DZ;
    let rtk = brtf(k);
    a[k] = select(0.0, -dts * cth * bp * gkm * brtf(k - 1u), k - 1u >= 1u);
    c[k] = select(0.0, -dts * cth * bp * gk * brtf(k + 1u), k + 1u <= NZ - 1u);
    b[k] = 1.0 + dts * cth * bp * (gk * rtk + gkm * rtk);
    rr[k] = S[2u * SIZE + q] + dts * (F[2u * SIZE + q] - cth * (bp * (Ek[k] - Ek[k - 1u]) + bm * (S[4u * SIZE + q] - S[4u * SIZE + q - PL])));
  }
  for (var k = 2u; k < NZ; k++) { let mm = a[k] / b[k - 1u]; b[k] -= mm * c[k - 1u]; rr[k] -= mm * rr[k - 1u]; }
  wn[0] = 0.0; wn[NZ] = 0.0;
  if (NZ > 1u) { wn[NZ - 1u] = rr[NZ - 1u] / b[NZ - 1u]; }
  for (var kk = i32(NZ) - 2; kk >= 1; kk--) { let k = u32(kk); wn[k] = (rr[k] - c[k] * wn[k + 1u]) / b[k]; }
  for (var k = 0u; k < NZ; k++) {
    let q = ix(i, j, k);
    S[4u * SIZE + q] = Ek[k] - dts * bcfac(k) * bp * (brtf(k + 1u) * wn[k + 1u] - brtf(k) * wn[k]) / DZ;
  }
  for (var k = 0u; k <= NZ; k++) { S[2u * SIZE + ix(i, j, k)] = wn[k]; }
}
`;

// Kessler microphysics per column (clip negatives, sedimentation, warm rain, saturation adjustment)
const KESSLER_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> S: array<f32>;
@group(0) @binding(1) var<storage, read> base: array<f32>;
${BASE_FNS}
@group(0) @binding(2) var<storage, read_write> A: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY) { return; }
  let j = t / NX; let i = t % NX;
  let dt = p.dtBig;
  let rhoS = brho0(0u);
  var col: array<f32, NZ>; var vt: array<f32, NZ>; var fl: array<f32, NZ + 1u>;
  var vmax = 0.0;
  for (var k = 0u; k < NZ; k++) {
    let q = ix(i, j, k);
    for (var f = 5u; f < 8u; f++) { S[f * SIZE + q] = max(S[f * SIZE + q], 0.0); }
    col[k] = S[7u * SIZE + q];
    vt[k] = select(0.0, 36.34 * pow(0.001 * brho0(k) * col[k], 0.1364) * sqrt(rhoS / brho0(k)), col[k] > 1e-12);
    vmax = max(vmax, vt[k]);
  }
  let nsub = max(1u, u32(ceil(vmax * dt / (0.8 * DZ))));
  let dts = dt / f32(nsub);
  let rq = ix(i, j, 0u);
  for (var s = 0u; s < nsub; s++) {
    for (var k = 0u; k < NZ; k++) { fl[k] = brho0(k) * vt[k] * col[k]; }
    fl[NZ] = 0.0;
    A[2u * SIZE + rq] += fl[0] * dts;
    for (var k = 0u; k < NZ; k++) { col[k] += dts * (fl[k + 1u] - fl[k]) / (brho0(k) * DZ); }
  }
  for (var k = 0u; k < NZ; k++) {
    let q = ix(i, j, k);
    var vv = S[5u * SIZE + q]; var cc = S[6u * SIZE + q]; var rr = col[k];
    let r = brho0(k);
    var factorn = 1.0;
    if (rr > 0.0) { factorn = 1.0 / (1.0 + 2.2 * dt * pow(rr, 0.875)); }
    let qrprod = cc - (cc - dt * max(0.001 * (cc - 0.001), 0.0)) * factorn;
    cc = max(cc - qrprod, 0.0);
    rr += qrprod;
    let pi = bpi0(k) + S[4u * SIZE + q];
    let pr = 1.0e5 * pow(pi, CP / RD);
    let T = S[3u * SIZE + q] * pi;
    let qvs = 380.0 / pr * exp(17.27 * (T - 273.15) / (T - 35.86));
    let f5c = 237.3 * 17.27 * XLV / CP;
    let prod = (vv - qvs) / (1.0 + qvs * f5c / ((T - 35.86) * (T - 35.86)));
    let rqq = max(r * rr, 0.0);
    var ern = 0.0;
    if (rqq > 0.0) {
      ern = min(min(dt * (((1.6 + 124.9 * pow(rqq, 0.2046)) * pow(rqq, 0.525)) / (2.55e8 / (pr * qvs) + 5.4e5)) * (max(qvs - vv, 0.0) / (r * qvs)),
                max(-prod - cc, 0.0)), rr);
    }
    let product = max(prod, -cc);
    S[3u * SIZE + q] += XLV / (CP * pi) * (product - ern);
    S[5u * SIZE + q] = max(vv - product + ern, 0.0);
    S[6u * SIZE + q] = cc + product;
    S[7u * SIZE + q] = max(rr - ern, 0.0);
  }
}
`;
