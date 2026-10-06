// WebGPU (f32) version of the regional compressible non-hydrostatic model, mirroring
// src/regional/{core,kessler,physics}.ts. All prognostic fields live in one buffer (field-major,
// same halo layout as the CPU model), the RK3 time-level-n copy in a second and tendencies in a third.
//
// Field slots: 0 u, 1 v, 2 w, 3 theta, 4 pi', then the moisture species: 5 qv, 6 qc, 7 qr and, with ice
// microphysics, 8 qi, 9 qs, 10 qg.

import { RegionalModel, H, BoundaryTargets } from '../regional/core.js';
import { T_FLOOR } from '../core/constants.js';
import { RegionalPhysicsConfig, surfaceState, BL_NOISE_PERIOD, BL_NOISE_DEPTH } from '../regional/physics.js';
import { ICE, LF, gammaFn, KOENIG_A1, KOENIG_A2 } from '../regional/ice.js';
import { COL, WV_PATH, ETOP_DBZ, VIL_ZMAX, level500, levelNear } from '../regional/diagnostics.js';
import { FORCING_TAU, MAX_FORCINGS, forcingTable, type WindForcing } from '../regional/forcing.js';
import { CU_TAU, CU_RH, CU_MIN_DEPTH, CU_DETRAIN, CU_DETRAIN_DEPTH, cumulusScale } from '../regional/cumulus.js';
import { EXT, EXT_MAX, subgridRHc } from '../regional/display.js';
import type { TracerParams } from '../regional/tracers.js';

export const WG = 64;
/** workgroups along x per dispatch row; kernels see gid.x = x + y * GX * WG */
export const GX = 32768;
/** kernel passes per command buffer: about PASS_CELLS / grid size (at least 4) */
const PASS_CELLS = 2e7;
type Pass = ((pass: GPUComputePassEncoder) => void) & { label: string };
/** values per column in the display readback (see readDisplay and columnDiagnostics) */
export { COL } from '../regional/diagnostics.js';
export interface DisplayPlanes { u: Float32Array; v: Float32Array; w: Float32Array; wTop: Float32Array; th: Float32Array; pp: Float32Array; sc: Float32Array[] }
export const linearGid = (code: string): string => code.replace(/fn main\(@builtin\(global_invocation_id\) gid: vec3<u32>\) \{/g,
  `fn main(@builtin(global_invocation_id) gid3: vec3<u32>) { let gid = vec3<u32>(gid3.x + gid3.y * ${GX * WG}u, 0u, 0u);`);

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

/** boundary: relaxation targets for open lateral boundaries (defaults to the CPU model's own targets). */
/** theta_rho = theta (1 + 0.61 qv - sum of condensates in slots 6 .. 5+NQ-1); for shaders that bind S. */
const THR_FNS = `
fn thr(q: u32) -> f32 {
  var f = 1.0;
  if (MOIST) { f += 0.61 * S[5u * SIZE + q]; for (var s = 6u; s < 5u + NQ; s++) { f -= S[s * SIZE + q]; } }
  return S[3u * SIZE + q] * f;
}
`;

export interface GpuRegionalOptions { moist: boolean; physics: RegionalPhysicsConfig | null; boundary?: BoundaryTargets | null; ice?: boolean }

export class GpuRegional {
  readonly device: GPUDevice;
  readonly cpu: RegionalModel;
  readonly S: GPUBuffer; readonly S0: GPUBuffer; readonly F: GPUBuffer;
  readonly aux: GPUBuffer;       // [0] ppOld, [1] eddy-viscosity deformation, [2] precipitation and [3] frozen precipitation accumulation (2-D, level 0)
  B: GPUBuffer | null = null;
  R: GPUBuffer | null = null;
  /** density potential temperature of the current RK stage (acoustic substeps) */
  readonly TR: GPUBuffer;
  /** sub-grid turbulence + surface-flux tendencies of the first RK stage (with physics) */
  FT: GPUBuffer | null = null;
  /** condensate column flags: raw, x-dilated, final (moist runs) */
  CF: GPUBuffer | null = null;    // positive-definite limiter ratios (moisture species)    // open-boundary relaxation targets: u, v, theta, qv, pi'
  /** number of prognostic fields (5 + moisture species) */
  readonly nf: number;
  readonly nq: number;
  time = 0;
  steps = 0;
  /** current time step (s); starts at the configured dt, changed by setDt (adaptive stepping) */
  dt: number;
  private readonly nsound: number;
  private readonly params: GPUBuffer[];
  private readonly consts: string;
  private readonly passes: Pass[];

  constructor(device: GPUDevice, m: RegionalModel, opts: GpuRegionalOptions) {
    this.device = device;
    this.cpu = m;
    const { nx, ny, nz, dx, dy, dz, f, beta, divDamp, nsound, dt } = m.c;
    this.dt = dt; this.nsound = nsound;
    const size = m.size;
    this.nq = opts.moist ? (opts.ice ? 6 : 3) : 0;
    if (this.nq > m.scalars.length) throw new Error(`GPU regional model needs ${this.nq} scalars, CPU model has ${m.scalars.length}`);
    const NF = 5 + this.nq;
    this.nf = NF;
    const buf = (bytes: number): GPUBuffer => device.createBuffer({ size: Math.max(16, bytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.S = buf(NF * size * 4); this.S0 = buf(NF * size * 4); this.F = buf(NF * size * 4);
    this.aux = buf(4 * size * 4);
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
    this.baseBuf = baseBuf;
    device.queue.writeBuffer(baseBuf, 0, base);
    const ph = opts.physics;
    this.blNoise = ph?.blNoise ?? 0;
    this.cuScale = ph?.cumulus ? cumulusScale(dx) : 0;
    const open = m.c.lateral === 'open';
    const sfc = ph ? surfaceState(m, ph) : null;
    const bnd = open ? (opts.boundary ?? m.boundary) : null;
    const consts = `
const NX: u32 = ${nx}u; const NY: u32 = ${ny}u; const NZ: u32 = ${nz}u; const HH: u32 = ${H}u;
const SX: u32 = ${m.sx}u; const SY: u32 = ${m.sy}u; const PL: u32 = ${m.plane}u; const SIZE: u32 = ${size}u; const L: u32 = ${L}u;
const DX: f32 = ${dx}; const DY: f32 = ${dy}; const DZ: f32 = ${dz}; const FCOR: f32 = ${f}; const GEOB: f32 = ${m.c.geostrophic ? 1 : 0};
const CP: f32 = ${cp}; const RD: f32 = ${rd}; const G: f32 = 9.80665; const XLV: f32 = 2.5e6; const TFLOOR: f32 = ${T_FLOOR}.0;
const BETA: f32 = ${beta}; const DIVD: f32 = ${divDamp};
const MOIST: bool = ${opts.moist}; const PHYS: bool = ${!!ph}; const NQ: u32 = ${this.nq}u; const NFLD: u32 = ${NF}u;
const OPEN: bool = ${open}; const NEST: bool = ${!!bnd}; const HASPP: bool = ${!!bnd?.pp};
const NRELAX: u32 = ${m.c.relaxCells ?? 5}u; const RTAU: f32 = ${m.c.relaxTau ?? 300};
const CYL: bool = ${!!m.c.relaxCyl}; const CYLR: f32 = ${m.c.relaxCyl?.r ?? 0}; const CYLW: f32 = ${m.c.relaxCyl?.w ?? 1};
const LH2: f32 = ${ph ? ph.lh * ph.lh : 0}; const LV2: f32 = ${ph ? ph.lv * ph.lv : 0};
const Z0: f32 = ${ph?.z0 ?? 0}; const FRU: f32 = ${ph?.frameVel?.u ?? 0}; const FRV: f32 = ${ph?.frameVel?.v ?? 0}; const PIS: f32 = ${sfc ? sfc.pis : 1}; const PSFC: f32 = ${sfc ? sfc.psfc : 1e5}; const CK: f32 = ${ph ? ph.ck : 0}; const RADTAU: f32 = ${ph ? ph.radTau : 0}; const RADMAX: f32 = ${ph ? ph.radMax : 0};
const VMIN: f32 = ${ph?.vmin ?? 1}; const RADC: f32 = ${ph?.radConst ?? 0}; const GUST: bool = ${!!ph?.gust}; const RHO1: f32 = ${m.rho0[0]};
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
    // 2-D dispatch (at most 65535 workgroups per dimension), folded back to a linear invocation index
    const pipe = (code: string): GPUComputePipeline => device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: consts + linearGid(code) }), entryPoint: 'main' } });
    const bg = (p: GPUComputePipeline, bufs: GPUBuffer[]): GPUBindGroup => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const disp = (p: GPUComputePipeline, g: GPUBindGroup, n: number, label: string): Pass => {
      const groups = Math.ceil(n / WG), gx = Math.min(groups, GX), gy = Math.ceil(groups / GX);
      return Object.assign((pass: GPUComputePassEncoder): void => { pass.setPipeline(p); pass.setBindGroup(0, g); pass.dispatchWorkgroups(gx, gy); }, { label });
    };

    // sub-grid turbulence and surface fluxes: computed in the first RK stage into FT, added in every stage
    if (ph) this.FT = buf(NF * size * 4);
    const addFT = (code: string, afterBinding: string, binding: number, writes: [string, string][]): string => {
      if (!ph) return code;
      let c = code.replace(afterBinding, `${afterBinding}\n@group(0) @binding(${binding}) var<storage, read> FT: array<f32>;`);
      for (const [a, b] of writes) { if (!c.includes(a)) throw new Error(`FT patch: ${a}`); c = c.replace(a, b); }
      return c;
    };
    const momCode = addFT(MOM_WGSL, '@group(0) @binding(2) var<storage, read> base: array<f32>;', 3,
      [['F[q] = fu;', 'F[q] = fu + FT[q];'], ['F[SIZE + q] = fv;', 'F[SIZE + q] = fv + FT[SIZE + q];'], ['F[2u * SIZE + q] = fw;', 'F[2u * SIZE + q] = fw + FT[2u * SIZE + q];']]);
    const moistQ = this.nq > 0;
    // binding of the column flags in the scalar kernel: after S, F, base, (R), (FT)
    const cfB = (pd: boolean): number => (moistQ ? 3 + (pd ? 1 : 0) + (ph ? 1 : 0) : -1);
    const scaCode = (pd: boolean): string => addFT(scalarWgsl(pd, cfB(pd)), pd ? '@group(0) @binding(3) var<storage, read> R: array<f32>;' : '@group(0) @binding(2) var<storage, read> base: array<f32>;', pd ? 4 : 3,
      pd ? [['F[off + q] = -((fR - fL) / DX + (gR - gL) / DY + (hT - hB) / (r0 * DZ)) + S[off + q] * dv;', 'F[off + q] = -((fR - fL) / DX + (gR - gL) / DY + (hT - hB) / (r0 * DZ)) + S[off + q] * dv + FT[off + q];']] : [['F[off + q] = tend;', 'F[off + q] = tend + FT[off + q];']]);
    const pHalo = pipe(HALO_WGSL), pMom = pipe(momCode), pSca = pipe(scaCode(false)), pScaPD = this.nq > 0 ? pipe(scaCode(true)) : null, pPD = this.nq > 0 ? pipe(PDRATIO_WGSL) : null, pStage = pipe(stageWgsl(this.nq > 0));
    const pAh = pipe(ACOUSTIC_H_WGSL), pAv = pipe(ACOUSTIC_V_WGSL), pSave = pipe(SAVE_WGSL), pThr = pipe(THRHO_WGSL);
    this.TR = buf(size * 4);
    const pKes = pipe(KESSLER_WGSL);
    let relax: Pass | null = null;
    if (bnd) {
      const bd = new Float32Array(5 * size);
      [bnd.u, bnd.v, bnd.th, bnd.qv, bnd.pp].forEach((a, f) => { if (a) for (let i = 0; i < size; i++) bd[f * size + i] = a[i]!; });
      this.B = device.createBuffer({ size: bd.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      device.queue.writeBuffer(this.B, 0, bd);
      const pRelax = pipe(RELAX_WGSL);
      relax = disp(pRelax, bg(pRelax, [this.S, this.F, this.B, baseBuf]), nx * ny * nz, 'relax');
    }
    const pTurbK = ph ? pipe(TURBK_WGSL) : null, pTurb = ph ? pipe(TURB_WGSL) : null, pSfc = sfc ? pipe(surfaceWgsl((ph?.z0 ?? 0) > 0)) : null;
    let sfcBuf: GPUBuffer | null = null;
    if (sfc) {
      const sd = new Float32Array(2 * nx * ny); sd.set(sfc.tsk); sd.set(sfc.wet, nx * ny);
      sfcBuf = device.createBuffer({ size: sd.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(sfcBuf, 0, sd);
      this.sfcBuf = sfcBuf;
    }
    // uniform params per stage: [dts, dtStage, dtBig]
    const stages: [number, number][] = [[dt / 3, Math.max(1, Math.round(nsound / 3))], [dt / 2, Math.max(1, Math.round(nsound / 2))], [dt, nsound]];
    this.consts = consts;
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
    const haloN = (nf: number, nk: number): number => (m.sx * m.sy - nx * ny) * nk * nf;
    const haloAll = disp(pHalo, bg(pHalo, [this.S, haloParams(0, NF, nz + 1)]), haloN(NF, nz + 1), 'halo all');
    const haloUV = disp(pHalo, bg(pHalo, [this.S, haloParams(0, 2, nz)]), haloN(2, nz), 'halo u v');
    const haloPP = disp(pHalo, bg(pHalo, [this.S, haloParams(4, 1, nz)]), haloN(1, nz), 'halo pp');
    const haloThQ = disp(pHalo, bg(pHalo, [this.S, haloParams(3, 5, nz)]), haloN(5, nz), 'halo th q');
    const haloPPold = disp(pHalo, bg(pHalo, [this.aux, haloParams(0, 1, nz)]), haloN(1, nz), 'halo pp old');
    const thrho = disp(pThr, bg(pThr, [this.S, this.TR]), nx * ny * nz, 'theta_rho');
    const haloTR = disp(pHalo, bg(pHalo, [this.TR, haloParams(0, 1, nz)]), haloN(1, nz), 'halo theta_rho');
    const haloK = disp(pHalo, bg(pHalo, [this.aux, haloParams(1, 1, nz)]), haloN(1, nz), 'halo K');
    if (this.nq > 0) this.R = buf(this.nq * size * 4);
    const nColF = nx * ny;
    if (moistQ) { this.CF = device.createBuffer({ size: 3 * nColF * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(this.CF, 0, new Uint32Array(3 * nColF).fill(1)); }
    const withCF = (b: GPUBuffer[]): GPUBuffer[] => (this.CF ? [...b, this.CF] : b);
    const pFlag = moistQ ? pipe(COLFLAG_WGSL) : null, pDx = moistQ ? pipe(dilateWgsl('x')) : null, pDy = moistQ ? pipe(dilateWgsl('y')) : null;
    const haloR = this.R ? disp(pHalo, bg(pHalo, [this.R, haloParams(0, this.nq, nz)]), haloN(this.nq, nz), 'halo PD ratio') : null;
    const nInt = nx * ny * nz, nInt1 = nx * ny * (nz + 1), nCol = nx * ny;
    const save = disp(pSave, bg(pSave, [this.S, this.S0]), NF * size, 'save state');
    const pIce = this.nq === 6 ? pipe(iceWgsl()) : null;
    const kes = pIce ? disp(pIce, bg(pIce, [this.S, baseBuf, this.aux, this.params[2]!]), nCol, 'microphysics (ice)') : disp(pKes, bg(pKes, [this.S, baseBuf, this.aux, this.params[2]!]), nCol, 'microphysics (Kessler)');
    this.passes = [];
    const seq: Pass[] = [];
    if (pFlag && pDx && pDy) {
      seq.push(disp(pFlag, bg(pFlag, [this.S, this.CF!]), nColF, 'condensate column flags'));
      seq.push(disp(pDx, bg(pDx, [this.CF!]), nColF, 'condensate flags dilate'), disp(pDy, bg(pDy, [this.CF!]), nColF, 'condensate flags dilate'));
    }
    seq.push(save);
    stages.forEach(([, ns], s) => {
      const prm = this.params[s]!;
      seq.push(haloAll);
      const withFT = (b: GPUBuffer[]): GPUBuffer[] => (this.FT ? [...b, this.FT] : b);
      if (ph && s === 0) {
        seq.push(disp(pTurbK!, bg(pTurbK!, [this.S, this.aux]), nInt, 'turbulence K'));
        seq.push(haloK);
        seq.push(disp(pTurb!, bg(pTurb!, [this.S, this.FT!, this.aux, baseBuf]), nInt1, 'turbulent mixing'));
        if (pSfc) seq.push(disp(pSfc, bg(pSfc, [this.S, this.FT!, sfcBuf!]), nCol, 'surface fluxes'));
      }
      seq.push(disp(pMom, bg(pMom, withFT([this.S, this.F, baseBuf])), nInt1, 'momentum advection'));
      if (s === 2 && pScaPD && pPD) {
        seq.push(disp(pPD, bg(pPD, [this.S, this.S0, this.R!, baseBuf, prm, this.CF!]), nInt, 'PD limiter ratio'));
        seq.push(haloR!);
        seq.push(disp(pScaPD, bg(pScaPD, withCF(withFT([this.S, this.F, baseBuf, this.R!]))), nInt, 'scalar advection'));
      } else seq.push(disp(pSca, bg(pSca, withCF(withFT([this.S, this.F, baseBuf]))), nInt, 'scalar advection'));
      if (relax) seq.push(relax);
      seq.push(disp(pStage, bg(pStage, [...withCF([this.S, this.S0, this.F, this.aux, prm]), baseBuf]), size, 'RK stage'));
      seq.push(haloThQ, thrho, haloTR);
      const ah = disp(pAh, bg(pAh, [this.S, this.F, this.aux, prm, this.TR]), nInt, 'acoustic horizontal');
      const av = disp(pAv, bg(pAv, [this.S, this.F, baseBuf, prm, this.aux, this.TR]), nCol, 'acoustic vertical (implicit)');
      for (let i = 0; i < ns; i++) seq.push(haloPP, haloPPold, ah, haloUV, av);
    });
    if (opts.moist) seq.push(kes);
    this.passes = seq;
  }

  /** Replace the open-boundary relaxation targets (time-dependent one-way nesting). */
  setBoundary(b: BoundaryTargets): void {
    if (!this.B) throw new Error('model was not built with open boundaries');
    const size = this.cpu.size, bd = new Float32Array(5 * size);
    [b.u, b.v, b.th, b.qv, b.pp].forEach((a, f) => { if (a) for (let i = 0; i < size; i++) bd[f * size + i] = a[i]!; });
    this.device.queue.writeBuffer(this.B, 0, bd);
  }

  private sfcBuf: GPUBuffer | null = null;
  /** true when the model has per-column surface fluxes (skin temperature and wetness can be changed) */
  get hasSurface(): boolean { return !!this.sfcBuf; }
  /** Replace the per-column skin temperature (K) and wetness ([j][i]). */
  setSurface(tsk: ArrayLike<number>, wet: ArrayLike<number>): void {
    if (!this.sfcBuf) return;
    const n = this.cpu.c.nx * this.cpu.c.ny, sd = new Float32Array(2 * n);
    sd.set(Array.from(tsk)); sd.set(Array.from(wet), n);
    this.device.queue.writeBuffer(this.sfcBuf, 0, sd);
  }

  /** Copy the CPU model state to the GPU (optionally with surface precipitation accumulations, [j][i]). */
  uploadFrom(m: RegionalModel, acc?: { rain: ArrayLike<number>; snow: ArrayLike<number> }): void {
    const size = m.size, d = new Float32Array(this.nf * size);
    const fields = [m.u, m.v, m.w, m.th, m.pp, ...m.scalars.slice(0, this.nq)];
    fields.forEach((a, f) => { if (a) d.set(Float32Array.from(a), f * size); });
    this.device.queue.writeBuffer(this.S, 0, d);
    const aux = new Float32Array(4 * size);
    if (acc) for (let j = 0; j < m.c.ny; j++) for (let i = 0; i < m.c.nx; i++) {
      const q = m.idx(i, j, 0), c = j * m.c.nx + i;
      aux[2 * size + q] = acc.rain[c]!; aux[3 * size + q] = acc.snow[c]!;
    }
    this.device.queue.writeBuffer(this.aux, 0, aux);
    this.time = m.time; this.steps = m.steps;
  }

  /** Release GPU buffers. */
  destroy(): void {
    for (const b of [this.S, this.S0, this.F, this.aux, this.TR, this.FT, this.CF, this.B, this.R, this.disp?.D, this.disp?.C, this.disp?.LB, this.disp?.MODE, this.cflK?.out, this.rzK?.RP, this.rzK?.O, this.trK?.T, this.trK?.P, this.sfcBuf, this.snap?.S, this.snap?.aux, this.editK?.E, ...this.params]) b?.destroy();
  }

  /** Change the time step (all kernels take it from the per-stage uniforms). */
  setDt(dt: number): void {
    this.dt = dt;
    const ns = this.nsound, stages: [number, number][] = [[dt / 3, Math.max(1, Math.round(ns / 3))], [dt / 2, Math.max(1, Math.round(ns / 2))], [dt, ns]];
    stages.forEach(([dts, n], i) => this.device.queue.writeBuffer(this.params[i]!, 0, new Float32Array([dts / n, dts, dt, 0])));
  }

  /** Largest advective Courant rate max(|u|/dx + |v|/dy + |w|/dz) over the domain (1/s). */
  async maxCourantRate(): Promise<number> { return (await this.maxCourant()).rate; }

  /** The largest advective Courant rate (1/s) and the speed of sound (m/s) of the hottest air, from a per-column GPU
   *  reduction (two values per column read back). Not finite when the state is not. */
  async maxCourant(): Promise<{ rate: number; cmax: number }> {
    const { nx, ny } = this.cpu.c, nCol = nx * ny, dev = this.device;
    if (!this.cflK) {
      const code = this.consts + `
@group(0) @binding(2) var<storage, read> base: array<f32>;
${BASE_FNS}
` + linearGid(`
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY) { return; }
  let j = t / NX; let i = t % NX;
  var r = 0.0; var c2 = 0.0;
  for (var k = 0u; k < NZ; k++) {
    let q = ix(i, j, k);
    let a = max(abs(S[q]), abs(S[q + 1u])) / DX + max(abs(S[SIZE + q]), abs(S[SIZE + q + SX])) / DY + max(abs(S[2u * SIZE + q]), abs(S[2u * SIZE + q + PL])) / DZ;
    // (a value that is not a number makes the whole reading one)
    if (a != a) { r = a; } else { r = max(r, a); }
    let th = S[3u * SIZE + q];
    if (th != th) { c2 = th; } else { c2 = max(c2, th * (bpi0(k) + S[4u * SIZE + q])); }
  }
  O[2u * t] = r;
  O[2u * t + 1u] = sqrt(1.4 * RD * max(c2, 0.0));
}`);
      const pipe = dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code }), entryPoint: 'main' } });
      const out = dev.createBuffer({ size: 2 * nCol * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      this.cflK = { pipe, out, bind: dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.S } }, { binding: 1, resource: { buffer: out } }, { binding: 2, resource: { buffer: this.baseBuf } }] }) };
    }
    const k = this.cflK, st = dev.createBuffer({ size: 2 * nCol * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
    const groups = Math.ceil(nCol / WG);
    pass.setPipeline(k.pipe); pass.setBindGroup(0, k.bind); pass.dispatchWorkgroups(Math.min(groups, GX), Math.ceil(groups / GX)); pass.end();
    enc.copyBufferToBuffer(k.out, 0, st, 0, 2 * nCol * 4);
    dev.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const a = new Float32Array(st.getMappedRange());
    let r = 0, c = 0;
    for (let i = 0; i < nCol; i++) { const x = a[2 * i]!, y = a[2 * i + 1]!; r = Number.isNaN(x) || Number.isNaN(r) ? NaN : Math.max(r, x); c = Number.isNaN(y) || Number.isNaN(c) ? NaN : Math.max(c, y); }
    st.unmap(); st.destroy();
    return { rate: r, cmax: c };
  }
  private cflK: { pipe: GPUComputePipeline; out: GPUBuffer; bind: GPUBindGroup } | null = null;

  /** stochastic boundary-layer perturbations (RegionalPhysicsConfig.blNoise, K) and the last interval applied */
  private readonly blNoise: number;
  private noiseEpoch = -1;
  private noiseK: { pipe: GPUComputePipeline; P: GPUBuffer; bind: GPUBindGroup } | null = null;
  /** Perturb theta below BL_NOISE_DEPTH in every column by blNoise * blNoiseValue(i, j, nx, epoch) (as RegionalPhysics.noise). */
  private noise(epoch: number): void {
    const dev = this.device, m = this.cpu;
    if (!this.noiseK) {
      let nk = 0; while (nk < m.c.nz && m.zc[nk]! < BL_NOISE_DEPTH) nk++;
      const pipe = dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code: this.consts + `
@group(0) @binding(0) var<storage, read_write> S: array<f32>;
@group(0) @binding(1) var<uniform> NP: vec4<u32>;
fn pcg(v: u32) -> u32 { let s = v * 747796405u + 2891336453u; let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY) { return; }
  let j = t / NX; let i = t % NX;
  let d = ${this.blNoise.toFixed(6)} * (2.0 * (f32(pcg((i + NX * j) ^ pcg(NP.x))) / 4294967296.0) - 1.0);
  for (var k = 0u; k < ${nk}u; k++) { let q = 3u * SIZE + k * PL + (j + HH) * SX + (i + HH); S[q] = S[q] + d; }
}` }), entryPoint: 'main' } });
      const P = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.noiseK = { pipe, P, bind: dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.S } }, { binding: 1, resource: { buffer: P } }] }) };
    }
    const k = this.noiseK;
    dev.queue.writeBuffer(k.P, 0, new Uint32Array([epoch >>> 0, 0, 0, 0]));
    const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(k.pipe); pass.setBindGroup(0, k.bind); pass.dispatchWorkgroups(Math.ceil(m.c.nx * m.c.ny / 64)); pass.end();
    dev.queue.submit([enc.finish()]);
  }

  /** scale-aware cumulus parameterization (cumulus.ts): strength (0: off), kernel and per-column state (cloud base and top
   *  level, rain rate mm/h, CAPE) of the last step */
  private readonly cuScale: number;
  private cuK: { pipe: GPUComputePipeline; P: GPUBuffer; bind: GPUBindGroup } | null = null;
  cuBuf: GPUBuffer | null = null;
  /** the per-column cumulus state (kb, kt, rate, CAPE), no convection until the first step (a 16-byte stand-in without the scheme) */
  private ensureCuBuf(): GPUBuffer {
    if (!this.cuBuf) {
      const n = this.cuScale > 0 ? this.cpu.c.nx * this.cpu.c.ny : 1, init = new Float32Array(4 * n);
      for (let c = 0; c < n; c++) { init[4 * c] = -1; init[4 * c + 1] = -1; }
      this.cuBuf = this.device.createBuffer({ size: 16 * n, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(this.cuBuf, 0, init);
    }
    return this.cuBuf;
  }
  private cumulus(): void {
    const dev = this.device, m = this.cpu, { nx, ny, nz, dz } = m.c;
    if (!this.cuK) {
      const nd = Math.max(1, Math.round(CU_DETRAIN_DEPTH / dz));
      const pipe = dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code: this.consts + `
@group(0) @binding(0) var<storage, read_write> S: array<f32>;
@group(0) @binding(1) var<storage, read> LC: array<f32>;      // per level: pi0, rho0
@group(0) @binding(2) var<storage, read_write> CU: array<f32>;  // per column: cloud base, top level, rain rate (mm/h), CAPE
@group(0) @binding(3) var<uniform> CP_: vec4<f32>;             // dt, strength
fn esat(T: f32) -> f32 { return 611.2 * exp(17.67 * (T - 273.15) / (T - 29.65)); }
// cumulusColumn / applyCumulus in src/regional/cumulus.ts
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY) { return; }
  let j = t / NX; let i = t % NX;
  CU[4u * t] = -1.0; CU[4u * t + 1u] = -1.0; CU[4u * t + 2u] = 0.0; CU[4u * t + 3u] = 0.0;
  var T: array<f32, ${nz}>; var p: array<f32, ${nz}>; var qv: array<f32, ${nz}>; var pk_: array<f32, ${nz}>; var Tp: array<f32, ${nz}>;
  for (var k = 0u; k < NZ; k++) {
    let q = ix(i, j, k); let pik = LC[2u * k] + S[4u * SIZE + q];
    pk_[k] = pik; T[k] = S[3u * SIZE + q] * pik; p[k] = 1e5 * pow(pik, CP / RD); qv[k] = S[5u * SIZE + q];
  }
  var pth = T[0] / pk_[0]; var pq = max(qv[0], 0.0); var kb = -1; var lfc = -1; var kt = -1; var cape = 0.0;
  Tp[0] = T[0];
  for (var k = 1u; k < NZ; k++) {
    let pk = p[k]; let pik = pk_[k];
    var tp = pth * pik; var d = 0.0;
    let es0 = esat(tp);
    if (pq > 0.622 * es0 / max(pk - es0, 1.0)) {
      for (var it = 0; it < 4; it++) {
        let tt = tp + XLV * d / CP; let es = esat(tt); let qs = 0.622 * es / max(pk - es, 1.0);
        let dqs = qs * pk / max(pk - es, 1.0) * 17.67 * 243.5 / ((tt - 29.65) * (tt - 29.65));
        d += (pq - d - qs) / (1.0 + XLV / CP * dqs);
      }
      d = clamp(d, 0.0, pq);
      if (kb < 0) { kb = i32(k); }
    }
    tp += XLV * d / CP; pq -= d; pth = tp / pik; Tp[k] = tp;
    let tve = T[k] * (1.0 + 0.61 * max(qv[k], 0.0)); let b = G * (tp * (1.0 + 0.61 * pq) - tve) / tve;
    if (lfc < 0) { if (kb >= 0 && b > 0.0) { lfc = i32(k); cape += b * DZ; kt = i32(k); } }
    else if (kt == i32(k) - 1 && b > 0.0) { cape += b * DZ; kt = i32(k); }
  }
  if (lfc < 0 || cape <= 0.0 || kt <= kb) { return; }
  let a = CP_.x / ${CU_TAU.toFixed(1)};
  var dT: array<f32, ${nz}>; var dq: array<f32, ${nz}>;
  var Pq = 0.0; var kq = -1;
  for (var k = 0; k <= kt; k++) {
    let Tr = Tp[k]; let es = esat(Tr); let qr = ${CU_RH} * 0.622 * es / max(p[k] - es, 1.0);
    dT[k] = -(T[k] - Tr) * a; dq[k] = -(qv[k] - qr) * a;
    Pq -= dq[k] * LC[2u * u32(k) + 1u] * DZ;
    if (Pq >= 0.0) { kq = k; }
  }
  // deep convection, else shallow convection up to the highest level where the vapour loss is still non-negative
  let deep = Pq > 0.0 && f32(kt - kb) * DZ >= ${CU_MIN_DEPTH.toFixed(1)};
  if (!deep) { if (kq <= kb) { return; } kt = kq; }
  var P0 = 0.0; var PT = 0.0; var mass = 0.0;
  for (var k = 0; k <= kt; k++) { let md = LC[2u * u32(k) + 1u] * DZ; P0 -= dq[k] * md; PT += CP * dT[k] * md / XLV; mass += md; }
  if (!(P0 >= 0.0)) { return; }
  let c = (P0 - PT) * XLV / (CP * mass); let sc = CP_.y; let P = sc * P0;
  for (var k = 0; k <= kt; k++) {
    let q = ix(i, j, u32(k));
    S[3u * SIZE + q] = S[3u * SIZE + q] + sc * (dT[k] + c) / pk_[k];
    S[5u * SIZE + q] = max(0.0, S[5u * SIZE + q] + sc * dq[k]);
  }
  CU[4u * t] = f32(kb); CU[4u * t + 1u] = select(f32(-kt - 2), f32(kt), deep); CU[4u * t + 2u] = P / CP_.x * 3600.0; CU[4u * t + 3u] = cape;
  if (!(P > 0.0)) { return; }
  let k0 = max(kb, kt - ${nd - 1});
  var mt = 0.0; for (var k = k0; k <= kt; k++) { mt += LC[2u * u32(k) + 1u] * DZ; }
  for (var k = k0; k <= kt; k++) {
    let q = ix(i, j, u32(k)); let add = ${CU_DETRAIN} * P / mt;
    if (NQ == 6u && T[k] < 273.15) { S[8u * SIZE + q] = S[8u * SIZE + q] + add; } else { S[6u * SIZE + q] = S[6u * SIZE + q] + add; }
  }
  let kr = max(1, kb);
  var mb = 0.0; for (var k = 0; k < kr; k++) { mb += LC[2u * u32(k) + 1u] * DZ; }
  for (var k = 0; k < kr; k++) { let q = ix(i, j, u32(k)); S[7u * SIZE + q] = S[7u * SIZE + q] + (1.0 - ${CU_DETRAIN}) * P / mb; }
}` }), entryPoint: 'main' } });
      const lc = new Float32Array(2 * nz);
      for (let k = 0; k < nz; k++) { lc[2 * k] = m.pi0[k]!; lc[2 * k + 1] = m.rho0[k]!; }
      const LC = dev.createBuffer({ size: lc.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(LC, 0, lc);
      this.ensureCuBuf();
      const P = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.cuK = { pipe, P, bind: dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.S } }, { binding: 1, resource: { buffer: LC } }, { binding: 2, resource: { buffer: this.cuBuf! } }, { binding: 3, resource: { buffer: P } }] }) };
    }
    const k = this.cuK;
    dev.queue.writeBuffer(k.P, 0, new Float32Array([this.dt, this.cuScale, 0, 0]));
    const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(k.pipe); pass.setBindGroup(0, k.bind); pass.dispatchWorkgroups(Math.ceil(nx * ny / 64)); pass.end();
    dev.queue.submit([enc.finish()]);
  }

  /** lasting wind forcings (forcing.ts) applied at the start of every step */
  private nForce = 0;
  private forceK: { pipe: GPUComputePipeline; T: GPUBuffer; FP: GPUBuffer; bind: GPUBindGroup } | null = null;
  /** Set the lasting wind forcings (at most MAX_FORCINGS; an empty list stops them). */
  setForcings(list: readonly WindForcing[]): void {
    this.forcings = list.slice(0, MAX_FORCINGS);
    this.nForce = this.forcings.length;
    if (!this.nForce) return;
    this.ensureForce();
    this.device.queue.writeBuffer(this.forceK!.T, 0, forcingTable(list));
  }
  /** the lasting forcings (the table is borrowed for a one-time wind) */
  private forcings: WindForcing[] = [];
  private ensureForce(): void {
    const dev = this.device, m = this.cpu, { nz } = m.c;
    if (!this.forceK) {
      const pipe = dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code: this.consts + `
@group(0) @binding(0) var<storage, read_write> S: array<f32>;
@group(0) @binding(1) var<uniform> FT: array<vec4<f32>, ${3 * MAX_FORCINGS}>;
@group(0) @binding(2) var<uniform> FP: vec4<f32>;
@group(0) @binding(3) var<storage, read> LV: array<f32>;     // ub[k], vb[k], zc[k] (nz each), zf[k] (nz + 1)
// envelope and target wind of forcing n at a point (forcingAt in src/regional/forcing.ts)
fn tgt(n: u32, px: f32, py: f32, pz: f32) -> vec4<f32> {
  let a = FT[3u * n]; let b = FT[3u * n + 1u]; let c = FT[3u * n + 2u];
  var dx = px - a.x; var dy = py - a.y;
  if (!OPEN) { let lx = f32(NX) * DX; let ly = f32(NY) * DY; dx -= floor(dx / lx + 0.5) * lx; dy -= floor(dy / ly + 0.5) * ly; }
  let r = sqrt(dx * dx + dy * dy); let rh = r / a.w; let rz = abs(pz - a.z) / b.x;
  if (rh >= 1.0 || rz >= 1.0) { return vec4<f32>(0.0); }
  let ch = cos(1.5707963 * rh); let cz = cos(1.5707963 * rz); let e = ch * ch * cz * cz;
  if (b.z < 0.5) { return vec4<f32>(e, b.y * c.x, b.y * c.y, b.y * c.z); }
  if (r < 1e-6) { return vec4<f32>(e, 0.0, 0.0, 0.0); }
  let sp = b.y * sin(3.14159265 * rh); let ux = dx / r; let uy = dy / r;
  if (b.z < 1.5) { return vec4<f32>(e, -b.w * sp * uy, b.w * sp * ux, 0.0); }
  return vec4<f32>(e, -b.w * sp * ux, -b.w * sp * uy, 0.0);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY * (NZ + 1u)) { return; }
  let k = t / (NX * NY); let r2 = t % (NX * NY); let j = r2 / NX; let i = r2 % NX;
  // FP: count, relaxation weight dt / tau, 1 for a one-time wind (add e * target, as applyWind(.., 'once'))
  let q = ix(i, j, k); let nf = u32(FP.x); let al = FP.y; let once = FP.z > 0.5;
  let xs = f32(i) * DX; let ys = f32(j) * DY; let xc = (f32(i) + 0.5) * DX; let yc = (f32(j) + 0.5) * DY;
  for (var n = 0u; n < nf; n++) {
    if (k < NZ) {
      let zc = LV[2u * NZ + k];
      let fu = tgt(n, xs, yc, zc);
      if (fu.x > 0.0) { S[q] = S[q] + select(al * fu.x * (fu.y - (S[q] - LV[k])), fu.x * fu.y, once); }
      let fv = tgt(n, xc, ys, zc);
      if (fv.x > 0.0) { S[SIZE + q] = S[SIZE + q] + select(al * fv.x * (fv.z - (S[SIZE + q] - LV[NZ + k])), fv.x * fv.z, once); }
    }
    if (k > 0u && k < NZ) {
      let fw = tgt(n, xc, yc, LV[3u * NZ + k]);
      if (fw.x > 0.0) { S[2u * SIZE + q] = S[2u * SIZE + q] + select(al * fw.x * (fw.w - S[2u * SIZE + q]), fw.x * fw.w, once); }
    }
  }
}` }), entryPoint: 'main' } });
      const T = dev.createBuffer({ size: 48 * MAX_FORCINGS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const FP = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const lv = new Float32Array(4 * nz + 1);
      for (let k = 0; k < nz; k++) { lv[k] = m.ub[k]!; lv[nz + k] = m.vb[k]!; lv[2 * nz + k] = m.zc[k]!; }
      for (let k = 0; k <= nz; k++) lv[3 * nz + k] = m.zf[k]!;
      const LV = dev.createBuffer({ size: lv.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(LV, 0, lv);
      this.forceK = { pipe, T, FP, bind: dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.S } }, { binding: 1, resource: { buffer: T } }, { binding: 2, resource: { buffer: FP } }, { binding: 3, resource: { buffer: LV } }] }) };
    }
  }
  /** A wind added once (applyWind(.., 'once') on the GPU state; f in this model's coordinates). */
  windOnce(f: WindForcing): void {
    this.ensureForce();
    const k = this.forceK!, dev = this.device, { nx, ny, nz } = this.cpu.c;
    dev.queue.writeBuffer(k.T, 0, forcingTable([f]));
    dev.queue.writeBuffer(k.FP, 0, new Float32Array([1, 0, 1, 0]));
    const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(k.pipe); pass.setBindGroup(0, k.bind); pass.dispatchWorkgroups(Math.ceil(nx * ny * (nz + 1) / 64)); pass.end();
    dev.queue.submit([enc.finish()]);
    dev.queue.writeBuffer(k.T, 0, forcingTable(this.forcings));
  }

  private readonly baseBuf: GPUBuffer;
  private editK: { pipe: GPUComputePipeline; E: GPUBuffer; bind: GPUBindGroup } | null = null;
  /**
   * Change the state in place (the interactions of the regional page; worker.ts does the same to a CPU model): a warm or
   * cold bubble (theta changes by `amp` cos^2 inside the ellipsoid, nothing colder than the temperature `floorT` K), or
   * water vapour multiplied by `fac` in a region (the excess over saturation condenses in the next step). Positions in m;
   * (ox, oy) is this model's origin in those coordinates (an inner grid's corner in the outer grid, else 0).
   */
  edit(e: { kind: 'bubble'; x: number; y: number; z: number; rh: number; rz: number; amp: number; floorT: number; warm: boolean; ox: number; oy: number }
    | { kind: 'moisture'; x: number; y: number; z: number; R: number; H: number; fac: number; ox: number; oy: number }): void {
    const dev = this.device, m = this.cpu, { nx, ny, nz } = m.c;
    if (!this.editK) {
      const code = this.consts + `
@group(0) @binding(0) var<storage, read_write> S: array<f32>;
@group(0) @binding(1) var<storage, read> base: array<f32>;
@group(0) @binding(2) var<uniform> E: array<vec4<f32>, 3>;
@group(0) @binding(3) var<storage, read> LZ: array<f32>;
// E[0]: kind (0 bubble, 1 moisture), centre x, y, z; E[1]: radius, vertical radius / half depth, amplitude or factor,
// cap; E[2]: origin ox, oy, warm
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY * NZ) { return; }
  let k = t / (NX * NY); let r2 = t % (NX * NY); let j = r2 / NX; let i = r2 % NX;
  let q = ix(i, j, k); let a = E[0]; let b = E[1]; let c = E[2];
  var ex = c.x + (f32(i) + 0.5) * DX - a.y; var ey = c.y + (f32(j) + 0.5) * DY - a.z;
  let zc = LZ[k];
  if (a.x < 0.5) {
    let r = sqrt((ex / b.x) * (ex / b.x) + (ey / b.x) * (ey / b.x) + ((zc - a.w) / b.y) * ((zc - a.w) / b.y));
    if (r >= 1.0) { return; }
    let cc = cos(1.5707963 * r); let d = b.z * cc * cc;
    let th = S[3u * SIZE + q]; let pk = max(base[L + k] + S[4u * SIZE + q], 0.05);
    S[3u * SIZE + q] = max(th + d, b.w / pk);
    return;
  }
  if (!OPEN) { let lx = f32(NX) * DX; let ly = f32(NY) * DY; ex -= floor(ex / lx + 0.5) * lx; ey -= floor(ey / ly + 0.5) * ly; }
  let rh = sqrt(ex * ex + ey * ey) / b.x; let rz = abs(zc - a.w) / b.y;
  if (rh >= 1.0 || rz >= 1.0) { return; }
  let ch = cos(1.5707963 * rh); let cz = cos(1.5707963 * rz); let e = ch * ch * cz * cz;
  let qv = S[5u * SIZE + q];
  S[5u * SIZE + q] = max(0.0, qv * (1.0 + (b.z - 1.0) * e));
}`;
      const pipe = dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code }), entryPoint: 'main' } });
      const E = dev.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const LZ = dev.createBuffer({ size: 4 * nz, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(LZ, 0, Float32Array.from(m.zc.subarray(0, nz)));
      this.editK = { pipe, E, bind: dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.S } }, { binding: 1, resource: { buffer: this.baseBuf } }, { binding: 2, resource: { buffer: E } }, { binding: 3, resource: { buffer: LZ } }] }) };
    }
    const E = e.kind === 'bubble' ? [0, e.x, e.y, e.z, e.rh, e.rz, e.amp, e.floorT, e.ox, e.oy, e.warm ? 1 : 0, 0]
      : [1, e.x, e.y, e.z, e.R, e.H, e.fac, 0, e.ox, e.oy, 0, 0];
    dev.queue.writeBuffer(this.editK.E, 0, new Float32Array(E));
    const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(this.editK.pipe); pass.setBindGroup(0, this.editK.bind); pass.dispatchWorkgroups(Math.ceil(nx * ny * nz / 64)); pass.end();
    dev.queue.submit([enc.finish()]);
  }

  private snap: { S: GPUBuffer; aux: GPUBuffer; time: number; steps: number } | null = null;
  /** Keep a copy of the state on the GPU (undo of an interaction), or restore it (false when there is none). */
  snapshot(): void {
    const dev = this.device, bytesS = this.nf * this.cpu.size * 4, bytesA = 4 * this.cpu.size * 4;
    if (!this.snap) this.snap = { S: dev.createBuffer({ size: bytesS, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }), aux: dev.createBuffer({ size: bytesA, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }), time: 0, steps: 0 };
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.S, 0, this.snap.S, 0, bytesS); enc.copyBufferToBuffer(this.aux, 0, this.snap.aux, 0, bytesA);
    dev.queue.submit([enc.finish()]);
    this.snap.time = this.time; this.snap.steps = this.steps;
  }
  restoreSnapshot(): boolean {
    const sn = this.snap; if (!sn) return false;
    const dev = this.device, enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(sn.S, 0, this.S, 0, this.nf * this.cpu.size * 4); enc.copyBufferToBuffer(sn.aux, 0, this.aux, 0, 4 * this.cpu.size * 4);
    dev.queue.submit([enc.finish()]);
    this.time = sn.time; this.steps = sn.steps;
    return true;
  }
  private force(): void {
    const k = this.forceK!, dev = this.device, { nx, ny, nz } = this.cpu.c;
    dev.queue.writeBuffer(k.FP, 0, new Float32Array([this.nForce, this.dt / FORCING_TAU, 0, 0]));
    const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(k.pipe); pass.setBindGroup(0, k.bind); pass.dispatchWorkgroups(Math.ceil(nx * ny * (nz + 1) / 64)); pass.end();
    dev.queue.submit([enc.finish()]);
  }

  step(n = 1): void {
    // several short command buffers per step: on large grids one step is ~1 s of GPU work, and a single
    // long submission can trip the OS GPU watchdog (Windows TDR, ~2 s), which resets the device
    const chunk = Math.max(4, Math.ceil(PASS_CELLS / this.cpu.size));
    for (let s = 0; s < n; s++) {
      // the CPU pre-step order: boundary-layer noise, cumulus, wind forcing
      if (this.blNoise > 0) {
        const e = Math.floor(this.time / BL_NOISE_PERIOD + 1e-6);
        if (this.noiseEpoch < 0 || e <= this.noiseEpoch) this.noiseEpoch = Math.max(this.noiseEpoch, e);
        else { this.noiseEpoch = e; this.noise(e); }
      }
      if (this.cuScale > 0) this.cumulus();
      if (this.nForce > 0) this.force();
      for (let p0 = 0; p0 < this.passes.length; p0 += chunk) {
        const enc = this.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        for (let p = p0; p < Math.min(this.passes.length, p0 + chunk); p++) this.passes[p]!(pass);
        pass.end();
        this.device.queue.submit([enc.finish()]);
      }
      this.time += this.dt;
      this.steps++;
    }
  }

  async readState(): Promise<Float32Array> {
    const bytes = this.nf * this.cpu.size * 4;
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

  /** Time the kernels of `steps` model steps (the model advances). With the 'timestamp-query' feature
   *  every kernel gets its own compute pass with GPU timestamps; otherwise each kernel is timed on the
   *  CPU as its own submission (includes submission overhead, coarser). Returns ms per step by kernel
   *  name, sorted, and the total. */
  async profile(steps = 3): Promise<{ method: 'gpu-timestamps' | 'cpu-timing'; total: number; rows: { label: string; ms: number; calls: number }[] }> {
    const dev = this.device, P = this.passes, acc = new Map<string, { ms: number; calls: number }>();
    const add = (label: string, ms: number): void => { const r = acc.get(label) ?? { ms: 0, calls: 0 }; r.ms += ms / steps; r.calls += 1 / steps; acc.set(label, r); };
    this.step(1); await dev.queue.onSubmittedWorkDone();                   // warm-up
    const ts = dev.features.has('timestamp-query');
    for (let s = 0; s < steps; s++) {
      if (ts) {
        const qs = dev.createQuerySet({ type: 'timestamp', count: 2 * P.length });
        const res = dev.createBuffer({ size: 16 * P.length, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        const rd = dev.createBuffer({ size: 16 * P.length, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = dev.createCommandEncoder();
        P.forEach((p, i) => { const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } }); p(pass); pass.end(); });
        enc.resolveQuerySet(qs, 0, 2 * P.length, res, 0);
        enc.copyBufferToBuffer(res, 0, rd, 0, 16 * P.length);
        dev.queue.submit([enc.finish()]);
        await rd.mapAsync(GPUMapMode.READ);
        const t = new BigInt64Array(rd.getMappedRange().slice(0));
        rd.unmap(); rd.destroy(); res.destroy(); qs.destroy();
        P.forEach((p, i) => add(p.label, Number(t[2 * i + 1]! - t[2 * i]!) / 1e6));
      } else {
        for (const p of P) {
          const t0 = performance.now();
          const enc = dev.createCommandEncoder(), pass = enc.beginComputePass(); p(pass); pass.end();
          dev.queue.submit([enc.finish()]);
          await dev.queue.onSubmittedWorkDone();
          add(p.label, performance.now() - t0);
        }
      }
      this.time += this.dt; this.steps++;
    }
    const rows = [...acc].map(([label, r]) => ({ label, ms: r.ms, calls: Math.round(r.calls) })).sort((a, b) => b.ms - a.ms);
    return { method: ts ? 'gpu-timestamps' : 'cpu-timing', total: rows.reduce((a, r) => a + r.ms, 0), rows };
  }

  private disp: { pipe: GPUComputePipeline; bind: GPUBindGroup; D: GPUBuffer; C: GPUBuffer; LB: GPUBuffer; MODE: GPUBuffer } | null = null;

  /**
   * Display data without reading the whole state back: a GPU kernel packs the 3-D view bytes and the
   * per-column composites; only those, the requested horizontal planes and the surface precipitation
   * accumulations are copied. `packed`: per cell cloud byte (qc + qi + qs) | channel-2 byte << 8 (channel 2:
   * precipitation qr + qg, or with `volMode` 1 updraft w, 2 cyclonic vertical vorticity); `col` (COL
   * values per column, offsets C in diagnostics.ts): w max, w min, condensate max, precipitation max,
   * column-max reflectivity (dBZ), cloud-top height (m), cloud-top temperature (K), 2-5 km updraft helicity
   * (m^2/s^2), surface-based CAPE and CIN (J/kg; same steps as parcelAscent); `planes`: every prognostic
   * field at the requested levels (w also at the level above); accumulated rain and snow at level 0.
   */
  async readDisplay(levels: number[], volMode = 0, subgrid = true): Promise<{ packed: Uint32Array; col: Float32Array; planes: Map<number, DisplayPlanes>; rain: Float32Array; snow: Float32Array; tracers: Float32Array | null; cu: Float32Array | null }> {
    const m = this.cpu, { nx, ny, nz, dx, dy } = m.c, n = nx * ny * nz, dev = this.device, PL = m.plane, SIZE = m.size, NFp = this.nf;
    if (!this.disp) {
      const code = `
const NX: u32 = ${nx}u; const NY: u32 = ${ny}u; const NZ: u32 = ${nz}u; const HH: u32 = ${H}u; const SX: u32 = ${m.sx}u; const PL: u32 = ${PL}u; const SIZE: u32 = ${SIZE}u; const ICE: bool = ${this.nq === 6}; const MOIST: bool = ${this.nq > 0};
const DX: f32 = ${dx}; const DY: f32 = ${dy}; const COL: u32 = ${COL}u; const K500: u32 = ${level500(m.pi0)}u; const K850: u32 = ${levelNear(m.pi0, 85000)}u; const K200: u32 = ${levelNear(m.pi0, 20000)}u;
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> D: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> LB: array<f32>;     // per level: pi0, rho0, zc, dz
@group(0) @binding(4) var<uniform> MODE: vec4<u32>;
const RHC: f32 = ${subgridRHc(dx)};
fn cs(f: u32, q: u32) -> f32 { return select(0.0, S[f * SIZE + q], MOIST); }
// display extinction (1/m) of cloud (cloudExtinction / subgridCloud in src/regional/display.ts)
fn extc(k: u32, q: u32, sub: bool) -> f32 {
  let pik = LB[4u * k] + S[4u * SIZE + q]; let T = S[3u * SIZE + q] * pik; let rho = LB[4u * k + 1u];
  let qc = max(cs(6u, q), 0.0);
  var qsub = 0.0;
  if (MOIST && sub && cs(6u, q) <= 1e-8) {
    let p = 1e5 * pow(pik, 1004.5 / 287.05); let es = 611.2 * exp(17.67 * (T - 273.15) / (T - 29.65));
    let qsat = 0.622 * es / max(p - es, 1.0); let rh = cs(5u, q) / qsat;
    if (rh > RHC && rh < 1.0) { let a = 1.0 + (rh - 1.0) / (1.0 - RHC); qsub = (1.0 - RHC) * qsat * a * a * a / 6.0; }
  }
  let ws = select(${EXT.ice.toFixed(1)}, ${EXT.liquid.toFixed(1)}, T > 253.15) * qsub;
  var e = ${EXT.liquid.toFixed(1)} * qc + ws;
  if (ICE) { e += ${EXT.ice.toFixed(1)} * max(S[8u * SIZE + q], 0.0) + ${EXT.snow.toFixed(1)} * max(S[9u * SIZE + q], 0.0); }
  return rho * e;
}
fn extb(beta: f32) -> u32 { if (beta <= 0.0) { return 0u; } return u32(min(255.0, round(255.0 * pow(beta / ${EXT_MAX}, 1.0 / 3.0)))); }
// cell-centred wind of column (i, j) at height h: linear between level centres, the lowest level below it (windIndices)
fn uvAt(i: u32, j: u32, h: f32) -> vec2<f32> {
  let fk = (h - LB[2]) / LB[3];
  let k0 = u32(clamp(floor(fk), 0.0, max(f32(NZ) - 2.0, 0.0)));
  let f = clamp(fk - f32(k0), 0.0, 1.0);
  let qa = k0 * PL + (j + HH) * SX + (i + HH); let qb = qa + PL;
  let a = vec2<f32>(0.5 * (S[qa] + S[qa + 1u]), 0.5 * (S[SIZE + qa] + S[SIZE + qa + SX]));
  let b = vec2<f32>(0.5 * (S[qb] + S[qb + 1u]), 0.5 * (S[SIZE + qb] + S[SIZE + qb + SX]));
  return a + (b - a) * f;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY) { return; }
  let j = t / NX; let i = t % NX;
  var wmax = 0.0; var wmin = 0.0; var cmax = 0.0; var pmax = 0.0; var zmax = 0.0; var ctz = 0.0; var ctt = 0.0; var uh = 0.0;
  var pw = 0.0; var tau = 0.0; var bet: array<f32, ${nz}>;
  // surface-based parcel (parcelAscent in src/regional/diagnostics.ts)
  var pth = 0.0; var pq = 0.0; var cape = 0.0; var cin = 0.0; var lcl = false; var lfc = false;
  var zlcl = -1.0; var li = 0.0; var etop = 0.0; var vil = 0.0;
  for (var k = 0u; k < NZ; k++) {
    let q = k * PL + (j + HH) * SX + (i + HH);
    var cl = cs(6u, q); var pr = cs(7u, q);
    var qs = 0.0; var qg = 0.0;
    if (ICE) { cl += S[8u * SIZE + q]; qs = max(S[9u * SIZE + q], 0.0); qg = max(S[10u * SIZE + q], 0.0); pr += qs + qg; }
    cl = max(cl, 0.0); pr = max(pr, 0.0);
    let rho = LB[4u * k + 1u];
    let rdz = rho * LB[4u * k + 3u];
    pw += rdz * max(cs(5u, q), 0.0);
    // reflectivity factor (mm^6 m^-3) from rain, snow and graupel contents (kg m^-3)
    let zr = 3.63e9 * pow(rho * max(cs(7u, q), 0.0), 1.75) + 9.80e8 * pow(rho * qs, 1.75) + 4.33e10 * pow(rho * qg, 1.75);
    zmax = max(zmax, zr);
    // echo top (${ETOP_DBZ} dBZ) and vertically integrated liquid (reflectivity capped at 56 dBZ)
    if (zr >= ${Math.pow(10, ETOP_DBZ / 10)}) { etop = LB[4u * k + 2u]; }
    if (zr > 0.0) { vil += 3.44e-6 * pow(min(zr, ${VIL_ZMAX}), 4.0 / 7.0) * LB[4u * k + 3u]; }
    let wc = 0.5 * (S[2u * SIZE + q] + S[2u * SIZE + q + PL]);
    let zeta = 0.25 * ((S[SIZE + q + 1u] + S[SIZE + q + 1u + SX]) - (S[SIZE + q - 1u] + S[SIZE + q - 1u + SX])) / DX
             - 0.25 * ((S[q + SX] + S[q + SX + 1u]) - (S[q - SX] + S[q - SX + 1u])) / DY;
    let zc = LB[4u * k + 2u];
    if (zc >= 2000.0 && zc <= 5000.0) { uh += wc * zeta * LB[4u * k + 3u]; }
    if (cl > 1e-5) { ctz = zc; ctt = S[3u * SIZE + q] * (LB[4u * k] + S[4u * SIZE + q]); }
    // 3-D view: extinction of cloud water, sub-grid cloud (MODE.y), ice and snow (anvils are mostly snow in this
    // scheme); channel 2: extinction of rain and graupel, or updraft, or cyclonic vorticity
    let bc = extc(k, q, MODE.y == 1u);
    bet[k] = bc; tau += bc * LB[4u * k + 3u];
    let cb = extb(bc);
    var pb = extb(rho * (${EXT.rain.toFixed(1)} * max(cs(7u, q), 0.0) + ${EXT.graupel.toFixed(1)} * qg));
    if (MODE.x == 1u) { pb = u32(min(255.0, round(sqrt(max(wc, 0.0) / 40.0) * 255.0))); }
    if (MODE.x == 2u) { pb = u32(min(255.0, round(sqrt(max(zeta, 0.0) / 0.05) * 255.0))); }
    D[(k * NY + j) * NX + i] = cb | (pb << 8u);
    let w = S[2u * SIZE + q];
    wmax = max(wmax, w); wmin = min(wmin, w); cmax = max(cmax, cl); pmax = max(pmax, pr);
    if (MOIST) {
      let pik = LB[4u * k] + S[4u * SIZE + q];
      let qvk = max(cs(5u, q), 0.0);
      if (k == 0u) { pth = S[3u * SIZE + q]; pq = qvk; }
      else {
        let pk = 1e5 * pow(pik, 1004.5 / 287.05);
        var tp = pth * pik; var d = 0.0;
        let es0 = 611.2 * exp(17.67 * (tp - 273.15) / (tp - 29.65));
        if (pq > 0.622 * es0 / max(pk - es0, 1.0)) {
          for (var it = 0; it < 4; it++) {
            let tt = tp + 2.5e6 * d / 1004.5;
            let es = 611.2 * exp(17.67 * (tt - 273.15) / (tt - 29.65));
            let qsat = 0.622 * es / max(pk - es, 1.0);
            let dq = qsat * pk / max(pk - es, 1.0) * 17.67 * 243.5 / ((tt - 29.65) * (tt - 29.65));
            d += (pq - d - qsat) / (1.0 + 2.5e6 / 1004.5 * dq);
          }
          d = clamp(d, 0.0, pq);
          if (!lcl) { zlcl = LB[4u * k + 2u]; }
          lcl = true;
        }
        tp += 2.5e6 * d / 1004.5; pq -= d; pth = tp / pik;
        if (k == K500) { li = S[3u * SIZE + q] * pik - tp; }
        let tve = S[3u * SIZE + q] * pik * (1.0 + 0.61 * qvk);
        let b = 9.80665 * (tp * (1.0 + 0.61 * pq) - tve) / tve;
        if (!lfc) { if (lcl && b > 0.0) { lfc = true; cape += b * LB[4u * k + 3u]; } else { cin += min(b, 0.0) * LB[4u * k + 3u]; } }
        else if (b > 0.0) { cape += b * LB[4u * k + 3u]; }
      }
    }
  }
  if (!lfc) { cape = 0.0; cin = 0.0; }
  if (ctz == 0.0) { let q0 = (j + HH) * SX + (i + HH); ctt = S[3u * SIZE + q0] * (LB[0] + S[4u * SIZE + q0]); }
  C[COL * t] = wmax; C[COL * t + 1u] = wmin; C[COL * t + 2u] = cmax; C[COL * t + 3u] = pmax;
  C[COL * t + 4u] = 10.0 * log(max(zmax, 1e-3)) / log(10.0); C[COL * t + 5u] = ctz; C[COL * t + 6u] = ctt; C[COL * t + 7u] = uh;
  C[COL * t + 8u] = cape; C[COL * t + 9u] = cin;
  // satellite-like values (columnDiagnostics in src/regional/diagnostics.ts): cloud albedo, precipitable water,
  // water-vapour channel temperature (WV_PATH ${WV_PATH} kg/m^2)
  var above = pw; var wvT = 0.0; var zEmit = LB[4u * (NZ - 1u) + 2u]; var found = false;
  for (var k = 0u; k < NZ; k++) {
    let q = k * PL + (j + HH) * SX + (i + HH);
    above -= LB[4u * k + 1u] * LB[4u * k + 3u] * max(cs(5u, q), 0.0);
    if (!found && above < ${WV_PATH}) { found = true; wvT = S[3u * SIZE + q] * (LB[4u * k] + S[4u * SIZE + q]); zEmit = LB[4u * k + 2u]; }
  }
  if (!found) { let q = (NZ - 1u) * PL + (j + HH) * SX + (i + HH); wvT = S[3u * SIZE + q] * (LB[4u * (NZ - 1u)] + S[4u * SIZE + q]); }
  if (ctz > zEmit) { wvT = ctt; }
  // visible image: cloud albedo of the column optical depth, and the height where the optical depth from the top reaches 1
  // (the tau-weighted mean height of thinner columns); columnDiagnostics in src/regional/diagnostics.ts
  var below = 0.0; var zvis = 0.0; var zw = 0.0;
  for (var k = 0u; k < NZ; k++) {
    let dt = bet[k] * LB[4u * k + 3u];
    if (tau - below >= 1.0) { zvis = LB[4u * k + 2u]; }
    zw += dt * LB[4u * k + 2u]; below += dt;
  }
  if (tau < 1.0) { zvis = select(0.0, zw / tau, tau > 1e-3); }
  C[COL * t + 10u] = tau / (tau + 7.7); C[COL * t + 11u] = pw; C[COL * t + 12u] = wvT; C[COL * t + 13u] = zvis;
  // 0-6 km shear and storm-relative helicity for the Bunkers right mover (windIndices in src/regional/diagnostics.ts)
  var mw = vec2<f32>(0.0, 0.0);
  for (var n = 0u; n <= 12u; n++) { mw += uvAt(i, j, f32(n) * 500.0); }
  mw /= 13.0;
  let a0 = uvAt(i, j, 0.0); let b1 = uvAt(i, j, 6000.0);
  let sh = 0.5 * (uvAt(i, j, 5500.0) + b1 - a0 - uvAt(i, j, 500.0)); let sl = length(sh);
  var cm = mw;
  if (sl > 0.1) { cm = mw + 7.5 * vec2<f32>(sh.y, -sh.x) / sl; }
  var srh1 = 0.0; var srh3 = 0.0; var pv = a0;
  for (var n = 1u; n <= 12u; n++) {
    let w = uvAt(i, j, f32(n) * 250.0);
    srh3 += (w.x - cm.x) * (pv.y - cm.y) - (pv.x - cm.x) * (w.y - cm.y);
    if (n <= 4u) { srh1 = srh3; }
    pv = w;
  }
  C[COL * t + 14u] = length(b1 - a0); C[COL * t + 15u] = srh1; C[COL * t + 16u] = srh3; C[COL * t + 17u] = zlcl;
  C[COL * t + 18u] = li; C[COL * t + 19u] = etop; C[COL * t + 20u] = vil;
  // winds at the 850 and 200 hPa levels (cell-centred) and the 0-1 km shear
  let q85 = K850 * PL + (j + HH) * SX + (i + HH); let q20 = K200 * PL + (j + HH) * SX + (i + HH);
  C[COL * t + 21u] = 0.5 * (S[q85] + S[q85 + 1u]); C[COL * t + 22u] = 0.5 * (S[SIZE + q85] + S[SIZE + q85 + SX]);
  C[COL * t + 23u] = 0.5 * (S[q20] + S[q20 + 1u]); C[COL * t + 24u] = 0.5 * (S[SIZE + q20] + S[SIZE + q20 + SX]);
  C[COL * t + 25u] = length(uvAt(i, j, 1000.0) - a0);
}`;
      const pipe = dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code }), entryPoint: 'main' } });
      const D = dev.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const C = dev.createBuffer({ size: nx * ny * COL * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const lb = new Float32Array(4 * nz);
      for (let k = 0; k < nz; k++) { lb[4 * k] = m.pi0[k]!; lb[4 * k + 1] = m.rho0[k]!; lb[4 * k + 2] = m.zc[k]!; lb[4 * k + 3] = m.c.dz; }
      const LB = dev.createBuffer({ size: lb.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(LB, 0, lb);
      const MODE = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const bind = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.S } }, { binding: 1, resource: { buffer: D } }, { binding: 2, resource: { buffer: C } }, { binding: 3, resource: { buffer: LB } }, { binding: 4, resource: { buffer: MODE } }] });
      this.disp = { pipe, bind, D, C, LB, MODE };
    }
    const d = this.disp, planeBytes = PL * 4;
    dev.queue.writeBuffer(d.MODE, 0, new Uint32Array([volMode, subgrid ? 1 : 0, 0, 0]));
    const perLevel = NFp + 1;
    const trBytes = this.trK ? this.trK.n * 16 : 0;
    const cuBytes = this.cuScale > 0 ? 16 * nx * ny : 0;
    const total = n * 4 + nx * ny * COL * 4 + levels.length * perLevel * planeBytes + 2 * planeBytes + trBytes + cuBytes;
    const st = dev.createBuffer({ size: total, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass(); pass.setPipeline(d.pipe); pass.setBindGroup(0, d.bind); pass.dispatchWorkgroups(Math.ceil(nx * ny / 64)); pass.end();
    let off = 0;
    enc.copyBufferToBuffer(d.D, 0, st, off, n * 4); off += n * 4;
    enc.copyBufferToBuffer(d.C, 0, st, off, nx * ny * COL * 4); off += nx * ny * COL * 4;
    for (const k of levels) {
      for (let f = 0; f < NFp; f++) { enc.copyBufferToBuffer(this.S, (f * SIZE + k * PL) * 4, st, off, planeBytes); off += planeBytes; }
      enc.copyBufferToBuffer(this.S, (2 * SIZE + (k + 1) * PL) * 4, st, off, planeBytes); off += planeBytes;   // w above
    }
    enc.copyBufferToBuffer(this.aux, 2 * SIZE * 4, st, off, planeBytes); off += planeBytes;
    enc.copyBufferToBuffer(this.aux, 3 * SIZE * 4, st, off, planeBytes); off += planeBytes;
    if (this.trK) { enc.copyBufferToBuffer(this.trK.T, 0, st, off, trBytes); off += trBytes; }
    if (cuBytes) enc.copyBufferToBuffer(this.ensureCuBuf(), 0, st, off, cuBytes);
    dev.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const buf = st.getMappedRange().slice(0);
    st.unmap(); st.destroy();
    let o = 0;
    const packed = new Uint32Array(buf, o, n); o += n * 4;
    const col = new Float32Array(buf, o, nx * ny * COL); o += nx * ny * COL * 4;
    const planes = new Map<number, DisplayPlanes>();
    for (const k of levels) {
      const g = (): Float32Array => { const a = new Float32Array(buf, o, PL); o += planeBytes; return a; };
      const f: Float32Array[] = []; for (let i = 0; i < NFp; i++) f.push(g());
      planes.set(k, { u: f[0]!, v: f[1]!, w: f[2]!, th: f[3]!, pp: f[4]!, sc: f.slice(5), wTop: g() });
    }
    const rain = new Float32Array(buf, o, PL); o += planeBytes;
    const snow = new Float32Array(buf, o, PL); o += planeBytes;
    const tracers = trBytes ? new Float32Array(buf, o, trBytes / 4) : null; o += trBytes;
    const cu = cuBytes ? new Float32Array(buf, o, cuBytes / 4) : null;
    return { packed, col, planes, rain, snow, tracers, cu };
  }

  /** Column profiles at grid columns (i, j): per point, per level, the cell-centred u, v, w, theta, pi' and
   *  moisture species (NF values). For cross-sections and soundings. */
  async readColumns(points: { i: number; j: number }[]): Promise<Float32Array> {
    const m = this.cpu, { nx, ny, nz } = m.c, dev = this.device, NF = this.nf, np = Math.max(1, points.length);
    if (!this.colK) {
      const code = this.consts + `
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read> PTS: array<u32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let np = arrayLength(&PTS) / 2u;
  if (t >= np * NZ) { return; }
  let p = t / NZ; let k = t % NZ;
  let q = ix(min(PTS[2u * p], NX - 1u), min(PTS[2u * p + 1u], NY - 1u), k);
  let o = (p * NZ + k) * NFLD;
  O[o] = 0.5 * (S[q] + S[q + 1u]);
  O[o + 1u] = 0.5 * (S[SIZE + q] + S[SIZE + q + SX]);
  O[o + 2u] = 0.5 * (S[2u * SIZE + q] + S[2u * SIZE + q + PL]);
  for (var f = 3u; f < NFLD; f++) { O[o + f] = S[f * SIZE + q]; }
}`;
      const pipe = dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code }), entryPoint: 'main' } });
      this.colK = { pipe };
    }
    const pts = new Uint32Array(2 * np); points.forEach((pt, i) => { pts[2 * i] = Math.max(0, Math.min(nx - 1, pt.i)); pts[2 * i + 1] = Math.max(0, Math.min(ny - 1, pt.j)); });
    const P = dev.createBuffer({ size: pts.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(P, 0, pts);
    const bytes = np * nz * NF * 4;
    const O = dev.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const st = dev.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const bind = dev.createBindGroup({ layout: this.colK.pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.S } }, { binding: 1, resource: { buffer: P } }, { binding: 2, resource: { buffer: O } }] });
    const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(this.colK.pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(Math.ceil(np * nz / 64)); pass.end();
    enc.copyBufferToBuffer(O, 0, st, 0, bytes);
    dev.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(st.getMappedRange().slice(0));
    st.unmap(); for (const b of [st, O, P]) b.destroy();
    return out;
  }
  private colK: { pipe: GPUComputePipeline } | null = null;

  /** Azimuthal means about (xc, yc) (m from the domain origin) in nr rings of width dr, per level:
   *  tangential wind, radial wind, w, theta - theta0, total condensate (5 values per ring and level). */
  async readRZ(xc: number, yc: number, dr: number, nr: number): Promise<Float32Array> {
    const m = this.cpu, { nz } = m.c, dev = this.device;
    if (!this.rzK) {
      const th0 = Array.from(m.th0).slice(0, nz).map((x) => x.toFixed(6)).join(', ');
      const code = this.consts + `
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<uniform> RP: vec4<f32>;          // xc, yc, dr, nr
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
const TH0 = array<f32, NZ>(${th0});
fn cellAt(x: f32, y: f32) -> vec2<i32> {
  var i = i32(floor(x / DX)); var j = i32(floor(y / DY));
  if (OPEN) { i = clamp(i, 0, i32(NX) - 1); j = clamp(j, 0, i32(NY) - 1); }
  else { i = ((i % i32(NX)) + i32(NX)) % i32(NX); j = ((j % i32(NY)) + i32(NY)) % i32(NY); }
  return vec2<i32>(i, j);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let nr = u32(RP.w);
  let t = gid.x;
  if (t >= nr * NZ) { return; }
  let ir = t / NZ; let k = t % NZ;
  let r = (f32(ir) + 0.5) * RP.z;
  let na = max(16u, min(256u, u32(6.2832 * r / (0.5 * DX))));
  var vt = 0.0; var vr = 0.0; var ww = 0.0; var tp = 0.0; var cd = 0.0;
  for (var a = 0u; a < na; a++) {
    let ang = 6.2831853 * (f32(a) + 0.5) / f32(na);
    let ca = cos(ang); let sa = sin(ang);
    let c = cellAt(RP.x + r * ca, RP.y + r * sa);
    let q = ix(u32(c.x), u32(c.y), k);
    let uc = 0.5 * (S[q] + S[q + 1u]); let vc = 0.5 * (S[SIZE + q] + S[SIZE + q + SX]);
    vt += -uc * sa + vc * ca; vr += uc * ca + vc * sa;
    ww += 0.5 * (S[2u * SIZE + q] + S[2u * SIZE + q + PL]);
    tp += S[3u * SIZE + q] - TH0[k];
    if (MOIST) { for (var f = 6u; f < NFLD; f++) { cd += max(S[f * SIZE + q], 0.0); } }
  }
  let inv = 1.0 / f32(na); let o = 5u * t;
  O[o] = vt * inv; O[o + 1u] = vr * inv; O[o + 2u] = ww * inv; O[o + 3u] = tp * inv; O[o + 4u] = cd * inv;
}`;
      const pipe = dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code }), entryPoint: 'main' } });
      const RP = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.rzK = { pipe, RP, O: null };
    }
    const k = this.rzK, bytes = nr * nz * 5 * 4;
    if (!k.O || k.O.size !== bytes) { k.O?.destroy(); k.O = dev.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }); }
    dev.queue.writeBuffer(k.RP, 0, new Float32Array([xc, yc, dr, nr]));
    const bind = dev.createBindGroup({ layout: k.pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.S } }, { binding: 1, resource: { buffer: k.RP } }, { binding: 2, resource: { buffer: k.O } }] });
    const st = dev.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(k.pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(Math.ceil(nr * nz / 64)); pass.end();
    enc.copyBufferToBuffer(k.O, 0, st, 0, bytes);
    dev.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(st.getMappedRange().slice(0));
    st.unmap(); st.destroy();
    return out;
  }
  private rzK: { pipe: GPUComputePipeline; RP: GPUBuffer; O: GPUBuffer | null } | null = null;

  private trK: { pipe: GPUComputePipeline; T: GPUBuffer; P: GPUBuffer; bind: GPUBindGroup; n: number } | null = null;

  /** Tracer particles (src/regional/tracers.ts): upload positions (x, y, z, age per particle). */
  initTracers(pos: Float32Array): void {
    const dev = this.device, m = this.cpu, { nx, ny, dx, dy } = m.c, n = pos.length / 4;
    this.trK?.T.destroy(); this.trK?.P.destroy();
    if (n === 0) { this.trK = null; return; }
    const pipe = this.trPipe ?? (this.trPipe = dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code: this.consts + `
const LX: f32 = ${nx * dx}; const LY: f32 = ${ny * dy};
struct TP { h: f32, nsub: u32, seed: u32, life: f32, cx: f32, cy: f32, rad: f32, zseed: f32, dt: f32, top: f32, n: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> T: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> TPR: TP;
fn samp(f: u32, gi: f32, gj: f32, gk: f32, nk: u32) -> f32 {
  let i0 = clamp(i32(floor(gi)), -2, i32(NX)); let j0 = clamp(i32(floor(gj)), -2, i32(NY));
  let kc = clamp(gk, 0.0, f32(nk - 1u)); let k0 = min(i32(nk) - 2, i32(floor(kc)));
  let fi = clamp(gi - f32(i0), 0.0, 1.0); let fj = clamp(gj - f32(j0), 0.0, 1.0); let fk = clamp(kc - f32(k0), 0.0, 1.0);
  let q = f * SIZE + u32(k0 * i32(PL) + (j0 + i32(HH)) * i32(SX) + (i0 + i32(HH)));
  let b = (S[q] * (1.0 - fi) + S[q + 1u] * fi) * (1.0 - fj) + (S[q + SX] * (1.0 - fi) + S[q + SX + 1u] * fi) * fj;
  let t = (S[q + PL] * (1.0 - fi) + S[q + PL + 1u] * fi) * (1.0 - fj) + (S[q + PL + SX] * (1.0 - fi) + S[q + PL + SX + 1u] * fi) * fj;
  return b * (1.0 - fk) + t * fk;
}
fn wind(p: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(samp(0u, p.x / DX, p.y / DY - 0.5, p.z / DZ - 0.5, NZ), samp(1u, p.x / DX - 0.5, p.y / DY, p.z / DZ - 0.5, NZ), samp(2u, p.x / DX - 0.5, p.y / DY - 0.5, p.z / DZ, NZ + 1u));
}
fn pcg(v: u32) -> u32 { let s = v * 747796405u + 2891336453u; let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
fn hh(p: u32, k: u32) -> f32 { return f32(pcg(p * 4u + k + TPR.seed)) / 4294967296.0; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= TPR.n) { return; }
  let x0 = T[p];
  var pos = x0.xyz;
  for (var s = 0u; s < TPR.nsub; s++) {
    let a = wind(pos);
    var mid = pos + 0.5 * TPR.h * a; mid.z = max(mid.z, 0.0);
    pos += TPR.h * wind(mid); pos.z = max(pos.z, 0.0);
    if (!OPEN) { pos.x -= floor(pos.x / LX) * LX; pos.y -= floor(pos.y / LY) * LY; }
  }
  var age = x0.w + TPR.dt;
  if (age > TPR.life || pos.z > TPR.top || (OPEN && (pos.x < 0.0 || pos.y < 0.0 || pos.x > LX || pos.y > LY))) {
    if (TPR.rad > 0.0 && (p & 1u) == 0u) {
      let a = 6.2831853 * hh(p, 0u); let r = TPR.rad * sqrt(hh(p, 1u));
      pos.x = clamp(TPR.cx + r * cos(a), 0.0, LX); pos.y = clamp(TPR.cy + r * sin(a), 0.0, LY);
    } else { pos.x = hh(p, 0u) * LX; pos.y = hh(p, 1u) * LY; }
    pos.z = 50.0 + hh(p, 2u) * (TPR.zseed - 50.0);
    age = 0.0;
  }
  T[p] = vec4<f32>(pos, age);
}` }), entryPoint: 'main' } }));
    const T = dev.createBuffer({ size: pos.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(T, 0, pos);
    const P = dev.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bind = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.S } }, { binding: 1, resource: { buffer: T } }, { binding: 2, resource: { buffer: P } }] });
    this.trK = { pipe, T, P, bind, n };
  }
  private trPipe: GPUComputePipeline | null = null;
  get tracerCount(): number { return this.trK?.n ?? 0; }

  /** Advance the tracers by dt in nsub midpoint steps in the current wind; seed: re-seeding hash offset. */
  advectTracers(dt: number, nsub: number, pr: TracerParams, seed: number): void {
    const k = this.trK; if (!k) return;
    const m = this.cpu, ns = Math.max(1, nsub | 0);
    const b = new ArrayBuffer(48), f = new Float32Array(b), u = new Uint32Array(b);
    f[0] = dt / ns; u[1] = ns; u[2] = seed >>> 0; f[3] = pr.life; f[4] = pr.cx; f[5] = pr.cy; f[6] = pr.rad; f[7] = pr.zSeed; f[8] = dt; f[9] = m.c.nz * m.c.dz - m.c.dampDepth; u[10] = k.n;
    this.device.queue.writeBuffer(k.P, 0, b);
    const enc = this.device.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(k.pipe); pass.setBindGroup(0, k.bind); pass.dispatchWorkgroups(Math.ceil(k.n / 64)); pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  async readTracers(): Promise<Float32Array> { return this.trK ? this.readBuffer(this.trK.T, this.trK.n * 16) : new Float32Array(0); }

  /** Accumulated frozen precipitation (snow + graupel + ice) at level 0, same layout as readRain. */
  async readSnow(): Promise<Float32Array> { const b = this.cpu.size * 4; const all = await this.readBuffer(this.aux, 4 * b); return all.slice(3 * this.cpu.size, 4 * this.cpu.size); }

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
  // one invocation per halo cell only: NH = SX*SY - NX*NY per level, enumerated as the full-width
  // bottom and top strips (2*HH rows of SX) followed by the side strips (NY rows of 2*HH)
  let t = gid.x;
  let NH = SX * SY - NX * NY;
  let per = NH * hp.nk;
  if (t >= per * hp.nf) { return; }
  let f = hp.f0 + t / per;
  let r = t % per;
  let k = r / NH;
  let h = r % NH;
  var ie: u32; var je: u32;
  if (h < 2u * HH * SX) {
    let row = h / SX;
    je = select(row - HH + NY + HH, row, row < HH);
    ie = h % SX;
  } else {
    let h2 = h - 2u * HH * SX;
    je = HH + h2 / (2u * HH);
    let c = h2 % (2u * HH);
    ie = select(c - HH + NX + HH, c, c < HH);
  }
  var si = (ie + NX - HH) % NX + HH;
  var sj = (je + NY - HH) % NY + HH;
  if (OPEN) { si = clamp(ie, HH, NX + HH - 1u); sj = clamp(je, HH, NY + HH - 1u); }
  A[f * SIZE + k * PL + je * SX + ie] = A[f * SIZE + k * PL + sj * SX + si];
}
`;

const SAVE_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> S0: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { if (gid.x < NFLD * SIZE) { S0[gid.x] = S[gid.x]; } }
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
${THR_FNS}
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
      fu += FCOR * (0.25 * (V(q) + V(q - 1u) + V(q + SX) + V(q - 1u + SX)) - GEOB * bvb(k));
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
      fv -= FCOR * (0.25 * (U(q) + U(q + 1u) + U(q - SX) + U(q + 1u - SX)) - GEOB * bub(k));
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
// With pd = true (final RK3 stage) the moisture species use the positive-definite flux limiter
// (Skamarock 2006): full face fluxes scaled by the donor cell's ratio R from PDRATIO, as in the CPU core.
const scalarWgsl = (pd: boolean, cfBinding = -1): string => /* wgsl */`
const PD: bool = ${pd};
${cfBinding >= 0 ? `@group(0) @binding(${cfBinding}) var<storage, read> CF: array<u32>;` : ''}
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> F: array<f32>;
@group(0) @binding(2) var<storage, read> base: array<f32>;
${pd ? '@group(0) @binding(3) var<storage, read> R: array<f32>;' : ''}
${BASE_FNS}
${ZFACE_FN}
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
  if (MOIST) { nf = 2u + NQ; }
  let wet = ${cfBinding >= 0 ? 'CF[2u * NX * NY + j * NX + i] != 0u' : 'true'};
  for (var s = 0u; s < nf; s++) {
    let off = (3u + s) * SIZE;
    if (s >= 3u && !wet) { F[off + q] = 0.0; continue; }   // hydrometeors in clear air: tendency exactly 0
    ${pd ? `if (s >= 2u) {
      let rs = (s - 2u) * SIZE;
      var fL = ul * f5(S[off + q - 3u], S[off + q - 2u], S[off + q - 1u], S[off + q], S[off + q + 1u], S[off + q + 2u], ul);
      var fR = ur * f5(S[off + q - 2u], S[off + q - 1u], S[off + q], S[off + q + 1u], S[off + q + 2u], S[off + q + 3u], ur);
      var gL = vl * f5(S[off + q - 3u * SX], S[off + q - 2u * SX], S[off + q - SX], S[off + q], S[off + q + SX], S[off + q + 2u * SX], vl);
      var gR = vr * f5(S[off + q - 2u * SX], S[off + q - SX], S[off + q], S[off + q + SX], S[off + q + 2u * SX], S[off + q + 3u * SX], vr);
      var hB = 0.0; var hT = 0.0;
      if (k > 0u) { hB = wb * zface(off, q, k, S[2u * SIZE + q], 0.0); }
      if (k < NZ - 1u) { hT = wt * zface(off, q + PL, k + 1u, S[2u * SIZE + q + PL], 0.0); }
      fL *= select(R[rs + q], R[rs + q - 1u], fL > 0.0);
      fR *= select(R[rs + q + 1u], R[rs + q], fR > 0.0);
      gL *= select(R[rs + q], R[rs + q - SX], gL > 0.0);
      gR *= select(R[rs + q + SX], R[rs + q], gR > 0.0);
      hB *= select(R[rs + q], R[rs + q - PL], hB > 0.0);
      hT *= select(R[rs + q + PL], R[rs + q], hT > 0.0);
      F[off + q] = -((fR - fL) / DX + (gR - gL) / DY + (hT - hB) / (r0 * DZ)) + S[off + q] * dv;
      continue;
    }` : ''}
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
      if (PHYS && RADC > 0.0 && bth0(k) * bpi0(k) > 210.0) { tend -= RADC / bpi0(k); }
      else if (PHYS && RADTAU > 0.0) { tend += max(-(S[off + q] - bth0(k)) / RADTAU, -RADMAX / bpi0(k)); }
    }
    F[off + q] = tend;
  }
}
`;

/** z-face interpolation of field slot `off` (values relative to c0), shared by the scalar kernels. */
const ZFACE_FN = `
fn zface(off: u32, qf: u32, kf: u32, vel: f32, c0: f32) -> f32 {
  if (kf >= 3u && kf <= NZ - 3u) { return f5(S[off + qf - 3u * PL] - c0, S[off + qf - 2u * PL] - c0, S[off + qf - PL] - c0, S[off + qf] - c0, S[off + qf + PL] - c0, S[off + qf + 2u * PL] - c0, vel); }
  if (kf >= 2u && kf <= NZ - 2u) { return f3(S[off + qf - 2u * PL] - c0, S[off + qf - PL] - c0, S[off + qf] - c0, S[off + qf + PL] - c0, vel); }
  return 0.5 * (S[off + qf - PL] + S[off + qf]) - c0;
}
`;

// positive-definite limiter ratios for the moisture species (final RK3 stage): outflow over dt from the
// unlimited face fluxes of the current stage vs. the content at time level n (S0) -> R
const PDRATIO_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read> S0: array<f32>;
@group(0) @binding(2) var<storage, read_write> R: array<f32>;
@group(0) @binding(3) var<storage, read> base: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> CF: array<u32>;
${BASE_FNS}
${ZFACE_FN}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY * NZ) { return; }
  let k = t / (NX * NY); let r = t % (NX * NY); let j = r / NX; let i = r % NX;
  let q = ix(i, j, k);
  let wet = CF[2u * NX * NY + j * NX + i] != 0u;
  let r0 = brho0(k);
  let ul = S[q]; let ur = S[q + 1u];
  let vl = S[SIZE + q]; let vr = S[SIZE + q + SX];
  let wb = S[2u * SIZE + q] * brho0f(k); let wt = S[2u * SIZE + q + PL] * brho0f(k + 1u);
  for (var s = 0u; s < NQ; s++) {
    let off = (5u + s) * SIZE;
    if (s >= 1u && !wet) { R[s * SIZE + q] = 1.0; continue; }
    let fL = ul * f5(S[off + q - 3u], S[off + q - 2u], S[off + q - 1u], S[off + q], S[off + q + 1u], S[off + q + 2u], ul);
    let fR = ur * f5(S[off + q - 2u], S[off + q - 1u], S[off + q], S[off + q + 1u], S[off + q + 2u], S[off + q + 3u], ur);
    let gL = vl * f5(S[off + q - 3u * SX], S[off + q - 2u * SX], S[off + q - SX], S[off + q], S[off + q + SX], S[off + q + 2u * SX], vl);
    let gR = vr * f5(S[off + q - 2u * SX], S[off + q - SX], S[off + q], S[off + q + SX], S[off + q + 2u * SX], S[off + q + 3u * SX], vr);
    var hB = 0.0; var hT = 0.0;
    if (k > 0u) { hB = wb * zface(off, q, k, S[2u * SIZE + q], 0.0); }
    if (k < NZ - 1u) { hT = wt * zface(off, q + PL, k + 1u, S[2u * SIZE + q + PL], 0.0); }
    let outflow = (max(fR, 0.0) - min(fL, 0.0)) / DX + (max(gR, 0.0) - min(gL, 0.0)) / DY + (max(hT, 0.0) - min(hB, 0.0)) / (r0 * DZ);
    let avail = max(S0[off + q], 0.0);
    R[s * SIZE + q] = select(1.0, avail / (outflow * p.dtBig), outflow * p.dtBig > avail);
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

// eddy diffusion of u, v, w, theta, moisture: written to the physics-tendency buffer (bound as F), which is
// computed in the first RK stage only and added to the slow tendencies of every stage (as in WRF)
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
    // eddy diffusivities at the cell and its six neighbours: the same for every field, read once
    let K0 = Kd(q);
    let kxr = 0.5 * LH2 * (K0 + Kd(q + 1u)) / DX; let kxl = 0.5 * LH2 * (K0 + Kd(q - 1u)) / DX;
    let kyr = 0.5 * LH2 * (K0 + Kd(q + SX)) / DY; let kyl = 0.5 * LH2 * (K0 + Kd(q - SX)) / DY;
    var kzt = 0.0; var kzb = 0.0;
    if (k < NZ - 1u) { kzt = 0.5 * (K0 + Kd(q + PL)) * LV2 * brho0f(k + 1u) / DZ; }
    if (k > 0u) { kzb = 0.5 * (K0 + Kd(q - PL)) * LV2 * brho0f(k) / DZ; }
    let rz = 1.0 / (brho0(k) * DZ);
    for (var f = 0u; f < NFLD; f++) {
      if (f == 2u || f == 4u) { continue; }
      let ipr = select(3.0, 1.0, f < 2u);                  // 1 / Prandtl number
      let off = f * SIZE;
      let c = S[off + q];
      let ac = c - bval(f, k);
      let fxr = kxr * (S[off + q + 1u] - c); let fxl = kxl * (c - S[off + q - 1u]);
      let fyr = kyr * (S[off + q + SX] - c); let fyl = kyl * (c - S[off + q - SX]);
      var fzt = 0.0; var fzb = 0.0;
      if (k < NZ - 1u) { fzt = kzt * ((S[off + q + PL] - bval(f, k + 1u)) - ac); }
      if (k > 0u) { fzb = kzb * (ac - (S[off + q - PL] - bval(f, k - 1u))); }
      F[off + q] = ipr * ((fxr - fxl) / DX + (fyr - fyl) / DY + (fzt - fzb) * rz);
    }
  }
  if (k >= 1u && k < NZ) {
    let kc = 0.5 * (Kd(q) + Kd(q - PL));
    let W = 2u * SIZE;
    F[W + q] = LH2 * kc * ((S[W + q + 1u] - 2.0 * S[W + q] + S[W + q - 1u]) / (DX * DX) + (S[W + q + SX] - 2.0 * S[W + q] + S[W + q - SX]) / (DY * DY))
      + LV2 * kc * (S[W + q + PL] - 2.0 * S[W + q] + S[W + q - PL]) / (DZ * DZ);
  }
}
`;

// bulk sea-surface fluxes at the lowest level (adds to F); one thread per column
const surfaceWgsl = (logDrag: boolean): string => /* wgsl */`
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> F: array<f32>;
@group(0) @binding(2) var<storage, read> SF: array<f32>;   // skin temperature, wetness (per column)
// saturation vapour pressure at the skin temperature: surfaceQs() in src/regional/physics.ts (any temperature gives a number)
fn esurf(tsk: f32) -> f32 {
  let T = max(tsk, TFLOOR);
  return min(611.2 * exp(17.67 * (T - 273.15) / (T - 29.65)), 0.9 * PSFC);
}
fn cdrag(spd: f32) -> f32 {
  ${logDrag ? 'let l = 0.4 / log(0.5 * DZ / Z0); return l * l;' : 'return min(2.4e-3, 1.0e-3 * (1.0 + 0.07 * spd));'}
}
// effective wind of the bulk fluxes of column c (lowest-level cell qq): gustSpeed() in src/regional/physics.ts
fn effspd(qq: u32, c: u32, ua: f32, va: f32) -> f32 {
  let s = sqrt(ua * ua + va * va);
  if (!GUST) { return max(s, VMIN); }
  let tsk = SF[c]; let th = S[3u * SIZE + qq];
  let es = esurf(tsk);
  var dq = 0.0; var qr = 0.0;
  if (MOIST) { dq = (0.622 * es / (PSFC - 0.378 * es) - S[5u * SIZE + qq]) * SF[NX * NY + c]; qr = max(S[7u * SIZE + qq], 0.0); }
  let b = CK * max(s, VMIN) * (tsk / PIS - th + 0.61 * th * dq);
  var ws = 0.0;
  if (b > 0.0) { ws = pow(9.80665 / th * b * 1000.0, 1.0 / 3.0); }
  let rq = RHO1 * qr;
  var rcd = 0.0;
  if (rq > 0.0) { rcd = min(7.0, rq * 36.34 * pow(1e-3 * rq, 0.1364) * 3600.0 * 2.4); }
  let ug = log(1.0 + 6.69 * rcd - 0.476 * rcd * rcd);
  return max(VMIN, sqrt(s * s + 1.44 * ws * ws + ug * ug));
}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY) { return; }
  let j = t / NX; let i = t % NX;
  let q = ix(i, j, 0u);
  let ua = 0.5 * (S[q] + S[q + 1u]) + FRU; let va = 0.5 * (S[SIZE + q] + S[SIZE + q + SX]) + FRV;
  let spd = effspd(q, t, ua, va);
  let cd = cdrag(spd);
  let taux = cd * spd * ua; let tauy = cd * spd * va;
  // each u/v face is shared by two columns: apply half of each column's stress to its two faces
  // (atomic-free: every thread only writes its own west/south face with its own and neighbour's share)
  let qw = ix((i + NX - 1u) % NX, j, 0u); let qs = ix(i, (j + NY - 1u) % NY, 0u);
  let uw = 0.5 * (S[qw] + S[qw + 1u]) + FRU; let vw = 0.5 * (S[SIZE + qw] + S[SIZE + qw + SX]) + FRV;
  let spw = effspd(qw, j * NX + (i + NX - 1u) % NX, uw, vw); let cdw = cdrag(spw);
  let us = 0.5 * (S[qs] + S[qs + 1u]) + FRU; let vs = 0.5 * (S[SIZE + qs] + S[SIZE + qs + SX]) + FRV;
  let sps = effspd(qs, ((j + NY - 1u) % NY) * NX + i, us, vs); let cds = cdrag(sps);
  var shw = cdw * spw * uw; var shs = cds * sps * vs;
  if (OPEN && i == 0u) { shw = 0.0; }
  if (OPEN && j == 0u) { shs = 0.0; }
  F[q] -= 0.5 * (taux + shw) / DZ;
  F[SIZE + q] -= 0.5 * (tauy + shs) / DZ;
  let tsk = SF[t];
  let esS = esurf(tsk);
  let qsS = 0.622 * esS / (PSFC - 0.378 * esS);
  F[3u * SIZE + q] += CK * spd * (tsk / PIS - S[3u * SIZE + q]) / DZ;
  if (MOIST) {
    var fq = CK * spd * (qsS - S[5u * SIZE + q]);
    if (fq > 0.0) { fq *= SF[NX * NY + t]; }
    F[5u * SIZE + q] += fq / DZ;
  }
}
`;

// open lateral boundaries: Davies relaxation toward the nesting targets in the outer NRELAX cells (or outside a
// cylinder: CYL, a two-way nest's ring)
// (u, v, theta, qv, pi'; w -> 0), and the lid sponge re-targeted from the base state to the 3-D targets
const RELAX_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> F: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read> base: array<f32>;
${BASE_FNS}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY * NZ) { return; }
  let k = t / (NX * NY); let r = t % (NX * NY); let j = r / NX; let i = r % NX;
  let q = ix(i, j, k);
  let rc = brc(k);
  if (rc > 0.0) {
    F[q] += rc * (B[q] - bub(k));
    F[SIZE + q] += rc * (B[SIZE + q] - bvb(k));
    F[3u * SIZE + q] += rc * (B[2u * SIZE + q] - bth0(k));
  }
  var x: f32;
  if (CYL) { x = min(1.0, (length(vec2<f32>(f32(i) + 0.5 - 0.5 * f32(NX), f32(j) + 0.5 - 0.5 * f32(NY))) - CYLR) / CYLW); }
  else { x = 1.0 - f32(min(min(i, j), min(NX - 1u - i, NY - 1u - j))) / f32(NRELAX); }
  if (x <= 0.0) { return; }
  let rr = x * x / RTAU;
  F[q] -= rr * (S[q] - B[q]);
  F[SIZE + q] -= rr * (S[SIZE + q] - B[SIZE + q]);
  F[3u * SIZE + q] -= rr * (S[3u * SIZE + q] - B[2u * SIZE + q]);
  if (MOIST) { F[5u * SIZE + q] -= rr * (S[5u * SIZE + q] - B[3u * SIZE + q]); }
  if (HASPP) { F[4u * SIZE + q] -= rr * (S[4u * SIZE + q] - B[4u * SIZE + q]); }
  F[2u * SIZE + q] -= rr * S[2u * SIZE + q];
}
`;

// RK3 stage: theta and moisture from time n + dtStage * F; restore u, v, w, pi' to time n; ppOld = pi'(n)
const stageWgsl = (cf: boolean): string => /* wgsl */`
@group(0) @binding(0) var<storage, read_write> S: array<f32>;
@group(0) @binding(1) var<storage, read> S0: array<f32>;
@group(0) @binding(2) var<storage, read> F: array<f32>;
@group(0) @binding(3) var<storage, read_write> A: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
${cf ? '@group(0) @binding(5) var<storage, read> CF: array<u32>;' : ''}
@group(0) @binding(${cf ? 6 : 5}) var<storage, read> base: array<f32>;
${BASE_FNS}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x;
  if (q >= SIZE) { return; }
  S[3u * SIZE + q] = S0[3u * SIZE + q] + p.dtStage * F[3u * SIZE + q];
  // the temperature floor (T_FLOOR in core/constants.ts): only an interaction asking for more cold than there is gets here
  { let kk = q / PL; if (kk < NZ) { S[3u * SIZE + q] = max(S[3u * SIZE + q], TFLOOR / bpi0(kk)); } }
  // hydrometeors of clear-air columns stay exactly 0 (S0 = F = 0 there): no update needed
  var fEnd = NFLD;
  ${cf ? `{
    let e = q % PL; let je = e / SX; let ie = e % SX;
    if (ie >= HH && ie < NX + HH && je >= HH && je < NY + HH && CF[2u * NX * NY + (je - HH) * NX + (ie - HH)] == 0u) { fEnd = 6u; }
  }` : ''}
  for (var f = 5u; f < fEnd; f++) { S[f * SIZE + q] = S0[f * SIZE + q] + p.dtStage * F[f * SIZE + q]; }
  S[q] = S0[q]; S[SIZE + q] = S0[SIZE + q]; S[2u * SIZE + q] = S0[2u * SIZE + q]; S[4u * SIZE + q] = S0[4u * SIZE + q];
  A[q] = S0[4u * SIZE + q];
}
`;

// condensate column flags (exact skipping of the hydrometeor species in clear air): raw flag per column
// (any cloud / precipitation species non-zero), then dilated by CF_R columns in x and y. Over one step the
// three RK stages can spread non-zero values by at most 3 columns each (5th-order stencil) and the last
// stage reads 3 more, so CF_R = 9 makes skipping exact: outside the flagged columns every hydrometeor
// tendency is exactly zero. Recomputed at the start of every step (after the microphysics).
const CF_R = 9;
const COLFLAG_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> CF: array<u32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY) { return; }
  let j = t / NX; let i = t % NX;
  var any = 0u;
  for (var k = 0u; k < NZ; k++) {
    let q = ix(i, j, k);
    for (var f = 6u; f < NFLD; f++) { if (S[f * SIZE + q] != 0.0) { any = 1u; } }
    if (any == 1u) { break; }
  }
  CF[t] = any;
}
`;
const dilateWgsl = (axis: 'x' | 'y'): string => /* wgsl */`
@group(0) @binding(0) var<storage, read_write> CF: array<u32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY) { return; }
  let j = i32(t / NX); let i = i32(t % NX);
  var any = 0u;
  for (var d = -${CF_R}; d <= ${CF_R}; d++) {
    ${axis === 'x'
      ? 'var ii = i + d; if (OPEN) { ii = clamp(ii, 0, i32(NX) - 1); } else { ii = (ii + i32(NX)) % i32(NX); } any |= CF[u32(j) * NX + u32(ii)];'
      : 'var jj = j + d; if (OPEN) { jj = clamp(jj, 0, i32(NY) - 1); } else { jj = (jj + i32(NY)) % i32(NY); } any |= CF[NX * NY + u32(jj) * NX + u32(i)];'}
  }
  CF[${axis === 'x' ? 'NX * NY' : '2u * NX * NY'} + t] = any;
}
`;

// density potential temperature theta (1 + 0.61 qv - q_condensate) of the interior cells, computed once per
// RK stage (theta and moisture are fixed during the acoustic substeps) instead of from seven fields at
// every use in the acoustic kernels
const THRHO_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> TR: array<f32>;
${THR_FNS}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY * NZ) { return; }
  let k = t / (NX * NY); let r = t % (NX * NY); let j = r / NX; let i = r % NX;
  let q = ix(i, j, k);
  TR[q] = thr(q);
}
`;

// acoustic step: horizontal momentum with divergence-damped pressure gradient
const ACOUSTIC_H_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> S: array<f32>;
@group(0) @binding(1) var<storage, read> F: array<f32>;
@group(0) @binding(2) var<storage, read> A: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<storage, read> TR: array<f32>;
fn thr(q: u32) -> f32 { return TR[q]; }
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

// acoustic step: vertically implicit w - pi' per column. Thomas algorithm with the coefficients formed on
// the fly and only three per-thread arrays (E, normalised c', d'): long per-thread arrays spill out of
// registers on real GPUs. Also stores the pre-update pi' in A (the previous-substep pi' used by the
// divergence damping of the next horizontal acoustic step; formerly a separate copy pass).
const ACOUSTIC_V_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> S: array<f32>;
@group(0) @binding(1) var<storage, read> F: array<f32>;
@group(0) @binding(2) var<storage, read> base: array<f32>;
${BASE_FNS}
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<storage, read_write> A: array<f32>;
@group(0) @binding(5) var<storage, read> TR: array<f32>;
fn thr(q: u32) -> f32 { return TR[q]; }
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY) { return; }
  let j = t / NX; let i = t % NX;
  let dts = p.dts;
  let bp = 0.5 * (1.0 + BETA); let bm = 0.5 * (1.0 - BETA);
  var Ek: array<f32, NZ>; var cp: array<f32, NZ + 1u>; var dp: array<f32, NZ + 1u>;
  for (var k = 0u; k < NZ; k++) {
    let q = ix(i, j, k);
    let dh = (S[q + 1u] - S[q]) / DX + (S[SIZE + q + SX] - S[SIZE + q]) / DY;
    let oldW = bm * (brtf(k + 1u) * S[2u * SIZE + q + PL] - brtf(k) * S[2u * SIZE + q]) / DZ;
    Ek[k] = S[4u * SIZE + q] + dts * F[4u * SIZE + q] - dts * bcfac(k) * (brtc(k) * dh + oldW);
  }
  // forward sweep over the interior w levels 1 .. NZ-1 (w = 0 at 0 and NZ)
  cp[0] = 0.0; dp[0] = 0.0;
  for (var k = 1u; k < NZ; k++) {
    let q = ix(i, j, k);
    let cth = CP * 0.5 * (thr(q) + thr(q - PL)) / DZ;
    let gk = dts * bcfac(k) * bp / DZ; let gkm = dts * bcfac(k - 1u) * bp / DZ;
    let rtk = brtf(k);
    let a = select(0.0, -dts * cth * bp * gkm * brtf(k - 1u), k - 1u >= 1u);
    let c = select(0.0, -dts * cth * bp * gk * brtf(k + 1u), k + 1u <= NZ - 1u);
    let b = 1.0 + dts * cth * bp * (gk * rtk + gkm * rtk);
    let r = S[2u * SIZE + q] + dts * (F[2u * SIZE + q] - cth * (bp * (Ek[k] - Ek[k - 1u]) + bm * (S[4u * SIZE + q] - S[4u * SIZE + q - PL])));
    let den = b - a * cp[k - 1u];
    cp[k] = c / den;
    dp[k] = (r - a * dp[k - 1u]) / den;
  }
  // back substitution: dp[k] becomes w[k]
  dp[NZ] = 0.0;
  for (var kk = i32(NZ) - 1; kk >= 1; kk--) { let k = u32(kk); dp[k] = dp[k] - cp[k] * dp[k + 1u]; }
  dp[0] = 0.0;
  for (var k = 0u; k < NZ; k++) {
    let q = ix(i, j, k);
    A[q] = S[4u * SIZE + q];
    S[4u * SIZE + q] = Ek[k] - dts * bcfac(k) * bp * (brtf(k + 1u) * dp[k + 1u] - brtf(k) * dp[k]) / DZ;
  }
  for (var k = 0u; k <= NZ; k++) { S[2u * SIZE + ix(i, j, k)] = dp[k]; }
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
    let pi = max(bpi0(k) + S[4u * SIZE + q], 0.05);
    let pr = 1.0e5 * pow(pi, CP / RD);
    let T = max(S[3u * SIZE + q] * pi, TFLOOR);
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

// Six-class ice microphysics per column: WGSL port of src/regional/ice.ts (see there for references).
const iceWgsl = (): string => {
  const c = (x: number): string => (Number.isInteger(x) ? `${x}.0` : `${x}`);
  const G3R = gammaFn(3 + ICE.BR), G3S = gammaFn(3 + ICE.BS), G3G = gammaFn(3 + ICE.BG), G4S = gammaFn(4 + ICE.BS), G4G = gammaFn(4 + ICE.BG);
  const G5S = gammaFn((ICE.BS + 5) / 2), G5G = gammaFn((ICE.BG + 5) / 2);
  return /* wgsl */`
@group(0) @binding(0) var<storage, read_write> S: array<f32>;
@group(0) @binding(1) var<storage, read> base: array<f32>;
${BASE_FNS}
@group(0) @binding(2) var<storage, read_write> A: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
const T0: f32 = ${c(ICE.T0)}; const LVI: f32 = ${c(ICE.LV)}; const LSI: f32 = ${c(ICE.LS)}; const LFI: f32 = ${c(LF)}; const RV: f32 = ${c(ICE.RV)};
const RHOW: f32 = 1000.0; const N0R: f32 = ${c(ICE.N0R)}; const AR: f32 = ${c(ICE.AR)}; const BR: f32 = ${c(ICE.BR)};
const RHOS: f32 = ${c(ICE.RHOS)}; const AS: f32 = ${c(ICE.AS)}; const BS: f32 = ${c(ICE.BS)};
const N0G: f32 = ${c(ICE.N0G)}; const RHOG: f32 = ${c(ICE.RHOG)}; const AG: f32 = ${c(ICE.AG)}; const BG: f32 = ${c(ICE.BG)};
const KAIR: f32 = ${c(ICE.KA)}; const MUA: f32 = ${c(ICE.MU)}; const MI0: f32 = ${c(ICE.MI0)}; const QI0: f32 = ${c(ICE.QI0)}; const QS0: f32 = ${c(ICE.QS0)};
const BIGGB: f32 = ${c(ICE.BIGG_B)}; const BIGGA: f32 = ${c(ICE.BIGG_A)};
const G3R: f32 = ${G3R}; const G3S: f32 = ${G3S}; const G3G: f32 = ${G3G}; const G4S: f32 = ${G4S}; const G4G: f32 = ${G4G}; const G5S: f32 = ${G5S}; const G5G: f32 = ${G5G};
const PI_: f32 = 3.14159265;
fn lam(rhox: f32, n0: f32, rq: f32) -> f32 { return pow(PI_ * rhox * n0 / max(rq, 1e-15), 0.25); }
fn n0snow(T: f32) -> f32 { return min(2e8, 2e6 * exp(0.12 * max(0.0, T0 - T))); }
fn inum(rqi: f32) -> f32 { return min(1e6, max(1e3, 5.38e7 * pow(max(rqi, 1e-20), 0.75))); }
var<private> KA1: array<f32, 31> = array<f32, 31>(${KOENIG_A1.map((v) => v.toExponential(4)).join(', ')});
var<private> KA2: array<f32, 31> = array<f32, 31>(${KOENIG_A2.map((v) => v.toFixed(4)).join(', ')});
// Bergeron-process snow production (bergeron() in src/regional/ice.ts): (psfi, psfw)
fn bergeron(T: f32, qi: f32, qc: f32, rho: f32, dt: f32) -> vec2<f32> {
  if (!(T < T0 && T > T0 - 31.0) || qc <= 1e-8 || qi <= 1e-12) { return vec2<f32>(0.0, 0.0); }
  let it = u32(clamp(round(T0 - T) - 1.0, 0.0, 30.0));
  let a1 = KA1[it]; let a2 = KA2[it];
  let dt1 = (pow(4.8e-7, 1.0 - a2) - pow(2.46e-7, 1.0 - a2)) / (a1 * (1.0 - a2));
  let ni50 = qi / 4.8e-10 * min(1.0, dt / dt1);
  return vec2<f32>(qi / dt1, ni50 * (a1 * pow(4.8e-7, a2) * 1e-3 + PI_ * rho * qc * 2.5e-9));
}
fn qvsw(T: f32, pr: f32) -> f32 { return 380.0 / pr * exp(17.27 * (T - 273.15) / (T - 35.86)); }
fn qvsi(T: f32, pr: f32) -> f32 { return 380.0 / pr * exp(21.875 * (T - 273.15) / (T - 7.66)); }
fn vfall(sp: u32, q: f32, rho: f32, rhoS: f32, T: f32) -> f32 {
  if (q <= 1e-12) { return 0.0; }
  let rq = rho * q; let dens = sqrt(rhoS / rho);
  if (sp == 7u) { return 36.34 * pow(0.001 * rq, 0.1364) * dens; }
  if (sp == 9u) { let l = lam(RHOS, n0snow(T), rq); return AS * G4S / (6.0 * pow(l, BS)) * dens; }
  if (sp == 10u) { let l = lam(RHOG, N0G, rq); return AG * G4G / (6.0 * pow(l, BG)) * dens; }
  let ni = inum(rq); let di = 11.9 * sqrt(rq / ni);
  return min(1.49e4 * pow(di, 1.31), 3.0);
}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NX * NY) { return; }
  let j = t / NX; let i = t % NX;
  let dt = p.dtBig;
  let rhoS = brho0(0u);
  let rq0 = ix(i, j, 0u);
  var col: array<f32, NZ>; var vt: array<f32, NZ>; var fl: array<f32, NZ + 1u>;
  for (var k = 0u; k < NZ; k++) {
    let q = ix(i, j, k);
    for (var f = 5u; f < 11u; f++) { S[f * SIZE + q] = max(S[f * SIZE + q], 0.0); }
  }
  // ---- sedimentation: rain, snow, graupel, cloud ice
  for (var n = 0u; n < 4u; n++) {
    let sp = select(select(select(8u, 10u, n == 2u), 9u, n == 1u), 7u, n == 0u);
    var vmax = 0.0;
    for (var k = 0u; k < NZ; k++) {
      let q = ix(i, j, k);
      col[k] = S[sp * SIZE + q];
      vt[k] = vfall(sp, col[k], brho0(k), rhoS, S[3u * SIZE + q] * (bpi0(k) + S[4u * SIZE + q]));
      vmax = max(vmax, vt[k]);
    }
    if (vmax == 0.0) { continue; }
    let nsub = max(1u, u32(ceil(vmax * dt / (0.8 * DZ))));
    let dts = dt / f32(nsub);
    for (var s = 0u; s < nsub; s++) {
      for (var k = 0u; k < NZ; k++) { fl[k] = brho0(k) * vt[k] * col[k]; }
      fl[NZ] = 0.0;
      A[2u * SIZE + rq0] += fl[0] * dts;
      if (sp != 7u) { A[3u * SIZE + rq0] += fl[0] * dts; }
      for (var k = 0u; k < NZ; k++) { col[k] += dts * (fl[k + 1u] - fl[k]) / (brho0(k) * DZ); }
    }
    for (var k = 0u; k < NZ; k++) { S[sp * SIZE + ix(i, j, k)] = max(0.0, col[k]); }
  }
  // ---- local processes
  for (var k = 0u; k < NZ; k++) {
    let q = ix(i, j, k);
    let rho = brho0(k);
    let pi = max(bpi0(k) + S[4u * SIZE + q], 0.05);
    let pr = 1.0e5 * pow(pi, CP / RD);
    let hv = LVI / (CP * pi); let hs = LSI / (CP * pi); let hf = LFI / (CP * pi);
    var th = S[3u * SIZE + q];
    var qv = S[5u * SIZE + q]; var qc = S[6u * SIZE + q]; var qr = S[7u * SIZE + q];
    var qi = S[8u * SIZE + q]; var qs = S[9u * SIZE + q]; var qg = S[10u * SIZE + q];
    var T = max(th * pi, TFLOOR);
    if (T > T0 && qi > 0.0) { qc += qi; th -= hf * qi; qi = 0.0; T = max(th * pi, TFLOOR); }
    if (T < T0 - 40.0) {
      if (qc > 0.0) { qi += qc; th += hf * qc; qc = 0.0; }
      if (qr > 0.0) { qg += qr; th += hf * qr; qr = 0.0; }
      T = max(th * pi, TFLOOR);
    }
    let dens = sqrt(rhoS / rho); let dens4 = sqrt(dens);
    let psi = 2.26e-5 * pow(T / T0, 1.81) * (1e5 / pr);
    let nu = MUA / rho; let sc3 = pow(nu / psi, 1.0 / 3.0);
    let qsi = qvsi(T, pr); let si = qv / qsi - 1.0;
    let Ai = (LSI / (RV * T) - 1.0) * LSI / (KAIR * T); let Bi = 1.0 / (psi * rho * qsi);
    let n0s = n0snow(T);
    var lS = 0.0; var lG = 0.0; var lR = 0.0;
    if (qs > 1e-12) { lS = lam(RHOS, n0s, rho * qs); }
    if (qg > 1e-12) { lG = lam(RHOG, N0G, rho * qg); }
    if (qr > 1e-12) { lR = lam(RHOW, N0R, rho * qr); }
    var kS = 0.0; var kG = 0.0; var kR = 0.0; var vS = 0.0; var vG = 0.0;
    if (lS > 0.0) { kS = PI_ / 4.0 * n0s * AS * G3S / pow(lS, 3.0 + BS) * dens; vS = 0.78 / (lS * lS) + 0.31 * sc3 * sqrt(AS / nu) * G5S * dens4 / pow(lS, (BS + 5.0) / 2.0); }
    if (lG > 0.0) { kG = PI_ / 4.0 * N0G * AG * G3G / pow(lG, 3.0 + BG) * dens; vG = 0.78 / (lG * lG) + 0.31 * sc3 * sqrt(AG / nu) * G5G * dens4 / pow(lG, (BG + 5.0) / 2.0); }
    if (lR > 0.0) { kR = PI_ / 4.0 * N0R * AR * G3R / pow(lR, 3.0 + BR) * dens; }
    var pidep = 0.0; var pigen = 0.0; var psdep = 0.0; var pgdep = 0.0; var psaut = 0.0; var pgaut = 0.0;
    var psaci = 0.0; var pgaci = 0.0; var praci = 0.0; var psacw = 0.0; var pgacw = 0.0; var pgfrz = 0.0; var psmlt = 0.0; var pgmlt = 0.0; var psfi = 0.0; var psfw = 0.0;
    let cold = T < T0;
    if (cold) {
      let eci = exp(0.05 * (T - T0));
      if (si > 0.0) {
        let nNuc = min(1e6, 1e3 * exp(0.1 * (T0 - T)));
        pigen = max(0.0, min(MI0 * nNuc / rho - qi, (qv - qsi) / (1.0 + LSI * LSI * qsi / (CP * RV * T * T)))) / dt;
      }
      if (qi > 1e-12) {
        let rqi = rho * qi; let ni = inum(rqi); let di = 11.9 * sqrt(rqi / ni);
        pidep = 4.0 * di * ni * si / (rho * (Ai + Bi));
        psaut = max(0.0, 1e-3 * exp(0.025 * (T - T0)) * (qi - QI0));
        let bg = bergeron(T, qi, qc, rho, dt); psfi = bg.x; psfw = bg.y;
        psaci = kS * eci * qi; pgaci = kG * eci * qi; praci = kR * qi;
      }
      if (lS > 0.0) { psdep = 4.0 * n0s * si * vS / (rho * (Ai + Bi)); }
      if (lG > 0.0) { pgdep = 2.0 * PI_ * N0G * si * vG / (rho * (Ai + Bi)); }
      if (qs > QS0) { pgaut = 1e-3 * exp(0.09 * (T - T0)) * (qs - QS0); }
      psacw = kS * qc; pgacw = kG * qc;
      if (lR > 0.0) { pgfrz = 20.0 * PI_ * PI_ * BIGGB * N0R * (RHOW / rho) * (exp(BIGGA * (T0 - T)) - 1.0) / pow(lR, 7.0); }
    } else {
      if (lS > 0.0) { psmlt = 2.0 * PI_ * n0s * KAIR * (T - T0) * vS / (rho * LFI); }
      if (lG > 0.0) { pgmlt = 2.0 * PI_ * N0G * KAIR * (T - T0) * vG / (rho * LFI); }
      psacw = kS * qc; pgacw = kG * qc;
    }
    let vSink = max(pidep, 0.0) + pigen + max(psdep, 0.0) + max(pgdep, 0.0);
    let fv = select(1.0, qv / (vSink * dt), vSink * dt > qv);
    let iSink = max(-pidep, 0.0) + psaut + psfi + psaci + pgaci + praci;
    let fi = select(1.0, qi / (iSink * dt), iSink * dt > qi);
    let sSink = max(-psdep, 0.0) + pgaut + psmlt;
    let fs = select(1.0, qs / (sSink * dt), sSink * dt > qs);
    let gSink = max(-pgdep, 0.0) + pgmlt;
    let fg = select(1.0, qg / (gSink * dt), gSink * dt > qg);
    let cSink = psacw + psfw + pgacw;
    let fc = select(1.0, qc / (cSink * dt), cSink * dt > qc);
    let fr = select(1.0, qr / (pgfrz * dt), pgfrz * dt > qr);
    let t_vi = (max(pidep, 0.0) + pigen) * fv * dt; let t_iv = max(-pidep, 0.0) * fi * dt;
    let t_vs = max(psdep, 0.0) * fv * dt; let t_sv = max(-psdep, 0.0) * fs * dt;
    let t_vg = max(pgdep, 0.0) * fv * dt; let t_gv = max(-pgdep, 0.0) * fg * dt;
    let t_is = (psaut + psfi + psaci) * fi * dt; let t_ig = (pgaci + praci) * fi * dt;
    let t_sg = pgaut * fs * dt; let t_sr = psmlt * fs * dt; let t_gr = pgmlt * fg * dt;
    let t_cs = (psacw + psfw) * fc * dt; let t_cg = pgacw * fc * dt; let t_rg = pgfrz * fr * dt;
    let cf = select(0.0, 1.0, cold);
    qv += -t_vi + t_iv - t_vs + t_sv - t_vg + t_gv;
    qi += t_vi - t_iv - t_is - t_ig;
    qs += t_vs - t_sv + t_is - t_sg - t_sr + cf * t_cs;
    qg += t_vg - t_gv + t_ig + t_sg - t_gr + cf * t_cg + t_rg;
    qc -= t_cs + t_cg;
    qr += t_sr + t_gr - t_rg + (1.0 - cf) * (t_cs + t_cg);
    th += hs * (t_vi - t_iv + t_vs - t_sv + t_vg - t_gv) + hf * (cf * (t_cs + t_cg) + t_rg - t_sr - t_gr);
    qv = max(qv, 0.0); qi = max(qi, 0.0); qs = max(qs, 0.0); qg = max(qg, 0.0); qc = max(qc, 0.0); qr = max(qr, 0.0);
    // warm rain + saturation adjustment (Kessler)
    T = max(th * pi, TFLOOR);
    var factorn = 1.0;
    if (qr > 0.0) { factorn = 1.0 / (1.0 + 2.2 * dt * pow(qr, 0.875)); }
    let qrprod = qc - (qc - dt * max(0.001 * (qc - 0.001), 0.0)) * factorn;
    qc = max(qc - qrprod, 0.0);
    qr += qrprod;
    let qvs = qvsw(T, pr);
    let f5c = 237.3 * 17.27 * LVI / CP;
    var prod = (qv - qvs) / (1.0 + qvs * f5c / ((T - 35.86) * (T - 35.86)));
    if (T < T0 - 40.0) { prod = min(prod, 0.0); }
    let rqq = max(rho * qr, 0.0);
    var ern = 0.0;
    if (rqq > 0.0) {
      ern = min(min(dt * (((1.6 + 124.9 * pow(rqq, 0.2046)) * pow(rqq, 0.525)) / (2.55e8 / (pr * qvs) + 5.4e5)) * (max(qvs - qv, 0.0) / (rho * qvs)), max(-prod - qc, 0.0)), qr);
    }
    let product = max(prod, -qc);
    th += hv * (product - ern);
    qv = max(qv - product + ern, 0.0);
    qc = qc + product;
    qr = max(qr - ern, 0.0);
    if (T < T0 - 40.0) {
      let Tn = max(th * pi, TFLOOR); let qsi2 = qvsi(Tn, pr);
      if (qv > qsi2) { let d = (qv - qsi2) / (1.0 + LSI * LSI * qsi2 / (CP * RV * Tn * Tn)); qv -= d; qi += d; th += hs * d; }
    }
    S[3u * SIZE + q] = th;
    S[5u * SIZE + q] = qv; S[6u * SIZE + q] = qc; S[7u * SIZE + q] = qr;
    S[8u * SIZE + q] = qi; S[9u * SIZE + q] = qs; S[10u * SIZE + q] = qg;
  }
}
`;
};
