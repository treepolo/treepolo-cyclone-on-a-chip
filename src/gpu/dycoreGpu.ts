// GPU version of the spectral primitive-equation dynamical core (f32), mirroring src/model/dycore.ts.
// One model step = one command-encoder submission:
//   1. psi/chi from vorticity/divergence              (spectral kernel)
//   2. Legendre synthesis + inverse FFT               (U, V, zeta, D, T per level; ln ps and its gradient)
//   3. grid-point dynamics per column                 (+ optional Held–Suarez forcing)
//   4. forward FFT + Legendre analysis of tendencies
//   5. semi-implicit solve per spectral coefficient   (+ del^8 hyperdiffusion, RAW time filter)
// The CPU Float64 core remains the reference; tests compare the two.

import { Dycore } from '../model/dycore.js';
import { GpuTransform, Recipe } from './transformGpu.js';
import { HS94 } from '../model/heldSuarez.js';

const WG = 64;

export interface GpuDycoreOptions { heldSuarez: boolean }

export class GpuDycore {
  readonly device: GPUDevice;
  readonly tr: GpuTransform;
  readonly cpu: Dycore;          // configuration + reference implementation
  readonly K: number;
  readonly nspec: number;
  readonly ng: number;
  readonly nState: number;       // spectral fields per state buffer (vor, div, tmp, lnps, psi, chi)
  steps = 0;
  time = 0;
  private spec: GPUBuffer[];     // three rotating state buffers
  private rot = 0;               // rotation index: old = spec[rot], cur = spec[rot+1], nxt = spec[rot+2]
  private readonly four: GPUBuffer; private readonly grid: GPUBuffer;
  private readonly fourT: GPUBuffer; private readonly gridT: GPUBuffer; private readonly tend: GPUBuffer;
  readonly sdot: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly ops: ((pass: GPUComputePassEncoder) => void)[][] = [];   // per rotation
  private readonly siFirst: GPUBuffer; private readonly siNormal: GPUBuffer;

  constructor(device: GPUDevice, cpu: Dycore, opts: GpuDycoreOptions) {
    this.device = device;
    this.cpu = cpu;
    const tr = new GpuTransform(device, cpu.tr);
    this.tr = tr;
    const K = cpu.K, nspec = cpu.tr.nspec, ng = cpu.ng, T = cpu.tr.trunc;
    this.K = K; this.nspec = nspec; this.ng = ng;
    this.nState = 5 * K + 1;
    this.spec = [0, 1, 2].map(() => tr.specBuffer(this.nState));
    const nSyn = 5 * K + 3, nAn = 6 * K + 1;
    this.four = tr.fourierBuffer(nSyn); this.grid = tr.gridBuffer(nSyn);
    this.fourT = tr.fourierBuffer(nAn); this.gridT = tr.gridBuffer(nAn);
    this.tend = tr.specBuffer(4 * K + 1);
    this.sdot = tr.gridBuffer(K);
    const lev = cpu.lev, a = cpu.planet.radius;

    // ---- static tables
    const levTab = new Float32Array(6 * (K + 1));
    for (let k = 0; k <= K; k++) levTab[k] = lev.sigmaHalf[k]!;
    for (let k = 0; k < K; k++) {
      levTab[(K + 1) + k] = lev.dsigma[k]!;
      levTab[2 * (K + 1) + k] = k === 0 ? 0 : lev.lnRatio[k]!;
      levTab[3 * (K + 1) + k] = lev.alpha[k]!;
      levTab[4 * (K + 1) + k] = lev.sigma[k]!;
    }
    const levels = tr.upload(levTab);
    const rowTab = new Float32Array(2 * cpu.tr.nlat);
    for (let j = 0; j < cpu.tr.nlat; j++) { rowTab[j] = cpu.tr.mu[j]!; rowTab[cpu.tr.nlat + j] = cpu.tr.coslat[j]!; }
    const rows = tr.upload(rowTab);
    const mats = new Float32Array(2 * K * K + K);
    mats.set(Float32Array.from(lev.G), 0); mats.set(Float32Array.from(lev.tau), K * K); mats.set(Float32Array.from(lev.dsigma), 2 * K * K);
    const matBuf = tr.upload(mats);
    const specTab = new Float32Array(3 * nspec);
    const hd = (cpu as unknown as { hyperdiff: Float64Array }).hyperdiff;
    for (let s = 0; s < nspec; s++) { specTab[s] = cpu.tr.nn1[s]!; specTab[nspec + s] = hd[s]!; specTab[2 * nspec + s] = cpu.tr.nOf[s]!; }
    const specInfo = tr.upload(specTab);
    const siMat = (h: number): GPUBuffer => tr.upload(Float32Array.from((cpu as unknown as { siMatrices(h: number): Float64Array }).siMatrices(h)));
    this.siFirst = siMat(cpu.dt / 2);
    this.siNormal = siMat(cpu.dt);
    const phisData = new Float32Array(2 * nspec);
    for (let s = 0; s < nspec; s++) { phisData[2 * s] = cpu.phis.re[s]!; phisData[2 * s + 1] = cpu.phis.im[s]!; }
    const phis = tr.upload(phisData);
    // params: [leap, h, radius, R*Tref, nu, alpha, filterOn, omega, rd, cp, kappa, hs, pRef, 0,0,0]
    this.params = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const consts = `
const K: u32 = ${K}u;
const NSPEC: u32 = ${nspec}u;
const NLAT: u32 = ${cpu.tr.nlat}u;
const NLON: u32 = ${cpu.tr.nlon}u;
const NG: u32 = ${ng}u;
const T: u32 = ${T}u;
const NSTATE: u32 = ${this.nState}u;
const TREF: f32 = ${lev.tRef};
struct Params { leap: f32, h: f32, radius: f32, RT: f32, nu: f32, alpha: f32, filterOn: f32, omega: f32,
                rd: f32, cp: f32, kappa: f32, hs: f32, pRef: f32, p1: f32, p2: f32, p3: f32 };
`;
    const pipe = (code: string): GPUComputePipeline => device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: consts + code }), entryPoint: 'main' } });
    const pPsi = pipe(PSI_WGSL), pGrid = pipe(gridWgsl(opts.heldSuarez)), pSI = pipe(SI_WGSL);

    // ---- recipes
    const ia = 1 / a;
    const synth: Recipe[] = [];
    for (let k = 0; k < K; k++) {
      const psi = 3 * K + 1 + k, chi = 4 * K + 1 + k;
      synth.push({ output: k, terms: [{ input: chi, kind: 'P', im: true, scale: ia }, { input: psi, kind: 'H', im: false, scale: -ia }] });
      synth.push({ output: K + k, terms: [{ input: psi, kind: 'P', im: true, scale: ia }, { input: chi, kind: 'H', im: false, scale: ia }] });
      synth.push({ output: 2 * K + k, terms: [{ input: k, kind: 'P', im: false, scale: 1 }] });
      synth.push({ output: 3 * K + k, terms: [{ input: K + k, kind: 'P', im: false, scale: 1 }] });
      synth.push({ output: 4 * K + k, terms: [{ input: 2 * K + k, kind: 'P', im: false, scale: 1 }] });
    }
    synth.push({ output: 5 * K, terms: [{ input: 3 * K, kind: 'P', im: false, scale: 1 }] });
    synth.push({ output: 5 * K + 1, terms: [{ input: 3 * K, kind: 'P', im: true, scale: 1 }] });
    synth.push({ output: 5 * K + 2, terms: [{ input: 3 * K, kind: 'H', im: false, scale: 1 }] });
    // analysis input slots: A 0.., B K.., UT 2K.., VT 3K.., E 4K.., TT 5K.., Np 6K
    const anal: Recipe[] = [];
    for (let k = 0; k < K; k++) {
      const A = k, B = K + k, UT = 2 * K + k, VT = 3 * K + k, E = 4 * K + k, TT = 5 * K + k;
      anal.push({ output: k, terms: [{ input: B, kind: 'P', im: true, scale: ia }, { input: A, kind: 'H', im: false, scale: ia }] });
      anal.push({ output: K + k, terms: [{ input: A, kind: 'P', im: true, scale: ia }, { input: B, kind: 'H', im: false, scale: -ia }] });
      anal.push({ output: 2 * K + k, terms: [{ input: TT, kind: 'P', im: false, scale: 1 }, { input: UT, kind: 'P', im: true, scale: -ia }, { input: VT, kind: 'H', im: false, scale: ia }] });
      anal.push({ output: 3 * K + 1 + k, terms: [{ input: E, kind: 'P', im: false, scale: 1 }] });
    }
    anal.push({ output: 3 * K, terms: [{ input: 6 * K, kind: 'P', im: false, scale: 1 }] });
    const fwdScaled = tr.prepareFftForward(this.gridT, this.fourT, 4 * K, 0, 0, true);
    const fwdPlain = tr.prepareFftForward(this.gridT, this.fourT, 2 * K + 1, 4 * K, 4 * K, false);
    const analOp = tr.prepareAnal(this.fourT, this.tend, anal);
    const invOp = tr.prepareFftInverse(this.four, this.grid, nSyn);
    const gridBG = device.createBindGroup({ layout: pGrid.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.grid } }, { binding: 1, resource: { buffer: this.gridT } },
      { binding: 2, resource: { buffer: levels } }, { binding: 3, resource: { buffer: rows } },
      { binding: 4, resource: { buffer: this.sdot } }, { binding: 5, resource: { buffer: this.params } },
    ] });
    const gridOp = (pass: GPUComputePassEncoder): void => { pass.setPipeline(pGrid); pass.setBindGroup(0, gridBG); pass.dispatchWorkgroups(Math.ceil(ng / WG)); };

    for (let r = 0; r < 3; r++) {
      const old = this.spec[r]!, cur = this.spec[(r + 1) % 3]!, nxt = this.spec[(r + 2) % 3]!;
      const psiBG = device.createBindGroup({ layout: pPsi.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: cur } }, { binding: 1, resource: { buffer: specInfo } }, { binding: 2, resource: { buffer: this.params } },
      ] });
      const synthOp = tr.prepareSynth(cur, this.four, synth);
      const mkSI = (mat: GPUBuffer): GPUBindGroup => device.createBindGroup({ layout: pSI.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: old } }, { binding: 1, resource: { buffer: cur } }, { binding: 2, resource: { buffer: nxt } },
        { binding: 3, resource: { buffer: this.tend } }, { binding: 4, resource: { buffer: matBuf } }, { binding: 5, resource: { buffer: mat } },
        { binding: 6, resource: { buffer: specInfo } }, { binding: 7, resource: { buffer: phis } }, { binding: 8, resource: { buffer: this.params } },
      ] });
      const siBGn = mkSI(this.siNormal), siBGf = mkSI(this.siFirst);
      const siOp = (first: boolean) => (pass: GPUComputePassEncoder): void => { pass.setPipeline(pSI); pass.setBindGroup(0, first ? siBGf : siBGn); pass.dispatchWorkgroups(Math.ceil(nspec / WG)); };
      const psiOp = (pass: GPUComputePassEncoder): void => { pass.setPipeline(pPsi); pass.setBindGroup(0, psiBG); pass.dispatchWorkgroups(Math.ceil(nspec / WG)); };
      this.ops.push([psiOp, synthOp, invOp, gridOp, fwdScaled, fwdPlain, analOp, siOp(false), siOp(true)]);
    }
    void HS94;
  }

  private writeParams(first: boolean): void {
    const c = this.cpu, lev = c.lev;
    const leap = first ? c.dt : 2 * c.dt;
    const robert = (c as unknown as { robert: number }).robert, williams = (c as unknown as { williams: number }).williams;
    const p = new Float32Array([leap, leap / 2, c.planet.radius, c.air.rd * lev.tRef, robert, williams, first ? 0 : 1, c.planet.omega,
      c.air.rd, c.air.cp, c.air.kappa, 1, c.air.pRef, 0, 0, 0]);
    this.device.queue.writeBuffer(this.params, 0, p);
  }

  /** Copy the CPU model's spectral state (old and current levels) to the GPU. */
  uploadFrom(cpu: Dycore): void {
    const nspec = this.nspec, K = this.K;
    const pack = (st: { vor: { re: Float64Array; im: Float64Array }[]; div: { re: Float64Array; im: Float64Array }[]; tmp: { re: Float64Array; im: Float64Array }[]; lnps: { re: Float64Array; im: Float64Array } }): Float32Array => {
      const d = new Float32Array(this.nState * nspec * 2);
      const put = (f: number, x: { re: Float64Array; im: Float64Array }): void => { for (let s = 0; s < nspec; s++) { d[(f * nspec + s) * 2] = x.re[s]!; d[(f * nspec + s) * 2 + 1] = x.im[s]!; } };
      for (let k = 0; k < K; k++) { put(k, st.vor[k]!); put(K + k, st.div[k]!); put(2 * K + k, st.tmp[k]!); }
      put(3 * K, st.lnps);
      return d;
    };
    this.rot = 0;
    this.tr.write(this.spec[0]!, pack(cpu.old));
    this.tr.write(this.spec[1]!, pack(cpu.cur));
    this.steps = cpu.steps;
    this.time = cpu.time;
  }

  /** Advance n steps (encoded into as few submissions as practical). */
  step(n = 1): void {
    for (let i = 0; i < n; i++) {
      const first = this.steps === 0;
      this.writeParams(first);
      const ops = this.ops[this.rot]!;
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (let o = 0; o < 7; o++) ops[o]!(pass);
      (first ? ops[8]! : ops[7]!)(pass);
      pass.end();
      this.device.queue.submit([enc.finish()]);
      this.rot = (this.rot + 1) % 3;
      this.steps++;
      this.time += this.cpu.dt;
    }
  }

  /** Read the current spectral state back (f32). Layout: field-major [5K+1][nspec] complex. */
  async readCurrent(): Promise<Float32Array> {
    return this.tr.read(this.spec[(this.rot + 1) % 3]!, this.nState * this.nspec * 8);
  }

  /** Grid-point u, v (m/s), T (K) of level k and ps (Pa) from the last synthesis (time level of the last step's input). */
  async readGrid(): Promise<Float32Array> { return this.tr.read(this.grid, (5 * this.K + 3) * this.ng * 4); }
}

// psi = -a^2/n(n+1) vor, chi = -a^2/n(n+1) div, into the psi/chi slots of the current state
const PSI_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> st: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> info: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s >= NSPEC) { return; }
  let nn = info[s];
  var f = 0.0;
  if (nn > 0.0) { f = -P.radius * P.radius / nn; }
  for (var k = 0u; k < K; k++) {
    st[(3u * K + 1u + k) * NSPEC + s] = st[k * NSPEC + s] * f;
    st[(4u * K + 1u + k) * NSPEC + s] = st[(K + k) * NSPEC + s] * f;
  }
}
`;

function gridWgsl(heldSuarez: boolean): string {
  const hs = heldSuarez ? `
      // Held–Suarez (1994) forcing, sigma = p / p_surface
      let sig = lev[4u * (K + 1u) + k];
      let bl = max(0.0, (sig - ${HS94.sigmaB}) / (1.0 - ${HS94.sigmaB}));
      let kv = ${HS94.kf} * bl;
      let kt = ${HS94.ka} + (${HS94.ks} - ${HS94.ka}) * bl * c2 * c2;
      let lnp = log(sig * ps / P.pRef);
      let teq = max(${HS94.Tmin}.0, (${HS94.T0}.0 - ${HS94.deltaTy}.0 * mu * mu - ${HS94.deltaThetaZ}.0 * lnp * c2) * exp(P.kappa * lnp));
      fu = -kv * U;
      fv = -kv * V;
      ft = -kt * (Tk - teq);` : '';
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read_write> O: array<f32>;
@group(0) @binding(2) var<storage, read> lev: array<f32>;
@group(0) @binding(3) var<storage, read> rows: array<f32>;
@group(0) @binding(4) var<storage, read_write> sdotOut: array<f32>;
@group(0) @binding(5) var<uniform> P: Params;

fn g(f: u32, p: u32) -> f32 { return G[f * NG + p]; }

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= NG) { return; }
  let j = p / NLON;
  let mu = rows[j];
  let c2 = 1.0 - mu * mu;
  let a = P.radius;
  let coef = 1.0 / (a * c2);
  let f = 2.0 * P.omega * mu;
  let lps = g(5u * K, p);
  let ps = exp(lps);
  let dl = g(5u * K + 1u, p);
  let dm = g(5u * K + 2u, p);
  var vgp: array<f32, K>;
  var C: array<f32, K + 1u>;
  var sd: array<f32, K + 1u>;
  C[0] = 0.0;
  var np = 0.0;
  for (var k = 0u; k < K; k++) {
    let ds = lev[(K + 1u) + k];
    vgp[k] = (g(k, p) * dl + g(K + k, p) * dm) * coef;
    C[k + 1u] = C[k] + ds * (g(3u * K + k, p) + vgp[k]);
    np -= ds * vgp[k];
  }
  O[6u * K * NG + p] = np;
  let Ctot = C[K];
  sd[0] = 0.0; sd[K] = 0.0;
  for (var k = 1u; k < K; k++) { sd[k] = lev[k] * Ctot - C[k]; }
  for (var k = 0u; k < K; k++) {
    let ds = lev[(K + 1u) + k];
    let lnR = lev[2u * (K + 1u) + k];
    let al = lev[3u * (K + 1u) + k];
    let ku = select(k - 1u, k, k == 0u);
    let kd = select(k + 1u, k, k == K - 1u);
    let i2 = 0.5 / ds;
    let sT = sd[k];
    let sB = sd[k + 1u];
    let U = g(k, p); let V = g(K + k, p); let Z = g(2u * K + k, p); let D = g(3u * K + k, p); let Tk = g(4u * K + k, p);
    let vaU = (sB * (g(kd, p) - U) + sT * (U - g(ku, p))) * i2;
    let vaV = (sB * (g(K + kd, p) - V) + sT * (V - g(K + ku, p))) * i2;
    let vaT = (sB * (g(4u * K + kd, p) - Tk) + sT * (Tk - g(4u * K + ku, p))) * i2;
    var lnC = 0.0;
    if (k > 0u) { lnC = lnR * C[k]; }
    let DG = D + vgp[k];
    let omp = vgp[k] - (lnC + al * ds * DG) / ds;
    let Tp = Tk - TREF;
    let absv = Z + f;
    var fu = 0.0; var fv = 0.0; var ft = 0.0;
    ${hs}
    O[k * NG + p] = absv * V - vaU - P.rd * Tp * dl / a + fu;
    O[(K + k) * NG + p] = -absv * U - vaV - P.rd * Tp * dm / a + fv;
    O[(2u * K + k) * NG + p] = U * Tp;
    O[(3u * K + k) * NG + p] = V * Tp;
    O[(4u * K + k) * NG + p] = 0.5 * (U * U + V * V) / c2;
    O[(5u * K + k) * NG + p] = Tp * D - vaT + P.kappa * Tk * omp + ft;
    sdotOut[k * NG + p] = 0.5 * (sT + sB);
  }
}
`;
}

// semi-implicit solve per spectral coefficient; hyperdiffusion; Robert–Asselin–Williams filter
const SI_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> old: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> cur: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> nxt: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> td: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> mats: array<f32>;
@group(0) @binding(5) var<storage, read> minv: array<f32>;
@group(0) @binding(6) var<storage, read> info: array<f32>;
@group(0) @binding(7) var<storage, read> phis: array<vec2<f32>>;
@group(0) @binding(8) var<uniform> P: Params;

fn idx(f: u32, s: u32) -> u32 { return f * NSPEC + s; }

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s >= NSPEC) { return; }
  let nn = info[s];
  let kd = info[NSPEC + s];
  let n = u32(info[2u * NSPEC + s]);
  let c = nn / (P.radius * P.radius);
  let h = P.h;
  let leap = P.leap;
  var Tst: array<vec2<f32>, K>;
  var rhs: array<vec2<f32>, K>;
  var Db: array<vec2<f32>, K>;
  // nonlinear T tendency = full + tau D_cur (linear part treated implicitly)
  for (var k = 0u; k < K; k++) {
    var nt = td[idx(2u * K + k, s)];
    for (var j = 0u; j <= k; j++) { nt += mats[K * K + k * K + j] * cur[idx(K + j, s)]; }
    Tst[k] = old[idx(2u * K + k, s)] + h * nt;
  }
  let Pst = old[idx(3u * K, s)] + h * td[idx(3u * K, s)];
  for (var k = 0u; k < K; k++) {
    var gT = vec2<f32>(0.0);
    for (var j = k; j < K; j++) { gT += mats[k * K + j] * Tst[j]; }
    let ND = td[idx(K + k, s)] + c * (td[idx(3u * K + 1u + k, s)] + phis[s]);
    rhs[k] = old[idx(K + k, s)] + h * ND + h * c * (gT + P.RT * Pst);
  }
  let mo = n * K * K;
  for (var i = 0u; i < K; i++) {
    var x = vec2<f32>(0.0);
    for (var j = 0u; j < K; j++) { x += minv[mo + i * K + j] * rhs[j]; }
    Db[i] = x;
  }
  var nuD = vec2<f32>(0.0);
  for (var k = 0u; k < K; k++) { nuD += mats[2u * K * K + k] * Db[k]; }
  let damp = 1.0 / (1.0 + leap * kd);
  let doF = P.filterOn > 0.5;
  let fnu = 0.5 * P.nu;
  // ln ps
  var pn = 2.0 * (Pst - h * nuD) - old[idx(3u * K, s)];
  if (doF) { let d = fnu * (old[idx(3u * K, s)] - 2.0 * cur[idx(3u * K, s)] + pn); cur[idx(3u * K, s)] += P.alpha * d; pn -= (1.0 - P.alpha) * d; }
  nxt[idx(3u * K, s)] = pn;
  for (var k = 0u; k < K; k++) {
    var tD = vec2<f32>(0.0);
    for (var j = 0u; j <= k; j++) { tD += mats[K * K + k * K + j] * Db[j]; }
    var tn = 2.0 * (Tst[k] - h * tD) - old[idx(2u * K + k, s)];
    if (n > 0u) { tn *= damp; }
    var dn = (2.0 * Db[k] - old[idx(K + k, s)]) * damp;
    var vn = (old[idx(k, s)] + leap * td[idx(k, s)]) * damp;
    if (doF) {
      var d = fnu * (old[idx(k, s)] - 2.0 * cur[idx(k, s)] + vn); cur[idx(k, s)] += P.alpha * d; vn -= (1.0 - P.alpha) * d;
      d = fnu * (old[idx(K + k, s)] - 2.0 * cur[idx(K + k, s)] + dn); cur[idx(K + k, s)] += P.alpha * d; dn -= (1.0 - P.alpha) * d;
      d = fnu * (old[idx(2u * K + k, s)] - 2.0 * cur[idx(2u * K + k, s)] + tn); cur[idx(2u * K + k, s)] += P.alpha * d; tn -= (1.0 - P.alpha) * d;
    }
    nxt[idx(k, s)] = vn;
    nxt[idx(K + k, s)] = dn;
    nxt[idx(2u * K + k, s)] = tn;
  }
}
`;
