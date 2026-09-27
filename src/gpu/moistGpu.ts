// GPU moisture and column physics (f32), mirroring the CPU path in src/model/dycore.ts
// (advectMoisture, applyColumnPhysics, fixMass) and src/model/moist/{aquaplanet,sbm}.ts.
//
// Inserted by GpuDycore between the semi-implicit solve and the time filter:
//   synthesis of the new state (U, V, T, ln ps)                         -> grid2
//   semi-Lagrangian transport of q (Cartesian winds, departure points, tricubic quasi-monotone)
//   global water fixer (two reductions)
//   column physics: gray radiation, surface fluxes + boundary layer, slab ocean / land bucket,
//                   sponge, Simplified Betts–Miller convection, large-scale condensation
//   forward transform of the u, v, T increments, accumulated into the new spectral state
//   global dry-mass fixer on ln ps

import { Dycore } from '../model/dycore.js';
import { GrayPhysics } from '../model/moist/aquaplanet.js';
import { SBM } from '../model/moist/sbm.js';
import { MOIST, EPS } from '../model/moist/thermo.js';
import { GpuDycore } from './dycoreGpu.js';
import { Recipe } from './transformGpu.js';

const WG = 64;

/** Surface-field slots in the SFC buffer (each NG long). */
export const SFC = { ts: 0, bucket: 1, land: 2, precipConv: 3, precipLS: 4, evap: 5, olr: 6, precipRate: 7, olrNow: 8, runoff: 9, shf: 10, snowAcc: 11, count: 12 };

export class GpuMoist {
  readonly q: GPUBuffer;
  readonly qNext: GPUBuffer;
  readonly sfc: GPUBuffer;
  readonly grid2: GPUBuffer;
  private readonly mp: GPUBuffer;
  private readonly physics: GrayPhysics;
  private readonly gd: GpuDycore;

  constructor(private readonly device: GPUDevice, gd: GpuDycore, cpu: Dycore, physics: GrayPhysics) {
    this.gd = gd;
    this.physics = physics;
    const tr = gd.tr, K = cpu.K, ng = cpu.ng, nlat = cpu.tr.nlat, nlon = cpu.tr.nlon, T = cpu.tr.trunc;
    const lev = cpu.lev, cfg = physics.cfg;
    this.q = tr.gridBuffer(K); this.qNext = tr.gridBuffer(K);
    this.sfc = tr.gridBuffer(SFC.count);
    const four2 = tr.fourierBuffer(3 * K + 1);
    this.grid2 = tr.gridBuffer(3 * K + 1);
    const inc = tr.gridBuffer(3 * K), fourInc = tr.fourierBuffer(3 * K);
    const cart = tr.gridBuffer(3 * K), dep = tr.gridBuffer(3 * K);
    const red = tr.buffer((nlat + 16) * 4);
    this.mp = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // tables
    const levTab = new Float32Array(4 * (K + 1));
    for (let k = 0; k <= K; k++) levTab[k] = lev.sigmaHalf[k]!;
    for (let k = 0; k < K; k++) { levTab[(K + 1) + k] = lev.sigma[k]!; levTab[2 * (K + 1) + k] = lev.dsigma[k]!; }
    const levels = tr.upload(levTab);
    const rowTab = new Float32Array(5 * nlat);
    const qf = (physics as unknown as { qfluxLat: Float64Array }).qfluxLat;
    for (let j = 0; j < nlat; j++) {
      rowTab[j] = cpu.tr.lat[j]!; rowTab[nlat + j] = cpu.tr.coslat[j]!; rowTab[2 * nlat + j] = qf[j]!;
      rowTab[3 * nlat + j] = cpu.tr.weight[j]!;
    }
    // extended latitudes for semi-Lagrangian interpolation (index r + 2, r = -2 .. nlat + 1)
    const ext = new Float32Array(nlat + 4);
    for (let r = -2; r <= nlat + 1; r++) ext[r + 2] = r < 0 ? Math.PI - cpu.tr.lat[-r - 1]! : r >= nlat ? -Math.PI - cpu.tr.lat[2 * nlat - 1 - r]! : cpu.tr.lat[r]!;
    const rows = tr.upload(rowTab), yext = tr.upload(ext);
    const lon = tr.upload(Float32Array.from(cpu.tr.lon));

    const consts = `
const K: u32 = ${K}u;
const NG: u32 = ${ng}u;
const NLAT: u32 = ${nlat}u;
const NLON: u32 = ${nlon}u;
const NSPEC: u32 = ${cpu.tr.nspec}u;
const PI: f32 = 3.14159265358979;
struct MP { dt: f32, decl: f32, radius: f32, g: f32, psTarget: f32, p1: f32, p2: f32, p3: f32 };
`;
    const pipe = (code: string): GPUComputePipeline => device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: consts + code }), entryPoint: 'main' } });
    const pCart = pipe(CART_WGSL), pDep = pipe(DEP_WGSL), pInterp = pipe(INTERP_WGSL);
    const pRowSum = pipe(ROWSUM_WGSL), pFinal = pipe(FINAL_WGSL), pScale = pipe(SCALE_WGSL);
    const pPhys = pipe(physicsWgsl(physics, cpu)), pMass = pipe(MASS_WGSL), pCopy = pipe(COPY_WGSL);

    const bg = (p: GPUComputePipeline, bufs: GPUBuffer[]): GPUBindGroup => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const disp = (p: GPUComputePipeline, g: GPUBindGroup, n: number) => (pass: GPUComputePassEncoder): void => { pass.setPipeline(p); pass.setBindGroup(0, g); pass.dispatchWorkgroups(Math.ceil(n / WG)); };

    const G = gd.gridBuffer;              // time-n synthesis: U 0.., V K.., zeta, D, T, lnps (5K), ...
    const cartOp = disp(pCart, bg(pCart, [G, cart, rows, lon]), K * ng);
    const depOp = disp(pDep, bg(pDep, [cart, gd.sdot, dep, rows, lon, levels, yext, this.mp]), K * ng);
    const interpOp = disp(pInterp, bg(pInterp, [this.q, this.qNext, dep, rows, levels, yext]), K * ng);
    // water sums: red[0..nlat) partials; red[nlat + 0] = W0, [nlat + 1] = W1, [nlat + 2] = factor, [nlat+3] = mean ps
    const rowParams = (qSrc: GPUBuffer, psSrc: GPUBuffer, psSlot: number, mode: number): GPUBuffer => tr.upload(Uint32Array.from([psSlot, mode, 0, 0]), GPUBufferUsage.UNIFORM);
    const sum0 = disp(pRowSum, bg(pRowSum, [this.q, G, red, levels, rows, rowParams(this.q, G, 5 * K, 0)]), nlat);
    const fin0 = disp(pFinal, bg(pFinal, [red, tr.upload(Uint32Array.from([0, 0, 0, 0]), GPUBufferUsage.UNIFORM)]), 1);
    const sum1 = disp(pRowSum, bg(pRowSum, [this.qNext, this.grid2, red, levels, rows, rowParams(this.qNext, this.grid2, 3 * K, 0)]), nlat);
    const fin1 = disp(pFinal, bg(pFinal, [red, tr.upload(Uint32Array.from([1, 0, 0, 0]), GPUBufferUsage.UNIFORM)]), 1);
    const scaleOp = disp(pScale, bg(pScale, [this.qNext, red]), K * ng);
    const physOp = disp(pPhys, bg(pPhys, [this.grid2, this.qNext, inc, this.sfc, levels, rows, this.mp]), ng);
    const fwdAB = tr.prepareFftForward(inc, fourInc, 2 * K, 0, 0, true);
    const fwdT = tr.prepareFftForward(inc, fourInc, K, 2 * K, 2 * K, false);
    const massSum = disp(pRowSum, bg(pRowSum, [this.qNext, this.grid2, red, levels, rows, rowParams(this.qNext, this.grid2, 3 * K, 1)]), nlat);
    const massFin = disp(pFinal, bg(pFinal, [red, tr.upload(Uint32Array.from([3, 0, 0, 0]), GPUBufferUsage.UNIFORM)]), 1);
    const copyOp = disp(pCopy, bg(pCopy, [this.qNext, this.q]), K * ng);
    const a = cpu.planet.radius, ia = 1 / a;
    const synth2: Recipe[] = [];
    for (let k = 0; k < K; k++) {
      const psi = 3 * K + 1 + k, chi = 4 * K + 1 + k;
      synth2.push({ output: k, terms: [{ input: chi, kind: 'P', im: true, scale: ia }, { input: psi, kind: 'H', im: false, scale: -ia }] });
      synth2.push({ output: K + k, terms: [{ input: psi, kind: 'P', im: true, scale: ia }, { input: chi, kind: 'H', im: false, scale: ia }] });
      synth2.push({ output: 2 * K + k, terms: [{ input: 2 * K + k, kind: 'P', im: false, scale: 1 }] });
    }
    synth2.push({ output: 3 * K, terms: [{ input: 3 * K, kind: 'P', im: false, scale: 1 }] });
    const incRecipes: Recipe[] = [];
    for (let k = 0; k < K; k++) {
      incRecipes.push({ output: k, terms: [{ input: K + k, kind: 'P', im: true, scale: ia }, { input: k, kind: 'H', im: false, scale: ia }] });        // vor
      incRecipes.push({ output: K + k, terms: [{ input: k, kind: 'P', im: true, scale: ia }, { input: K + k, kind: 'H', im: false, scale: -ia }] });   // div
      incRecipes.push({ output: 2 * K + k, terms: [{ input: 2 * K + k, kind: 'P', im: false, scale: 1 }] });                                         // T
    }
    const inv2 = tr.prepareFftInverse(four2, this.grid2, 3 * K + 1);

    gd.setHooks((_r, _old, _cur, nxt) => {
      const psiNxt = disp(gd.pipePsi, bg(gd.pipePsi, [nxt, gd.specInfo, gd.paramsBuffer]), cpu.tr.nspec);
      const synthOp = tr.prepareSynth(nxt, four2, synth2);
      const analInc = tr.prepareAnal(fourInc, nxt, incRecipes, true);
      const massOp = disp(pMass, bg(pMass, [nxt, red, this.mp]), 1);
      return [psiNxt, synthOp, inv2, cartOp, depOp, sum0, fin0, interpOp, sum1, fin1, scaleOp, physOp, fwdAB, fwdT, analInc, massSum, massFin, massOp, copyOp];
    });
    gd.beforeStep = (time: number): void => {
      physics.setTime(time + cpu.dt);
      device.queue.writeBuffer(this.mp, 0, new Float32Array([cpu.dt, physics.declination, a, cpu.planet.gravity, cpu.massTarget, 0, 0, 0]));
    };
    void T;
  }

  /** Upload moisture and surface state from the CPU model/physics. */
  uploadFrom(cpu: Dycore): void {
    const tr = this.gd.tr, ng = cpu.ng, f = this.physics.f;
    tr.write(this.q, Float32Array.from(cpu.q));
    const s = new Float32Array(SFC.count * ng);
    s.set(Float32Array.from(f.sst), SFC.ts * ng);
    s.set(Float32Array.from(f.bucket), SFC.bucket * ng);
    s.set(Float32Array.from(this.physics.surface.land), SFC.land * ng);
    tr.write(this.sfc, s);
  }

  async readQ(): Promise<Float32Array> { return this.gd.tr.read(this.q, this.gd.K * this.gd.ng * 4); }
  async readSurface(): Promise<Float32Array> { return this.gd.tr.read(this.sfc, SFC.count * this.gd.ng * 4); }
}

// ---------------------------------------------------------------------------------------------
// Semi-Lagrangian transport

const CART_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read_write> cart: array<f32>;
@group(0) @binding(2) var<storage, read> rows: array<f32>;
@group(0) @binding(3) var<storage, read> lon: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x;
  if (q >= K * NG) { return; }
  let k = q / NG; let p = q % NG; let j = p / NLON; let i = p % NLON;
  let la = rows[j]; let cl = rows[NLAT + j];
  let u = G[k * NG + p] / cl; let v = G[(K + k) * NG + p] / cl;
  let so = sin(lon[i]); let co = cos(lon[i]); let sl = sin(la);
  cart[q] = -so * u - sl * co * v;
  cart[K * NG + q] = co * u - sl * so * v;
  cart[2u * K * NG + q] = cl * v;
}
`;

const SL_COMMON = /* wgsl */`
fn latInterval(lat: f32) -> i32 {
  if (lat > rows[0]) { return -1; }
  if (lat <= rows[NLAT - 1u]) { return i32(NLAT) - 1; }
  var lo = 0u; var hi = NLAT - 1u;
  loop { if (hi - lo <= 1u) { break; } let mid = (lo + hi) / 2u; if (rows[mid] >= lat) { lo = mid; } else { hi = mid; } }
  return i32(lo);
}
fn sigInterval(s: f32) -> u32 {
  if (s <= lev[K + 1u]) { return 0u; }
  if (s >= lev[K + 1u + K - 1u]) { return K - 2u; }
  var lo = 0u; var hi = K - 1u;
  loop { if (hi - lo <= 1u) { break; } let mid = (lo + hi) / 2u; if (lev[K + 1u + mid] <= s) { lo = mid; } else { hi = mid; } }
  return lo;
}
fn rowOf(r: i32) -> vec2<u32> {
  if (r < 0) { return vec2<u32>(u32(-r - 1), NLON / 2u); }
  if (r >= i32(NLAT)) { return vec2<u32>(u32(2 * i32(NLAT) - 1 - r), NLON / 2u); }
  return vec2<u32>(u32(r), 0u);
}
fn sig(k: u32) -> f32 { return lev[K + 1u + k]; }
`;

const DEP_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> cart: array<f32>;
@group(0) @binding(1) var<storage, read> sdot: array<f32>;
@group(0) @binding(2) var<storage, read_write> dep: array<f32>;
@group(0) @binding(3) var<storage, read> rows: array<f32>;
@group(0) @binding(4) var<storage, read> lon: array<f32>;
@group(0) @binding(5) var<storage, read> lev: array<f32>;
@group(0) @binding(6) var<storage, read> yext: array<f32>;
@group(0) @binding(7) var<uniform> M: MP;
${SL_COMMON}
fn sample(lat: f32, lo: f32, s: f32) -> vec4<f32> {
  let j = latInterval(lat);
  let y0 = yext[u32(j + 2)]; let y1 = yext[u32(j + 3)];
  let wy = (y0 - lat) / (y0 - y1);
  var x = lo / (2.0 * PI) * f32(NLON);
  x = x - floor(x / f32(NLON)) * f32(NLON);
  let i0 = u32(floor(x)) % NLON; let wx = x - floor(x);
  let k = sigInterval(s);
  let wz = clamp((s - sig(k)) / (sig(k + 1u) - sig(k)), 0.0, 1.0);
  var acc = vec4<f32>(0.0);
  for (var dk = 0u; dk < 2u; dk++) {
    let wk = select(1.0 - wz, wz, dk == 1u);
    let o = (k + dk) * NG;
    for (var dj = 0; dj < 2; dj++) {
      let rs = rowOf(j + dj);
      let w = wk * select(1.0 - wy, wy, dj == 1);
      let ia = o + rs.x * NLON + (i0 + rs.y) % NLON;
      let ib = o + rs.x * NLON + (i0 + 1u + rs.y) % NLON;
      let wa = w * (1.0 - wx); let wb = w * wx;
      acc += wa * vec4<f32>(cart[ia], cart[K * NG + ia], cart[2u * K * NG + ia], sdot[ia])
           + wb * vec4<f32>(cart[ib], cart[K * NG + ib], cart[2u * K * NG + ib], sdot[ib]);
    }
  }
  return acc;
}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x;
  if (q >= K * NG) { return; }
  let k = q / NG; let p = q % NG; let j = p / NLON; let i = p % NLON;
  let dt = M.dt; let a = M.radius;
  let la = rows[j]; let cl = rows[NLAT + j];
  let P = vec3<f32>(cl * cos(lon[i]), cl * sin(lon[i]), sin(la));
  var w = vec4<f32>(cart[q], cart[K * NG + q], cart[2u * K * NG + q], sdot[q]);
  var Mv = P;
  let sMin = sig(0u); let sMax = sig(K - 1u);
  for (var it = 0; it < 2; it++) {
    Mv = normalize(P - 0.5 * dt * w.xyz / a);
    let mlat = asin(clamp(Mv.z, -1.0, 1.0));
    let mlon = atan2(Mv.y, Mv.x);
    let msig = clamp(sig(k) - 0.5 * dt * w.w, sMin, sMax);
    w = sample(mlat, mlon, msig);
  }
  let d = dot(P, Mv);
  let D = 2.0 * d * Mv - P;
  dep[q] = asin(clamp(D.z, -1.0, 1.0));
  dep[K * NG + q] = atan2(D.y, D.x);
  dep[2u * K * NG + q] = clamp(sig(k) - dt * w.w, sMin, sMax);
}
`;

const INTERP_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> dep: array<f32>;
@group(0) @binding(3) var<storage, read> rows: array<f32>;
@group(0) @binding(4) var<storage, read> lev: array<f32>;
@group(0) @binding(5) var<storage, read> yext: array<f32>;
${SL_COMMON}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x;
  if (q >= K * NG) { return; }
  let lat = dep[q]; let lo = dep[K * NG + q]; let s = dep[2u * K * NG + q];
  var x = lo / (2.0 * PI) * f32(NLON);
  x = x - floor(x / f32(NLON)) * f32(NLON);
  let i0 = u32(floor(x)) % NLON; let t = x - floor(x);
  var wl = array<f32, 4>(-t * (t - 1.0) * (t - 2.0) / 6.0, (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0,
                         -(t + 1.0) * t * (t - 2.0) / 2.0, (t + 1.0) * t * (t - 1.0) / 6.0);
  let j = latInterval(lat);
  var wy: array<f32, 4>;
  var rr: array<vec2<u32>, 4>;
  for (var a = 0; a < 4; a++) {
    let ya = yext[u32(j - 1 + a + 2)];
    var w = 1.0;
    for (var b = 0; b < 4; b++) { if (b != a) { let yb = yext[u32(j - 1 + b + 2)]; w *= (lat - yb) / (ya - yb); } }
    wy[a] = w;
    rr[a] = rowOf(j - 1 + a);
  }
  let k = sigInterval(s);
  var k0: u32; var nk: u32;
  var wz = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  if (k >= 1u && k + 2u <= K - 1u) {
    k0 = k - 1u; nk = 4u;
    for (var a = 0u; a < 4u; a++) {
      var w = 1.0;
      for (var b = 0u; b < 4u; b++) { if (b != a) { w *= (s - sig(k0 + b)) / (sig(k0 + a) - sig(k0 + b)); } }
      wz[a] = w;
    }
  } else {
    k0 = k; nk = 2u;
    let w1 = clamp((s - sig(k)) / (sig(k + 1u) - sig(k)), 0.0, 1.0);
    wz[0] = 1.0 - w1; wz[1] = w1;
  }
  let inner = select(0u, 1u, nk == 4u);
  var acc = 0.0; var lo2 = 1e30; var hi2 = -1e30;
  for (var c = 0u; c < nk; c++) {
    let o = (k0 + c) * NG;
    var accK = 0.0;
    for (var a = 0u; a < 4u; a++) {
      let ro = o + rr[a].x * NLON;
      var accR = 0.0;
      for (var b = 0u; b < 4u; b++) {
        let v = src[ro + (i0 + NLON - 1u + b + rr[a].y) % NLON];
        accR += wl[b] * v;
        if ((a == 1u || a == 2u) && (b == 1u || b == 2u) && (c == inner || c == inner + 1u)) { lo2 = min(lo2, v); hi2 = max(hi2, v); }
      }
      accK += wy[a] * accR;
    }
    acc += wz[c] * accK;
  }
  dst[q] = clamp(acc, lo2, hi2);
}
`;

// ---------------------------------------------------------------------------------------------
// Reductions and fixers

// mode 0: sum_k q dsigma * ps ; mode 1: sum ps.  partial per row (Gaussian-weighted zonal mean)
const ROWSUM_WGSL = /* wgsl */`
struct RP { psSlot: u32, mode: u32, a: u32, b: u32 };
@group(0) @binding(0) var<storage, read> qf: array<f32>;
@group(0) @binding(1) var<storage, read> psg: array<f32>;
@group(0) @binding(2) var<storage, read_write> red: array<f32>;
@group(0) @binding(3) var<storage, read> lev: array<f32>;
@group(0) @binding(4) var<storage, read> rows: array<f32>;
@group(0) @binding(5) var<uniform> R: RP;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= NLAT) { return; }
  var s = 0.0;
  for (var i = 0u; i < NLON; i++) {
    let p = j * NLON + i;
    let ps = exp(psg[R.psSlot * NG + p]);
    if (R.mode == 1u) { s += ps; continue; }
    var col = 0.0;
    for (var k = 0u; k < K; k++) { col += qf[k * NG + p] * lev[2u * (K + 1u) + k]; }
    s += col * ps;
  }
  red[j] = rows[3u * NLAT + j] * s / f32(NLON);
}
`;

// slot 0: W0 = sum -> red[NLAT]; slot 1: W1 -> red[NLAT+1], factor W0/W1 -> red[NLAT+2]; slot 3: mean ps -> red[NLAT+3]
const FINAL_WGSL = /* wgsl */`
struct FP { slot: u32, a: u32, b: u32, c: u32 };
@group(0) @binding(0) var<storage, read_write> red: array<f32>;
@group(0) @binding(1) var<uniform> F: FP;
@compute @workgroup_size(1)
fn main() {
  var s = 0.0;
  for (var j = 0u; j < NLAT; j++) { s += red[j]; }
  s *= 0.5;
  red[NLAT + F.slot] = s;
  if (F.slot == 1u) { red[NLAT + 2u] = select(1.0, red[NLAT] / s, s > 0.0); }
}
`;

const SCALE_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> qf: array<f32>;
@group(0) @binding(1) var<storage, read> red: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= K * NG) { return; }
  qf[gid.x] *= red[NLAT + 2u];
}
`;

const MASS_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> st: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> red: array<f32>;
@group(0) @binding(2) var<uniform> M: MP;
@compute @workgroup_size(1)
fn main() {
  let mean = red[NLAT + 3u];
  st[3u * K * NSPEC].x += log(M.psTarget / mean) * 1.41421356237;
}
`;

const COPY_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= K * NG) { return; }
  b[gid.x] = a[gid.x];
}
`;

// ---------------------------------------------------------------------------------------------
// Column physics

function f(x: number): string { return Number.isInteger(x) ? `${x}.0` : `${x}`; }

function physicsWgsl(ph: GrayPhysics, cpu: Dycore): string {
  const c = ph.cfg, air = cpu.air;
  const rad = c.radiation === 'frierson'
    ? `
    let tau0 = ${f(c.tauEq)} + (${f(c.tauPole)} - ${f(c.tauEq)}) * sl * sl;
    for (var k = 0u; k < K; k++) {
      let x0 = ph[k] / 1.0e5; let x1 = ph[k + 1u] / 1.0e5;
      let t0 = tau0 * (${f(c.linearTau)} * x0 + (1.0 - ${f(c.linearTau)}) * x0 * x0 * x0 * x0);
      let t1 = tau0 * (${f(c.linearTau)} * x1 + (1.0 - ${f(c.linearTau)}) * x1 * x1 * x1 * x1);
      tr[k] = exp(-(t1 - t0));
    }`
    : `
    for (var k = 0u; k < K; k++) { tr[k] = exp(-(${f(c.byrneA)} + ${f(c.byrneB)} * Q[k]) * (ph[k + 1u] - ph[k]) / 1.0e5); }`;
  const insol = c.seasonal
    ? `
    let dcl = M.decl;
    let hx = clamp(-tan(la) * tan(dcl), -1.0, 1.0);
    let h0 = acos(hx);
    let insol = ${f(c.solarConstant)} / PI * (h0 * sin(la) * sin(dcl) + cos(la) * cos(dcl) * sin(h0));`
    : `
    let insol = 0.25 * ${f(c.solarConstant)} * (1.0 + ${f(c.delSol)} * (1.0 - 3.0 * sl * sl) / 4.0);`;
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> G2: array<f32>;
@group(0) @binding(1) var<storage, read_write> QN: array<f32>;
@group(0) @binding(2) var<storage, read_write> INC: array<f32>;
@group(0) @binding(3) var<storage, read_write> S: array<f32>;
@group(0) @binding(4) var<storage, read> lev: array<f32>;
@group(0) @binding(5) var<storage, read> rows: array<f32>;
@group(0) @binding(6) var<uniform> M: MP;

const RD: f32 = ${f(air.rd)};
const CP: f32 = ${f(air.cp)};
const KAPPA: f32 = ${air.kappa};
const PREF: f32 = ${f(air.pRef)};
const LV: f32 = ${f(MOIST.Lv)};
const RV: f32 = ${f(MOIST.Rv)};
const EPSV: f32 = ${EPS};
const STEF: f32 = ${MOIST.stefan};
const VK: f32 = ${f(MOIST.vonKarman)};
const RHOW: f32 = ${f(MOIST.rhoWater)};
const CPW: f32 = ${f(MOIST.cpWater)};

fn esat(T: f32) -> f32 { return ${f(MOIST.e0)} * exp(-(LV / RV) * (1.0 / T - 1.0 / ${f(MOIST.T0)})); }
fn qsat(T: f32, p: f32) -> f32 { let es = esat(T); return EPSV * es / max(p - (1.0 - EPSV) * es, 1.0e-3 * p); }
fn dqsat(T: f32, p: f32) -> f32 { let es = esat(T); let den = max(p - (1.0 - EPSV) * es, 1.0e-3 * p); return EPSV * es * LV / (RV * T * T) * p / (den * den); }
fn mixr(e: f32, p: f32) -> f32 { return EPSV * e / max(p - e, 1.0e-3 * p); }
fn virt(T: f32, r: f32) -> f32 { let q = r / (1.0 + r); return T * (1.0 + q * (RV / RD - 1.0)); }
fn lclTemp(value: f32) -> f32 {
  var T = 250.0;
  for (var it = 0; it < 30; it++) {
    let fx = log(esat(T)) - log(T) / KAPPA - value;
    let df = LV / (RV * T * T) - 1.0 / (KAPPA * T);
    let d = fx / df;
    T = max(100.0, T - d);
    if (abs(d) < 1.0e-4) { break; }
  }
  return T;
}

var<private> Tc: array<f32, K>;
var<private> Q: array<f32, K>;
var<private> U: array<f32, K>;
var<private> V: array<f32, K>;
var<private> pf: array<f32, K>;
var<private> ph: array<f32, K + 1u>;
var<private> zf: array<f32, K>;
var<private> zh: array<f32, K + 1u>;
var<private> lwu: array<f32, K + 1u>;
var<private> lwd: array<f32, K + 1u>;
var<private> tr: array<f32, K>;
var<private> sw: array<f32, K + 1u>;
var<private> A: array<f32, K + 1u>;
var<private> Mm: array<f32, K>;
var<private> cc: array<f32, K>;
var<private> dd: array<f32, K>;
var<private> X: array<f32, K>;
var<private> Tp: array<f32, K>;
var<private> rp: array<f32, K>;
var<private> rin: array<f32, K>;
var<private> Tv: array<f32, K>;
var<private> Tref: array<f32, K>;
var<private> qref: array<f32, K>;
var<private> dTc: array<f32, K>;
var<private> dq: array<f32, K>;

// backward-Euler diffusion of X with conductances A, masses Mm, surface node xs via A[K]; returns surface flux
fn diffuse(xs: f32, dt: f32) -> f32 {
  var pc = 0.0; var pd = 0.0;
  for (var k = 0u; k < K; k++) {
    let lower = -dt * A[k];
    var upper = 0.0;
    if (k < K - 1u) { upper = -dt * A[k + 1u]; }
    let diag = Mm[k] + dt * (A[k] + A[k + 1u]);
    var rhs = Mm[k] * X[k];
    if (k == K - 1u) { rhs += dt * A[K] * xs; }
    let den = diag - lower * pc;
    cc[k] = upper / den;
    dd[k] = (rhs - lower * pd) / den;
    pc = cc[k]; pd = dd[k];
  }
  X[K - 1u] = dd[K - 1u];
  for (var kk = i32(K) - 2; kk >= 0; kk--) { let k = u32(kk); X[k] = dd[k] - cc[k] * X[k + 1u]; }
  return A[K] * (xs - X[K - 1u]);
}

fn noCape() { for (var k = 0u; k < K; k++) { Tp[k] = Tc[k]; rp[k] = rin[k]; } }

// Simplified Betts–Miller; returns convective precipitation (kg m^-2 over dt)
fn sbm(dt: f32, g: f32) -> f32 {
  let ks = K - 1u;
  let tau = ${f(SBM.tauBm)};
  let top = max(ph[0], 0.1 * pf[0]);
  for (var k = 0u; k < K; k++) {
    rin[k] = Q[k] / (1.0 - Q[k]); Tv[k] = virt(Tc[k], rin[k]); Tp[k] = Tc[k]; rp[k] = rin[k];
    dTc[k] = 0.0; dq[k] = 0.0; Tref[k] = Tc[k]; qref[k] = Q[k];
  }
  var cape = 0.0; var nocape = true; var kLZB = -1; var kLCL = i32(ks); var skip = false;
  let T0 = Tc[ks]; let r0 = rin[ks];
  let rs0 = mixr(esat(T0), pf[ks]);
  if (r0 >= rs0) {
    Tp[ks] = T0 + (r0 - rs0) / (CP / LV + LV * rs0 / RV / (T0 * T0));
    rp[ks] = mixr(esat(Tp[ks]), pf[ks]);
  } else {
    let theta0 = T0 * pow(PREF / pf[ks], KAPPA);
    if (r0 <= 0.0) { skip = true; } else {
      let value = log(pow(theta0, -1.0 / KAPPA) * PREF * r0 / (EPSV + r0));
      var TLCL = lclTemp(value);
      var pLCL = PREF * pow(TLCL / theta0, 1.0 / KAPPA);
      if (pLCL < pf[0]) { pLCL = pf[0]; TLCL = theta0 * pow(pLCL / PREF, KAPPA); }
      var k = i32(ks);
      while (k > 0 && pf[u32(k)] > pLCL) {
        let uk = u32(k);
        Tp[uk] = theta0 * pow(pf[uk] / PREF, KAPPA);
        rp[uk] = mixr(esat(Tp[uk]), pf[uk]);
        k--;
      }
      kLCL = k;
      let ul = u32(kLCL);
      var aa = KAPPA * TLCL + (LV / CP) * r0;
      var bb = LV * LV * r0 / (CP * RV * TLCL * TLCL);
      Tp[ul] = TLCL + (aa / (1.0 + bb)) * log(pf[ul] / pLCL) / 2.0;
      if (Tp[ul] < ${f(SBM.Tmin)}) { skip = true; noCape(); } else {
        rp[ul] = mixr(esat(Tp[ul]), (pf[ul] + pLCL) / 2.0);
        aa = KAPPA * Tp[ul] + (LV / CP) * rp[ul];
        bb = LV * LV * rp[ul] / (CP * RV * Tp[ul] * Tp[ul]);
        Tp[ul] = TLCL + (aa / (1.0 + bb)) * log(pf[ul] / pLCL);
        if (Tp[ul] < ${f(SBM.Tmin)}) { skip = true; noCape(); } else {
          rp[ul] = mixr(esat(Tp[ul]), pf[ul]);
          let buoy = virt(Tp[ul], rp[ul]) - Tv[ul];
          let phu = select(ph[ul], top, ul == 0u);
          if (buoy > 0.0) { cape += RD * buoy * log(ph[ul + 1u] / phu); nocape = false; }
        }
      }
    }
  }
  if (!skip) {
    for (var kk = kLCL - 1; kk >= 0; kk--) {
      let k = u32(kk);
      var aa = KAPPA * Tp[k + 1u] + (LV / CP) * rp[k + 1u];
      var bb = LV * LV * rp[k + 1u] / (CP * RV * Tp[k + 1u] * Tp[k + 1u]);
      Tp[k] = Tp[k + 1u] + (aa / (1.0 + bb)) * log(pf[k] / pf[k + 1u]) / 2.0;
      if (Tp[k] < ${f(SBM.Tmin)}) { if (nocape) { noCape(); } break; }
      rp[k] = mixr(esat(Tp[k]), (pf[k] + pf[k + 1u]) / 2.0);
      aa = KAPPA * Tp[k] + (LV / CP) * rp[k];
      bb = LV * LV * rp[k] / (CP * RV * Tp[k] * Tp[k]);
      Tp[k] = Tp[k + 1u] + (aa / (1.0 + bb)) * log(pf[k] / pf[k + 1u]);
      if (Tp[k] < ${f(SBM.Tmin)}) { if (nocape) { noCape(); } break; }
      rp[k] = mixr(esat(Tp[k]), pf[k]);
      let buoy = virt(Tp[k], rp[k]) - Tv[k];
      if (buoy < 0.0) {
        if (!nocape) { kLZB = kk + 1; break; }
      } else {
        let phu = select(ph[k], top, k == 0u);
        cape += RD * buoy * log(ph[k + 1u] / phu);
        nocape = false;
      }
    }
  }
  if (!(cape > 0.0)) { return 0.0; }
  if (kLZB < 0) { kLZB = 0; }
  let kz = u32(kLZB);
  for (var k = 0u; k < K; k++) { Tref[k] = Tp[k]; }
  for (var k = kz; k <= ks; k++) {
    let eref = ${f(SBM.rhbm)} * pf[k] * rp[k] / (rp[k] + EPSV);
    let r = mixr(eref, pf[k]);
    qref[k] = r / (1.0 + r);
  }
  for (var k = 0u; k < kz; k++) { Tref[k] = Tc[k]; qref[k] = Q[k]; }
  var Pq = 0.0; var Pt = 0.0;
  for (var k = kz; k <= ks; k++) {
    let dp = ph[k + 1u] - ph[k];
    dq[k] = -(Q[k] - qref[k]) * dt / tau;
    Pq -= dq[k] * dp;
    dTc[k] = -(Tc[k] - Tref[k]) * dt / tau;
    Pt += (CP / LV) * dTc[k] * dp;
  }
  Pq /= g; Pt /= g;
  if (Pq > 0.0 && Pt > 0.0) {
    if (Pq > Pt) {
      let fr = Pt / Pq;
      for (var k = kz; k <= ks; k++) { dq[k] *= fr; }
      Pq = Pt;
    } else {
      var dk = 0.0;
      for (var k = kz; k <= ks; k++) { dk -= (dTc[k] + (LV / CP) * dq[k]) * (ph[k + 1u] - ph[k]); }
      dk /= ph[ks + 1u] - ph[kz];
      for (var k = kz; k <= ks; k++) { dTc[k] += dk; }
    }
  } else if (Pt > 0.0) {
    var k = kz;
    while (Pq < 0.0 && k <= ks) { Pq += dq[k] * (ph[k + 1u] - ph[k]) / g; k++; }
    let kTop = k - 1u;
    let found = Pq > 0.0;
    if (kTop > kz) { for (var kk = kz; kk < kTop; kk++) { dTc[kk] = 0.0; dq[kk] = 0.0; } }
    if (found) {
      let cfac = Pq * g / (dq[kTop] * (ph[kTop] - ph[kTop + 1u]) * -1.0);
      dq[kTop] *= cfac; dTc[kTop] *= cfac;
      var dk = 0.0;
      for (var kk = kTop; kk <= ks; kk++) { dk += dTc[kk] * (ph[kk] - ph[kk + 1u]); }
      dk /= ph[ks + 1u] - ph[kTop];
      if (kTop != ks) { for (var kk = kTop; kk <= ks; kk++) { dTc[kk] += dk; } }
    } else {
      var k1 = kz; var k2 = kTop;
      if (kTop == kz) { k1 = ks; k2 = ks; }
      for (var kk = k1; kk <= k2; kk++) { dTc[kk] = 0.0; dq[kk] = 0.0; }
    }
    Pq = 0.0;
  } else {
    return 0.0;
  }
  for (var k = 0u; k < K; k++) { Tc[k] += dTc[k]; Q[k] += dq[k]; }
  return max(0.0, Pq);
}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= NG) { return; }
  let j = p / NLON;
  let la = rows[j]; let cl = rows[NLAT + j]; let qflux = rows[2u * NLAT + j];
  let sl = sin(la);
  let dt = M.dt; let g = M.g;
  let ps = exp(G2[3u * K * NG + p]);
  let isLand = S[${SFC.land}u * NG + p] > 0.5;
  for (var k = 0u; k < K; k++) {
    U[k] = G2[k * NG + p] / cl; V[k] = G2[(K + k) * NG + p] / cl; Tc[k] = G2[(2u * K + k) * NG + p]; Q[k] = QN[k * NG + p];
  }
  let u0 = U; let v0 = V; let T0c = Tc;
  for (var k = 0u; k <= K; k++) { ph[k] = lev[k] * ps; }
  for (var k = 0u; k < K; k++) { pf[k] = lev[K + 1u + k] * ps; }
  zh[K] = 0.0;
  for (var kk = i32(K) - 1; kk >= 0; kk--) {
    let k = u32(kk);
    var lnr = 1.3862944; var alpha = 0.6931472;
    if (k > 0u) { lnr = log(ph[k + 1u] / ph[k]); alpha = 1.0 - ph[k] / (ph[k + 1u] - ph[k]) * lnr; }
    zf[k] = zh[k + 1u] + alpha * RD * Tc[k] / g;
    zh[k] = zh[k + 1u] + RD * Tc[k] * lnr / g;
  }
  // ---- radiation
  let ts = S[${SFC.ts}u * NG + p];
  ${rad}
  lwd[0] = 0.0;
  for (var k = 0u; k < K; k++) { let B = STEF * Tc[k] * Tc[k] * Tc[k] * Tc[k]; lwd[k + 1u] = lwd[k] * tr[k] + B * (1.0 - tr[k]); }
  lwu[K] = STEF * ts * ts * ts * ts;
  for (var kk = i32(K) - 1; kk >= 0; kk--) { let k = u32(kk); let B = STEF * Tc[k] * Tc[k] * Tc[k] * Tc[k]; lwu[k] = lwu[k + 1u] * tr[k] + B * (1.0 - tr[k]); }
  ${insol}
  for (var k = 0u; k <= K; k++) { let x = ph[k] / 1.0e5; sw[k] = insol * exp(-${f(c.atmAbs)} * x * x * x * x); }
  var albedo = select(${f(c.albedo)}, ${f(c.albedoLand)}, isLand);
  if (!isLand && ${f(c.albedoIce)} != ${f(c.albedo)}) {
    let wi = clamp((273.15 - ts) / 10.0, 0.0, 1.0);
    albedo = ${f(c.albedo)} + wi * (${f(c.albedoIce)} - ${f(c.albedo)});
  }
  let swSfc = sw[K]; let swUp = albedo * swSfc;
  for (var k = 0u; k < K; k++) {
    let Ftop = lwu[k] - lwd[k] + swUp - sw[k];
    let Fbot = lwu[k + 1u] - lwd[k + 1u] + swUp - sw[k + 1u];
    Tc[k] += dt * g * (Fbot - Ftop) / (CP * (ph[k + 1u] - ph[k]));
  }
  let olr = lwu[0];
  // ---- surface fluxes + boundary layer
  let z0 = select(${f(c.roughness)}, ${f(c.roughnessLand)}, isLand);
  let ka = K - 1u; let za = zf[ka];
  let speed = max(sqrt(U[ka] * U[ka] + V[ka] * V[ka]), 1.0e-3);
  let rhoA = pf[ka] / (RD * Tc[ka]);
  let ri = g * za * (Tc[ka] + g * za / CP - ts) / (ts * speed * speed);
  let cn = pow(VK / log(za / z0), 2.0);
  let rc = ${f(c.richCrit)};
  var cd = 0.0;
  if (ri <= 0.0) { cd = cn; } else if (ri < rc) { cd = cn * (1.0 - ri / rc) * (1.0 - ri / rc); }
  let ustar = sqrt(cd) * speed;
  let svBot = Tc[ka] + g * za / CP;
  var h = za; var rich1 = 0.0; var h1 = za; var found = false;
  for (var kk = i32(ka) - 1; kk >= 1; kk--) {
    let k = u32(kk);
    let sv = Tc[k] + g * zf[k] / CP;
    let rich = zf[k] * g * (sv - svBot) / svBot / (U[k] * U[k] + V[k] * V[k] + 1.0e-10);
    if (rich > rc) { h = zf[k] + (h1 - zf[k]) * (rich - rc) / (rich - rich1); found = true; break; }
    rich1 = rich; h1 = zf[k];
  }
  if (!found) { h = zf[1]; }
  let hIn = ${f(c.fracInner)} * h;
  A[0] = 0.0;
  for (var i = 1u; i < K; i++) {
    let z = zh[i];
    var Kd = 0.0;
    var zz = z;
    var scaleK = 1.0;
    if (z >= hIn) { zz = hIn; scaleK = (z / hIn) * pow(1.0 - (z - hIn) / (h - hIn), 2.0); }
    if (z < h) {
      if (ri <= 0.0) { Kd = VK * ustar * zz; }
      else if (ri < rc) { let x = ri / rc; Kd = VK * ustar * zz / (1.0 + x * log(zz / z0) / (1.0 - x)); }
      Kd *= scaleK;
    }
    let Th = 0.5 * (Tc[i - 1u] + Tc[i]);
    A[i] = (ph[i] / (RD * Th)) * Kd / (zf[i - 1u] - zf[i]);
  }
  let aSfc = rhoA * cd * speed;
  for (var k = 0u; k < K; k++) { Mm[k] = (ph[k + 1u] - ph[k]) / g; }
  let beta = select(1.0, min(1.0, S[${SFC.bucket}u * NG + p] / (0.75 * ${f(c.bucketMax)})), isLand);
  // dry static energy
  for (var k = 0u; k < K; k++) { X[k] = CP * Tc[k] + g * zf[k]; }
  A[K] = aSfc;
  let fluxS = diffuse(CP * ts, dt);
  for (var k = 0u; k < K; k++) { Tc[k] = (X[k] - g * zf[k]) / CP; }
  for (var k = 0u; k < K; k++) { X[k] = Q[k]; }
  A[K] = aSfc * beta;
  var fluxQ = diffuse(qsat(ts, ps), dt);
  for (var k = 0u; k < K; k++) { Q[k] = X[k]; }
  A[K] = aSfc;
  for (var k = 0u; k < K; k++) { X[k] = U[k]; }
  let fu = diffuse(0.0, dt);
  for (var k = 0u; k < K; k++) { U[k] = X[k]; X[k] = V[k]; }
  let fv = diffuse(0.0, dt);
  for (var k = 0u; k < K; k++) { V[k] = X[k]; }
  if (isLand && fluxQ * dt > S[${SFC.bucket}u * NG + p] * RHOW) {
    let excess = fluxQ * dt - S[${SFC.bucket}u * NG + p] * RHOW;
    Q[ka] -= excess / Mm[ka];
    fluxQ -= excess / dt;
  }
  let lh = LV * fluxQ;
  let heatCap = select(RHOW * CPW * ${f(c.mixedLayerDepth)}, ${f(c.landHeatCapacity)}, isLand);
  let net = swSfc * (1.0 - albedo) + lwd[K] - STEF * ts * ts * ts * ts - fluxS - lh + select(qflux, 0.0, isLand);
  S[${SFC.ts}u * NG + p] = ts + dt * net / heatCap;
  // ---- sponge
  for (var k = 0u; k < K; k++) {
    if (pf[k] >= ${f(c.spongePBottom)}) { break; }
    let x = (${f(c.spongePBottom)} - pf[k]) / ${f(c.spongePBottom)};
    let r = x * x / ${f(c.spongeTau)};
    let un = U[k] / (1.0 + dt * r); let vn = V[k] / (1.0 + dt * r);
    Tc[k] += 0.5 * (U[k] * U[k] + V[k] * V[k] - un * un - vn * vn) / CP;
    U[k] = un; V[k] = vn;
  }
  // ---- convection
  var rainConv = 0.0;
  ${c.useConvection ? 'rainConv = sbm(dt, g);' : ''}
  // ---- large-scale condensation with re-evaporation
  let hlcp = LV / CP;
  var exq = 0.0;
  for (var k = 0u; k < K; k++) {
    let mass = (ph[k + 1u] - ph[k]) / g;
    let qs = qsat(Tc[k], pf[k]); let dqs = dqsat(Tc[k], pf[k]);
    if (Q[k] > qs) {
      let d = (qs - Q[k]) / (1.0 + hlcp * dqs);
      Q[k] += d; Tc[k] -= hlcp * d; exq -= d * mass;
    } else if (exq > 0.0) {
      let def = min(max((qs - Q[k]) / (1.0 + hlcp * dqs), 0.0), exq / mass);
      Q[k] += def; Tc[k] -= hlcp * def; exq -= def * mass;
    }
  }
  let rainLS = max(0.0, exq);
  for (var k = 0u; k < K; k++) { Q[k] = max(Q[k], 0.0); }
  // ---- land bucket
  if (isLand) {
    var w = S[${SFC.bucket}u * NG + p] + (rainConv + rainLS - fluxQ * dt) / RHOW;
    w = max(w, 0.0);
    if (w > ${f(c.bucketMax)}) { S[${SFC.runoff}u * NG + p] += (w - ${f(c.bucketMax)}) * RHOW; w = ${f(c.bucketMax)}; }
    S[${SFC.bucket}u * NG + p] = w;
  }
  S[${SFC.precipConv}u * NG + p] += rainConv;
  S[${SFC.precipLS}u * NG + p] += rainLS;
  S[${SFC.evap}u * NG + p] += fluxQ * dt;
  S[${SFC.shf}u * NG + p] += fluxS * dt;
  S[${SFC.olr}u * NG + p] += olr * dt;
  S[${SFC.olrNow}u * NG + p] = olr;
  S[${SFC.precipRate}u * NG + p] = (rainConv + rainLS) / dt;
  if (Tc[K - 1u] < 273.15) { S[${SFC.snowAcc}u * NG + p] += rainConv + rainLS; }
  // ---- increments (cos-weighted winds for the vector transform) and moisture
  for (var k = 0u; k < K; k++) {
    INC[k * NG + p] = (U[k] - u0[k]) * cl;
    INC[(K + k) * NG + p] = (V[k] - v0[k]) * cl;
    INC[(2u * K + k) * NG + p] = Tc[k] - T0c[k];
    QN[k * NG + p] = Q[k];
  }
  _ = fu; _ = fv;
}
`;
}
