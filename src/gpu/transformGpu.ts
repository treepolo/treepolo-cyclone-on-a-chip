// Spherical-harmonic transforms on the GPU (WebGPU / WGSL, f32).
//
// Mirrors src/spectral/transform.ts. All transforms are batched: a "recipe" lists, for each output
// field, the input fields and operators (P or H Legendre functions, optional i*m factor, scale).
//
// Buffer layouts (all f32; complex = vec2<f32>):
//   spectral : [field][nspec]            complex
//   fourier  : [field][lat][T+1]         complex
//   grid     : [field][lat][lon]         real
// Latitudes run north -> south, as on the CPU.

import { SpectralTransform } from '../spectral/transform.js';

export interface Term { input: number; kind: 'P' | 'H'; im: boolean; scale: number }
export interface Recipe { output: number; terms: Term[] }

const WG = 64;

export class GpuTransform {
  readonly device: GPUDevice;
  readonly cpu: SpectralTransform;
  readonly trunc: number; readonly nlat: number; readonly nlon: number; readonly nspec: number; readonly half: number;
  readonly nfour: number; // T + 1
  private readonly tables: { P: GPUBuffer; H: GPUBuffer; PW: GPUBuffer; HW: GPUBuffer; mOf: GPUBuffer; nOf: GPUBuffer; mStart: GPUBuffer; invc2: GPUBuffer; tw: GPUBuffer };
  private readonly pipes: { synth: GPUComputePipeline; anal: GPUComputePipeline; fftInv: GPUComputePipeline; fftFwd: GPUComputePipeline };

  constructor(device: GPUDevice, cpu: SpectralTransform) {
    this.device = device;
    this.cpu = cpu;
    this.trunc = cpu.trunc; this.nlat = cpu.nlat; this.nlon = cpu.nlon; this.nspec = cpu.nspec;
    this.half = cpu.nlat / 2; this.nfour = cpu.trunc + 1;
    const hlf = this.half;
    // Legendre tables from the CPU transform (private fields accessed via a typed view)
    const src = cpu as unknown as { P: Float64Array; H: Float64Array };
    const P = Float32Array.from(src.P), H = Float32Array.from(src.H);
    const PW = new Float32Array(P.length), HW = new Float32Array(H.length);
    for (let s = 0; s < cpu.nspec; s++) for (let j = 0; j < hlf; j++) {
      PW[s * hlf + j] = src.P[s * hlf + j]! * cpu.weight[j]!;
      HW[s * hlf + j] = src.H[s * hlf + j]! * cpu.weight[j]!;
    }
    const invc2 = new Float32Array(cpu.nlat);
    for (let j = 0; j < cpu.nlat; j++) invc2[j] = 1 / (1 - cpu.mu[j]! ** 2);
    // FFT twiddles exp(-2 pi i k / N) computed in double precision (WGSL sin/cos are only ~1e-3 accurate)
    const tw = new Float32Array(this.nlon);
    for (let k = 0; k < this.nlon / 2; k++) { tw[2 * k] = Math.cos(2 * Math.PI * k / this.nlon); tw[2 * k + 1] = -Math.sin(2 * Math.PI * k / this.nlon); }
    this.tables = {
      tw: this.upload(tw),
      P: this.upload(P), H: this.upload(H), PW: this.upload(PW), HW: this.upload(HW),
      mOf: this.upload(Uint32Array.from(cpu.mOf)), nOf: this.upload(Uint32Array.from(cpu.nOf)),
      mStart: this.upload(Uint32Array.from(cpu.mStart)), invc2: this.upload(invc2),
    };
    const consts = `
const T: u32 = ${this.trunc}u;
const NF: u32 = ${this.nfour}u;
const NLAT: u32 = ${this.nlat}u;
const NLON: u32 = ${this.nlon}u;
const HALF: u32 = ${hlf}u;
const NSPEC: u32 = ${this.nspec}u;
struct Term { input: u32, kind: u32, im: u32, scale: f32 };
struct Range { start: u32, end: u32, output: u32, pad: u32 };
`;
    const mk = (code: string, entry: string): GPUComputePipeline => device.createComputePipeline({
      layout: 'auto', compute: { module: device.createShaderModule({ code: consts + code }), entryPoint: entry },
    });
    this.pipes = {
      synth: mk(SYNTH_WGSL, 'main'),
      anal: mk(ANAL_WGSL, 'main'),
      fftInv: mk(fftWgsl(this.nlon, true), 'main'),
      fftFwd: mk(fftWgsl(this.nlon, false), 'main'),
    };
  }

  upload(data: Float32Array | Uint32Array, usage = GPUBufferUsage.STORAGE): GPUBuffer {
    const b = this.device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    const Ctor = data instanceof Float32Array ? Float32Array : Uint32Array;
    new Ctor(b.getMappedRange()).set(data);
    b.unmap();
    return b;
  }

  buffer(bytes: number, extra = 0): GPUBuffer {
    return this.device.createBuffer({ size: Math.max(16, bytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extra });
  }
  specBuffer(fields: number): GPUBuffer { return this.buffer(fields * this.nspec * 8); }
  fourierBuffer(fields: number): GPUBuffer { return this.buffer(fields * this.nlat * this.nfour * 8); }
  gridBuffer(fields: number): GPUBuffer { return this.buffer(fields * this.nlat * this.nlon * 4); }

  private encodeRecipes(recipes: Recipe[]): { ranges: GPUBuffer; terms: GPUBuffer } {
    const ranges = new Uint32Array(recipes.length * 4);
    const termData = new ArrayBuffer(Math.max(1, recipes.reduce((a, r) => a + r.terms.length, 0)) * 16);
    const tu = new Uint32Array(termData), tf = new Float32Array(termData);
    let t = 0;
    recipes.forEach((r, o) => {
      ranges[o * 4] = t; ranges[o * 4 + 2] = r.output;
      for (const term of r.terms) {
        tu[t * 4] = term.input; tu[t * 4 + 1] = term.kind === 'P' ? 0 : 1; tu[t * 4 + 2] = term.im ? 1 : 0; tf[t * 4 + 3] = term.scale;
        t++;
      }
      ranges[o * 4 + 1] = t;
    });
    return { ranges: this.upload(ranges), terms: this.upload(new Float32Array(termData)) };
  }

  /** Prepared Legendre synthesis: spectral buffer -> fourier buffer, for a fixed recipe set. */
  prepareSynth(spec: GPUBuffer, fourier: GPUBuffer, recipes: Recipe[]): (pass: GPUComputePassEncoder) => void {
    const { ranges, terms } = this.encodeRecipes(recipes);
    const t = this.tables;
    const bg = this.device.createBindGroup({ layout: this.pipes.synth.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: spec } }, { binding: 1, resource: { buffer: fourier } },
      { binding: 2, resource: { buffer: t.P } }, { binding: 3, resource: { buffer: t.H } },
      { binding: 4, resource: { buffer: t.mStart } }, { binding: 5, resource: { buffer: ranges } }, { binding: 6, resource: { buffer: terms } },
    ] });
    const nx = Math.ceil(this.nfour * this.half / WG), ny = recipes.length;
    return (pass) => { pass.setPipeline(this.pipes.synth); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(nx, ny); };
  }

  /** Prepared Legendre analysis: fourier buffer -> spectral buffer (overwrite or accumulate). */
  prepareAnal(fourier: GPUBuffer, spec: GPUBuffer, recipes: Recipe[], accumulate = false): (pass: GPUComputePassEncoder) => void {
    const { ranges, terms } = this.encodeRecipes(recipes);
    const t = this.tables;
    const flags = this.upload(Uint32Array.from([accumulate ? 1 : 0, 0, 0, 0]), GPUBufferUsage.UNIFORM);
    const bg = this.device.createBindGroup({ layout: this.pipes.anal.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: fourier } }, { binding: 1, resource: { buffer: spec } },
      { binding: 2, resource: { buffer: t.PW } }, { binding: 3, resource: { buffer: t.HW } },
      { binding: 4, resource: { buffer: t.mOf } }, { binding: 5, resource: { buffer: t.nOf } },
      { binding: 6, resource: { buffer: ranges } }, { binding: 7, resource: { buffer: terms } }, { binding: 8, resource: { buffer: flags } },
    ] });
    const nx = Math.ceil(this.nspec / WG), ny = recipes.length;
    return (pass) => { pass.setPipeline(this.pipes.anal); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(nx, ny); };
  }

  /** Prepared inverse FFT of fields [f0, f0+count) of a fourier buffer into the same slots of a grid buffer. */
  prepareFftInverse(fourier: GPUBuffer, grid: GPUBuffer, count: number, f0 = 0, g0 = f0): (pass: GPUComputePassEncoder) => void {
    const params = this.upload(Uint32Array.from([f0, g0, 0, 0]), GPUBufferUsage.UNIFORM);
    const bg = this.device.createBindGroup({ layout: this.pipes.fftInv.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: fourier } }, { binding: 1, resource: { buffer: grid } }, { binding: 2, resource: { buffer: params } },
      { binding: 3, resource: { buffer: this.tables.tw } },
    ] });
    return (pass) => { pass.setPipeline(this.pipes.fftInv); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(this.nlat, count); };
  }

  /**
   * Prepared forward FFT of grid fields [g0, g0+count) into fourier slots [f0, ...).
   * scaleMask: per-field flag (bit i for field g0+i, up to 32 per call) to multiply by 1/(1-mu^2).
   */
  prepareFftForward(grid: GPUBuffer, fourier: GPUBuffer, count: number, g0: number, f0: number, invCos2: boolean): (pass: GPUComputePassEncoder) => void {
    const params = this.upload(Uint32Array.from([g0, f0, invCos2 ? 1 : 0, 0]), GPUBufferUsage.UNIFORM);
    const bg = this.device.createBindGroup({ layout: this.pipes.fftFwd.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: grid } }, { binding: 1, resource: { buffer: fourier } }, { binding: 2, resource: { buffer: params } },
      { binding: 3, resource: { buffer: this.tables.invc2 } }, { binding: 4, resource: { buffer: this.tables.tw } },
    ] });
    return (pass) => { pass.setPipeline(this.pipes.fftFwd); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(this.nlat, count); };
  }

  async read(buf: GPUBuffer, bytes: number, offset = 0): Promise<Float32Array> {
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, offset, staging, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  write(buf: GPUBuffer, data: Float32Array, offsetBytes = 0): void {
    this.device.queue.writeBuffer(buf, offsetBytes, data.buffer, data.byteOffset, data.byteLength);
  }

  run(...ops: ((pass: GPUComputePassEncoder) => void)[]): void {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (const op of ops) op(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}

const SYNTH_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> spec: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> four: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> P: array<f32>;
@group(0) @binding(3) var<storage, read> H: array<f32>;
@group(0) @binding(4) var<storage, read> mStart: array<u32>;
@group(0) @binding(5) var<storage, read> ranges: array<Range>;
@group(0) @binding(6) var<storage, read> terms: array<Term>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= NF * HALF) { return; }
  let m = idx / HALF;
  let j = idx % HALF;
  let r = ranges[gid.y];
  let s0 = mStart[m];
  let fm = f32(m);
  var north = vec2<f32>(0.0);
  var south = vec2<f32>(0.0);
  for (var t = r.start; t < r.end; t++) {
    let term = terms[t];
    var ev = vec2<f32>(0.0);
    var od = vec2<f32>(0.0);
    let base = term.input * NSPEC;
    for (var n = m; n <= T; n++) {
      let si = s0 + n - m;
      var l: f32;
      if (term.kind == 0u) { l = P[si * HALF + j]; } else { l = H[si * HALF + j]; }
      var c = spec[base + si];
      if (term.im == 1u) { c = vec2<f32>(-fm * c.y, fm * c.x); }
      if (((n - m) & 1u) == 0u) { ev += c * l; } else { od += c * l; }
    }
    north += term.scale * (ev + od);
    if (term.kind == 0u) { south += term.scale * (ev - od); } else { south += term.scale * (od - ev); }
  }
  four[(r.output * NLAT + j) * NF + m] = north;
  four[(r.output * NLAT + (NLAT - 1u - j)) * NF + m] = south;
}
`;

const ANAL_WGSL = /* wgsl */`
struct Flags { accumulate: u32, a: u32, b: u32, c: u32 };
@group(0) @binding(0) var<storage, read> four: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> spec: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> PW: array<f32>;
@group(0) @binding(3) var<storage, read> HW: array<f32>;
@group(0) @binding(4) var<storage, read> mOf: array<u32>;
@group(0) @binding(5) var<storage, read> nOf: array<u32>;
@group(0) @binding(6) var<storage, read> ranges: array<Range>;
@group(0) @binding(7) var<storage, read> terms: array<Term>;
@group(0) @binding(8) var<uniform> flags: Flags;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let si = gid.x;
  if (si >= NSPEC) { return; }
  let r = ranges[gid.y];
  let m = mOf[si];
  let n = nOf[si];
  let fm = f32(m);
  var acc = vec2<f32>(0.0);
  for (var t = r.start; t < r.end; t++) {
    let term = terms[t];
    let useSym = (((n - m) + term.kind) & 1u) == 0u;
    var s = vec2<f32>(0.0);
    for (var j = 0u; j < HALF; j++) {
      let a = four[(term.input * NLAT + j) * NF + m];
      let b = four[(term.input * NLAT + (NLAT - 1u - j)) * NF + m];
      var l: f32;
      if (term.kind == 0u) { l = PW[si * HALF + j]; } else { l = HW[si * HALF + j]; }
      if (useSym) { s += (a + b) * l; } else { s += (a - b) * l; }
    }
    if (term.im == 1u) { s = vec2<f32>(-fm * s.y, fm * s.x); }
    acc += term.scale * s;
  }
  let o = r.output * NSPEC + si;
  if (flags.accumulate == 1u) { spec[o] += acc; } else { spec[o] = acc; }
}
`;

/** Radix-2 Stockham FFT of one latitude row per workgroup (complex FFT of the real row). */
function fftWgsl(n: number, inverse: boolean): string {
  const threads = Math.min(256, n / 2);
  const log2 = Math.round(Math.log2(n));
  const io = inverse ? `
struct Params { f0: u32, g0: u32, a: u32, b: u32 };
@group(0) @binding(0) var<storage, read> four: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> grid: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> tw: array<vec2<f32>>;` : `
struct Params { g0: u32, f0: u32, invCos2: u32, b: u32 };
@group(0) @binding(0) var<storage, read> grid: array<f32>;
@group(0) @binding(1) var<storage, read_write> four: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> invc2: array<f32>;
@group(0) @binding(4) var<storage, read> tw: array<vec2<f32>>;`;
  const load = inverse ? `
  for (var k = lid; k < NLON; k += ${threads}u) {
    var v = vec2<f32>(0.0);
    if (k <= T) { v = four[((params.f0 + field) * NLAT + row) * NF + k]; if (k == 0u) { v.y = 0.0; } }
    else if (NLON - k <= T) { let c = four[((params.f0 + field) * NLAT + row) * NF + (NLON - k)]; v = vec2<f32>(c.x, -c.y); }
    buf0[k] = v;
  }` : `
  for (var k = lid; k < NLON; k += ${threads}u) {
    buf0[k] = vec2<f32>(grid[((params.g0 + field) * NLAT + row) * NLON + k], 0.0);
  }`;
  const store = inverse ? `
  for (var k = lid; k < NLON; k += ${threads}u) {
    grid[((params.g0 + field) * NLAT + row) * NLON + k] = RES[k].x;
  }` : `
  var sc = 1.0 / f32(NLON);
  if (params.invCos2 == 1u) { sc *= invc2[row]; }
  for (var k = lid; k <= T; k += ${threads}u) {
    four[((params.f0 + field) * NLAT + row) * NF + k] = RES[k] * sc;
  }`;
  // Stockham autosort (OTFFT formulation): stage with length n, stride s, ping-pong buf0/buf1
  let stages = '';
  for (let st = 0; st < log2; st++) {
    const src = st % 2 === 0 ? 'buf0' : 'buf1', dst = st % 2 === 0 ? 'buf1' : 'buf0';
    const len = n >> st, stride = 1 << st, m = len / 2;
    stages += `
  for (var t = lid; t < NLON / 2u; t += ${threads}u) {
    let p = t / ${stride}u;
    let q = t % ${stride}u;
    let tws = tw[p * ${stride}u];
    let w = vec2<f32>(tws.x, ${inverse ? '-' : ''}tws.y);
    let a = ${src}[q + ${stride}u * p];
    let b = ${src}[q + ${stride}u * (p + ${m}u)];
    let d = a - b;
    ${dst}[q + ${stride}u * (2u * p)] = a + b;
    ${dst}[q + ${stride}u * (2u * p + 1u)] = vec2<f32>(d.x * w.x - d.y * w.y, d.x * w.y + d.y * w.x);
  }
  workgroupBarrier();`;
  }
  const res = log2 % 2 === 0 ? 'buf0' : 'buf1';
  return `${io}
var<workgroup> buf0: array<vec2<f32>, ${n}>;
var<workgroup> buf1: array<vec2<f32>, ${n}>;
@compute @workgroup_size(${threads})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let row = wid.x;
  let field = wid.y;
  ${load}
  workgroupBarrier();
  ${stages}
  ${store.replace(/RES/g, res)}
}
`;
}
