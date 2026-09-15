import { DRY_AIR } from '../core/constants.js';
import type { HydrostaticReference1D } from '../corev2/hydrostaticReference.js';
import {
  MAX_RECONSTRUCTION_NEIGHBORS,
  radialFaceCentroid,
  sideFaceCentroid,
  type LinearReconstructionStencil,
} from '../corev2/reconstruction.js';
import type { SphericalShellGeometry } from '../corev2/sphericalShellGeometry.js';
import { coreV2ReferenceTotalEnergyF32 } from './coreV2ReferenceF32.js';

export const CORE_V2_GPU_GRADIENT_FLOATS_PER_CELL = 20;

type GPUAny = any;
const USAGE = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 } as const;
const MAP_READ = 1;

function alignedSize(bytes: number): number {
  return Math.max(4, Math.ceil(bytes / 4) * 4);
}

function upload(device: GPUAny, data: ArrayBufferView, usage: number, label: string): GPUAny {
  const buffer = device.createBuffer({
    label,
    size: alignedSize(data.byteLength),
    usage: usage | USAGE.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  );
  buffer.unmap();
  return buffer;
}

function empty(device: GPUAny, bytes: number, usage: number, label: string): GPUAny {
  return device.createBuffer({ label, size: alignedSize(bytes), usage });
}

async function readback(device: GPUAny, source: GPUAny, floats: number): Promise<Float32Array> {
  const bytes = floats * 4;
  const staging = empty(device, bytes, USAGE.MAP_READ | USAGE.COPY_DST, 'core-v2-reconstruction-readback');
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(MAP_READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

export interface CoreV2GpuReconstructionStaticData {
  cellCount: number;
  nz: number;
  /** Two vec4<u32> per cell: four ids, then ids 4/5 + neighbor count. */
  neighborMeta: Uint32Array;
  /** Six vec4<f32> per cell: xyz least-squares coefficient. */
  neighborCoeff: Float32Array;
  /** Six vec4<f32> per cell: face-centroid minus cell-centroid. */
  faceDisplacement: Float32Array;
  /** One vec4<f32> per vertical layer: rho_ref, p_ref, Phi_ref, rhoE_ref. */
  referenceLayer: Float32Array;
}

export function buildCoreV2GpuReconstructionStaticData(
  geometry: SphericalShellGeometry,
  stencil: LinearReconstructionStencil,
  reference: HydrostaticReference1D,
): CoreV2GpuReconstructionStaticData {
  if (stencil.geometry !== geometry) {
    throw new Error('Core v2 GPU reconstruction received foreign stencil');
  }
  const cellCount = geometry.horizontal.cellCount * geometry.nz;
  if (stencil.neighborCount.length !== cellCount) {
    throw new Error('Core v2 GPU reconstruction stencil size mismatch');
  }
  if (
    reference.cellDensity.length !== geometry.nz ||
    reference.cellPressure.length !== geometry.nz ||
    reference.cellGeopotential.length !== geometry.nz
  ) {
    throw new Error('Core v2 GPU reconstruction reference size mismatch');
  }

  const neighborMeta = new Uint32Array(cellCount * 8);
  const neighborCoeff = new Float32Array(cellCount * MAX_RECONSTRUCTION_NEIGHBORS * 4);
  const faceDisplacement = new Float32Array(cellCount * 6 * 4);
  const referenceLayer = new Float32Array(geometry.nz * 4);

  for (let k = 0; k < geometry.nz; k++) {
    const density = reference.cellDensity[k]!;
    const pressure = reference.cellPressure[k]!;
    const geopotential = reference.cellGeopotential[k]!;
    referenceLayer[k * 4] = density;
    referenceLayer[k * 4 + 1] = pressure;
    referenceLayer[k * 4 + 2] = geopotential;
    referenceLayer[k * 4 + 3] = coreV2ReferenceTotalEnergyF32(density, pressure, geopotential);
  }

  for (let q = 0; q < cellCount; q++) {
    const count = stencil.neighborCount[q]!;
    if (count > MAX_RECONSTRUCTION_NEIGHBORS) {
      throw new Error(`Core v2 GPU reconstruction too many neighbors at ${q}: ${count}`);
    }
    for (let j = 0; j < MAX_RECONSTRUCTION_NEIGHBORS; j++) {
      const sourceSlot = q * MAX_RECONSTRUCTION_NEIGHBORS + j;
      const neighbor = j < count ? stencil.neighborCell[sourceSlot]! : q;
      neighborMeta[q * 8 + j] = neighbor;
      const coefficientSlot = (q * MAX_RECONSTRUCTION_NEIGHBORS + j) * 4;
      neighborCoeff[coefficientSlot] = stencil.coeffX[sourceSlot] ?? 0;
      neighborCoeff[coefficientSlot + 1] = stencil.coeffY[sourceSlot] ?? 0;
      neighborCoeff[coefficientSlot + 2] = stencil.coeffZ[sourceSlot] ?? 0;
    }
    neighborMeta[q * 8 + 6] = count;

    const c = Math.floor(q / geometry.nz);
    const k = q % geometry.nz;
    const x0 = [
      stencil.cellCentroid[q * 3]!,
      stencil.cellCentroid[q * 3 + 1]!,
      stencil.cellCentroid[q * 3 + 2]!,
    ] as const;
    const positions = [] as Array<readonly [number, number, number]>;
    for (let s = 0; s < 4; s++) {
      const edge = geometry.horizontal.cellEdges[c * 4 + s]!;
      positions.push(sideFaceCentroid(geometry, edge, k));
    }
    positions.push(radialFaceCentroid(geometry, c, k));
    positions.push(radialFaceCentroid(geometry, c, k + 1));
    for (let s = 0; s < 6; s++) {
      const position = positions[s]!;
      const base = (q * 6 + s) * 4;
      faceDisplacement[base] = position[0] - x0[0];
      faceDisplacement[base + 1] = position[1] - x0[1];
      faceDisplacement[base + 2] = position[2] - x0[2];
    }
  }

  return {
    cellCount,
    nz: geometry.nz,
    neighborMeta,
    neighborCoeff,
    faceDisplacement,
    referenceLayer,
  };
}

const SHADER = /* wgsl */`
struct Params {
  cellCount:u32,
  nz:u32,
  _pad0:u32,
  _pad1:u32,
  gamma:f32,
  _pad2:f32,
  _pad3:f32,
  _pad4:f32,
};
@group(0) @binding(0) var<uniform> P:Params;
@group(0) @binding(1) var<storage,read> state:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> neighborMeta:array<vec4<u32>>;
@group(0) @binding(3) var<storage,read> coeff:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read> faceDx:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read> refLayer:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read_write> gradient:array<vec4<f32>>;

fn neighbor_id(q:u32,j:u32)->u32 {
  if(j<4u){return neighborMeta[2u*q][j];}
  return neighborMeta[2u*q+1u][j-4u];
}
fn neighbor_count(q:u32)->u32{return neighborMeta[2u*q+1u].z;}

fn primitive_value(q:u32,variable:u32)->f32 {
  let a=state[2u*q];
  let b=state[2u*q+1u];
  let rho=a.x;
  if(variable==0u){return rho;}
  if(variable==1u){return a.y/rho;}
  if(variable==2u){return a.z/rho;}
  if(variable==3u){return a.w/rho;}
  let k=q-P.nz*(q/P.nz);
  let refState=refLayer[k];
  let kinetic=0.5*(a.y*a.y+a.z*a.z+a.w*a.w)/rho;
  let internalPerturbation=(b.x-refState.w)-kinetic-(rho-refState.x)*refState.z;
  return (P.gamma-1.0)*internalPerturbation;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let q=gid.x;
  if(q>=P.cellCount){return;}
  let count=neighbor_count(q);
  for(var variable:u32=0u; variable<5u; variable++){
    let q0=primitive_value(q,variable);
    var g=vec3<f32>(0.0);
    var qMin=q0;
    var qMax=q0;
    for(var j:u32=0u;j<6u;j++){
      if(j>=count){break;}
      let nq=neighbor_id(q,j);
      let qn=primitive_value(nq,variable);
      let delta=qn-q0;
      g += coeff[q*6u+j].xyz*delta;
      qMin=min(qMin,qn);
      qMax=max(qMax,qn);
    }
    var limiter=1.0;
    for(var s:u32=0u;s<6u;s++){
      let delta=dot(g,faceDx[q*6u+s].xyz);
      if(delta>0.0){
        limiter=min(limiter,(qMax-q0)/delta);
      }else if(delta<0.0){
        limiter=min(limiter,(qMin-q0)/delta);
      }
    }
    limiter=clamp(limiter,0.0,1.0);
    gradient[q*5u+variable]=vec4<f32>(limiter*g,0.0);
  }
}
`;

export class CoreV2GpuReconstruction {
  readonly device: GPUAny;
  readonly staticData: CoreV2GpuReconstructionStaticData;
  private readonly pipeline: GPUAny;
  private readonly pipelineValidation: Promise<any>;
  private readonly compilationInfo: Promise<any>;
  private readonly params: GPUAny;
  private readonly neighborMeta: GPUAny;
  private readonly neighborCoeff: GPUAny;
  private readonly faceDisplacement: GPUAny;
  private readonly referenceLayer: GPUAny;

  constructor(device: GPUAny, staticData: CoreV2GpuReconstructionStaticData) {
    this.device = device;
    this.staticData = staticData;
    const paramRaw = new ArrayBuffer(32);
    const u32 = new Uint32Array(paramRaw);
    const f32 = new Float32Array(paramRaw);
    u32[0] = staticData.cellCount;
    u32[1] = staticData.nz;
    f32[4] = DRY_AIR.gamma;
    this.params = upload(device, new Uint8Array(paramRaw), USAGE.UNIFORM, 'core-v2-recon-params');
    this.neighborMeta = upload(device, staticData.neighborMeta, USAGE.STORAGE, 'core-v2-recon-neighbors');
    this.neighborCoeff = upload(device, staticData.neighborCoeff, USAGE.STORAGE, 'core-v2-recon-coeff');
    this.faceDisplacement = upload(device, staticData.faceDisplacement, USAGE.STORAGE, 'core-v2-recon-facedx');
    this.referenceLayer = upload(device, staticData.referenceLayer, USAGE.STORAGE, 'core-v2-recon-reference');
    this.device.pushErrorScope('validation');
    const module = device.createShaderModule({ label: 'core-v2-reconstruction-module', code: SHADER });
    this.compilationInfo = module.getCompilationInfo();
    this.pipeline = device.createComputePipeline({
      label: 'core-v2-reconstruction-pipeline',
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    this.pipelineValidation = this.device.popErrorScope();
  }

  async computeGradients(packedState: Float32Array): Promise<Float32Array> {
    if (packedState.length !== this.staticData.cellCount * 8) {
      throw new Error('Core v2 GPU reconstruction packed-state size mismatch');
    }
    const setupError = await this.pipelineValidation;
    if (setupError) {
      const info = await this.compilationInfo;
      const messages = Array.from(info.messages ?? [])
        .map((message: any) => `${message.type ?? 'message'} ${message.lineNum ?? '?'}:${message.linePos ?? '?'} ${message.message ?? String(message)}`)
        .join(' | ');
      throw new Error(`Core v2 GPU reconstruction pipeline error: ${setupError.message}${messages ? `; shader: ${messages}` : ''}`);
    }
    const state = upload(
      this.device,
      packedState,
      USAGE.STORAGE,
      'core-v2-reconstruction-state',
    );
    const output = empty(
      this.device,
      this.staticData.cellCount * CORE_V2_GPU_GRADIENT_FLOATS_PER_CELL * 4,
      USAGE.STORAGE | USAGE.COPY_SRC,
      'core-v2-reconstruction-output',
    );
    this.device.pushErrorScope('validation');
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: state } },
        { binding: 2, resource: { buffer: this.neighborMeta } },
        { binding: 3, resource: { buffer: this.neighborCoeff } },
        { binding: 4, resource: { buffer: this.faceDisplacement } },
        { binding: 5, resource: { buffer: this.referenceLayer } },
        { binding: 6, resource: { buffer: output } },
      ],
    });
    const encoder = this.device.createCommandEncoder({ label: 'core-v2-reconstruction-encoder' });
    const pass = encoder.beginComputePass({ label: 'core-v2-reconstruction-pass' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.staticData.cellCount / 128));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const validationError = await this.device.popErrorScope();
    if (validationError) {
      state.destroy();
      output.destroy();
      throw new Error(`Core v2 GPU reconstruction validation error: ${validationError.message}`);
    }
    const result = await readback(
      this.device,
      output,
      this.staticData.cellCount * CORE_V2_GPU_GRADIENT_FLOATS_PER_CELL,
    );
    state.destroy();
    output.destroy();
    return result;
  }

  destroy(): void {
    this.params.destroy();
    this.neighborMeta.destroy();
    this.neighborCoeff.destroy();
    this.faceDisplacement.destroy();
    this.referenceLayer.destroy();
  }
}
