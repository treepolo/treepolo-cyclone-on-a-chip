import { DRY_AIR, EARTH, type PlanetConfig } from '../core/constants.js';
import type { HydrostaticReference1D } from '../corev2/hydrostaticReference.js';
import {
  radialFaceVectorArea,
  shellCellIndex,
  type SphericalShellGeometry,
} from '../corev2/sphericalShellGeometry.js';
import { coreV2ReferenceTotalEnergyF32 } from './coreV2ReferenceF32.js';

export const CORE_V2_GPU_HEVI_RATE_FLOATS_PER_CELL = 8;

type GPUAny = any;
const USAGE = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 } as const;
const MAP_READ = 1;

function aligned(bytes: number): number {
  return Math.max(4, Math.ceil(bytes / 4) * 4);
}

function upload(device: GPUAny, data: ArrayBufferView, usage: number, label: string): GPUAny {
  const buffer = device.createBuffer({
    label,
    size: aligned(data.byteLength),
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
  return device.createBuffer({ label, size: aligned(bytes), usage });
}

async function readback(device: GPUAny, source: GPUAny, floats: number): Promise<Float32Array> {
  const bytes = floats * 4;
  const staging = empty(device, bytes, USAGE.MAP_READ | USAGE.COPY_DST, 'core-v2-hevi-readback');
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(MAP_READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

export interface CoreV2GpuHeviVerticalStaticData {
  cellCount: number;
  horizontalCellCount: number;
  nz: number;
  internalFaceCount: number;
  /** Two vec4 per internal radial face: area.xyz,p_ref then Phi,0,0,0. */
  faceStatic: Float32Array;
  /** vec4 per cell: exact integrated gravity coefficient xyz, unused. */
  gravityCoefficient: Float32Array;
  /** Two vec4 per horizontal column: bottom outward area, top outward area. */
  boundaryArea: Float32Array;
  /** vec4 per layer: rho_ref,p_ref,Phi_ref,canonical f32 rhoE_ref. */
  referenceLayer: Float32Array;
}

export function buildCoreV2GpuHeviVerticalStaticData(
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  planet: PlanetConfig = EARTH,
): CoreV2GpuHeviVerticalStaticData {
  const hc = geometry.horizontal.cellCount;
  const nz = geometry.nz;
  if (
    reference.cellDensity.length !== nz ||
    reference.cellPressure.length !== nz ||
    reference.cellGeopotential.length !== nz ||
    reference.radialFacePressure.length !== nz + 1
  ) throw new Error('Core v2 GPU HEVI reference/geometry size mismatch');

  const cellCount = hc * nz;
  const internalFaceCount = hc * Math.max(0, nz - 1);
  const faceStatic = new Float32Array(internalFaceCount * 8);
  const gravityCoefficient = new Float32Array(cellCount * 4);
  const boundaryArea = new Float32Array(hc * 8);
  const referenceLayer = new Float32Array(nz * 4);

  for (let k = 0; k < nz; k++) {
    const rho = reference.cellDensity[k]!;
    const p = reference.cellPressure[k]!;
    const phi = reference.cellGeopotential[k]!;
    referenceLayer[k * 4] = rho;
    referenceLayer[k * 4 + 1] = p;
    referenceLayer[k * 4 + 2] = phi;
    referenceLayer[k * 4 + 3] = coreV2ReferenceTotalEnergyF32(rho, p, phi);
  }

  for (let c = 0; c < hc; c++) {
    const bottom = radialFaceVectorArea(geometry, c, 0);
    const top = radialFaceVectorArea(geometry, c, nz);
    const b = c * 8;
    boundaryArea[b] = -bottom[0];
    boundaryArea[b + 1] = -bottom[1];
    boundaryArea[b + 2] = -bottom[2];
    boundaryArea[b + 4] = top[0];
    boundaryArea[b + 5] = top[1];
    boundaryArea[b + 6] = top[2];

    for (let k = 0; k < nz; k++) {
      const q = shellCellIndex(c, k, nz);
      const r0 = geometry.radiusInterface[k]!;
      const r1 = geometry.radiusInterface[k + 1]!;
      const scale = -planet.gravity * (r1 ** 3 - r0 ** 3) / 3;
      const i = q * 4;
      gravityCoefficient[i] = scale * geometry.cellVectorAreaUnit[c * 3]!;
      gravityCoefficient[i + 1] = scale * geometry.cellVectorAreaUnit[c * 3 + 1]!;
      gravityCoefficient[i + 2] = scale * geometry.cellVectorAreaUnit[c * 3 + 2]!;
    }

    for (let ki = 1; ki < nz; ki++) {
      const f = c * (nz - 1) + (ki - 1);
      const area = radialFaceVectorArea(geometry, c, ki);
      const i = f * 8;
      faceStatic[i] = area[0];
      faceStatic[i + 1] = area[1];
      faceStatic[i + 2] = area[2];
      faceStatic[i + 3] = reference.radialFacePressure[ki]!;
      faceStatic[i + 4] = planet.gravity * (
        geometry.radiusInterface[ki]! - geometry.radiusInterface[0]!
      );
    }
  }

  return {
    cellCount,
    horizontalCellCount: hc,
    nz,
    internalFaceCount,
    faceStatic,
    gravityCoefficient,
    boundaryArea,
    referenceLayer,
  };
}

const FACE_SHADER = /* wgsl */`
struct Params{faceCount:u32,nz:u32,_p0:u32,_p1:u32,gamma:f32,_p2:f32,_p3:f32,_p4:f32};
@group(0) @binding(0) var<uniform> P:Params;
@group(0) @binding(1) var<storage,read> state:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> faceStatic:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> refLayer:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> faceFlux:array<vec4<f32>>;

fn beta_plus(m:f32)->f32{if(abs(m)<1.0){return .25*(2.0-m)*(m+1.0)*(m+1.0);}return select(0.0,1.0,m>=0.0);}
fn beta_minus(m:f32)->f32{if(abs(m)<1.0){return .25*(2.0+m)*(m-1.0)*(m-1.0);}return select(1.0,0.0,m>=0.0);}
fn pprime(q:u32)->f32{
  let a=state[2u*q];let b=state[2u*q+1u];let rho=a.x;let k=q-P.nz*(q/P.nz);let r=refLayer[k];
  let ke=.5*(a.y*a.y+a.z*a.z+a.w*a.w)/rho;
  return (P.gamma-1.0)*((b.x-r.w)-ke-(rho-r.x)*r.z);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let f=gid.x;if(f>=P.faceCount){return;}
  let slots=P.nz-1u;let c=f/slots;let ki=f-c*slots+1u;let lo=c*P.nz+ki-1u;let hi=lo+1u;
  let fs0=faceStatic[2u*f];let fs1=faceStatic[2u*f+1u];let av=fs0.xyz;let pRef=fs0.w;let phi=fs1.x;
  let area=length(av);let n=av/area;
  let aL=state[2u*lo];let aR=state[2u*hi];let rhoL=aL.x;let rhoR=aR.x;
  let uL=aL.yzw/rhoL;let uR=aR.yzw/rhoR;let pL=pRef+pprime(lo);let pR=pRef+pprime(hi);
  let csL=sqrt(P.gamma*pL/rhoL);let csR=sqrt(P.gamma*pR/rhoR);let cs=.5*(csL+csR);
  let unL=dot(uL,n);let unR=dot(uR,n);let s2L=dot(uL,uL);let s2R=dot(uR,uR);let mL=unL/cs;let mR=unR/cs;
  let vr=sqrt(.5*(s2L+s2R));let mh=min(1.0,vr/cs);let chi=(1.0-mh)*(1.0-mh);
  let ex=-max(min(mL,0.0),-1.0)*min(max(mR,0.0),1.0);let wa=(rhoL*abs(unL)+rhoR*abs(unR))/(rhoL+rhoR);
  let al=(1.0-ex)*wa+ex*abs(unL);let ar=(1.0-ex)*wa+ex*abs(unR);
  let ppL=pL-pRef;let ppR=pR-pRef;
  let mass=.5*(rhoL*(unL+al)+rhoR*(unR-ar)-(chi/cs)*(ppR-ppL));
  let bp=beta_plus(mL);let bm=beta_minus(mR);
  let pressure=.5*(ppL+ppR)+.5*(bp-bm)*(ppL-ppR)+vr*(bp+bm-1.0)*cs*.5*(rhoL+rhoR);
  let leftUp=mass>=0.0;let uu=select(uR,uL,leftUp);let pu=select(pR,pL,leftUp);let ru=select(rhoR,rhoL,leftUp);let s2=select(s2R,s2L,leftUp);
  let h=P.gamma/(P.gamma-1.0)*pu/ru+.5*s2+phi;let mf=mass*uu+pressure*n;
  faceFlux[2u*f]=vec4<f32>(area*mass,area*mf.x,area*mf.y,area*mf.z);
  faceFlux[2u*f+1u]=vec4<f32>(area*mass*h,0.0,0.0,0.0);
}
`;

const GATHER_SHADER = /* wgsl */`
struct Params{cellCount:u32,nz:u32,_p0:u32,_p1:u32,gamma:f32,_p2:f32,_p3:f32,_p4:f32};
@group(0) @binding(0) var<uniform> P:Params;
@group(0) @binding(1) var<storage,read> state:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> faceFlux:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> gravityCoeff:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read> boundaryArea:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read> refLayer:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read_write> rate:array<vec4<f32>>;

fn pprime(q:u32)->f32{
  let a=state[2u*q];let b=state[2u*q+1u];let rho=a.x;let k=q-P.nz*(q/P.nz);let r=refLayer[k];
  let ke=.5*(a.y*a.y+a.z*a.z+a.w*a.w)/rho;
  return (P.gamma-1.0)*((b.x-r.w)-ke-(rho-r.x)*r.z);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let q=gid.x;if(q>=P.cellCount){return;}let c=q/P.nz;let k=q-c*P.nz;let slots=P.nz-1u;
  var a=vec4<f32>(0.0);var e=0.0;
  if(k>0u){let f=c*slots+k-1u;a+=faceFlux[2u*f];e+=faceFlux[2u*f+1u].x;}
  if(k+1u<P.nz){let f=c*slots+k;a-=faceFlux[2u*f];e-=faceFlux[2u*f+1u].x;}
  let r=refLayer[k];let rhoPrime=state[2u*q].x-r.x;a.yzw+=rhoPrime*gravityCoeff[q].xyz;
  let pp=pprime(q);
  if(k==0u){a.yzw-=pp*boundaryArea[2u*c].xyz;}
  if(k+1u==P.nz){a.yzw-=pp*boundaryArea[2u*c+1u].xyz;}
  rate[2u*q]=a;rate[2u*q+1u]=vec4<f32>(e,0.0,0.0,0.0);
}
`;

export class CoreV2GpuHeviVertical {
  private readonly device: GPUAny;
  readonly staticData: CoreV2GpuHeviVerticalStaticData;
  private readonly facePipeline: GPUAny;
  private readonly gatherPipeline: GPUAny;
  private readonly paramsFace: GPUAny;
  private readonly paramsCell: GPUAny;
  private readonly faceStatic: GPUAny;
  private readonly gravityCoefficient: GPUAny;
  private readonly boundaryArea: GPUAny;
  private readonly referenceLayer: GPUAny;

  constructor(device: GPUAny, data: CoreV2GpuHeviVerticalStaticData) {
    this.device = device;
    this.staticData = data;
    const makeParams = (count: number): Uint8Array => {
      const raw = new ArrayBuffer(32);
      const u = new Uint32Array(raw);
      const f = new Float32Array(raw);
      u[0] = count;
      u[1] = data.nz;
      f[4] = DRY_AIR.gamma;
      return new Uint8Array(raw);
    };
    this.paramsFace = upload(device, makeParams(data.internalFaceCount), USAGE.UNIFORM, 'core-v2-hevi-face-params');
    this.paramsCell = upload(device, makeParams(data.cellCount), USAGE.UNIFORM, 'core-v2-hevi-cell-params');
    this.faceStatic = upload(device, data.faceStatic, USAGE.STORAGE, 'core-v2-hevi-face-static');
    this.gravityCoefficient = upload(device, data.gravityCoefficient, USAGE.STORAGE, 'core-v2-hevi-gravity');
    this.boundaryArea = upload(device, data.boundaryArea, USAGE.STORAGE, 'core-v2-hevi-boundary');
    this.referenceLayer = upload(device, data.referenceLayer, USAGE.STORAGE, 'core-v2-hevi-reference');
    const faceModule = device.createShaderModule({ label: 'core-v2-hevi-face-module', code: FACE_SHADER });
    const gatherModule = device.createShaderModule({ label: 'core-v2-hevi-gather-module', code: GATHER_SHADER });
    this.facePipeline = device.createComputePipeline({ label: 'core-v2-hevi-face-pipeline', layout: 'auto', compute: { module: faceModule, entryPoint: 'main' } });
    this.gatherPipeline = device.createComputePipeline({ label: 'core-v2-hevi-gather-pipeline', layout: 'auto', compute: { module: gatherModule, entryPoint: 'main' } });
  }

  async computeIntegratedRate(packedState: Float32Array): Promise<Float32Array> {
    const d = this.staticData;
    if (packedState.length !== d.cellCount * 8) throw new Error('Core v2 GPU HEVI packed-state size mismatch');
    const state = upload(this.device, packedState, USAGE.STORAGE, 'core-v2-hevi-state');
    const faceFlux = empty(this.device, Math.max(1, d.internalFaceCount * 8) * 4, USAGE.STORAGE | USAGE.COPY_SRC, 'core-v2-hevi-face-flux');
    const rate = empty(this.device, d.cellCount * 8 * 4, USAGE.STORAGE | USAGE.COPY_SRC, 'core-v2-hevi-rate');

    if (d.internalFaceCount > 0) {
      const bg = this.device.createBindGroup({ layout: this.facePipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.paramsFace } },
        { binding: 1, resource: { buffer: state } },
        { binding: 2, resource: { buffer: this.faceStatic } },
        { binding: 3, resource: { buffer: this.referenceLayer } },
        { binding: 4, resource: { buffer: faceFlux } },
      ] });
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.facePipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(d.internalFaceCount / 128));
      pass.end();
      this.device.queue.submit([encoder.finish()]);
    }

    const gatherBg = this.device.createBindGroup({ layout: this.gatherPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.paramsCell } },
      { binding: 1, resource: { buffer: state } },
      { binding: 2, resource: { buffer: faceFlux } },
      { binding: 3, resource: { buffer: this.gravityCoefficient } },
      { binding: 4, resource: { buffer: this.boundaryArea } },
      { binding: 5, resource: { buffer: this.referenceLayer } },
      { binding: 6, resource: { buffer: rate } },
    ] });
    const gatherEncoder = this.device.createCommandEncoder();
    const gatherPass = gatherEncoder.beginComputePass();
    gatherPass.setPipeline(this.gatherPipeline);
    gatherPass.setBindGroup(0, gatherBg);
    gatherPass.dispatchWorkgroups(Math.ceil(d.cellCount / 128));
    gatherPass.end();
    this.device.queue.submit([gatherEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    const result = await readback(this.device, rate, d.cellCount * 8);
    state.destroy();
    faceFlux.destroy();
    rate.destroy();
    return result;
  }

  destroy(): void {
    this.paramsFace.destroy();
    this.paramsCell.destroy();
    this.faceStatic.destroy();
    this.gravityCoefficient.destroy();
    this.boundaryArea.destroy();
    this.referenceLayer.destroy();
  }
}
