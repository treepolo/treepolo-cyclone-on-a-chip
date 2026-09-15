import { DRY_AIR, EARTH, type PlanetConfig } from '../core/constants.js';
import type { HydrostaticReference1D } from '../corev2/hydrostaticReference.js';
import type { SphericalShellGeometry } from '../corev2/sphericalShellGeometry.js';
import {
  buildCoreV2GpuHeviVerticalStaticData,
  type CoreV2GpuHeviVerticalStaticData,
} from './coreV2HeviVerticalGpu.js';

export const CORE_V2_GPU_HEVI_JACOBIAN_FLOATS_PER_CELL = 75;
export const CORE_V2_GPU_HEVI_JACOBIAN_BLOCK_FLOATS = 25;

export interface CoreV2GpuHeviJacobianStaticData extends CoreV2GpuHeviVerticalStaticData {
  inverseCellVolume: Float32Array;
}

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
  const staging = empty(device, bytes, USAGE.MAP_READ | USAGE.COPY_DST, 'core-v2-hevi-jacobian-readback');
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(MAP_READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

export function buildCoreV2GpuHeviJacobianStaticData(
  geometry: SphericalShellGeometry,
  reference: HydrostaticReference1D,
  planet: PlanetConfig = EARTH,
): CoreV2GpuHeviJacobianStaticData {
  const base = buildCoreV2GpuHeviVerticalStaticData(geometry, reference, planet);
  const inverseCellVolume = new Float32Array(base.cellCount);
  for (let q = 0; q < base.cellCount; q++) {
    inverseCellVolume[q] = 1 / geometry.cellVolume[q]!;
  }
  return { ...base, inverseCellVolume };
}

/**
 * The CPU nonlinear HEVI stage uses forward finite differences to form the
 * block-tridiagonal Jacobian of V(U). Float32 cannot use the CPU f64 step
 * sqrt(Number.EPSILON), because that perturbation often rounds away entirely.
 * This is the corresponding f32 forward-difference scale sqrt(FLT_EPSILON).
 */
export const CORE_V2_GPU_FD_SQRT_EPSILON = Math.sqrt(2 ** -23);

const SHADER = /* wgsl */`
struct Params{
  cellCount:u32,
  nz:u32,
  _p0:u32,
  _p1:u32,
  gamma:f32,
  fdScale:f32,
  _p2:f32,
  _p3:f32,
};
@group(0) @binding(0) var<uniform> P:Params;
@group(0) @binding(1) var<storage,read> state:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> faceStatic:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> gravityCoeff:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read> boundaryArea:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read> refLayer:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read> invVolume:array<f32>;
@group(0) @binding(7) var<storage,read_write> jacobian:array<f32>;

struct Rate5{a:vec4<f32>,e:f32};

fn beta_plus(m:f32)->f32{
  if(abs(m)<1.0){return .25*(2.0-m)*(m+1.0)*(m+1.0);}
  return select(0.0,1.0,m>=0.0);
}
fn beta_minus(m:f32)->f32{
  if(abs(m)<1.0){return .25*(2.0+m)*(m-1.0)*(m-1.0);}
  return select(1.0,0.0,m>=0.0);
}

fn load_a(q:u32,sourceQ:u32,sourceVar:u32,h:f32)->vec4<f32>{
  var a=state[2u*q];
  if(q==sourceQ){
    if(sourceVar==0u){a.x+=h;}
    else if(sourceVar==1u){a.y+=h;}
    else if(sourceVar==2u){a.z+=h;}
    else if(sourceVar==3u){a.w+=h;}
  }
  return a;
}
fn load_e(q:u32,sourceQ:u32,sourceVar:u32,h:f32)->f32{
  var e=state[2u*q+1u].x;
  if(q==sourceQ && sourceVar==4u){e+=h;}
  return e;
}

fn pprime(q:u32,sourceQ:u32,sourceVar:u32,h:f32)->f32{
  let a=load_a(q,sourceQ,sourceVar,h);
  let e=load_e(q,sourceQ,sourceVar,h);
  let rho=a.x;
  let k=q-P.nz*(q/P.nz);
  let r=refLayer[k];
  let ke=.5*(a.y*a.y+a.z*a.z+a.w*a.w)/rho;
  return (P.gamma-1.0)*((e-r.w)-ke-(rho-r.x)*r.z);
}

fn radial_flux(f:u32,sourceQ:u32,sourceVar:u32,h:f32)->Rate5{
  let slots=P.nz-1u;
  let c=f/slots;
  let ki=f-c*slots+1u;
  let lo=c*P.nz+ki-1u;
  let hi=lo+1u;
  let fs0=faceStatic[2u*f];
  let fs1=faceStatic[2u*f+1u];
  let av=fs0.xyz;
  let pRef=fs0.w;
  let phi=fs1.x;
  let area=length(av);
  let n=av/area;
  let aL=load_a(lo,sourceQ,sourceVar,h);
  let aR=load_a(hi,sourceQ,sourceVar,h);
  let rhoL=aL.x;
  let rhoR=aR.x;
  let uL=aL.yzw/rhoL;
  let uR=aR.yzw/rhoR;
  let pL=pRef+pprime(lo,sourceQ,sourceVar,h);
  let pR=pRef+pprime(hi,sourceQ,sourceVar,h);
  let csL=sqrt(P.gamma*pL/rhoL);
  let csR=sqrt(P.gamma*pR/rhoR);
  let cs=.5*(csL+csR);
  let unL=dot(uL,n);
  let unR=dot(uR,n);
  let s2L=dot(uL,uL);
  let s2R=dot(uR,uR);
  let mL=unL/cs;
  let mR=unR/cs;
  let vr=sqrt(.5*(s2L+s2R));
  let mh=min(1.0,vr/cs);
  let chi=(1.0-mh)*(1.0-mh);
  let ex=-max(min(mL,0.0),-1.0)*min(max(mR,0.0),1.0);
  let wa=(rhoL*abs(unL)+rhoR*abs(unR))/(rhoL+rhoR);
  let al=(1.0-ex)*wa+ex*abs(unL);
  let ar=(1.0-ex)*wa+ex*abs(unR);
  let ppL=pL-pRef;
  let ppR=pR-pRef;
  let mass=.5*(rhoL*(unL+al)+rhoR*(unR-ar)-(chi/cs)*(ppR-ppL));
  let bp=beta_plus(mL);
  let bm=beta_minus(mR);
  let pressure=.5*(ppL+ppR)+.5*(bp-bm)*(ppL-ppR)+vr*(bp+bm-1.0)*cs*.5*(rhoL+rhoR);
  let leftUp=mass>=0.0;
  let uu=select(uR,uL,leftUp);
  let pu=select(pR,pL,leftUp);
  let ru=select(rhoR,rhoL,leftUp);
  let s2=select(s2R,s2L,leftUp);
  let hTotal=P.gamma/(P.gamma-1.0)*pu/ru+.5*s2+phi;
  let mf=mass*uu+pressure*n;
  var out:Rate5;
  out.a=vec4<f32>(area*mass,area*mf.x,area*mf.y,area*mf.z);
  out.e=area*mass*hTotal;
  return out;
}

fn tendency(q:u32,sourceQ:u32,sourceVar:u32,h:f32)->Rate5{
  let c=q/P.nz;
  let k=q-c*P.nz;
  let slots=P.nz-1u;
  var out:Rate5;
  out.a=vec4<f32>(0.0);
  out.e=0.0;
  if(k>0u){
    let f=c*slots+k-1u;
    let flux=radial_flux(f,sourceQ,sourceVar,h);
    out.a+=flux.a;
    out.e+=flux.e;
  }
  if(k+1u<P.nz){
    let f=c*slots+k;
    let flux=radial_flux(f,sourceQ,sourceVar,h);
    out.a-=flux.a;
    out.e-=flux.e;
  }
  let r=refLayer[k];
  let rhoPrime=load_a(q,sourceQ,sourceVar,h).x-r.x;
  out.a.yzw+=rhoPrime*gravityCoeff[q].xyz;
  let pp=pprime(q,sourceQ,sourceVar,h);
  if(k==0u){out.a.yzw-=pp*boundaryArea[2u*c].xyz;}
  if(k+1u==P.nz){out.a.yzw-=pp*boundaryArea[2u*c+1u].xyz;}
  let invV=invVolume[q];
  out.a*=invV;
  out.e*=invV;
  return out;
}

fn component_value(q:u32,v:u32)->f32{
  let a=state[2u*q];
  if(v==0u){return a.x;}
  if(v==1u){return a.y;}
  if(v==2u){return a.z;}
  if(v==3u){return a.w;}
  return state[2u*q+1u].x;
}
fn component_scale(q:u32,v:u32)->f32{
  let k=q-P.nz*(q/P.nz);
  let r=refLayer[k];
  if(v==0u){return max(r.x,1e-8);}
  if(v>=1u && v<=3u){
    let sound=sqrt(P.gamma*r.y/r.x);
    return max(r.x*sound,1e-6);
  }
  let internal=r.y/(P.gamma-1.0);
  let potential=abs(r.x*r.z);
  return max(internal+potential,1.0);
}
fn rate_component(rate:Rate5,v:u32)->f32{
  if(v==0u){return rate.a.x;}
  if(v==1u){return rate.a.y;}
  if(v==2u){return rate.a.z;}
  if(v==3u){return rate.a.w;}
  return rate.e;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let sourceLinear=gid.x;
  let total=P.cellCount*5u;
  if(sourceLinear>=total){return;}
  let sourceQ=sourceLinear/5u;
  let sourceVar=sourceLinear-sourceQ*5u;
  let sourceK=sourceQ-P.nz*(sourceQ/P.nz);
  let value=component_value(sourceQ,sourceVar);
  let scale=max(abs(value),component_scale(sourceQ,sourceVar));
  let h=P.fdScale*scale;
  let rowBegin=select(sourceK-1u,0u,sourceK==0u);
  let rowEnd=min(P.nz-1u,sourceK+1u);
  let column=sourceQ/P.nz;
  for(var rowK:u32=rowBegin;rowK<=rowEnd;rowK++){
    let rowQ=column*P.nz+rowK;
    let base=tendency(rowQ,sourceQ,sourceVar,0.0);
    let pert=tendency(rowQ,sourceQ,sourceVar,h);
    var block:u32=1u;
    if(rowK>sourceK){block=0u;}
    else if(rowK<sourceK){block=2u;}
    let blockBase=(rowQ*3u+block)*25u;
    for(var rowVar:u32=0u;rowVar<5u;rowVar++){
      let derivative=(rate_component(pert,rowVar)-rate_component(base,rowVar))/h;
      jacobian[blockBase+rowVar*5u+sourceVar]=derivative;
    }
  }
}
`;

export class CoreV2GpuHeviJacobian {
  private readonly device: GPUAny;
  readonly staticData: CoreV2GpuHeviJacobianStaticData;
  private readonly pipeline: GPUAny;
  private readonly pipelineValidation: Promise<any>;
  private readonly compilationInfo: Promise<any>;
  private readonly params: GPUAny;
  private readonly faceStatic: GPUAny;
  private readonly gravityCoefficient: GPUAny;
  private readonly boundaryArea: GPUAny;
  private readonly referenceLayer: GPUAny;
  private readonly inverseCellVolume: GPUAny;

  constructor(device: GPUAny, data: CoreV2GpuHeviJacobianStaticData) {
    this.device = device;
    this.staticData = data;
    const raw = new ArrayBuffer(32);
    const u = new Uint32Array(raw);
    const f = new Float32Array(raw);
    u[0] = data.cellCount;
    u[1] = data.nz;
    f[4] = DRY_AIR.gamma;
    f[5] = CORE_V2_GPU_FD_SQRT_EPSILON;
    this.params = upload(device, new Uint8Array(raw), USAGE.UNIFORM, 'core-v2-hevi-jacobian-params');
    this.faceStatic = upload(device, data.faceStatic, USAGE.STORAGE, 'core-v2-hevi-jacobian-faces');
    this.gravityCoefficient = upload(device, data.gravityCoefficient, USAGE.STORAGE, 'core-v2-hevi-jacobian-gravity');
    this.boundaryArea = upload(device, data.boundaryArea, USAGE.STORAGE, 'core-v2-hevi-jacobian-boundary');
    this.referenceLayer = upload(device, data.referenceLayer, USAGE.STORAGE, 'core-v2-hevi-jacobian-reference');
    this.inverseCellVolume = upload(device, data.inverseCellVolume, USAGE.STORAGE, 'core-v2-hevi-jacobian-inv-volume');

    this.device.pushErrorScope('validation');
    const module = device.createShaderModule({ label: 'core-v2-hevi-jacobian-module', code: SHADER });
    this.compilationInfo = module.getCompilationInfo();
    this.pipeline = device.createComputePipeline({
      label: 'core-v2-hevi-jacobian-pipeline',
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    this.pipelineValidation = this.device.popErrorScope();
  }

  async computeTendencyJacobian(packedState: Float32Array): Promise<Float32Array> {
    const d = this.staticData;
    if (packedState.length !== d.cellCount * 8) {
      throw new Error('Core v2 GPU HEVI Jacobian packed-state size mismatch');
    }
    const setupError = await this.pipelineValidation;
    if (setupError) {
      const info = await this.compilationInfo;
      const messages = Array.from(info.messages ?? [])
        .map((message: any) => `${message.type ?? 'message'} ${message.lineNum ?? '?'}:${message.linePos ?? '?'} ${message.message ?? String(message)}`)
        .join(' | ');
      throw new Error(`Core v2 GPU HEVI Jacobian pipeline error: ${setupError.message}${messages ? `; shader: ${messages}` : ''}`);
    }

    const state = upload(this.device, packedState, USAGE.STORAGE, 'core-v2-hevi-jacobian-state');
    const output = empty(
      this.device,
      d.cellCount * CORE_V2_GPU_HEVI_JACOBIAN_FLOATS_PER_CELL * 4,
      USAGE.STORAGE | USAGE.COPY_SRC,
      'core-v2-hevi-jacobian-output',
    );

    this.device.pushErrorScope('validation');
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: state } },
        { binding: 2, resource: { buffer: this.faceStatic } },
        { binding: 3, resource: { buffer: this.gravityCoefficient } },
        { binding: 4, resource: { buffer: this.boundaryArea } },
        { binding: 5, resource: { buffer: this.referenceLayer } },
        { binding: 6, resource: { buffer: this.inverseCellVolume } },
        { binding: 7, resource: { buffer: output } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(d.cellCount * 5 / 64));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const validationError = await this.device.popErrorScope();
    if (validationError) {
      state.destroy();
      output.destroy();
      throw new Error(`Core v2 GPU HEVI Jacobian dispatch validation error: ${validationError.message}`);
    }

    const result = await readback(
      this.device,
      output,
      d.cellCount * CORE_V2_GPU_HEVI_JACOBIAN_FLOATS_PER_CELL,
    );
    state.destroy();
    output.destroy();
    return result;
  }

  destroy(): void {
    this.params.destroy();
    this.faceStatic.destroy();
    this.gravityCoefficient.destroy();
    this.boundaryArea.destroy();
    this.referenceLayer.destroy();
    this.inverseCellVolume.destroy();
  }
}
