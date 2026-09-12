import { DRY_AIR, EARTH, type PlanetConfig } from '../core/constants.js';
import type { HydrostaticReference1D } from '../corev2/hydrostaticReference.js';
import { radialFaceCentroid, type LinearReconstructionStencil } from '../corev2/reconstruction.js';
import {
  radialFaceVectorArea,
  shellCellIndex,
  type SphericalShellGeometry,
} from '../corev2/sphericalShellGeometry.js';
import type { CoreV2GpuInternalFaceStaticData } from './coreV2FaceFluxGpu.js';

export const CORE_V2_GPU_RATE_FLOATS_PER_CELL = 8;

type GPUAny=any;
const USAGE={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128} as const;
const MAP_READ=1;
function aligned(n:number):number{return Math.max(4,Math.ceil(n/4)*4);}
function upload(device:GPUAny,data:ArrayBufferView,usage:number,label:string):GPUAny{const b=device.createBuffer({label,size:aligned(data.byteLength),usage:usage|USAGE.COPY_DST,mappedAtCreation:true});new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));b.unmap();return b;}
function empty(device:GPUAny,bytes:number,usage:number,label:string):GPUAny{return device.createBuffer({label,size:aligned(bytes),usage});}
async function readback(device:GPUAny,source:GPUAny,floats:number):Promise<Float32Array>{const bytes=floats*4;const s=empty(device,bytes,USAGE.MAP_READ|USAGE.COPY_DST,'core-v2-rate-readback');const e=device.createCommandEncoder();e.copyBufferToBuffer(source,0,s,0,bytes);device.queue.submit([e.finish()]);await s.mapAsync(MAP_READ);const out=new Float32Array(s.getMappedRange().slice(0));s.unmap();s.destroy();return out;}

export interface CoreV2GpuRateStaticData {
  cellCount:number;
  horizontalCellCount:number;
  nz:number;
  /** Three vec4<i32> per cell, packing six (faceId,sign) pairs. faceId=-1 is unused. */
  cellFacePairs:Int32Array;
  /** vec4 per cell: vector coefficient multiplying rho-rho_ref for gravity force. */
  gravityCoefficient:Float32Array;
  /** Four vec4 per horizontal cell: bottom dx+p_ref, bottom area, top dx+p_ref, top area. */
  boundaryStatic:Float32Array;
  /** vec4 per layer: rho_ref,p_ref,Phi_ref,unused. */
  referenceLayer:Float32Array;
}

export function buildCoreV2GpuRateStaticData(
  geometry:SphericalShellGeometry,
  stencil:LinearReconstructionStencil,
  reference:HydrostaticReference1D,
  faces:CoreV2GpuInternalFaceStaticData,
  planet:PlanetConfig=EARTH,
):CoreV2GpuRateStaticData{
  if(stencil.geometry!==geometry)throw new Error('Core v2 GPU rate received foreign stencil');
  const hc=geometry.horizontal.cellCount, nz=geometry.nz, cellCount=hc*nz;
  if(faces.cellCount!==cellCount||faces.nz!==nz)throw new Error('Core v2 GPU rate face geometry mismatch');
  const pairs=new Int32Array(cellCount*12);pairs.fill(-1);
  const used=new Uint8Array(cellCount);
  for(let f=0;f<faces.faceCount;f++){
    const l=faces.faceCells[f*2]!,r=faces.faceCells[f*2+1]!;
    for(const [q,sign] of [[l,-1],[r,1]] as const){
      const slot=used[q]!;if(slot>=6)throw new Error(`Core v2 GPU cell ${q} exceeds six internal faces`);
      pairs[q*12+slot*2]=f;pairs[q*12+slot*2+1]=sign;used[q]=slot+1;
    }
  }
  const gravityCoefficient=new Float32Array(cellCount*4);
  const referenceLayer=new Float32Array(nz*4);
  for(let k=0;k<nz;k++){
    referenceLayer[k*4]=reference.cellDensity[k]!;referenceLayer[k*4+1]=reference.cellPressure[k]!;referenceLayer[k*4+2]=reference.cellGeopotential[k]!;
  }
  for(let c=0;c<hc;c++)for(let k=0;k<nz;k++){
    const q=shellCellIndex(c,k,nz),r0=geometry.radiusInterface[k]!,r1=geometry.radiusInterface[k+1]!;
    const scale=-planet.gravity*(r1**3-r0**3)/3, base=q*4;
    gravityCoefficient[base]=scale*geometry.cellVectorAreaUnit[c*3]!;
    gravityCoefficient[base+1]=scale*geometry.cellVectorAreaUnit[c*3+1]!;
    gravityCoefficient[base+2]=scale*geometry.cellVectorAreaUnit[c*3+2]!;
  }
  const boundaryStatic=new Float32Array(hc*16);
  for(let c=0;c<hc;c++){
    const qb=shellCellIndex(c,0,nz),qt=shellCellIndex(c,nz-1,nz);
    const xb=[stencil.cellCentroid[qb*3]!,stencil.cellCentroid[qb*3+1]!,stencil.cellCentroid[qb*3+2]!] as const;
    const xt=[stencil.cellCentroid[qt*3]!,stencil.cellCentroid[qt*3+1]!,stencil.cellCentroid[qt*3+2]!] as const;
    const fb=radialFaceCentroid(geometry,c,0),ft=radialFaceCentroid(geometry,c,nz);
    const ab=radialFaceVectorArea(geometry,c,0),at=radialFaceVectorArea(geometry,c,nz);const i=c*16;
    boundaryStatic[i]=fb[0]-xb[0];boundaryStatic[i+1]=fb[1]-xb[1];boundaryStatic[i+2]=fb[2]-xb[2];boundaryStatic[i+3]=reference.radialFacePressure[0]!;
    boundaryStatic[i+4]=-ab[0];boundaryStatic[i+5]=-ab[1];boundaryStatic[i+6]=-ab[2];
    boundaryStatic[i+8]=ft[0]-xt[0];boundaryStatic[i+9]=ft[1]-xt[1];boundaryStatic[i+10]=ft[2]-xt[2];boundaryStatic[i+11]=reference.radialFacePressure[nz]!;
    boundaryStatic[i+12]=at[0];boundaryStatic[i+13]=at[1];boundaryStatic[i+14]=at[2];
  }
  return {cellCount,horizontalCellCount:hc,nz,cellFacePairs:pairs,gravityCoefficient,boundaryStatic,referenceLayer};
}

const SHADER=/* wgsl */`
struct Params{cellCount:u32,nz:u32,_a:u32,_b:u32,gamma:f32,_pad0:f32,_pad1:f32,_pad2:f32};
@group(0)@binding(0)var<uniform>P:Params;
@group(0)@binding(1)var<storage,read>state:array<vec4<f32>>;
@group(0)@binding(2)var<storage,read>gradient:array<vec4<f32>>;
@group(0)@binding(3)var<storage,read>faceFlux:array<vec4<f32>>;
@group(0)@binding(4)var<storage,read>pairs:array<vec4<i32>>;
@group(0)@binding(5)var<storage,read>gravityCoeff:array<vec4<f32>>;
@group(0)@binding(6)var<storage,read>boundaryStatic:array<vec4<f32>>;
@group(0)@binding(7)var<storage,read>refLayer:array<vec4<f32>>;
@group(0)@binding(8)var<storage,read_write>rate:array<vec4<f32>>;
fn pprime(q:u32)->f32{let a=state[2u*q];let b=state[2u*q+1u];let rho=a.x;let k=q-P.nz*(q/P.nz);let refState=refLayer[k];let ke=.5*(a.y*a.y+a.z*a.z+a.w*a.w)/rho;let eref=refState.y/(P.gamma-1.0)+refState.x*refState.z;return(P.gamma-1.0)*((b.x-eref)-ke-(rho-refState.x)*refState.z);}
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){
 let q=gid.x;if(q>=P.cellCount){return;}var a=vec4<f32>(0.0);var e=0.0;
 for(var s:u32=0u;s<6u;s++){let v=pairs[q*3u+s/2u];let fid=select(v.x,v.z,(s&1u)==1u);let sign=select(v.y,v.w,(s&1u)==1u);if(fid>=0){let f=faceFlux[2u*u32(fid)];let fe=faceFlux[2u*u32(fid)+1u].x;let sf=f32(sign);a+=sf*f;e+=sf*fe;}}
 let k=q-P.nz*(q/P.nz);let c=q/P.nz;let rhoPrime=state[2u*q].x-refLayer[k].x;a.yzw+=rhoPrime*gravityCoeff[q].xyz;
 if(k==0u){let d=boundaryStatic[c*4u];let area=boundaryStatic[c*4u+1u];let pp=pprime(q)+dot(gradient[q*5u+4u].xyz,d.xyz);a.yzw-=pp*area.xyz;}
 if(k+1u==P.nz){let d=boundaryStatic[c*4u+2u];let area=boundaryStatic[c*4u+3u];let pp=pprime(q)+dot(gradient[q*5u+4u].xyz,d.xyz);a.yzw-=pp*area.xyz;}
 rate[2u*q]=a;rate[2u*q+1u]=vec4<f32>(e,0.0,0.0,0.0);
}
`;

export class CoreV2GpuRateGather{
 private readonly device:GPUAny;
 readonly staticData:CoreV2GpuRateStaticData;
 private readonly pipeline:GPUAny;
 private readonly pipelineValidation:Promise<any>;
 private readonly compilationInfo:Promise<any>;
 private readonly buffers:GPUAny[]=[];
 constructor(device:GPUAny,d:CoreV2GpuRateStaticData){
  this.device=device;this.staticData=d;
  const raw=new ArrayBuffer(32),u=new Uint32Array(raw),f=new Float32Array(raw);u[0]=d.cellCount;u[1]=d.nz;f[4]=DRY_AIR.gamma;
  const arrays:[ArrayBufferView,string][]=[[new Uint8Array(raw),'params'],[d.cellFacePairs,'pairs'],[d.gravityCoefficient,'gravity'],[d.boundaryStatic,'boundary'],[d.referenceLayer,'reference']];
  for(let i=0;i<arrays.length;i++)this.buffers.push(upload(device,arrays[i]![0],i===0?USAGE.UNIFORM:USAGE.STORAGE,`core-v2-rate-${arrays[i]![1]}`));
  this.device.pushErrorScope('validation');
  const module=device.createShaderModule({label:'core-v2-rate-gather-module',code:SHADER});
  this.compilationInfo=module.getCompilationInfo();
  this.pipeline=device.createComputePipeline({label:'core-v2-rate-gather-pipeline',layout:'auto',compute:{module,entryPoint:'main'}});
  this.pipelineValidation=this.device.popErrorScope();
 }
 async compute(stateData:Float32Array,gradients:Float32Array,fluxes:Float32Array):Promise<Float32Array>{
  const d=this.staticData;
  if(stateData.length!==d.cellCount*8||gradients.length!==d.cellCount*20)throw new Error('Core v2 GPU rate input size mismatch');
  const setupError=await this.pipelineValidation;
  if(setupError){
   const info=await this.compilationInfo;
   const messages=Array.from(info.messages??[]).map((message:any)=>`${message.type??'message'} ${message.lineNum??'?'}:${message.linePos??'?'} ${message.message??String(message)}`).join(' | ');
   throw new Error(`Core v2 GPU rate-gather pipeline error: ${setupError.message}${messages?`; shader: ${messages}`:''}`);
  }
  const state=upload(this.device,stateData,USAGE.STORAGE,'rate-state'),grad=upload(this.device,gradients,USAGE.STORAGE,'rate-grad'),flux=upload(this.device,fluxes,USAGE.STORAGE,'rate-flux'),out=empty(this.device,d.cellCount*8*4,USAGE.STORAGE|USAGE.COPY_SRC,'rate-out');
  this.device.pushErrorScope('validation');
  const b=this.buffers;const bg=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:b[0]}},{binding:1,resource:{buffer:state}},{binding:2,resource:{buffer:grad}},{binding:3,resource:{buffer:flux}},{binding:4,resource:{buffer:b[1]}},{binding:5,resource:{buffer:b[2]}},{binding:6,resource:{buffer:b[3]}},{binding:7,resource:{buffer:b[4]}},{binding:8,resource:{buffer:out}}]});
  const enc=this.device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(this.pipeline);pass.setBindGroup(0,bg);pass.dispatchWorkgroups(Math.ceil(d.cellCount/128));pass.end();this.device.queue.submit([enc.finish()]);await this.device.queue.onSubmittedWorkDone();
  const validationError=await this.device.popErrorScope();
  if(validationError){state.destroy();grad.destroy();flux.destroy();out.destroy();throw new Error(`Core v2 GPU rate-gather validation error: ${validationError.message}`);}
  const result=await readback(this.device,out,d.cellCount*8);state.destroy();grad.destroy();flux.destroy();out.destroy();return result;
 }
 destroy():void{for(const b of this.buffers)b.destroy();}
}
