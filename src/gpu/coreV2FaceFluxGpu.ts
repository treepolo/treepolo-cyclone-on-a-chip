import { DRY_AIR, EARTH, type PlanetConfig } from '../core/constants.js';
import type { HydrostaticReference1D } from '../corev2/hydrostaticReference.js';
import {
  radialFaceCentroid,
  sideFaceCentroid,
  type LinearReconstructionStencil,
} from '../corev2/reconstruction.js';
import {
  radialFaceVectorArea,
  shellCellIndex,
  sideFaceVectorArea,
  type SphericalShellGeometry,
} from '../corev2/sphericalShellGeometry.js';

export const CORE_V2_GPU_FACE_FLUX_FLOATS = 8;

type GPUAny = any;
const USAGE = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 } as const;
const MAP_READ = 1;

function aligned(bytes: number): number { return Math.max(4, Math.ceil(bytes / 4) * 4); }
function upload(device: GPUAny, data: ArrayBufferView, usage: number, label: string): GPUAny {
  const buffer = device.createBuffer({ label, size: aligned(data.byteLength), usage: usage | USAGE.COPY_DST, mappedAtCreation: true });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
}
function empty(device: GPUAny, bytes: number, usage: number, label: string): GPUAny {
  return device.createBuffer({ label, size: aligned(bytes), usage });
}
async function readback(device: GPUAny, source: GPUAny, floats: number): Promise<Float32Array> {
  const bytes = floats * 4;
  const staging = empty(device, bytes, USAGE.MAP_READ | USAGE.COPY_DST, 'core-v2-face-readback');
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(MAP_READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return result;
}

export interface CoreV2GpuInternalFaceStaticData {
  faceCount: number;
  cellCount: number;
  nz: number;
  /** vec2<u32> per face, padded as two u32. */
  faceCells: Uint32Array;
  /** vec4: displacement from left cell centroid to face centroid, p_ref. */
  leftDxReferencePressure: Float32Array;
  /** vec4: displacement from right cell centroid to face centroid, Phi_face. */
  rightDxGeopotential: Float32Array;
  /** vec4: exact finite-volume vector area xyz, unused. */
  vectorArea: Float32Array;
  /** vec4 per layer: rho_ref, p_ref, Phi_ref, unused. */
  referenceLayer: Float32Array;
}

function sideFaceRadialMean(geometry: SphericalShellGeometry, k: number): number {
  const r0 = geometry.radiusInterface[k]!;
  const r1 = geometry.radiusInterface[k + 1]!;
  return (2 / 3) * (r1 ** 3 - r0 ** 3) / (r1 * r1 - r0 * r0);
}

function geopotentialAtRadius(
  geometry: SphericalShellGeometry,
  radius: number,
  planet: PlanetConfig,
): number {
  return planet.gravity * (radius - geometry.radiusInterface[0]!);
}

export function buildCoreV2GpuInternalFaceStaticData(
  geometry: SphericalShellGeometry,
  stencil: LinearReconstructionStencil,
  reference: HydrostaticReference1D,
  planet: PlanetConfig = EARTH,
): CoreV2GpuInternalFaceStaticData {
  if (stencil.geometry !== geometry) throw new Error('Core v2 GPU face data received foreign stencil');
  const horizontal = geometry.horizontal;
  const cellCount = horizontal.cellCount * geometry.nz;
  const faceCount = horizontal.edgeCount * geometry.nz + horizontal.cellCount * (geometry.nz - 1);
  const faceCells = new Uint32Array(faceCount * 2);
  const leftDxReferencePressure = new Float32Array(faceCount * 4);
  const rightDxGeopotential = new Float32Array(faceCount * 4);
  const vectorArea = new Float32Array(faceCount * 4);
  const referenceLayer = new Float32Array(geometry.nz * 4);
  for (let k = 0; k < geometry.nz; k++) {
    referenceLayer[k * 4] = reference.cellDensity[k]!;
    referenceLayer[k * 4 + 1] = reference.cellPressure[k]!;
    referenceLayer[k * 4 + 2] = reference.cellGeopotential[k]!;
  }

  const centroid = (q: number): readonly [number, number, number] => [
    stencil.cellCentroid[q * 3]!,
    stencil.cellCentroid[q * 3 + 1]!,
    stencil.cellCentroid[q * 3 + 2]!,
  ];
  const writeFace = (
    f: number,
    left: number,
    right: number,
    position: readonly [number, number, number],
    area: readonly [number, number, number],
    pRef: number,
    phi: number,
  ): void => {
    faceCells[f * 2] = left;
    faceCells[f * 2 + 1] = right;
    const xl = centroid(left);
    const xr = centroid(right);
    const l = f * 4;
    leftDxReferencePressure[l] = position[0] - xl[0];
    leftDxReferencePressure[l + 1] = position[1] - xl[1];
    leftDxReferencePressure[l + 2] = position[2] - xl[2];
    leftDxReferencePressure[l + 3] = pRef;
    rightDxGeopotential[l] = position[0] - xr[0];
    rightDxGeopotential[l + 1] = position[1] - xr[1];
    rightDxGeopotential[l + 2] = position[2] - xr[2];
    rightDxGeopotential[l + 3] = phi;
    vectorArea[l] = area[0];
    vectorArea[l + 1] = area[1];
    vectorArea[l + 2] = area[2];
  };

  let f = 0;
  for (let e = 0; e < horizontal.edgeCount; e++) {
    const edge = horizontal.edges[e]!;
    for (let k = 0; k < geometry.nz; k++) {
      const left = shellCellIndex(edge.leftCell, k, geometry.nz);
      const right = shellCellIndex(edge.rightCell, k, geometry.nz);
      writeFace(
        f++,
        left,
        right,
        sideFaceCentroid(geometry, e, k),
        sideFaceVectorArea(geometry, e, k),
        reference.sideFacePressure[k]!,
        geopotentialAtRadius(geometry, sideFaceRadialMean(geometry, k), planet),
      );
    }
  }
  for (let c = 0; c < horizontal.cellCount; c++) {
    for (let ki = 1; ki < geometry.nz; ki++) {
      writeFace(
        f++,
        shellCellIndex(c, ki - 1, geometry.nz),
        shellCellIndex(c, ki, geometry.nz),
        radialFaceCentroid(geometry, c, ki),
        radialFaceVectorArea(geometry, c, ki),
        reference.radialFacePressure[ki]!,
        geopotentialAtRadius(geometry, geometry.radiusInterface[ki]!, planet),
      );
    }
  }
  if (f !== faceCount) throw new Error(`Core v2 GPU internal face count mismatch ${f} != ${faceCount}`);
  return { faceCount, cellCount, nz: geometry.nz, faceCells, leftDxReferencePressure, rightDxGeopotential, vectorArea, referenceLayer };
}

const SHADER = /* wgsl */`
struct Params{faceCount:u32,nz:u32,_p0:u32,_p1:u32,gamma:f32,_pad:vec3<f32>};
@group(0) @binding(0) var<uniform> P:Params;
@group(0) @binding(1) var<storage,read> state:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> gradient:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> faceCells:array<vec2<u32>>;
@group(0) @binding(4) var<storage,read> leftDxRef:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read> rightDxPhi:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read> vectorArea:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read> refLayer:array<vec4<f32>>;
@group(0) @binding(8) var<storage,read_write> outFlux:array<vec4<f32>>;

fn beta_plus(m:f32)->f32{if(abs(m)<1.0){return .25*(2.0-m)*(m+1.0)*(m+1.0);}return select(0.0,1.0,m>=0.0);}
fn beta_minus(m:f32)->f32{if(abs(m)<1.0){return .25*(2.0+m)*(m-1.0)*(m-1.0);}return select(1.0,0.0,m>=0.0);}

fn base_value(q:u32,v:u32)->f32{
  let a=state[2u*q];let b=state[2u*q+1u];let rho=a.x;
  if(v==0u){return rho;} if(v==1u){return a.y/rho;} if(v==2u){return a.z/rho;} if(v==3u){return a.w/rho;}
  let k=q-P.nz*(q/P.nz);let ref=refLayer[k];let kinetic=.5*(a.y*a.y+a.z*a.z+a.w*a.w)/rho;
  let referenceEnergy=ref.y/(P.gamma-1.0)+ref.x*ref.z;
  return (P.gamma-1.0)*((b.x-referenceEnergy)-kinetic-(rho-ref.x)*ref.z);
}
fn reconstruct(q:u32,dx:vec3<f32>,pRef:f32)->array<vec4<f32>,2>{
  let rho=base_value(q,0u)+dot(gradient[q*5u].xyz,dx);
  let ux=base_value(q,1u)+dot(gradient[q*5u+1u].xyz,dx);
  let uy=base_value(q,2u)+dot(gradient[q*5u+2u].xyz,dx);
  let uz=base_value(q,3u)+dot(gradient[q*5u+3u].xyz,dx);
  let pp=base_value(q,4u)+dot(gradient[q*5u+4u].xyz,dx);
  var out:array<vec4<f32>,2>;
  out[0]=vec4<f32>(rho,ux,uy,uz);out[1]=vec4<f32>(pRef+pp,0.0,0.0,0.0);return out;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let f=gid.x;if(f>=P.faceCount){return;}
  let cells=faceCells[f];let ld=leftDxRef[f];let rd=rightDxPhi[f];let av=vectorArea[f];
  let pRef=ld.w;let phi=rd.w;let L=reconstruct(cells.x,ld.xyz,pRef);let R=reconstruct(cells.y,rd.xyz,pRef);
  let area=length(av.xyz);let n=av.xyz/area;
  let rhoL=L[0].x;let rhoR=R[0].x;let uL=L[0].yzw;let uR=R[0].yzw;let pL=L[1].x;let pR=R[1].x;
  let aL=sqrt(P.gamma*pL/rhoL);let aR=sqrt(P.gamma*pR/rhoR);let a=.5*(aL+aR);
  let unL=dot(uL,n);let unR=dot(uR,n);let s2L=dot(uL,uL);let s2R=dot(uR,uR);let mL=unL/a;let mR=unR/a;
  let vr=sqrt(.5*(s2L+s2R));let mh=min(1.0,vr/a);let chi=(1.0-mh)*(1.0-mh);
  let ex=-max(min(mL,0.0),-1.0)*min(max(mR,0.0),1.0);let wa=(rhoL*abs(unL)+rhoR*abs(unR))/(rhoL+rhoR);
  let al=(1.0-ex)*wa+ex*abs(unL);let ar=(1.0-ex)*wa+ex*abs(unR);
  let ppL=pL-pRef;let ppR=pR-pRef;let mass=.5*(rhoL*(unL+al)+rhoR*(unR-ar)-(chi/a)*(ppR-ppL));
  let bp=beta_plus(mL);let bm=beta_minus(mR);let pressure=.5*(ppL+ppR)+.5*(bp-bm)*(ppL-ppR)+vr*(bp+bm-1.0)*a*.5*(rhoL+rhoR);
  let leftUp=mass>=0.0;let uu=select(uR,uL,leftUp);let pu=select(pR,pL,leftUp);let ru=select(rhoR,rhoL,leftUp);let s2=select(s2R,s2L,leftUp);
  let h=P.gamma/(P.gamma-1.0)*pu/ru+.5*s2+phi;let mf=mass*uu+pressure*n;
  outFlux[2u*f]=vec4<f32>(area*mass,area*mf.x,area*mf.y,area*mf.z);outFlux[2u*f+1u]=vec4<f32>(area*mass*h,0.0,0.0,0.0);
}
`;

export class CoreV2GpuInternalFaceFlux {
  private readonly device: GPUAny;
  readonly staticData: CoreV2GpuInternalFaceStaticData;
  private readonly pipeline: GPUAny;
  private readonly params: GPUAny;
  private readonly faceCells: GPUAny;
  private readonly leftDxRef: GPUAny;
  private readonly rightDxPhi: GPUAny;
  private readonly vectorArea: GPUAny;
  private readonly referenceLayer: GPUAny;

  constructor(device: GPUAny, staticData: CoreV2GpuInternalFaceStaticData) {
    this.device=device;this.staticData=staticData;
    const raw=new ArrayBuffer(32);const u=new Uint32Array(raw);const f=new Float32Array(raw);u[0]=staticData.faceCount;u[1]=staticData.nz;f[4]=DRY_AIR.gamma;
    this.params=upload(device,new Uint8Array(raw),USAGE.UNIFORM,'core-v2-face-params');
    this.faceCells=upload(device,staticData.faceCells,USAGE.STORAGE,'core-v2-face-cells');
    this.leftDxRef=upload(device,staticData.leftDxReferencePressure,USAGE.STORAGE,'core-v2-face-leftdx');
    this.rightDxPhi=upload(device,staticData.rightDxGeopotential,USAGE.STORAGE,'core-v2-face-rightdx');
    this.vectorArea=upload(device,staticData.vectorArea,USAGE.STORAGE,'core-v2-face-area');
    this.referenceLayer=upload(device,staticData.referenceLayer,USAGE.STORAGE,'core-v2-face-reference');
    const module=device.createShaderModule({label:'core-v2-face-flux-module',code:SHADER});
    this.pipeline=device.createComputePipeline({label:'core-v2-face-flux-pipeline',layout:'auto',compute:{module,entryPoint:'main'}});
  }

  async compute(packedState:Float32Array, gradients:Float32Array):Promise<Float32Array>{
    if(packedState.length!==this.staticData.cellCount*8)throw new Error('Core v2 GPU face state size mismatch');
    if(gradients.length!==this.staticData.cellCount*20)throw new Error('Core v2 GPU face gradient size mismatch');
    const state=upload(this.device,packedState,USAGE.STORAGE,'core-v2-face-state');
    const grad=upload(this.device,gradients,USAGE.STORAGE,'core-v2-face-gradient');
    const output=empty(this.device,this.staticData.faceCount*8*4,USAGE.STORAGE|USAGE.COPY_SRC,'core-v2-face-output');
    const bg=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:this.params}},{binding:1,resource:{buffer:state}},{binding:2,resource:{buffer:grad}},
      {binding:3,resource:{buffer:this.faceCells}},{binding:4,resource:{buffer:this.leftDxRef}},{binding:5,resource:{buffer:this.rightDxPhi}},
      {binding:6,resource:{buffer:this.vectorArea}},{binding:7,resource:{buffer:this.referenceLayer}},{binding:8,resource:{buffer:output}},
    ]});
    const encoder=this.device.createCommandEncoder();const pass=encoder.beginComputePass();pass.setPipeline(this.pipeline);pass.setBindGroup(0,bg);pass.dispatchWorkgroups(Math.ceil(this.staticData.faceCount/128));pass.end();this.device.queue.submit([encoder.finish()]);await this.device.queue.onSubmittedWorkDone();
    const result=await readback(this.device,output,this.staticData.faceCount*8);state.destroy();grad.destroy();output.destroy();return result;
  }

  destroy():void{this.params.destroy();this.faceCells.destroy();this.leftDxRef.destroy();this.rightDxPhi.destroy();this.vectorArea.destroy();this.referenceLayer.destroy();}
}
