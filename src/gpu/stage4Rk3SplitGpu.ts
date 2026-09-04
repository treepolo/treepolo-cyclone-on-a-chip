import { DRY_AIR, EARTH } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { buildRk3SplitSchedule } from '../solver/rk3SplitSchedule.js';
import { STAGE4_HEVI_OFFCENTERING } from '../solver/stage4Config.js';
import { Stage4Rk3SplitOptions } from '../solver/stage4Rk3SplitCpu.js';
import { DryState } from '../solver/state.js';
import { GpuAcousticDivergenceDamping } from './acousticDivergenceDampingGpu.js';
import { GpuRotatingDryCore } from './rotatingDryCoreGpu.js';
import { GpuStage4SlowTendencyReference } from './stage4SlowTendenciesGpu.js';

type GPUAny=any;
function makeEmpty(device:GPUAny,size:number,usage:number,label:string):GPUAny{return device.createBuffer({label,size:Math.max(4,Math.ceil(size/4)*4),usage});}

const COMMON=/* wgsl */`
struct Params{
  nz:u32,edgeCount:u32,cellCount:u32,flags:u32,
  dt:f32,radius:f32,gravity:f32,rd:f32,
  gamma:f32,pRef:f32,offCenter:f32,_pad0:f32,
};
@group(0)@binding(0)var<uniform>P:Params;
fn pressure(x:f32)->f32{return P.pRef*pow(P.rd*x/P.pRef,P.gamma);}
`;

const PACK_BASE=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>rho:array<f32>;
@group(0)@binding(2)var<storage,read>x:array<f32>;
@group(0)@binding(3)var<storage,read_write>baseCell:array<vec2<f32>>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.cellCount*P.nz){return;}baseCell[q]=vec2<f32>(rho[q],x[q]);}
`;

const PREP_CELL=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>rho:array<f32>;
@group(0)@binding(2)var<storage,read>x:array<f32>;
@group(0)@binding(3)var<storage,read>baseCell:array<vec2<f32>>;
@group(0)@binding(4)var<storage,read>slowScalar:array<vec2<f32>>;
@group(0)@binding(5)var<storage,read_write>predCell:array<vec2<f32>>;
@group(0)@binding(6)var<storage,read_write>acousticCell:array<vec2<f32>>;
@group(0)@binding(7)var<storage,read_write>frozenScalar:array<vec2<f32>>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.cellCount*P.nz){return;}predCell[q]=vec2<f32>(rho[q],x[q]);acousticCell[q]=baseCell[q];frozenScalar[q]=slowScalar[q];}
`;

const ADD_HREF=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>cellEdges:array<vec4<u32>>;
@group(0)@binding(2)var<storage,read>cellSigns:array<vec4<i32>>;
@group(0)@binding(3)var<storage,read>edgeMetric:array<vec2<f32>>;
@group(0)@binding(4)var<storage,read>cellArea:array<f32>;
@group(0)@binding(5)var<storage,read>layerRef:array<f32>;
@group(0)@binding(6)var<storage,read>u:array<f32>;
@group(0)@binding(7)var<storage,read_write>frozenScalar:array<vec2<f32>>;
fn r0(k:u32)->f32{return layerRef[k*5u+3u];}fn x0(k:u32)->f32{return layerRef[k*5u+4u];}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){
  let q=gid.x;if(q>=P.cellCount*P.nz){return;}let c=q/P.nz;let k=q-c*P.nz;let ee=cellEdges[c];let ss=cellSigns[c];var sum=0.0;
  for(var s:u32=0u;s<4u;s++){let e=ee[s];sum+=f32(ss[s])*u[e*P.nz+k]*edgeMetric[e].x;}
  frozenScalar[q]=frozenScalar[q]-vec2<f32>(r0(k),x0(k))*sum/cellArea[c];
}
`;

const ADD_VREF=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>layerRef:array<f32>;
@group(0)@binding(2)var<storage,read>interfaceRef:array<f32>;
@group(0)@binding(3)var<storage,read>w:array<f32>;
@group(0)@binding(4)var<storage,read_write>frozenScalar:array<vec2<f32>>;
fn dz(k:u32)->f32{return layerRef[k*5u+1u];}fn irho(i:u32)->f32{return interfaceRef[i*2u];}fn ix(i:u32)->f32{return interfaceRef[i*2u+1u];}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){
  let q=gid.x;if(q>=P.cellCount*P.nz){return;}let c=q/P.nz;let k=q-c*P.nz;let b=c*(P.nz+1u)+k;let wb=w[b];let wt=w[b+1u];
  frozenScalar[q]=frozenScalar[q]-(vec2<f32>(irho(k+1u),ix(k+1u))*wt-vec2<f32>(irho(k),ix(k))*wb)/dz(k);
}
`;

const PREP_U=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>edgeCells:array<vec2<u32>>;
@group(0)@binding(2)var<storage,read>edgeMetric:array<vec2<f32>>;
@group(0)@binding(3)var<storage,read>predCell:array<vec2<f32>>;
@group(0)@binding(4)var<storage,read>slowU:array<f32>;
@group(0)@binding(5)var<storage,read_write>frozenU:array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){
  let q=gid.x;if(q>=P.edgeCount*P.nz){return;}let e=q/P.nz;let k=q-e*P.nz;let cc=edgeCells[e];let l=cc.x*P.nz+k;let r=cc.y*P.nz+k;let pl=pressure(predCell[l].y);let pr=pressure(predCell[r].y);let ravg=max(.5*(predCell[l].x+predCell[r].x),1e-12);let dist=max(edgeMetric[e].y,1.0);
  frozenU[q]=slowU[q]-(pr-pl)/(ravg*dist);
}
`;

const PREP_IFACE=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>layerRef:array<f32>;
@group(0)@binding(2)var<storage,read>predCell:array<vec2<f32>>;
@group(0)@binding(3)var<storage,read>predW:array<f32>;
@group(0)@binding(4)var<storage,read>baseW:array<f32>;
@group(0)@binding(5)var<storage,read>slowW:array<f32>;
@group(0)@binding(6)var<storage,read>rayleigh:array<f32>;
@group(0)@binding(7)var<storage,read_write>iface:array<vec4<f32>>;
fn z(k:u32)->f32{return layerRef[k*5u];}fn p0(k:u32)->f32{return layerRef[k*5u+2u];}fn r0(k:u32)->f32{return layerRef[k*5u+3u];}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){
  let q=gid.x;let count=P.cellCount*(P.nz+1u);if(q>=count){return;}let c=q/(P.nz+1u);let i=q-c*(P.nz+1u);var acc=0.0;
  if(i>0u&&i<P.nz){let l=i-1u;let u=i;let cl=predCell[c*P.nz+l];let cu=predCell[c*P.nz+u];let rr=.5*(cl.x+cu.x);let rr0=.5*(r0(l)+r0(u));let den=max(.5*(rr0+rr),1e-12);let dzc=z(u)-z(l);let dpPrime=(pressure(cu.y)-p0(u))-(pressure(cl.y)-p0(l));acc=-dpPrime/(den*dzc)-P.gravity*(rr-rr0)/max(rr,1e-12);}
  let boundary=i==0u||i==P.nz;let rate=select(rayleigh[i],0.0,(P.flags&1u)==0u||boundary);let fw=select(slowW[q]+acc,0.0,boundary);iface[q]=vec4<f32>(predW[q],select(baseW[q],0.0,boundary),fw,rate);
}
`;

const HVEL=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>edgeCells:array<vec2<u32>>;
@group(0)@binding(2)var<storage,read>edgeMetric:array<vec2<f32>>;
@group(0)@binding(3)var<storage,read>predCell:array<vec2<f32>>;
@group(0)@binding(4)var<storage,read>acousticCell:array<vec2<f32>>;
@group(0)@binding(5)var<storage,read>frozenU:array<f32>;
@group(0)@binding(6)var<storage,read_write>acousticU:array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){
  let q=gid.x;if(q>=P.edgeCount*P.nz){return;}let e=q/P.nz;let k=q-e*P.nz;let cc=edgeCells[e];let l=cc.x*P.nz+k;let r=cc.y*P.nz+k;let pl=pressure(predCell[l].y);let pr=pressure(predCell[r].y);let dpl=P.gamma*pl/max(predCell[l].y,1e-12);let dpr=P.gamma*pr/max(predCell[r].y,1e-12);let dl=acousticCell[l]-predCell[l];let dr=acousticCell[r]-predCell[r];let ravg=max(.5*(predCell[l].x+predCell[r].x),1e-12);let dRavg=.5*(dl.x+dr.x);let dist=max(edgeMetric[e].y,1.0);let dDp=dpr*dr.y-dpl*dl.y;let lin=-dDp/(ravg*dist)+(pr-pl)*dRavg/(ravg*ravg*dist);acousticU[q]=acousticU[q]+P.dt*(frozenU[q]+lin);
}
`;

const HREF_FLUX=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>edgeMetric:array<vec2<f32>>;
@group(0)@binding(2)var<storage,read>layerRef:array<f32>;
@group(0)@binding(3)var<storage,read>predU:array<f32>;
@group(0)@binding(4)var<storage,read>acousticU:array<f32>;
@group(0)@binding(5)var<storage,read_write>flux:array<vec2<f32>>;
fn dz(k:u32)->f32{return layerRef[k*5u+1u];}fn r0(k:u32)->f32{return layerRef[k*5u+3u];}fn x0(k:u32)->f32{return layerRef[k*5u+4u];}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.edgeCount*P.nz){return;}let e=q/P.nz;let k=q-e*P.nz;let du=acousticU[q]-predU[q];let A=edgeMetric[e].x*dz(k);flux[q]=vec2<f32>(r0(k),x0(k))*du*A;}
`;

const HREF_DIV=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>cellArea:array<f32>;
@group(0)@binding(2)var<storage,read>layerRef:array<f32>;
@group(0)@binding(3)var<storage,read>cellEdges:array<vec4<u32>>;
@group(0)@binding(4)var<storage,read>cellSigns:array<vec4<i32>>;
@group(0)@binding(5)var<storage,read>flux:array<vec2<f32>>;
@group(0)@binding(6)var<storage,read_write>corr:array<vec2<f32>>;
fn dz(k:u32)->f32{return layerRef[k*5u+1u];}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.cellCount*P.nz){return;}let c=q/P.nz;let k=q-c*P.nz;let ee=cellEdges[c];let ss=cellSigns[c];var h=vec2<f32>(0.0);for(var s:u32=0u;s<4u;s++){h+=f32(ss[s])*flux[ee[s]*P.nz+k];}corr[q]=-h/(cellArea[c]*dz(k));}
`;

const VERTICAL=COMMON+/* wgsl */`
const MAX_NZ:u32=128u;
@group(0)@binding(1)var<storage,read>layerRef:array<f32>;
@group(0)@binding(2)var<storage,read>interfaceRef:array<f32>;
@group(0)@binding(3)var<storage,read>predCell:array<vec2<f32>>;
@group(0)@binding(4)var<storage,read_write>acousticCell:array<vec2<f32>>;
@group(0)@binding(5)var<storage,read>frozenScalar:array<vec2<f32>>;
@group(0)@binding(6)var<storage,read>hCorr:array<vec2<f32>>;
@group(0)@binding(7)var<storage,read_write>iface:array<vec4<f32>>;
@group(0)@binding(8)var<storage,read_write>refFlux:array<vec2<f32>>;
fn lz(k:u32)->f32{return layerRef[k*5u];}fn ldz(k:u32)->f32{return layerRef[k*5u+1u];}fn lp(k:u32)->f32{return layerRef[k*5u+2u];}fn lrho(k:u32)->f32{return layerRef[k*5u+3u];}fn irho(i:u32)->f32{return interfaceRef[i*2u];}fn ix(i:u32)->f32{return interfaceRef[i*2u+1u];}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){
  let c=gid.x;if(c>=P.cellCount||P.nz>MAX_NZ||P.nz<2u){return;}let nz=P.nz;let n=nz-1u;let theta=.5*(1.0+clamp(P.offCenter,0.0,.999));let oldWeight=1.0-theta;
  var drOld:array<f32,128>;var dxOld:array<f32,128>;var dwOld:array<f32,129>;var pPred:array<f32,128>;var dpdX:array<f32,128>;var drBase:array<f32,128>;var dxBase:array<f32,128>;var lOld:array<f32,129>;var lo:array<f32,127>;var di:array<f32,127>;var up:array<f32,127>;var rhs:array<f32,127>;var cp:array<f32,127>;var dp:array<f32,127>;var sol:array<f32,127>;var dwNew:array<f32,129>;
  let bc=c*nz;let bw=c*(nz+1u);
  for(var k:u32=0u;k<nz;k++){let pc=predCell[bc+k];let ac=acousticCell[bc+k];drOld[k]=ac.x-pc.x;dxOld[k]=ac.y-pc.y;pPred[k]=pressure(pc.y);dpdX[k]=P.gamma*pPred[k]/max(pc.y,1e-12);}
  for(var i:u32=0u;i<=nz;i++){let f=iface[bw+i];dwOld[i]=f.y-f.x;dwNew[i]=0.0;}
  for(var k:u32=0u;k<nz;k++){let rr=frozenScalar[bc+k]+hCorr[bc+k];let divR=(irho(k+1u)*dwOld[k+1u]-irho(k)*dwOld[k])/ldz(k);let divX=(ix(k+1u)*dwOld[k+1u]-ix(k)*dwOld[k])/ldz(k);drBase[k]=drOld[k]+P.dt*rr.x-P.dt*oldWeight*divR;dxBase[k]=dxOld[k]+P.dt*rr.y-P.dt*oldWeight*divX;}
  for(var i:u32=1u;i<nz;i++){let l=i-1u;let u=i;let cl=predCell[bc+l];let cu=predCell[bc+u];let rr=.5*(cl.x+cu.x);let rr0=.5*(lrho(l)+lrho(u));let den=max(.5*(rr0+rr),1e-12);let dzc=lz(u)-lz(l);let dpPrime=(pPred[u]-lp(u))-(pPred[l]-lp(l));let cr=dpPrime*.25/(den*den*dzc)-P.gravity*rr0*.5/max(rr*rr,1e-24);let cxl=dpdX[l]/(den*dzc);let cxu=-dpdX[u]/(den*dzc);lOld[i]=cr*(drOld[l]+drOld[u])+cxl*dxOld[l]+cxu*dxOld[u];}
  for(var ii:u32=0u;ii<n;ii++){let i=ii+1u;let l=i-1u;let u=i;let cl=predCell[bc+l];let cu=predCell[bc+u];let rr=.5*(cl.x+cu.x);let rr0=.5*(lrho(l)+lrho(u));let den=max(.5*(rr0+rr),1e-12);let dzc=lz(u)-lz(l);let dpPrime=(pPred[u]-lp(u))-(pPred[l]-lp(l));let cr=dpPrime*.25/(den*den*dzc)-P.gravity*rr0*.5/max(rr*rr,1e-24);let cxl=dpdX[l]/(den*dzc);let cxu=-dpdX[u]/(den*dzc);let l0=cr*(drBase[l]+drBase[u])+cxl*dxBase[l]+cxu*dxBase[u];let alm=cr*(P.dt*theta*irho(i-1u)/ldz(l))+cxl*(P.dt*theta*ix(i-1u)/ldz(l));let ali=cr*(-P.dt*theta*irho(i)/ldz(l)+P.dt*theta*irho(i)/ldz(u))+cxl*(-P.dt*theta*ix(i)/ldz(l))+cxu*(P.dt*theta*ix(i)/ldz(u));let alp=cr*(-P.dt*theta*irho(i+1u)/ldz(u))+cxu*(-P.dt*theta*ix(i+1u)/ldz(u));let f=iface[bw+i];let rate=max(0.0,f.w);lo[ii]=-P.dt*theta*alm;di[ii]=1.0+P.dt*rate-P.dt*theta*ali;up[ii]=-P.dt*theta*alp;rhs[ii]=dwOld[i]+P.dt*f.z+P.dt*oldWeight*lOld[i]+P.dt*theta*l0-P.dt*rate*f.x;}
  lo[0u]=0.0;up[n-1u]=0.0;var denom=di[0u];cp[0u]=select(up[0u]/denom,0.0,n==1u);dp[0u]=rhs[0u]/denom;for(var j:u32=1u;j<n;j++){denom=di[j]-lo[j]*cp[j-1u];cp[j]=select(up[j]/denom,0.0,j==n-1u);dp[j]=(rhs[j]-lo[j]*dp[j-1u])/denom;}sol[n-1u]=dp[n-1u];var jj:i32=i32(n)-2;loop{if(jj<0){break;}let j=u32(jj);sol[j]=dp[j]-cp[j]*sol[j+1u];jj=jj-1;}for(var i:u32=1u;i<nz;i++){dwNew[i]=sol[i-1u];}
  for(var k:u32=0u;k<nz;k++){let dr=drBase[k]-P.dt*theta*(irho(k+1u)*dwNew[k+1u]-irho(k)*dwNew[k])/ldz(k);let dx=dxBase[k]-P.dt*theta*(ix(k+1u)*dwNew[k+1u]-ix(k)*dwNew[k])/ldz(k);acousticCell[bc+k]=predCell[bc+k]+vec2<f32>(dr,dx);}
  for(var i:u32=0u;i<=nz;i++){var f=iface[bw+i];let nw=select(f.x+dwNew[i],0.0,i==0u||i==nz);let weighted=f.x+oldWeight*dwOld[i]+theta*dwNew[i];f.y=nw;iface[bw+i]=f;refFlux[bw+i]=vec2<f32>(irho(i)*weighted,ix(i)*weighted);}
}
`;

const UNPACK_CELL=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>acousticCell:array<vec2<f32>>;
@group(0)@binding(2)var<storage,read_write>rho:array<f32>;
@group(0)@binding(3)var<storage,read_write>x:array<f32>;
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.cellCount*P.nz){return;}let a=acousticCell[q];rho[q]=a.x;x[q]=a.y;}
`;
const UNPACK_W=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>iface:array<vec4<f32>>;
@group(0)@binding(2)var<storage,read_write>w:array<f32>;
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;let count=P.cellCount*(P.nz+1u);if(q>=count){return;}let i=q%(P.nz+1u);w[q]=select(iface[q].y,0.0,i==0u||i==P.nz);}
`;

/**
 * Full GPU mirror of Stage4Rk3SplitCpu.  The production Stage-4 integrator is
 * intentionally left untouched until this path passes direct hardware
 * agreement.  Core buffers hold the frozen RK predictor during each stage;
 * acoustic buffers restart from the immutable large-step base and are copied
 * back only after the stage is complete.
 */
export class GpuStage4Rk3SplitReference{
  readonly device:GPUAny;readonly buffers:Record<string,GPUAny>={};readonly pipelines:Record<string,GPUAny>={};readonly groups:Record<string,GPUAny>={};
  readonly slow:GpuStage4SlowTendencyReference;readonly divergence:GpuAcousticDivergenceDamping;
  private readonly params:GPUAny;
  private constructor(public readonly core:GpuRotatingDryCore,public readonly acousticRatio=4){
    buildRk3SplitSchedule(acousticRatio);this.device=core.device;const U=(globalThis as any).GPUBufferUsage,storage=U.STORAGE|U.COPY_DST|U.COPY_SRC,uniform=U.UNIFORM|U.COPY_DST,h=core.h,v=core.v,cellN=h.cellCount*v.nz,edgeN=h.edgeCount*v.nz,wN=h.cellCount*(v.nz+1);
    this.params=makeEmpty(this.device,48,uniform,'Stage4 RK3 split params');
    this.buffers.baseCell=makeEmpty(this.device,cellN*8,storage,'RK3 immutable base cell');this.buffers.baseU=makeEmpty(this.device,edgeN*4,storage,'RK3 immutable base u');this.buffers.baseW=makeEmpty(this.device,wN*4,storage,'RK3 immutable base w');this.buffers.predCell=makeEmpty(this.device,cellN*8,storage,'RK3 packed predictor cell');this.buffers.acousticCell=makeEmpty(this.device,cellN*8,storage,'RK3 acoustic cell');this.buffers.frozenScalar=makeEmpty(this.device,cellN*8,storage,'RK3 frozen scalar RHS');this.buffers.acousticU=makeEmpty(this.device,edgeN*4,storage,'RK3 acoustic u');this.buffers.frozenU=makeEmpty(this.device,edgeN*4,storage,'RK3 frozen u RHS');this.buffers.iface=makeEmpty(this.device,wN*16,storage,'RK3 predictor/acoustic/frozen w');this.buffers.hRefFlux=makeEmpty(this.device,edgeN*8,storage,'RK3 horizontal reference correction flux');this.buffers.hCorr=makeEmpty(this.device,cellN*8,storage,'RK3 horizontal reference correction tendency');this.buffers.refFlux=makeEmpty(this.device,wN*8,storage,'RK3 vertical reference flux');
    this.slow=GpuStage4SlowTendencyReference.create(core);this.divergence=new GpuAcousticDivergenceDamping(core,this.buffers.acousticU);this.build();
  }
  static async create(h:CubedSphereGrid,v:VerticalGrid,ref:ReferenceAtmosphere,state:DryState,acousticRatio=4):Promise<GpuStage4Rk3SplitReference>{
    const core=await GpuRotatingDryCore.create(h,v,ref,state);core.device.pushErrorScope?.('validation');let out:GpuStage4Rk3SplitReference|undefined;
    try{out=new GpuStage4Rk3SplitReference(core,acousticRatio);const err=await core.device.popErrorScope?.();if(err){out.destroy();throw new Error(`Stage 4 RK3 split WebGPU validation: ${err.message||err}`);}return out;}catch(e){if(!out){try{await core.device.popErrorScope?.();}catch{}core.destroy();}throw e;}
  }
  private pipe(code:string,label:string):GPUAny{return this.device.createComputePipeline({label,layout:'auto',compute:{module:this.device.createShaderModule({label:`${label} shader`,code}),entryPoint:'main'}});}
  private bind(name:string,code:string,buffers:GPUAny[]):void{const p=this.pipe(code,`Stage4 RK3 ${name}`);this.pipelines[name]=p;this.groups[name]=this.device.createBindGroup({layout:p.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.params}},...buffers.map((buffer,i)=>({binding:i+1,resource:{buffer}}))]});}
  private build():void{
    const b=this.core.core.buffers,s=this.slow,rb=this.buffers;
    this.bind('packBase',PACK_BASE,[b.rho,b.rhoTheta,rb.baseCell]);
    this.bind('prepCell',PREP_CELL,[b.rho,b.rhoTheta,rb.baseCell,s.scalarT,rb.predCell,rb.acousticCell,rb.frozenScalar]);
    this.bind('addHref',ADD_HREF,[b.cellEdges,b.cellSigns,b.edgeMetric,b.cellArea,b.layerRef,b.u,rb.frozenScalar]);
    this.bind('addVref',ADD_VREF,[b.layerRef,b.interfaceRef,b.w,rb.frozenScalar]);
    this.bind('prepU',PREP_U,[b.edgeCells,b.edgeMetric,rb.predCell,s.uT,rb.frozenU]);
    this.bind('prepIface',PREP_IFACE,[b.layerRef,rb.predCell,b.w,rb.baseW,s.wT,b.heviRayleigh,rb.iface]);
    this.bind('hvel',HVEL,[b.edgeCells,b.edgeMetric,rb.predCell,rb.acousticCell,rb.frozenU,rb.acousticU]);
    this.bind('hrefFlux',HREF_FLUX,[b.edgeMetric,b.layerRef,b.u,rb.acousticU,rb.hRefFlux]);
    this.bind('hrefDiv',HREF_DIV,[b.cellArea,b.layerRef,b.cellEdges,b.cellSigns,rb.hRefFlux,rb.hCorr]);
    this.bind('vertical',VERTICAL,[b.layerRef,b.interfaceRef,rb.predCell,rb.acousticCell,rb.frozenScalar,rb.hCorr,rb.iface,rb.refFlux]);
    this.bind('unpackCell',UNPACK_CELL,[rb.acousticCell,b.rho,b.rhoTheta]);this.bind('unpackW',UNPACK_W,[rb.iface,b.w]);
  }
  private writeParams(dt:number,topAbsorber:boolean):void{const ab=new ArrayBuffer(48),u=new Uint32Array(ab),f=new Float32Array(ab);u[0]=this.core.v.nz;u[1]=this.core.h.edgeCount;u[2]=this.core.h.cellCount;u[3]=topAbsorber?1:0;f[4]=dt;f[5]=EARTH.radius;f[6]=EARTH.gravity;f[7]=DRY_AIR.rd;f[8]=DRY_AIR.gamma;f[9]=DRY_AIR.pRef;f[10]=STAGE4_HEVI_OFFCENTERING;this.device.queue.writeBuffer(this.params,0,ab);}
  private dispatch(p:GPUAny,name:string,count:number,size=128):void{p.setPipeline(this.pipelines[name]);p.setBindGroup(0,this.groups[name]);p.dispatchWorkgroups(Math.ceil(count/size));}
  private one(enc:GPUAny,name:string,count:number,size=128):void{const p=enc.beginComputePass();this.dispatch(p,name,count,size);p.end();}
  private snapshotBase():void{this.writeParams(0,true);const h=this.core.h,v=this.core.v,b=this.core.core.buffers,enc=this.device.createCommandEncoder({label:'Stage4 RK3 snapshot immutable base'});this.one(enc,'packBase',h.cellCount*v.nz);enc.copyBufferToBuffer(b.u,0,this.buffers.baseU,0,h.edgeCount*v.nz*4);enc.copyBufferToBuffer(b.w,0,this.buffers.baseW,0,h.cellCount*(v.nz+1)*4);this.device.queue.submit([enc.finish()]);}
  private prepareStage(dtFast:number,topAbsorber:boolean):void{
    this.writeParams(dtFast,topAbsorber);const h=this.core.h,v=this.core.v,b=this.core.core.buffers,enc=this.device.createCommandEncoder({label:'Stage4 RK3 prepare frozen stage'});
    this.one(enc,'prepCell',h.cellCount*v.nz);this.one(enc,'addHref',h.cellCount*v.nz);this.one(enc,'addVref',h.cellCount*v.nz);this.one(enc,'prepU',h.edgeCount*v.nz);this.one(enc,'prepIface',h.cellCount*(v.nz+1));enc.copyBufferToBuffer(this.buffers.baseU,0,this.buffers.acousticU,0,h.edgeCount*v.nz*4);this.device.queue.submit([enc.finish()]);
  }
  private advanceStage(dtFast:number,steps:number,divergenceDamping:boolean):void{
    const h=this.core.h,v=this.core.v,b=this.core.core.buffers;this.writeParams(dtFast,true);if(divergenceDamping)this.divergence.prepare(dtFast);const enc=this.device.createCommandEncoder({label:`Stage4 RK3 acoustic stage ${steps}x${dtFast}`});
    for(let n=0;n<steps;n++){this.one(enc,'hvel',h.edgeCount*v.nz);this.one(enc,'hrefFlux',h.edgeCount*v.nz);this.one(enc,'hrefDiv',h.cellCount*v.nz);this.one(enc,'vertical',h.cellCount,1);if(divergenceDamping)this.divergence.encode(enc);}
    this.one(enc,'unpackCell',h.cellCount*v.nz);this.one(enc,'unpackW',h.cellCount*(v.nz+1));enc.copyBufferToBuffer(this.buffers.acousticU,0,b.u,0,h.edgeCount*v.nz*4);this.device.queue.submit([enc.finish()]);
  }
  step(dt:number,options:Stage4Rk3SplitOptions={}):void{
    if(!(dt>0))throw new Error('Stage4 GPU RK3 dt must be positive');const schedule=buildRk3SplitSchedule(this.acousticRatio),held=options.heldSuarez!==false,momentum=options.momentumTransport!==false,coriolis=options.coriolis!==false,divergence=options.divergenceDamping!==false,top=options.topAbsorber!==false;this.snapshotBase();
    for(const stage of schedule){this.slow.compute(held,momentum,coriolis);const dtFast=dt*stage.acousticDtFraction;this.prepareStage(dtFast,top);this.advanceStage(dtFast,stage.acousticSteps,divergence);}
  }
  uploadState(state:DryState):void{this.core.core.uploadState(state);}
  downloadState(time=0):Promise<DryState>{return this.core.downloadState(time);}
  destroy():void{this.divergence.destroy();this.slow.destroy();for(const b of Object.values(this.buffers))b.destroy?.();this.params.destroy?.();this.core.destroy();}
}