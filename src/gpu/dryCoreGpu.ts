import { DRY_AIR, EARTH } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { DryState } from '../solver/state.js';

type GPUAny = any;

function f32(src:ArrayLike<number>):Float32Array { return Float32Array.from(src); }
function makeBuffer(device:GPUAny,data:ArrayBufferView,usage:number,label:string):GPUAny {
  const size=Math.max(4,Math.ceil(data.byteLength/4)*4); const b=device.createBuffer({label,size,usage,mappedAtCreation:true});
  const target=new Uint8Array(b.getMappedRange()); target.set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength)); b.unmap(); return b;
}
function makeEmpty(device:GPUAny,size:number,usage:number,label:string):GPUAny { return device.createBuffer({label,size:Math.max(4,Math.ceil(size/4)*4),usage}); }

export interface GpuCoreStatus { adapterInfo?:unknown; limits:Record<string,number>; }

const COMMON = /* wgsl */`
struct Params {
  nz:u32, edgeCount:u32, cellCount:u32, _pad0:u32,
  dt:f32, radius:f32, gravity:f32, rd:f32,
  gamma:f32, pRef:f32, _pad1:vec2<f32>,
};
@group(0) @binding(0) var<uniform> params:Params;
fn p_from_x(x:f32)->f32 { return params.pRef * pow(params.rd*x/params.pRef, params.gamma); }
`;

const PRESSURE_SHADER = COMMON + /* wgsl */`
@group(0) @binding(1) var<storage,read> rhoTheta:array<f32>;
@group(0) @binding(2) var<storage,read_write> pressure:array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let q=gid.x; let count=params.cellCount*params.nz; if(q>=count){return;} pressure[q]=p_from_x(rhoTheta[q]);
}`;

const HVEL_SHADER = COMMON + /* wgsl */`
@group(0) @binding(1) var<storage,read> edgeCells:array<vec2<u32>>;
@group(0) @binding(2) var<storage,read> edgeMetric:array<vec2<f32>>;
@group(0) @binding(3) var<storage,read> rho:array<f32>;
@group(0) @binding(4) var<storage,read> pressure:array<f32>;
@group(0) @binding(5) var<storage,read_write> uEdge:array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let q=gid.x; let count=params.edgeCount*params.nz; if(q>=count){return;}
  let e=q/params.nz; let k=q-e*params.nz; let cells=edgeCells[e];
  let l=cells.x*params.nz+k; let r=cells.y*params.nz+k; let rhoAvg=max(0.5*(rho[l]+rho[r]),1e-8);
  let dist=edgeMetric[e].y; uEdge[q]=uEdge[q]-params.dt*(pressure[r]-pressure[l])/(rhoAvg*dist);
}`;

const HEVI_SHADER = COMMON + /* wgsl */`
const MAX_NZ:u32=128u;
@group(0) @binding(1) var<storage,read> zCenter:array<f32>;
@group(0) @binding(2) var<storage,read> dz:array<f32>;
@group(0) @binding(3) var<storage,read> refP:array<f32>;
@group(0) @binding(4) var<storage,read> refRho:array<f32>;
@group(0) @binding(5) var<storage,read> refX:array<f32>;
@group(0) @binding(6) var<storage,read> refRhoI:array<f32>;
@group(0) @binding(7) var<storage,read> refXI:array<f32>;
@group(0) @binding(8) var<storage,read_write> rho:array<f32>;
@group(0) @binding(9) var<storage,read_write> rhoTheta:array<f32>;
@group(0) @binding(10) var<storage,read_write> w:array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let c=gid.x; if(c>=params.cellCount || params.nz>MAX_NZ){return;} let nz=params.nz; let n=nz-1u;
  var oldW:array<f32,129>; var pPrime:array<f32,128>; var Lold:array<f32,128>;
  var lo:array<f32,127>; var di:array<f32,127>; var up:array<f32,127>; var rhs:array<f32,127>;
  var cp:array<f32,127>; var dp:array<f32,127>; var sol:array<f32,127>;
  for(var k:u32=0u;k<=nz;k++){ oldW[k]=w[c*(nz+1u)+k]; }
  for(var k:u32=0u;k<nz;k++){
    let q=c*nz+k; pPrime[k]=p_from_x(rhoTheta[q])-p_from_x(refX[k]);
    Lold[k]=(refXI[k+1u]*oldW[k+1u]-refXI[k]*oldW[k])/dz[k];
  }
  for(var ii:u32=0u;ii<n;ii++){
    let i=ii+1u; let l=i-1u; let u=i; let dzc=zCenter[u]-zCenter[l];
    let ql=c*nz+l; let qu=c*nz+u; let rho0i=0.5*(refRho[l]+refRho[u]); let rhoi=max(0.5*(rho0i+0.5*(rho[ql]+rho[qu])),1e-8);
    let Al=params.gamma*refP[l]/refX[l]; let Au=params.gamma*refP[u]/refX[u]; let fac=0.25*params.dt*params.dt/(rhoi*dzc);
    let xim=refXI[i-1u]; let xi=refXI[i]; let xip=refXI[i+1u];
    lo[ii]=-fac*Al*xim/dz[l]; di[ii]=1.0+fac*(Au*xi/dz[u]+Al*xi/dz[l]); up[ii]=-fac*Au*xip/dz[u];
    rhs[ii]=oldW[i]-params.dt*(pPrime[u]-pPrime[l])/(rhoi*dzc)+fac*(Au*Lold[u]-Al*Lold[l]);
  }
  if(n>0u){
    lo[0u]=0.0; up[n-1u]=0.0; var denom=di[0u]; cp[0u]=select(up[0u]/denom,0.0,n==1u); dp[0u]=rhs[0u]/denom;
    for(var i:u32=1u;i<n;i++){denom=di[i]-lo[i]*cp[i-1u]; cp[i]=select(up[i]/denom,0.0,i==n-1u); dp[i]=(rhs[i]-lo[i]*dp[i-1u])/denom;}
    sol[n-1u]=dp[n-1u]; var jj:i32=i32(n)-2; loop{if(jj<0){break;} let j=u32(jj);sol[j]=dp[j]-cp[j]*sol[j+1u];jj=jj-1;}
  }
  w[c*(nz+1u)]=0.0; w[c*(nz+1u)+nz]=0.0; for(var i:u32=1u;i<nz;i++){w[c*(nz+1u)+i]=sol[i-1u];}
  for(var k:u32=0u;k<nz;k++){
    let q=c*nz+k; let Lnew=(refXI[k+1u]*w[c*(nz+1u)+k+1u]-refXI[k]*w[c*(nz+1u)+k])/dz[k];
    rhoTheta[q]=rhoTheta[q]-0.5*params.dt*(Lold[k]+Lnew);
    let Rold=(refRhoI[k+1u]*oldW[k+1u]-refRhoI[k]*oldW[k])/dz[k];
    let Rnew=(refRhoI[k+1u]*w[c*(nz+1u)+k+1u]-refRhoI[k]*w[c*(nz+1u)+k])/dz[k];
    rho[q]=rho[q]-0.5*params.dt*(Rold+Rnew);
  }
}`;

const BUOYANCY_SHADER = COMMON + /* wgsl */`
@group(0) @binding(1) var<storage,read> refRho:array<f32>;
@group(0) @binding(2) var<storage,read> rho:array<f32>;
@group(0) @binding(3) var<storage,read_write> w:array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let q=gid.x; let count=params.cellCount*(params.nz-1u); if(q>=count){return;} let c=q/(params.nz-1u); let i=1u+(q-c*(params.nz-1u));
  let l=c*params.nz+i-1u; let u=c*params.nz+i; let wi=c*(params.nz+1u)+i;
  let rr=0.5*(rho[l]+rho[u]); let r0=0.5*(refRho[i-1u]+refRho[i]); w[wi]=w[wi]-params.dt*params.gravity*(rr-r0)/max(rr,1e-8);
}`;

const HFLUX_SHADER = COMMON + /* wgsl */`
@group(0) @binding(1) var<storage,read> edgeCells:array<vec2<u32>>;
@group(0) @binding(2) var<storage,read> edgeMetric:array<vec2<f32>>;
@group(0) @binding(3) var<storage,read> dz:array<f32>;
@group(0) @binding(4) var<storage,read> rho:array<f32>;
@group(0) @binding(5) var<storage,read> rhoTheta:array<f32>;
@group(0) @binding(6) var<storage,read> uEdge:array<f32>;
@group(0) @binding(7) var<storage,read_write> fluxRho:array<f32>;
@group(0) @binding(8) var<storage,read_write> fluxX:array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let q=gid.x; let count=params.edgeCount*params.nz;if(q>=count){return;} let e=q/params.nz;let k=q-e*params.nz;let cc=edgeCells[e];let l=cc.x*params.nz+k;let r=cc.y*params.nz+k;let vel=uEdge[q];let src=select(r,l,vel>=0.0);let area=edgeMetric[e].x*dz[k];fluxRho[q]=rho[src]*vel*area;fluxX[q]=rhoTheta[src]*vel*area;
}`;

const VFLUX_SHADER = COMMON + /* wgsl */`
@group(0) @binding(1) var<storage,read> cellArea:array<f32>;
@group(0) @binding(2) var<storage,read> refRho:array<f32>;
@group(0) @binding(3) var<storage,read> refX:array<f32>;
@group(0) @binding(4) var<storage,read> rho:array<f32>;
@group(0) @binding(5) var<storage,read> rhoTheta:array<f32>;
@group(0) @binding(6) var<storage,read> w:array<f32>;
@group(0) @binding(7) var<storage,read_write> fluxRho:array<f32>;
@group(0) @binding(8) var<storage,read_write> fluxX:array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let q=gid.x;let count=params.cellCount*(params.nz+1u);if(q>=count){return;}let c=q/(params.nz+1u);let i=q-c*(params.nz+1u);if(i==0u||i==params.nz){fluxRho[q]=0.0;fluxX[q]=0.0;return;}let vel=w[q];let srcK=select(i,i-1u,vel>=0.0);let src=c*params.nz+srcK;let area=cellArea[c];fluxRho[q]=(rho[src]-refRho[srcK])*vel*area;fluxX[q]=(rhoTheta[src]-refX[srcK])*vel*area;
}`;

const DIVERGENCE_SHADER = COMMON + /* wgsl */`
@group(0) @binding(1) var<storage,read> cellArea:array<f32>;
@group(0) @binding(2) var<storage,read> dz:array<f32>;
@group(0) @binding(3) var<storage,read> cellEdges:array<vec4<u32>>;
@group(0) @binding(4) var<storage,read> cellSigns:array<vec4<i32>>;
@group(0) @binding(5) var<storage,read> hFluxRho:array<f32>;
@group(0) @binding(6) var<storage,read> hFluxX:array<f32>;
@group(0) @binding(7) var<storage,read> vFluxRho:array<f32>;
@group(0) @binding(8) var<storage,read> vFluxX:array<f32>;
@group(0) @binding(9) var<storage,read_write> rho:array<f32>;
@group(0) @binding(10) var<storage,read_write> rhoTheta:array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let q=gid.x;let count=params.cellCount*params.nz;if(q>=count){return;}let c=q/params.nz;let k=q-c*params.nz;let ee=cellEdges[c];let ss=cellSigns[c];
 let hR=f32(ss.x)*hFluxRho[ee.x*params.nz+k]+f32(ss.y)*hFluxRho[ee.y*params.nz+k]+f32(ss.z)*hFluxRho[ee.z*params.nz+k]+f32(ss.w)*hFluxRho[ee.w*params.nz+k];
 let hX=f32(ss.x)*hFluxX[ee.x*params.nz+k]+f32(ss.y)*hFluxX[ee.y*params.nz+k]+f32(ss.z)*hFluxX[ee.z*params.nz+k]+f32(ss.w)*hFluxX[ee.w*params.nz+k];
 let vb=c*(params.nz+1u)+k;let vt=vb+1u;let volume=cellArea[c]*dz[k];rho[q]=rho[q]+params.dt*(-hR+vFluxRho[vb]-vFluxRho[vt])/volume;rhoTheta[q]=rhoTheta[q]+params.dt*(-hX+vFluxX[vb]-vFluxX[vt])/volume;
}`;

export class GpuDryCorePrototype {
  readonly device:GPUAny; readonly buffers:Record<string,GPUAny>={}; readonly pipelines:Record<string,GPUAny>={}; readonly bindGroups:Record<string,GPUAny>={};
  readonly status:GpuCoreStatus; private readonly paramsBuffer:GPUAny;
  private constructor(device:GPUAny,adapter:GPUAny,public readonly h:CubedSphereGrid,public readonly v:VerticalGrid,public readonly ref:ReferenceAtmosphere,state:DryState){
    this.device=device; this.status={adapterInfo:adapter.info,limits:Object.fromEntries(Object.entries(adapter.limits).filter(([,v])=>typeof v==='number')) as Record<string,number>};
    const U=(globalThis as any).GPUBufferUsage; const storage=U.STORAGE|U.COPY_DST|U.COPY_SRC, uniform=U.UNIFORM|U.COPY_DST;
    this.paramsBuffer=makeEmpty(device,48,uniform,'core params');
    const edgeCells=new Uint32Array(h.edgeCount*2),edgeMetric=new Float32Array(h.edgeCount*2);
    for(let e=0;e<h.edgeCount;e++){const ge=h.edges[e]!;edgeCells[e*2]=ge.leftCell;edgeCells[e*2+1]=ge.rightCell;edgeMetric[e*2]=ge.angularLength*EARTH.radius;edgeMetric[e*2+1]=ge.centerDistanceAngle*EARTH.radius;}
    const cellArea=new Float32Array(h.cellCount);for(let c=0;c<h.cellCount;c++)cellArea[c]=h.cellAreaUnit[c]!*EARTH.radius*EARTH.radius;
    this.buffers.edgeCells=makeBuffer(device,edgeCells,storage,'edgeCells');this.buffers.edgeMetric=makeBuffer(device,edgeMetric,storage,'edgeMetric');
    this.buffers.cellArea=makeBuffer(device,cellArea,storage,'cellArea');this.buffers.cellEdges=makeBuffer(device,new Uint32Array(h.cellEdges),storage,'cellEdges');this.buffers.cellSigns=makeBuffer(device,new Int32Array(h.cellEdgeSigns),storage,'cellSigns');
    this.buffers.zCenter=makeBuffer(device,f32(v.zCenter),storage,'zCenter');this.buffers.dz=makeBuffer(device,f32(v.dz),storage,'dz');
    this.buffers.refP=makeBuffer(device,f32(ref.pCenter),storage,'refP');this.buffers.refRho=makeBuffer(device,f32(ref.rhoCenter),storage,'refRho');this.buffers.refX=makeBuffer(device,f32(ref.rhoThetaCenter),storage,'refX');this.buffers.refRhoI=makeBuffer(device,f32(ref.rhoInterface),storage,'refRhoI');this.buffers.refXI=makeBuffer(device,f32(ref.rhoThetaInterface),storage,'refXI');
    this.buffers.rho=makeBuffer(device,f32(state.rhoD),storage,'rhoD');this.buffers.rhoTheta=makeBuffer(device,f32(state.rhoThetaM),storage,'rhoThetaM');this.buffers.u=makeBuffer(device,f32(state.uEdge),storage,'uEdge');this.buffers.w=makeBuffer(device,f32(state.wInterface),storage,'wInterface');
    this.buffers.pressure=makeEmpty(device,state.rhoD.length*4,storage,'pressure');this.buffers.hFluxRho=makeEmpty(device,state.uEdge.length*4,storage,'hFluxRho');this.buffers.hFluxX=makeEmpty(device,state.uEdge.length*4,storage,'hFluxX');this.buffers.vFluxRho=makeEmpty(device,state.wInterface.length*4,storage,'vFluxRho');this.buffers.vFluxX=makeEmpty(device,state.wInterface.length*4,storage,'vFluxX');
    this.buildPipelines();
  }
  static async create(h:CubedSphereGrid,v:VerticalGrid,ref:ReferenceAtmosphere,state:DryState):Promise<GpuDryCorePrototype>{
    const nav=(globalThis as any).navigator;if(!nav?.gpu)throw new Error('WebGPU unavailable');const adapter=await nav.gpu.requestAdapter();if(!adapter)throw new Error('No WebGPU adapter');const device=await adapter.requestDevice();if(v.nz>128)throw new Error('Stage 3 GPU HEVI supports nz<=128');device.pushErrorScope?.('validation');const core=new GpuDryCorePrototype(device,adapter,h,v,ref,state);const err=await device.popErrorScope?.();if(err){core.destroy();throw new Error(`WebGPU validation: ${err.message||err}`);}return core;
  }
  private pipeline(code:string,label:string):GPUAny{return this.device.createComputePipeline({label,layout:'auto',compute:{module:this.device.createShaderModule({label:`${label} shader`,code}),entryPoint:'main'}});}
  private entries(names:string[]):any[]{return names.map((n,i)=>({binding:i,resource:{buffer:n==='params'?this.paramsBuffer:this.buffers[n]}}));}
  private buildPipelines():void{
    const defs:[string,string,string[]][]=[
      ['pressure',PRESSURE_SHADER,['params','rhoTheta','pressure']],
      ['hvel',HVEL_SHADER,['params','edgeCells','edgeMetric','rho','pressure','u']],
      ['hevi',HEVI_SHADER,['params','zCenter','dz','refP','refRho','refX','refRhoI','refXI','rho','rhoTheta','w']],
      ['buoyancy',BUOYANCY_SHADER,['params','refRho','rho','w']],
      ['hflux',HFLUX_SHADER,['params','edgeCells','edgeMetric','dz','rho','rhoTheta','u','hFluxRho','hFluxX']],
      ['vflux',VFLUX_SHADER,['params','cellArea','refRho','refX','rho','rhoTheta','w','vFluxRho','vFluxX']],
      ['divergence',DIVERGENCE_SHADER,['params','cellArea','dz','cellEdges','cellSigns','hFluxRho','hFluxX','vFluxRho','vFluxX','rho','rhoTheta']],
    ];
    for(const [name,code,bindings] of defs){const p=this.pipeline(code,name);this.pipelines[name]=p;this.bindGroups[name]=this.device.createBindGroup({layout:p.getBindGroupLayout(0),entries:this.entries(bindings)});}
  }
  private writeParams(dt:number):void{const ab=new ArrayBuffer(48),u=new Uint32Array(ab),f=new Float32Array(ab);u[0]=this.v.nz;u[1]=this.h.edgeCount;u[2]=this.h.cellCount;f[4]=dt;f[5]=EARTH.radius;f[6]=EARTH.gravity;f[7]=DRY_AIR.rd;f[8]=DRY_AIR.gamma;f[9]=DRY_AIR.pRef;this.device.queue.writeBuffer(this.paramsBuffer,0,ab);}
  private dispatch(pass:GPUAny,name:string,count:number,size=128):void{pass.setPipeline(this.pipelines[name]);pass.setBindGroup(0,this.bindGroups[name]);pass.dispatchWorkgroups(Math.ceil(count/size));}
  step(dt:number):void{
    this.writeParams(dt);const enc=this.device.createCommandEncoder({label:'dry core step'});const pass=enc.beginComputePass();
    this.dispatch(pass,'pressure',this.h.cellCount*this.v.nz);this.dispatch(pass,'hvel',this.h.edgeCount*this.v.nz);this.dispatch(pass,'hevi',this.h.cellCount,1);this.dispatch(pass,'buoyancy',this.h.cellCount*(this.v.nz-1));this.dispatch(pass,'hflux',this.h.edgeCount*this.v.nz);this.dispatch(pass,'vflux',this.h.cellCount*(this.v.nz+1));this.dispatch(pass,'divergence',this.h.cellCount*this.v.nz);pass.end();this.device.queue.submit([enc.finish()]);
  }
  uploadState(state:DryState):void{this.device.queue.writeBuffer(this.buffers.rho,0,f32(state.rhoD));this.device.queue.writeBuffer(this.buffers.rhoTheta,0,f32(state.rhoThetaM));this.device.queue.writeBuffer(this.buffers.u,0,f32(state.uEdge));this.device.queue.writeBuffer(this.buffers.w,0,f32(state.wInterface));}

  private async readF32(buffer:GPUAny,count:number):Promise<Float32Array>{
    const U=(globalThis as any).GPUBufferUsage, M=(globalThis as any).GPUMapMode; const bytes=count*4;
    const stage=this.device.createBuffer({size:Math.max(4,bytes),usage:U.COPY_DST|U.MAP_READ}); const enc=this.device.createCommandEncoder();enc.copyBufferToBuffer(buffer,0,stage,0,bytes);this.device.queue.submit([enc.finish()]);await stage.mapAsync(M.READ);const out=new Float32Array(stage.getMappedRange().slice(0));stage.unmap();stage.destroy();return out;
  }
  async downloadState(time=0):Promise<DryState>{
    await this.device.queue.onSubmittedWorkDone();
    const [rho,x,u,w]=await Promise.all([this.readF32(this.buffers.rho,this.h.cellCount*this.v.nz),this.readF32(this.buffers.rhoTheta,this.h.cellCount*this.v.nz),this.readF32(this.buffers.u,this.h.edgeCount*this.v.nz),this.readF32(this.buffers.w,this.h.cellCount*(this.v.nz+1))]);
    return {rhoD:Float64Array.from(rho),rhoThetaM:Float64Array.from(x),uEdge:Float64Array.from(u),wInterface:Float64Array.from(w),time};
  }
  destroy():void{for(const b of Object.values(this.buffers))b.destroy?.();this.paramsBuffer.destroy?.();this.device.destroy?.();}
}
