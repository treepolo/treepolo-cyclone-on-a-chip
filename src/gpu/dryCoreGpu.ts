import { DRY_AIR, EARTH } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { DryState } from '../solver/state.js';

type GPUAny = any;

function f32(src:ArrayLike<number>):Float32Array { return Float32Array.from(src); }
function makeBuffer(device:GPUAny,data:ArrayBufferView,usage:number,label:string):GPUAny {
  const size=Math.max(4,Math.ceil(data.byteLength/4)*4);
  const b=device.createBuffer({label,size,usage,mappedAtCreation:true});
  const target=new Uint8Array(b.getMappedRange());
  target.set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));
  b.unmap();
  return b;
}
function makeEmpty(device:GPUAny,size:number,usage:number,label:string):GPUAny {
  return device.createBuffer({label,size:Math.max(4,Math.ceil(size/4)*4),usage});
}

export interface GpuCoreStatus {
  adapterInfo?:unknown;
  limits:Record<string,number>;
  requiredStorageBuffersPerStage:number;
}

const COMMON = /* wgsl */`
struct Params {
  nz:u32, edgeCount:u32, cellCount:u32, _pad0:u32,
  dt:f32, radius:f32, gravity:f32, rd:f32,
  gamma:f32, pRef:f32, heviOffCentering:f32, _pad1:f32,
};
@group(0) @binding(0) var<uniform> params:Params;
fn p_from_x(x:f32)->f32 { return params.pRef * pow(params.rd*x/params.pRef, params.gamma); }
`;

const PRESSURE_SHADER = COMMON + /* wgsl */`
@group(0) @binding(1) var<storage,read> rhoTheta:array<f32>;
@group(0) @binding(2) var<storage,read_write> pressure:array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let q=gid.x; let count=params.cellCount*params.nz; if(q>=count){return;}
  pressure[q]=p_from_x(rhoTheta[q]);
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
  let l=cells.x*params.nz+k; let r=cells.y*params.nz+k;
  let rhoAvg=max(0.5*(rho[l]+rho[r]),1e-8);
  let dist=edgeMetric[e].y;
  uEdge[q]=uEdge[q]-params.dt*(pressure[r]-pressure[l])/(rhoAvg*dist);
}`;

const HEVI_SHADER = COMMON + /* wgsl */`
const MAX_NZ:u32=128u;
@group(0) @binding(1) var<storage,read> layerRef:array<f32>;
@group(0) @binding(2) var<storage,read> interfaceRef:array<f32>;
@group(0) @binding(3) var<storage,read_write> rho:array<f32>;
@group(0) @binding(4) var<storage,read_write> rhoTheta:array<f32>;
@group(0) @binding(5) var<storage,read_write> w:array<f32>;
fn lz(k:u32)->f32{return layerRef[k*5u];}
fn ldz(k:u32)->f32{return layerRef[k*5u+1u];}
fn lp(k:u32)->f32{return layerRef[k*5u+2u];}
fn lrho(k:u32)->f32{return layerRef[k*5u+3u];}
fn lx(k:u32)->f32{return layerRef[k*5u+4u];}
fn irho(i:u32)->f32{return interfaceRef[i*2u];}
fn ix(i:u32)->f32{return interfaceRef[i*2u+1u];}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let c=gid.x; if(c>=params.cellCount || params.nz>MAX_NZ){return;}
  let nz=params.nz; let n=nz-1u;
  let theta=0.5*(1.0+clamp(params.heviOffCentering,0.0,0.999));
  let oldWeight=1.0-theta;
  var oldW:array<f32,129>; var pPrime:array<f32,128>; var Lold:array<f32,128>;
  var lo:array<f32,127>; var di:array<f32,127>; var up:array<f32,127>; var rhs:array<f32,127>;
  var cp:array<f32,127>; var dp:array<f32,127>; var sol:array<f32,127>;
  for(var k:u32=0u;k<=nz;k++){ oldW[k]=w[c*(nz+1u)+k]; }
  for(var k:u32=0u;k<nz;k++){
    let q=c*nz+k;
    pPrime[k]=p_from_x(rhoTheta[q])-lp(k);
    Lold[k]=(ix(k+1u)*oldW[k+1u]-ix(k)*oldW[k])/ldz(k);
  }
  for(var ii:u32=0u;ii<n;ii++){
    let i=ii+1u; let l=i-1u; let u=i;
    let dzc=lz(u)-lz(l);
    let ql=c*nz+l; let qu=c*nz+u;
    let rho0i=0.5*(lrho(l)+lrho(u));
    let rhoi=max(0.5*(rho0i+0.5*(rho[ql]+rho[qu])),1e-8);
    let Al=params.gamma*lp(l)/lx(l);
    let Au=params.gamma*lp(u)/lx(u);
    let base=params.dt*params.dt/(rhoi*dzc);
    let facNew=theta*theta*base;
    let facOld=theta*oldWeight*base;
    let xim=ix(i-1u); let xi=ix(i); let xip=ix(i+1u);
    lo[ii]=-facNew*Al*xim/ldz(l);
    di[ii]=1.0+facNew*(Au*xi/ldz(u)+Al*xi/ldz(l));
    up[ii]=-facNew*Au*xip/ldz(u);
    rhs[ii]=oldW[i]-params.dt*(pPrime[u]-pPrime[l])/(rhoi*dzc)+facOld*(Au*Lold[u]-Al*Lold[l]);
  }
  if(n>0u){
    lo[0u]=0.0; up[n-1u]=0.0;
    var denom=di[0u]; cp[0u]=select(up[0u]/denom,0.0,n==1u); dp[0u]=rhs[0u]/denom;
    for(var i:u32=1u;i<n;i++){
      denom=di[i]-lo[i]*cp[i-1u];
      cp[i]=select(up[i]/denom,0.0,i==n-1u);
      dp[i]=(rhs[i]-lo[i]*dp[i-1u])/denom;
    }
    sol[n-1u]=dp[n-1u];
    var jj:i32=i32(n)-2;
    loop{if(jj<0){break;} let j=u32(jj); sol[j]=dp[j]-cp[j]*sol[j+1u]; jj=jj-1;}
  }
  w[c*(nz+1u)]=0.0; w[c*(nz+1u)+nz]=0.0;
  for(var i:u32=1u;i<nz;i++){w[c*(nz+1u)+i]=sol[i-1u];}
  for(var k:u32=0u;k<nz;k++){
    let q=c*nz+k;
    let dz=ldz(k);
    let Lnew=(ix(k+1u)*w[c*(nz+1u)+k+1u]-ix(k)*w[c*(nz+1u)+k])/dz;
    rhoTheta[q]=rhoTheta[q]-params.dt*(oldWeight*Lold[k]+theta*Lnew);
    let Rold=(irho(k+1u)*oldW[k+1u]-irho(k)*oldW[k])/dz;
    let Rnew=(irho(k+1u)*w[c*(nz+1u)+k+1u]-irho(k)*w[c*(nz+1u)+k])/dz;
    rho[q]=rho[q]-params.dt*(oldWeight*Rold+theta*Rnew);
  }
}`;

const BUOYANCY_SHADER = COMMON + /* wgsl */`
@group(0) @binding(1) var<storage,read> layerRef:array<f32>;
@group(0) @binding(2) var<storage,read> rho:array<f32>;
@group(0) @binding(3) var<storage,read_write> w:array<f32>;
fn lrho(k:u32)->f32{return layerRef[k*5u+3u];}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let q=gid.x; let count=params.cellCount*(params.nz-1u); if(q>=count){return;}
  let c=q/(params.nz-1u); let i=1u+(q-c*(params.nz-1u));
  let l=c*params.nz+i-1u; let u=c*params.nz+i; let wi=c*(params.nz+1u)+i;
  let rr=0.5*(rho[l]+rho[u]);
  let r0=0.5*(lrho(i-1u)+lrho(i));
  w[wi]=w[wi]-params.dt*params.gravity*(rr-r0)/max(rr,1e-8);
}`;

const HFLUX_SHADER = COMMON + /* wgsl */`
@group(0) @binding(1) var<storage,read> edgeCells:array<vec2<u32>>;
@group(0) @binding(2) var<storage,read> edgeMetric:array<vec2<f32>>;
@group(0) @binding(3) var<storage,read> layerRef:array<f32>;
@group(0) @binding(4) var<storage,read> rho:array<f32>;
@group(0) @binding(5) var<storage,read> rhoTheta:array<f32>;
@group(0) @binding(6) var<storage,read> uEdge:array<f32>;
@group(0) @binding(7) var<storage,read_write> flux:array<vec2<f32>>;
fn ldz(k:u32)->f32{return layerRef[k*5u+1u];}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let q=gid.x; let count=params.edgeCount*params.nz; if(q>=count){return;}
  let e=q/params.nz; let k=q-e*params.nz; let cc=edgeCells[e];
  let l=cc.x*params.nz+k; let r=cc.y*params.nz+k; let vel=uEdge[q];
  let src=select(r,l,vel>=0.0); let area=edgeMetric[e].x*ldz(k);
  flux[q]=vec2<f32>(rho[src],rhoTheta[src])*vel*area;
}`;

const VFLUX_SHADER = COMMON + /* wgsl */`
@group(0) @binding(1) var<storage,read> cellArea:array<f32>;
@group(0) @binding(2) var<storage,read> layerRef:array<f32>;
@group(0) @binding(3) var<storage,read> rho:array<f32>;
@group(0) @binding(4) var<storage,read> rhoTheta:array<f32>;
@group(0) @binding(5) var<storage,read> w:array<f32>;
@group(0) @binding(6) var<storage,read_write> flux:array<vec2<f32>>;
fn lrho(k:u32)->f32{return layerRef[k*5u+3u];}
fn lx(k:u32)->f32{return layerRef[k*5u+4u];}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let q=gid.x; let count=params.cellCount*(params.nz+1u); if(q>=count){return;}
  let c=q/(params.nz+1u); let i=q-c*(params.nz+1u);
  if(i==0u||i==params.nz){flux[q]=vec2<f32>(0.0); return;}
  let vel=w[q]; let srcK=select(i,i-1u,vel>=0.0); let src=c*params.nz+srcK; let area=cellArea[c];
  flux[q]=vec2<f32>(rho[src]-lrho(srcK),rhoTheta[src]-lx(srcK))*vel*area;
}`;

const DIVERGENCE_SHADER = COMMON + /* wgsl */`
@group(0) @binding(1) var<storage,read> cellArea:array<f32>;
@group(0) @binding(2) var<storage,read> layerRef:array<f32>;
@group(0) @binding(3) var<storage,read> cellEdges:array<vec4<u32>>;
@group(0) @binding(4) var<storage,read> cellSigns:array<vec4<i32>>;
@group(0) @binding(5) var<storage,read> hFlux:array<vec2<f32>>;
@group(0) @binding(6) var<storage,read> vFlux:array<vec2<f32>>;
@group(0) @binding(7) var<storage,read_write> rho:array<f32>;
@group(0) @binding(8) var<storage,read_write> rhoTheta:array<f32>;
fn ldz(k:u32)->f32{return layerRef[k*5u+1u];}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let q=gid.x; let count=params.cellCount*params.nz; if(q>=count){return;}
  let c=q/params.nz; let k=q-c*params.nz; let ee=cellEdges[c]; let ss=cellSigns[c];
  let f0=hFlux[ee.x*params.nz+k]*f32(ss.x);
  let f1=hFlux[ee.y*params.nz+k]*f32(ss.y);
  let f2=hFlux[ee.z*params.nz+k]*f32(ss.z);
  let f3=hFlux[ee.w*params.nz+k]*f32(ss.w);
  let h=f0+f1+f2+f3;
  let vb=c*(params.nz+1u)+k; let vt=vb+1u;
  let volume=cellArea[c]*ldz(k);
  let tendency=-h+vFlux[vb]-vFlux[vt];
  rho[q]=rho[q]+params.dt*tendency.x/volume;
  rhoTheta[q]=rhoTheta[q]+params.dt*tendency.y/volume;
}`;

export class GpuDryCorePrototype {
  readonly device:GPUAny;
  readonly buffers:Record<string,GPUAny>={};
  readonly pipelines:Record<string,GPUAny>={};
  readonly bindGroups:Record<string,GPUAny>={};
  readonly status:GpuCoreStatus;
  private readonly paramsBuffer:GPUAny;

  private constructor(device:GPUAny,adapter:GPUAny,public readonly h:CubedSphereGrid,public readonly v:VerticalGrid,public readonly ref:ReferenceAtmosphere,state:DryState,public readonly heviOffCentering=0){
    this.device=device;
    this.status={adapterInfo:adapter.info,limits:Object.fromEntries(Object.entries(adapter.limits).filter(([,x])=>typeof x==='number')) as Record<string,number>,requiredStorageBuffersPerStage:8};
    const U=(globalThis as any).GPUBufferUsage;
    const storage=U.STORAGE|U.COPY_DST|U.COPY_SRC, uniform=U.UNIFORM|U.COPY_DST;
    this.paramsBuffer=makeEmpty(device,48,uniform,'core params');
    const edgeCells=new Uint32Array(h.edgeCount*2),edgeMetric=new Float32Array(h.edgeCount*2);
    for(let e=0;e<h.edgeCount;e++){
      const ge=h.edges[e]!;
      edgeCells[e*2]=ge.leftCell; edgeCells[e*2+1]=ge.rightCell;
      edgeMetric[e*2]=ge.angularLength*EARTH.radius; edgeMetric[e*2+1]=ge.centerDistanceAngle*EARTH.radius;
    }
    const cellArea=new Float32Array(h.cellCount);
    for(let c=0;c<h.cellCount;c++) cellArea[c]=h.cellAreaUnit[c]!*EARTH.radius*EARTH.radius;
    const layerRef=new Float32Array(v.nz*5);
    for(let k=0;k<v.nz;k++){
      const o=k*5;
      layerRef[o]=v.zCenter[k]!; layerRef[o+1]=v.dz[k]!; layerRef[o+2]=ref.pCenter[k]!; layerRef[o+3]=ref.rhoCenter[k]!; layerRef[o+4]=ref.rhoThetaCenter[k]!;
    }
    const interfaceRef=new Float32Array((v.nz+1)*2);
    for(let i=0;i<=v.nz;i++){
      interfaceRef[i*2]=ref.rhoInterface[i]!; interfaceRef[i*2+1]=ref.rhoThetaInterface[i]!;
    }
    this.buffers.edgeCells=makeBuffer(device,edgeCells,storage,'edgeCells');
    this.buffers.edgeMetric=makeBuffer(device,edgeMetric,storage,'edgeMetric');
    this.buffers.cellArea=makeBuffer(device,cellArea,storage,'cellArea');
    this.buffers.cellEdges=makeBuffer(device,new Uint32Array(h.cellEdges),storage,'cellEdges');
    this.buffers.cellSigns=makeBuffer(device,new Int32Array(h.cellEdgeSigns),storage,'cellSigns');
    this.buffers.layerRef=makeBuffer(device,layerRef,storage,'layerRef[z,dz,p0,rho0,rhoTheta0]');
    this.buffers.interfaceRef=makeBuffer(device,interfaceRef,storage,'interfaceRef[rho0,rhoTheta0]');
    this.buffers.rho=makeBuffer(device,f32(state.rhoD),storage,'rhoD');
    this.buffers.rhoTheta=makeBuffer(device,f32(state.rhoThetaM),storage,'rhoThetaM');
    this.buffers.u=makeBuffer(device,f32(state.uEdge),storage,'uEdge');
    this.buffers.w=makeBuffer(device,f32(state.wInterface),storage,'wInterface');
    this.buffers.pressure=makeEmpty(device,state.rhoD.length*4,storage,'pressure');
    this.buffers.hFlux=makeEmpty(device,state.uEdge.length*8,storage,'hFlux[rho,rhoTheta]');
    this.buffers.vFlux=makeEmpty(device,state.wInterface.length*8,storage,'vFlux[rho,rhoTheta]');
    this.buildPipelines();
  }

  static async create(h:CubedSphereGrid,v:VerticalGrid,ref:ReferenceAtmosphere,state:DryState,heviOffCentering=0):Promise<GpuDryCorePrototype>{
    const nav=(globalThis as any).navigator;
    if(!nav?.gpu) throw new Error('WebGPU unavailable');
    if(!(heviOffCentering>=0&&heviOffCentering<1))throw new Error('HEVI offCentering must be in [0,1)');
    const adapter=await nav.gpu.requestAdapter();
    if(!adapter) throw new Error('No WebGPU adapter');
    if(v.nz>128) throw new Error('Stage 3 GPU HEVI supports nz<=128');
    const requiredStorageBuffersPerStage=8;
    const supported=Number(adapter.limits?.maxStorageBuffersPerShaderStage ?? 0);
    if(supported<requiredStorageBuffersPerStage) throw new Error(`WebGPU adapter exposes maxStorageBuffersPerShaderStage=${supported}; Stage 3 requires ${requiredStorageBuffersPerStage}.`);
    const device=await adapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:requiredStorageBuffersPerStage}});
    device.pushErrorScope?.('validation');
    const core=new GpuDryCorePrototype(device,adapter,h,v,ref,state,heviOffCentering);
    const err=await device.popErrorScope?.();
    if(err){core.destroy();throw new Error(`WebGPU validation: ${err.message||err}`);}
    return core;
  }
  private pipeline(code:string,label:string):GPUAny{return this.device.createComputePipeline({label,layout:'auto',compute:{module:this.device.createShaderModule({label:`${label} shader`,code}),entryPoint:'main'}});}
  private entries(names:string[]):any[]{return names.map((n,i)=>({binding:i,resource:{buffer:n==='params'?this.paramsBuffer:this.buffers[n]}}));}
  private buildPipelines():void{
    const defs:[string,string,string[]][]=[
      ['pressure',PRESSURE_SHADER,['params','rhoTheta','pressure']],
      ['hvel',HVEL_SHADER,['params','edgeCells','edgeMetric','rho','pressure','u']],
      ['hevi',HEVI_SHADER,['params','layerRef','interfaceRef','rho','rhoTheta','w']],
      ['buoyancy',BUOYANCY_SHADER,['params','layerRef','rho','w']],
      ['hflux',HFLUX_SHADER,['params','edgeCells','edgeMetric','layerRef','rho','rhoTheta','u','hFlux']],
      ['vflux',VFLUX_SHADER,['params','cellArea','layerRef','rho','rhoTheta','w','vFlux']],
      ['divergence',DIVERGENCE_SHADER,['params','cellArea','layerRef','cellEdges','cellSigns','hFlux','vFlux','rho','rhoTheta']],
    ];
    for(const [name,code,bindings] of defs){const p=this.pipeline(code,name);this.pipelines[name]=p;this.bindGroups[name]=this.device.createBindGroup({layout:p.getBindGroupLayout(0),entries:this.entries(bindings)});}
  }
  private writeParams(dt:number):void{
    const ab=new ArrayBuffer(48),u=new Uint32Array(ab),f=new Float32Array(ab);
    u[0]=this.v.nz;u[1]=this.h.edgeCount;u[2]=this.h.cellCount;
    f[4]=dt;f[5]=EARTH.radius;f[6]=EARTH.gravity;f[7]=DRY_AIR.rd;f[8]=DRY_AIR.gamma;f[9]=DRY_AIR.pRef;f[10]=this.heviOffCentering;
    this.device.queue.writeBuffer(this.paramsBuffer,0,ab);
  }
  private dispatch(pass:GPUAny,name:string,count:number,size=128):void{pass.setPipeline(this.pipelines[name]);pass.setBindGroup(0,this.bindGroups[name]);pass.dispatchWorkgroups(Math.ceil(count/size));}
  step(dt:number):void{
    this.writeParams(dt);const enc=this.device.createCommandEncoder({label:'dry core step'});const pass=enc.beginComputePass();
    this.dispatch(pass,'pressure',this.h.cellCount*this.v.nz);
    this.dispatch(pass,'hvel',this.h.edgeCount*this.v.nz);
    this.dispatch(pass,'hevi',this.h.cellCount,1);
    this.dispatch(pass,'buoyancy',this.h.cellCount*(this.v.nz-1));
    this.dispatch(pass,'hflux',this.h.edgeCount*this.v.nz);
    this.dispatch(pass,'vflux',this.h.cellCount*(this.v.nz+1));
    this.dispatch(pass,'divergence',this.h.cellCount*this.v.nz);
    pass.end();this.device.queue.submit([enc.finish()]);
  }
  uploadState(state:DryState):void{
    this.device.queue.writeBuffer(this.buffers.rho,0,f32(state.rhoD));
    this.device.queue.writeBuffer(this.buffers.rhoTheta,0,f32(state.rhoThetaM));
    this.device.queue.writeBuffer(this.buffers.u,0,f32(state.uEdge));
    this.device.queue.writeBuffer(this.buffers.w,0,f32(state.wInterface));
  }
  private async readF32(buffer:GPUAny,count:number):Promise<Float32Array>{
    const U=(globalThis as any).GPUBufferUsage,M=(globalThis as any).GPUMapMode,bytes=count*4;
    const stage=this.device.createBuffer({size:Math.max(4,bytes),usage:U.COPY_DST|U.MAP_READ});
    const enc=this.device.createCommandEncoder();enc.copyBufferToBuffer(buffer,0,stage,0,bytes);this.device.queue.submit([enc.finish()]);
    await stage.mapAsync(M.READ);const out=new Float32Array(stage.getMappedRange().slice(0));stage.unmap();stage.destroy();return out;
  }
  async downloadState(time=0):Promise<DryState>{
    await this.device.queue.onSubmittedWorkDone();
    const [rho,x,u,w]=await Promise.all([
      this.readF32(this.buffers.rho,this.h.cellCount*this.v.nz),
      this.readF32(this.buffers.rhoTheta,this.h.cellCount*this.v.nz),
      this.readF32(this.buffers.u,this.h.edgeCount*this.v.nz),
      this.readF32(this.buffers.w,this.h.cellCount*(this.v.nz+1)),
    ]);
    return {rhoD:Float64Array.from(rho),rhoThetaM:Float64Array.from(x),uEdge:Float64Array.from(u),wInterface:Float64Array.from(w),time};
  }
  destroy():void{for(const b of Object.values(this.buffers))b.destroy?.();this.paramsBuffer.destroy?.();this.device.destroy?.();}
}
