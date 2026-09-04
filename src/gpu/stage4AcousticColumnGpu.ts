import { DRY_AIR, EARTH } from '../core/constants.js';
import { buildModelTopSpongeRates } from '../physics/modelTopSponge.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { Stage4FrozenRhs } from '../solver/stage4Rk3SplitCpu.js';
import { DryState, cell3DIndex, w3DIndex } from '../solver/state.js';
import { GpuDryCorePrototype } from './dryCoreGpu.js';

type GPUAny=any;
function makeEmpty(device:GPUAny,size:number,usage:number,label:string):GPUAny{return device.createBuffer({label,size:Math.max(4,Math.ceil(size/4)*4),usage});}

const VERTICAL_ACOUSTIC=/* wgsl */`
const MAX_NZ:u32=128u;
struct Params{
  nz:u32,cellCount:u32,_pad0:vec2<u32>,
  dt:f32,gravity:f32,rd:f32,gamma:f32,
  pRef:f32,offCenter:f32,_pad1:vec2<f32>,
};
@group(0)@binding(0)var<uniform>P:Params;
// layerRef = [zCenter,dz,p0,rho0,rhoTheta0]
@group(0)@binding(1)var<storage,read>layerRef:array<f32>;
// interfaceRef = [rho0,rhoTheta0]
@group(0)@binding(2)var<storage,read>interfaceRef:array<f32>;
// cellA = [predictor rho, predictor X, acoustic rho, acoustic X]
@group(0)@binding(3)var<storage,read_write>cellA:array<vec4<f32>>;
// cellB = [frozen rho RHS, frozen X RHS, unused, unused]
@group(0)@binding(4)var<storage,read>cellB:array<vec4<f32>>;
// iface = [predictor w, acoustic w, frozen w RHS, Rayleigh rate]
@group(0)@binding(5)var<storage,read_write>iface:array<vec4<f32>>;
// reference flux per unit area = [rho0*w_weighted, X0*w_weighted]
@group(0)@binding(6)var<storage,read_write>refFlux:array<vec2<f32>>;
fn lz(k:u32)->f32{return layerRef[k*5u];}
fn ldz(k:u32)->f32{return layerRef[k*5u+1u];}
fn lp(k:u32)->f32{return layerRef[k*5u+2u];}
fn lrho(k:u32)->f32{return layerRef[k*5u+3u];}
fn lx(k:u32)->f32{return layerRef[k*5u+4u];}
fn irho(i:u32)->f32{return interfaceRef[i*2u];}
fn ix(i:u32)->f32{return interfaceRef[i*2u+1u];}
fn pressure(x:f32)->f32{return P.pRef*pow(P.rd*x/P.pRef,P.gamma);}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){
  let c=gid.x;if(c>=P.cellCount||P.nz>MAX_NZ||P.nz<2u){return;}
  let nz=P.nz;let n=nz-1u;let theta=.5*(1.0+clamp(P.offCenter,0.0,.999));let oldWeight=1.0-theta;
  var drOld:array<f32,128>;var dxOld:array<f32,128>;var dwOld:array<f32,129>;
  var pPred:array<f32,128>;var dpdX:array<f32,128>;var drBase:array<f32,128>;var dxBase:array<f32,128>;
  var lOld:array<f32,129>;var lo:array<f32,127>;var di:array<f32,127>;var up:array<f32,127>;var rhs:array<f32,127>;
  var cp:array<f32,127>;var dp:array<f32,127>;var sol:array<f32,127>;var dwNew:array<f32,129>;
  let baseCell=c*nz;let baseW=c*(nz+1u);
  for(var k:u32=0u;k<nz;k++){
    let a=cellA[baseCell+k];drOld[k]=a.z-a.x*0.0-a.x*0.0-a.x+a.x; // overwritten below; keeps parser scalarized
    drOld[k]=a.z-a.x;dxOld[k]=a.w-a.y;pPred[k]=pressure(a.y);dpdX[k]=P.gamma*pPred[k]/max(a.y,1e-12);
  }
  for(var i:u32=0u;i<=nz;i++){let f=iface[baseW+i];dwOld[i]=f.y-f.x;dwNew[i]=0.0;}
  for(var k:u32=0u;k<nz;k++){
    let rb=cellB[baseCell+k];
    let divR=(irho(k+1u)*dwOld[k+1u]-irho(k)*dwOld[k])/ldz(k);
    let divX=(ix(k+1u)*dwOld[k+1u]-ix(k)*dwOld[k])/ldz(k);
    drBase[k]=drOld[k]+P.dt*rb.x-P.dt*oldWeight*divR;
    dxBase[k]=dxOld[k]+P.dt*rb.y-P.dt*oldWeight*divX;
  }
  for(var i:u32=1u;i<nz;i++){
    let l=i-1u;let u=i;let al=cellA[baseCell+l];let au=cellA[baseCell+u];
    let rr=.5*(al.x+au.x);let r0=.5*(lrho(l)+lrho(u));let den=max(.5*(r0+rr),1e-12);let dzc=lz(u)-lz(l);
    let dpPrime=(pPred[u]-lp(u))-(pPred[l]-lp(l));
    let cr=dpPrime*.25/(den*den*dzc)-P.gravity*r0*.5/max(rr*rr,1e-24);
    let cxl=dpdX[l]/(den*dzc);let cxu=-dpdX[u]/(den*dzc);
    lOld[i]=cr*(drOld[l]+drOld[u])+cxl*dxOld[l]+cxu*dxOld[u];
  }
  for(var ii:u32=0u;ii<n;ii++){
    let i=ii+1u;let l=i-1u;let u=i;let al=cellA[baseCell+l];let au=cellA[baseCell+u];
    let rr=.5*(al.x+au.x);let r0=.5*(lrho(l)+lrho(u));let den=max(.5*(r0+rr),1e-12);let dzc=lz(u)-lz(l);
    let dpPrime=(pPred[u]-lp(u))-(pPred[l]-lp(l));
    let cr=dpPrime*.25/(den*den*dzc)-P.gravity*r0*.5/max(rr*rr,1e-24);
    let cxl=dpdX[l]/(den*dzc);let cxu=-dpdX[u]/(den*dzc);
    let l0=cr*(drBase[l]+drBase[u])+cxl*dxBase[l]+cxu*dxBase[u];
    let alm=cr*(P.dt*theta*irho(i-1u)/ldz(l))+cxl*(P.dt*theta*ix(i-1u)/ldz(l));
    let ali=cr*(-P.dt*theta*irho(i)/ldz(l)+P.dt*theta*irho(i)/ldz(u))+cxl*(-P.dt*theta*ix(i)/ldz(l))+cxu*(P.dt*theta*ix(i)/ldz(u));
    let alp=cr*(-P.dt*theta*irho(i+1u)/ldz(u))+cxu*(-P.dt*theta*ix(i+1u)/ldz(u));
    let f=iface[baseW+i];let rate=max(0.0,f.w);
    lo[ii]=-P.dt*theta*alm;di[ii]=1.0+P.dt*rate-P.dt*theta*ali;up[ii]=-P.dt*theta*alp;
    rhs[ii]=dwOld[i]+P.dt*f.z+P.dt*oldWeight*lOld[i]+P.dt*theta*l0-P.dt*rate*f.x;
  }
  lo[0u]=0.0;up[n-1u]=0.0;
  var denom=di[0u];cp[0u]=select(up[0u]/denom,0.0,n==1u);dp[0u]=rhs[0u]/denom;
  for(var j:u32=1u;j<n;j++){denom=di[j]-lo[j]*cp[j-1u];cp[j]=select(up[j]/denom,0.0,j==n-1u);dp[j]=(rhs[j]-lo[j]*dp[j-1u])/denom;}
  sol[n-1u]=dp[n-1u];var jj:i32=i32(n)-2;loop{if(jj<0){break;}let j=u32(jj);sol[j]=dp[j]-cp[j]*sol[j+1u];jj=jj-1;}
  for(var i:u32=1u;i<nz;i++){dwNew[i]=sol[i-1u];}
  for(var k:u32=0u;k<nz;k++){
    let dr=drBase[k]-P.dt*theta*(irho(k+1u)*dwNew[k+1u]-irho(k)*dwNew[k])/ldz(k);
    let dx=dxBase[k]-P.dt*theta*(ix(k+1u)*dwNew[k+1u]-ix(k)*dwNew[k])/ldz(k);
    var a=cellA[baseCell+k];a.z=a.x+dr;a.w=a.y+dx;cellA[baseCell+k]=a;
  }
  for(var i:u32=0u;i<=nz;i++){
    var f=iface[baseW+i];let newW=select(f.x+dwNew[i],0.0,i==0u||i==nz);let weighted=f.x+oldWeight*dwOld[i]+theta*dwNew[i];
    f.y=newW;iface[baseW+i]=f;refFlux[baseW+i]=vec2<f32>(irho(i)*weighted,ix(i)*weighted);
  }
}`;

/**
 * GPU mirror of the CPU predictor-relative vertical acoustic/gravity column.
 * It is intentionally isolated from the production Stage-4 path until direct
 * GPU/CPU agreement is established.
 */
export class GpuStage4AcousticColumnReference{
  readonly device:GPUAny;readonly cellA:GPUAny;readonly cellB:GPUAny;readonly iface:GPUAny;readonly refFlux:GPUAny;
  private readonly params:GPUAny;private readonly pipeline:GPUAny;private readonly group:GPUAny;
  private constructor(public readonly core:GpuDryCorePrototype){
    this.device=core.device;const U=(globalThis as any).GPUBufferUsage,storage=U.STORAGE|U.COPY_DST|U.COPY_SRC,uniform=U.UNIFORM|U.COPY_DST,h=core.h,v=core.v;
    this.params=makeEmpty(this.device,48,uniform,'RK3 acoustic column params');
    this.cellA=makeEmpty(this.device,h.cellCount*v.nz*16,storage,'RK3 acoustic cell predictor/state');
    this.cellB=makeEmpty(this.device,h.cellCount*v.nz*16,storage,'RK3 acoustic frozen scalar RHS');
    this.iface=makeEmpty(this.device,h.cellCount*(v.nz+1)*16,storage,'RK3 acoustic interface predictor/state/RHS/rayleigh');
    this.refFlux=makeEmpty(this.device,h.cellCount*(v.nz+1)*8,storage,'RK3 acoustic reference flux');
    this.pipeline=this.device.createComputePipeline({label:'Stage4 predictor-relative vertical acoustic',layout:'auto',compute:{module:this.device.createShaderModule({label:'Stage4 predictor-relative vertical acoustic shader',code:VERTICAL_ACOUSTIC}),entryPoint:'main'}});
    this.group=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:this.params}},{binding:1,resource:{buffer:core.buffers.layerRef}},{binding:2,resource:{buffer:core.buffers.interfaceRef}},{binding:3,resource:{buffer:this.cellA}},{binding:4,resource:{buffer:this.cellB}},{binding:5,resource:{buffer:this.iface}},{binding:6,resource:{buffer:this.refFlux}},
    ]});
  }
  static create(core:GpuDryCorePrototype):GpuStage4AcousticColumnReference{return new GpuStage4AcousticColumnReference(core);}
  upload(predictor:DryState,acoustic:DryState,frozen:Stage4FrozenRhs):void{
    const h=this.core.h,v=this.core.v;if(predictor.rhoD.length!==h.cellCount*v.nz||acoustic.rhoD.length!==predictor.rhoD.length)throw new Error('GPU acoustic upload cell shape mismatch');
    const a=new Float32Array(h.cellCount*v.nz*4),b=new Float32Array(a.length),rates=buildModelTopSpongeRates(v),f=new Float32Array(h.cellCount*(v.nz+1)*4);
    for(let c=0;c<h.cellCount;c++)for(let k=0;k<v.nz;k++){const q=cell3DIndex(c,k,v.nz),o=q*4;a[o]=predictor.rhoD[q]!;a[o+1]=predictor.rhoThetaM[q]!;a[o+2]=acoustic.rhoD[q]!;a[o+3]=acoustic.rhoThetaM[q]!;b[o]=frozen.rhoD[q]!;b[o+1]=frozen.rhoThetaM[q]!;}
    for(let c=0;c<h.cellCount;c++)for(let i=0;i<=v.nz;i++){const q=w3DIndex(c,i,v.nz),o=q*4;f[o]=predictor.wInterface[q]!;f[o+1]=acoustic.wInterface[q]!;f[o+2]=frozen.wInterface[q]!;f[o+3]=rates[i]!;}
    this.device.queue.writeBuffer(this.cellA,0,a);this.device.queue.writeBuffer(this.cellB,0,b);this.device.queue.writeBuffer(this.iface,0,f);
  }
  private writeParams(dt:number,offCenter:number):void{
    const ab=new ArrayBuffer(48),u=new Uint32Array(ab),f=new Float32Array(ab);u[0]=this.core.v.nz;u[1]=this.core.h.cellCount;f[4]=dt;f[5]=EARTH.gravity;f[6]=DRY_AIR.rd;f[7]=DRY_AIR.gamma;f[8]=DRY_AIR.pRef;f[9]=offCenter;this.device.queue.writeBuffer(this.params,0,ab);
  }
  encode(enc:GPUAny,dt:number,offCenter:number):void{this.writeParams(dt,offCenter);const p=enc.beginComputePass();p.setPipeline(this.pipeline);p.setBindGroup(0,this.group);p.dispatchWorkgroups(this.core.h.cellCount);p.end();}
  step(dt:number,offCenter:number):void{const enc=this.device.createCommandEncoder({label:'Stage4 predictor-relative vertical acoustic step'});this.encode(enc,dt,offCenter);this.device.queue.submit([enc.finish()]);}
  private async read(buffer:GPUAny,count:number):Promise<Float32Array>{const U=(globalThis as any).GPUBufferUsage,M=(globalThis as any).GPUMapMode,bytes=count*4,stage=this.device.createBuffer({size:Math.max(4,bytes),usage:U.COPY_DST|U.MAP_READ}),enc=this.device.createCommandEncoder();enc.copyBufferToBuffer(buffer,0,stage,0,bytes);this.device.queue.submit([enc.finish()]);await stage.mapAsync(M.READ);const out=new Float32Array(stage.getMappedRange().slice(0));stage.unmap();stage.destroy();return out;}
  async downloadAcousticState(time=0):Promise<DryState>{
    await this.device.queue.onSubmittedWorkDone();const h=this.core.h,v=this.core.v,[a,f]=await Promise.all([this.read(this.cellA,h.cellCount*v.nz*4),this.read(this.iface,h.cellCount*(v.nz+1)*4)]),rho=new Float64Array(h.cellCount*v.nz),x=new Float64Array(rho.length),w=new Float64Array(h.cellCount*(v.nz+1));
    for(let q=0;q<rho.length;q++){rho[q]=a[q*4+2]!;x[q]=a[q*4+3]!;}for(let q=0;q<w.length;q++)w[q]=f[q*4+1]!;
    return{rhoD:rho,rhoThetaM:x,uEdge:new Float64Array(h.edgeCount*v.nz),wInterface:w,time};
  }
  destroy():void{this.params.destroy?.();this.cellA.destroy?.();this.cellB.destroy?.();this.iface.destroy?.();this.refFlux.destroy?.();}
}
