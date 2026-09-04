import { DRY_AIR, EARTH } from '../core/constants.js';
import { Stage4SlowTendencies } from '../solver/stage4SlowTendenciesCpu.js';
import { DryState } from '../solver/state.js';
import { GpuRotatingDryCore } from './rotatingDryCoreGpu.js';

type GPUAny=any;
function makeEmpty(device:GPUAny,size:number,usage:number,label:string):GPUAny{return device.createBuffer({label,size:Math.max(4,Math.ceil(size/4)*4),usage});}
const COMMON=/* wgsl */`
struct Params{nz:u32,edgeCount:u32,cellCount:u32,_pad0:u32,radius:f32,omega:f32,rd:f32,gamma:f32,pRef:f32,kappa:f32,_pad1:vec2<f32>};
@group(0)@binding(0)var<uniform>P:Params;
fn pressure(x:f32)->f32{return P.pRef*pow(P.rd*x/P.pRef,P.gamma);}
`;
const HPERT=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>edgeCells:array<vec2<u32>>;@group(0)@binding(2)var<storage,read>edgeMetric:array<vec2<f32>>;@group(0)@binding(3)var<storage,read>layerRef:array<f32>;@group(0)@binding(4)var<storage,read>rho:array<f32>;@group(0)@binding(5)var<storage,read>x:array<f32>;@group(0)@binding(6)var<storage,read>u:array<f32>;@group(0)@binding(7)var<storage,read_write>flux:array<vec2<f32>>;
fn dz(k:u32)->f32{return layerRef[k*5u+1u];}fn r0(k:u32)->f32{return layerRef[k*5u+3u];}fn x0(k:u32)->f32{return layerRef[k*5u+4u];}
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.edgeCount*P.nz){return;}let e=q/P.nz;let k=q-e*P.nz;let cc=edgeCells[e];let vel=u[q];let l=cc.x*P.nz+k;let r=cc.y*P.nz+k;let src=select(r,l,vel>=0.0);let A=edgeMetric[e].x*dz(k);flux[q]=vec2<f32>(rho[src]-r0(k),x[src]-x0(k))*vel*A;}
`;
const VPERT=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>cellArea:array<f32>;@group(0)@binding(2)var<storage,read>layerRef:array<f32>;@group(0)@binding(3)var<storage,read>rho:array<f32>;@group(0)@binding(4)var<storage,read>x:array<f32>;@group(0)@binding(5)var<storage,read>w:array<f32>;@group(0)@binding(6)var<storage,read_write>flux:array<vec2<f32>>;
fn r0(k:u32)->f32{return layerRef[k*5u+3u];}fn x0(k:u32)->f32{return layerRef[k*5u+4u];}
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;let count=P.cellCount*(P.nz+1u);if(q>=count){return;}let c=q/(P.nz+1u);let i=q-c*(P.nz+1u);if(i==0u||i==P.nz){flux[q]=vec2<f32>(0.0);return;}let vel=w[q];let k=select(i,i-1u,vel>=0.0);let src=c*P.nz+k;flux[q]=vec2<f32>(rho[src]-r0(k),x[src]-x0(k))*vel*cellArea[c];}
`;
const SDIV=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>cellArea:array<f32>;@group(0)@binding(2)var<storage,read>layerRef:array<f32>;@group(0)@binding(3)var<storage,read>cellEdges:array<vec4<u32>>;@group(0)@binding(4)var<storage,read>cellSigns:array<vec4<i32>>;@group(0)@binding(5)var<storage,read>hFlux:array<vec2<f32>>;@group(0)@binding(6)var<storage,read>vFlux:array<vec2<f32>>;@group(0)@binding(7)var<storage,read_write>out:array<vec2<f32>>;
fn dz(k:u32)->f32{return layerRef[k*5u+1u];}
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.cellCount*P.nz){return;}let c=q/P.nz;let k=q-c*P.nz;let ee=cellEdges[c];let ss=cellSigns[c];var h=vec2<f32>(0.0);for(var s:u32=0u;s<4u;s++){h+=f32(ss[s])*hFlux[ee[s]*P.nz+k];}let vb=c*(P.nz+1u)+k;let t=(-h+vFlux[vb]-vFlux[vb+1u])/(cellArea[c]*dz(k));out[q]=t;}
`;
const CELLWIND=COMMON+/* wgsl */`
struct CellGeom{east:vec4<f32>,north:vec4<f32>};@group(0)@binding(1)var<storage,read>cellGeom:array<CellGeom>;@group(0)@binding(2)var<storage,read>cellEdges:array<vec4<u32>>;@group(0)@binding(3)var<storage,read>recon:array<vec2<f32>>;@group(0)@binding(4)var<storage,read>u:array<f32>;@group(0)@binding(5)var<storage,read_write>cw:array<vec4<f32>>;
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.cellCount*P.nz){return;}let c=q/P.nz;let k=q-c*P.nz;let ee=cellEdges[c];var le=0.0;var ln=0.0;for(var s:u32=0u;s<4u;s++){let co=recon[c*4u+s];let uv=u[ee[s]*P.nz+k];le+=co.x*uv;ln+=co.y*uv;}cw[q]=vec4<f32>(le*cellGeom[c].east.xyz+ln*cellGeom[c].north.xyz,0.0);}
`;
const HADV=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>cellEdges:array<vec4<u32>>;@group(0)@binding(2)var<storage,read>cellSigns:array<vec4<i32>>;@group(0)@binding(3)var<storage,read>edgeCells:array<vec2<u32>>;@group(0)@binding(4)var<storage,read>edgeMetric:array<vec2<f32>>;@group(0)@binding(5)var<storage,read>cellArea:array<f32>;@group(0)@binding(6)var<storage,read>u:array<f32>;@group(0)@binding(7)var<storage,read>cw:array<vec4<f32>>;@group(0)@binding(8)var<storage,read_write>dv:array<vec4<f32>>;
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.cellCount*P.nz){return;}let c=q/P.nz;let k=q-c*P.nz;let ee=cellEdges[c];let ss=cellSigns[c];let cur=cw[q].xyz;var d=vec3<f32>(0.0);for(var s:u32=0u;s<4u;s++){let e=ee[s];let outward=f32(ss[s])*u[e*P.nz+k];if(outward<0.0){let cc=edgeCells[e];let n=select(cc.x,cc.y,cc.x==c);let coef=-outward*edgeMetric[e].x/cellArea[c];d+=coef*(cw[n*P.nz+k].xyz-cur);}}dv[q]=vec4<f32>(d,0.0);}
`;
const VADV=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>layerRef:array<f32>;@group(0)@binding(2)var<storage,read>w:array<f32>;@group(0)@binding(3)var<storage,read>cw:array<vec4<f32>>;@group(0)@binding(4)var<storage,read_write>dv:array<vec4<f32>>;
fn z(k:u32)->f32{return layerRef[k*5u];}
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.cellCount*P.nz){return;}let c=q/P.nz;let k=q-c*P.nz;let wb=w[c*(P.nz+1u)+k];let wt=w[c*(P.nz+1u)+k+1u];let wc=.5*(wb+wt);var d=dv[q].xyz;let cur=cw[q].xyz;if(wc>0.0&&k>0u){d-=wc*(cur-cw[q-1u].xyz)/(z(k)-z(k-1u));}else if(wc<0.0&&k+1u<P.nz){d-=wc*(cw[q+1u].xyz-cur)/(z(k+1u)-z(k));}dv[q]=vec4<f32>(d,0.0);}
`;
const CORIOLIS=COMMON+/* wgsl */`
struct CellGeom{east:vec4<f32>,north:vec4<f32>};@group(0)@binding(1)var<storage,read>cellGeom:array<CellGeom>;@group(0)@binding(2)var<storage,read>cw:array<vec4<f32>>;@group(0)@binding(3)var<storage,read_write>dv:array<vec4<f32>>;
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.cellCount*P.nz){return;}let c=q/P.nz;let g=cellGeom[c];let wind=cw[q].xyz;let ue=dot(wind,g.east.xyz);let vn=dot(wind,g.north.xyz);let f=2.0*P.omega*g.east.w;let d=dv[q].xyz+f*vn*g.east.xyz-f*ue*g.north.xyz;dv[q]=vec4<f32>(d,0.0);}
`;
const PROJECT=COMMON+/* wgsl */`
struct CellGeom{east:vec4<f32>,north:vec4<f32>};
@group(0)@binding(1)var<storage,read>edgeCells:array<vec2<u32>>;@group(0)@binding(2)var<storage,read>edgeGeom:array<vec4<f32>>;@group(0)@binding(3)var<storage,read>cellGeom:array<CellGeom>;@group(0)@binding(4)var<storage,read>dv:array<vec4<f32>>;@group(0)@binding(5)var<storage,read_write>uT:array<f32>;
fn tang(c:u32,d:vec3<f32>)->vec3<f32>{let r=normalize(cross(cellGeom[c].east.xyz,cellGeom[c].north.xyz));return d-r*dot(d,r);}
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.edgeCount*P.nz){return;}let e=q/P.nz;let k=q-e*P.nz;let cc=edgeCells[e];let d=.5*(tang(cc.x,dv[cc.x*P.nz+k].xyz)+tang(cc.y,dv[cc.y*P.nz+k].xyz));uT[q]=dot(d,edgeGeom[e].xyz);}
`;
const WTEND=COMMON+/* wgsl */`
struct SlotMeta{edge:i32,neighbor:i32,sign:i32,_pad:i32};@group(0)@binding(1)var<storage,read>slotMeta:array<SlotMeta>;@group(0)@binding(2)var<storage,read>edgeMetric:array<vec2<f32>>;@group(0)@binding(3)var<storage,read>layerRef:array<f32>;@group(0)@binding(4)var<storage,read>cellArea:array<f32>;@group(0)@binding(5)var<storage,read>u:array<f32>;@group(0)@binding(6)var<storage,read>w:array<f32>;@group(0)@binding(7)var<storage,read_write>wT:array<f32>;
fn dz(k:u32)->f32{return layerRef[k*5u+1u];}
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;let count=P.cellCount*(P.nz+1u);if(q>=count){return;}let c=q/(P.nz+1u);let i=q-c*(P.nz+1u);if(i==0u||i==P.nz){wT[q]=0.0;return;}let wi=w[q];var t=0.0;for(var s:u32=0u;s<4u;s++){let m=slotMeta[c*4u+s];let e=u32(m.edge);let ue=.5*(u[e*P.nz+i-1u]+u[e*P.nz+i]);let outward=f32(m.sign)*ue;if(outward<0.0){let wn=w[u32(m.neighbor)*(P.nz+1u)+i];t-=outward*edgeMetric[e].x*(wn-wi)/cellArea[c];}}if(wi>0.0){t-=wi*(wi-w[q-1u])/dz(i-1u);}else if(wi<0.0){t-=wi*(w[q+1u]-wi)/dz(i);}wT[q]=t;}
`;
const THERMAL=COMMON+/* wgsl */`
struct CellGeom{east:vec4<f32>,north:vec4<f32>};@group(0)@binding(1)var<storage,read>cellGeom:array<CellGeom>;@group(0)@binding(2)var<storage,read>rho:array<f32>;@group(0)@binding(3)var<storage,read>x:array<f32>;@group(0)@binding(4)var<storage,read_write>sT:array<vec2<f32>>;
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.cellCount*P.nz){return;}let c=q/P.nz;let lat=asin(clamp(cellGeom[c].east.w,-1.0,1.0));let p=pressure(x[q]);let sig=max(1e-6,p/P.pRef);let sn=sin(lat);let cs=cos(lat);let teq=max(200.0,(315.0-60.0*sn*sn-10.0*log(sig)*cs*cs)*pow(sig,P.kappa));let theta=x[q]/max(rho[q],1e-12);let thetaEq=teq*pow(P.pRef/p,P.kappa);let sf=max(0.0,(sig-0.7)/0.3);let ka=1.0/(40.0*86400.0);let ks=1.0/(4.0*86400.0);let rate=ka+(ks-ka)*sf*pow(cs,4.0);var t=sT[q];t.y+=rho[q]*rate*(thetaEq-theta);sT[q]=t;}
`;
const DRAG=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>edgeCells:array<vec2<u32>>;@group(0)@binding(2)var<storage,read>x:array<f32>;@group(0)@binding(3)var<storage,read>u:array<f32>;@group(0)@binding(4)var<storage,read_write>uT:array<f32>;
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.edgeCount*P.nz){return;}let e=q/P.nz;let k=q-e*P.nz;let cc=edgeCells[e];let sig=.5*(pressure(x[cc.x*P.nz+k])+pressure(x[cc.y*P.nz+k]))/P.pRef;let rate=(1.0/86400.0)*max(0.0,(sig-0.7)/0.3);uT[q]=uT[q]-rate*u[q];}
`;

/** GPU mirror of computeStage4SlowTendencies(); isolated until agreement passes. */
export class GpuStage4SlowTendencyReference{
  readonly device:GPUAny;readonly scalarT:GPUAny;readonly uT:GPUAny;readonly wT:GPUAny;readonly hFlux:GPUAny;readonly vFlux:GPUAny;
  private readonly params:GPUAny;private readonly cw:GPUAny;private readonly dv:GPUAny;private readonly pipelines:Record<string,GPUAny>={};private readonly groups:Record<string,GPUAny>={};
  private constructor(public readonly core:GpuRotatingDryCore){
    this.device=core.device;const U=(globalThis as any).GPUBufferUsage,storage=U.STORAGE|U.COPY_DST|U.COPY_SRC,uniform=U.UNIFORM|U.COPY_DST,h=core.h,v=core.v;
    this.params=makeEmpty(this.device,48,uniform,'Stage4 slow tendency params');this.scalarT=makeEmpty(this.device,h.cellCount*v.nz*8,storage,'Stage4 slow scalar tendency');this.uT=makeEmpty(this.device,h.edgeCount*v.nz*4,storage,'Stage4 slow u tendency');this.wT=makeEmpty(this.device,h.cellCount*(v.nz+1)*4,storage,'Stage4 slow w tendency');this.hFlux=makeEmpty(this.device,h.edgeCount*v.nz*8,storage,'Stage4 slow horizontal perturbation flux');this.vFlux=makeEmpty(this.device,h.cellCount*(v.nz+1)*8,storage,'Stage4 slow vertical perturbation flux');this.cw=makeEmpty(this.device,h.cellCount*v.nz*16,storage,'Stage4 slow cell wind');this.dv=makeEmpty(this.device,h.cellCount*v.nz*16,storage,'Stage4 slow cell velocity tendency');
    const defs:[string,string,Array<GPUAny>][]=[
      ['hpert',HPERT,[core.core.buffers.edgeCells,core.core.buffers.edgeMetric,core.core.buffers.layerRef,core.core.buffers.rho,core.core.buffers.rhoTheta,core.core.buffers.u,this.hFlux]],
      ['vpert',VPERT,[core.core.buffers.cellArea,core.core.buffers.layerRef,core.core.buffers.rho,core.core.buffers.rhoTheta,core.core.buffers.w,this.vFlux]],
      ['sdiv',SDIV,[core.core.buffers.cellArea,core.core.buffers.layerRef,core.core.buffers.cellEdges,core.core.buffers.cellSigns,this.hFlux,this.vFlux,this.scalarT]],
      ['cellwind',CELLWIND,[core.buffers.cellGeom,core.core.buffers.cellEdges,core.buffers.recon,core.core.buffers.u,this.cw]],
      ['hadv',HADV,[core.core.buffers.cellEdges,core.core.buffers.cellSigns,core.core.buffers.edgeCells,core.core.buffers.edgeMetric,core.core.buffers.cellArea,core.core.buffers.u,this.cw,this.dv]],
      ['vadv',VADV,[core.core.buffers.layerRef,core.core.buffers.w,this.cw,this.dv]],
      ['coriolis',CORIOLIS,[core.buffers.cellGeom,this.cw,this.dv]],
      ['project',PROJECT,[core.core.buffers.edgeCells,core.buffers.edgeGeom,core.buffers.cellGeom,this.dv,this.uT]],
      ['wtend',WTEND,[core.buffers.wAdvMeta,core.core.buffers.edgeMetric,core.core.buffers.layerRef,core.core.buffers.cellArea,core.core.buffers.u,core.core.buffers.w,this.wT]],
      ['thermal',THERMAL,[core.buffers.cellGeom,core.core.buffers.rho,core.core.buffers.rhoTheta,this.scalarT]],
      ['drag',DRAG,[core.core.buffers.edgeCells,core.core.buffers.rhoTheta,core.core.buffers.u,this.uT]],
    ];
    for(const [name,code,buffers] of defs){const p=this.device.createComputePipeline({label:`Stage4 slow ${name}`,layout:'auto',compute:{module:this.device.createShaderModule({label:`Stage4 slow ${name} shader`,code}),entryPoint:'main'}});this.pipelines[name]=p;this.groups[name]=this.device.createBindGroup({layout:p.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.params}},...buffers.map((buffer,i)=>({binding:i+1,resource:{buffer}}))]});}
  }
  static create(core:GpuRotatingDryCore):GpuStage4SlowTendencyReference{return new GpuStage4SlowTendencyReference(core);}
  uploadPredictor(state:DryState):void{this.core.core.uploadState(state);}
  prepare():void{const ab=new ArrayBuffer(48),u=new Uint32Array(ab),f=new Float32Array(ab);u[0]=this.core.v.nz;u[1]=this.core.h.edgeCount;u[2]=this.core.h.cellCount;f[4]=EARTH.radius;f[5]=EARTH.omega;f[6]=DRY_AIR.rd;f[7]=DRY_AIR.gamma;f[8]=DRY_AIR.pRef;f[9]=DRY_AIR.kappa;this.device.queue.writeBuffer(this.params,0,ab);}
  private dispatch(p:GPUAny,name:string,count:number):void{p.setPipeline(this.pipelines[name]);p.setBindGroup(0,this.groups[name]);p.dispatchWorkgroups(Math.ceil(count/128));}
  encode(enc:GPUAny,heldSuarez=true,momentum=true,coriolis=true):void{
    const h=this.core.h,v=this.core.v;enc.clearBuffer(this.scalarT);enc.clearBuffer(this.uT);enc.clearBuffer(this.wT);enc.clearBuffer(this.dv);
    let p=enc.beginComputePass();this.dispatch(p,'hpert',h.edgeCount*v.nz);this.dispatch(p,'vpert',h.cellCount*(v.nz+1));this.dispatch(p,'sdiv',h.cellCount*v.nz);p.end();
    if(momentum||coriolis){p=enc.beginComputePass();this.dispatch(p,'cellwind',h.cellCount*v.nz);if(momentum){this.dispatch(p,'hadv',h.cellCount*v.nz);this.dispatch(p,'vadv',h.cellCount*v.nz);}if(coriolis)this.dispatch(p,'coriolis',h.cellCount*v.nz);this.dispatch(p,'project',h.edgeCount*v.nz);p.end();}
    if(momentum){p=enc.beginComputePass();this.dispatch(p,'wtend',h.cellCount*(v.nz+1));p.end();}
    if(heldSuarez){p=enc.beginComputePass();this.dispatch(p,'thermal',h.cellCount*v.nz);this.dispatch(p,'drag',h.edgeCount*v.nz);p.end();}
  }
  compute(heldSuarez=true,momentum=true,coriolis=true):void{this.prepare();const enc=this.device.createCommandEncoder({label:'Stage4 frozen slow tendency reference'});this.encode(enc,heldSuarez,momentum,coriolis);this.device.queue.submit([enc.finish()]);}
  private async read(buffer:GPUAny,count:number):Promise<Float32Array>{const U=(globalThis as any).GPUBufferUsage,M=(globalThis as any).GPUMapMode,bytes=count*4,stage=this.device.createBuffer({size:Math.max(4,bytes),usage:U.COPY_DST|U.MAP_READ}),enc=this.device.createCommandEncoder();enc.copyBufferToBuffer(buffer,0,stage,0,bytes);this.device.queue.submit([enc.finish()]);await stage.mapAsync(M.READ);const out=new Float32Array(stage.getMappedRange().slice(0));stage.unmap();stage.destroy();return out;}
  async download():Promise<Stage4SlowTendencies>{
    await this.device.queue.onSubmittedWorkDone();const h=this.core.h,v=this.core.v,[st,u,w,hf,vf]=await Promise.all([this.read(this.scalarT,h.cellCount*v.nz*2),this.read(this.uT,h.edgeCount*v.nz),this.read(this.wT,h.cellCount*(v.nz+1)),this.read(this.hFlux,h.edgeCount*v.nz*2),this.read(this.vFlux,h.cellCount*(v.nz+1)*2)]),rho=new Float64Array(h.cellCount*v.nz),x=new Float64Array(rho.length),hm=new Float64Array(h.edgeCount*v.nz),vm=new Float64Array(h.cellCount*(v.nz+1));
    for(let q=0;q<rho.length;q++){rho[q]=st[q*2]!;x[q]=st[q*2+1]!;}for(let q=0;q<hm.length;q++)hm[q]=hf[q*2]!;for(let q=0;q<vm.length;q++)vm[q]=vf[q*2]!;
    return{rhoD:rho,rhoThetaM:x,uEdge:Float64Array.from(u),wInterface:Float64Array.from(w),hMassFlux:hm,vMassFlux:vm};
  }
  destroy():void{this.params.destroy?.();this.scalarT.destroy?.();this.uT.destroy?.();this.wT.destroy?.();this.hFlux.destroy?.();this.vFlux.destroy?.();this.cw.destroy?.();this.dv.destroy?.();}
}
