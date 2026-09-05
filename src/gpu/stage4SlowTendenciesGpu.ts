import { DRY_AIR, EARTH } from '../core/constants.js';
import { buildRotationGeometry, type RotationGeometry } from '../physics/rotation.js';
import { Stage4SlowTendencies } from '../solver/stage4SlowTendenciesCpu.js';
import { DryState } from '../solver/state.js';
import { buildStage4GradientStencilData } from './stage4GradientStencil.js';
import { GpuRotatingDryCore } from './rotatingDryCoreGpu.js';

type GPUAny=any;
function makeBuffer(device:GPUAny,data:ArrayBufferView,usage:number,label:string):GPUAny{const size=Math.max(4,Math.ceil(data.byteLength/4)*4),b=device.createBuffer({label,size,usage,mappedAtCreation:true});new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));b.unmap();return b;}
function makeEmpty(device:GPUAny,size:number,usage:number,label:string):GPUAny{return device.createBuffer({label,size:Math.max(4,Math.ceil(size/4)*4),usage});}
function clampDot(x:number):number{return Math.max(-1,Math.min(1,x));}
function faceLocalDelta(h:GpuRotatingDryCore['h'],g:RotationGeometry,c:number,eid:number):readonly[number,number]{
  const o=c*3,m=h.edges[eid]!.midpoint,rx=g.radial[o]!,ry=g.radial[o+1]!,rz=g.radial[o+2]!,mu=clampDot(rx*m[0]+ry*m[1]+rz*m[2]),ang=Math.acos(mu),tx=m[0]-mu*rx,ty=m[1]-mu*ry,tz=m[2]-mu*rz,tm=Math.hypot(tx,ty,tz);
  if(!(ang>0)||!(tm>0))return[0,0];
  const scale=EARTH.radius*ang/tm,dx=tx*scale,dy=ty*scale,dz=tz*scale;
  return[dx*g.east[o]!+dy*g.east[o+1]!+dz*g.east[o+2]!,dx*g.north[o]!+dy*g.north[o+1]!+dz*g.north[o+2]!];
}
/** Per cell-slot data packed as WGSL HadvSlot = 3 vec4 = 48 bytes. */
function buildHadvSlots(core:GpuRotatingDryCore,g:RotationGeometry):Uint8Array{
  const h=core.h,ab=new ArrayBuffer(h.cellCount*4*48),ii=new Int32Array(ab),ff=new Float32Array(ab);
  for(let c=0;c<h.cellCount;c++)for(let slot=0;slot<4;slot++){
    const eid=h.cellEdges[c*4+slot]!,edge=h.edges[eid]!,nb=edge.leftCell===c?edge.rightCell:edge.leftCell,base=(c*4+slot)*12,self=faceLocalDelta(h,g,c,eid),other=faceLocalDelta(h,g,nb,eid);
    ii[base]=eid;ii[base+1]=nb;ii[base+2]=h.cellEdgeSigns[c*4+slot]!;ii[base+3]=0;
    ff[base+4]=self[0];ff[base+5]=self[1];ff[base+6]=other[0];ff[base+7]=other[1];
    ff[base+8]=edge.midpoint[0];ff[base+9]=edge.midpoint[1];ff[base+10]=edge.midpoint[2];ff[base+11]=0;
  }
  return new Uint8Array(ab);
}
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
const HGRADLIM=COMMON+/* wgsl */`
struct HadvSlot{info:vec4<i32>,face:vec4<f32>,mid:vec4<f32>};struct GradPair{east:vec4<f32>,north:vec4<f32>};
@group(0)@binding(1)var<storage,read>slots:array<HadvSlot>;@group(0)@binding(2)var<storage,read>weights:array<GradPair>;@group(0)@binding(3)var<storage,read>cw:array<vec4<f32>>;@group(0)@binding(4)var<storage,read_write>grad:array<GradPair>;
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){
 let q=gid.x;if(q>=P.cellCount*P.nz){return;}let c=q/P.nz;let k=q-c*P.nz;let base=cw[q].xyz;var lo=base;var hi=base;var ge=vec3<f32>(0.0);var gn=vec3<f32>(0.0);let w=weights[c];
 for(var s:u32=0u;s<4u;s++){let nb=u32(slots[c*4u+s].info.y);let nv=cw[nb*P.nz+k].xyz;lo=min(lo,nv);hi=max(hi,nv);let d=nv-base;ge+=w.east[s]*d;gn+=w.north[s]*d;}
 var phi=1.0;for(var s:u32=0u;s<4u;s++){let fd=slots[c*4u+s].face.xy;let d=ge*fd.x+gn*fd.y;for(var j:u32=0u;j<3u;j++){let x=d[j];if(x>1e-15){phi=min(phi,(hi[j]-base[j])/x);}else if(x< -1e-15){phi=min(phi,(lo[j]-base[j])/x);}}}
 phi=clamp(phi,0.0,1.0);grad[q].east=vec4<f32>(phi*ge,0.0);grad[q].north=vec4<f32>(phi*gn,0.0);
}
`;
const HADV=COMMON+/* wgsl */`
struct HadvSlot{info:vec4<i32>,face:vec4<f32>,mid:vec4<f32>};struct GradPair{east:vec4<f32>,north:vec4<f32>};
@group(0)@binding(1)var<storage,read>slots:array<HadvSlot>;@group(0)@binding(2)var<storage,read>edgeMetric:array<vec2<f32>>;@group(0)@binding(3)var<storage,read>rho:array<f32>;@group(0)@binding(4)var<storage,read>u:array<f32>;@group(0)@binding(5)var<storage,read>cw:array<vec4<f32>>;@group(0)@binding(6)var<storage,read>grad:array<GradPair>;@group(0)@binding(7)var<storage,read>cellArea:array<f32>;@group(0)@binding(8)var<storage,read_write>dv:array<vec4<f32>>;
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){
 let q=gid.x;if(q>=P.cellCount*P.nz){return;}let c=q/P.nz;let k=q-c*P.nz;let cur=cw[q].xyz;var sumM=0.0;var sumQ=vec3<f32>(0.0);
 for(var s:u32=0u;s<4u;s++){let sg=slots[c*4u+s];let e=u32(sg.info.x);let nb=u32(sg.info.y);let sign=f32(sg.info.z);let ue=u[e*P.nz+k];let selfUp=ue*sign>=0.0;let up=select(nb,c,selfUp);let fd=select(sg.face.zw,sg.face.xy,selfUp);let gg=grad[up*P.nz+k];var qf=cw[up*P.nz+k].xyz+gg.east.xyz*fd.x+gg.north.xyz*fd.y;let m=sg.mid.xyz;qf-=m*dot(qf,m);let M=sign*rho[up*P.nz+k]*ue*edgeMetric[e].x;sumM+=M;sumQ+=M*qf;}
 let den=max(rho[q]*cellArea[c],1e-12);let a=(-sumQ+cur*sumM)/den;dv[q]=vec4<f32>(a,0.0);
}
`;
const VADV=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>cellArea:array<f32>;@group(0)@binding(2)var<storage,read>layerRef:array<f32>;@group(0)@binding(3)var<storage,read>rhoInterfaceRef:array<f32>;@group(0)@binding(4)var<storage,read>rho:array<f32>;@group(0)@binding(5)var<storage,read>w:array<f32>;@group(0)@binding(6)var<storage,read>cw:array<vec4<f32>>;@group(0)@binding(7)var<storage,read_write>dv:array<vec4<f32>>;
fn dz(k:u32)->f32{return layerRef[k*5u+1u];}fn r0(k:u32)->f32{return layerRef[k*5u+3u];}
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){
 let q=gid.x;if(q>=P.cellCount*P.nz){return;}let c=q/P.nz;let k=q-c*P.nz;let area=cellArea[c];let cur=cw[q].xyz;var sumM=0.0;var sumQ=vec3<f32>(0.0);
 let il=k;if(il>0u){let wl=w[c*(P.nz+1u)+il];let sk=select(il,il-1u,wl>=0.0);let src=c*P.nz+sk;let m=(rhoInterfaceRef[il]+rho[src]-r0(sk))*wl*area;let outward=-m;sumM+=outward;sumQ+=outward*cw[c*P.nz+sk].xyz;}
 let iu=k+1u;if(iu<P.nz){let wu=w[c*(P.nz+1u)+iu];let sk=select(iu,iu-1u,wu>=0.0);let src=c*P.nz+sk;let m=(rhoInterfaceRef[iu]+rho[src]-r0(sk))*wu*area;let outward=m;sumM+=outward;sumQ+=outward*cw[c*P.nz+sk].xyz;}
 let den=max(rho[q]*area*dz(k),1e-12);let a=(-sumQ+cur*sumM)/den;dv[q]=vec4<f32>(dv[q].xyz+a,0.0);
}
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
  private readonly params:GPUAny;private readonly cw:GPUAny;private readonly dv:GPUAny;private readonly hGrad:GPUAny;private readonly gradWeights:GPUAny;private readonly hadvSlots:GPUAny;private readonly rhoInterfaceRef:GPUAny;private readonly pipelines:Record<string,GPUAny>={};private readonly groups:Record<string,GPUAny>={};
  private constructor(public readonly core:GpuRotatingDryCore){
    this.device=core.device;const U=(globalThis as any).GPUBufferUsage,storage=U.STORAGE|U.COPY_DST|U.COPY_SRC,uniform=U.UNIFORM|U.COPY_DST,h=core.h,v=core.v,g=buildRotationGeometry(h),stencil=buildStage4GradientStencilData(h);
    this.params=makeEmpty(this.device,48,uniform,'Stage4 slow tendency params');this.scalarT=makeEmpty(this.device,h.cellCount*v.nz*8,storage,'Stage4 slow scalar tendency');this.uT=makeEmpty(this.device,h.edgeCount*v.nz*4,storage,'Stage4 slow u tendency');this.wT=makeEmpty(this.device,h.cellCount*(v.nz+1)*4,storage,'Stage4 slow w tendency');this.hFlux=makeEmpty(this.device,h.edgeCount*v.nz*8,storage,'Stage4 slow horizontal perturbation flux');this.vFlux=makeEmpty(this.device,h.cellCount*(v.nz+1)*8,storage,'Stage4 slow vertical perturbation flux');this.cw=makeEmpty(this.device,h.cellCount*v.nz*16,storage,'Stage4 slow cell wind');this.dv=makeEmpty(this.device,h.cellCount*v.nz*16,storage,'Stage4 slow cell velocity tendency');this.hGrad=makeEmpty(this.device,h.cellCount*v.nz*32,storage,'Stage4 slow limited horizontal wind gradients');this.gradWeights=makeBuffer(this.device,stencil.weights,storage,'Stage4 slow horizontal gradient weights');this.hadvSlots=makeBuffer(this.device,buildHadvSlots(core,g),storage,'Stage4 slow horizontal advection slots');this.rhoInterfaceRef=makeBuffer(this.device,Float32Array.from(core.ref.rhoInterface),storage,'Stage4 slow interface reference density');
    const defs:[string,string,Array<GPUAny>][]=[
      ['hpert',HPERT,[core.core.buffers.edgeCells,core.core.buffers.edgeMetric,core.core.buffers.layerRef,core.core.buffers.rho,core.core.buffers.rhoTheta,core.core.buffers.u,this.hFlux]],
      ['vpert',VPERT,[core.core.buffers.cellArea,core.core.buffers.layerRef,core.core.buffers.rho,core.core.buffers.rhoTheta,core.core.buffers.w,this.vFlux]],
      ['sdiv',SDIV,[core.core.buffers.cellArea,core.core.buffers.layerRef,core.core.buffers.cellEdges,core.core.buffers.cellSigns,this.hFlux,this.vFlux,this.scalarT]],
      ['cellwind',CELLWIND,[core.buffers.cellGeom,core.core.buffers.cellEdges,core.buffers.recon,core.core.buffers.u,this.cw]],
      ['hgradlim',HGRADLIM,[this.hadvSlots,this.gradWeights,this.cw,this.hGrad]],
      ['hadv',HADV,[this.hadvSlots,core.core.buffers.edgeMetric,core.core.buffers.rho,core.core.buffers.u,this.cw,this.hGrad,core.core.buffers.cellArea,this.dv]],
      ['vadv',VADV,[core.core.buffers.cellArea,core.core.buffers.layerRef,this.rhoInterfaceRef,core.core.buffers.rho,core.core.buffers.w,this.cw,this.dv]],
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
  clear(enc:GPUAny):void{enc.clearBuffer(this.scalarT);enc.clearBuffer(this.uT);enc.clearBuffer(this.wT);enc.clearBuffer(this.dv);}
  encodeIntoPass(p:GPUAny,heldSuarez=true,momentum=true,coriolis=true):void{
    const h=this.core.h,v=this.core.v;
    this.dispatch(p,'hpert',h.edgeCount*v.nz);this.dispatch(p,'vpert',h.cellCount*(v.nz+1));this.dispatch(p,'sdiv',h.cellCount*v.nz);
    if(momentum||coriolis){this.dispatch(p,'cellwind',h.cellCount*v.nz);if(momentum){this.dispatch(p,'hgradlim',h.cellCount*v.nz);this.dispatch(p,'hadv',h.cellCount*v.nz);this.dispatch(p,'vadv',h.cellCount*v.nz);}if(coriolis)this.dispatch(p,'coriolis',h.cellCount*v.nz);this.dispatch(p,'project',h.edgeCount*v.nz);}
    if(momentum)this.dispatch(p,'wtend',h.cellCount*(v.nz+1));
    if(heldSuarez){this.dispatch(p,'thermal',h.cellCount*v.nz);this.dispatch(p,'drag',h.edgeCount*v.nz);}
  }
  encode(enc:GPUAny,heldSuarez=true,momentum=true,coriolis=true):void{this.clear(enc);const p=enc.beginComputePass();this.encodeIntoPass(p,heldSuarez,momentum,coriolis);p.end();}
  compute(heldSuarez=true,momentum=true,coriolis=true):void{this.prepare();const enc=this.device.createCommandEncoder({label:'Stage4 frozen slow tendency reference'});this.encode(enc,heldSuarez,momentum,coriolis);this.device.queue.submit([enc.finish()]);}
  private async read(buffer:GPUAny,count:number):Promise<Float32Array>{const U=(globalThis as any).GPUBufferUsage,M=(globalThis as any).GPUMapMode,bytes=count*4,stage=this.device.createBuffer({size:Math.max(4,bytes),usage:U.COPY_DST|U.MAP_READ}),enc=this.device.createCommandEncoder();enc.copyBufferToBuffer(buffer,0,stage,0,bytes);this.device.queue.submit([enc.finish()]);await stage.mapAsync(M.READ);const out=new Float32Array(stage.getMappedRange().slice(0));stage.unmap();stage.destroy();return out;}
  async download():Promise<Stage4SlowTendencies>{
    await this.device.queue.onSubmittedWorkDone();const h=this.core.h,v=this.core.v,[st,u,w,hf,vf]=await Promise.all([this.read(this.scalarT,h.cellCount*v.nz*2),this.read(this.uT,h.edgeCount*v.nz),this.read(this.wT,h.cellCount*(v.nz+1)),this.read(this.hFlux,h.edgeCount*v.nz*2),this.read(this.vFlux,h.cellCount*(v.nz+1)*2)]),rho=new Float64Array(h.cellCount*v.nz),x=new Float64Array(rho.length),hm=new Float64Array(h.edgeCount*v.nz),vm=new Float64Array(h.cellCount*(v.nz+1));
    for(let q=0;q<rho.length;q++){rho[q]=st[q*2]!;x[q]=st[q*2+1]!;}for(let q=0;q<hm.length;q++)hm[q]=hf[q*2]!;for(let q=0;q<vm.length;q++)vm[q]=vf[q*2]!;
    return{rhoD:rho,rhoThetaM:x,uEdge:Float64Array.from(u),wInterface:Float64Array.from(w),hMassFlux:hm,vMassFlux:vm};
  }
  destroy():void{this.params.destroy?.();this.scalarT.destroy?.();this.uT.destroy?.();this.wT.destroy?.();this.hFlux.destroy?.();this.vFlux.destroy?.();this.cw.destroy?.();this.dv.destroy?.();this.hGrad.destroy?.();this.gradWeights.destroy?.();this.hadvSlots.destroy?.();this.rhoInterfaceRef.destroy?.();}
}
