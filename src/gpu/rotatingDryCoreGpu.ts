import { DRY_AIR, EARTH } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { buildRotationGeometry } from '../physics/rotation.js';
import { STAGE4_HEVI_OFFCENTERING } from '../solver/stage4Config.js';
import { DryState } from '../solver/state.js';
import { GpuDryCorePrototype } from './dryCoreGpu.js';

type GPUAny=any;
function makeBuffer(device:GPUAny,data:ArrayBufferView,usage:number,label:string):GPUAny{const size=Math.max(4,Math.ceil(data.byteLength/4)*4),b=device.createBuffer({label,size,usage,mappedAtCreation:true});new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));b.unmap();return b;}
function makeEmpty(device:GPUAny,size:number,usage:number,label:string):GPUAny{return device.createBuffer({label,size:Math.max(4,Math.ceil(size/4)*4),usage});}
const COMMON=/* wgsl */`
struct Params{nz:u32,edgeCount:u32,cellCount:u32,_p0:u32,dt:f32,omega:f32,rd:f32,gamma:f32,pRef:f32,kappa:f32,_p1:vec2<f32>};
@group(0) @binding(0) var<uniform> P:Params;
fn pressure(x:f32)->f32{return P.pRef*pow(P.rd*x/P.pRef,P.gamma);}
`;
const CORIOLIS=COMMON+/* wgsl */`
struct CellGeom{east:vec4<f32>,north:vec4<f32>};
@group(0) @binding(1)var<storage,read> edgeCells:array<vec2<u32>>;
@group(0) @binding(2)var<storage,read> edgeGeom:array<vec4<f32>>;
@group(0) @binding(3)var<storage,read> cellGeom:array<CellGeom>;
@group(0) @binding(4)var<storage,read> cellEdges:array<vec4<u32>>;
@group(0) @binding(5)var<storage,read> recon:array<vec2<f32>>;
@group(0) @binding(6)var<storage,read> u:array<f32>;
@group(0) @binding(7)var<storage,read_write> uOut:array<f32>;
fn cw(c:u32,k:u32)->vec3<f32>{let ee=cellEdges[c];var le=0.0;var ln=0.0;for(var s:u32=0u;s<4u;s++){let eid=ee[s];let co=recon[c*4u+s];let uv=u[eid*P.nz+k];le+=co.x*uv;ln+=co.y*uv;}return le*cellGeom[c].east.xyz+ln*cellGeom[c].north.xyz;}
fn delta(c:u32,w:vec3<f32>)->vec3<f32>{let g=cellGeom[c];let ue=dot(w,g.east.xyz);let vn=dot(w,g.north.xyz);let a=2.0*P.omega*g.east.w*P.dt;let co=cos(a);let si=sin(a);let ue2=ue*co+vn*si;let vn2=vn*co-ue*si;return(ue2-ue)*g.east.xyz+(vn2-vn)*g.north.xyz;}
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.edgeCount*P.nz){return;}let e=q/P.nz;let k=q-e*P.nz;let cc=edgeCells[e];let d=.5*(delta(cc.x,cw(cc.x,k))+delta(cc.y,cw(cc.y,k)));uOut[q]=u[q]+dot(d,edgeGeom[e].xyz);}
`;
const CELLWIND=COMMON+/* wgsl */`
struct CellGeom{east:vec4<f32>,north:vec4<f32>};@group(0)@binding(1)var<storage,read>cellGeom:array<CellGeom>;@group(0)@binding(2)var<storage,read>cellEdges:array<vec4<u32>>;@group(0)@binding(3)var<storage,read>recon:array<vec2<f32>>;@group(0)@binding(4)var<storage,read>u:array<f32>;@group(0)@binding(5)var<storage,read_write>cw:array<vec4<f32>>;
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.cellCount*P.nz){return;}let c=q/P.nz;let k=q-c*P.nz;let ee=cellEdges[c];var le=0.0;var ln=0.0;for(var s:u32=0u;s<4u;s++){let co=recon[c*4u+s];let uv=u[ee[s]*P.nz+k];le+=co.x*uv;ln+=co.y*uv;}cw[q]=vec4<f32>(le*cellGeom[c].east.xyz+ln*cellGeom[c].north.xyz,0.0);}
`;
const HMOM=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>edgeCells:array<vec2<u32>>;@group(0)@binding(2)var<storage,read>hFlux:array<vec2<f32>>;@group(0)@binding(3)var<storage,read>cw:array<vec4<f32>>;@group(0)@binding(4)var<storage,read_write>mf:array<vec4<f32>>;
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.edgeCount*P.nz){return;}let e=q/P.nz;let k=q-e*P.nz;let m=hFlux[q].x;let cc=edgeCells[e];let src=select(cc.y,cc.x,m>=0.0);mf[q]=vec4<f32>(cw[src*P.nz+k].xyz*m,0.0);}
`;
const VMOM=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>vFlux:array<vec2<f32>>;@group(0)@binding(2)var<storage,read>cw:array<vec4<f32>>;@group(0)@binding(3)var<storage,read_write>mf:array<vec4<f32>>;
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;let count=P.cellCount*(P.nz+1u);if(q>=count){return;}let c=q/(P.nz+1u);let i=q-c*(P.nz+1u);let m=vFlux[q].x;if(i==0u||i==P.nz){mf[q]=vec4<f32>(0.0);return;}let k=select(i,i-1u,m>=0.0);mf[q]=vec4<f32>(cw[c*P.nz+k].xyz*m,0.0);}
`;
const RHOBEFORE=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>volume:array<f32>;@group(0)@binding(2)var<storage,read>cellEdges:array<vec4<u32>>;@group(0)@binding(3)var<storage,read>cellSigns:array<vec4<i32>>;@group(0)@binding(4)var<storage,read>hFlux:array<vec2<f32>>;@group(0)@binding(5)var<storage,read>vFlux:array<vec2<f32>>;@group(0)@binding(6)var<storage,read>rho:array<f32>;@group(0)@binding(7)var<storage,read_write>rhoOld:array<f32>;
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.cellCount*P.nz){return;}let c=q/P.nz;let k=q-c*P.nz;let ee=cellEdges[c];let ss=cellSigns[c];var mt=0.0;for(var s:u32=0u;s<4u;s++){mt-=f32(ss[s])*hFlux[ee[s]*P.nz+k].x;}let vb=c*(P.nz+1u)+k;mt+=vFlux[vb].x-vFlux[vb+1u].x;rhoOld[q]=rho[q]-P.dt*mt/volume[q];}
`;
const MOMDELTA=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>volume:array<f32>;@group(0)@binding(2)var<storage,read>cellEdges:array<vec4<u32>>;@group(0)@binding(3)var<storage,read>cellSigns:array<vec4<i32>>;@group(0)@binding(4)var<storage,read>hm:array<vec4<f32>>;@group(0)@binding(5)var<storage,read>vm:array<vec4<f32>>;@group(0)@binding(6)var<storage,read>rhoOld:array<f32>;@group(0)@binding(7)var<storage,read>rho:array<f32>;@group(0)@binding(8)var<storage,read_write>cw:array<vec4<f32>>;
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.cellCount*P.nz){return;}let c=q/P.nz;let k=q-c*P.nz;let ee=cellEdges[c];let ss=cellSigns[c];var mt=vec3<f32>(0.0);for(var s:u32=0u;s<4u;s++){mt-=f32(ss[s])*hm[ee[s]*P.nz+k].xyz;}let vb=c*(P.nz+1u)+k;mt+=vm[vb].xyz-vm[vb+1u].xyz;let old=cw[q].xyz;let nw=(rhoOld[q]*old+P.dt*mt/volume[q])/max(rho[q],1e-8);cw[q]=vec4<f32>(nw-old,0.0);}
`;
const PROJECT=COMMON+/* wgsl */`
struct CellGeom{east:vec4<f32>,north:vec4<f32>};@group(0)@binding(1)var<storage,read>edgeCells:array<vec2<u32>>;@group(0)@binding(2)var<storage,read>edgeGeom:array<vec4<f32>>;@group(0)@binding(3)var<storage,read>cellGeom:array<CellGeom>;@group(0)@binding(4)var<storage,read>delta:array<vec4<f32>>;@group(0)@binding(5)var<storage,read>u:array<f32>;@group(0)@binding(6)var<storage,read_write>uOut:array<f32>;
fn tang(c:u32,d:vec3<f32>)->vec3<f32>{let r=normalize(cross(cellGeom[c].east.xyz,cellGeom[c].north.xyz));return d-r*dot(d,r);}
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.edgeCount*P.nz){return;}let e=q/P.nz;let k=q-e*P.nz;let cc=edgeCells[e];let d=.5*(tang(cc.x,delta[cc.x*P.nz+k].xyz)+tang(cc.y,delta[cc.y*P.nz+k].xyz));uOut[q]=u[q]+dot(d,edgeGeom[e].xyz);}
`;
const THERMAL=COMMON+/* wgsl */`
struct CellGeom{east:vec4<f32>,north:vec4<f32>};@group(0)@binding(1)var<storage,read>cellGeom:array<CellGeom>;@group(0)@binding(2)var<storage,read>rho:array<f32>;@group(0)@binding(3)var<storage,read_write>x:array<f32>;
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.cellCount*P.nz){return;}let c=q/P.nz;let lat=asin(clamp(cellGeom[c].east.w,-1.0,1.0));let p=pressure(x[q]);let sig=max(1e-6,p/P.pRef);let sn=sin(lat);let cs=cos(lat);let teq=max(200.0,(315.0-60.0*sn*sn-10.0*log(sig)*cs*cs)*pow(sig,P.kappa));let theta=x[q]/rho[q];let thetaEq=teq*pow(P.pRef/p,P.kappa);let sf=max(0.0,(sig-0.7)/0.3);let ka=1.0/(40.0*86400.0);let ks=1.0/(4.0*86400.0);let rate=ka+(ks-ka)*sf*pow(cs,4.0);let tn=thetaEq+(theta-thetaEq)*exp(-rate*P.dt);x[q]=rho[q]*tn;}
`;
const DRAG=COMMON+/* wgsl */`
@group(0)@binding(1)var<storage,read>edgeCells:array<vec2<u32>>;@group(0)@binding(2)var<storage,read>rhoTheta:array<f32>;@group(0)@binding(3)var<storage,read_write>u:array<f32>;
@compute @workgroup_size(128)fn main(@builtin(global_invocation_id)gid:vec3<u32>){let q=gid.x;if(q>=P.edgeCount*P.nz){return;}let e=q/P.nz;let k=q-e*P.nz;let cc=edgeCells[e];let sig=.5*(pressure(rhoTheta[cc.x*P.nz+k])+pressure(rhoTheta[cc.y*P.nz+k]))/P.pRef;let rate=(1.0/86400.0)*max(0.0,(sig-0.7)/0.3);u[q]=u[q]*exp(-rate*P.dt);}
`;
export class GpuRotatingDryCore{
  readonly device:GPUAny;readonly buffers:Record<string,GPUAny>={};readonly pipelines:Record<string,GPUAny>={};readonly groups:Record<string,GPUAny>={};private paramsFull:GPUAny;private paramsHalf:GPUAny;
  private constructor(public readonly core:GpuDryCorePrototype,public readonly h:CubedSphereGrid,public readonly v:VerticalGrid,public readonly ref:ReferenceAtmosphere){
    this.device=core.device;const U=(globalThis as any).GPUBufferUsage,storage=U.STORAGE|U.COPY_DST|U.COPY_SRC,uniform=U.UNIFORM|U.COPY_DST,g=buildRotationGeometry(h);this.paramsFull=makeEmpty(this.device,48,uniform,'stage4 params full');this.paramsHalf=makeEmpty(this.device,48,uniform,'stage4 params half');
    const edgeGeom=new Float32Array(h.edgeCount*4);for(let e=0;e<h.edgeCount;e++){const ge=h.edges[e]!;edgeGeom.set([ge.normal[0],ge.normal[1],ge.normal[2],ge.midpoint[2]],e*4)}const cellGeom=new Float32Array(h.cellCount*8);for(let c=0;c<h.cellCount;c++)cellGeom.set([g.east[c*3]!,g.east[c*3+1]!,g.east[c*3+2]!,g.radial[c*3+2]!,g.north[c*3]!,g.north[c*3+1]!,g.north[c*3+2]!,0],c*8);const volume=new Float32Array(h.cellCount*v.nz);for(let c=0;c<h.cellCount;c++)for(let k=0;k<v.nz;k++)volume[c*v.nz+k]=h.cellAreaUnit[c]!*EARTH.radius*EARTH.radius*v.dz[k]!;
    this.buffers.edgeGeom=makeBuffer(this.device,edgeGeom,storage,'stage4 edge geom');this.buffers.cellGeom=makeBuffer(this.device,cellGeom,storage,'stage4 cell geom');this.buffers.recon=makeBuffer(this.device,Float32Array.from(g.reconstruction),storage,'stage4 reconstruction');this.buffers.volume=makeBuffer(this.device,volume,storage,'stage4 cell volume');this.buffers.uTmp=makeEmpty(this.device,core.buffers.u.size??h.edgeCount*v.nz*4,storage,'stage4 u temp');this.buffers.cellWind=makeEmpty(this.device,h.cellCount*v.nz*16,storage,'stage4 cell wind/delta');this.buffers.hMom=makeEmpty(this.device,h.edgeCount*v.nz*16,storage,'stage4 h momentum flux');this.buffers.vMom=makeEmpty(this.device,h.cellCount*(v.nz+1)*16,storage,'stage4 v momentum flux');this.buffers.rhoOld=makeEmpty(this.device,h.cellCount*v.nz*4,storage,'stage4 rho before outer transport');this.build();
  }
  static async create(h:CubedSphereGrid,v:VerticalGrid,ref:ReferenceAtmosphere,state:DryState):Promise<GpuRotatingDryCore>{const core=await GpuDryCorePrototype.create(h,v,ref,state,STAGE4_HEVI_OFFCENTERING);core.device.pushErrorScope?.('validation');const out=new GpuRotatingDryCore(core,h,v,ref);const err=await core.device.popErrorScope?.();if(err){out.destroy();throw new Error(`Stage 4 WebGPU validation: ${err.message||err}`)}return out}
  private pipe(code:string,label:string):GPUAny{return this.device.createComputePipeline({label,layout:'auto',compute:{module:this.device.createShaderModule({label:`${label} shader`,code}),entryPoint:'main'}})}
  private entries(names:string[]):any[]{return names.map((n,i)=>({binding:i,resource:{buffer:n==='params'?this.paramsFull:n==='paramsHalf'?this.paramsHalf:(this.buffers[n]??this.core.buffers[n])}}))}
  private build():void{const d:[string,string,string[]][]=[['coriolis',CORIOLIS,['paramsHalf','edgeCells','edgeGeom','cellGeom','cellEdges','recon','u','uTmp']],['cellWind',CELLWIND,['params','cellGeom','cellEdges','recon','u','cellWind']],['hMom',HMOM,['params','edgeCells','hFlux','cellWind','hMom']],['vMom',VMOM,['params','vFlux','cellWind','vMom']],['rhoOld',RHOBEFORE,['params','volume','cellEdges','cellSigns','hFlux','vFlux','rho','rhoOld']],['momDelta',MOMDELTA,['params','volume','cellEdges','cellSigns','hMom','vMom','rhoOld','rho','cellWind']],['project',PROJECT,['params','edgeCells','edgeGeom','cellGeom','cellWind','u','uTmp']],['thermal',THERMAL,['params','cellGeom','rho','rhoTheta']],['drag',DRAG,['params','edgeCells','rhoTheta','u']]];for(const[name,code,b]of d){const p=this.pipe(code,name);this.pipelines[name]=p;this.groups[name]=this.device.createBindGroup({layout:p.getBindGroupLayout(0),entries:this.entries(b)})}}
  private writeBuffer(buf:GPUAny,dt:number):void{const ab=new ArrayBuffer(48),u=new Uint32Array(ab),f=new Float32Array(ab);u[0]=this.v.nz;u[1]=this.h.edgeCount;u[2]=this.h.cellCount;f[4]=dt;f[5]=EARTH.omega;f[6]=DRY_AIR.rd;f[7]=DRY_AIR.gamma;f[8]=DRY_AIR.pRef;f[9]=DRY_AIR.kappa;this.device.queue.writeBuffer(buf,0,ab)}
  private prepareBase(dt:number):void{const ab=new ArrayBuffer(48),u=new Uint32Array(ab),f=new Float32Array(ab);u[0]=this.v.nz;u[1]=this.h.edgeCount;u[2]=this.h.cellCount;f[4]=dt;f[5]=EARTH.radius;f[6]=EARTH.gravity;f[7]=DRY_AIR.rd;f[8]=DRY_AIR.gamma;f[9]=DRY_AIR.pRef;f[10]=this.core.heviOffCentering;this.device.queue.writeBuffer((this.core as any).paramsBuffer,0,ab)}
  private prepare(dt:number):void{this.writeBuffer(this.paramsFull,dt);this.writeBuffer(this.paramsHalf,.5*dt);this.prepareBase(dt)}
  private dispatch(pass:GPUAny,name:string,count:number):void{pass.setPipeline(this.pipelines[name]);pass.setBindGroup(0,this.groups[name]);pass.dispatchWorkgroups(Math.ceil(count/128))}
  private onePass(enc:GPUAny,name:string,count:number):void{const p=enc.beginComputePass();this.dispatch(p,name,count);p.end()}
  private encodeOne(enc:GPUAny,heldSuarez:boolean):void{
    this.onePass(enc,'coriolis',this.h.edgeCount*this.v.nz);enc.copyBufferToBuffer(this.buffers.uTmp,0,this.core.buffers.u,0,this.h.edgeCount*this.v.nz*4);
    {const p=enc.beginComputePass(),seq:[string,number,number][]=[['pressure',this.h.cellCount*this.v.nz,128],['hvel',this.h.edgeCount*this.v.nz,128],['hevi',this.h.cellCount,1],['buoyancy',this.h.cellCount*(this.v.nz-1),128],['hflux',this.h.edgeCount*this.v.nz,128],['vflux',this.h.cellCount*(this.v.nz+1),128],['divergence',this.h.cellCount*this.v.nz,128]];for(const[name,count,size]of seq){p.setPipeline(this.core.pipelines[name]);p.setBindGroup(0,this.core.bindGroups[name]);p.dispatchWorkgroups(Math.ceil(count/size))}p.end()}
    {const p=enc.beginComputePass();this.dispatch(p,'cellWind',this.h.cellCount*this.v.nz);this.dispatch(p,'hMom',this.h.edgeCount*this.v.nz);this.dispatch(p,'vMom',this.h.cellCount*(this.v.nz+1));this.dispatch(p,'rhoOld',this.h.cellCount*this.v.nz);this.dispatch(p,'momDelta',this.h.cellCount*this.v.nz);this.dispatch(p,'project',this.h.edgeCount*this.v.nz);p.end()}
    enc.copyBufferToBuffer(this.buffers.uTmp,0,this.core.buffers.u,0,this.h.edgeCount*this.v.nz*4);if(heldSuarez){const p=enc.beginComputePass();this.dispatch(p,'thermal',this.h.cellCount*this.v.nz);this.dispatch(p,'drag',this.h.edgeCount*this.v.nz);p.end()}this.onePass(enc,'coriolis',this.h.edgeCount*this.v.nz);enc.copyBufferToBuffer(this.buffers.uTmp,0,this.core.buffers.u,0,this.h.edgeCount*this.v.nz*4);
  }
  step(dt:number,heldSuarez=true):void{this.prepare(dt);const enc=this.device.createCommandEncoder({label:'stage4 rotating dry step'});this.encodeOne(enc,heldSuarez);this.device.queue.submit([enc.finish()])}
  stepBatch(dt:number,count:number,heldSuarez=true):void{if(!Number.isInteger(count)||count<1)throw new Error('batch count must be positive integer');this.prepare(dt);const enc=this.device.createCommandEncoder({label:`stage4 batch ${count}`});for(let i=0;i<count;i++)this.encodeOne(enc,heldSuarez);this.device.queue.submit([enc.finish()])}
  downloadState(time=0):Promise<DryState>{return this.core.downloadState(time)}destroy():void{for(const b of Object.values(this.buffers))b.destroy?.();this.paramsFull.destroy?.();this.paramsHalf.destroy?.();this.core.destroy()}
}
