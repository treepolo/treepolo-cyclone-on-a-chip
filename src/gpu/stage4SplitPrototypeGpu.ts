import { DRY_AIR, EARTH } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { DryState } from '../solver/state.js';
import { GpuAcousticDivergenceDamping } from './acousticDivergenceDampingGpu.js';
import { GpuRotatingDryCore } from './rotatingDryCoreGpu.js';

type GPUAny=any;

const ACCUM_REF_MASS=/* wgsl */`
struct Params{count:u32,weight:f32,_pad:vec2<f32>};
@group(0)@binding(0)var<uniform>P:Params;
@group(0)@binding(1)var<storage,read>src:array<f32>;
@group(0)@binding(2)var<storage,read_write>dst:array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){
  let q=gid.x;if(q>=P.count){return;}dst[q]=dst[q]+P.weight*src[q];
}`;

/**
 * Diagnostic split-explicit prototype for Stage 4.
 *
 * This intentionally does NOT replace the production Stage-4 integrator yet.
 * A meteorological outer step is kept at dtOuter, while the pressure/acoustic,
 * HEVI, explicit buoyancy and acoustic-divergence operators are advanced with
 * equal acoustic substeps. Horizontal/vertical scalar transport, full 3-D
 * momentum transport, Held-Suarez forcing and surface drag remain one outer
 * update. The HEVI reference mass flux is averaged across the acoustic steps
 * before the outer momentum bookkeeping so Fref + Fpert remains consistent
 * with the total continuity update over dtOuter.
 *
 * This is the first structural split test before adding the locked outer RK3
 * stages. It exists to answer whether acoustic substepping fixes the observed
 * 38-40 km fast-mode growth without paying the mass-drift cost of shrinking
 * every slow operator to 2.5 s.
 */
export class GpuStage4SplitPrototype{
  readonly device:GPUAny;
  readonly divergence:GpuAcousticDivergenceDamping;
  private readonly fastParams:GPUAny;
  private readonly fastGroups:Record<string,GPUAny>={};
  private readonly refMassAverage:GPUAny;
  private readonly accumParams:GPUAny;
  private readonly accumPipeline:GPUAny;
  private readonly accumGroup:GPUAny;

  private constructor(
    public readonly core:GpuRotatingDryCore,
    public readonly acousticSubsteps:number,
  ){
    if(!Number.isInteger(acousticSubsteps)||acousticSubsteps<2)throw new Error('acousticSubsteps must be an integer >= 2');
    this.device=core.device;
    const U=(globalThis as any).GPUBufferUsage;
    const storage=U.STORAGE|U.COPY_DST|U.COPY_SRC;
    this.fastParams=this.device.createBuffer({label:'stage4 split fast params',size:48,usage:U.UNIFORM|U.COPY_DST});
    this.accumParams=this.device.createBuffer({label:'stage4 split reference-flux accumulator params',size:16,usage:U.UNIFORM|U.COPY_DST});
    const wCount=core.h.cellCount*(core.v.nz+1);
    this.refMassAverage=this.device.createBuffer({label:'stage4 split mean HEVI reference mass flux',size:Math.max(4,wCount*4),usage:storage});
    this.accumPipeline=this.device.createComputePipeline({label:'stage4 split average HEVI reference mass flux',layout:'auto',compute:{module:this.device.createShaderModule({label:'stage4 split reference mass accumulator shader',code:ACCUM_REF_MASS}),entryPoint:'main'}});
    this.accumGroup=this.device.createBindGroup({layout:this.accumPipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:this.accumParams}},
      {binding:1,resource:{buffer:core.core.buffers.heviRefMass}},
      {binding:2,resource:{buffer:this.refMassAverage}},
    ]});
    this.buildFastGroups();
    this.divergence=new GpuAcousticDivergenceDamping(core);
  }

  static async create(h:CubedSphereGrid,v:VerticalGrid,ref:ReferenceAtmosphere,state:DryState,acousticSubsteps=4):Promise<GpuStage4SplitPrototype>{
    const core=await GpuRotatingDryCore.create(h,v,ref,state);
    core.device.pushErrorScope?.('validation');
    try{
      const out=new GpuStage4SplitPrototype(core,acousticSubsteps);
      const err=await core.device.popErrorScope?.();
      if(err){out.destroy();throw new Error(`Stage 4 split prototype WebGPU validation: ${err.message||err}`);}
      return out;
    }catch(e){try{await core.device.popErrorScope?.();}catch{}core.destroy();throw e;}
  }

  private buildFastGroups():void{
    const c=this.core.core,b=c.buffers,p=c.pipelines,fp=this.fastParams;
    const make=(name:string,buffers:GPUAny[])=>this.device.createBindGroup({layout:p[name].getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:fp}},...buffers.map((buffer,i)=>({binding:i+1,resource:{buffer}}))]});
    this.fastGroups.pressure=make('pressure',[b.rhoTheta,b.pressure]);
    this.fastGroups.hvel=make('hvel',[b.edgeCells,b.edgeMetric,b.rho,b.pressure,b.u]);
    this.fastGroups.hevi=make('hevi',[b.layerRef,b.interfaceRef,b.cellArea,b.rho,b.rhoTheta,b.w,b.heviRayleigh,b.heviRefMass]);
    this.fastGroups.buoyancy=make('buoyancy',[b.layerRef,b.rho,b.w]);
  }

  private writeFastParams(dt:number):void{
    const ab=new ArrayBuffer(48),u=new Uint32Array(ab),f=new Float32Array(ab);
    u[0]=this.core.v.nz;u[1]=this.core.h.edgeCount;u[2]=this.core.h.cellCount;
    f[4]=dt;f[5]=EARTH.radius;f[6]=EARTH.gravity;f[7]=DRY_AIR.rd;f[8]=DRY_AIR.gamma;f[9]=DRY_AIR.pRef;f[10]=this.core.core.heviOffCentering;
    this.device.queue.writeBuffer(this.fastParams,0,ab);
    const aa=new ArrayBuffer(16),au=new Uint32Array(aa),af=new Float32Array(aa);au[0]=this.core.h.cellCount*(this.core.v.nz+1);af[1]=1/this.acousticSubsteps;this.device.queue.writeBuffer(this.accumParams,0,aa);
  }

  private dispatchBase(pass:GPUAny,name:string,count:number,size=128,group?:GPUAny):void{
    const c=this.core.core;pass.setPipeline(c.pipelines[name]);pass.setBindGroup(0,group??c.bindGroups[name]);pass.dispatchWorkgroups(Math.ceil(count/size));
  }

  private encodeFastSubstep(enc:GPUAny):void{
    const h=this.core.h,v=this.core.v;
    let p=enc.beginComputePass();
    this.dispatchBase(p,'pressure',h.cellCount*v.nz,128,this.fastGroups.pressure);
    this.dispatchBase(p,'hvel',h.edgeCount*v.nz,128,this.fastGroups.hvel);
    this.dispatchBase(p,'hevi',h.cellCount,1,this.fastGroups.hevi);
    this.dispatchBase(p,'buoyancy',h.cellCount*(v.nz-1),128,this.fastGroups.buoyancy);
    p.end();
    p=enc.beginComputePass();p.setPipeline(this.accumPipeline);p.setBindGroup(0,this.accumGroup);p.dispatchWorkgroups(Math.ceil(h.cellCount*(v.nz+1)/128));p.end();
    this.divergence.encode(enc);
  }

  private encodeCoriolisHalf(enc:GPUAny):void{
    const r=this.core as any,h=this.core.h,v=this.core.v;
    r.onePass(enc,'coriolis',h.edgeCount*v.nz);
    enc.copyBufferToBuffer(r.buffers.uTmp,0,this.core.core.buffers.u,0,h.edgeCount*v.nz*4);
  }

  private encodeOuterSlow(enc:GPUAny,heldSuarez:boolean):void{
    const r=this.core as any,h=this.core.h,v=this.core.v,c=this.core.core;
    let p=enc.beginComputePass();
    this.dispatchBase(p,'hflux',h.edgeCount*v.nz);
    this.dispatchBase(p,'vflux',h.cellCount*(v.nz+1));
    this.dispatchBase(p,'divergence',h.cellCount*v.nz);
    p.end();
    p=enc.beginComputePass();
    r.dispatch(p,'cellWind',h.cellCount*v.nz);
    r.dispatch(p,'hMom',h.edgeCount*v.nz);
    r.dispatch(p,'vMom',h.cellCount*(v.nz+1));
    r.dispatch(p,'rhoOld',h.cellCount*v.nz);
    r.dispatch(p,'momDelta',h.cellCount*v.nz);
    r.dispatch(p,'project',h.edgeCount*v.nz);
    r.dispatch(p,'wAdvect',h.cellCount*(v.nz+1));
    p.end();
    enc.copyBufferToBuffer(r.buffers.uTmp,0,c.buffers.u,0,h.edgeCount*v.nz*4);
    enc.copyBufferToBuffer(r.buffers.wTmp,0,c.buffers.w,0,h.cellCount*(v.nz+1)*4);
    if(heldSuarez){p=enc.beginComputePass();r.dispatch(p,'thermal',h.cellCount*v.nz);r.dispatch(p,'drag',h.edgeCount*v.nz);p.end();}
  }

  private encodeOuterStep(enc:GPUAny,heldSuarez:boolean):void{
    const c=this.core.core,h=this.core.h,v=this.core.v;
    this.encodeCoriolisHalf(enc);
    enc.clearBuffer(this.refMassAverage);
    for(let n=0;n<this.acousticSubsteps;n++)this.encodeFastSubstep(enc);
    enc.copyBufferToBuffer(this.refMassAverage,0,c.buffers.heviRefMass,0,h.cellCount*(v.nz+1)*4);
    this.encodeOuterSlow(enc,heldSuarez);
    this.encodeCoriolisHalf(enc);
  }

  private prepare(dtOuter:number):void{
    if(!(dtOuter>0))throw new Error('outer dt must be positive');
    const dtFast=dtOuter/this.acousticSubsteps;
    (this.core as any).prepare(dtOuter);
    this.writeFastParams(dtFast);
    this.divergence.prepare(dtFast);
  }

  step(dtOuter:number,heldSuarez=true):void{
    this.prepare(dtOuter);const enc=this.device.createCommandEncoder({label:`stage4 split outer dt=${dtOuter}`});this.encodeOuterStep(enc,heldSuarez);this.device.queue.submit([enc.finish()]);
  }

  stepBatch(dtOuter:number,count:number,heldSuarez=true):void{
    if(!Number.isInteger(count)||count<1)throw new Error('batch count must be positive integer');
    this.prepare(dtOuter);const enc=this.device.createCommandEncoder({label:`stage4 split batch outer=${dtOuter} n=${count}`});
    for(let i=0;i<count;i++)this.encodeOuterStep(enc,heldSuarez);
    this.device.queue.submit([enc.finish()]);
  }

  downloadState(time=0):Promise<DryState>{return this.core.downloadState(time);}
  destroy():void{this.divergence.destroy();this.fastParams.destroy?.();this.refMassAverage.destroy?.();this.accumParams.destroy?.();this.core.destroy();}
}
