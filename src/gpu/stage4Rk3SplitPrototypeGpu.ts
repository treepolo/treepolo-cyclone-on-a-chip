import { DRY_AIR, EARTH } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { acousticDivergenceCoefficientForDt } from '../physics/acousticDivergenceDamping.js';
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

const RK_REANCHOR=/* wgsl */`
struct Params{count:u32,_pad:vec3<u32>};
@group(0)@binding(0)var<uniform>P:Params;
@group(0)@binding(1)var<storage,read>base:array<f32>;
@group(0)@binding(2)var<storage,read>predictor:array<f32>;
@group(0)@binding(3)var<storage,read_write>state:array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){
  let q=gid.x;if(q>=P.count){return;}
  state[q]=base[q]+(state[q]-predictor[q]);
}`;

interface StageResources{
  durationFraction:number;
  acousticSubsteps:number;
  rotFull:GPUAny;
  rotHalf:GPUAny;
  coreSlow:GPUAny;
  coreFast:GPUAny;
  divParams:GPUAny;
  accumParams:GPUAny;
  coreFastGroups:Record<string,GPUAny>;
  coreSlowGroups:Record<string,GPUAny>;
  rotGroups:Record<string,GPUAny>;
  divGroup:GPUAny;
  divAdjustGroup:GPUAny;
  accumGroup:GPUAny;
}

/**
 * Diagnostic Wicker/Skamarock-style RK3 + split-acoustic scaffold.
 *
 * The three predictor-reset stages use the ARW/Wicker-Skamarock large-step
 * fractions 1/3, 1/2 and 1.  With ns=4 the acoustic schedule is:
 *
 *   stage 1: 1 x (dt/3)
 *   stage 2: 2 x (dt/4)
 *   stage 3: 4 x (dt/4)
 *
 * For a stage predictor P, the split stage is advanced for the corresponding
 * stage duration, then ONLY its increment (advanced - P) is retained and
 * re-anchored to the large-step base state B:
 *
 *   stage_out = B + (advanced - P)
 *
 * This implements the RK3 predictor-reset algebra without chaining three full
 * timesteps. HEVI reference mass flux is time-averaged over each acoustic loop
 * before that stage's scalar/horizontal-momentum transport, preserving the
 * continuity-consistent Fref + Fpert bookkeeping demonstrated by the Stage-4
 * split prototype.
 *
 * IMPORTANT: this remains a diagnostic prototype, not the production RK3 core.
 * Slow momentum/forcing operators are still applied after each stage's acoustic
 * loop rather than being frozen as RHS terms inside every acoustic small step.
 * The purpose of this layer is to validate the RK3 predictor-reset and stage
 * schedule before the final slow-RHS-freezing refactor.
 */
export class GpuStage4Rk3SplitPrototype{
  readonly device:GPUAny;
  readonly divergence:GpuAcousticDivergenceDamping;
  readonly stages:StageResources[]=[];
  private readonly refMassAverage:GPUAny;
  private readonly accumPipeline:GPUAny;
  private readonly reanchorPipeline:GPUAny;
  private readonly base:Record<string,GPUAny>={};
  private readonly predictor:Record<string,GPUAny>={};
  private readonly reanchorParams:Record<string,GPUAny>={};
  private readonly reanchorGroups:Record<string,GPUAny>={};

  private constructor(public readonly core:GpuRotatingDryCore){
    this.device=core.device;
    const U=(globalThis as any).GPUBufferUsage;
    const storage=U.STORAGE|U.COPY_DST|U.COPY_SRC,uniform=U.UNIFORM|U.COPY_DST;
    const h=core.h,v=core.v,c=core.core,b=c.buffers,r:any=core,d:any=new GpuAcousticDivergenceDamping(core);
    this.divergence=d;
    const wCount=h.cellCount*(v.nz+1);
    this.refMassAverage=this.device.createBuffer({label:'rk3 mean HEVI reference mass flux',size:Math.max(4,wCount*4),usage:storage});
    this.accumPipeline=this.device.createComputePipeline({label:'rk3 HEVI reference mass accumulator',layout:'auto',compute:{module:this.device.createShaderModule({label:'rk3 reference mass accumulator shader',code:ACCUM_REF_MASS}),entryPoint:'main'}});
    this.reanchorPipeline=this.device.createComputePipeline({label:'rk3 predictor reanchor',layout:'auto',compute:{module:this.device.createShaderModule({label:'rk3 predictor reanchor shader',code:RK_REANCHOR}),entryPoint:'main'}});

    const fields:[string,GPUAny,number][]=[
      ['rho',b.rho,h.cellCount*v.nz],
      ['rhoTheta',b.rhoTheta,h.cellCount*v.nz],
      ['u',b.u,h.edgeCount*v.nz],
      ['w',b.w,h.cellCount*(v.nz+1)],
    ];
    for(const[name,src,count]of fields){
      const bytes=Math.max(4,count*4);
      this.base[name]=this.device.createBuffer({label:`rk3 base ${name}`,size:bytes,usage:storage});
      this.predictor[name]=this.device.createBuffer({label:`rk3 predictor ${name}`,size:bytes,usage:storage});
      const p=this.device.createBuffer({label:`rk3 reanchor params ${name}`,size:16,usage:uniform});
      const ab=new ArrayBuffer(16);new Uint32Array(ab)[0]=count;this.device.queue.writeBuffer(p,0,ab);this.reanchorParams[name]=p;
      this.reanchorGroups[name]=this.device.createBindGroup({layout:this.reanchorPipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer:p}},
        {binding:1,resource:{buffer:this.base[name]}},
        {binding:2,resource:{buffer:this.predictor[name]}},
        {binding:3,resource:{buffer:src}},
      ]});
    }

    const makeUniform=(label:string)=>this.device.createBuffer({label,size:48,usage:uniform});
    const makeAccumParams=(label:string)=>this.device.createBuffer({label,size:16,usage:uniform});
    const stageDefs:[number,number][]=[[1/3,1],[1/2,2],[1,4]];
    for(let si=0;si<stageDefs.length;si++){
      const[durationFraction,acousticSubsteps]=stageDefs[si]!;
      const rotFull=makeUniform(`rk3 stage ${si+1} rotating full params`),rotHalf=makeUniform(`rk3 stage ${si+1} rotating half params`),coreSlow=makeUniform(`rk3 stage ${si+1} core slow params`),coreFast=makeUniform(`rk3 stage ${si+1} core fast params`),divParams=makeUniform(`rk3 stage ${si+1} divergence params`),accumParams=makeAccumParams(`rk3 stage ${si+1} accumulator params`);
      const coreGroup=(name:string,params:GPUAny,buffers:GPUAny[])=>this.device.createBindGroup({layout:c.pipelines[name].getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:params}},...buffers.map((buffer:GPUAny,i:number)=>({binding:i+1,resource:{buffer}}))]});
      const rotGroup=(name:string,params:GPUAny,buffers:GPUAny[])=>this.device.createBindGroup({layout:r.pipelines[name].getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:params}},...buffers.map((buffer:GPUAny,i:number)=>({binding:i+1,resource:{buffer}}))]});
      const coreFastGroups:Record<string,GPUAny>={
        pressure:coreGroup('pressure',coreFast,[b.rhoTheta,b.pressure]),
        hvel:coreGroup('hvel',coreFast,[b.edgeCells,b.edgeMetric,b.rho,b.pressure,b.u]),
        hevi:coreGroup('hevi',coreFast,[b.layerRef,b.interfaceRef,b.cellArea,b.rho,b.rhoTheta,b.w,b.heviRayleigh,b.heviRefMass]),
        buoyancy:coreGroup('buoyancy',coreFast,[b.layerRef,b.rho,b.w]),
      };
      const coreSlowGroups:Record<string,GPUAny>={
        hflux:coreGroup('hflux',coreSlow,[b.edgeCells,b.edgeMetric,b.layerRef,b.rho,b.rhoTheta,b.u,b.hFlux]),
        vflux:coreGroup('vflux',coreSlow,[b.cellArea,b.layerRef,b.rho,b.rhoTheta,b.w,b.vFlux]),
        divergence:coreGroup('divergence',coreSlow,[b.cellArea,b.layerRef,b.cellEdges,b.cellSigns,b.hFlux,b.vFlux,b.rho,b.rhoTheta]),
      };
      const rotGroups:Record<string,GPUAny>={
        coriolis:rotGroup('coriolis',rotHalf,[b.edgeCells,r.buffers.edgeGeom,r.buffers.cellGeom,b.cellEdges,r.buffers.recon,b.u,r.buffers.uTmp]),
        cellWind:rotGroup('cellWind',rotFull,[r.buffers.cellGeom,b.cellEdges,r.buffers.recon,b.u,r.buffers.cellWind]),
        hMom:rotGroup('hMom',rotFull,[b.edgeCells,b.hFlux,r.buffers.cellWind,r.buffers.hMom]),
        vMom:rotGroup('vMom',rotFull,[b.vFlux,b.heviRefMass,r.buffers.cellWind,r.buffers.vMom]),
        wAdvect:rotGroup('wAdvect',rotFull,[r.buffers.wAdvMeta,b.edgeMetric,b.layerRef,b.cellArea,b.u,b.w,r.buffers.wTmp]),
        rhoOld:rotGroup('rhoOld',rotFull,[r.buffers.volume,b.cellEdges,b.cellSigns,b.hFlux,b.vFlux,b.heviRefMass,b.rho,r.buffers.rhoOld]),
        momDelta:rotGroup('momDelta',rotFull,[r.buffers.volume,b.cellEdges,b.cellSigns,r.buffers.hMom,r.buffers.vMom,r.buffers.rhoOld,b.rho,r.buffers.cellWind]),
        project:rotGroup('project',rotFull,[b.edgeCells,r.buffers.edgeGeom,r.buffers.cellGeom,r.buffers.cellWind,b.u,r.buffers.uTmp]),
        thermal:rotGroup('thermal',rotFull,[r.buffers.cellGeom,b.rho,b.rhoTheta]),
        drag:rotGroup('drag',rotFull,[b.edgeCells,b.rhoTheta,b.u]),
      };
      const dd:any=this.divergence;
      const divGroup=this.device.createBindGroup({layout:dd.divergencePipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer:divParams}},{binding:1,resource:{buffer:b.cellArea}},{binding:2,resource:{buffer:b.edgeMetric}},{binding:3,resource:{buffer:b.cellEdges}},{binding:4,resource:{buffer:b.cellSigns}},{binding:5,resource:{buffer:b.u}},{binding:6,resource:{buffer:dd.divergence}},
      ]});
      const divAdjustGroup=this.device.createBindGroup({layout:dd.adjustPipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer:divParams}},{binding:1,resource:{buffer:b.edgeCells}},{binding:2,resource:{buffer:b.edgeMetric}},{binding:3,resource:{buffer:dd.divergence}},{binding:4,resource:{buffer:b.u}},
      ]});
      const accumGroup=this.device.createBindGroup({layout:this.accumPipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer:accumParams}},{binding:1,resource:{buffer:b.heviRefMass}},{binding:2,resource:{buffer:this.refMassAverage}},
      ]});
      this.stages.push({durationFraction,acousticSubsteps,rotFull,rotHalf,coreSlow,coreFast,divParams,accumParams,coreFastGroups,coreSlowGroups,rotGroups,divGroup,divAdjustGroup,accumGroup});
    }
  }

  static async create(h:CubedSphereGrid,v:VerticalGrid,ref:ReferenceAtmosphere,state:DryState):Promise<GpuStage4Rk3SplitPrototype>{
    const core=await GpuRotatingDryCore.create(h,v,ref,state);core.device.pushErrorScope?.('validation');
    try{
      const out=new GpuStage4Rk3SplitPrototype(core),err=await core.device.popErrorScope?.();
      if(err){out.destroy();throw new Error(`Stage 4 RK3 split prototype WebGPU validation: ${err.message||err}`);}
      return out;
    }catch(e){try{await core.device.popErrorScope?.();}catch{}core.destroy();throw e;}
  }

  private writeRotParams(buf:GPUAny,dt:number):void{
    const ab=new ArrayBuffer(48),u=new Uint32Array(ab),f=new Float32Array(ab);u[0]=this.core.v.nz;u[1]=this.core.h.edgeCount;u[2]=this.core.h.cellCount;f[4]=dt;f[5]=EARTH.omega;f[6]=DRY_AIR.rd;f[7]=DRY_AIR.gamma;f[8]=DRY_AIR.pRef;f[9]=DRY_AIR.kappa;this.device.queue.writeBuffer(buf,0,ab);
  }
  private writeCoreParams(buf:GPUAny,dt:number):void{
    const ab=new ArrayBuffer(48),u=new Uint32Array(ab),f=new Float32Array(ab);u[0]=this.core.v.nz;u[1]=this.core.h.edgeCount;u[2]=this.core.h.cellCount;f[4]=dt;f[5]=EARTH.radius;f[6]=EARTH.gravity;f[7]=DRY_AIR.rd;f[8]=DRY_AIR.gamma;f[9]=DRY_AIR.pRef;f[10]=this.core.core.heviOffCentering;this.device.queue.writeBuffer(buf,0,ab);
  }
  private writeDivParams(buf:GPUAny,dt:number):void{
    const ab=new ArrayBuffer(48),u=new Uint32Array(ab),f=new Float32Array(ab);u[0]=this.core.v.nz;u[1]=this.core.h.edgeCount;u[2]=this.core.h.cellCount;f[4]=acousticDivergenceCoefficientForDt(dt);this.device.queue.writeBuffer(buf,0,ab);
  }
  private prepare(dtOuter:number):void{
    if(!(dtOuter>0))throw new Error('RK3 outer dt must be positive');
    const ns=4;
    for(const s of this.stages){
      const duration=dtOuter*s.durationFraction;
      const fastDt=s===this.stages[0]?duration:dtOuter/ns;
      this.writeRotParams(s.rotFull,duration);this.writeRotParams(s.rotHalf,.5*duration);this.writeCoreParams(s.coreSlow,duration);this.writeCoreParams(s.coreFast,fastDt);this.writeDivParams(s.divParams,fastDt);
      const ab=new ArrayBuffer(16),u=new Uint32Array(ab),f=new Float32Array(ab);u[0]=this.core.h.cellCount*(this.core.v.nz+1);f[1]=1/s.acousticSubsteps;this.device.queue.writeBuffer(s.accumParams,0,ab);
    }
  }

  private dispatch(pass:GPUAny,pipeline:GPUAny,group:GPUAny,count:number,size=128):void{pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/size));}
  private encodeCoriolisHalf(enc:GPUAny,s:StageResources):void{
    const r:any=this.core,h=this.core.h,v=this.core.v;
    let p=enc.beginComputePass();this.dispatch(p,r.pipelines.coriolis,s.rotGroups.coriolis,h.edgeCount*v.nz);p.end();
    enc.copyBufferToBuffer(r.buffers.uTmp,0,this.core.core.buffers.u,0,h.edgeCount*v.nz*4);
  }
  private encodeFast(enc:GPUAny,s:StageResources):void{
    const c=this.core.core as any,h=this.core.h,v=this.core.v,dd:any=this.divergence;
    let p=enc.beginComputePass();
    this.dispatch(p,c.pipelines.pressure,s.coreFastGroups.pressure,h.cellCount*v.nz);
    this.dispatch(p,c.pipelines.hvel,s.coreFastGroups.hvel,h.edgeCount*v.nz);
    this.dispatch(p,c.pipelines.hevi,s.coreFastGroups.hevi,h.cellCount,1);
    this.dispatch(p,c.pipelines.buoyancy,s.coreFastGroups.buoyancy,h.cellCount*(v.nz-1));p.end();
    p=enc.beginComputePass();this.dispatch(p,this.accumPipeline,s.accumGroup,h.cellCount*(v.nz+1));p.end();
    p=enc.beginComputePass();this.dispatch(p,dd.divergencePipeline,s.divGroup,h.cellCount*v.nz);p.end();
    p=enc.beginComputePass();this.dispatch(p,dd.adjustPipeline,s.divAdjustGroup,h.edgeCount*v.nz);p.end();
  }
  private encodeSlow(enc:GPUAny,s:StageResources,heldSuarez:boolean):void{
    const c:any=this.core.core,r:any=this.core,h=this.core.h,v=this.core.v;
    let p=enc.beginComputePass();
    this.dispatch(p,c.pipelines.hflux,s.coreSlowGroups.hflux,h.edgeCount*v.nz);
    this.dispatch(p,c.pipelines.vflux,s.coreSlowGroups.vflux,h.cellCount*(v.nz+1));
    this.dispatch(p,c.pipelines.divergence,s.coreSlowGroups.divergence,h.cellCount*v.nz);p.end();
    p=enc.beginComputePass();
    for(const[name,count]of [['cellWind',h.cellCount*v.nz],['hMom',h.edgeCount*v.nz],['vMom',h.cellCount*(v.nz+1)],['rhoOld',h.cellCount*v.nz],['momDelta',h.cellCount*v.nz],['project',h.edgeCount*v.nz],['wAdvect',h.cellCount*(v.nz+1)]] as [string,number][]){this.dispatch(p,r.pipelines[name],s.rotGroups[name],count)}p.end();
    enc.copyBufferToBuffer(r.buffers.uTmp,0,c.buffers.u,0,h.edgeCount*v.nz*4);enc.copyBufferToBuffer(r.buffers.wTmp,0,c.buffers.w,0,h.cellCount*(v.nz+1)*4);
    if(heldSuarez){p=enc.beginComputePass();this.dispatch(p,r.pipelines.thermal,s.rotGroups.thermal,h.cellCount*v.nz);this.dispatch(p,r.pipelines.drag,s.rotGroups.drag,h.edgeCount*v.nz);p.end();}
  }
  private copyState(enc:GPUAny,target:Record<string,GPUAny>):void{
    const b=this.core.core.buffers,h=this.core.h,v=this.core.v;
    enc.copyBufferToBuffer(b.rho,0,target.rho,0,h.cellCount*v.nz*4);enc.copyBufferToBuffer(b.rhoTheta,0,target.rhoTheta,0,h.cellCount*v.nz*4);enc.copyBufferToBuffer(b.u,0,target.u,0,h.edgeCount*v.nz*4);enc.copyBufferToBuffer(b.w,0,target.w,0,h.cellCount*(v.nz+1)*4);
  }
  private encodeReanchor(enc:GPUAny):void{
    const counts:Record<string,number>={rho:this.core.h.cellCount*this.core.v.nz,rhoTheta:this.core.h.cellCount*this.core.v.nz,u:this.core.h.edgeCount*this.core.v.nz,w:this.core.h.cellCount*(this.core.v.nz+1)};
    for(const name of ['rho','rhoTheta','u','w']){const p=enc.beginComputePass();this.dispatch(p,this.reanchorPipeline,this.reanchorGroups[name],counts[name]!);p.end();}
  }
  private encodeStage(enc:GPUAny,s:StageResources,heldSuarez:boolean):void{
    const b=this.core.core.buffers,h=this.core.h,v=this.core.v;
    this.copyState(enc,this.predictor);
    this.encodeCoriolisHalf(enc,s);
    enc.clearBuffer(this.refMassAverage);
    for(let i=0;i<s.acousticSubsteps;i++)this.encodeFast(enc,s);
    enc.copyBufferToBuffer(this.refMassAverage,0,b.heviRefMass,0,h.cellCount*(v.nz+1)*4);
    this.encodeSlow(enc,s,heldSuarez);
    this.encodeCoriolisHalf(enc,s);
    this.encodeReanchor(enc);
  }
  private encodeOuter(enc:GPUAny,heldSuarez:boolean):void{
    this.copyState(enc,this.base);
    for(const s of this.stages)this.encodeStage(enc,s,heldSuarez);
  }

  step(dtOuter:number,heldSuarez=true):void{this.prepare(dtOuter);const enc=this.device.createCommandEncoder({label:`stage4 RK3 split outer dt=${dtOuter}`});this.encodeOuter(enc,heldSuarez);this.device.queue.submit([enc.finish()]);}
  stepBatch(dtOuter:number,count:number,heldSuarez=true):void{if(!Number.isInteger(count)||count<1)throw new Error('batch count must be positive integer');this.prepare(dtOuter);const enc=this.device.createCommandEncoder({label:`stage4 RK3 split batch outer=${dtOuter} n=${count}`});for(let i=0;i<count;i++)this.encodeOuter(enc,heldSuarez);this.device.queue.submit([enc.finish()]);}
  downloadState(time=0):Promise<DryState>{return this.core.downloadState(time);}
  destroy():void{
    this.divergence.destroy();this.refMassAverage.destroy?.();this.accumPipeline?.destroy?.();this.reanchorPipeline?.destroy?.();
    for(const b of Object.values(this.base))b.destroy?.();for(const b of Object.values(this.predictor))b.destroy?.();for(const b of Object.values(this.reanchorParams))b.destroy?.();
    for(const s of this.stages){s.rotFull.destroy?.();s.rotHalf.destroy?.();s.coreSlow.destroy?.();s.coreFast.destroy?.();s.divParams.destroy?.();s.accumParams.destroy?.();}
    this.core.destroy();
  }
}
