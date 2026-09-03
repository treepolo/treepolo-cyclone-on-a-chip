import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { DryState } from '../solver/state.js';
import { GpuAcousticDivergenceDamping } from './acousticDivergenceDampingGpu.js';
import { GpuModelTopSponge } from './modelTopSpongeGpu.js';
import { GpuRotatingDryCore } from './rotatingDryCoreGpu.js';

/** Stage 4 GPU integrator: rotating core + per-timestep acoustic-divergence filter + top sponge. */
export class GpuStage4Integrator{
  readonly device:any;
  private constructor(
    public readonly core:GpuRotatingDryCore,
    public readonly divergence:GpuAcousticDivergenceDamping,
    public readonly sponge:GpuModelTopSponge,
  ){this.device=core.device;}
  static async create(h:CubedSphereGrid,v:VerticalGrid,ref:ReferenceAtmosphere,state:DryState):Promise<GpuStage4Integrator>{
    const core=await GpuRotatingDryCore.create(h,v,ref,state);core.device.pushErrorScope?.('validation');
    try{
      const divergence=new GpuAcousticDivergenceDamping(core),sponge=new GpuModelTopSponge(core),err=await core.device.popErrorScope?.();
      if(err){divergence.destroy();sponge.destroy();core.destroy();throw new Error(`Stage 4 stabilizer WebGPU validation: ${err.message||err}`);}
      return new GpuStage4Integrator(core,divergence,sponge);
    }catch(e){try{await core.device.popErrorScope?.();}catch{}core.destroy();throw e;}
  }
  private prepare(dt:number):void{
    // GpuRotatingDryCore keeps these encoder helpers private to its implementation;
    // Stage 4 intentionally composes them here so the long-run and agreement gates
    // use the exact same per-timestep operator ordering without extra queue submits.
    (this.core as any).prepare(dt);
    this.divergence.prepare();
    this.sponge.prepare(dt);
  }
  private encodeOne(enc:any,heldSuarez:boolean):void{
    (this.core as any).encodeOne(enc,heldSuarez);
    this.divergence.encode(enc);
    this.sponge.encode(enc);
  }
  step(dt:number,heldSuarez=true):void{
    this.prepare(dt);const enc=this.device.createCommandEncoder({label:'stage4 full timestep'});this.encodeOne(enc,heldSuarez);this.device.queue.submit([enc.finish()]);
  }
  /**
   * Batch many complete Stage 4 timesteps into one command buffer. Every timestep
   * receives the same operator sequence as `step`: rotating dry core -> horizontal
   * divergence damping -> model-top sponge. This is a correctness requirement, not
   * merely a performance choice.
   */
  stepBatch(dt:number,count:number,heldSuarez=true):void{
    if(!Number.isInteger(count)||count<1)throw new Error('batch count must be positive integer');
    this.prepare(dt);const enc=this.device.createCommandEncoder({label:`stage4 full batch ${count}`});
    for(let i=0;i<count;i++)this.encodeOne(enc,heldSuarez);
    this.device.queue.submit([enc.finish()]);
  }
  downloadState(time=0):Promise<DryState>{return this.core.downloadState(time);}
  destroy():void{this.divergence.destroy();this.sponge.destroy();this.core.destroy();}
}
