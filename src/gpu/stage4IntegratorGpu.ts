import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { DryState } from '../solver/state.js';
import { GpuAcousticDivergenceDamping } from './acousticDivergenceDampingGpu.js';
import { GpuRotatingDryCore } from './rotatingDryCoreGpu.js';

/**
 * Stage 4 GPU integrator: rotating core + per-timestep horizontal divergence
 * damping. The model-top gravity-wave absorber now lives inside the HEVI
 * vertical acoustic solve, so no separate post-step w sponge is applied.
 */
export class GpuStage4Integrator{
  readonly device:any;
  private constructor(
    public readonly core:GpuRotatingDryCore,
    public readonly divergence:GpuAcousticDivergenceDamping,
  ){this.device=core.device;}
  static async create(h:CubedSphereGrid,v:VerticalGrid,ref:ReferenceAtmosphere,state:DryState):Promise<GpuStage4Integrator>{
    const core=await GpuRotatingDryCore.create(h,v,ref,state);core.device.pushErrorScope?.('validation');
    try{
      const divergence=new GpuAcousticDivergenceDamping(core),err=await core.device.popErrorScope?.();
      if(err){divergence.destroy();core.destroy();throw new Error(`Stage 4 stabilizer WebGPU validation: ${err.message||err}`);}
      return new GpuStage4Integrator(core,divergence);
    }catch(e){try{await core.device.popErrorScope?.();}catch{}core.destroy();throw e;}
  }
  private prepare(dt:number):void{
    // Long-run and agreement gates use identical timestep ordering. Horizontal
    // divergence damping is normalized by physical dt; upper-level Rayleigh
    // damping is already embedded in core HEVI.
    (this.core as any).prepare(dt);
    this.divergence.prepare(dt);
  }
  private encodeOne(enc:any,heldSuarez:boolean):void{
    (this.core as any).encodeOne(enc,heldSuarez);
    this.divergence.encode(enc);
  }
  step(dt:number,heldSuarez=true):void{
    this.prepare(dt);const enc=this.device.createCommandEncoder({label:'stage4 full timestep'});this.encodeOne(enc,heldSuarez);this.device.queue.submit([enc.finish()]);
  }
  /** Batch complete Stage 4 timesteps into one command buffer without changing the numerical operator. */
  stepBatch(dt:number,count:number,heldSuarez=true):void{
    if(!Number.isInteger(count)||count<1)throw new Error('batch count must be positive integer');
    this.prepare(dt);const enc=this.device.createCommandEncoder({label:`stage4 full batch ${count}`});
    for(let i=0;i<count;i++)this.encodeOne(enc,heldSuarez);
    this.device.queue.submit([enc.finish()]);
  }
  downloadState(time=0):Promise<DryState>{return this.core.downloadState(time);}
  destroy():void{this.divergence.destroy();this.core.destroy();}
}
