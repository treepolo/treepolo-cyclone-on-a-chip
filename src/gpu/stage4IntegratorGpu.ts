import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { DryState } from '../solver/state.js';
import { GpuAcousticDivergenceDamping } from './acousticDivergenceDampingGpu.js';
import { GpuModelTopSponge } from './modelTopSpongeGpu.js';
import { GpuRotatingDryCore } from './rotatingDryCoreGpu.js';

/** Stage 4 GPU integrator: rotating core + acoustic-divergence filter + top sponge. */
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
  step(dt:number,heldSuarez=true):void{this.core.step(dt,heldSuarez);this.divergence.apply();this.sponge.apply(dt);}
  /**
   * `stabilizerStride` controls only how often the slow JS-side stabilizer passes
   * are submitted. Core dynamics still advances every dt. The long-run Stage 4
   * gate uses stride=10 (100 s at dt=10 s), far shorter than the horizontal
   * acoustic crossing time of this very coarse global grid.
   */
  stepBatch(dt:number,count:number,heldSuarez=true,stabilizerStride=1):void{
    if(!Number.isInteger(count)||count<1)throw new Error('batch count must be positive integer');
    if(!Number.isInteger(stabilizerStride)||stabilizerStride<1)throw new Error('stabilizerStride must be positive integer');
    let left=count;
    while(left>0){const n=Math.min(stabilizerStride,left);this.core.stepBatch(dt,n,heldSuarez);this.divergence.apply();this.sponge.apply(dt*n);left-=n;}
  }
  downloadState(time=0):Promise<DryState>{return this.core.downloadState(time);}
  destroy():void{this.divergence.destroy();this.sponge.destroy();this.core.destroy();}
}
