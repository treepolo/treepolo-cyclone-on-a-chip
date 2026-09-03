import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { DryState } from '../solver/state.js';
import { GpuModelTopSponge } from './modelTopSpongeGpu.js';
import { GpuRotatingDryCore } from './rotatingDryCoreGpu.js';

/**
 * Stage 4 production-facing GPU integrator. The rotating dry core and the
 * artificial-model-top absorbing layer are advanced as operator-split pieces.
 * A spongeStride of 1 applies the boundary treatment every atmospheric step;
 * larger strides are allowed for long development runs to reduce submission
 * overhead, but the accumulated damping uses the exact elapsed time.
 */
export class GpuStage4Integrator{
  readonly device:any;
  private constructor(public readonly core:GpuRotatingDryCore,public readonly sponge:GpuModelTopSponge){this.device=core.device;}
  static async create(h:CubedSphereGrid,v:VerticalGrid,ref:ReferenceAtmosphere,state:DryState):Promise<GpuStage4Integrator>{
    const core=await GpuRotatingDryCore.create(h,v,ref,state);
    core.device.pushErrorScope?.('validation');
    try{
      const sponge=new GpuModelTopSponge(core);
      const err=await core.device.popErrorScope?.();
      if(err){sponge.destroy();core.destroy();throw new Error(`Stage 4 sponge WebGPU validation: ${err.message||err}`);}
      return new GpuStage4Integrator(core,sponge);
    }catch(e){
      try{await core.device.popErrorScope?.();}catch{}
      core.destroy();throw e;
    }
  }
  step(dt:number,heldSuarez=true):void{this.core.step(dt,heldSuarez);this.sponge.apply(dt);}
  stepBatch(dt:number,count:number,heldSuarez=true,spongeStride=1):void{
    if(!Number.isInteger(count)||count<1)throw new Error('batch count must be positive integer');
    if(!Number.isInteger(spongeStride)||spongeStride<1)throw new Error('spongeStride must be positive integer');
    let left=count;
    while(left>0){const n=Math.min(spongeStride,left);this.core.stepBatch(dt,n,heldSuarez);this.sponge.apply(dt*n);left-=n;}
  }
  downloadState(time=0):Promise<DryState>{return this.core.downloadState(time);}
  destroy():void{this.sponge.destroy();this.core.destroy();}
}
