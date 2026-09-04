import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { buildRk3SplitSchedule } from '../solver/rk3SplitSchedule.js';
import { advanceStage4AcousticSmallStep, computeStage4FrozenRhs } from '../solver/stage4Rk3SplitCpu.js';
import { cloneState, createHydrostaticState, edge3DIndex, w3DIndex } from '../solver/state.js';
import { GpuStage4Rk3SplitReference } from '../gpu/stage4Rk3SplitGpu.js';

export interface Stage4Rk3StageDiagnosticSample{stage:number;targetFraction:number;rhoRelativeL2:number;rhoThetaRelativeL2:number;maxDeltaU:number;maxDeltaW:number;}
function relL2(a:ArrayLike<number>,b:ArrayLike<number>):number{let n=0,d=0;for(let i=0;i<a.length;i++){const e=a[i]!-b[i]!;n+=e*e;d+=b[i]!*b[i]!;}return Math.sqrt(n/Math.max(d,Number.MIN_VALUE));}
function maxDiff(a:ArrayLike<number>,b:ArrayLike<number>):number{let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]!-b[i]!));return m;}

/** Diagnostic-only stage snapshots. It intentionally reaches the isolated
 * reference integrator's private orchestration through `any`; no production API
 * is expanded solely for a failure diagnostic. */
export async function runStage4Rk3GpuStageDiagnostic():Promise<Stage4Rk3StageDiagnosticSample[]>{
  const h=buildCubedSphere(2),v=buildStretchedVerticalGrid(24,40000,1.35),ref=buildHeldSuarezReference(v),dt=10,nz=v.nz,base=createHydrostaticState(h,v,ref),schedule=buildRk3SplitSchedule(4);
  for(let c=0;c<h.cellCount;c++)for(let k=0;k<nz;k++){const q=c*nz+k,z=v.zCenter[k]!/v.top,phase=(c+1)*.173+(k+1)*.287;base.rhoD[q]=base.rhoD[q]!*(1+1.8e-4*Math.sin(phase)*Math.sin(Math.PI*z));base.rhoThetaM[q]=base.rhoThetaM[q]!*(1+2.2e-4*Math.cos(phase*.83)*Math.sin(Math.PI*z));}
  for(let e=0;e<h.edgeCount;e++)for(let k=0;k<nz;k++)base.uEdge[edge3DIndex(e,k,nz)]=1.4*Math.sin((e+1)*.137+(k+1)*.211)*(.35+.65*Math.cos(.5*Math.PI*v.zCenter[k]!/v.top));
  for(let c=0;c<h.cellCount;c++)for(let i=1;i<nz;i++)base.wInterface[w3DIndex(c,i,nz)]=.025*Math.sin(Math.PI*v.zInterface[i]!/v.top)*Math.sin((c+1)*.229+i*.317);
  const opts={heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true} as const;

  // CPU predictor sequence, explicitly preserving immutable large-step base.
  const cpuStages:ReturnType<typeof cloneState>[]=[];let cpuPred=cloneState(base);
  for(const stage of schedule){const frozen=computeStage4FrozenRhs(h,v,ref,cpuPred,opts),acoustic=cloneState(base),dtFast=dt*stage.acousticDtFraction;for(let n=0;n<stage.acousticSteps;n++)advanceStage4AcousticSmallStep(h,v,ref,cpuPred,acoustic,frozen,dtFast,opts);cpuPred=acoustic;cpuStages.push(cloneState(cpuPred));}

  const gpu=await GpuStage4Rk3SplitReference.create(h,v,ref,base,4),g:any=gpu,samples:Stage4Rk3StageDiagnosticSample[]=[];
  try{
    g.snapshotBase();
    for(let si=0;si<schedule.length;si++){
      const stage=schedule[si]!,dtFast=dt*stage.acousticDtFraction;gpu.slow.compute(true,true,true);g.prepareStage(dtFast,true);g.advanceStage(dtFast,stage.acousticSteps,true);const gs=await gpu.downloadState(stage.targetFraction*dt),cs=cpuStages[si]!;
      samples.push({stage:si+1,targetFraction:stage.targetFraction,rhoRelativeL2:relL2(gs.rhoD,cs.rhoD),rhoThetaRelativeL2:relL2(gs.rhoThetaM,cs.rhoThetaM),maxDeltaU:maxDiff(gs.uEdge,cs.uEdge),maxDeltaW:maxDiff(gs.wInterface,cs.wInterface)});
    }
    return samples;
  }finally{gpu.destroy();}
}
