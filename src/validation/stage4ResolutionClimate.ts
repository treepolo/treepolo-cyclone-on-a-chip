import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { GpuStage4Rk3SplitReference } from '../gpu/stage4Rk3SplitGpu.js';
import { acousticDivergenceRms } from '../physics/acousticDivergenceDamping.js';
import { addHeldSuarezWavePerturbation, buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { MODEL_TOP_SPONGE, buildModelTopSpongeRates } from '../physics/modelTopSponge.js';
import { diagnoseState } from '../solver/diagnostics.js';
import { diagnoseZonalMeans, type ZonalMeanDiagnostics } from '../solver/stage4Diagnostics.js';
import { createHydrostaticState, type DryState } from '../solver/state.js';
import type { ClimateDaySample } from './stage4Gpu.js';

const NZ=48,TOP=40000,STRETCH=1.4,DT=10,ACOUSTIC_RATIO=4,BATCH=40,ZONAL_BINS=24;
async function yieldToBrowser():Promise<void>{const scheduler=(globalThis as any).scheduler;if(typeof scheduler?.yield==='function')await scheduler.yield();else await new Promise<void>(r=>setTimeout(r,0));}
export interface ResolutionClimateProgress{horizontalN:number;simulatedDay:number;targetDays:number;completedOuterSteps:number;totalOuterSteps:number;elapsedMs:number;}
export interface ResolutionClimateResult{horizontalN:number;days:number;samples:ClimateDaySample[];failures:string[];elapsedMs:number;finalState:DryState;finalZonal:ZonalMeanDiagnostics;}

function extra(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,ref:ReturnType<typeof buildHeldSuarezReference>,s:DryState){
  const spongeStart=MODEL_TOP_SPONGE.startFraction*v.top,wStride=v.nz+1;let maxWBelowSponge=0,maxWInSponge=0,maxW=0,maxWAltitude=0,maxWLatitude=0,maxEdgeWind=0,maxHorizontalCfl=0,maxVerticalCfl=0;
  for(let c=0;c<h.cellCount;c++){const lat=Math.asin(Math.max(-1,Math.min(1,h.cellCenters[c*3+2]!)))*180/Math.PI;for(let i=1;i<v.nz;i++){const z=v.zInterface[i]!,aw=Math.abs(s.wInterface[c*wStride+i]!);if(z>=spongeStart)maxWInSponge=Math.max(maxWInSponge,aw);else maxWBelowSponge=Math.max(maxWBelowSponge,aw);maxVerticalCfl=Math.max(maxVerticalCfl,aw*DT/Math.min(v.dz[i-1]!,v.dz[i]!));if(aw>maxW){maxW=aw;maxWAltitude=z;maxWLatitude=lat;}}}
  for(let e=0;e<h.edgeCount;e++){const dist=Math.max(h.edges[e]!.centerDistanceAngle*EARTH.radius,1);for(let k=0;k<v.nz;k++){const au=Math.abs(s.uEdge[e*v.nz+k]!);maxEdgeWind=Math.max(maxEdgeWind,au);maxHorizontalCfl=Math.max(maxHorizontalCfl,au*DT/dist);}}
  return{maxWBelowSponge,maxWInSponge,maxWAltitude,maxWLatitude,maxEdgeWind,divergenceRms:acousticDivergenceRms(h,v,ref,s),maxHorizontalCfl,maxVerticalCfl};
}

/**
 * Resolution experiment using exactly the Stage 4 production equations and
 * time stepping.  No climate target, nudging, checkpoint resume, or result
 * correction is applied.  horizontalN is the only horizontal-grid control.
 */
export async function runStage4ResolutionClimate(horizontalN:number,days:number,onSample?:(s:ClimateDaySample)=>void,onProgress?:(p:ResolutionClimateProgress)=>void):Promise<ResolutionClimateResult>{
  if(!Number.isInteger(horizontalN)||horizontalN<4)throw new Error('horizontalN must be an integer >= 4');
  if(!Number.isInteger(days)||days<1)throw new Error('days must be a positive integer');
  const h=buildCubedSphere(horizontalN),v=buildStretchedVerticalGrid(NZ,TOP,STRETCH),ref=buildHeldSuarezReference(v),rates=buildModelTopSpongeRates(v),active=Array.from(rates.slice(1,-1)).filter(x=>x>0).length;
  if(active<6)throw new Error(`model-top sponge under-resolved: ${active} active interfaces`);
  const seed=createHydrostaticState(h,v,ref);addHeldSuarezWavePerturbation(h,v,ref,seed,.05);
  const gpu=await GpuStage4Rk3SplitReference.create(h,v,ref,seed,ACOUSTIC_RATIO),samples:ClimateDaySample[]=[],failures:string[]=[],t0=performance.now(),stepsPerQuarter=Math.round(21600/DT),totalOuterSteps=days*Math.round(86400/DT),opts={heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true} as const;
  let completedOuterSteps=0,finalState=await gpu.downloadState(0),finalZonal=diagnoseZonalMeans(h,v,finalState,ZONAL_BINS);const m0=diagnoseState(h,v,finalState).dryMass;
  try{
    onProgress?.({horizontalN,simulatedDay:0,targetDays:days,completedOuterSteps,totalOuterSteps,elapsedMs:0});
    for(let segment=1;segment<=days*4;segment++){
      let left=stepsPerQuarter;while(left>0){const n=Math.min(BATCH,left);gpu.stepBatch(DT,n,opts);await gpu.device.queue.onSubmittedWorkDone();left-=n;completedOuterSteps+=n;onProgress?.({horizontalN,simulatedDay:completedOuterSteps*DT/86400,targetDays:days,completedOuterSteps,totalOuterSteps,elapsedMs:performance.now()-t0});await yieldToBrowser();}
      const day=segment/4;finalState=await gpu.downloadState(day*86400);const d=diagnoseState(h,v,finalState);finalZonal=diagnoseZonalMeans(h,v,finalState,ZONAL_BINS);const x=extra(h,v,ref,finalState),s:ClimateDaySample={day,massDrift:(d.dryMass-m0)/m0,maxW:d.maxAbsW,jet:finalZonal.maxUpperMidlatitudeWesterly,trade:finalZonal.meanTropicalLowLevelZonal,psi:finalZonal.maxAbsStreamfunction,nhPsi:finalZonal.nhDominantStreamfunction,shPsi:finalZonal.shDominantStreamfunction,invalid:d.nan||d.minRho<=0||d.minP<=0,...x};samples.push(s);onSample?.(s);
      if(s.invalid){failures.push(`day ${day}: invalid state`);break;}if(Math.abs(s.massDrift)>5e-5){failures.push(`day ${day}: mass drift ${s.massDrift}`);break;}if(!(s.maxW<10)){failures.push(`day ${day}: vertical velocity guard ${s.maxW}`);break;}
    }
    if(samples.length===0||samples[samples.length-1]!.day<days)failures.push(`resolution run stopped before day ${days}`);
    return{horizontalN,days,samples,failures,elapsedMs:performance.now()-t0,finalState,finalZonal};
  }finally{gpu.destroy();}
}
