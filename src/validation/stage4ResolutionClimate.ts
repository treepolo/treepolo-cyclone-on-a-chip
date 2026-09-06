import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { GpuStage4Rk3SplitReference } from '../gpu/stage4Rk3SplitGpu.js';
import {
  STAGE4_RK3_CHECKPOINT_SCHEMA_VERSION,
  assertStage4Rk3CheckpointCompatible,
  type Stage4Rk3ClimateCheckpoint,
} from '../persistence/stage4Rk3Checkpoint.js';
import { stage4ResolutionModelSignature } from '../persistence/stage4ResolutionCheckpoint.js';
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
export interface ResolutionClimateRunOptions{resume?:Stage4Rk3ClimateCheckpoint|null;onCheckpoint?:(checkpoint:Stage4Rk3ClimateCheckpoint)=>void|Promise<void>;}

function extra(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,ref:ReturnType<typeof buildHeldSuarezReference>,s:DryState){
  const spongeStart=MODEL_TOP_SPONGE.startFraction*v.top,wStride=v.nz+1;let maxWBelowSponge=0,maxWInSponge=0,maxW=0,maxWAltitude=0,maxWLatitude=0,maxEdgeWind=0,maxHorizontalCfl=0,maxVerticalCfl=0;
  for(let c=0;c<h.cellCount;c++){const lat=Math.asin(Math.max(-1,Math.min(1,h.cellCenters[c*3+2]!)))*180/Math.PI;for(let i=1;i<v.nz;i++){const z=v.zInterface[i]!,aw=Math.abs(s.wInterface[c*wStride+i]!);if(z>=spongeStart)maxWInSponge=Math.max(maxWInSponge,aw);else maxWBelowSponge=Math.max(maxWBelowSponge,aw);maxVerticalCfl=Math.max(maxVerticalCfl,aw*DT/Math.min(v.dz[i-1]!,v.dz[i]!));if(aw>maxW){maxW=aw;maxWAltitude=z;maxWLatitude=lat;}}}
  for(let e=0;e<h.edgeCount;e++){const dist=Math.max(h.edges[e]!.centerDistanceAngle*EARTH.radius,1);for(let k=0;k<v.nz;k++){const au=Math.abs(s.uEdge[e*v.nz+k]!);maxEdgeWind=Math.max(maxEdgeWind,au);maxHorizontalCfl=Math.max(maxHorizontalCfl,au*DT/dist);}}
  return{maxWBelowSponge,maxWInSponge,maxWAltitude,maxWLatitude,maxEdgeWind,divergenceRms:acousticDivergenceRms(h,v,ref,s),maxHorizontalCfl,maxVerticalCfl};
}

function assertResumeState(cp:Stage4Rk3ClimateCheckpoint,h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,horizontalN:number,days:number,stepsPerQuarter:number,totalOuterSteps:number):void{
  assertStage4Rk3CheckpointCompatible(cp,stage4ResolutionModelSignature(horizontalN),days);
  if(!Number.isInteger(cp.completedOuterSteps)||cp.completedOuterSteps<0||cp.completedOuterSteps>totalOuterSteps)throw new Error(`resolution checkpoint completed step invalid: ${cp.completedOuterSteps}`);
  if(cp.completedOuterSteps%stepsPerQuarter!==0)throw new Error(`resolution checkpoint is not on a quarter-day boundary: ${cp.completedOuterSteps}`);
  const expectedSamples=cp.completedOuterSteps/stepsPerQuarter,expectedTime=cp.completedOuterSteps*DT;
  if(cp.samples.length!==expectedSamples)throw new Error(`resolution checkpoint sample count mismatch: ${cp.samples.length}; expected ${expectedSamples}`);
  if(Math.abs(cp.state.time-expectedTime)>1e-6)throw new Error(`resolution checkpoint state time mismatch: ${cp.state.time}; expected ${expectedTime}`);
  const cells=h.cellCount*v.nz,edges=h.edgeCount*v.nz,w=h.cellCount*(v.nz+1);
  if(cp.state.rhoD.length!==cells||cp.state.rhoThetaM.length!==cells||cp.state.uEdge.length!==edges||cp.state.wInterface.length!==w)throw new Error('resolution checkpoint prognostic arrays do not match the requested grid');
  if(!(cp.initialDryMass>0)||!Number.isFinite(cp.initialDryMass))throw new Error('resolution checkpoint initial dry mass invalid');
  if(expectedSamples>0){const last=cp.samples[expectedSamples-1]!,expectedDay=expectedTime/86400;if(Math.abs(last.day-expectedDay)>1e-10)throw new Error(`resolution checkpoint last sample day mismatch: ${last.day}; expected ${expectedDay}`);}
}

/** Resolution experiment using exactly the Stage 4 production equations and time stepping. */
export async function runStage4ResolutionClimate(horizontalN:number,days:number,onSample?:(s:ClimateDaySample)=>void,onProgress?:(p:ResolutionClimateProgress)=>void,runOptions:ResolutionClimateRunOptions={}):Promise<ResolutionClimateResult>{
  if(!Number.isInteger(horizontalN)||horizontalN<4)throw new Error('horizontalN must be an integer >= 4');
  if(!Number.isInteger(days)||days<1)throw new Error('days must be a positive integer');
  const h=buildCubedSphere(horizontalN),v=buildStretchedVerticalGrid(NZ,TOP,STRETCH),ref=buildHeldSuarezReference(v),rates=buildModelTopSpongeRates(v),active=Array.from(rates.slice(1,-1)).filter(x=>x>0).length;
  if(active<6)throw new Error(`model-top sponge under-resolved: ${active} active interfaces`);
  // Keep the production equations and outer dt unchanged. At N16+, use shorter
  // host submissions so Windows/D3D12 gets a queue completion + browser yield
  // four times as often; this avoids making 40 full RK3 outer steps one long GPU
  // submission without changing the simulated trajectory.
  const batchOuterSteps=horizontalN>=16?10:BATCH;
  const stepsPerQuarter=Math.round(21600/DT),totalOuterSteps=days*Math.round(86400/DT),seed=createHydrostaticState(h,v,ref);addHeldSuarezWavePerturbation(h,v,ref,seed,.05);
  const resume=runOptions.resume??null;if(resume)assertResumeState(resume,h,v,horizontalN,days,stepsPerQuarter,totalOuterSteps);
  const startState=resume?.state??seed,gpu=await GpuStage4Rk3SplitReference.create(h,v,ref,startState,ACOUSTIC_RATIO),samples:ClimateDaySample[]=resume?resume.samples.map(s=>({...s})):[],failures:string[]=[],t0=performance.now(),opts={heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true} as const;
  let completedOuterSteps=resume?.completedOuterSteps??0,finalState=startState,finalZonal=diagnoseZonalMeans(h,v,finalState,ZONAL_BINS);const m0=resume?.initialDryMass??diagnoseState(h,v,finalState).dryMass;
  try{
    onProgress?.({horizontalN,simulatedDay:completedOuterSteps*DT/86400,targetDays:days,completedOuterSteps,totalOuterSteps,elapsedMs:0});
    const firstSegment=completedOuterSteps/stepsPerQuarter+1;
    for(let segment=firstSegment;segment<=days*4;segment++){
      let left=stepsPerQuarter;while(left>0){const n=Math.min(batchOuterSteps,left);gpu.stepBatch(DT,n,opts);await gpu.device.queue.onSubmittedWorkDone();left-=n;completedOuterSteps+=n;onProgress?.({horizontalN,simulatedDay:completedOuterSteps*DT/86400,targetDays:days,completedOuterSteps,totalOuterSteps,elapsedMs:performance.now()-t0});await yieldToBrowser();}
      const day=segment/4;finalState=await gpu.downloadState(day*86400);const d=diagnoseState(h,v,finalState);finalZonal=diagnoseZonalMeans(h,v,finalState,ZONAL_BINS);const x=extra(h,v,ref,finalState),s:ClimateDaySample={day,massDrift:(d.dryMass-m0)/m0,maxW:d.maxAbsW,jet:finalZonal.maxUpperMidlatitudeWesterly,trade:finalZonal.meanTropicalLowLevelZonal,psi:finalZonal.maxAbsStreamfunction,nhPsi:finalZonal.nhDominantStreamfunction,shPsi:finalZonal.shDominantStreamfunction,invalid:d.nan||d.minRho<=0||d.minP<=0,...x};samples.push(s);onSample?.(s);
      if(s.invalid){failures.push(`day ${day}: invalid state`);break;}if(Math.abs(s.massDrift)>5e-5){failures.push(`day ${day}: mass drift ${s.massDrift}`);break;}if(!(s.maxW<10)){failures.push(`day ${day}: vertical velocity guard ${s.maxW}`);break;}
      if(runOptions.onCheckpoint)await runOptions.onCheckpoint({schemaVersion:STAGE4_RK3_CHECKPOINT_SCHEMA_VERSION,modelSignature:stage4ResolutionModelSignature(horizontalN),savedAt:Date.now(),targetDays:days,completedOuterSteps,initialDryMass:m0,state:finalState,samples:samples.map(sample=>({...sample}))});
    }
    if(samples.length===0||samples[samples.length-1]!.day<days)failures.push(`resolution run stopped before day ${days}`);
    return{horizontalN,days,samples,failures,elapsedMs:performance.now()-t0,finalState,finalZonal};
  }finally{gpu.destroy();}
}
