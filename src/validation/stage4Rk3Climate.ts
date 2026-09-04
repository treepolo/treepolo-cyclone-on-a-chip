import { EARTH } from '../core/constants.js';
import { buildCubedSphere, CubedSphereGrid } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid, VerticalGrid } from '../grid/vertical.js';
import { acousticDivergenceRms } from '../physics/acousticDivergenceDamping.js';
import { addHeldSuarezWavePerturbation, buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { buildModelTopSpongeRates, MODEL_TOP_SPONGE } from '../physics/modelTopSponge.js';
import { diagnoseState } from '../solver/diagnostics.js';
import { diagnoseZonalMeans, ZonalMeanDiagnostics } from '../solver/stage4Diagnostics.js';
import { createHydrostaticState, DryState } from '../solver/state.js';
import { GpuStage4Rk3SplitReference } from '../gpu/stage4Rk3SplitGpu.js';
import {
  assertStage4Rk3CheckpointCompatible,
  STAGE4_RK3_CHECKPOINT_SCHEMA_VERSION,
  STAGE4_RK3_PRODUCTION_MODEL_SIGNATURE,
  type Stage4Rk3ClimateCheckpoint,
} from '../persistence/stage4Rk3Checkpoint.js';
import { ClimateDaySample } from './stage4Gpu.js';

const HORIZONTAL_N=8;
const VERTICAL_NZ=48;
const TOP_METERS=40000;
const VERTICAL_STRETCH=1.4;
const ZONAL_BINS=24;
const MIN_ACTIVE_SPONGE_INTERFACES=6;
const DT=10;
const ACOUSTIC_RATIO=4;
// 40 outer steps is the largest batch that has already passed real-device
// CPU/GPU agreement.  Do not silently increase this for the climate gate.
const PRODUCTION_BATCH=40;

async function yieldToBrowser():Promise<void>{
  const scheduler=(globalThis as any).scheduler;
  if(typeof scheduler?.yield==='function'){
    await scheduler.yield();
    return;
  }
  await new Promise<void>(resolve=>setTimeout(resolve,0));
}

interface ExtraDiagnostics{
  maxWBelowSponge:number;maxWInSponge:number;maxWAltitude:number;maxWLatitude:number;
  maxEdgeWind:number;divergenceRms:number;maxHorizontalCfl:number;maxVerticalCfl:number;
}
function extra(h:CubedSphereGrid,v:VerticalGrid,s:DryState,dt:number):ExtraDiagnostics{
  const spongeStart=MODEL_TOP_SPONGE.startFraction*v.top,wStride=v.nz+1;
  let maxWBelowSponge=0,maxWInSponge=0,maxW=0,maxWAltitude=0,maxWLatitude=0,maxEdgeWind=0,maxHorizontalCfl=0,maxVerticalCfl=0;
  for(let c=0;c<h.cellCount;c++){
    const lat=Math.asin(Math.max(-1,Math.min(1,h.cellCenters[c*3+2]!)))*180/Math.PI;
    for(let i=1;i<v.nz;i++){
      const z=v.zInterface[i]!,aw=Math.abs(s.wInterface[c*wStride+i]!);
      if(z>=spongeStart)maxWInSponge=Math.max(maxWInSponge,aw);else maxWBelowSponge=Math.max(maxWBelowSponge,aw);
      maxVerticalCfl=Math.max(maxVerticalCfl,aw*dt/Math.min(v.dz[i-1]!,v.dz[i]!));
      if(aw>maxW){maxW=aw;maxWAltitude=z;maxWLatitude=lat;}
    }
  }
  for(let e=0;e<h.edgeCount;e++){
    const dist=Math.max(h.edges[e]!.centerDistanceAngle*EARTH.radius,1);
    for(let k=0;k<v.nz;k++){
      const au=Math.abs(s.uEdge[e*v.nz+k]!);maxEdgeWind=Math.max(maxEdgeWind,au);maxHorizontalCfl=Math.max(maxHorizontalCfl,au*dt/dist);
    }
  }
  return{maxWBelowSponge,maxWInSponge,maxWAltitude,maxWLatitude,maxEdgeWind,divergenceRms:acousticDivergenceRms(h,v,buildHeldSuarezReference(v),s),maxHorizontalCfl,maxVerticalCfl};
}

export interface Stage4Rk3ClimateProgress{
  simulatedDay:number;
  targetDays:number;
  completedOuterSteps:number;
  totalOuterSteps:number;
  batchSize:number;
  elapsedMs:number;
}
export interface Stage4Rk3ClimateRunOptions{
  resume?:Stage4Rk3ClimateCheckpoint|null;
  onCheckpoint?:(checkpoint:Stage4Rk3ClimateCheckpoint)=>void|Promise<void>;
}
export interface Stage4Rk3ClimateResult{passed:boolean;samples:ClimateDaySample[];failures:string[];elapsedMs:number;finalZonal?:ZonalMeanDiagnostics;}

function assertResumeState(cp:Stage4Rk3ClimateCheckpoint,h:CubedSphereGrid,v:VerticalGrid,days:number,stepsPerQuarter:number,totalOuterSteps:number):void{
  assertStage4Rk3CheckpointCompatible(cp,STAGE4_RK3_PRODUCTION_MODEL_SIGNATURE,days);
  if(!Number.isInteger(cp.completedOuterSteps)||cp.completedOuterSteps<0||cp.completedOuterSteps>totalOuterSteps)throw new Error(`checkpoint completed step invalid: ${cp.completedOuterSteps}`);
  if(cp.completedOuterSteps%stepsPerQuarter!==0)throw new Error(`checkpoint is not on a quarter-day boundary: step ${cp.completedOuterSteps}`);
  const expectedSamples=cp.completedOuterSteps/stepsPerQuarter;
  if(cp.samples.length!==expectedSamples)throw new Error(`checkpoint sample count mismatch: ${cp.samples.length}; expected ${expectedSamples}`);
  const expectedTime=cp.completedOuterSteps*DT;
  if(Math.abs(cp.state.time-expectedTime)>1e-6)throw new Error(`checkpoint state time mismatch: ${cp.state.time}; expected ${expectedTime}`);
  const cells=h.cellCount*v.nz,edges=h.edgeCount*v.nz,w=h.cellCount*(v.nz+1);
  if(cp.state.rhoD.length!==cells||cp.state.rhoThetaM.length!==cells||cp.state.uEdge.length!==edges||cp.state.wInterface.length!==w)throw new Error('checkpoint prognostic array dimensions do not match the production grid');
  if(!(cp.initialDryMass>0)||!Number.isFinite(cp.initialDryMass))throw new Error('checkpoint initial dry mass invalid');
  if(expectedSamples>0){
    const last=cp.samples[expectedSamples-1]!;
    const expectedDay=expectedTime/86400;
    if(Math.abs(last.day-expectedDay)>1e-10)throw new Error(`checkpoint last sample day mismatch: ${last.day}; expected ${expectedDay}`);
  }
}

export async function runStage4Rk3Climate(days=30,onSample?:(s:ClimateDaySample)=>void,onProgress?:(p:Stage4Rk3ClimateProgress)=>void,runOptions:Stage4Rk3ClimateRunOptions={}):Promise<Stage4Rk3ClimateResult>{
  if(!Number.isInteger(days)||days<1)throw new Error('RK3 climate days must be a positive integer');
  const h=buildCubedSphere(HORIZONTAL_N),v=buildStretchedVerticalGrid(VERTICAL_NZ,TOP_METERS,VERTICAL_STRETCH),rates=buildModelTopSpongeRates(v),active=Array.from(rates.slice(1,-1)).filter(x=>x>0).length;
  if(active<MIN_ACTIVE_SPONGE_INTERFACES)throw new Error(`RK3 climate sponge under-resolved: ${active} active interfaces; require >=${MIN_ACTIVE_SPONGE_INTERFACES}`);
  const stepsPerQuarter=Math.round(21600/DT),stepsPerDay=Math.round(86400/DT),totalOuterSteps=days*stepsPerDay;
  const ref=buildHeldSuarezReference(v),seed=createHydrostaticState(h,v,ref);addHeldSuarezWavePerturbation(h,v,ref,seed,.05);
  const resume=runOptions.resume??null;
  if(resume)assertResumeState(resume,h,v,days,stepsPerQuarter,totalOuterSteps);
  const startState=resume?.state??seed;
  const gpu=await GpuStage4Rk3SplitReference.create(h,v,ref,startState,ACOUSTIC_RATIO);
  const samples:ClimateDaySample[]=resume?resume.samples.map(s=>({...s})):[];
  const failures:string[]=[],t0=performance.now(),opts={heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true} as const;
  let finalZonal:ZonalMeanDiagnostics|undefined,completedOuterSteps=resume?.completedOuterSteps??0;
  const m0=resume?.initialDryMass??diagnoseState(h,v,await gpu.downloadState(0)).dryMass;
  try{
    onProgress?.({simulatedDay:completedOuterSteps*DT/86400,targetDays:days,completedOuterSteps,totalOuterSteps,batchSize:0,elapsedMs:0});
    const firstSegment=completedOuterSteps/stepsPerQuarter+1;
    for(let segment=firstSegment;segment<=days*4;segment++){
      let left=stepsPerQuarter;
      while(left>0){
        const n=Math.min(PRODUCTION_BATCH,left);
        gpu.stepBatch(DT,n,opts);
        await gpu.device.queue.onSubmittedWorkDone();
        left-=n;completedOuterSteps+=n;
        onProgress?.({simulatedDay:completedOuterSteps*DT/86400,targetDays:days,completedOuterSteps,totalOuterSteps,batchSize:n,elapsedMs:performance.now()-t0});
        await yieldToBrowser();
      }
      const day=segment/4,state=await gpu.downloadState(day*86400),d=diagnoseState(h,v,state),z=diagnoseZonalMeans(h,v,state,ZONAL_BINS),x=extra(h,v,state,DT),s:ClimateDaySample={day,massDrift:(d.dryMass-m0)/m0,maxW:d.maxAbsW,jet:z.maxUpperMidlatitudeWesterly,trade:z.meanTropicalLowLevelZonal,psi:z.maxAbsStreamfunction,nhPsi:z.nhDominantStreamfunction,shPsi:z.shDominantStreamfunction,invalid:d.nan||d.minRho<=0||d.minP<=0,...x};
      samples.push(s);onSample?.(s);finalZonal=z;
      if(s.invalid){failures.push(`day ${day}: invalid state`);break;}
      if(Math.abs(s.massDrift)>5e-5){failures.push(`day ${day}: mass drift ${s.massDrift}`);break;}
      if(!(s.maxW<10)){failures.push(`day ${day}: vertical-velocity stability guard exceeded: ${s.maxW} at z=${(s.maxWAltitude/1000).toFixed(2)} km, lat=${s.maxWLatitude.toFixed(1)} deg; below=${s.maxWBelowSponge}, absorber=${s.maxWInSponge}, hCFL=${s.maxHorizontalCfl}, vCFL=${s.maxVerticalCfl}, divRMS=${s.divergenceRms}`);break;}
      if(runOptions.onCheckpoint){
        await runOptions.onCheckpoint({
          schemaVersion:STAGE4_RK3_CHECKPOINT_SCHEMA_VERSION,
          modelSignature:STAGE4_RK3_PRODUCTION_MODEL_SIGNATURE,
          savedAt:Date.now(),
          targetDays:days,
          completedOuterSteps,
          initialDryMass:m0,
          state,
          samples:samples.map(sample=>({...sample})),
        });
      }
    }
    const f=samples[samples.length-1];
    if(!f)failures.push('no RK3 climate samples');
    else if(f.day<days)failures.push(`RK3 climate validation stopped at day ${f.day} before ${days}`);
    else if(days>=30&&!f.invalid){
      if(!(f.jet>.5))failures.push(`midlatitude upper westerly too weak: ${f.jet}`);
      if(!(f.trade<0))failures.push(`tropical low-level easterly absent: ${f.trade}`);
      if(!(f.psi>1e9))failures.push(`overturning too weak: ${f.psi}`);
      if(!(f.nhPsi*f.shPsi<0))failures.push(`hemisphere overturning signs not opposite: ${f.nhPsi},${f.shPsi}`);
    }
    return{passed:failures.length===0,samples,failures,elapsedMs:performance.now()-t0,finalZonal};
  }finally{gpu.destroy();}
}

export async function runStage4Rk3TwoDayClimate(onSample?:(s:ClimateDaySample)=>void):Promise<Stage4Rk3ClimateResult>{return runStage4Rk3Climate(2,onSample);}
