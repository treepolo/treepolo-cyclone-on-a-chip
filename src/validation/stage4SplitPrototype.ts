import { EARTH } from '../core/constants.js';
import { buildCubedSphere, CubedSphereGrid } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid, VerticalGrid } from '../grid/vertical.js';
import { acousticDivergenceRms } from '../physics/acousticDivergenceDamping.js';
import { addHeldSuarezWavePerturbation, buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { MODEL_TOP_SPONGE } from '../physics/modelTopSponge.js';
import { diagnoseState } from '../solver/diagnostics.js';
import { createHydrostaticState, DryState } from '../solver/state.js';
import { GpuStage4SplitPrototype } from '../gpu/stage4SplitPrototypeGpu.js';

const N=8,NZ=48,TOP=40000,OUTER_DT=10;

export interface Stage4SplitSample{
  acousticSubsteps:number;
  acousticDt:number;
  day:number;
  massDrift:number;
  maxW:number;
  maxWBelowAbsorber:number;
  maxWInAbsorber:number;
  maxWAltitude:number;
  maxWLatitude:number;
  maxEdgeWind:number;
  divergenceRms:number;
  maxHorizontalCfl:number;
  maxVerticalCflOuter:number;
  maxVerticalCflAcoustic:number;
  invalid:boolean;
}
export interface Stage4SplitRun{acousticSubsteps:number;acousticDt:number;samples:Stage4SplitSample[];elapsedMs:number}
export interface Stage4SplitResult{runs:Stage4SplitRun[];elapsedMs:number}

function extra(h:CubedSphereGrid,v:VerticalGrid,s:DryState,acousticDt:number){
  const absorberStart=MODEL_TOP_SPONGE.startFraction*v.top,stride=v.nz+1;
  let maxW=0,maxWBelowAbsorber=0,maxWInAbsorber=0,maxWAltitude=0,maxWLatitude=0,maxEdgeWind=0,maxHorizontalCfl=0,maxVerticalCflOuter=0,maxVerticalCflAcoustic=0;
  for(let c=0;c<h.cellCount;c++){
    const lat=Math.asin(Math.max(-1,Math.min(1,h.cellCenters[c*3+2]!)))*180/Math.PI;
    for(let i=1;i<v.nz;i++){
      const z=v.zInterface[i]!,aw=Math.abs(s.wInterface[c*stride+i]!),dz=Math.min(v.dz[i-1]!,v.dz[i]!);
      if(z>=absorberStart)maxWInAbsorber=Math.max(maxWInAbsorber,aw);else maxWBelowAbsorber=Math.max(maxWBelowAbsorber,aw);
      maxVerticalCflOuter=Math.max(maxVerticalCflOuter,aw*OUTER_DT/dz);maxVerticalCflAcoustic=Math.max(maxVerticalCflAcoustic,aw*acousticDt/dz);
      if(aw>maxW){maxW=aw;maxWAltitude=z;maxWLatitude=lat}
    }
  }
  for(let e=0;e<h.edgeCount;e++){
    const dist=Math.max(h.edges[e]!.centerDistanceAngle*EARTH.radius,1);
    for(let k=0;k<v.nz;k++){const au=Math.abs(s.uEdge[e*v.nz+k]!);maxEdgeWind=Math.max(maxEdgeWind,au);maxHorizontalCfl=Math.max(maxHorizontalCfl,au*OUTER_DT/dist)}
  }
  return {maxW,maxWBelowAbsorber,maxWInAbsorber,maxWAltitude,maxWLatitude,maxEdgeWind,divergenceRms:acousticDivergenceRms(h,v,buildHeldSuarezReference(v),s),maxHorizontalCfl,maxVerticalCflOuter,maxVerticalCflAcoustic};
}

async function runOne(acousticSubsteps:number,onSample?:(s:Stage4SplitSample)=>void):Promise<Stage4SplitRun>{
  const acousticDt=OUTER_DT/acousticSubsteps,h=buildCubedSphere(N),v=buildStretchedVerticalGrid(NZ,TOP,1.4),ref=buildHeldSuarezReference(v),seed=createHydrostaticState(h,v,ref);
  addHeldSuarezWavePerturbation(h,v,ref,seed,.05);
  const gpu=await GpuStage4SplitPrototype.create(h,v,ref,seed,acousticSubsteps),initial=await gpu.downloadState(0),m0=diagnoseState(h,v,initial).dryMass;
  const stepsPerQuarter=Math.round(21600/OUTER_DT),samples:Stage4SplitSample[]=[],t0=performance.now();
  try{
    for(let segment=1;segment<=8;segment++){
      let left=stepsPerQuarter;
      while(left>0){const n=Math.min(100,left);gpu.stepBatch(OUTER_DT,n,true);left-=n;if(left%500===0){await gpu.device.queue.onSubmittedWorkDone();await new Promise<void>(r=>setTimeout(r,0))}}
      await gpu.device.queue.onSubmittedWorkDone();
      const day=segment/4,state=await gpu.downloadState(day*86400),d=diagnoseState(h,v,state),x=extra(h,v,state,acousticDt),sample:Stage4SplitSample={acousticSubsteps,acousticDt,day,massDrift:(d.dryMass-m0)/m0,invalid:d.nan||d.minRho<=0||d.minP<=0,...x};
      samples.push(sample);onSample?.(sample);if(sample.invalid||sample.maxW>50)break;
    }
    return {acousticSubsteps,acousticDt,samples,elapsedMs:performance.now()-t0};
  }finally{gpu.destroy()}
}

export async function runStage4SplitPrototype(onSample?:(s:Stage4SplitSample)=>void):Promise<Stage4SplitResult>{
  const t0=performance.now(),runs:Stage4SplitRun[]=[];
  for(const substeps of [2,4])runs.push(await runOne(substeps,onSample));
  return {runs,elapsedMs:performance.now()-t0};
}
