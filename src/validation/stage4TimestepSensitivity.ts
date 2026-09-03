import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid, VerticalGrid } from '../grid/vertical.js';
import { acousticDivergenceRms } from '../physics/acousticDivergenceDamping.js';
import { addHeldSuarezWavePerturbation, buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { MODEL_TOP_SPONGE } from '../physics/modelTopSponge.js';
import { diagnoseState } from '../solver/diagnostics.js';
import { createHydrostaticState, DryState } from '../solver/state.js';
import { GpuStage4Integrator } from '../gpu/stage4IntegratorGpu.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';

const N=8;
const NZ=48;
const TOP=40000;

export interface Stage4TimestepSensitivitySample{
  dt:number;
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
  maxVerticalCfl:number;
  invalid:boolean;
}

export interface Stage4TimestepSensitivityRun{
  dt:number;
  samples:Stage4TimestepSensitivitySample[];
  elapsedMs:number;
}

export interface Stage4TimestepSensitivityResult{
  runs:Stage4TimestepSensitivityRun[];
  elapsedMs:number;
}

function extra(h:CubedSphereGrid,v:VerticalGrid,s:DryState,dt:number){
  const absorberStart=MODEL_TOP_SPONGE.startFraction*v.top;
  let maxW=0,maxWBelowAbsorber=0,maxWInAbsorber=0,maxWAltitude=0,maxWLatitude=0,maxEdgeWind=0,maxHorizontalCfl=0,maxVerticalCfl=0;
  const stride=v.nz+1;
  for(let c=0;c<h.cellCount;c++){
    const lat=Math.asin(Math.max(-1,Math.min(1,h.cellCenters[c*3+2]!)))*180/Math.PI;
    for(let i=1;i<v.nz;i++){
      const z=v.zInterface[i]!,aw=Math.abs(s.wInterface[c*stride+i]!);
      if(z>=absorberStart)maxWInAbsorber=Math.max(maxWInAbsorber,aw);else maxWBelowAbsorber=Math.max(maxWBelowAbsorber,aw);
      maxVerticalCfl=Math.max(maxVerticalCfl,aw*dt/Math.min(v.dz[i-1]!,v.dz[i]!));
      if(aw>maxW){maxW=aw;maxWAltitude=z;maxWLatitude=lat}
    }
  }
  for(let e=0;e<h.edgeCount;e++){
    const dist=Math.max(h.edges[e]!.centerDistanceAngle*EARTH.radius,1);
    for(let k=0;k<v.nz;k++){
      const au=Math.abs(s.uEdge[e*v.nz+k]!);
      maxEdgeWind=Math.max(maxEdgeWind,au);
      maxHorizontalCfl=Math.max(maxHorizontalCfl,au*dt/dist);
    }
  }
  return {maxW,maxWBelowAbsorber,maxWInAbsorber,maxWAltitude,maxWLatitude,maxEdgeWind,divergenceRms:acousticDivergenceRms(h,v,buildHeldSuarezReference(v),s),maxHorizontalCfl,maxVerticalCfl};
}

async function runOne(dt:number,onSample?:(s:Stage4TimestepSensitivitySample)=>void):Promise<Stage4TimestepSensitivityRun>{
  const h=buildCubedSphere(N),v=buildStretchedVerticalGrid(NZ,TOP,1.4),ref=buildHeldSuarezReference(v),seed=createHydrostaticState(h,v,ref);
  addHeldSuarezWavePerturbation(h,v,ref,seed,.05);
  const gpu=await GpuStage4Integrator.create(h,v,ref,seed),initial=await gpu.downloadState(0),m0=diagnoseState(h,v,initial).dryMass;
  const stepsPerQuarter=Math.round(21600/dt),samples:Stage4TimestepSensitivitySample[]=[],t0=performance.now();
  try{
    for(let segment=1;segment<=8;segment++){
      let left=stepsPerQuarter;
      while(left>0){
        const n=Math.min(200,left);gpu.stepBatch(dt,n,true);left-=n;
        if(left%1000===0){await gpu.device.queue.onSubmittedWorkDone();await new Promise<void>(r=>setTimeout(r,0))}
      }
      await gpu.device.queue.onSubmittedWorkDone();
      const day=segment/4,state=await gpu.downloadState(day*86400),d=diagnoseState(h,v,state),x=extra(h,v,state,dt);
      const sample:Stage4TimestepSensitivitySample={dt,day,massDrift:(d.dryMass-m0)/m0,invalid:d.nan||d.minRho<=0||d.minP<=0,...x};
      samples.push(sample);onSample?.(sample);
      if(sample.invalid||sample.maxW>50)break;
    }
    return {dt,samples,elapsedMs:performance.now()-t0};
  }finally{gpu.destroy()}
}

export async function runStage4TimestepSensitivity(onSample?:(s:Stage4TimestepSensitivitySample)=>void):Promise<Stage4TimestepSensitivityResult>{
  const t0=performance.now(),runs:Stage4TimestepSensitivityRun[]=[];
  for(const dt of [10,5,2.5])runs.push(await runOne(dt,onSample));
  return {runs,elapsedMs:performance.now()-t0};
}
