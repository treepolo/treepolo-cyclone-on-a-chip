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
import { ClimateDaySample } from './stage4Gpu.js';

const HORIZONTAL_N=8;
const VERTICAL_NZ=48;
const TOP_METERS=40000;
const ZONAL_BINS=24;
const MIN_ACTIVE_SPONGE_INTERFACES=6;
const DT=10;
// 40 outer steps is the largest batch that has already passed real-device
// CPU/GPU agreement.  Do not silently increase this for the climate gate:
// much larger command buffers can spend minutes synchronously encoding on the
// browser main thread before the GPU receives any work.
const PRODUCTION_BATCH=40;

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
export interface Stage4Rk3ClimateResult{passed:boolean;samples:ClimateDaySample[];failures:string[];elapsedMs:number;finalZonal?:ZonalMeanDiagnostics;}
export async function runStage4Rk3Climate(days=30,onSample?:(s:ClimateDaySample)=>void,onProgress?:(p:Stage4Rk3ClimateProgress)=>void):Promise<Stage4Rk3ClimateResult>{
  if(!Number.isInteger(days)||days<1)throw new Error('RK3 climate days must be a positive integer');
  const h=buildCubedSphere(HORIZONTAL_N),v=buildStretchedVerticalGrid(VERTICAL_NZ,TOP_METERS,1.4),rates=buildModelTopSpongeRates(v),active=Array.from(rates.slice(1,-1)).filter(x=>x>0).length;
  if(active<MIN_ACTIVE_SPONGE_INTERFACES)throw new Error(`RK3 climate sponge under-resolved: ${active} active interfaces; require >=${MIN_ACTIVE_SPONGE_INTERFACES}`);
  const ref=buildHeldSuarezReference(v),seed=createHydrostaticState(h,v,ref);addHeldSuarezWavePerturbation(h,v,ref,seed,.05);
  const gpu=await GpuStage4Rk3SplitReference.create(h,v,ref,seed,4),initial=await gpu.downloadState(0),m0=diagnoseState(h,v,initial).dryMass,stepsPerQuarter=Math.round(21600/DT),stepsPerDay=Math.round(86400/DT),totalOuterSteps=days*stepsPerDay,samples:ClimateDaySample[]=[],failures:string[]=[],t0=performance.now(),opts={heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true} as const;
  let finalZonal:ZonalMeanDiagnostics|undefined,completedOuterSteps=0;
  try{
    for(let segment=1;segment<=days*4;segment++){
      let left=stepsPerQuarter;
      while(left>0){
        const n=Math.min(PRODUCTION_BATCH,left);
        gpu.stepBatch(DT,n,opts);
        // Waiting for submitted work already yields to the browser event loop.
        // Do not add a timer-based yield here: background-tab timer throttling
        // can delay setTimeout(0) for a minute or more and leave the GPU idle
        // between otherwise healthy batches.
        await gpu.device.queue.onSubmittedWorkDone();
        left-=n;completedOuterSteps+=n;
        onProgress?.({simulatedDay:completedOuterSteps*DT/86400,targetDays:days,completedOuterSteps,totalOuterSteps,batchSize:n,elapsedMs:performance.now()-t0});
      }
      const day=segment/4,state=await gpu.downloadState(day*86400),d=diagnoseState(h,v,state),z=diagnoseZonalMeans(h,v,state,ZONAL_BINS),x=extra(h,v,state,DT),s:ClimateDaySample={day,massDrift:(d.dryMass-m0)/m0,maxW:d.maxAbsW,jet:z.maxUpperMidlatitudeWesterly,trade:z.meanTropicalLowLevelZonal,psi:z.maxAbsStreamfunction,nhPsi:z.nhDominantStreamfunction,shPsi:z.shDominantStreamfunction,invalid:d.nan||d.minRho<=0||d.minP<=0,...x};
      samples.push(s);onSample?.(s);finalZonal=z;
      if(s.invalid){failures.push(`day ${day}: invalid state`);break;}
      if(Math.abs(s.massDrift)>5e-5){failures.push(`day ${day}: mass drift ${s.massDrift}`);break;}
      if(!(s.maxW<10)){failures.push(`day ${day}: vertical-velocity stability guard exceeded: ${s.maxW} at z=${(s.maxWAltitude/1000).toFixed(2)} km, lat=${s.maxWLatitude.toFixed(1)} deg; below=${s.maxWBelowSponge}, absorber=${s.maxWInSponge}, hCFL=${s.maxHorizontalCfl}, vCFL=${s.maxVerticalCfl}, divRMS=${s.divergenceRms}`);break;}
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
