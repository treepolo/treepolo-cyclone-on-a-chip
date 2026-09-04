import { EARTH } from '../core/constants.js';
import { buildCubedSphere, CubedSphereGrid } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid, VerticalGrid } from '../grid/vertical.js';
import { acousticDivergenceRms } from '../physics/acousticDivergenceDamping.js';
import { addHeldSuarezWavePerturbation, buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { buildModelTopSpongeRates, MODEL_TOP_SPONGE } from '../physics/modelTopSponge.js';
import { diagnoseState } from '../solver/diagnostics.js';
import { diagnoseZonalMeans } from '../solver/stage4Diagnostics.js';
import { createHydrostaticState, DryState } from '../solver/state.js';
import { GpuStage4Rk3SplitReference } from '../gpu/stage4Rk3SplitGpu.js';
import { ClimateDaySample } from './stage4Gpu.js';

const HORIZONTAL_N=8;
const VERTICAL_NZ=48;
const TOP_METERS=40000;
const ZONAL_BINS=24;
const MIN_ACTIVE_SPONGE_INTERFACES=6;
const DT=10;
const BATCH=40;

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

export interface Stage4Rk3TwoDayResult{passed:boolean;samples:ClimateDaySample[];failures:string[];elapsedMs:number;}
export async function runStage4Rk3TwoDayClimate(onSample?:(s:ClimateDaySample)=>void):Promise<Stage4Rk3TwoDayResult>{
  const h=buildCubedSphere(HORIZONTAL_N),v=buildStretchedVerticalGrid(VERTICAL_NZ,TOP_METERS,1.4),rates=buildModelTopSpongeRates(v),active=Array.from(rates.slice(1,-1)).filter(x=>x>0).length;
  if(active<MIN_ACTIVE_SPONGE_INTERFACES)throw new Error(`RK3 two-day sponge under-resolved: ${active} active interfaces; require >=${MIN_ACTIVE_SPONGE_INTERFACES}`);
  const ref=buildHeldSuarezReference(v),seed=createHydrostaticState(h,v,ref);addHeldSuarezWavePerturbation(h,v,ref,seed,.05);
  const gpu=await GpuStage4Rk3SplitReference.create(h,v,ref,seed,4),initial=await gpu.downloadState(0),m0=diagnoseState(h,v,initial).dryMass,stepsPerQuarter=Math.round(21600/DT),samples:ClimateDaySample[]=[],failures:string[]=[],t0=performance.now(),opts={heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true} as const;
  try{
    for(let segment=1;segment<=8;segment++){
      let left=stepsPerQuarter,batches=0;
      while(left>0){const n=Math.min(BATCH,left);gpu.stepBatch(DT,n,opts);left-=n;batches++;if(batches%8===0)await new Promise<void>(r=>setTimeout(r,0));}
      await gpu.device.queue.onSubmittedWorkDone();
      const day=segment/4,state=await gpu.downloadState(day*86400),d=diagnoseState(h,v,state),z=diagnoseZonalMeans(h,v,state,ZONAL_BINS),x=extra(h,v,state,DT),s:ClimateDaySample={day,massDrift:(d.dryMass-m0)/m0,maxW:d.maxAbsW,jet:z.maxUpperMidlatitudeWesterly,trade:z.meanTropicalLowLevelZonal,psi:z.maxAbsStreamfunction,nhPsi:z.nhDominantStreamfunction,shPsi:z.shDominantStreamfunction,invalid:d.nan||d.minRho<=0||d.minP<=0,...x};
      samples.push(s);onSample?.(s);
      if(s.invalid){failures.push(`day ${day}: invalid state`);break;}
      if(Math.abs(s.massDrift)>5e-5){failures.push(`day ${day}: mass drift ${s.massDrift}`);break;}
      if(!(s.maxW<10)){failures.push(`day ${day}: vertical-velocity stability guard exceeded: ${s.maxW} at z=${(s.maxWAltitude/1000).toFixed(2)} km, lat=${s.maxWLatitude.toFixed(1)} deg; below=${s.maxWBelowSponge}, absorber=${s.maxWInSponge}, hCFL=${s.maxHorizontalCfl}, vCFL=${s.maxVerticalCfl}, divRMS=${s.divergenceRms}`);break;}
    }
    const f=samples[samples.length-1];if(!f)failures.push('no RK3 two-day samples');else if(f.day<2)failures.push(`RK3 two-day validation stopped at day ${f.day}`);
    return{passed:failures.length===0,samples,failures,elapsedMs:performance.now()-t0};
  }finally{gpu.destroy();}
}
