import { buildCubedSphere } from './grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from './grid/vertical.js';
import { GpuStage4Rk3SplitReference } from './gpu/stage4Rk3SplitGpu.js';
import type { Stage4Rk3ClimateCheckpoint } from './persistence/stage4Rk3Checkpoint.js';
import { buildHeldSuarezReference } from './physics/heldSuarez.js';
import { buildRotationGeometry } from './physics/rotation.js';
import {
  diagnoseAxialAngularMomentum,
  diagnoseEddies,
  type AxialAngularMomentumDiagnostics,
  type EddyDiagnostics,
} from './solver/stage4CirculationDiagnostics.js';
import { diagnoseStage4InstantAamBreakdown } from './solver/stage4MomentumBudgetDiagnostics.js';
import type { DryState } from './solver/state.js';
import { runStage4Rk3Climate, type Stage4Rk3ClimateProgress } from './validation/stage4Rk3Climate.js';
import type { ClimateDaySample } from './validation/stage4Gpu.js';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const runBtn=$<HTMLButtonElement>('run');
const logEl=$<HTMLPreElement>('log');
const DT=10;
const STEPS_PER_QUARTER=Math.round(21600/DT);
const BATCH=40;

interface BudgetPoint{day:number;aam:AxialAngularMomentumDiagnostics;eddy:EddyDiagnostics;}

function linearSlope(samples:ClimateDaySample[]):number{
  if(samples.length<2)return NaN;
  let sx=0,sy=0,sxx=0,sxy=0;
  for(const s of samples){sx+=s.day;sy+=s.trade;sxx+=s.day*s.day;sxy+=s.day*s.trade;}
  const n=samples.length,den=n*sxx-sx*sx;
  return Math.abs(den)>0?(n*sxy-sx*sy)/den:NaN;
}

async function yieldToBrowser():Promise<void>{
  const scheduler=(globalThis as any).scheduler;
  if(typeof scheduler?.yield==='function')await scheduler.yield();
  else await new Promise<void>(resolve=>setTimeout(resolve,0));
}

function point(
  h:ReturnType<typeof buildCubedSphere>,
  v:ReturnType<typeof buildStretchedVerticalGrid>,
  state:DryState,
  day:number,
  rotation:ReturnType<typeof buildRotationGeometry>,
):BudgetPoint{
  return{
    day,
    aam:diagnoseAxialAngularMomentum(h,v,state,rotation),
    eddy:diagnoseEddies(h,v,state,24,rotation),
  };
}

async function oneDayIntegratedBudget(
  h:ReturnType<typeof buildCubedSphere>,
  v:ReturnType<typeof buildStretchedVerticalGrid>,
  ref:ReturnType<typeof buildHeldSuarezReference>,
  rotation:ReturnType<typeof buildRotationGeometry>,
  state:DryState,
  startDay:number,
){
  const points:BudgetPoint[]=[point(h,v,state,startDay,rotation)];
  const gpu=await GpuStage4Rk3SplitReference.create(h,v,ref,state,4);
  const opts={heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true} as const;
  try{
    for(let quarter=1;quarter<=4;quarter++){
      let left=STEPS_PER_QUARTER;
      while(left>0){
        const n=Math.min(BATCH,left);
        gpu.stepBatch(DT,n,opts);
        await gpu.device.queue.onSubmittedWorkDone();
        left-=n;
        await yieldToBrowser();
      }
      const elapsed=quarter*21600;
      const next=await gpu.downloadState(state.time+elapsed);
      points.push(point(h,v,next,startDay+quarter*.25,rotation));
      $('status').textContent=`30-day complete; AAM budget day ${(startDay+quarter*.25).toFixed(2)}`;
    }
  }finally{gpu.destroy();}

  const first=points[0]!,last=points[points.length-1]!;
  const totalSeconds=(last.day-first.day)*86400;
  const observed=(last.aam.absolute-first.aam.absolute)/totalSeconds;
  let dragIntegral=0;
  for(let i=1;i<points.length;i++){
    dragIntegral+=.5*(points[i-1]!.aam.dragTorque+points[i]!.aam.dragTorque)*(points[i]!.day-points[i-1]!.day)*86400;
  }
  const meanDrag=dragIntegral/totalSeconds;
  const residual=observed-meanDrag;
  const lever=.5*(first.aam.torqueLeverMass+last.aam.torqueLeverMass);
  return{
    points:points.map(p=>({
      day:p.day,
      absoluteAam:p.aam.absolute,
      relativeAam:p.aam.relative,
      dragTorque:p.aam.dragTorque,
      midlatitudeEke:p.eddy.midlatitudeEke,
      polewardHeatFlux:p.eddy.midlatitudePolewardHeatFlux,
      polewardMomentumFlux:p.eddy.midlatitudePolewardMomentumFlux,
    })),
    observedTorque:observed,
    meanDragTorque:meanDrag,
    numericalResidualTorque:residual,
    equivalentMsPerDay:{
      observed:observed/lever*86400,
      drag:meanDrag/lever*86400,
      residual:residual/lever*86400,
    },
  };
}

runBtn.onclick=()=>void(async()=>{
  runBtn.disabled=true;
  logEl.textContent='';
  $('status').textContent='Initializing fresh day-0 state';
  const started=performance.now();
  let finalCheckpoint:Stage4Rk3ClimateCheckpoint|null=null;
  try{
    const samples:ClimateDaySample[]=[];
    const progress=(p:Stage4Rk3ClimateProgress)=>{
      $('status').textContent=`Fresh Held–Suarez day ${p.simulatedDay.toFixed(3)} / 30`;
      $('day').textContent=p.simulatedDay.toFixed(3);
      $('elapsed').textContent=`${(p.elapsedMs/1000).toFixed(1)} s`;
    };
    const climate=await runStage4Rk3Climate(
      30,
      s=>{
        samples.push({...s});
        $('day').textContent=s.day.toFixed(2);
        $('trade').textContent=`${s.trade.toFixed(4)} m/s`;
        $('jet').textContent=`${s.jet.toFixed(4)} m/s`;
        $('psi').textContent=s.psi.toExponential(4);
        $('mass').textContent=s.massDrift.toExponential(4);
      },
      progress,
      {onCheckpoint:cp=>{finalCheckpoint=cp;}},
    );
    if(!finalCheckpoint)throw new Error('fresh climate run produced no final checkpoint');

    const cp=finalCheckpoint as Stage4Rk3ClimateCheckpoint;
    const h=buildCubedSphere(8);
    const v=buildStretchedVerticalGrid(48,40000,1.4);
    const ref=buildHeldSuarezReference(v);
    const rotation=buildRotationGeometry(h);
    const finalState=cp.state;
    const aam=diagnoseAxialAngularMomentum(h,v,finalState,rotation);
    const eddy=diagnoseEddies(h,v,finalState,24,rotation);
    const instant=diagnoseStage4InstantAamBreakdown(h,v,ref,finalState,rotation);
    const recent=samples.filter(s=>s.day>=25);
    const tradeSlope=linearSlope(recent);

    $('status').textContent='30-day complete; running independent day-30 → 31 AAM budget';
    const integrated=await oneDayIntegratedBudget(h,v,ref,rotation,finalState,30);

    const z=climate.finalZonal;
    const result={
      model:'Stage4 production RK3 split-explicit; fresh day-0 initial condition',
      elapsedMs:performance.now()-started,
      climate:{
        passed:climate.passed,
        failures:climate.failures,
        samples,
        final:samples[samples.length-1]??null,
        tradeSlopeDay25To30:tradeSlope,
      },
      finalZonal:z?{
        bins:z.bins,
        latDeg:Array.from(z.latDeg),
        temperature:Array.from(z.temperature),
        zonalWind:Array.from(z.zonalWind),
        meridionalWind:Array.from(z.meridionalWind),
        streamfunction:Array.from(z.streamfunction),
        maxUpperMidlatitudeWesterly:z.maxUpperMidlatitudeWesterly,
        meanTropicalLowLevelZonal:z.meanTropicalLowLevelZonal,
        maxAbsStreamfunction:z.maxAbsStreamfunction,
        nhDominantStreamfunction:z.nhDominantStreamfunction,
        shDominantStreamfunction:z.shDominantStreamfunction,
      }:null,
      finalAam:{
        absolute:aam.absolute,
        relative:aam.relative,
        dragTorque:aam.dragTorque,
        torqueLeverMass:aam.torqueLeverMass,
      },
      finalEddy:{
        midlatitudeEke:eddy.midlatitudeEke,
        midlatitudePolewardHeatFlux:eddy.midlatitudePolewardHeatFlux,
        midlatitudePolewardMomentumFlux:eddy.midlatitudePolewardMomentumFlux,
      },
      instantaneousAamBreakdown:instant,
      integratedDay30To31:integrated,
    };

    logEl.textContent=JSON.stringify(result,null,2);
    $('elapsed').textContent=`${((performance.now()-started)/1000).toFixed(1)} s`;
    $('status').textContent='COMPLETE';
  }catch(e){
    $('status').textContent='ERROR';
    logEl.textContent=String(e instanceof Error?e.stack||e.message:e);
  }finally{
    runBtn.disabled=false;
  }
})();
