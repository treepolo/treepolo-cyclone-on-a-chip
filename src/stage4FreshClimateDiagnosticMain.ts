import { EARTH } from './core/constants.js';
import { buildCubedSphere } from './grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from './grid/vertical.js';
import { GpuStage4Rk3SplitReference } from './gpu/stage4Rk3SplitGpu.js';
import {
  assertStage4Rk3CheckpointCompatible,
  clearStage4Rk3Checkpoint,
  loadStage4Rk3Checkpoint,
  saveStage4Rk3Checkpoint,
  STAGE4_RK3_PRODUCTION_MODEL_SIGNATURE,
  type Stage4Rk3ClimateCheckpoint,
} from './persistence/stage4Rk3Checkpoint.js';
import { acousticDivergenceCoefficientForDt, applyAcousticDivergenceDamping, computeAcousticDivergence } from './physics/acousticDivergenceDamping.js';
import { buildHeldSuarezReference } from './physics/heldSuarez.js';
import { reconstructEdgeNormalScalarGradient } from './physics/horizontalGradient.js';
import { buildRotationGeometry } from './physics/rotation.js';
import {
  diagnoseAxialAngularMomentum,
  diagnoseEddies,
  type AxialAngularMomentumDiagnostics,
  type EddyDiagnostics,
} from './solver/stage4CirculationDiagnostics.js';
import { diagnoseStage4InstantAamBreakdown } from './solver/stage4MomentumBudgetDiagnostics.js';
import { cloneState, edge3DIndex, type DryState } from './solver/state.js';
import { runStage4Rk3Climate, type Stage4Rk3ClimateProgress } from './validation/stage4Rk3Climate.js';
import type { ClimateDaySample } from './validation/stage4Gpu.js';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const runBtn=$<HTMLButtonElement>('run');
const logEl=$<HTMLPreElement>('log');
const DT=10;
const STEPS_PER_QUARTER=Math.round(21600/DT);
const BATCH=40;
const TARGET_DAYS=30;

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

async function loadCompatibleResume():Promise<Stage4Rk3ClimateCheckpoint|null>{
  try{
    const cp=await loadStage4Rk3Checkpoint();
    if(!cp)return null;
    try{
      assertStage4Rk3CheckpointCompatible(cp,STAGE4_RK3_PRODUCTION_MODEL_SIGNATURE,TARGET_DAYS);
      return cp;
    }catch{
      await clearStage4Rk3Checkpoint();
      return null;
    }
  }catch{
    // A damaged/obsolete browser store must never change the equations or make
    // the diagnostic silently accept a bad state.  Discard it and start fresh.
    try{await clearStage4Rk3Checkpoint();}catch{}
    return null;
  }
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

function applyNonorthDivergenceCandidate(
  h:ReturnType<typeof buildCubedSphere>,
  v:ReturnType<typeof buildStretchedVerticalGrid>,
  ref:ReturnType<typeof buildHeldSuarezReference>,
  s:DryState,
  coefficient:number,
):void{
  const div=computeAcousticDivergence(h,v,ref,s),g=buildRotationGeometry(h),R=EARTH.radius;
  for(let e=0;e<h.edgeCount;e++){
    const L=h.edges[e]!.centerDistanceAngle*R,L2=L*L;
    for(let k=0;k<v.nz;k++){
      const q=edge3DIndex(e,k,v.nz),gradN=reconstructEdgeNormalScalarGradient(h,g,e,c=>div[c*v.nz+k]!);
      s.uEdge[q]=s.uEdge[q]!+coefficient*L2*gradN;
    }
  }
}

async function oneStepOperatorAttribution(
  h:ReturnType<typeof buildCubedSphere>,
  v:ReturnType<typeof buildStretchedVerticalGrid>,
  ref:ReturnType<typeof buildHeldSuarezReference>,
  rotation:ReturnType<typeof buildRotationGeometry>,
  state:DryState,
){
  const before=diagnoseAxialAngularMomentum(h,v,state,rotation),lever=before.torqueLeverMass;
  const variants=[
    ['production',{heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true}],
    ['noDivergence',{heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:false,topAbsorber:true}],
    ['noTopAbsorber',{heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:false}],
    ['noDivergenceNoTop',{heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:false,topAbsorber:false}],
  ] as const;
  const slopes:Record<string,number>={};
  for(const [name,opts] of variants){
    const gpu=await GpuStage4Rk3SplitReference.create(h,v,ref,state,4);
    try{
      gpu.step(DT,opts);
      await gpu.device.queue.onSubmittedWorkDone();
      const next=await gpu.downloadState(state.time+DT),a=diagnoseAxialAngularMomentum(h,v,next,rotation);
      slopes[name]=(a.absolute-before.absolute)/DT/lever*86400;
    }finally{gpu.destroy();}
  }

  const instant=diagnoseStage4InstantAamBreakdown(h,v,ref,state,rotation);
  const directDt=DT/4,coef=acousticDivergenceCoefficientForDt(directDt);
  const legacy=cloneState(state),nonorth=cloneState(state);
  applyAcousticDivergenceDamping(h,v,ref,legacy,coef);
  applyNonorthDivergenceCandidate(h,v,ref,nonorth,coef);
  const legacyA=diagnoseAxialAngularMomentum(h,v,legacy,rotation),nonorthA=diagnoseAxialAngularMomentum(h,v,nonorth,rotation);
  const directScale=86400/(directDt*lever);
  const frozenMsDay=instant.full/lever*86400;
  return{
    oneOuterStepEquivalentMsPerDay:slopes,
    differencesEquivalentMsPerDay:{
      divergenceContribution:slopes.production!-slopes.noDivergence!,
      topAbsorberContribution:slopes.production!-slopes.noTopAbsorber!,
      divergenceContributionWithoutTop:slopes.noTopAbsorber!-slopes.noDivergenceNoTop!,
      topContributionWithoutDivergence:slopes.noDivergence!-slopes.noDivergenceNoTop!,
      productionMinusFrozenRhs:slopes.production!-frozenMsDay,
      noDivergenceMinusFrozenRhs:slopes.noDivergence!-frozenMsDay,
    },
    frozenInstantEquivalentMsPerDay:frozenMsDay,
    directSingleAcousticFilterEquivalentMsPerDay:{
      dt:directDt,
      coefficient:coef,
      legacy:(legacyA.absolute-before.absolute)*directScale,
      nonorthCandidate:(nonorthA.absolute-before.absolute)*directScale,
    },
  };
}

runBtn.onclick=()=>void(async()=>{
  runBtn.disabled=true;
  logEl.textContent='';
  $('status').textContent='Checking resumable climate checkpoint';
  const started=performance.now();
  let finalCheckpoint:Stage4Rk3ClimateCheckpoint|null=null;
  try{
    const resume=await loadCompatibleResume();
    finalCheckpoint=resume;
    const resumeDay=resume?resume.state.time/86400:0;
    const samples:ClimateDaySample[]=resume?resume.samples.map(s=>({...s})):[];
    $('status').textContent=resume?`Resuming Held–Suarez from day ${resumeDay.toFixed(2)}`:'Initializing fresh day-0 state';
    if(resume){
      const last=samples[samples.length-1];
      $('day').textContent=resumeDay.toFixed(2);
      if(last){
        $('trade').textContent=`${last.trade.toFixed(4)} m/s`;
        $('jet').textContent=`${last.jet.toFixed(4)} m/s`;
        $('psi').textContent=last.psi.toExponential(4);
        $('mass').textContent=last.massDrift.toExponential(4);
      }
    }
    const progress=(p:Stage4Rk3ClimateProgress)=>{
      $('status').textContent=`Held–Suarez day ${p.simulatedDay.toFixed(3)} / 30${resume?' (resumed)':''}`;
      $('day').textContent=p.simulatedDay.toFixed(3);
      $('elapsed').textContent=`${(p.elapsedMs/1000).toFixed(1)} s`;
    };
    const climate=await runStage4Rk3Climate(
      TARGET_DAYS,
      s=>{
        samples.push({...s});
        $('day').textContent=s.day.toFixed(2);
        $('trade').textContent=`${s.trade.toFixed(4)} m/s`;
        $('jet').textContent=`${s.jet.toFixed(4)} m/s`;
        $('psi').textContent=s.psi.toExponential(4);
        $('mass').textContent=s.massDrift.toExponential(4);
      },
      progress,
      {
        resume,
        onCheckpoint:async cp=>{
          finalCheckpoint=cp;
          await saveStage4Rk3Checkpoint(cp);
        },
      },
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

    $('status').textContent='30-day complete; attributing one-step AAM operators';
    const attribution=await oneStepOperatorAttribution(h,v,ref,rotation,finalState);

    $('status').textContent='30-day complete; running independent day-30 → 31 AAM budget';
    const integrated=await oneDayIntegratedBudget(h,v,ref,rotation,finalState,30);

    const z=climate.finalZonal;
    const result={
      model:'Stage4 production RK3 split-explicit; fresh/resumable day-0 initial condition',
      resumedFromDay:resumeDay,
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
      oneStepOperatorAttribution:attribution,
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
