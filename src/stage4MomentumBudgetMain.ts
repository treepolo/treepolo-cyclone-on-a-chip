import { buildCubedSphere } from './grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from './grid/vertical.js';
import { GpuStage4Rk3SplitReference } from './gpu/stage4Rk3SplitGpu.js';
import {
  assertStage4Rk3CheckpointShape,
  loadStage4Rk3Checkpoint,
  STAGE4_RK3_PRODUCTION_MODEL_SIGNATURE,
  type Stage4Rk3ClimateCheckpoint,
} from './persistence/stage4Rk3Checkpoint.js';
import { buildHeldSuarezReference } from './physics/heldSuarez.js';
import { buildRotationGeometry } from './physics/rotation.js';
import { diagnoseAxialAngularMomentum, diagnoseEddies, type AxialAngularMomentumDiagnostics, type EddyDiagnostics } from './solver/stage4CirculationDiagnostics.js';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const logEl=$<HTMLPreElement>('log');
const runBtn=$<HTMLButtonElement>('run');
const DIAGNOSTIC_KEY='stage4-rk3-spinup-diagnostic';
const DT=10,STEPS_PER_QUARTER=Math.round(21600/DT),BATCH=40;

type Point={day:number;aam:AxialAngularMomentumDiagnostics;eddy:EddyDiagnostics};
function log(s:string):void{logEl.textContent=`${s}\n${logEl.textContent||''}`.slice(0,50000);}
function valid(cp:Stage4Rk3ClimateCheckpoint|null):cp is Stage4Rk3ClimateCheckpoint{
  if(!cp)return false;
  try{assertStage4Rk3CheckpointShape(cp);}catch{return false;}
  return cp.modelSignature===STAGE4_RK3_PRODUCTION_MODEL_SIGNATURE;
}
async function latestCheckpoint():Promise<{cp:Stage4Rk3ClimateCheckpoint;kind:string}|null>{
  const production=await loadStage4Rk3Checkpoint(),diagnostic=await loadStage4Rk3Checkpoint(DIAGNOSTIC_KEY);
  const items=[{cp:production,kind:'production'},{cp:diagnostic,kind:'spin-up diagnostic'}].filter((x):x is {cp:Stage4Rk3ClimateCheckpoint;kind:string}=>valid(x.cp));
  items.sort((a,b)=>b.cp.completedOuterSteps-a.cp.completedOuterSteps);
  return items[0]??null;
}
function fmtExp(x:number):string{return Number.isFinite(x)?x.toExponential(4):'—';}
function fmt(x:number,d=4):string{return Number.isFinite(x)?x.toFixed(d):'—';}
async function yieldToBrowser():Promise<void>{
  const scheduler=(globalThis as any).scheduler;
  if(typeof scheduler?.yield==='function')await scheduler.yield();else await new Promise<void>(r=>setTimeout(r,0));
}
function diagnose(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,state:any,day:number,rotation:ReturnType<typeof buildRotationGeometry>):Point{
  return{day,aam:diagnoseAxialAngularMomentum(h,v,state,rotation),eddy:diagnoseEddies(h,v,state,24,rotation)};
}
function renderPoint(p:Point):void{
  $('day').textContent=p.day.toFixed(2);
  $('aam').textContent=fmtExp(p.aam.absolute);
  $('relAam').textContent=fmtExp(p.aam.relative);
  $('drag').textContent=fmtExp(p.aam.dragTorque)+' N m';
  $('eke').textContent=fmt(p.eddy.midlatitudeEke,3)+' m²/s²';
  $('heat').textContent=fmt(p.eddy.midlatitudePolewardHeatFlux,4)+' K·m/s';
  $('mom').textContent=fmt(p.eddy.midlatitudePolewardMomentumFlux,4)+' m²/s²';
}

async function refresh():Promise<void>{
  try{
    const src=await latestCheckpoint();
    if(!src){$('source').textContent='找不到相容 checkpoint / No compatible checkpoint';runBtn.disabled=true;return;}
    const day=src.cp.completedOuterSteps*DT/86400;
    $('source').textContent=`${src.kind}: day ${day.toFixed(2)}`;
    $('status').textContent='就緒 / Ready';runBtn.disabled=false;
  }catch(e){$('status').textContent='錯誤 / ERROR';$('source').textContent=String(e);runBtn.disabled=true;}
}

runBtn.onclick=()=>void(async()=>{
  runBtn.disabled=true;$('status').textContent='初始化 / Initializing';
  try{
    const src=await latestCheckpoint();if(!src)throw new Error('no compatible checkpoint');
    const h=buildCubedSphere(8),v=buildStretchedVerticalGrid(48,40000,1.4),ref=buildHeldSuarezReference(v),rotation=buildRotationGeometry(h);
    const startDay=src.cp.completedOuterSteps*DT/86400,startState=src.cp.state;
    const points:Point[]=[diagnose(h,v,startState,startDay,rotation)];renderPoint(points[0]!);
    log(`start day ${startDay.toFixed(2)} AAM=${fmtExp(points[0]!.aam.absolute)} relAAM=${fmtExp(points[0]!.aam.relative)} dragTorque=${fmtExp(points[0]!.aam.dragTorque)} EKE=${fmt(points[0]!.eddy.midlatitudeEke,4)} poleward_vT=${fmt(points[0]!.eddy.midlatitudePolewardHeatFlux,5)} poleward_uv=${fmt(points[0]!.eddy.midlatitudePolewardMomentumFlux,5)}`);
    const gpu=await GpuStage4Rk3SplitReference.create(h,v,ref,startState,4),opts={heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true} as const,t0=performance.now();
    try{
      for(let quarter=1;quarter<=4;quarter++){
        let left=STEPS_PER_QUARTER;
        while(left>0){const n=Math.min(BATCH,left);gpu.stepBatch(DT,n,opts);await gpu.device.queue.onSubmittedWorkDone();left-=n;await yieldToBrowser();}
        const elapsed=quarter*21600,day=startDay+quarter*.25,state=await gpu.downloadState(startState.time+elapsed),p=diagnose(h,v,state,day,rotation);points.push(p);renderPoint(p);
        $('status').textContent=`執行中 / Running — day ${day.toFixed(2)}`;
        log(`day ${day.toFixed(2)} AAM=${fmtExp(p.aam.absolute)} relAAM=${fmtExp(p.aam.relative)} dragTorque=${fmtExp(p.aam.dragTorque)} EKE=${fmt(p.eddy.midlatitudeEke,4)} poleward_vT=${fmt(p.eddy.midlatitudePolewardHeatFlux,5)} poleward_uv=${fmt(p.eddy.midlatitudePolewardMomentumFlux,5)}`);
      }
    }finally{gpu.destroy();}
    const first=points[0]!,last=points[points.length-1]!,totalSeconds=(last.day-first.day)*86400;
    const observed=(last.aam.absolute-first.aam.absolute)/totalSeconds;
    let dragIntegral=0;
    for(let i=1;i<points.length;i++)dragIntegral+=.5*(points[i-1]!.aam.dragTorque+points[i]!.aam.dragTorque)*(points[i]!.day-points[i-1]!.day)*86400;
    const meanDrag=dragIntegral/totalSeconds,residual=observed-meanDrag,lever=.5*(first.aam.torqueLeverMass+last.aam.torqueLeverMass),obsAccel=observed/lever*86400,dragAccel=meanDrag/lever*86400,resAccel=residual/lever*86400;
    $('observed').textContent=`${fmtExp(observed)} N m  (${obsAccel>=0?'+':''}${fmt(obsAccel,5)} m/s/day)`;
    $('meanDrag').textContent=`${fmtExp(meanDrag)} N m  (${dragAccel>=0?'+':''}${fmt(dragAccel,5)} m/s/day)`;
    $('residual').textContent=`${fmtExp(residual)} N m  (${resAccel>=0?'+':''}${fmt(resAccel,5)} m/s/day)`;
    $('ekeChange').textContent=`${fmt(first.eddy.midlatitudeEke,3)} → ${fmt(last.eddy.midlatitudeEke,3)} m²/s²`;
    $('heatChange').textContent=`${fmt(first.eddy.midlatitudePolewardHeatFlux,4)} → ${fmt(last.eddy.midlatitudePolewardHeatFlux,4)} K·m/s`;
    $('momChange').textContent=`${fmt(first.eddy.midlatitudePolewardMomentumFlux,4)} → ${fmt(last.eddy.midlatitudePolewardMomentumFlux,4)} m²/s²`;
    $('elapsed').textContent=((performance.now()-t0)/1000).toFixed(1)+' s';$('status').textContent='完成 / COMPLETE';
    log(`BUDGET observed=${fmtExp(observed)} Nm meanDrag=${fmtExp(meanDrag)} Nm residual=${fmtExp(residual)} Nm; equivalent accel observed=${obsAccel.toFixed(7)} drag=${dragAccel.toFixed(7)} residual=${resAccel.toFixed(7)} m/s/day.`);
  }catch(e){$('status').textContent='錯誤 / ERROR';log(`ERROR: ${String(e)}`);}finally{runBtn.disabled=false;}
})();

void refresh();
