import { buildCubedSphere } from './grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from './grid/vertical.js';
import { loadCompatibleStage4ResolutionCheckpoint, saveStage4ResolutionCheckpoint } from './persistence/stage4ResolutionCheckpoint.js';
import { buildHeldSuarezReference } from './physics/heldSuarez.js';
import { buildRotationGeometry } from './physics/rotation.js';
import { diagnoseAxialAngularMomentum, diagnoseEddies } from './solver/stage4CirculationDiagnostics.js';
import { diagnoseStage4InstantAamBreakdown } from './solver/stage4MomentumBudgetDiagnostics.js';
import { runStage4ResolutionClimate } from './validation/stage4ResolutionClimate.js';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const params=new URLSearchParams(location.search),n=Math.max(4,Math.round(Number(params.get('n')??16))),days=Math.max(1,Math.round(Number(params.get('days')??5)));
const run=$<HTMLButtonElement>('run'),log=$<HTMLPreElement>('log');
$('config').textContent=`N=${n} × 48, ${days} day${days===1?'':'s'}, dt=10 s`;
run.onclick=()=>void(async()=>{
  run.disabled=true;log.textContent='';$('status').textContent='CHECKING RESUMABLE CHECKPOINT';const started=performance.now();
  try{
    const resume=await loadCompatibleStage4ResolutionCheckpoint(n,days),resumeDay=resume?resume.state.time/86400:0;
    if(resume){const last=resume.samples[resume.samples.length-1];$('status').textContent=`RESUMING N${n} FROM DAY ${resumeDay.toFixed(2)}`;$('day').textContent=resumeDay.toFixed(2);if(last){$('trade').textContent=`${last.trade.toFixed(4)} m/s`;$('jet').textContent=`${last.jet.toFixed(4)} m/s`;$('mass').textContent=last.massDrift.toExponential(4);}}
    else $('status').textContent='INITIALIZING FRESH DAY-0 STATE';
    const result=await runStage4ResolutionClimate(n,days,s=>{
      $('day').textContent=s.day.toFixed(2);$('trade').textContent=`${s.trade.toFixed(4)} m/s`;$('jet').textContent=`${s.jet.toFixed(4)} m/s`;$('mass').textContent=s.massDrift.toExponential(4);
    },p=>{$('status').textContent=`N${n} day ${p.simulatedDay.toFixed(3)} / ${days}${resume?' (resumed)':''}`;$('elapsed').textContent=`${(p.elapsedMs/1000).toFixed(1)} s`;},{resume,onCheckpoint:cp=>saveStage4ResolutionCheckpoint(n,days,cp)});
    const h=buildCubedSphere(n),v=buildStretchedVerticalGrid(48,40000,1.4),ref=buildHeldSuarezReference(v),g=buildRotationGeometry(h),aam=diagnoseAxialAngularMomentum(h,v,result.finalState,g),eddy=diagnoseEddies(h,v,result.finalState,24,g),instant=diagnoseStage4InstantAamBreakdown(h,v,ref,result.finalState,g),z=result.finalZonal;
    log.textContent=JSON.stringify({
      model:'Stage4 production equations; resumable fresh day-0 resolution-controlled diagnostic',horizontalN:n,days,resumedFromDay:resumeDay,elapsedMs:result.elapsedMs,failures:result.failures,samples:result.samples,
      finalZonal:{bins:z.bins,latDeg:Array.from(z.latDeg),temperature:Array.from(z.temperature),zonalWind:Array.from(z.zonalWind),meridionalWind:Array.from(z.meridionalWind),streamfunction:Array.from(z.streamfunction),maxUpperMidlatitudeWesterly:z.maxUpperMidlatitudeWesterly,meanTropicalLowLevelZonal:z.meanTropicalLowLevelZonal,maxAbsStreamfunction:z.maxAbsStreamfunction,nhDominantStreamfunction:z.nhDominantStreamfunction,shDominantStreamfunction:z.shDominantStreamfunction},
      finalAam:aam,finalEddy:eddy,instantaneousAamBreakdown:instant,
    },null,2);
    $('elapsed').textContent=`${((performance.now()-started)/1000).toFixed(1)} s`;$('status').textContent=result.failures.length?'COMPLETE WITH STABILITY FAILURE':'COMPLETE';
  }catch(e){$('status').textContent='ERROR';log.textContent=String(e instanceof Error?e.stack||e.message:e);}finally{run.disabled=false;}
})();
