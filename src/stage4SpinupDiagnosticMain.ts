import {
  assertStage4Rk3CheckpointShape,
  clearStage4Rk3Checkpoint,
  cloneStage4Rk3Checkpoint,
  loadStage4Rk3Checkpoint,
  saveStage4Rk3Checkpoint,
  STAGE4_RK3_PRODUCTION_MODEL_SIGNATURE,
  type Stage4Rk3ClimateCheckpoint,
} from './persistence/stage4Rk3Checkpoint.js';
import { runStage4Rk3Climate, type Stage4Rk3ClimateProgress } from './validation/stage4Rk3Climate.js';
import type { ClimateDaySample } from './validation/stage4Gpu.js';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const runBtn=$<HTMLButtonElement>('run');
const clearBtn=$<HTMLButtonElement>('clear');
const targetSelect=$<HTMLSelectElement>('target');
const logEl=$<HTMLPreElement>('log');
const DIAGNOSTIC_KEY='stage4-rk3-spinup-diagnostic';
const DT=10;
let busy=false;
let source:Stage4Rk3ClimateCheckpoint|null=null;

function log(message:string):void{
  logEl.textContent=`${message}\n${logEl.textContent||''}`.slice(0,60000);
}
function dayOf(cp:Stage4Rk3ClimateCheckpoint):number{return cp.completedOuterSteps*DT/86400;}
function setBusy(value:boolean):void{
  busy=value;
  runBtn.disabled=value||source===null;
  clearBtn.disabled=value;
  targetSelect.disabled=value;
}
function validModel(cp:Stage4Rk3ClimateCheckpoint|null):cp is Stage4Rk3ClimateCheckpoint{
  if(!cp)return false;
  try{assertStage4Rk3CheckpointShape(cp);}catch{return false;}
  return cp.modelSignature===STAGE4_RK3_PRODUCTION_MODEL_SIGNATURE;
}
function linearSlope(samples:ClimateDaySample[]):number{
  if(samples.length<2)return NaN;
  let sx=0,sy=0,sxx=0,sxy=0;
  for(const s of samples){sx+=s.day;sy+=s.trade;sxx+=s.day*s.day;sxy+=s.day*s.trade;}
  const n=samples.length,den=n*sxx-sx*sx;
  return Math.abs(den)>0?(n*sxy-sx*sy)/den:NaN;
}
function renderSample(s:ClimateDaySample):void{
  $('day').textContent=s.day.toFixed(2);
  $('trade').textContent=`${s.trade.toFixed(3)} m/s`;
  $('jet').textContent=`${s.jet.toFixed(3)} m/s`;
  $('psi').textContent=s.psi.toExponential(3);
  $('maxW').textContent=`${s.maxW.toExponential(3)} m/s`;
  $('mass').textContent=s.massDrift.toExponential(3);
}
function progress(p:Stage4Rk3ClimateProgress):void{
  const pct=100*p.completedOuterSteps/p.totalOuterSteps;
  $('status').textContent=`執行中 ${pct.toFixed(1)}% / Running ${pct.toFixed(1)}%`;
  $('day').textContent=p.simulatedDay.toFixed(3);
  $('elapsed').textContent=`${(p.elapsedMs/1000).toFixed(1)} s`;
}
async function refresh():Promise<void>{
  setBusy(true);
  try{
    const production=await loadStage4Rk3Checkpoint();
    const diagnostic=await loadStage4Rk3Checkpoint(DIAGNOSTIC_KEY);
    const candidates=[production,diagnostic].filter(validModel).sort((a,b)=>b.completedOuterSteps-a.completedOuterSteps);
    source=candidates[0]??null;
    if(!source){
      $('source').textContent='找不到相容 checkpoint / No compatible checkpoint';
      $('source').className='bad';
      $('status').textContent='等待 checkpoint / Waiting for checkpoint';
      return;
    }
    const d=dayOf(source),kind=source===diagnostic?'spin-up diagnostic':'production';
    $('source').textContent=`${kind}: day ${d.toFixed(2)}`;
    $('source').className='ok';
    const last=source.samples[source.samples.length-1];
    if(last)renderSample(last);
    $('status').textContent='就緒 / Ready';
    const target=Math.max(60,Number(targetSelect.value)||60);
    if(d>=target)log(`目前 checkpoint day ${d.toFixed(2)} 已達所選 target day ${target}. / Current checkpoint already reaches selected target.`);
  }catch(e){
    source=null;
    $('source').textContent=`錯誤 / ERROR — ${String(e)}`;
    $('source').className='bad';
    $('status').textContent='錯誤 / ERROR';
  }finally{setBusy(false);}
}

runBtn.onclick=()=>void(async()=>{
  const base=source;
  if(!base)return;
  const target=Number(targetSelect.value);
  if(!Number.isInteger(target)||target<1){log('Invalid target day.');return;}
  const startDay=dayOf(base);
  if(startDay>=target){log(`checkpoint day ${startDay.toFixed(2)} 已經 >= target day ${target}。`);return;}
  const resume=cloneStage4Rk3Checkpoint(base);
  // targetDays is persistence metadata, not part of the numerical state. Retag a
  // compatible checkpoint so the generic climate runner can continue farther.
  resume.targetDays=target;
  resume.savedAt=Date.now();
  setBusy(true);
  $('status').textContent='初始化 / Initializing';
  log(`從 day ${startDay.toFixed(2)} 延伸 Held–Suarez spin-up 到 day ${target}；使用獨立 diagnostic checkpoint，不改 production checkpoint。 / Extending spin-up from day ${startDay.toFixed(2)} to day ${target}.`);
  try{
    const result=await runStage4Rk3Climate(
      target,
      s=>{renderSample(s);log(`day ${s.day.toFixed(2)} trade=${s.trade.toFixed(3)} m/s jet=${s.jet.toFixed(3)} psi=${s.psi.toExponential(3)} max|w|=${s.maxW.toExponential(3)} mass=${s.massDrift.toExponential(3)}`);},
      progress,
      {resume,onCheckpoint:cp=>saveStage4Rk3Checkpoint(cp,DIAGNOSTIC_KEY)},
    );
    const final=result.samples[result.samples.length-1];
    if(final)renderSample(final);
    const recent=result.samples.filter(s=>final&&s.day>=final.day-5);
    const slope=linearSlope(recent);
    $('slope').textContent=Number.isFinite(slope)?`${slope>=0?'+':''}${slope.toFixed(3)} m/s/day`:'—';
    $('elapsed').textContent=`${(result.elapsedMs/1000).toFixed(1)} s`;
    $('endpoint').textContent=result.passed?'原 30-day-style endpoint gates PASS / PASS':'原 endpoint gates FAIL / FAIL';
    $('endpoint').className=result.passed?'ok':'bad';
    $('status').textContent='完成 / COMPLETE';
    log(`spin-up continuation complete to day ${target}; endpoint=${result.passed?'PASS':'FAIL'}; failures=${result.failures.join(' | ')||'none'}; last-5-day trade slope=${Number.isFinite(slope)?slope.toFixed(6):'n/a'} m/s/day.`);
  }catch(e){
    $('status').textContent='錯誤 / ERROR';
    $('status').className='bad';
    log(`spin-up diagnostic error: ${String(e)}`);
  }finally{
    await refresh();
    setBusy(false);
  }
})();

clearBtn.onclick=()=>void(async()=>{
  setBusy(true);
  try{
    await clearStage4Rk3Checkpoint(DIAGNOSTIC_KEY);
    log('已清除 spin-up diagnostic checkpoint；production checkpoint 未更動。 / Cleared diagnostic checkpoint; production checkpoint is unchanged.');
  }catch(e){log(`clear diagnostic checkpoint error: ${String(e)}`);}finally{await refresh();setBusy(false);}
})();

targetSelect.onchange=()=>void refresh();
void refresh();
