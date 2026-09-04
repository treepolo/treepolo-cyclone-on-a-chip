import { buildCubedSphere } from './grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from './grid/vertical.js';
import { addHeldSuarezWavePerturbation, buildHeldSuarezReference } from './physics/heldSuarez.js';
import { diagnoseState } from './solver/diagnostics.js';
import { createHydrostaticState, type DryState } from './solver/state.js';
import { GpuStage4Rk3SplitReference } from './gpu/stage4Rk3SplitGpu.js';
import {
  assertStage4Rk3CheckpointCompatible,
  clearStage4Rk3Checkpoint,
  loadStage4Rk3Checkpoint,
  saveStage4Rk3Checkpoint,
  stage4Rk3StoragePersistenceStatus,
  STAGE4_RK3_CHECKPOINT_SCHEMA_VERSION,
  STAGE4_RK3_CHECKPOINT_TEST_KEY,
  type Stage4Rk3ClimateCheckpoint,
} from './persistence/stage4Rk3Checkpoint.js';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const status=$<HTMLSpanElement>('status'),phaseEl=$<HTMLSpanElement>('phase'),storageEl=$<HTMLSpanElement>('storage'),rhoEl=$<HTMLSpanElement>('rho'),xEl=$<HTMLSpanElement>('x'),uEl=$<HTMLSpanElement>('u'),wEl=$<HTMLSpanElement>('w'),deleteEl=$<HTMLSpanElement>('deleted'),compatEl=$<HTMLSpanElement>('compat'),logEl=$<HTMLDivElement>('log'),runBtn=$<HTMLButtonElement>('run');
const log=(x:string)=>{logEl.textContent=`${x}\n${logEl.textContent||''}`.slice(0,30000)};
const PHASE_KEY='stage4-rk3-checkpoint-test-phase';
const TEST_SIGNATURE='stage4-rk3-checkpoint-self-test-v1|N2|NZ24|H40000|stretch1.35|dt10|acoustic4';
const DT=10,BEFORE=4,AFTER=4;
const opts={heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true} as const;

function fixture(){
  const h=buildCubedSphere(2),v=buildStretchedVerticalGrid(24,40000,1.35),ref=buildHeldSuarezReference(v),seed=createHydrostaticState(h,v,ref);
  addHeldSuarezWavePerturbation(h,v,ref,seed,.05);
  return{h,v,ref,seed};
}
function relativeL2(a:Float64Array,b:Float64Array):number{
  let n=0,d=0;
  for(let i=0;i<a.length;i++){const av=a[i]!,dv=av-b[i]!;n+=dv*dv;d+=av*av;}
  return Math.sqrt(n/Math.max(d,1e-300));
}
function maxDelta(a:Float64Array,b:Float64Array):number{
  let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]!-b[i]!));return m;
}
async function runGpu(state:DryState,steps:number,time:number):Promise<DryState>{
  const{h,v,ref}=fixture();
  const gpu=await GpuStage4Rk3SplitReference.create(h,v,ref,state,4);
  try{
    gpu.stepBatch(DT,steps,opts);
    await gpu.device.queue.onSubmittedWorkDone();
    return await gpu.downloadState(time);
  }finally{gpu.destroy();}
}
async function phaseOne():Promise<void>{
  runBtn.disabled=true;status.textContent='執行中 / Running';status.className='';phaseEl.textContent='1 / 2：建立 checkpoint 後重新載入頁面';
  await clearStage4Rk3Checkpoint(STAGE4_RK3_CHECKPOINT_TEST_KEY);
  const{h,v,ref,seed}=fixture(),m0=diagnoseState(h,v,seed).dryMass;
  log(`phase 1: run ${BEFORE} steps, download full prognostic state, save to IndexedDB.`);
  const checkpointState=await runGpu(seed,BEFORE,BEFORE*DT);
  const cp:Stage4Rk3ClimateCheckpoint={schemaVersion:STAGE4_RK3_CHECKPOINT_SCHEMA_VERSION,modelSignature:TEST_SIGNATURE,savedAt:Date.now(),targetDays:1,completedOuterSteps:BEFORE,initialDryMass:m0,state:checkpointState,samples:[]};
  await saveStage4Rk3Checkpoint(cp,STAGE4_RK3_CHECKPOINT_TEST_KEY);
  const immediate=await loadStage4Rk3Checkpoint(STAGE4_RK3_CHECKPOINT_TEST_KEY);
  if(!immediate)throw new Error('checkpoint vanished immediately after IndexedDB save');
  if(immediate.state.rhoD.length!==checkpointState.rhoD.length||immediate.state.uEdge.length!==checkpointState.uEdge.length)throw new Error('checkpoint array lengths changed during IndexedDB round-trip');
  sessionStorage.setItem(PHASE_KEY,'resume');
  phaseEl.textContent='1 / 2 PASS：checkpoint 已寫入；正在重新載入整個頁面…';
  log('phase 1 PASS: IndexedDB write/read succeeded; forcing a real page reload now.');
  location.reload();
}
async function phaseTwo():Promise<void>{
  runBtn.disabled=true;status.textContent='執行中 / Running';status.className='';phaseEl.textContent='2 / 2：重載後恢復並驗證 continuation';
  const cp=await loadStage4Rk3Checkpoint(STAGE4_RK3_CHECKPOINT_TEST_KEY);
  if(!cp)throw new Error('checkpoint missing after real page reload');
  assertStage4Rk3CheckpointCompatible(cp,TEST_SIGNATURE,1);
  let incompatibleRejected=false;
  try{assertStage4Rk3CheckpointCompatible({...cp,modelSignature:'intentionally-incompatible'},TEST_SIGNATURE,1);}catch{incompatibleRejected=true;}
  compatEl.textContent=incompatibleRejected?'PASS':'FAIL';compatEl.className=incompatibleRejected?'ok':'bad';
  if(!incompatibleRejected)throw new Error('incompatible checkpoint was not rejected');
  log(`phase 2: loaded checkpoint after reload at t=${cp.state.time}s; destroy/recreate GPU boundary is real.`);
  const resumed=await runGpu(cp.state,AFTER,(BEFORE+AFTER)*DT);
  const{seed}=fixture();
  const uninterrupted=await runGpu(seed,BEFORE+AFTER,(BEFORE+AFTER)*DT);
  const rho=relativeL2(uninterrupted.rhoD,resumed.rhoD),x=relativeL2(uninterrupted.rhoThetaM,resumed.rhoThetaM),u=maxDelta(uninterrupted.uEdge,resumed.uEdge),w=maxDelta(uninterrupted.wInterface,resumed.wInterface);
  rhoEl.textContent=rho.toExponential(3);xEl.textContent=x.toExponential(3);uEl.textContent=`${u.toExponential(3)} m/s`;wEl.textContent=`${w.toExponential(3)} m/s`;
  const numericalPass=rho<=1e-7&&x<=1e-7&&u<=1e-6&&w<=1e-5;
  await clearStage4Rk3Checkpoint(STAGE4_RK3_CHECKPOINT_TEST_KEY);
  const afterDelete=await loadStage4Rk3Checkpoint(STAGE4_RK3_CHECKPOINT_TEST_KEY);
  const deletePass=afterDelete===null;deleteEl.textContent=deletePass?'PASS':'FAIL';deleteEl.className=deletePass?'ok':'bad';
  sessionStorage.removeItem(PHASE_KEY);
  const pass=numericalPass&&deletePass&&incompatibleRejected;
  status.textContent=pass?'持久化 checkpoint gate 通過 / PERSISTENT CHECKPOINT GATE PASS':'持久化 checkpoint gate 失敗 / FAIL';status.className=pass?'ok':'bad';phaseEl.textContent='2 / 2 完成 / COMPLETE';
  log(`[checkpoint reload gate] ${pass?'PASS':'FAIL'} rho=${rho} X=${x} maxDu=${u} maxDw=${w} delete=${deletePass} incompatibleRejected=${incompatibleRejected}`);
  runBtn.disabled=false;
}
async function init():Promise<void>{
  storageEl.textContent=await stage4Rk3StoragePersistenceStatus();
  if(sessionStorage.getItem(PHASE_KEY)==='resume'){
    try{await phaseTwo();}catch(e){sessionStorage.removeItem(PHASE_KEY);status.textContent=`錯誤 / ERROR — ${String(e)}`;status.className='bad';log(String(e));runBtn.disabled=false;}
  }
}
runBtn.onclick=()=>void phaseOne().catch(e=>{sessionStorage.removeItem(PHASE_KEY);status.textContent=`錯誤 / ERROR — ${String(e)}`;status.className='bad';log(String(e));runBtn.disabled=false;});
void init();
