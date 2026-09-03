import { buildCubedSphere } from './grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from './grid/vertical.js';
import { buildIsothermalReference } from './physics/referenceAtmosphere.js';
import { diagnoseState } from './solver/diagnostics.js';
import { DryCoreCpu } from './solver/dryCoreCpu.js';
import { addThermalBubble } from './solver/initialConditions.js';
import { createHydrostaticState } from './solver/state.js';
import { DebugViewer } from './render/debugViewer.js';
import { GpuDryCorePrototype } from './gpu/dryCoreGpu.js';
import { GPU_VALIDATION_DEFAULTS, GpuValidationSample, runHydrostaticGpuValidation } from './validation/gpuMultiStep.js';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const logEl=$<HTMLDivElement>('log');
const log=(x:string)=>{logEl.textContent=`${x}\n${logEl.textContent||''}`.slice(0,10000);};
const h=buildCubedSphere(8),v=buildStretchedVerticalGrid(32,40000,1.7),ref=buildIsothermalReference(v,288),cpu=new DryCoreCpu(h,v,ref);
let state=createHydrostaticState(h,v,ref),initialMass=diagnoseState(h,v,state).dryMass,running=false,gpuSmokePassed=false;
const viewer=new DebugViewer($<HTMLCanvasElement>('view'),h,v,ref);
const validateBtn=$<HTMLButtonElement>('gpuValidate');
const validationProgress=$<HTMLProgressElement>('gpuValidationProgress');
validateBtn.disabled=true;
validationProgress.max=GPU_VALIDATION_DEFAULTS.steps;
$('grid').textContent=`6 × ${h.n} × ${h.n} × ${v.nz}`;
$('cells').textContent=(h.cellCount*v.nz).toLocaleString();

function reset(){
  state=createHydrostaticState(h,v,ref);
  initialMass=diagnoseState(h,v,state).dryMass;
  running=false;
  $('run').textContent='執行 / Run';
  log('CPU 狀態已重設為等溫靜力參考大氣。 / CPU state reset to isothermal hydrostatic reference.');
}
function bubble(){
  addThermalBubble(h,v,ref,state,{lonDeg:0,latDeg:0,altitude:3500,horizontalRadius:1.1e6,verticalRadius:2200,deltaTheta:3});
  initialMass=diagnoseState(h,v,state).dryMass;
  log('已加入定壓 +3 K 熱泡：只改變物理初始條件，沒有強制指定向上速度。 / Injected constant-pressure +3 K thermal bubble: physics initial condition only; no upward velocity was forced.');
}
function step(n=1){
  try{
    for(let i=0;i<n;i++){
      const c=cpu.step(state,0.5);
      if(c.maxHorizontalCfl>0.8||c.maxVerticalCfl>0.8){
        running=false;
        $('run').textContent='執行 / Run';
        log(`CFL 防護觸發 / CFL guard: h=${c.maxHorizontalCfl.toFixed(3)}, v=${c.maxVerticalCfl.toFixed(3)}`);
        break;
      }
    }
  }catch(e){
    running=false;
    $('run').textContent='執行 / Run';
    log(`已暫停 / PAUSED: ${String(e)}`);
  }
}
function updateUi(){
  const d=diagnoseState(h,v,state),dr=(d.dryMass-initialMass)/initialMass;
  $('mass').textContent=dr.toExponential(3);
  $('wmax').textContent=`${d.maxAbsW.toFixed(4)} m/s`;
  $('rhomin').textContent=`${d.minRho.toExponential(3)} kg/m³`;
  $('pmin').textContent=`${d.minP.toFixed(1)} Pa`;
  $('time').textContent=`${state.time.toFixed(1)} s`;
  const bad=d.nan||d.minRho<=0||d.minP<=0;
  $('mass').className=Math.abs(dr)<1e-9?'ok':'bad';
  $('rhomin').className=bad?'bad':'ok';
  $('pmin').className=bad?'bad':'ok';
  viewer.draw(state);
}
$('reset').onclick=reset;
$('bubble').onclick=bubble;
$('step').onclick=()=>step(1);
$('run').onclick=()=>{running=!running;$('run').textContent=running?'暫停 / Pause':'執行 / Run';};
function frame(){if(running)step(2);updateUi();requestAnimationFrame(frame);}frame();

function showValidationSample(s:GpuValidationSample,totalSteps:number){
  validationProgress.value=s.step;
  $('gpuValidationProgressText').textContent=`${s.step.toLocaleString()} / ${totalSteps.toLocaleString()} steps · ${s.simTime.toFixed(2)} s`;
  $('gpuLongMass').textContent=s.gpuMassDrift.toExponential(3);
  $('gpuLongW').textContent=`${s.gpuMaxAbsW.toExponential(3)} m/s`;
  $('gpuRhoL2').textContent=s.rhoRelL2.toExponential(3);
  $('gpuRhoThetaL2').textContent=s.rhoThetaRelL2.toExponential(3);
  $('gpuUDiff').textContent=`${s.maxAbsUDiff.toExponential(3)} m/s`;
  $('gpuWDiff').textContent=`${s.maxAbsWDiff.toExponential(3)} m/s`;
}

async function runGpuValidation(){
  if(!gpuSmokePassed)return;
  running=false;
  $('run').textContent='執行 / Run';
  validateBtn.disabled=true;
  validationProgress.value=0;
  $('gpuValidationStatus').textContent='執行中 / Running';
  $('gpuValidationStatus').className='';
  $('gpuValidationProgressText').textContent=`0 / ${GPU_VALIDATION_DEFAULTS.steps.toLocaleString()} steps`;
  $('gpuValidationElapsed').textContent='—';
  log(`開始 GPU 多步靜力驗證：${GPU_VALIDATION_DEFAULTS.steps} steps × ${GPU_VALIDATION_DEFAULTS.dt} s。 / Starting GPU multi-step hydrostatic validation: ${GPU_VALIDATION_DEFAULTS.steps} steps × ${GPU_VALIDATION_DEFAULTS.dt} s.`);
  try{
    const result=await runHydrostaticGpuValidation(h,v,ref,(sample,totalSteps)=>{
      showValidationSample(sample,totalSteps);
      log(`GPU checkpoint ${sample.step}: mass=${sample.gpuMassDrift.toExponential(3)}, max|w|=${sample.gpuMaxAbsW.toExponential(3)} m/s, rho L2=${sample.rhoRelL2.toExponential(3)}, rhoTheta L2=${sample.rhoThetaRelL2.toExponential(3)}.`);
    });
    $('gpuValidationElapsed').textContent=`${(result.elapsedMs/1000).toFixed(2)} s`;
    if(result.passed){
      $('gpuValidationStatus').textContent='通過 / PASS';
      $('gpuValidationStatus').className='ok';
      log('GPU 多步靜力平衡、守恆與 CPU/GPU 一致性 gate 通過。 / GPU multi-step hydrostatic, conservation, and CPU/GPU agreement gate PASS.');
    }else{
      $('gpuValidationStatus').textContent='失敗 / FAIL';
      $('gpuValidationStatus').className='bad';
      log(`GPU 多步驗證失敗 / GPU multi-step validation FAIL:\n${result.failures.join('\n')}`);
    }
  }catch(e){
    $('gpuValidationStatus').textContent='錯誤 / ERROR';
    $('gpuValidationStatus').className='bad';
    log(`GPU 多步驗證無法完成 / GPU multi-step validation could not complete: ${String(e)}`);
  }finally{
    validateBtn.disabled=!gpuSmokePassed;
  }
}
validateBtn.onclick=()=>{void runGpuValidation();};

(async()=>{
  const gpuEl=$('gpu');
  try{
    const gpu=await GpuDryCorePrototype.create(h,v,ref,createHydrostaticState(h,v,ref));
    const supported=gpu.status.limits.maxStorageBuffersPerShaderStage;
    log(`WebGPU 儲存緩衝區上限 / storage-buffer limit: adapter=${supported ?? 'unknown'}, core requires=${gpu.status.requiredStorageBuffersPerStage}.`);
    gpu.step(0.25);
    const gs=await gpu.downloadState(0.25),d=diagnoseState(h,v,gs);
    if(d.nan||d.minRho<=0||d.minP<=0)throw new Error('GPU smoke produced invalid state');
    gpuEl.textContent='計算 smoke 測試通過 / Compute smoke OK';
    gpuEl.className='ok';
    gpuSmokePassed=true;
    validateBtn.disabled=false;
    log(`WebGPU pipeline 已編譯；靜力 smoke 最大 |w|=${d.maxAbsW.toExponential(3)} m/s。 / WebGPU pipelines compiled; hydrostatic smoke max |w|=${d.maxAbsW.toExponential(3)} m/s.`);
    gpu.destroy();
  }catch(e){
    gpuEl.textContent='不可用 / 失敗 / Unavailable / failed';
    gpuEl.className='bad';
    gpuSmokePassed=false;
    validateBtn.disabled=true;
    log(`WebGPU smoke 測試無法完成 / WebGPU smoke not available: ${String(e)}`);
  }
})();
