import { buildCubedSphere } from './grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from './grid/vertical.js';
import { buildHeldSuarezReference } from './physics/heldSuarez.js';
import { createHydrostaticState } from './solver/state.js';
import { diagnoseState } from './solver/diagnostics.js';
import { GpuStage4Rk3SplitReference } from './gpu/stage4Rk3SplitGpu.js';
import { runStage4Rk3GpuMultistepAgreement } from './validation/stage4Rk3GpuMultistepAgreement.js';
import { runStage4Rk3Climate } from './validation/stage4Rk3Climate.js';
import type { Stage4Rk3ClimateProgress } from './validation/stage4Rk3Climate.js';
import type { ClimateDaySample } from './validation/stage4Gpu.js';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const logEl=$<HTMLDivElement>('log');
const log=(x:string)=>{logEl.textContent=`${x}\n${logEl.textContent||''}`.slice(0,50000)};
const agreeBtn=$<HTMLButtonElement>('agree'),climateBtn=$<HTMLButtonElement>('climate');
let smoke=false;
function lock(v:boolean){agreeBtn.disabled=v||!smoke;climateBtn.disabled=v||!smoke;}
function climateProgress(p:Stage4Rk3ClimateProgress){
  const pct=100*p.completedOuterSteps/p.totalOuterSteps;
  $('climateStatus').textContent=`執行中 ${pct.toFixed(1)}% / Running ${pct.toFixed(1)}%`;
  $('climateDay').textContent=`${p.simulatedDay.toFixed(3)} / ${p.targetDays} (GPU-complete)`;
  $('climateElapsed').textContent=`${(p.elapsedMs/1000).toFixed(1)} s`;
}
function climateSample(s:ClimateDaySample){
  $('climateDay').textContent=`${s.day.toFixed(s.day%1===0?0:2)} / 30`;
  $('climateMass').textContent=s.massDrift.toExponential(3);
  $('climateJet').textContent=`${s.jet.toFixed(3)} m/s`;
  $('climateTrade').textContent=`${s.trade.toFixed(3)} m/s`;
  $('climatePsi').textContent=s.psi.toExponential(3);
  $('climateW').textContent=`${s.maxW.toExponential(3)} m/s`;
  $('climateWBelow').textContent=`${s.maxWBelowSponge.toExponential(3)} m/s`;
  $('climateWSponge').textContent=`${s.maxWInSponge.toExponential(3)} m/s`;
  $('climateWLocation').textContent=`${(s.maxWAltitude/1000).toFixed(2)} km, ${s.maxWLatitude.toFixed(1)}°`;
  $('climateMaxU').textContent=`${s.maxEdgeWind.toFixed(3)} m/s`;
  $('climateDiv').textContent=`${s.divergenceRms.toExponential(3)} s⁻¹`;
  $('climateHCfl').textContent=s.maxHorizontalCfl.toExponential(3);
  $('climateVCfl').textContent=s.maxVerticalCfl.toExponential(3);
  log(`RK3 Held–Suarez day ${s.day.toFixed(2)}: mass=${s.massDrift.toExponential(3)}, jet=${s.jet.toFixed(3)}, trade=${s.trade.toFixed(3)}, psi=${s.psi.toExponential(3)}, max|w|=${s.maxW.toExponential(3)} [below=${s.maxWBelowSponge.toExponential(3)}, absorber=${s.maxWInSponge.toExponential(3)}, z=${(s.maxWAltitude/1000).toFixed(2)}km, lat=${s.maxWLatitude.toFixed(1)}deg], max|u_edge|=${s.maxEdgeWind.toFixed(3)}, divRMS=${s.divergenceRms.toExponential(3)}, CFL(h/v)=${s.maxHorizontalCfl.toExponential(2)}/${s.maxVerticalCfl.toExponential(2)}.`);
}

agreeBtn.onclick=()=>void(async()=>{
  lock(true);$('agreeStatus').textContent='執行中 / Running';$('agreeStatus').className='';
  try{
    const r=await runStage4Rk3GpuMultistepAgreement();
    $('agreeStep').textContent=String(r.steps);
    $('agreeMass').textContent=r.gpuMassDrift.toExponential(3);
    $('agreeRho').textContent=r.rhoRelativeL2.toExponential(3);
    $('agreeTheta').textContent=r.rhoThetaRelativeL2.toExponential(3);
    $('agreeU').textContent=`${r.maxDeltaU.toExponential(3)} m/s`;
    $('agreeW').textContent=`${r.maxDeltaW.toExponential(3)} m/s`;
    $('agreeElapsed').textContent=`${(r.elapsedMs/1000).toFixed(2)} s`;
    $('agreeStatus').textContent=r.pass?'通過 / PASS':'失敗 / FAIL';$('agreeStatus').className=r.pass?'ok':'bad';
    log(`[RK3 40-step batched agreement] ${r.pass?'PASS':'FAIL'} rho=${r.rhoRelativeL2} X=${r.rhoThetaRelativeL2} du=${r.maxDeltaU} dw=${r.maxDeltaW} massGPU=${r.gpuMassDrift} massCPU=${r.cpuMassDrift}.`);
  }catch(e){$('agreeStatus').textContent='錯誤 / ERROR';$('agreeStatus').className='bad';log(`RK3 agreement error: ${String(e)}`);}finally{lock(false);}
})();

climateBtn.onclick=()=>void(async()=>{
  lock(true);$('climateStatus').textContent='執行中 0.0% / Running 0.0%';$('climateStatus').className='';$('climateDay').textContent='0.000 / 30 (GPU-complete)';$('climateElapsed').textContent='0.0 s';
  log('30 日 production gate：使用真機已驗證的 40-step GPU batch；每批完成後等待 GPU queue 並回報實際完成進度。數值核心仍為 3-stage RK3 predictor restart + predictor-relative split-explicit acoustic substeps (1×dt/3, 2×dt/4, 4×dt/4) + vertically implicit acoustic/gravity solve + Held–Suarez + complete 3-D momentum + Coriolis + implicit top absorber + acoustic divergence damping；outer dt=10 s。 / 30-day production gate uses the verified 40-step GPU batch and reports GPU-completed progress after every batch.');
  try{
    const r=await runStage4Rk3Climate(30,climateSample,climateProgress);
    $('climateElapsed').textContent=`${(r.elapsedMs/1000).toFixed(2)} s`;
    $('climateStatus').textContent=r.passed?'通過 / PASS':'失敗 / FAIL';$('climateStatus').className=r.passed?'ok':'bad';
    log(r.passed?'30 日 RK3 Held–Suarez production gate 通過。 / 30-day RK3 Held–Suarez production gate PASS.':`30-day RK3 Held–Suarez gate FAIL:\n${r.failures.join('\n')}`);
  }catch(e){$('climateStatus').textContent='錯誤 / ERROR';$('climateStatus').className='bad';log(`RK3 Held–Suarez climate error: ${String(e)}`);}finally{lock(false);}
})();

(async()=>{
  try{
    const h=buildCubedSphere(4),v=buildStretchedVerticalGrid(12,30000,1.4),ref=buildHeldSuarezReference(v),gpu=await GpuStage4Rk3SplitReference.create(h,v,ref,createHydrostaticState(h,v,ref),4);
    gpu.step(10,{heldSuarez:false,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true});
    const s=await gpu.downloadState(10),d=diagnoseState(h,v,s);
    if(d.nan||d.minRho<=0||d.minP<=0)throw new Error('invalid Stage 4 RK3 smoke state');
    gpu.destroy();smoke=true;
    $('gpuStatus').textContent='RK3 split-explicit production GPU smoke 通過 / RK3 split-explicit production GPU smoke PASS';$('gpuStatus').className='ok';
    log(`Stage 4 RK3 production smoke PASS; max|w|=${d.maxAbsW.toExponential(3)} m/s.`);lock(false);
  }catch(e){smoke=false;$('gpuStatus').textContent='失敗 / FAIL';$('gpuStatus').className='bad';agreeBtn.disabled=true;climateBtn.disabled=true;log(`Stage 4 RK3 production smoke failed: ${String(e)}`);}
})();
