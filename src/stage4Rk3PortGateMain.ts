import { runStage4AcousticGpuAgreement } from './validation/stage4AcousticGpuAgreement.js';
import { runStage4SlowGpuAgreement } from './validation/stage4SlowGpuAgreement.js';
import { runStage4Rk3GpuAgreement } from './validation/stage4Rk3GpuAgreement.js';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const run=$<HTMLButtonElement>('run'),overall=$<HTMLElement>('overall'),log=$<HTMLPreElement>('log');
const set=(id:string,v:string,cls='')=>{const e=$<HTMLElement>(id);e.textContent=v;e.className=cls;};
const exp=(x:number)=>x.toExponential(3);

run.onclick=()=>void(async()=>{
  run.disabled=true;log.textContent='';set('overall','執行中：聲學柱 / Running: acoustic column');
  for(const id of ['aStatus','aRho','aX','aFrefRho','aFrefX','aW','aRestW','sStatus','sRho','sX','sHF','sVF','sU','sW','sRest','rStatus','rRho','rX','rU','rW','rMass','rCpuW','rGpuW','rRestRho','rRestX','rRestU','rRestW','rRestMass','rElapsed'])set(id,'—');
  try{
    const a=await runStage4AcousticGpuAgreement();
    set('aStatus',a.pass?'通過 / PASS':'失敗 / FAIL',a.pass?'ok':'bad');set('aRho',exp(a.rhoRelativeL2));set('aX',exp(a.rhoThetaRelativeL2));set('aFrefRho',exp(a.referenceMassFluxRelativeL2));set('aFrefX',exp(a.referenceRhoThetaFluxRelativeL2));set('aW',`${exp(a.maxDeltaW)} m/s`);set('aRestW',`${exp(a.hydrostaticMaxW)} m/s`);
    log.textContent+=`[Acoustic] ${a.pass?'PASS':'FAIL'} rho=${a.rhoRelativeL2} X=${a.rhoThetaRelativeL2} FrefRho=${a.referenceMassFluxRelativeL2} FrefX=${a.referenceRhoThetaFluxRelativeL2} maxDw=${a.maxDeltaW} restW=${a.hydrostaticMaxW}\n`;
    if(!a.pass){set('overall','停止：聲學柱 gate 失敗 / STOP: acoustic gate FAIL','bad');set('sStatus','未執行 / NOT RUN');set('rStatus','未執行 / NOT RUN');return;}

    set('overall','聲學柱通過；執行 slow RHS / Acoustic PASS; running slow RHS','ok');
    const s=await runStage4SlowGpuAgreement();
    set('sStatus',s.pass?'通過 / PASS':'失敗 / FAIL',s.pass?'ok':'bad');set('sRho',exp(s.rhoRelativeL2));set('sX',exp(s.rhoThetaRelativeL2));set('sHF',exp(s.hFluxRelativeL2));set('sVF',exp(s.vFluxRelativeL2));set('sU',`${exp(s.maxDeltaU)} m/s²`);set('sW',`${exp(s.maxDeltaW)} m/s²`);set('sRest',exp(s.restMax));
    log.textContent+=`[Slow RHS] ${s.pass?'PASS':'FAIL'} rho=${s.rhoRelativeL2} X=${s.rhoThetaRelativeL2} hF=${s.hFluxRelativeL2} vF=${s.vFluxRelativeL2} maxDu=${s.maxDeltaU} maxDw=${s.maxDeltaW} rest=${s.restMax}\n`;
    if(!s.pass){set('overall','停止：slow RHS gate 失敗 / STOP: slow RHS gate FAIL','bad');set('rStatus','未執行 / NOT RUN');return;}

    set('overall','兩個子系統通過；執行完整單步 RK3 / Subsystems PASS; running full one-step RK3','ok');
    const r=await runStage4Rk3GpuAgreement();
    set('rStatus',r.pass?'通過 / PASS':'失敗 / FAIL',r.pass?'ok':'bad');set('rRho',exp(r.rhoRelativeL2));set('rX',exp(r.rhoThetaRelativeL2));set('rU',`${exp(r.maxDeltaU)} m/s`);set('rW',`${exp(r.maxDeltaW)} m/s`);set('rMass',`CPU ${exp(r.cpuMassDrift)} / GPU ${exp(r.gpuMassDrift)}`);set('rCpuW',`${exp(r.cpuMaxW)} m/s`);set('rGpuW',`${exp(r.gpuMaxW)} m/s`);set('rRestRho',exp(r.restRhoRelativeL2));set('rRestX',exp(r.restRhoThetaRelativeL2));set('rRestU',`${exp(r.restGpuMaxU)} m/s`);set('rRestW',`${exp(r.restGpuMaxW)} m/s`);set('rRestMass',exp(r.restGpuMassDrift));set('rElapsed',`${(r.elapsedMs/1000).toFixed(2)} s`);
    log.textContent+=`[Full RK3] ${r.pass?'PASS':'FAIL'} rho=${r.rhoRelativeL2} X=${r.rhoThetaRelativeL2} maxDu=${r.maxDeltaU} maxDw=${r.maxDeltaW} massCPU=${r.cpuMassDrift} massGPU=${r.gpuMassDrift} cpuMaxW=${r.cpuMaxW} gpuMaxW=${r.gpuMaxW} restRho=${r.restRhoRelativeL2} restX=${r.restRhoThetaRelativeL2} restMaxU=${r.restGpuMaxU} restMaxW=${r.restGpuMaxW} restMass=${r.restGpuMassDrift}\n`;
    set('overall',r.pass?'三層 RK3 GPU gate 全部通過 / ALL THREE RK3 GPU GATES PASS':'子系統通過；完整單步 RK3 gate 失敗 / Subsystems PASS; full one-step RK3 FAIL',r.pass?'ok':'bad');
  }catch(e){set('overall','WebGPU 錯誤 / WebGPU ERROR','bad');log.textContent+=`${String(e)}\n`;}finally{run.disabled=false;}
})();
