import { runStage4AcousticGpuAgreement } from './validation/stage4AcousticGpuAgreement.js';
import { runStage4SlowGpuAgreement } from './validation/stage4SlowGpuAgreement.js';
import { runStage4Rk3GpuAgreement } from './validation/stage4Rk3GpuAgreement.js';
import { runStage4Rk3GpuMultistepAgreement } from './validation/stage4Rk3GpuMultistepAgreement.js';
import { runStage4Rk3GpuStageDiagnostic } from './validation/stage4Rk3GpuStageDiagnostic.js';
import { runStage4Rk3TwoDayClimate } from './validation/stage4Rk3Climate.js';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const run=$<HTMLButtonElement>('run'),overall=$<HTMLElement>('overall'),log=$<HTMLPreElement>('log');
const set=(id:string,v:string,cls='')=>{const e=$<HTMLElement>(id);e.textContent=v;e.className=cls;};
const exp=(x:number)=>x.toExponential(3);
const notRun=()=>set('cStatus','未執行 / NOT RUN');

run.onclick=()=>void(async()=>{
  run.disabled=true;log.textContent='';set('overall','執行中：聲學柱 / Running: acoustic column');
  for(const id of ['aStatus','aRho','aX','aFrefRho','aFrefX','aFrefSelfRho','aFrefSelfX','aW','aRestW','sStatus','sRho','sX','sHF','sVF','sHFSelf','sVFSelf','sRhoDivSelf','sXDivSelf','sThermal','sU','sW','sRest','rStatus','rRho','rX','rU','rW','rMass','rCpuW','rGpuW','rRestRho','rRestX','rRestU','rRestW','rRestMass','rElapsed','mStatus','mRho','mX','mU','mW','mMass','mCpuW','mGpuW','mElapsed','cStatus','cDay','cMass','cW','cBelow','cAbsorber','cU','cDiv','cCfl','cElapsed'])set(id,'—');
  try{
    const a=await runStage4AcousticGpuAgreement();
    set('aStatus',a.pass?'通過 / PASS':'失敗 / FAIL',a.pass?'ok':'bad');set('aRho',exp(a.rhoRelativeL2));set('aX',exp(a.rhoThetaRelativeL2));set('aFrefRho',exp(a.referenceMassFluxRelativeL2));set('aFrefX',exp(a.referenceRhoThetaFluxRelativeL2));set('aFrefSelfRho',exp(a.referenceMassFluxSelfRelativeL2));set('aFrefSelfX',exp(a.referenceRhoThetaFluxSelfRelativeL2));set('aW',`${exp(a.maxDeltaW)} m/s`);set('aRestW',`${exp(a.hydrostaticMaxW)} m/s`);
    log.textContent+=`[Acoustic] ${a.pass?'PASS':'FAIL'} rho=${a.rhoRelativeL2} X=${a.rhoThetaRelativeL2} FrefRawRho=${a.referenceMassFluxRelativeL2} FrefRawX=${a.referenceRhoThetaFluxRelativeL2} FrefSelfRho=${a.referenceMassFluxSelfRelativeL2} FrefSelfX=${a.referenceRhoThetaFluxSelfRelativeL2} maxDw=${a.maxDeltaW} restW=${a.hydrostaticMaxW}\n`;
    if(!a.pass){set('overall','停止：聲學柱 gate 失敗 / STOP: acoustic gate FAIL','bad');set('sStatus','未執行 / NOT RUN');set('rStatus','未執行 / NOT RUN');set('mStatus','未執行 / NOT RUN');notRun();return;}

    set('overall','聲學柱通過；執行 slow RHS / Acoustic PASS; running slow RHS','ok');
    const s=await runStage4SlowGpuAgreement();
    set('sStatus',s.pass?'通過 / PASS':'失敗 / FAIL',s.pass?'ok':'bad');set('sRho',exp(s.rhoRelativeL2));set('sX',exp(s.rhoThetaRelativeL2));set('sHF',exp(s.hFluxRelativeL2));set('sVF',exp(s.vFluxRelativeL2));set('sHFSelf',exp(s.hFluxSelfRelativeL2));set('sVFSelf',exp(s.vFluxSelfRelativeL2));set('sRhoDivSelf',exp(s.rhoDivergenceSelfRelativeL2));set('sXDivSelf',exp(s.rhoThetaDivergenceSelfRelativeL2));set('sThermal',exp(s.thermalRelativeL2));set('sU',`${exp(s.maxDeltaU)} m/s²`);set('sW',`${exp(s.maxDeltaW)} m/s²`);set('sRest',exp(s.restMax));
    log.textContent+=`[Slow RHS] ${s.pass?'PASS':'FAIL'} rawRho=${s.rhoRelativeL2} rawX=${s.rhoThetaRelativeL2} rawHF=${s.hFluxRelativeL2} rawVF=${s.vFluxRelativeL2} hFSelf=${s.hFluxSelfRelativeL2} vFSelf=${s.vFluxSelfRelativeL2} rhoDivSelf=${s.rhoDivergenceSelfRelativeL2} XDivSelf=${s.rhoThetaDivergenceSelfRelativeL2} thermal=${s.thermalRelativeL2} maxDu=${s.maxDeltaU} maxDw=${s.maxDeltaW} rest=${s.restMax}\n`;
    if(!s.pass){set('overall','停止：slow RHS gate 失敗 / STOP: slow RHS gate FAIL','bad');set('rStatus','未執行 / NOT RUN');set('mStatus','未執行 / NOT RUN');notRun();return;}

    set('overall','兩個子系統通過；執行完整單步 RK3 / Subsystems PASS; running full one-step RK3','ok');
    const r=await runStage4Rk3GpuAgreement();
    set('rStatus',r.pass?'通過 / PASS':'失敗 / FAIL',r.pass?'ok':'bad');set('rRho',exp(r.rhoRelativeL2));set('rX',exp(r.rhoThetaRelativeL2));set('rU',`${exp(r.maxDeltaU)} m/s`);set('rW',`${exp(r.maxDeltaW)} m/s`);set('rMass',`CPU ${exp(r.cpuMassDrift)} / GPU ${exp(r.gpuMassDrift)}`);set('rCpuW',`${exp(r.cpuMaxW)} m/s`);set('rGpuW',`${exp(r.gpuMaxW)} m/s`);set('rRestRho',exp(r.restRhoRelativeL2));set('rRestX',exp(r.restRhoThetaRelativeL2));set('rRestU',`${exp(r.restGpuMaxU)} m/s`);set('rRestW',`${exp(r.restGpuMaxW)} m/s`);set('rRestMass',exp(r.restGpuMassDrift));set('rElapsed',`${(r.elapsedMs/1000).toFixed(2)} s`);
    log.textContent+=`[Full RK3] ${r.pass?'PASS':'FAIL'} rho=${r.rhoRelativeL2} X=${r.rhoThetaRelativeL2} maxDu=${r.maxDeltaU} maxDw=${r.maxDeltaW} massCPU=${r.cpuMassDrift} massGPU=${r.gpuMassDrift} cpuMaxW=${r.cpuMaxW} gpuMaxW=${r.gpuMaxW} restRho=${r.restRhoRelativeL2} restX=${r.restRhoThetaRelativeL2} restMaxU=${r.restGpuMaxU} restMaxW=${r.restGpuMaxW} restMass=${r.restGpuMassDrift}\n`;
    if(!r.pass){
      set('overall','單步失敗；執行 RK stage 診斷 / One-step FAIL; running RK-stage diagnostic','bad');set('mStatus','未執行 / NOT RUN');notRun();
      try{const ds=await runStage4Rk3GpuStageDiagnostic();for(const d of ds)log.textContent+=`[Stage ${d.stage} @ ${d.targetFraction}] rho=${d.rhoRelativeL2} X=${d.rhoThetaRelativeL2} maxDu=${d.maxDeltaU} maxDw=${d.maxDeltaW}\n`;set('overall','停止：完整單步 RK3 失敗；stage 診斷已附於 Log / STOP: one-step RK3 FAIL; stage diagnostic appended','bad');}catch(de){log.textContent+=`[Stage diagnostic ERROR] ${String(de)}\n`;set('overall','停止：完整單步 RK3 失敗；stage 診斷亦錯誤 / STOP: one-step and stage diagnostic failed','bad');}
      return;
    }

    set('overall','單步通過；執行 40-step batched agreement / One-step PASS; running 40-step batched agreement','ok');
    const m=await runStage4Rk3GpuMultistepAgreement();
    set('mStatus',m.pass?'通過 / PASS':'失敗 / FAIL',m.pass?'ok':'bad');set('mRho',exp(m.rhoRelativeL2));set('mX',exp(m.rhoThetaRelativeL2));set('mU',`${exp(m.maxDeltaU)} m/s`);set('mW',`${exp(m.maxDeltaW)} m/s`);set('mMass',`CPU ${exp(m.cpuMassDrift)} / GPU ${exp(m.gpuMassDrift)}`);set('mCpuW',`${exp(m.cpuMaxW)} m/s`);set('mGpuW',`${exp(m.gpuMaxW)} m/s`);set('mElapsed',`${(m.elapsedMs/1000).toFixed(2)} s (${m.steps}×${m.dt}s = ${m.simulatedSeconds}s; one GPU batch)`);
    log.textContent+=`[40-step batched RK3] ${m.pass?'PASS':'FAIL'} rho=${m.rhoRelativeL2} X=${m.rhoThetaRelativeL2} maxDu=${m.maxDeltaU} maxDw=${m.maxDeltaW} massCPU=${m.cpuMassDrift} massGPU=${m.gpuMassDrift} cpuMaxW=${m.cpuMaxW} gpuMaxW=${m.gpuMaxW}\n`;
    if(!m.pass){set('overall','前三層通過；40-step batched agreement 失敗 / First three PASS; batched agreement FAIL','bad');notRun();return;}

    set('overall','短期 RK3 全通過；執行 production-scale 2-day gate / Short RK3 gates PASS; running 2-day production gate','ok');set('cStatus','執行中 / Running');
    const c=await runStage4Rk3TwoDayClimate(sample=>{set('cDay',sample.day.toFixed(2));set('cMass',exp(sample.massDrift));set('cW',`${sample.maxW.toFixed(4)} m/s`);set('cBelow',`${sample.maxWBelowSponge.toFixed(4)} m/s`);set('cAbsorber',`${sample.maxWInSponge.toFixed(4)} m/s`);set('cU',`${sample.maxEdgeWind.toFixed(3)} m/s`);set('cDiv',`${exp(sample.divergenceRms)} s⁻¹`);set('cCfl',`${exp(sample.maxHorizontalCfl)} / ${exp(sample.maxVerticalCfl)}`);log.textContent+=`[RK3 climate day ${sample.day.toFixed(2)}] mass=${sample.massDrift} maxW=${sample.maxW} below=${sample.maxWBelowSponge} absorber=${sample.maxWInSponge} z=${(sample.maxWAltitude/1000).toFixed(2)}km lat=${sample.maxWLatitude.toFixed(1)} maxU=${sample.maxEdgeWind} div=${sample.divergenceRms} CFLh=${sample.maxHorizontalCfl} CFLv=${sample.maxVerticalCfl}\n`;});
    set('cStatus',c.passed?'通過 / PASS':'失敗 / FAIL',c.passed?'ok':'bad');set('cElapsed',`${(c.elapsedMs/1000).toFixed(2)} s`);if(c.failures.length)log.textContent+=`[RK3 2-day failures]\n${c.failures.join('\n')}\n`;
    set('overall',c.passed?'RK3 batch + production-scale 2-day gate 全部通過 / RK3 BATCH + 2-DAY PRODUCTION GATE PASS':'短 gate 通過；2-day production gate 失敗 / Short gates PASS; 2-day production FAIL',c.passed?'ok':'bad');
  }catch(e){set('overall','WebGPU 錯誤 / WebGPU ERROR','bad');log.textContent+=`${String(e)}\n`;}finally{run.disabled=false;}
})();
