import { runStage4AcousticGpuAgreement } from './validation/stage4AcousticGpuAgreement.js';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const run=$<HTMLButtonElement>('run'),status=$<HTMLSpanElement>('status'),logEl=$<HTMLPreElement>('log');
const set=(id:string,x:string)=>{$<HTMLElement>(id).textContent=x};
run.onclick=()=>void(async()=>{
  run.disabled=true;status.textContent='執行中 / Running';status.className='';logEl.textContent='';
  for(const id of ['rho','x','frefRho','frefX','w','cpuW','gpuW','restW','restRho','elapsed'])set(id,'—');
  try{
    const r=await runStage4AcousticGpuAgreement();
    set('rho',r.rhoRelativeL2.toExponential(3));set('x',r.rhoThetaRelativeL2.toExponential(3));set('frefRho',r.referenceMassFluxRelativeL2.toExponential(3));set('frefX',r.referenceRhoThetaFluxRelativeL2.toExponential(3));set('w',`${r.maxDeltaW.toExponential(3)} m/s`);set('cpuW',`${r.cpuMaxW.toExponential(3)} m/s`);set('gpuW',`${r.gpuMaxW.toExponential(3)} m/s`);set('restW',`${r.hydrostaticMaxW.toExponential(3)} m/s`);set('restRho',r.hydrostaticRhoRelativeL2.toExponential(3));set('elapsed',`${(r.elapsedMs/1000).toFixed(2)} s`);
    status.textContent=r.pass?'通過 / PASS':'失敗 / FAIL';status.className=r.pass?'ok':'bad';
    logEl.textContent=`GPU/CPU acoustic-column agreement ${r.pass?'PASS':'FAIL'}\nrhoL2=${r.rhoRelativeL2}\nrhoThetaL2=${r.rhoThetaRelativeL2}\nreferenceMassFluxL2=${r.referenceMassFluxRelativeL2}\nreferenceRhoThetaFluxL2=${r.referenceRhoThetaFluxRelativeL2}\nmaxDw=${r.maxDeltaW}\ncpuMaxW=${r.cpuMaxW}\ngpuMaxW=${r.gpuMaxW}\nhydrostaticMaxW=${r.hydrostaticMaxW}\nhydrostaticRhoL2=${r.hydrostaticRhoRelativeL2}`;
  }catch(e){status.textContent='錯誤 / ERROR';status.className='bad';logEl.textContent=String(e);}finally{run.disabled=false;}
})();
