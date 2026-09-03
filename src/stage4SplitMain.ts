import { runStage4SplitPrototype, Stage4SplitSample } from './validation/stage4SplitPrototype.js';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const runBtn=$<HTMLButtonElement>('run'),status=$<HTMLSpanElement>('status'),tbody=$<HTMLTableSectionElement>('results'),logEl=$<HTMLDivElement>('log');
const log=(x:string)=>{logEl.textContent=`${x}\n${logEl.textContent||''}`.slice(0,30000)};
function addRow(s:Stage4SplitSample){
  const tr=document.createElement('tr');
  const cells=[`${s.acousticSubsteps} × ${s.acousticDt.toFixed(1)} s`,s.day.toFixed(2),s.maxW.toFixed(3),s.maxWBelowAbsorber.toFixed(3),s.maxWInAbsorber.toFixed(3),`${(s.maxWAltitude/1000).toFixed(2)} km, ${s.maxWLatitude.toFixed(1)}°`,s.maxEdgeWind.toFixed(3),s.massDrift.toExponential(3),s.maxVerticalCflOuter.toExponential(3),s.maxVerticalCflAcoustic.toExponential(3),s.divergenceRms.toExponential(3)];
  for(const x of cells){const td=document.createElement('td');td.textContent=x;tr.appendChild(td)}tbody.appendChild(tr);
  log(`substeps=${s.acousticSubsteps} acousticDt=${s.acousticDt}s day=${s.day.toFixed(2)} max|w|=${s.maxW.toFixed(4)} below=${s.maxWBelowAbsorber.toFixed(4)} absorber=${s.maxWInAbsorber.toFixed(4)} z=${(s.maxWAltitude/1000).toFixed(2)}km lat=${s.maxWLatitude.toFixed(1)} max|u|=${s.maxEdgeWind.toFixed(3)} mass=${s.massDrift.toExponential(3)} CFLouter/acoustic=${s.maxVerticalCflOuter.toExponential(3)}/${s.maxVerticalCflAcoustic.toExponential(3)} div=${s.divergenceRms.toExponential(3)}`);
}
runBtn.onclick=()=>void(async()=>{
  runBtn.disabled=true;tbody.textContent='';logEl.textContent='';status.textContent='執行中 / Running';status.className='';$('verdict').textContent='—';$('elapsed').textContent='—';
  try{
    const r=await runStage4SplitPrototype(addRow),last2=r.runs[0]?.samples.at(-1),last4=r.runs[1]?.samples.at(-1);
    $('elapsed').textContent=`${(r.elapsedMs/1000).toFixed(2)} s`;
    if(!last2||!last4)throw new Error('missing split-prototype samples');
    const stable4=!last4.invalid&&last4.maxWInAbsorber<2&&Math.abs(last4.massDrift)<5e-5;
    const dynamic4=!last4.invalid&&last4.maxWInAbsorber<2;
    const mass4=Math.abs(last4.massDrift)<5e-5;
    $('verdict').textContent=stable4?'4× acoustic split 同時壓住上層 w 與 mass drift / 4× split controls both w and mass drift':dynamic4&&!mass4?'4× split 壓住 fast mode，但 mass drift 仍需修 / fast mode controlled; mass drift remains':'4× split 尚未控制失穩 / split not sufficient';
    $('wRatio').textContent=(last4.maxWInAbsorber/13.639930725097656).toFixed(3);
    $('mass4').textContent=last4.massDrift.toExponential(3);
    status.textContent='完成 / COMPLETE';status.className=stable4?'ok':dynamic4?'warn':'bad';
  }catch(e){status.textContent='錯誤 / ERROR';status.className='bad';log(`split prototype error: ${String(e)}`)}finally{runBtn.disabled=false}
})();
