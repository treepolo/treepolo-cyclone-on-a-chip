import { runStage4TimestepSensitivity, Stage4TimestepSensitivitySample } from './validation/stage4TimestepSensitivity.js';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const runBtn=$<HTMLButtonElement>('run');
const logEl=$<HTMLDivElement>('log');
const statusEl=$<HTMLSpanElement>('status');
const elapsedEl=$<HTMLSpanElement>('elapsed');
const body=$<HTMLTableSectionElement>('results');

function log(x:string){logEl.textContent=`${x}\n${logEl.textContent||''}`.slice(0,40000)}
function key(dt:number,day:number){return `${dt}-${day.toFixed(2)}`}
const rows=new Map<string,HTMLTableRowElement>();

function sample(s:Stage4TimestepSensitivitySample){
  if(s.day<1.5)return;
  const tr=document.createElement('tr');
  const vals=[
    `${s.dt.toFixed(s.dt%1===0?0:1)} s`,
    s.day.toFixed(2),
    s.maxW.toFixed(3),
    s.maxWBelowAbsorber.toFixed(3),
    s.maxWInAbsorber.toFixed(3),
    `${(s.maxWAltitude/1000).toFixed(2)} km, ${s.maxWLatitude.toFixed(1)}°`,
    s.maxEdgeWind.toFixed(3),
    s.massDrift.toExponential(3),
    s.maxVerticalCfl.toExponential(3),
    s.divergenceRms.toExponential(3),
  ];
  for(const v of vals){const td=document.createElement('td');td.textContent=v;tr.appendChild(td)}
  const k=key(s.dt,s.day),old=rows.get(k);if(old)old.replaceWith(tr);else body.appendChild(tr);rows.set(k,tr);
  log(`dt=${s.dt}s day=${s.day.toFixed(2)} max|w|=${s.maxW.toFixed(4)} below=${s.maxWBelowAbsorber.toFixed(4)} absorber=${s.maxWInAbsorber.toFixed(4)} z=${(s.maxWAltitude/1000).toFixed(2)}km lat=${s.maxWLatitude.toFixed(1)} max|u|=${s.maxEdgeWind.toFixed(3)} vCFL=${s.maxVerticalCfl.toExponential(3)} div=${s.divergenceRms.toExponential(3)} mass=${s.massDrift.toExponential(3)}`);
}

runBtn.onclick=()=>void(async()=>{
  runBtn.disabled=true;body.textContent='';rows.clear();logEl.textContent='';statusEl.textContent='執行中 / Running';statusEl.className='';elapsedEl.textContent='—';
  try{
    const r=await runStage4TimestepSensitivity(sample);
    elapsedEl.textContent=`${(r.elapsedMs/1000).toFixed(2)} s`;
    const finals=r.runs.map(x=>x.samples[x.samples.length-1]).filter(Boolean) as Stage4TimestepSensitivitySample[];
    const f10=finals.find(x=>x.dt===10),f5=finals.find(x=>x.dt===5),f25=finals.find(x=>x.dt===2.5);
    if(f10&&f5&&f25){
      const r5=f5.maxWInAbsorber/Math.max(f10.maxWInAbsorber,1e-12),r25=f25.maxWInAbsorber/Math.max(f10.maxWInAbsorber,1e-12);
      $('ratio5').textContent=r5.toFixed(3);$('ratio25').textContent=r25.toFixed(3);
      if(r25<.5){$('verdict').textContent='強 timestep 敏感 / Strong timestep sensitivity';$('verdict').className='ok'}
      else if(r25>.8){$('verdict').textContent='弱 timestep 敏感 / Weak timestep sensitivity';$('verdict').className='bad'}
      else{$('verdict').textContent='中等 timestep 敏感 / Moderate timestep sensitivity';$('verdict').className=''}
    }
    statusEl.textContent='完成 / COMPLETE';statusEl.className='ok';
  }catch(e){statusEl.textContent='失敗 / FAIL';statusEl.className='bad';log(`Sensitivity run failed: ${String(e)}`)}finally{runBtn.disabled=false}
})();
