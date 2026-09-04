import { buildCubedSphere } from './grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from './grid/vertical.js';
import { addHeldSuarezWavePerturbation, buildHeldSuarezReference } from './physics/heldSuarez.js';
import { createHydrostaticState } from './solver/state.js';
import { GpuStage4Rk3SplitReference } from './gpu/stage4Rk3SplitGpu.js';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const runBtn=$<HTMLButtonElement>('run');
const logEl=$<HTMLPreElement>('log');
const KEY='cyclone-stage4-runtime-diag-v1';
const DT=10,BATCH=40,TARGET_DAYS=.5;
const opts={heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true} as const;
interface Trace{phase:string;batch:number;completed:number;day:number;encodeMs:number;waitMs:number;wallMs:number;deviceLost?:string;uncaptured?:string;}
let trace:Trace={phase:'idle',batch:0,completed:0,day:0,encodeMs:0,waitMs:0,wallMs:0};
function persist(){sessionStorage.setItem(KEY,JSON.stringify(trace));}
function render(){
  $('phase').textContent=trace.phase;$('batch').textContent=String(trace.batch);$('day').textContent=trace.day.toFixed(4);$('encode').textContent=`${trace.encodeMs.toFixed(2)} ms`;$('wait').textContent=`${trace.waitMs.toFixed(2)} ms`;$('wall').textContent=`${(trace.wallMs/1000).toFixed(2)} s`;$('lost').textContent=trace.deviceLost??'—';$('error').textContent=trace.uncaptured??'—';
  logEl.textContent=JSON.stringify(trace,null,2);
}
function mark(p:Partial<Trace>){trace={...trace,...p};persist();render();}
const prev=sessionStorage.getItem(KEY);if(prev){try{trace=JSON.parse(prev) as Trace;}catch{}render();}
runBtn.onclick=()=>void(async()=>{
  runBtn.disabled=true;sessionStorage.removeItem(KEY);trace={phase:'initializing',batch:0,completed:0,day:0,encodeMs:0,waitMs:0,wallMs:0};persist();render();
  let gpu:GpuStage4Rk3SplitReference|undefined;const t0=performance.now();
  try{
    const h=buildCubedSphere(8),v=buildStretchedVerticalGrid(48,40000,1.4),ref=buildHeldSuarezReference(v),seed=createHydrostaticState(h,v,ref);addHeldSuarezWavePerturbation(h,v,ref,seed,.05);
    gpu=await GpuStage4Rk3SplitReference.create(h,v,ref,seed,4);
    gpu.device.lost.then((info:any)=>{mark({deviceLost:`reason=${String(info.reason)} message=${String(info.message)}`,phase:'DEVICE LOST',wallMs:performance.now()-t0});});
    gpu.device.addEventListener?.('uncapturederror',(ev:any)=>{mark({uncaptured:String(ev?.error?.message??ev?.error??ev),wallMs:performance.now()-t0});});
    const total=Math.round(TARGET_DAYS*86400/DT);let completed=0,batch=0;
    while(completed<total){
      const n=Math.min(BATCH,total-completed);batch++;
      mark({phase:'before encode',batch,completed,day:completed*DT/86400,wallMs:performance.now()-t0});
      const te=performance.now();gpu.stepBatch(DT,n,opts);const encodeMs=performance.now()-te;
      mark({phase:'submitted / waiting GPU',batch,completed,day:completed*DT/86400,encodeMs,wallMs:performance.now()-t0});
      const tw=performance.now();await gpu.device.queue.onSubmittedWorkDone();const waitMs=performance.now()-tw;
      completed+=n;
      mark({phase:'GPU complete',batch,completed,day:completed*DT/86400,encodeMs,waitMs,wallMs:performance.now()-t0});
    }
    mark({phase:'COMPLETE',completed,day:completed*DT/86400,wallMs:performance.now()-t0});
  }catch(e){mark({phase:'ERROR',uncaptured:String(e),wallMs:performance.now()-t0});}
  finally{gpu?.destroy();runBtn.disabled=false;}
})();
