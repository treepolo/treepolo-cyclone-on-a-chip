import { buildCubedSphere } from './grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from './grid/vertical.js';
import { addHeldSuarezWavePerturbation, buildHeldSuarezReference } from './physics/heldSuarez.js';
import { createHydrostaticState } from './solver/state.js';
import { GpuStage4Rk3SplitReference } from './gpu/stage4Rk3SplitGpu.js';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const runBtn=$<HTMLButtonElement>('run');
const logEl=$<HTMLPreElement>('log');
const KEY='cyclone-stage4-runtime-diag-v3';
const DT=10,BATCH=40,TARGET_DAYS=3,WAIT_TIMEOUT_MS=15000;
const opts={heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true} as const;

interface BatchTiming{batch:number;day:number;encodeMs:number;waitMs:number;wallMs:number;}
interface Trace{
  phase:string;batch:number;completed:number;day:number;encodeMs:number;waitMs:number;wallMs:number;
  deviceLost?:string;adapterAfterLoss?:string;uncaptured?:string;history:BatchTiming[];
}
let trace:Trace={phase:'idle',batch:0,completed:0,day:0,encodeMs:0,waitMs:0,wallMs:0,history:[]};
function persist(){sessionStorage.setItem(KEY,JSON.stringify(trace));}
function render(){
  $('phase').textContent=trace.phase;$('batch').textContent=String(trace.batch);$('day').textContent=trace.day.toFixed(4);$('encode').textContent=`${trace.encodeMs.toFixed(2)} ms`;$('wait').textContent=`${trace.waitMs.toFixed(2)} ms`;$('wall').textContent=`${(trace.wallMs/1000).toFixed(2)} s`;$('lost').textContent=trace.deviceLost??'—';$('error').textContent=trace.uncaptured??'—';
  logEl.textContent=JSON.stringify(trace,null,2);
}
function mark(p:Partial<Trace>){trace={...trace,...p};persist();render();}
function addTiming(t:BatchTiming){mark({history:[...trace.history.slice(-23),t]});}
const prev=sessionStorage.getItem(KEY);if(prev){try{trace=JSON.parse(prev) as Trace;}catch{}render();}

async function waitForGpu(queue:any,timeoutMs:number):Promise<'done'|'timeout'>{
  let timer:number|undefined;
  try{
    return await Promise.race([
      queue.onSubmittedWorkDone().then(()=> 'done' as const),
      new Promise<'timeout'>(resolve=>{timer=window.setTimeout(()=>resolve('timeout'),timeoutMs);}),
    ]);
  }finally{if(timer!==undefined)window.clearTimeout(timer);}
}

async function yieldToBrowser():Promise<void>{
  const scheduler=(globalThis as any).scheduler;
  if(typeof scheduler?.yield==='function'){
    await scheduler.yield();
    return;
  }
  await new Promise<void>(resolve=>setTimeout(resolve,0));
}

runBtn.onclick=()=>void(async()=>{
  runBtn.disabled=true;sessionStorage.removeItem(KEY);
  trace={phase:'initializing',batch:0,completed:0,day:0,encodeMs:0,waitMs:0,wallMs:0,history:[]};persist();render();
  let gpu:GpuStage4Rk3SplitReference|undefined;
  let intentionalTeardown=false,unexpectedLoss=false;
  const t0=performance.now();
  try{
    const h=buildCubedSphere(8),v=buildStretchedVerticalGrid(48,40000,1.4),ref=buildHeldSuarezReference(v),seed=createHydrostaticState(h,v,ref);addHeldSuarezWavePerturbation(h,v,ref,seed,.05);
    gpu=await GpuStage4Rk3SplitReference.create(h,v,ref,seed,4);
    gpu.device.lost.then((info:any)=>void(async()=>{
      if(intentionalTeardown)return;
      unexpectedLoss=true;
      const lost=`reason=${String(info.reason)} message=${String(info.message)}`;
      mark({deviceLost:lost,phase:'DEVICE LOST',wallMs:performance.now()-t0});
      try{
        const adapter=await (navigator as any).gpu?.requestAdapter?.();
        mark({adapterAfterLoss:adapter?'GPUAdapter available':'null / no adapter',wallMs:performance.now()-t0});
      }catch(e){mark({adapterAfterLoss:`requestAdapter error: ${String(e)}`,wallMs:performance.now()-t0});}
    })());
    gpu.device.addEventListener?.('uncapturederror',(ev:any)=>{mark({uncaptured:String(ev?.error?.message??ev?.error??ev),wallMs:performance.now()-t0});});

    const total=Math.round(TARGET_DAYS*86400/DT);let completed=0,batch=0;
    while(completed<total&&!unexpectedLoss){
      const n=Math.min(BATCH,total-completed);batch++;
      mark({phase:'before encode',batch,completed,day:completed*DT/86400,wallMs:performance.now()-t0});
      const te=performance.now();gpu.stepBatch(DT,n,opts);const encodeMs=performance.now()-te;
      mark({phase:'submitted / waiting GPU',batch,completed,day:completed*DT/86400,encodeMs,wallMs:performance.now()-t0});
      const tw=performance.now();const waitResult=await waitForGpu(gpu.device.queue,WAIT_TIMEOUT_MS);const waitMs=performance.now()-tw;
      if(unexpectedLoss)break;
      if(waitResult==='timeout'){
        mark({phase:'GPU WAIT TIMEOUT',batch,completed,day:completed*DT/86400,encodeMs,waitMs,wallMs:performance.now()-t0});
        return;
      }
      completed+=n;
      const day=completed*DT/86400;
      mark({phase:'GPU complete',batch,completed,day,encodeMs,waitMs,wallMs:performance.now()-t0});
      addTiming({batch,day,encodeMs,waitMs,wallMs:performance.now()-t0});
      await yieldToBrowser();
    }
    if(unexpectedLoss)return;
    mark({phase:'COMPLETE',completed,day:completed*DT/86400,wallMs:performance.now()-t0});
  }catch(e){
    if(!unexpectedLoss)mark({phase:'ERROR',uncaptured:String(e),wallMs:performance.now()-t0});
  }finally{
    intentionalTeardown=true;
    gpu?.destroy();
    runBtn.disabled=false;
  }
})();
