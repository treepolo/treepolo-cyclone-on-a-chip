import { buildCubedSphere } from './grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from './grid/vertical.js';
import { addHeldSuarezWavePerturbation, buildHeldSuarezReference } from './physics/heldSuarez.js';
import { createHydrostaticState } from './solver/state.js';
import { GpuStage4Rk3SplitReference } from './gpu/stage4Rk3SplitGpu.js';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const runBtn=$<HTMLButtonElement>('run');
const logEl=$<HTMLPreElement>('log');
const DT=10;
const BATCH=40;
const WARMUP_BATCHES=2;
const MEASURE_BATCHES=10;
const opts={heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true} as const;

async function yieldToBrowser():Promise<'scheduler.yield'|'setTimeout(0)'>{
  const scheduler=(globalThis as any).scheduler;
  if(typeof scheduler?.yield==='function'){
    await scheduler.yield();
    return 'scheduler.yield';
  }
  await new Promise<void>(resolve=>setTimeout(resolve,0));
  return 'setTimeout(0)';
}

function mean(xs:number[]):number{return xs.reduce((a,b)=>a+b,0)/Math.max(1,xs.length);}
function median(xs:number[]):number{const s=[...xs].sort((a,b)=>a-b);const n=s.length;return n%2?s[(n-1)/2]!:0.5*(s[n/2-1]!+s[n/2]!);}

runBtn.onclick=()=>void(async()=>{
  runBtn.disabled=true;
  logEl.textContent='Initializing production N=8×48 benchmark…';
  let gpu:GpuStage4Rk3SplitReference|undefined;
  let intentionalTeardown=false;
  let unexpectedLoss:string|undefined;
  try{
    const h=buildCubedSphere(8),v=buildStretchedVerticalGrid(48,40000,1.4),ref=buildHeldSuarezReference(v),seed=createHydrostaticState(h,v,ref);addHeldSuarezWavePerturbation(h,v,ref,seed,.05);
    gpu=await GpuStage4Rk3SplitReference.create(h,v,ref,seed,4);
    gpu.device.lost.then((info:any)=>{if(!intentionalTeardown)unexpectedLoss=`reason=${String(info.reason)} message=${String(info.message)}`;});

    let yieldMode:'scheduler.yield'|'setTimeout(0)'='setTimeout(0)';
    for(let i=0;i<WARMUP_BATCHES;i++){
      gpu.stepBatch(DT,BATCH,opts);
      await gpu.device.queue.onSubmittedWorkDone();
      if(unexpectedLoss)throw new Error(`GPU device lost during warmup: ${unexpectedLoss}`);
      yieldMode=await yieldToBrowser();
    }

    const encodeMs:number[]=[],waitMs:number[]=[],wallMs:number[]=[];
    for(let i=0;i<MEASURE_BATCHES;i++){
      const t0=performance.now();
      const te=performance.now();gpu.stepBatch(DT,BATCH,opts);encodeMs.push(performance.now()-te);
      const tw=performance.now();await gpu.device.queue.onSubmittedWorkDone();waitMs.push(performance.now()-tw);
      if(unexpectedLoss)throw new Error(`GPU device lost during measured batch ${i+1}: ${unexpectedLoss}`);
      yieldMode=await yieldToBrowser();
      wallMs.push(performance.now()-t0);
      logEl.textContent=`Measured ${i+1}/${MEASURE_BATCHES} batches…`;
    }

    const meanWall=mean(wallMs),stepsPerSecond=BATCH/(meanWall/1000),simDaysPerWallMinute=stepsPerSecond*DT*60/86400,estimated30DayMinutes=30/simDaysPerWallMinute;
    const result={
      config:{grid:'N=8 × 48',cells:18432,dtSeconds:DT,batch:BATCH,warmupBatches:WARMUP_BATCHES,measuredBatches:MEASURE_BATCHES,yieldMode},
      timingMs:{encodeMean:mean(encodeMs),encodeMedian:median(encodeMs),gpuWaitMean:mean(waitMs),gpuWaitMedian:median(waitMs),batchWallMean:meanWall,batchWallMedian:median(wallMs)},
      throughput:{outerStepsPerSecond:stepsPerSecond,simulatedDaysPerWallMinute:simDaysPerWallMinute,estimated30DayMinutes},
      batches:wallMs.map((wall,i)=>({batch:i+1,encodeMs:encodeMs[i],gpuWaitMs:waitMs[i],wallMs:wall})),
      deviceLost:unexpectedLoss??null,
    };
    $('status').textContent='COMPLETE';
    $('wait').textContent=`${result.timingMs.gpuWaitMean.toFixed(2)} ms`;
    $('throughput').textContent=`${result.throughput.outerStepsPerSecond.toFixed(2)} steps/s`;
    $('estimate').textContent=`${result.throughput.estimated30DayMinutes.toFixed(2)} min`;
    $('yield').textContent=yieldMode;
    logEl.textContent=JSON.stringify(result,null,2);
  }catch(e){
    $('status').textContent='ERROR';
    logEl.textContent=String(e);
  }finally{
    intentionalTeardown=true;
    gpu?.destroy();
    runBtn.disabled=false;
  }
})();
