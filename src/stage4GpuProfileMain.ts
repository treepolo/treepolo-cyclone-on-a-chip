import { buildCubedSphere } from './grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from './grid/vertical.js';
import { addHeldSuarezWavePerturbation, buildHeldSuarezReference } from './physics/heldSuarez.js';
import { buildModelTopSpongeRates } from './physics/modelTopSponge.js';
import { GpuDryCorePrototype } from './gpu/dryCoreGpu.js';
import { GpuRotatingDryCore } from './gpu/rotatingDryCoreGpu.js';
import { GpuStage4Rk3SplitReference } from './gpu/stage4Rk3SplitGpu.js';
import { buildRk3SplitSchedule } from './solver/rk3SplitSchedule.js';
import { STAGE4_HEVI_OFFCENTERING } from './solver/stage4Config.js';
import { createHydrostaticState } from './solver/state.js';

type GPUAny=any;
type ProfileRecord={
  outer:number;
  stage:number|null;
  acousticStep:number|null;
  category:string;
  kernel:string;
  begin:number;
  end:number;
};

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const runBtn=$<HTMLButtonElement>('run');
const logEl=$<HTMLPreElement>('log');
const DT=10;
const WARMUP_BATCH=40;
const PROFILE_OUTER_STEPS=10;
const MAX_QUERIES=4096;
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

function mean(xs:number[]):number{return xs.reduce((a,b)=>a+b,0)/Math.max(xs.length,1);}
function median(xs:number[]):number{const s=[...xs].sort((a,b)=>a-b);const n=s.length;return n===0?0:n%2?s[(n-1)/2]!:0.5*(s[n/2-1]!+s[n/2]!);}
function pct(x:number,total:number):number{return total>0?100*x/total:0;}

async function createTimestampGpu():Promise<GpuStage4Rk3SplitReference>{
  const nav=(globalThis as any).navigator;
  if(!nav?.gpu)throw new Error('WebGPU unavailable');
  const adapter=await nav.gpu.requestAdapter();
  if(!adapter)throw new Error('No WebGPU adapter');
  if(!adapter.features?.has?.('timestamp-query'))throw new Error('This WebGPU adapter does not expose timestamp-query');
  const requiredStorageBuffersPerStage=8;
  const supported=Number(adapter.limits?.maxStorageBuffersPerShaderStage??0);
  if(supported<requiredStorageBuffersPerStage)throw new Error(`WebGPU adapter exposes maxStorageBuffersPerShaderStage=${supported}; Stage 4 requires ${requiredStorageBuffersPerStage}.`);
  const device=await adapter.requestDevice({
    requiredLimits:{maxStorageBuffersPerShaderStage:requiredStorageBuffersPerStage},
    requiredFeatures:['timestamp-query'],
  });
  if(!device.features?.has?.('timestamp-query')){
    device.destroy?.();
    throw new Error('timestamp-query was supported by the adapter but was not enabled on the GPUDevice');
  }

  const h=buildCubedSphere(8),v=buildStretchedVerticalGrid(48,40000,1.4),ref=buildHeldSuarezReference(v),seed=createHydrostaticState(h,v,ref);
  addHeldSuarezWavePerturbation(h,v,ref,seed,.05);
  device.pushErrorScope?.('validation');
  let dry:GpuDryCorePrototype|undefined;
  let rotating:GpuRotatingDryCore|undefined;
  let gpu:GpuStage4Rk3SplitReference|undefined;
  try{
    dry=new (GpuDryCorePrototype as any)(device,adapter,h,v,ref,seed,STAGE4_HEVI_OFFCENTERING,buildModelTopSpongeRates(v)) as GpuDryCorePrototype;
    rotating=new (GpuRotatingDryCore as any)(dry,h,v,ref) as GpuRotatingDryCore;
    gpu=new (GpuStage4Rk3SplitReference as any)(rotating,4) as GpuStage4Rk3SplitReference;
    const err=await device.popErrorScope?.();
    if(err)throw new Error(`Stage 4 timestamp profiler WebGPU validation: ${err.message||err}`);
    return gpu;
  }catch(e){
    try{await device.popErrorScope?.();}catch{}
    try{gpu?.destroy();}catch{}
    if(!gpu){try{rotating?.destroy();}catch{};if(!rotating){try{dry?.destroy();}catch{};if(!dry)device.destroy?.();}}
    throw e;
  }
}

function aggregate(records:ProfileRecord[],values:BigUint64Array){
  const samples=records.map(r=>{
    const ns=Number(values[r.end]!-values[r.begin]!);
    return {...r,ms:ns/1e6};
  });
  const totalMs=samples.reduce((a,b)=>a+b.ms,0);
  const kernelMap=new Map<string,number[]>();
  const categoryMap=new Map<string,number[]>();
  for(const s of samples){
    const ks=kernelMap.get(s.kernel)??[];ks.push(s.ms);kernelMap.set(s.kernel,ks);
    const cs=categoryMap.get(s.category)??[];cs.push(s.ms);categoryMap.set(s.category,cs);
  }
  const kernels=[...kernelMap.entries()].map(([kernel,xs])=>({
    kernel,
    totalMs:xs.reduce((a,b)=>a+b,0),
    msPerOuter:xs.reduce((a,b)=>a+b,0)/PROFILE_OUTER_STEPS,
    dispatchesPerOuter:xs.length/PROFILE_OUTER_STEPS,
    meanDispatchMs:mean(xs),
    medianDispatchMs:median(xs),
    sharePct:pct(xs.reduce((a,b)=>a+b,0),totalMs),
  })).sort((a,b)=>b.totalMs-a.totalMs);
  const categories=[...categoryMap.entries()].map(([category,xs])=>({
    category,
    totalMs:xs.reduce((a,b)=>a+b,0),
    msPerOuter:xs.reduce((a,b)=>a+b,0)/PROFILE_OUTER_STEPS,
    sharePct:pct(xs.reduce((a,b)=>a+b,0),totalMs),
  })).sort((a,b)=>b.totalMs-a.totalMs);
  return{samples,totalMs,kernels,categories};
}

async function profile(gpu:GpuStage4Rk3SplitReference){
  const g=gpu as any;
  const device=gpu.device as GPUAny;
  const U=(globalThis as any).GPUBufferUsage,M=(globalThis as any).GPUMapMode;
  const querySet=device.createQuerySet({label:'Stage4 kernel timestamp queries',type:'timestamp',count:MAX_QUERIES});
  const resolveBuffer=device.createBuffer({label:'Stage4 timestamp resolve',size:MAX_QUERIES*8,usage:U.QUERY_RESOLVE|U.COPY_SRC});
  const readBuffer=device.createBuffer({label:'Stage4 timestamp readback',size:MAX_QUERIES*8,usage:U.COPY_DST|U.MAP_READ});
  const records:ProfileRecord[]=[];
  let cursor=0;
  const enc=device.createCommandEncoder({label:`Stage4 detailed timestamp profile ${PROFILE_OUTER_STEPS} outer steps`});
  const h=g.core.h,v=g.core.v,b=g.core.core.buffers,rb=g.buffers;
  const schedule=buildRk3SplitSchedule(g.acousticRatio);
  g.prepareBatch(DT,true);

  const timed=(meta:Omit<ProfileRecord,'begin'|'end'>,fn:(p:GPUAny)=>void)=>{
    if(cursor+2>MAX_QUERIES)throw new Error(`Timestamp query budget exceeded (${MAX_QUERIES})`);
    const begin=cursor++,end=cursor++;
    const p=enc.beginComputePass({label:`profile ${meta.kernel}`,timestampWrites:{querySet,beginningOfPassWriteIndex:begin,endOfPassWriteIndex:end}});
    fn(p);p.end();records.push({...meta,begin,end});
  };
  const rkDispatch=(p:GPUAny,name:string,count:number,stage:number,size=128)=>g.dispatch(p,name,count,stage,size);
  const slowDispatch=(p:GPUAny,name:string,count:number)=>g.slow.dispatch(p,name,count);
  const divDispatch=(p:GPUAny,div:GPUAny,name:'divergence'|'adjust')=>{
    if(name==='divergence'){
      p.setPipeline(div.divergencePipeline);p.setBindGroup(0,div.divergenceGroup);p.dispatchWorkgroups(Math.ceil(h.cellCount*v.nz/128));
    }else{
      p.setPipeline(div.adjustPipeline);p.setBindGroup(0,div.adjustGroup);p.dispatchWorkgroups(Math.ceil(h.edgeCount*v.nz/128));
    }
  };

  for(let outer=0;outer<PROFILE_OUTER_STEPS;outer++){
    timed({outer,stage:null,acousticStep:null,category:'snapshot',kernel:'snapshot.packBase'},p=>rkDispatch(p,'packBase',h.cellCount*v.nz,0));
    enc.copyBufferToBuffer(b.u,0,rb.baseU,0,h.edgeCount*v.nz*4);
    enc.copyBufferToBuffer(b.w,0,rb.baseW,0,h.cellCount*(v.nz+1)*4);

    for(let stageIndex=0;stageIndex<schedule.length;stageIndex++){
      const stage=stageIndex+1,steps=schedule[stageIndex]!.acousticSteps;
      g.slow.clear(enc);
      const slow=(name:string,count:number)=>timed({outer,stage,acousticStep:null,category:'slow',kernel:`slow.${name}`},p=>slowDispatch(p,name,count));
      slow('hpert',h.edgeCount*v.nz);slow('vpert',h.cellCount*(v.nz+1));slow('sdiv',h.cellCount*v.nz);
      slow('cellwind',h.cellCount*v.nz);slow('hadv',h.cellCount*v.nz);slow('vadv',h.cellCount*v.nz);slow('coriolis',h.cellCount*v.nz);slow('project',h.edgeCount*v.nz);slow('wtend',h.cellCount*(v.nz+1));slow('thermal',h.cellCount*v.nz);slow('drag',h.edgeCount*v.nz);

      const prep=(name:string,count:number,size=128)=>timed({outer,stage,acousticStep:null,category:'prep',kernel:`prep.${name}`},p=>rkDispatch(p,name,count,stageIndex,size));
      prep('prepCell',h.cellCount*v.nz);prep('addHref',h.cellCount*v.nz);prep('addVref',h.cellCount*v.nz);prep('prepU',h.edgeCount*v.nz);prep('prepIface',h.cellCount*(v.nz+1));
      enc.copyBufferToBuffer(rb.baseU,0,rb.acousticU,0,h.edgeCount*v.nz*4);

      const div=stageIndex===0?g.divergenceStage1:g.divergence;
      for(let n=0;n<steps;n++){
        const acousticStep=n+1;
        const acoustic=(name:string,count:number,size=128)=>timed({outer,stage,acousticStep,category:'acoustic',kernel:`acoustic.${name}`},p=>rkDispatch(p,name,count,stageIndex,size));
        acoustic('hvel',h.edgeCount*v.nz);acoustic('hrefFlux',h.edgeCount*v.nz);acoustic('hrefDiv',h.cellCount*v.nz);acoustic('vertical',h.cellCount,1);
        timed({outer,stage,acousticStep,category:'damping',kernel:'damping.divergence'},p=>divDispatch(p,div,'divergence'));
        timed({outer,stage,acousticStep,category:'damping',kernel:'damping.adjust'},p=>divDispatch(p,div,'adjust'));
      }
      timed({outer,stage,acousticStep:null,category:'unpack',kernel:'unpack.cell'},p=>rkDispatch(p,'unpackCell',h.cellCount*v.nz,stageIndex));
      timed({outer,stage,acousticStep:null,category:'unpack',kernel:'unpack.w'},p=>rkDispatch(p,'unpackW',h.cellCount*(v.nz+1),stageIndex));
      enc.copyBufferToBuffer(rb.acousticU,0,b.u,0,h.edgeCount*v.nz*4);
    }
  }

  const queryCount=cursor,queryBytes=queryCount*8;
  enc.resolveQuerySet(querySet,0,queryCount,resolveBuffer,0);
  enc.copyBufferToBuffer(resolveBuffer,0,readBuffer,0,queryBytes);
  const submitStart=performance.now();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  const gpuWaitMs=performance.now()-submitStart;
  await readBuffer.mapAsync(M.READ);
  const values=new BigUint64Array(readBuffer.getMappedRange().slice(0,queryBytes));
  readBuffer.unmap();
  const agg=aggregate(records,values);
  querySet.destroy?.();resolveBuffer.destroy?.();readBuffer.destroy?.();
  return{
    queryCount,
    dispatches:records.length,
    profileGpuWaitMs:gpuWaitMs,
    measuredComputeMs:agg.totalMs,
    measuredComputeMsPerOuter:agg.totalMs/PROFILE_OUTER_STEPS,
    measuredVsProfileWaitPct:pct(agg.totalMs,gpuWaitMs),
    categories:agg.categories,
    kernels:agg.kernels,
    samples:agg.samples,
  };
}

function renderTable(rows:Array<{kernel:string;msPerOuter:number;dispatchesPerOuter:number;meanDispatchMs:number;sharePct:number}>):void{
  const body=$<HTMLTableSectionElement>('kernelRows');body.textContent='';
  for(const r of rows){
    const tr=document.createElement('tr');
    for(const x of [r.kernel,r.msPerOuter.toFixed(4),r.dispatchesPerOuter.toFixed(1),r.meanDispatchMs.toFixed(4),`${r.sharePct.toFixed(1)}%`]){
      const td=document.createElement('td');td.textContent=x;tr.appendChild(td);
    }
    body.appendChild(tr);
  }
}

runBtn.onclick=()=>void(async()=>{
  runBtn.disabled=true;$('status').textContent='INITIALIZING';logEl.textContent='Requesting a timestamp-query GPUDevice and building the production Stage 4 pipelines…';
  let gpu:GpuStage4Rk3SplitReference|undefined;let intentionalTeardown=false;let unexpectedLoss:string|undefined;
  try{
    gpu=await createTimestampGpu();
    gpu.device.lost.then((info:any)=>{if(!intentionalTeardown)unexpectedLoss=`reason=${String(info.reason)} message=${String(info.message)}`;});
    $('feature').textContent='enabled';
    $('status').textContent='WARMUP';
    gpu.stepBatch(DT,WARMUP_BATCH,opts);await gpu.device.queue.onSubmittedWorkDone();
    if(unexpectedLoss)throw new Error(`GPU device lost during warmup: ${unexpectedLoss}`);
    const yieldMode=await yieldToBrowser();
    $('status').textContent='PROFILING';
    const result=await profile(gpu);
    if(unexpectedLoss)throw new Error(`GPU device lost during profiling: ${unexpectedLoss}`);
    renderTable(result.kernels);
    $('status').textContent='COMPLETE';
    $('profileWait').textContent=`${result.profileGpuWaitMs.toFixed(2)} ms / ${PROFILE_OUTER_STEPS} steps`;
    $('compute').textContent=`${result.measuredComputeMsPerOuter.toFixed(3)} ms / outer step`;
    $('topKernel').textContent=result.kernels[0]?`${result.kernels[0].kernel} — ${result.kernels[0].sharePct.toFixed(1)}%`:'—';
    $('yield').textContent=yieldMode;
    logEl.textContent=JSON.stringify({
      config:{grid:'N=8 × 48',cells:18432,dtSeconds:DT,warmupOuterSteps:WARMUP_BATCH,profileOuterSteps:PROFILE_OUTER_STEPS,timestampFeature:'timestamp-query',yieldMode},
      summary:{queryCount:result.queryCount,dispatches:result.dispatches,profileGpuWaitMs:result.profileGpuWaitMs,measuredComputeMs:result.measuredComputeMs,measuredComputeMsPerOuter:result.measuredComputeMsPerOuter,measuredVsProfileWaitPct:result.measuredVsProfileWaitPct},
      categories:result.categories,
      kernels:result.kernels,
      deviceLost:unexpectedLoss??null,
    },null,2);
  }catch(e){$('status').textContent='ERROR';logEl.textContent=String(e);}
  finally{intentionalTeardown=true;gpu?.destroy();runBtn.disabled=false;}
})();
