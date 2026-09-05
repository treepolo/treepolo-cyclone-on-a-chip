import type { DryState } from '../solver/state.js';
import type { ClimateDaySample } from '../validation/stage4Gpu.js';

const DB_NAME='cyclone-on-a-chip';
const DB_VERSION=1;
const STORE_NAME='stage4-rk3-checkpoints';
export const STAGE4_RK3_PRODUCTION_CHECKPOINT_KEY='stage4-rk3-production';
export const STAGE4_RK3_CHECKPOINT_TEST_KEY='stage4-rk3-checkpoint-self-test';
export const STAGE4_RK3_CHECKPOINT_SCHEMA_VERSION=1;
// Bump this whenever a numerical change makes a prognostic state unsafe to
// resume under the current Stage 4 production integrator.
export const STAGE4_RK3_PRODUCTION_MODEL_SIGNATURE='stage4-rk3-split-v1|N8|NZ48|H40000|stretch1.4|dt10|acoustic4|offcenter0.10|held-suarez-v1|material-momentum-muscl-bj-v2|vertical-mass-momentum-v1|fref-fpert-3d-v1|rayleigh-v1|acoustic-divergence-v1|face-conormal-v1|nonorth-pressure-v1';

export interface Stage4Rk3ClimateCheckpoint{
  schemaVersion:number;
  modelSignature:string;
  savedAt:number;
  targetDays:number;
  completedOuterSteps:number;
  initialDryMass:number;
  state:DryState;
  samples:ClimateDaySample[];
}

function requestResult<T>(request:IDBRequest<T>):Promise<T>{
  return new Promise<T>((resolve,reject)=>{
    request.onsuccess=()=>resolve(request.result);
    request.onerror=()=>reject(request.error??new Error('IndexedDB request failed'));
  });
}
function transactionDone(tx:IDBTransaction):Promise<void>{
  return new Promise<void>((resolve,reject)=>{
    tx.oncomplete=()=>resolve();
    tx.onabort=()=>reject(tx.error??new Error('IndexedDB transaction aborted'));
    tx.onerror=()=>reject(tx.error??new Error('IndexedDB transaction failed'));
  });
}
function openDb():Promise<IDBDatabase>{
  if(typeof indexedDB==='undefined')return Promise.reject(new Error('IndexedDB is unavailable'));
  return new Promise<IDBDatabase>((resolve,reject)=>{
    const request=indexedDB.open(DB_NAME,DB_VERSION);
    request.onupgradeneeded=()=>{
      const db=request.result;
      if(!db.objectStoreNames.contains(STORE_NAME))db.createObjectStore(STORE_NAME);
    };
    request.onsuccess=()=>resolve(request.result);
    request.onerror=()=>reject(request.error??new Error('Failed to open IndexedDB'));
    request.onblocked=()=>reject(new Error('IndexedDB upgrade blocked by another tab'));
  });
}
function cloneState(state:DryState):DryState{
  return{
    rhoD:state.rhoD.slice(),
    rhoThetaM:state.rhoThetaM.slice(),
    uEdge:state.uEdge.slice(),
    wInterface:state.wInterface.slice(),
    time:state.time,
  };
}
function normalizeState(value:DryState):DryState{
  const asF64=(x:Float64Array|ArrayLike<number>,name:string):Float64Array=>{
    if(x instanceof Float64Array)return x.slice();
    if(x&&typeof x.length==='number')return Float64Array.from(x);
    throw new Error(`checkpoint ${name} is not an array`);
  };
  return{
    rhoD:asF64(value.rhoD,'rhoD'),
    rhoThetaM:asF64(value.rhoThetaM,'rhoThetaM'),
    uEdge:asF64(value.uEdge,'uEdge'),
    wInterface:asF64(value.wInterface,'wInterface'),
    time:Number(value.time),
  };
}
export function cloneStage4Rk3Checkpoint(cp:Stage4Rk3ClimateCheckpoint):Stage4Rk3ClimateCheckpoint{
  return{
    schemaVersion:cp.schemaVersion,
    modelSignature:cp.modelSignature,
    savedAt:cp.savedAt,
    targetDays:cp.targetDays,
    completedOuterSteps:cp.completedOuterSteps,
    initialDryMass:cp.initialDryMass,
    state:cloneState(cp.state),
    samples:cp.samples.map(s=>({...s})),
  };
}
export function assertStage4Rk3CheckpointShape(value:unknown):asserts value is Stage4Rk3ClimateCheckpoint{
  if(!value||typeof value!=='object')throw new Error('checkpoint is not an object');
  const cp=value as Partial<Stage4Rk3ClimateCheckpoint>;
  if(cp.schemaVersion!==STAGE4_RK3_CHECKPOINT_SCHEMA_VERSION)throw new Error(`checkpoint schema mismatch: ${String(cp.schemaVersion)}`);
  if(typeof cp.modelSignature!=='string'||!cp.modelSignature)throw new Error('checkpoint model signature missing');
  if(!Number.isFinite(cp.savedAt)||!Number.isFinite(cp.targetDays)||!Number.isFinite(cp.completedOuterSteps)||!Number.isFinite(cp.initialDryMass))throw new Error('checkpoint scalar metadata invalid');
  if(!cp.state||!Array.isArray(cp.samples))throw new Error('checkpoint state/samples missing');
  const s=cp.state as DryState;
  if(!s.rhoD||!s.rhoThetaM||!s.uEdge||!s.wInterface||!Number.isFinite(s.time))throw new Error('checkpoint prognostic state invalid');
}
export function assertStage4Rk3CheckpointCompatible(cp:Stage4Rk3ClimateCheckpoint,modelSignature:string,targetDays:number):void{
  assertStage4Rk3CheckpointShape(cp);
  if(cp.modelSignature!==modelSignature)throw new Error(`checkpoint model mismatch: ${cp.modelSignature}`);
  if(cp.targetDays!==targetDays)throw new Error(`checkpoint target mismatch: ${cp.targetDays} days; expected ${targetDays}`);
}
export async function saveStage4Rk3Checkpoint(cp:Stage4Rk3ClimateCheckpoint,key=STAGE4_RK3_PRODUCTION_CHECKPOINT_KEY):Promise<void>{
  assertStage4Rk3CheckpointShape(cp);
  const db=await openDb();
  try{
    const tx=db.transaction(STORE_NAME,'readwrite');
    tx.objectStore(STORE_NAME).put(cloneStage4Rk3Checkpoint(cp),key);
    await transactionDone(tx);
  }finally{db.close();}
}
export async function loadStage4Rk3Checkpoint(key=STAGE4_RK3_PRODUCTION_CHECKPOINT_KEY):Promise<Stage4Rk3ClimateCheckpoint|null>{
  const db=await openDb();
  try{
    const tx=db.transaction(STORE_NAME,'readonly');
    const raw=await requestResult<unknown>(tx.objectStore(STORE_NAME).get(key));
    await transactionDone(tx);
    if(raw===undefined)return null;
    assertStage4Rk3CheckpointShape(raw);
    const cp=raw as Stage4Rk3ClimateCheckpoint;
    return{
      schemaVersion:cp.schemaVersion,
      modelSignature:cp.modelSignature,
      savedAt:cp.savedAt,
      targetDays:cp.targetDays,
      completedOuterSteps:cp.completedOuterSteps,
      initialDryMass:cp.initialDryMass,
      state:normalizeState(cp.state),
      samples:cp.samples.map(s=>({...s})),
    };
  }finally{db.close();}
}
export async function clearStage4Rk3Checkpoint(key=STAGE4_RK3_PRODUCTION_CHECKPOINT_KEY):Promise<void>{
  const db=await openDb();
  try{
    const tx=db.transaction(STORE_NAME,'readwrite');
    tx.objectStore(STORE_NAME).delete(key);
    await transactionDone(tx);
  }finally{db.close();}
}
export async function stage4Rk3StoragePersistenceStatus():Promise<'persistent'|'best-effort'|'unknown'>{
  try{
    if(!navigator.storage?.persisted)return'unknown';
    if(await navigator.storage.persisted())return'persistent';
    if(navigator.storage.persist&&await navigator.storage.persist())return'persistent';
    return'best-effort';
  }catch{return'unknown';}
}
