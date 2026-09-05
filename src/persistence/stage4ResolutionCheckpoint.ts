import {
  STAGE4_RK3_PRODUCTION_MODEL_SIGNATURE,
  assertStage4Rk3CheckpointCompatible,
  clearStage4Rk3Checkpoint,
  loadStage4Rk3Checkpoint,
  saveStage4Rk3Checkpoint,
  type Stage4Rk3ClimateCheckpoint,
} from './stage4Rk3Checkpoint.js';

const PRODUCTION_N_MARKER='|N8|';

export function stage4ResolutionModelSignature(horizontalN:number):string{
  if(!Number.isInteger(horizontalN)||horizontalN<4)throw new Error('horizontalN must be an integer >= 4');
  if(horizontalN===8)return STAGE4_RK3_PRODUCTION_MODEL_SIGNATURE;
  if(!STAGE4_RK3_PRODUCTION_MODEL_SIGNATURE.includes(PRODUCTION_N_MARKER))throw new Error('Stage 4 production model signature is missing the N8 marker');
  return STAGE4_RK3_PRODUCTION_MODEL_SIGNATURE.replace(PRODUCTION_N_MARKER,`|N${horizontalN}|`);
}

export function stage4ResolutionCheckpointKey(horizontalN:number,targetDays:number):string{
  if(!Number.isInteger(horizontalN)||horizontalN<4)throw new Error('horizontalN must be an integer >= 4');
  if(!Number.isInteger(targetDays)||targetDays<1)throw new Error('targetDays must be a positive integer');
  return `stage4-rk3-resolution-n${horizontalN}-d${targetDays}`;
}

export async function loadCompatibleStage4ResolutionCheckpoint(horizontalN:number,targetDays:number):Promise<Stage4Rk3ClimateCheckpoint|null>{
  const key=stage4ResolutionCheckpointKey(horizontalN,targetDays);
  try{
    const cp=await loadStage4Rk3Checkpoint(key);
    if(!cp)return null;
    assertStage4Rk3CheckpointCompatible(cp,stage4ResolutionModelSignature(horizontalN),targetDays);
    return cp;
  }catch{
    try{await clearStage4Rk3Checkpoint(key);}catch{}
    return null;
  }
}

export async function saveStage4ResolutionCheckpoint(horizontalN:number,targetDays:number,cp:Stage4Rk3ClimateCheckpoint):Promise<void>{
  assertStage4Rk3CheckpointCompatible(cp,stage4ResolutionModelSignature(horizontalN),targetDays);
  await saveStage4Rk3Checkpoint(cp,stage4ResolutionCheckpointKey(horizontalN,targetDays));
}
