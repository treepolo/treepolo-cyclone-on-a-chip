import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { diagnoseState } from '../solver/diagnostics.js';
import { Stage4Rk3SplitCpu } from '../solver/stage4Rk3SplitCpu.js';
import { cloneState, createHydrostaticState, edge3DIndex, w3DIndex } from '../solver/state.js';
import { GpuStage4Rk3SplitReference } from '../gpu/stage4Rk3SplitGpu.js';

export interface Stage4Rk3GpuMultistepAgreementResult{
  steps:number;dt:number;simulatedSeconds:number;
  rhoRelativeL2:number;rhoThetaRelativeL2:number;maxDeltaU:number;maxDeltaW:number;
  gpuMassDrift:number;cpuMassDrift:number;cpuMaxU:number;gpuMaxU:number;cpuMaxW:number;gpuMaxW:number;
  elapsedMs:number;pass:boolean;
}
function relL2(a:ArrayLike<number>,b:ArrayLike<number>):number{let n=0,d=0;for(let i=0;i<a.length;i++){const e=a[i]!-b[i]!;n+=e*e;d+=b[i]!*b[i]!;}return Math.sqrt(n/Math.max(d,Number.MIN_VALUE));}
function maxDiff(a:ArrayLike<number>,b:ArrayLike<number>):number{let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]!-b[i]!));return m;}
function maxAbs(a:ArrayLike<number>):number{let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]!));return m;}

export async function runStage4Rk3GpuMultistepAgreement():Promise<Stage4Rk3GpuMultistepAgreementResult>{
  const t0=performance.now(),h=buildCubedSphere(2),v=buildStretchedVerticalGrid(16,30000,1.3),ref=buildHeldSuarezReference(v),dt=10,steps=40,nz=v.nz,initial=createHydrostaticState(h,v,ref);
  for(let c=0;c<h.cellCount;c++)for(let k=0;k<nz;k++){
    const q=c*nz+k,z=v.zCenter[k]!/v.top,p=(c+1)*.151+(k+1)*.263;
    initial.rhoD[q]=initial.rhoD[q]!*(1+8e-5*Math.sin(p)*Math.sin(Math.PI*z));
    initial.rhoThetaM[q]=initial.rhoThetaM[q]!*(1+1.1e-4*Math.cos(p*.91)*Math.sin(Math.PI*z));
  }
  for(let e=0;e<h.edgeCount;e++)for(let k=0;k<nz;k++)initial.uEdge[edge3DIndex(e,k,nz)]=.35*Math.sin((e+1)*.117+(k+1)*.193);
  for(let c=0;c<h.cellCount;c++)for(let i=1;i<nz;i++)initial.wInterface[w3DIndex(c,i,nz)]=.006*Math.sin(Math.PI*v.zInterface[i]!/v.top)*Math.sin((c+1)*.217+i*.281);
  const d0=diagnoseState(h,v,initial),cpuState=cloneState(initial),cpu=new Stage4Rk3SplitCpu(h,v,ref,4),opts={heldSuarez:false,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true} as const;
  for(let n=0;n<steps;n++)cpu.step(cpuState,dt,opts);
  const gpu=await GpuStage4Rk3SplitReference.create(h,v,ref,initial,4);let gpuState:ReturnType<typeof cloneState>;
  try{
    gpu.device.pushErrorScope?.('validation');for(let n=0;n<steps;n++)gpu.step(dt,opts);const err=await gpu.device.popErrorScope?.();if(err)throw new Error(`multistep RK3 GPU validation: ${err.message||err}`);gpuState=await gpu.downloadState(steps*dt);
  }finally{gpu.destroy();}
  const dc=diagnoseState(h,v,cpuState),dg=diagnoseState(h,v,gpuState),rhoRelativeL2=relL2(gpuState.rhoD,cpuState.rhoD),rhoThetaRelativeL2=relL2(gpuState.rhoThetaM,cpuState.rhoThetaM),maxDeltaU=maxDiff(gpuState.uEdge,cpuState.uEdge),maxDeltaW=maxDiff(gpuState.wInterface,cpuState.wInterface),gpuMassDrift=(dg.dryMass-d0.dryMass)/d0.dryMass,cpuMassDrift=(dc.dryMass-d0.dryMass)/d0.dryMass,cpuMaxU=maxAbs(cpuState.uEdge),gpuMaxU=maxAbs(gpuState.uEdge),cpuMaxW=maxAbs(cpuState.wInterface),gpuMaxW=maxAbs(gpuState.wInterface),finite=[rhoRelativeL2,rhoThetaRelativeL2,maxDeltaU,maxDeltaW,gpuMassDrift,cpuMassDrift,cpuMaxU,gpuMaxU,cpuMaxW,gpuMaxW].every(Number.isFinite);
  const pass=finite&&!dc.nan&&!dg.nan&&dc.minRho>0&&dg.minRho>0&&dc.minP>0&&dg.minP>0&&rhoRelativeL2<=2e-3&&rhoThetaRelativeL2<=2e-3&&maxDeltaU<=.12&&maxDeltaW<=.03&&Math.abs(gpuMassDrift)<=5e-5;
  return{steps,dt,simulatedSeconds:steps*dt,rhoRelativeL2,rhoThetaRelativeL2,maxDeltaU,maxDeltaW,gpuMassDrift,cpuMassDrift,cpuMaxU,gpuMaxU,cpuMaxW,gpuMaxW,elapsedMs:performance.now()-t0,pass};
}
