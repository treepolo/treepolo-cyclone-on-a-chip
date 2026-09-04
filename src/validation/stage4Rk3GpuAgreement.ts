import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { diagnoseState } from '../solver/diagnostics.js';
import { Stage4Rk3SplitCpu } from '../solver/stage4Rk3SplitCpu.js';
import { cloneState, createHydrostaticState, edge3DIndex, w3DIndex } from '../solver/state.js';
import { GpuStage4Rk3SplitReference } from '../gpu/stage4Rk3SplitGpu.js';

export interface Stage4Rk3GpuAgreementResult{
  rhoRelativeL2:number;
  rhoThetaRelativeL2:number;
  maxDeltaU:number;
  maxDeltaW:number;
  cpuMassDrift:number;
  gpuMassDrift:number;
  cpuMaxU:number;
  gpuMaxU:number;
  cpuMaxW:number;
  gpuMaxW:number;
  restRhoRelativeL2:number;
  restRhoThetaRelativeL2:number;
  restMaxDeltaU:number;
  restMaxDeltaW:number;
  restGpuMaxU:number;
  restGpuMaxW:number;
  restGpuMassDrift:number;
  elapsedMs:number;
  pass:boolean;
}

function relL2(a:ArrayLike<number>,b:ArrayLike<number>):number{let n=0,d=0;for(let i=0;i<a.length;i++){const e=a[i]!-b[i]!;n+=e*e;d+=b[i]!*b[i]!;}return Math.sqrt(n/Math.max(d,Number.MIN_VALUE));}
function maxDiff(a:ArrayLike<number>,b:ArrayLike<number>):number{let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]!-b[i]!));return m;}
function maxAbs(a:ArrayLike<number>):number{let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]!));return m;}
function massDrift(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,before:ReturnType<typeof createHydrostaticState>,after:ReturnType<typeof createHydrostaticState>):number{
  const m0=diagnoseState(h,v,before).dryMass,m1=diagnoseState(h,v,after).dryMass;return(m1-m0)/m0;
}
function valid(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,s:ReturnType<typeof createHydrostaticState>):boolean{const d=diagnoseState(h,v,s);return!d.nan&&d.minRho>0&&d.minP>0;}

export async function runStage4Rk3GpuAgreement():Promise<Stage4Rk3GpuAgreementResult>{
  const t0=performance.now(),h=buildCubedSphere(2),v=buildStretchedVerticalGrid(24,40000,1.35),ref=buildHeldSuarezReference(v),dt=10;
  const initial=createHydrostaticState(h,v,ref),nz=v.nz;

  // Smooth deterministic 3-D perturbation.  Magnitudes are intentionally small
  // enough to remain in the one-step agreement regime, but every slow/fast term
  // receives a nonzero input: scalar Fpert, pressure correction, u/w material
  // transport, Coriolis, Held-Suarez forcing, Rayleigh, and divergence damping.
  for(let c=0;c<h.cellCount;c++)for(let k=0;k<nz;k++){
    const q=c*nz+k,z=v.zCenter[k]!/v.top,phase=(c+1)*.173+(k+1)*.287;
    initial.rhoD[q]=initial.rhoD[q]!*(1+1.8e-4*Math.sin(phase)*Math.sin(Math.PI*z));
    initial.rhoThetaM[q]=initial.rhoThetaM[q]!*(1+2.2e-4*Math.cos(phase*.83)*Math.sin(Math.PI*z));
  }
  for(let e=0;e<h.edgeCount;e++)for(let k=0;k<nz;k++){
    const q=edge3DIndex(e,k,nz),z=v.zCenter[k]!/v.top;
    initial.uEdge[q]=1.4*Math.sin((e+1)*.137+(k+1)*.211)*(.35+.65*Math.cos(.5*Math.PI*z));
  }
  for(let c=0;c<h.cellCount;c++)for(let i=1;i<nz;i++){
    const z=v.zInterface[i]!/v.top;
    initial.wInterface[w3DIndex(c,i,nz)]=.025*Math.sin(Math.PI*z)*Math.sin((c+1)*.229+i*.317);
  }

  const cpuState=cloneState(initial),cpu=new Stage4Rk3SplitCpu(h,v,ref,4);
  cpu.step(cpuState,dt,{heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true});

  const gpuCore=await GpuStage4Rk3SplitReference.create(h,v,ref,initial,4);
  let gpuState:ReturnType<typeof cloneState>;
  try{
    gpuCore.device.pushErrorScope?.('validation');gpuCore.step(dt,{heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true});
    const err=await gpuCore.device.popErrorScope?.();if(err)throw new Error(`full RK3 GPU validation: ${err.message||err}`);
    gpuState=await gpuCore.downloadState(dt);
  }finally{gpuCore.destroy();}

  const rhoRelativeL2=relL2(gpuState.rhoD,cpuState.rhoD),rhoThetaRelativeL2=relL2(gpuState.rhoThetaM,cpuState.rhoThetaM),maxDeltaU=maxDiff(gpuState.uEdge,cpuState.uEdge),maxDeltaW=maxDiff(gpuState.wInterface,cpuState.wInterface),cpuMassDrift=massDrift(h,v,initial,cpuState),gpuMassDrift=massDrift(h,v,initial,gpuState),cpuMaxU=maxAbs(cpuState.uEdge),gpuMaxU=maxAbs(gpuState.uEdge),cpuMaxW=maxAbs(cpuState.wInterface),gpuMaxW=maxAbs(gpuState.wInterface);

  // Independent full-outer-step hydrostatic rest case through the exact same
  // orchestration. Held-Suarez is disabled so the physically correct result is
  // zero velocity and unchanged hydrostatic scalar state.
  const rest=createHydrostaticState(h,v,ref),restCpu=cloneState(rest);cpu.step(restCpu,dt,{heldSuarez:false,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true});
  const restGpuCore=await GpuStage4Rk3SplitReference.create(h,v,ref,rest,4);let restGpu:ReturnType<typeof cloneState>;
  try{
    restGpuCore.device.pushErrorScope?.('validation');restGpuCore.step(dt,{heldSuarez:false,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true});const err=await restGpuCore.device.popErrorScope?.();if(err)throw new Error(`hydrostatic full RK3 GPU validation: ${err.message||err}`);restGpu=await restGpuCore.downloadState(dt);
  }finally{restGpuCore.destroy();}
  const restRhoRelativeL2=relL2(restGpu.rhoD,restCpu.rhoD),restRhoThetaRelativeL2=relL2(restGpu.rhoThetaM,restCpu.rhoThetaM),restMaxDeltaU=maxDiff(restGpu.uEdge,restCpu.uEdge),restMaxDeltaW=maxDiff(restGpu.wInterface,restCpu.wInterface),restGpuMaxU=maxAbs(restGpu.uEdge),restGpuMaxW=maxAbs(restGpu.wInterface),restGpuMassDrift=massDrift(h,v,rest,restGpu);

  const finite=[rhoRelativeL2,rhoThetaRelativeL2,maxDeltaU,maxDeltaW,cpuMassDrift,gpuMassDrift,restRhoRelativeL2,restRhoThetaRelativeL2,restMaxDeltaU,restMaxDeltaW,restGpuMaxU,restGpuMaxW,restGpuMassDrift].every(Number.isFinite);
  const pass=finite&&valid(h,v,cpuState)&&valid(h,v,gpuState)&&valid(h,v,restGpu)&&rhoRelativeL2<=2e-4&&rhoThetaRelativeL2<=2e-4&&maxDeltaU<=2e-2&&maxDeltaW<=5e-3&&Math.abs(gpuMassDrift)<=2e-5&&restRhoRelativeL2<=1e-4&&restRhoThetaRelativeL2<=1e-4&&restMaxDeltaU<=2e-4&&restMaxDeltaW<=1e-4&&restGpuMaxU<=2e-4&&restGpuMaxW<=1e-4&&Math.abs(restGpuMassDrift)<=2e-5;
  return{rhoRelativeL2,rhoThetaRelativeL2,maxDeltaU,maxDeltaW,cpuMassDrift,gpuMassDrift,cpuMaxU,gpuMaxU,cpuMaxW,gpuMaxW,restRhoRelativeL2,restRhoThetaRelativeL2,restMaxDeltaU,restMaxDeltaW,restGpuMaxU,restGpuMaxW,restGpuMassDrift,elapsedMs:performance.now()-t0,pass};
}
