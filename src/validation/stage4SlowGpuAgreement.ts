import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { GpuRotatingDryCore } from '../gpu/rotatingDryCoreGpu.js';
import { GpuStage4SlowTendencyReference } from '../gpu/stage4SlowTendenciesGpu.js';
import { computeStage4SlowTendencies } from '../solver/stage4SlowTendenciesCpu.js';
import { createHydrostaticState, w3DIndex } from '../solver/state.js';

export interface SlowGpuAgreementResult{
  rhoRelativeL2:number;rhoThetaRelativeL2:number;hFluxRelativeL2:number;vFluxRelativeL2:number;maxDeltaU:number;maxDeltaW:number;restMax:number;elapsedMs:number;pass:boolean;
}
function relL2(a:ArrayLike<number>,b:ArrayLike<number>):number{let n=0,d=0;for(let i=0;i<a.length;i++){const e=a[i]!-b[i]!;n+=e*e;d+=b[i]!*b[i]!;}return Math.sqrt(n/Math.max(d,Number.MIN_VALUE));}
function maxDiff(a:ArrayLike<number>,b:ArrayLike<number>):number{let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]!-b[i]!));return m;}
function maxAbs(a:ArrayLike<number>):number{let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]!));return m;}

export async function runStage4SlowGpuAgreement():Promise<SlowGpuAgreementResult>{
  const t0=performance.now(),h=buildCubedSphere(2),v=buildStretchedVerticalGrid(16,26000,1.3),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref);
  for(let q=0;q<s.rhoD.length;q++){s.rhoD[q]=s.rhoD[q]!*(1+4e-4*Math.sin((q+1)*.173));s.rhoThetaM[q]=s.rhoThetaM[q]!*(1+6e-4*Math.cos((q+2)*.137));}
  for(let q=0;q<s.uEdge.length;q++)s.uEdge[q]=2.4*Math.sin((q+1)*.113)-.9*Math.cos((q+3)*.071);
  for(let c=0;c<h.cellCount;c++)for(let i=1;i<v.nz;i++)s.wInterface[w3DIndex(c,i,v.nz)]=.12*Math.sin(Math.PI*v.zInterface[i]!/v.top)*Math.sin((c+1)*.23+i*.37);
  const cpu=computeStage4SlowTendencies(h,v,ref,s,{heldSuarez:true,momentumTransport:true,coriolis:true}),core=await GpuRotatingDryCore.create(h,v,ref,s);
  core.device.pushErrorScope?.('validation');let gpu:GpuStage4SlowTendencyReference|undefined;
  try{
    gpu=GpuStage4SlowTendencyReference.create(core);const err=await core.device.popErrorScope?.();if(err)throw new Error(`slow tendency GPU validation: ${err.message||err}`);
    gpu.uploadPredictor(s);gpu.compute(true,true,true);const got=await gpu.download();
    const rhoRelativeL2=relL2(got.rhoD,cpu.rhoD),rhoThetaRelativeL2=relL2(got.rhoThetaM,cpu.rhoThetaM),hFluxRelativeL2=relL2(got.hMassFlux,cpu.hMassFlux),vFluxRelativeL2=relL2(got.vMassFlux,cpu.vMassFlux),maxDeltaU=maxDiff(got.uEdge,cpu.uEdge),maxDeltaW=maxDiff(got.wInterface,cpu.wInterface);

    const rest=createHydrostaticState(h,v,ref);gpu.uploadPredictor(rest);gpu.compute(false,true,true);const rg=await gpu.download();const restMax=Math.max(maxAbs(rg.rhoD),maxAbs(rg.rhoThetaM),maxAbs(rg.uEdge),maxAbs(rg.wInterface));
    const pass=rhoRelativeL2<=1.2e-4&&rhoThetaRelativeL2<=1.2e-4&&hFluxRelativeL2<=1.2e-4&&vFluxRelativeL2<=1.2e-4&&maxDeltaU<=4e-4&&maxDeltaW<=2e-5&&restMax<=2e-5;
    return{rhoRelativeL2,rhoThetaRelativeL2,hFluxRelativeL2,vFluxRelativeL2,maxDeltaU,maxDeltaW,restMax,elapsedMs:performance.now()-t0,pass};
  }finally{gpu?.destroy();core.destroy();}
}
