import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { buildModelTopSpongeRates } from '../physics/modelTopSponge.js';
import { GpuDryCorePrototype } from '../gpu/dryCoreGpu.js';
import { GpuStage4AcousticColumnReference } from '../gpu/stage4AcousticColumnGpu.js';
import { predictorRelativeVerticalAcousticStep } from '../solver/stage4AcousticColumnCpu.js';
import { computeStage4FrozenRhs } from '../solver/stage4Rk3SplitCpu.js';
import { cloneState, createHydrostaticState, w3DIndex } from '../solver/state.js';

export interface AcousticGpuAgreementResult{
  rhoRelativeL2:number;
  rhoThetaRelativeL2:number;
  /** Informational CPU-f64 versus GPU-f32 Fref difference. */
  referenceMassFluxRelativeL2:number;
  /** Informational CPU-f64 versus GPU-f32 Fref difference. */
  referenceRhoThetaFluxRelativeL2:number;
  /** Tight identity check: GPU refFlux versus Fref recomputed from GPU time levels. */
  referenceMassFluxSelfRelativeL2:number;
  /** Tight identity check: GPU refFlux versus Fref-X recomputed from GPU time levels. */
  referenceRhoThetaFluxSelfRelativeL2:number;
  maxDeltaW:number;
  cpuMaxW:number;
  gpuMaxW:number;
  hydrostaticMaxW:number;
  hydrostaticRhoRelativeL2:number;
  elapsedMs:number;
  pass:boolean;
}
function relL2(a:ArrayLike<number>,b:ArrayLike<number>):number{let n=0,d=0;for(let i=0;i<a.length;i++){const e=a[i]!-b[i]!;n+=e*e;d+=b[i]!*b[i]!;}return Math.sqrt(n/Math.max(d,Number.MIN_VALUE));}
function maxAbs(a:ArrayLike<number>):number{let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]!));return m;}
function maxDiff(a:ArrayLike<number>,b:ArrayLike<number>):number{let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]!-b[i]!));return m;}
async function readF32(device:any,buffer:any,count:number):Promise<Float32Array>{const U=(globalThis as any).GPUBufferUsage,M=(globalThis as any).GPUMapMode,bytes=count*4,stage=device.createBuffer({size:Math.max(4,bytes),usage:U.COPY_DST|U.MAP_READ}),enc=device.createCommandEncoder();enc.copyBufferToBuffer(buffer,0,stage,0,bytes);device.queue.submit([enc.finish()]);await stage.mapAsync(M.READ);const out=new Float32Array(stage.getMappedRange().slice(0));stage.unmap();stage.destroy();return out;}

function cpuVerticalStep(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,ref:ReturnType<typeof buildHeldSuarezReference>,predictor:ReturnType<typeof createHydrostaticState>,initial:ReturnType<typeof cloneState>,dt:number,off:number){
  const out=cloneState(initial),frozen=computeStage4FrozenRhs(h,v,ref,predictor,{heldSuarez:false,momentumTransport:false,coriolis:false}),rates=buildModelTopSpongeRates(v),nz=v.nz,refMass=new Float64Array(h.cellCount*(nz+1)),refX=new Float64Array(refMass.length);
  for(let c=0;c<h.cellCount;c++){
    const p={rho:new Float64Array(nz),rhoTheta:new Float64Array(nz),w:new Float64Array(nz+1)},a={rho:new Float64Array(nz),rhoTheta:new Float64Array(nz),w:new Float64Array(nz+1)},r={rho:new Float64Array(nz),rhoTheta:new Float64Array(nz),w:new Float64Array(nz+1)};
    for(let k=0;k<nz;k++){const q=c*nz+k;p.rho[k]=predictor.rhoD[q]!;p.rhoTheta[k]=predictor.rhoThetaM[q]!;a.rho[k]=out.rhoD[q]!;a.rhoTheta[k]=out.rhoThetaM[q]!;r.rho[k]=frozen.rhoD[q]!;r.rhoTheta[k]=frozen.rhoThetaM[q]!;}
    for(let i=0;i<=nz;i++){const q=w3DIndex(c,i,nz);p.w[i]=predictor.wInterface[q]!;a.w[i]=out.wInterface[q]!;r.w[i]=frozen.wInterface[q]!;}
    const result=predictorRelativeVerticalAcousticStep(v,ref,p,a,r,dt,off,rates);
    for(let k=0;k<nz;k++){const q=c*nz+k;out.rhoD[q]=a.rho[k]!;out.rhoThetaM[q]=a.rhoTheta[k]!;}
    for(let i=0;i<=nz;i++){const q=w3DIndex(c,i,nz);out.wInterface[q]=a.w[i]!;refMass[q]=result.referenceMassFlux[i]!;refX[q]=result.referenceRhoThetaFlux[i]!;}
  }
  return{out,frozen,refMass,refX};
}

/**
 * Recompute the exact bookkeeping quantity that the WGSL kernel is supposed to
 * write, using f32-rounded predictor/old/new time levels.  This separates an
 * actual Fref-output bug from the expected CPU-f64 versus GPU-f32 solver error
 * in w.  The raw CPU/GPU Fref comparison is still reported as a diagnostic.
 */
function recomputeGpuFref(
  v:ReturnType<typeof buildStretchedVerticalGrid>,
  ref:ReturnType<typeof buildHeldSuarezReference>,
  predictor:ReturnType<typeof createHydrostaticState>,
  initial:ReturnType<typeof cloneState>,
  got:ReturnType<typeof cloneState>,
  off:number,
):{mass:Float64Array;x:Float64Array}{
  const n=predictor.wInterface.length,mass=new Float64Array(n),x=new Float64Array(n);
  const off32=Math.fround(off),theta=Math.fround(Math.fround(1+off32)*Math.fround(.5)),oldWeight=Math.fround(1-theta);
  for(let c=0;c<predictor.wInterface.length/(v.nz+1);c++)for(let i=0;i<=v.nz;i++){
    const q=w3DIndex(c,i,v.nz),pred=Math.fround(predictor.wInterface[q]!),old=Math.fround(initial.wInterface[q]!),nw=Math.fround(got.wInterface[q]!);
    const dwOld=Math.fround(old-pred),dwNew=Math.fround(nw-pred),oldTerm=Math.fround(oldWeight*dwOld),newTerm=Math.fround(theta*dwNew),weighted=Math.fround(Math.fround(pred+oldTerm)+newTerm);
    mass[q]=Math.fround(Math.fround(ref.rhoInterface[i]!)*weighted);
    x[q]=Math.fround(Math.fround(ref.rhoThetaInterface[i]!)*weighted);
  }
  return{mass,x};
}

export async function runStage4AcousticGpuAgreement():Promise<AcousticGpuAgreementResult>{
  const t0=performance.now(),h=buildCubedSphere(2),v=buildStretchedVerticalGrid(48,40000,1.4),ref=buildHeldSuarezReference(v),dt=2.5,off=.1;
  const predictor=createHydrostaticState(h,v,ref);
  for(let q=0;q<predictor.rhoD.length;q++){
    predictor.rhoD[q]=predictor.rhoD[q]!*(1+8e-5*Math.sin((q+1)*.173));
    predictor.rhoThetaM[q]=predictor.rhoThetaM[q]!*(1+1.1e-4*Math.cos((q+3)*.119));
  }
  for(let c=0;c<h.cellCount;c++)for(let i=1;i<v.nz;i++)predictor.wInterface[w3DIndex(c,i,v.nz)]=.012*Math.sin(Math.PI*v.zInterface[i]!/v.top)*Math.sin((c+1)*.27+i*.13);
  const acoustic=cloneState(predictor);
  for(let q=0;q<acoustic.rhoD.length;q++){
    acoustic.rhoD[q]=acoustic.rhoD[q]!*(1+3e-5*Math.cos((q+2)*.211));
    acoustic.rhoThetaM[q]=acoustic.rhoThetaM[q]!*(1-4e-5*Math.sin((q+5)*.157));
  }
  for(let c=0;c<h.cellCount;c++)for(let i=1;i<v.nz;i++)acoustic.wInterface[w3DIndex(c,i,v.nz)]=acoustic.wInterface[w3DIndex(c,i,v.nz)]!+.004*Math.sin(2*Math.PI*v.zInterface[i]!/v.top)*Math.cos((c+1)*.19+i*.17);
  const cpu=cpuVerticalStep(h,v,ref,predictor,acoustic,dt,off),core=await GpuDryCorePrototype.create(h,v,ref,predictor,off,buildModelTopSpongeRates(v));
  core.device.pushErrorScope?.('validation');
  let gpu:GpuStage4AcousticColumnReference|undefined;
  try{
    gpu=GpuStage4AcousticColumnReference.create(core);
    const err=await core.device.popErrorScope?.();if(err)throw new Error(`predictor-relative acoustic GPU validation: ${err.message||err}`);
    gpu.upload(predictor,acoustic,cpu.frozen);gpu.step(dt,off);
    const got=await gpu.downloadAcousticState(),rawFlux=await readF32(core.device,gpu.refFlux,h.cellCount*(v.nz+1)*2),gpuMass=new Float64Array(cpu.refMass.length),gpuX=new Float64Array(cpu.refX.length);
    for(let q=0;q<gpuMass.length;q++){gpuMass[q]=rawFlux[q*2]!;gpuX[q]=rawFlux[q*2+1]!;}
    const self=recomputeGpuFref(v,ref,predictor,acoustic,got,off);
    const rhoRelativeL2=relL2(got.rhoD,cpu.out.rhoD),rhoThetaRelativeL2=relL2(got.rhoThetaM,cpu.out.rhoThetaM),referenceMassFluxRelativeL2=relL2(gpuMass,cpu.refMass),referenceRhoThetaFluxRelativeL2=relL2(gpuX,cpu.refX),referenceMassFluxSelfRelativeL2=relL2(gpuMass,self.mass),referenceRhoThetaFluxSelfRelativeL2=relL2(gpuX,self.x),maxDeltaW=maxDiff(got.wInterface,cpu.out.wInterface),cpuMaxW=maxAbs(cpu.out.wInterface),gpuMaxW=maxAbs(got.wInterface);

    // Independent hydrostatic zero case through the same compiled kernel.
    const rest=createHydrostaticState(h,v,ref),restCpu=cpuVerticalStep(h,v,ref,rest,cloneState(rest),dt,off);gpu.upload(rest,rest,restCpu.frozen);gpu.step(dt,off);const restGpu=await gpu.downloadAcousticState(),hydrostaticMaxW=maxAbs(restGpu.wInterface),hydrostaticRhoRelativeL2=relL2(restGpu.rhoD,restCpu.out.rhoD);
    // Raw CPU-f64/GPU-f32 Fref error is only a loose sanity guard because it is
    // mathematically downstream of the allowed w solver error.  The tight
    // bookkeeping gate is the GPU self-identity above.
    const pass=rhoRelativeL2<=8e-5&&rhoThetaRelativeL2<=8e-5&&referenceMassFluxRelativeL2<=5e-3&&referenceRhoThetaFluxRelativeL2<=5e-3&&referenceMassFluxSelfRelativeL2<=5e-6&&referenceRhoThetaFluxSelfRelativeL2<=5e-6&&maxDeltaW<=3e-4&&hydrostaticMaxW<=2e-5&&hydrostaticRhoRelativeL2<=8e-5;
    return{rhoRelativeL2,rhoThetaRelativeL2,referenceMassFluxRelativeL2,referenceRhoThetaFluxRelativeL2,referenceMassFluxSelfRelativeL2,referenceRhoThetaFluxSelfRelativeL2,maxDeltaW,cpuMaxW,gpuMaxW,hydrostaticMaxW,hydrostaticRhoRelativeL2,elapsedMs:performance.now()-t0,pass};
  }finally{gpu?.destroy();core.destroy();}
}
