import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { GpuRotatingDryCore } from '../gpu/rotatingDryCoreGpu.js';
import { GpuStage4SlowTendencyReference } from '../gpu/stage4SlowTendenciesGpu.js';
import { computeStage4SlowTendencies } from '../solver/stage4SlowTendenciesCpu.js';
import { createHydrostaticState, w3DIndex } from '../solver/state.js';

export interface SlowGpuAgreementResult{
  rhoRelativeL2:number;rhoThetaRelativeL2:number;hFluxRelativeL2:number;vFluxRelativeL2:number;
  hFluxSelfRelativeL2:number;vFluxSelfRelativeL2:number;rhoDivergenceSelfRelativeL2:number;rhoThetaDivergenceSelfRelativeL2:number;thermalRelativeL2:number;
  maxDeltaU:number;maxDeltaW:number;restMax:number;elapsedMs:number;pass:boolean;
}
function relL2(a:ArrayLike<number>,b:ArrayLike<number>):number{let n=0,d=0;for(let i=0;i<a.length;i++){const e=a[i]!-b[i]!;n+=e*e;d+=b[i]!*b[i]!;}return Math.sqrt(n/Math.max(d,Number.MIN_VALUE));}
function maxDiff(a:ArrayLike<number>,b:ArrayLike<number>):number{let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]!-b[i]!));return m;}
function maxAbs(a:ArrayLike<number>):number{let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]!));return m;}
const F=Math.fround;
const add=(a:number,b:number)=>F(F(a)+F(b)),sub=(a:number,b:number)=>F(F(a)-F(b)),mul=(a:number,b:number)=>F(F(a)*F(b)),div=(a:number,b:number)=>F(F(a)/F(b));

/** Rebuild HPERT/VPERT using the exact f32 inputs uploaded by the GPU path. */
function rebuildPerturbationFluxF32(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,ref:ReturnType<typeof buildHeldSuarezReference>,s:ReturnType<typeof createHydrostaticState>){
  const nz=v.nz,rho=Float32Array.from(s.rhoD),x=Float32Array.from(s.rhoThetaM),u=Float32Array.from(s.uEdge),w=Float32Array.from(s.wInterface),dz=Float32Array.from(v.dz),r0=Float32Array.from(ref.rhoCenter),x0=Float32Array.from(ref.rhoThetaCenter),edgeLength=new Float32Array(h.edgeCount),cellArea=new Float32Array(h.cellCount);
  for(let e=0;e<h.edgeCount;e++)edgeLength[e]=h.edges[e]!.angularLength*EARTH.radius;
  for(let c=0;c<h.cellCount;c++)cellArea[c]=h.cellAreaUnit[c]!*EARTH.radius*EARTH.radius;
  const hfR=new Float64Array(h.edgeCount*nz),hfX=new Float64Array(hfR.length),vfR=new Float64Array(h.cellCount*(nz+1)),vfX=new Float64Array(vfR.length);
  for(let e=0;e<h.edgeCount;e++)for(let k=0;k<nz;k++){
    const q=e*nz+k,vel=u[q]!,ge=h.edges[e]!,src=(vel>=0?ge.leftCell:ge.rightCell)*nz+k,A=mul(edgeLength[e]!,dz[k]!);
    hfR[q]=mul(mul(sub(rho[src]!,r0[k]!),vel),A);hfX[q]=mul(mul(sub(x[src]!,x0[k]!),vel),A);
  }
  for(let c=0;c<h.cellCount;c++)for(let i=1;i<nz;i++){
    const q=c*(nz+1)+i,vel=w[q]!,k=vel>=0?i-1:i,src=c*nz+k,A=cellArea[c]!;
    vfR[q]=mul(mul(sub(rho[src]!,r0[k]!),vel),A);vfX[q]=mul(mul(sub(x[src]!,x0[k]!),vel),A);
  }
  return{hfR,hfX,vfR,vfX,dz,cellArea};
}

/** Rebuild SDIV from the GPU-produced f32 fluxes; this is the conservation identity gate. */
function rebuildScalarDivergenceF32(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,hFlux:ArrayLike<number>,vFlux:ArrayLike<number>){
  const nz=v.nz,dz=Float32Array.from(v.dz),cellArea=new Float32Array(h.cellCount),rho=new Float64Array(h.cellCount*nz),x=new Float64Array(rho.length);
  for(let c=0;c<h.cellCount;c++)cellArea[c]=h.cellAreaUnit[c]!*EARTH.radius*EARTH.radius;
  for(let c=0;c<h.cellCount;c++)for(let k=0;k<nz;k++){
    let hr=F(0),hx=F(0);
    for(let slot=0;slot<4;slot++){
      const e=h.cellEdges[c*4+slot]!,sgn=h.cellEdgeSigns[c*4+slot]!,q=e*nz+k;
      hr=add(hr,mul(sgn,hFlux[q]!));hx=add(hx,mul(sgn,hFlux[q+h.edgeCount*nz]!));
    }
    const vb=c*(nz+1)+k,den=mul(cellArea[c]!,dz[k]!),q=c*nz+k;
    rho[q]=div(add(sub(F(0),hr),sub(vFlux[vb]!,vFlux[vb+1]!)),den);
    x[q]=div(add(sub(F(0),hx),sub(vFlux[vb+h.cellCount*(nz+1)]!,vFlux[vb+1+h.cellCount*(nz+1)]!)),den);
  }
  return{rho,x};
}

function packFluxPair(r:ArrayLike<number>,x:ArrayLike<number>):Float64Array{const out=new Float64Array(r.length*2);for(let q=0;q<r.length;q++){out[q*2]=r[q]!;out[q*2+1]=x[q]!;}return out;}
function unpackPairFlux(f:Float64Array,n:number):{r:Float64Array;x:Float64Array}{const r=new Float64Array(n),x=new Float64Array(n);for(let q=0;q<n;q++){r[q]=f[q*2]!;x[q]=f[q*2+1]!;}return{r,x};}

export async function runStage4SlowGpuAgreement():Promise<SlowGpuAgreementResult>{
  const t0=performance.now(),h=buildCubedSphere(2),v=buildStretchedVerticalGrid(16,26000,1.3),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref);
  for(let q=0;q<s.rhoD.length;q++){s.rhoD[q]=s.rhoD[q]!*(1+4e-4*Math.sin((q+1)*.173));s.rhoThetaM[q]=s.rhoThetaM[q]!*(1+6e-4*Math.cos((q+2)*.137));}
  for(let q=0;q<s.uEdge.length;q++)s.uEdge[q]=2.4*Math.sin((q+1)*.113)-.9*Math.cos((q+3)*.071);
  for(let c=0;c<h.cellCount;c++)for(let i=1;i<v.nz;i++)s.wInterface[w3DIndex(c,i,v.nz)]=.12*Math.sin(Math.PI*v.zInterface[i]!/v.top)*Math.sin((c+1)*.23+i*.37);
  const cpu=computeStage4SlowTendencies(h,v,ref,s,{heldSuarez:true,momentumTransport:true,coriolis:true}),cpuTransport=computeStage4SlowTendencies(h,v,ref,s,{heldSuarez:false,momentumTransport:false,coriolis:false}),core=await GpuRotatingDryCore.create(h,v,ref,s);
  core.device.pushErrorScope?.('validation');let gpu:GpuStage4SlowTendencyReference|undefined;
  try{
    gpu=GpuStage4SlowTendencyReference.create(core);const err=await core.device.popErrorScope?.();if(err)throw new Error(`slow tendency GPU validation: ${err.message||err}`);

    // Full RHS: keep the raw Float64-CPU vs Float32-GPU diagnostics visible.
    gpu.uploadPredictor(s);gpu.compute(true,true,true);const got=await gpu.download();
    const rhoRelativeL2=relL2(got.rhoD,cpu.rhoD),rhoThetaRelativeL2=relL2(got.rhoThetaM,cpu.rhoThetaM),hFluxRelativeL2=relL2(got.hMassFlux,cpu.hMassFlux),vFluxRelativeL2=relL2(got.vMassFlux,cpu.vMassFlux),maxDeltaU=maxDiff(got.uEdge,cpu.uEdge),maxDeltaW=maxDiff(got.wInterface,cpu.wInterface);

    // Transport-only rerun. This isolates HPERT/VPERT/SDIV from thermal forcing.
    gpu.uploadPredictor(s);gpu.compute(false,false,false);const transport=await gpu.download();
    const rebuilt=rebuildPerturbationFluxF32(h,v,ref,s),expH=packFluxPair(rebuilt.hfR,rebuilt.hfX),expV=packFluxPair(rebuilt.vfR,rebuilt.vfX),gotHPair=packFluxPair(transport.hMassFlux,new Float64Array(transport.hMassFlux.length)),gotVPair=packFluxPair(transport.vMassFlux,new Float64Array(transport.vMassFlux.length));
    // download() returns only mass carriers in hMassFlux/vMassFlux; read the packed GPU flux buffers directly for rhoTheta too.
    const U=(globalThis as any).GPUBufferUsage,M=(globalThis as any).GPUMapMode;
    const readPacked=async(buffer:any,count:number):Promise<Float64Array>=>{const bytes=count*8,stage=core.device.createBuffer({size:bytes,usage:U.COPY_DST|U.MAP_READ}),enc=core.device.createCommandEncoder();enc.copyBufferToBuffer(buffer,0,stage,0,bytes);core.device.queue.submit([enc.finish()]);await stage.mapAsync(M.READ);const f=new Float32Array(stage.getMappedRange().slice(0));stage.unmap();stage.destroy();return Float64Array.from(f);};
    const hPacked=await readPacked(gpu.hFlux,h.edgeCount*v.nz),vPacked=await readPacked(gpu.vFlux,h.cellCount*(v.nz+1)),hp=unpackPairFlux(hPacked,h.edgeCount*v.nz),vp=unpackPairFlux(vPacked,h.cellCount*(v.nz+1));
    const hFluxSelfRelativeL2=Math.max(relL2(hp.r,rebuilt.hfR),relL2(hp.x,rebuilt.hfX)),vFluxSelfRelativeL2=Math.max(relL2(vp.r,rebuilt.vfR),relL2(vp.x,rebuilt.vfX));

    // Strict discrete conservation identity from the GPU's own flux outputs.
    // Repack as [all rho, all X] for compact reconstruction helper.
    const hSplit=new Float64Array(hp.r.length*2),vSplit=new Float64Array(vp.r.length*2);for(let q=0;q<hp.r.length;q++){hSplit[q]=hp.r[q]!;hSplit[q+hp.r.length]=hp.x[q]!;}for(let q=0;q<vp.r.length;q++){vSplit[q]=vp.r[q]!;vSplit[q+vp.r.length]=vp.x[q]!;}
    const divSelf=rebuildScalarDivergenceF32(h,v,hSplit,vSplit),rhoDivergenceSelfRelativeL2=relL2(transport.rhoD,divSelf.rho),rhoThetaDivergenceSelfRelativeL2=relL2(transport.rhoThetaM,divSelf.x);

    // Thermal-only comparison by subtracting transport-only scalar RHS from the full scalar RHS.
    const gpuThermal=new Float64Array(got.rhoThetaM.length),cpuThermal=new Float64Array(got.rhoThetaM.length);for(let q=0;q<gpuThermal.length;q++){gpuThermal[q]=got.rhoThetaM[q]!-transport.rhoThetaM[q]!;cpuThermal[q]=cpu.rhoThetaM[q]!-cpuTransport.rhoThetaM[q]!;}
    const thermalRelativeL2=relL2(gpuThermal,cpuThermal);

    const rest=createHydrostaticState(h,v,ref);gpu.uploadPredictor(rest);gpu.compute(false,true,true);const rg=await gpu.download();const restMax=Math.max(maxAbs(rg.rhoD),maxAbs(rg.rhoThetaM),maxAbs(rg.uEdge),maxAbs(rg.wInterface));
    const pass=hFluxSelfRelativeL2<=5e-6&&vFluxSelfRelativeL2<=5e-6&&rhoDivergenceSelfRelativeL2<=5e-6&&rhoThetaDivergenceSelfRelativeL2<=5e-6&&thermalRelativeL2<=2e-4&&maxDeltaU<=4e-4&&maxDeltaW<=2e-5&&restMax<=2e-5;
    return{rhoRelativeL2,rhoThetaRelativeL2,hFluxRelativeL2,vFluxRelativeL2,hFluxSelfRelativeL2,vFluxSelfRelativeL2,rhoDivergenceSelfRelativeL2,rhoThetaDivergenceSelfRelativeL2,thermalRelativeL2,maxDeltaU,maxDeltaW,restMax,elapsedMs:performance.now()-t0,pass};
  }finally{gpu?.destroy();core.destroy();}
}
