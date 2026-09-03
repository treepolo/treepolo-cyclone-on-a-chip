import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { addHeldSuarezWavePerturbation, buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { buildRotationGeometry, setAnalyticCellWind } from '../physics/rotation.js';
import { diagnoseState } from '../solver/diagnostics.js';
import { RotatingDryCoreCpu } from '../solver/rotatingDryCoreCpu.js';
import { diagnoseZonalMeans, ZonalMeanDiagnostics } from '../solver/stage4Diagnostics.js';
import { createHydrostaticState, DryState } from '../solver/state.js';
import { GpuStage4Integrator } from '../gpu/stage4IntegratorGpu.js';

const CLIMATE_HORIZONTAL_N=8;
const CLIMATE_VERTICAL_NZ=20;
const CLIMATE_TOP_METERS=30000;
const CLIMATE_ZONAL_BINS=24;

function relL2(a:ArrayLike<number>,b:ArrayLike<number>):number{let n=0,d=0;for(let i=0;i<a.length;i++){const x=a[i]!-b[i]!;n+=x*x;d+=b[i]!*b[i]!}return Math.sqrt(n/Math.max(d,Number.MIN_VALUE))}
function maxDiff(a:ArrayLike<number>,b:ArrayLike<number>):number{let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]!-b[i]!));return m}
export interface Stage4AgreementSample{step:number;massDrift:number;rhoL2:number;rhoThetaL2:number;uMaxDiff:number;wMaxDiff:number;gpuMaxW:number;invalid:boolean}
export interface Stage4AgreementResult{passed:boolean;samples:Stage4AgreementSample[];failures:string[];elapsedMs:number}
export async function runStage4GpuAgreement(onSample?:(s:Stage4AgreementSample)=>void):Promise<Stage4AgreementResult>{
  const h=buildCubedSphere(4),v=buildStretchedVerticalGrid(12,30000,1.4),ref=buildHeldSuarezReference(v),cpuState=createHydrostaticState(h,v,ref),gpuState=createHydrostaticState(h,v,ref),g=buildRotationGeometry(h);addHeldSuarezWavePerturbation(h,v,ref,cpuState,.05);addHeldSuarezWavePerturbation(h,v,ref,gpuState,.05);const wind=(r:readonly[number,number,number])=>[-10*r[1],10*r[0],0] as const;setAnalyticCellWind(h,g,cpuState,r=>wind(r));setAnalyticCellWind(h,g,gpuState,r=>wind(r));const cpu=new RotatingDryCoreCpu(h,v,ref),gpu=await GpuStage4Integrator.create(h,v,ref,gpuState),initial=await gpu.downloadState(0),m0=diagnoseState(h,v,initial).dryMass,dt=5,check=[1,10,100,250,500],samples:Stage4AgreementSample[]=[],failures:string[]=[],t0=performance.now();let done=0;
  try{for(const target of check){const n=target-done;for(let i=0;i<n;i++)cpu.step(cpuState,dt);gpu.stepBatch(dt,n,true,1);await gpu.device.queue.onSubmittedWorkDone();const gs=await gpu.downloadState(target*dt),gd=diagnoseState(h,v,gs),s:Stage4AgreementSample={step:target,massDrift:(gd.dryMass-m0)/m0,rhoL2:relL2(gs.rhoD,cpuState.rhoD),rhoThetaL2:relL2(gs.rhoThetaM,cpuState.rhoThetaM),uMaxDiff:maxDiff(gs.uEdge,cpuState.uEdge),wMaxDiff:maxDiff(gs.wInterface,cpuState.wInterface),gpuMaxW:gd.maxAbsW,invalid:gd.nan||gd.minRho<=0||gd.minP<=0};samples.push(s);onSample?.(s);done=target}
    for(const s of samples){if(s.invalid)failures.push(`step ${s.step}: invalid state`);if(Math.abs(s.massDrift)>2e-6)failures.push(`step ${s.step}: mass drift ${s.massDrift}`);if(s.rhoL2>1e-4)failures.push(`step ${s.step}: rho L2 ${s.rhoL2}`);if(s.rhoThetaL2>1e-4)failures.push(`step ${s.step}: rhoTheta L2 ${s.rhoThetaL2}`);if(s.uMaxDiff>.05)failures.push(`step ${s.step}: u diff ${s.uMaxDiff}`);if(s.wMaxDiff>.02)failures.push(`step ${s.step}: w diff ${s.wMaxDiff}`)}return{passed:failures.length===0,samples,failures,elapsedMs:performance.now()-t0};
  }finally{gpu.destroy()}
}
export interface ClimateDaySample{day:number;massDrift:number;maxW:number;jet:number;trade:number;psi:number;nhPsi:number;shPsi:number;invalid:boolean}
export interface Stage4ClimateResult{passed:boolean;samples:ClimateDaySample[];failures:string[];elapsedMs:number;finalZonal?:ZonalMeanDiagnostics}
export async function runHeldSuarezGpuClimate(days=30,onSample?:(s:ClimateDaySample)=>void):Promise<Stage4ClimateResult>{
  const h=buildCubedSphere(CLIMATE_HORIZONTAL_N),v=buildStretchedVerticalGrid(CLIMATE_VERTICAL_NZ,CLIMATE_TOP_METERS,1.4),ref=buildHeldSuarezReference(v),seed=createHydrostaticState(h,v,ref);addHeldSuarezWavePerturbation(h,v,ref,seed,.05);const gpu=await GpuStage4Integrator.create(h,v,ref,seed),initial=await gpu.downloadState(0),m0=diagnoseState(h,v,initial).dryMass,dt=10,stepsPerQuarterDay=Math.round(21600/dt),segments=days*4,samples:ClimateDaySample[]=[],failures:string[]=[],t0=performance.now();let finalZonal:ZonalMeanDiagnostics|undefined;
  try{for(let segment=1;segment<=segments;segment++){let left=stepsPerQuarterDay;while(left>0){const n=Math.min(200,left);gpu.stepBatch(dt,n,true,10);left-=n;if(left%1000===0){await gpu.device.queue.onSubmittedWorkDone();await new Promise<void>(r=>setTimeout(r,0))}}await gpu.device.queue.onSubmittedWorkDone();const simDay=segment/4,state=await gpu.downloadState(simDay*86400),d=diagnoseState(h,v,state),z=diagnoseZonalMeans(h,v,state,CLIMATE_ZONAL_BINS),s:ClimateDaySample={day:simDay,massDrift:(d.dryMass-m0)/m0,maxW:d.maxAbsW,jet:z.maxUpperMidlatitudeWesterly,trade:z.meanTropicalLowLevelZonal,psi:z.maxAbsStreamfunction,nhPsi:z.nhDominantStreamfunction,shPsi:z.shDominantStreamfunction,invalid:d.nan||d.minRho<=0||d.minP<=0};samples.push(s);onSample?.(s);finalZonal=z;
      if(s.invalid){failures.push(`day ${simDay}: invalid state`);break}
      if(Math.abs(s.massDrift)>5e-5){failures.push(`day ${simDay}: mass drift ${s.massDrift}`);break}
      if(!(s.maxW<10)){failures.push(`day ${simDay}: vertical-velocity stability guard exceeded: ${s.maxW}`);break}
    }
    const f=samples[samples.length-1];if(!f)failures.push('no climate samples');else if(f.day===days&&!f.invalid){if(!(f.jet>.5))failures.push(`midlatitude upper westerly too weak: ${f.jet}`);if(!(f.trade<0))failures.push(`tropical low-level easterly absent: ${f.trade}`);if(!(f.psi>1e9))failures.push(`overturning too weak: ${f.psi}`);if(!(f.nhPsi*f.shPsi<0))failures.push(`hemisphere overturning signs not opposite: ${f.nhPsi},${f.shPsi}`)}else if(f.day<days)failures.push(`validation stopped at day ${f.day} before ${days}`);
    return{passed:failures.length===0,samples,failures,elapsedMs:performance.now()-t0,finalZonal};
  }finally{gpu.destroy()}
}