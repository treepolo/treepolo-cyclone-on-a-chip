import { DRY_AIR, EARTH } from './core/constants.js';
import { buildCubedSphere } from './grid/cubedSphere.js';
import { createConservativeFields } from './corev2/fields.js';
import { buildIsothermalHydrostaticReference, type HydrostaticReference1D } from './corev2/hydrostaticReference.js';
import { buildLinearReconstructionStencil } from './corev2/reconstruction.js';
import { conservedFromPrimitive } from './corev2/state.js';
import { buildSphericalShellGeometry, shellCellIndex } from './corev2/sphericalShellGeometry.js';
import { closedShellWellBalancedEulerGravityRate } from './corev2/wellBalancedEulerGravity.js';
import { buildCoreV2GpuInternalFaceStaticData, CoreV2GpuInternalFaceFlux } from './gpu/coreV2FaceFluxGpu.js';
import { buildCoreV2GpuRateStaticData, CoreV2GpuRateGather } from './gpu/coreV2RateGatherGpu.js';
import { buildCoreV2GpuReconstructionStaticData, CoreV2GpuReconstruction } from './gpu/coreV2ReconstructionGpu.js';

const $=(id:string):HTMLElement=>{const n=document.getElementById(id);if(!n)throw new Error(`missing #${id}`);return n;};
const f32=Math.fround;
function f32Reference(r:HydrostaticReference1D):HydrostaticReference1D{const cast=(a:Float64Array)=>Float64Array.from(a,f32);return{temperature:f32(r.temperature),surfacePressure:f32(r.surfacePressure),cellDensity:cast(r.cellDensity),cellPressure:cast(r.cellPressure),radialFacePressure:cast(r.radialFacePressure),sideFacePressure:cast(r.sideFacePressure),cellGeopotential:cast(r.cellGeopotential)};}
function pack(fields:ReturnType<typeof createConservativeFields>):Float32Array{const a=new Float32Array(fields.rho.length*8);for(let q=0;q<fields.rho.length;q++){let i=q*8;a[i]=fields.rho[q]!;a[i+1]=fields.momX[q]!;a[i+2]=fields.momY[q]!;a[i+3]=fields.momZ[q]!;a[i+4]=fields.rhoE[q]!;}return a;}
function unpack(a:Float32Array){const f=createConservativeFields(a.length/8);for(let q=0;q<f.rho.length;q++){let i=q*8;f.rho[q]=a[i]!;f.momX[q]=a[i+1]!;f.momY[q]=a[i+2]!;f.momZ[q]=a[i+3]!;f.rhoE[q]=a[i+4]!;}return f;}
function rel(actual:ArrayLike<number>,expected:ArrayLike<number>):number{let n=0,d=0;for(let i=0;i<actual.length;i++){let e=expected[i]!,x=actual[i]!-e;n+=x*x;d+=e*e;}return Math.sqrt(n/Math.max(d,1e-30));}
function referenceEnergyF32(rho:number,p:number,phi:number):number{const gamma=f32(DRY_AIR.gamma),gm1=f32(gamma-f32(1));const internal=f32(f32(p)/gm1);const potential=f32(f32(rho)*f32(phi));return f32(internal+potential);}

async function gpuRate(device:any,packed:Float32Array,geometry:ReturnType<typeof buildSphericalShellGeometry>,stencil:ReturnType<typeof buildLinearReconstructionStencil>,reference:HydrostaticReference1D):Promise<Float32Array>{
 const reconData=buildCoreV2GpuReconstructionStaticData(geometry,stencil,reference);const faceData=buildCoreV2GpuInternalFaceStaticData(geometry,stencil,reference);const rateData=buildCoreV2GpuRateStaticData(geometry,stencil,reference,faceData);
 const recon=new CoreV2GpuReconstruction(device,reconData),faces=new CoreV2GpuInternalFaceFlux(device,faceData),gather=new CoreV2GpuRateGather(device,rateData);
 try{const gradients=await recon.computeGradients(packed);const flux=await faces.compute(packed,gradients);return await gather.compute(packed,gradients,flux);}finally{recon.destroy();faces.destroy();gather.destroy();}
}

async function run():Promise<void>{
 $('status').textContent='RUNNING';const webgpu=(navigator as Navigator&{gpu?:any}).gpu;if(!webgpu)throw new Error('navigator.gpu unavailable');const adapter=await webgpu.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw new Error('WebGPU adapter unavailable');const device=await adapter.requestDevice();
 const horizontal=buildCubedSphere(3);const geometry=buildSphericalShellGeometry(horizontal,new Float64Array([EARTH.radius,EARTH.radius+150,EARTH.radius+700,EARTH.radius+2600,EARTH.radius+9000]));const reference=f32Reference(buildIsothermalHydrostaticReference(geometry,286,100200));const stencil=buildLinearReconstructionStencil(geometry);const fields=createConservativeFields(horizontal.cellCount*geometry.nz);
 for(let c=0;c<horizontal.cellCount;c++){const x=horizontal.cellCenters[c*3]!,y=horizontal.cellCenters[c*3+1]!,z=horizontal.cellCenters[c*3+2]!;for(let k=0;k<geometry.nz;k++){const q=shellCellIndex(c,k,geometry.nz),v=(k+.5)/geometry.nz;const s=conservedFromPrimitive({rho:reference.cellDensity[k]!*(1+.006*x-.004*y+.002*v),pressure:reference.cellPressure[k]!*(1+.007*y+.003*z-.0015*v),velocity:[14*y+1.5*v,-11*x+z,2.5*z-.3*v]},reference.cellGeopotential[k]!);fields.rho[q]=f32(s.rho);fields.momX[q]=f32(s.momentum[0]);fields.momY[q]=f32(s.momentum[1]);fields.momZ[q]=f32(s.momentum[2]);fields.rhoE[q]=f32(s.rhoE);}}
 const packed=pack(fields);const cpu=closedShellWellBalancedEulerGravityRate(unpack(packed),geometry,stencil,reference);const gpu=await gpuRate(device,packed,geometry,stencil,reference);
 const massGpu=new Float64Array(fields.rho.length),massCpu=new Float64Array(fields.rho.length),momGpu=new Float64Array(fields.rho.length*3),momCpu=new Float64Array(fields.rho.length*3),energyGpu=new Float64Array(fields.rho.length),energyCpu=new Float64Array(fields.rho.length);
 for(let q=0;q<fields.rho.length;q++){massGpu[q]=gpu[q*8]!;massCpu[q]=cpu.rho[q]!;momGpu[q*3]=gpu[q*8+1]!;momGpu[q*3+1]=gpu[q*8+2]!;momGpu[q*3+2]=gpu[q*8+3]!;momCpu[q*3]=cpu.momX[q]!;momCpu[q*3+1]=cpu.momY[q]!;momCpu[q*3+2]=cpu.momZ[q]!;energyGpu[q]=gpu[q*8+4]!;energyCpu[q]=cpu.rhoE[q]!;}
 const massErr=rel(massGpu,massCpu),momErr=rel(momGpu,momCpu),energyErr=rel(energyGpu,energyCpu);$('massErr').textContent=massErr.toExponential(6);$('momErr').textContent=momErr.toExponential(6);$('energyErr').textContent=energyErr.toExponential(6);
 const hydro=new Float32Array(fields.rho.length*8);for(let c=0;c<horizontal.cellCount;c++)for(let k=0;k<geometry.nz;k++){const q=shellCellIndex(c,k,geometry.nz),i=q*8,rho=f32(reference.cellDensity[k]!),p=f32(reference.cellPressure[k]!),phi=f32(reference.cellGeopotential[k]!);hydro[i]=rho;hydro[i+4]=referenceEnergyF32(rho,p,phi);}const hydroRate=await gpuRate(device,hydro,geometry,stencil,reference);let hydroMax=0;for(const x of hydroRate)hydroMax=Math.max(hydroMax,Math.abs(x));$('hydroMax').textContent=hydroMax.toExponential(6);$('adapter').textContent=JSON.stringify(adapter.info??{},null,2);
 const pass=massErr<8e-3&&momErr<8e-3&&energyErr<8e-3&&hydroMax===0;$('status').textContent=pass?'GATE PASS':'GATE FAIL';$('status').className=pass?'ok':'bad';if(!pass)throw new Error(`Core v2 GPU full spatial operator gate: mass=${massErr} mom=${momErr} energy=${energyErr} hydro=${hydroMax}`);
}
run().catch(e=>{console.error(e);$('status').textContent='GATE FAIL';$('status').className='bad';$('log').textContent=e instanceof Error?(e.stack??e.message):String(e);});
