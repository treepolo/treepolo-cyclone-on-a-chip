import { buildCubedSphere } from './grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from './grid/vertical.js';
import { buildIsothermalReference } from './physics/referenceAtmosphere.js';
import { diagnoseState } from './solver/diagnostics.js';
import { DryCoreCpu } from './solver/dryCoreCpu.js';
import { addThermalBubble } from './solver/initialConditions.js';
import { cloneState, createHydrostaticState, DryState } from './solver/state.js';
import { DebugViewer } from './render/debugViewer.js';
import { GpuDryCorePrototype } from './gpu/dryCoreGpu.js';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const logEl=$<HTMLDivElement>('log');const log=(x:string)=>{logEl.textContent=`${x}\n${logEl.textContent||''}`.slice(0,5000);};
const h=buildCubedSphere(8),v=buildStretchedVerticalGrid(32,40000,1.7),ref=buildIsothermalReference(v,288),cpu=new DryCoreCpu(h,v,ref);
let state=createHydrostaticState(h,v,ref),initialMass=diagnoseState(h,v,state).dryMass,running=false;
const viewer=new DebugViewer($<HTMLCanvasElement>('view'),h,v,ref);
$('grid').textContent=`6 × ${h.n} × ${h.n} × ${v.nz}`;$('cells').textContent=(h.cellCount*v.nz).toLocaleString();

function reset(){state=createHydrostaticState(h,v,ref);initialMass=diagnoseState(h,v,state).dryMass;running=false;$('run').textContent='Run';log('CPU state reset to isothermal hydrostatic reference.');}
function bubble(){addThermalBubble(h,v,ref,state,{lonDeg:0,latDeg:0,altitude:3500,horizontalRadius:1.1e6,verticalRadius:2200,deltaTheta:3});initialMass=diagnoseState(h,v,state).dryMass;log('Injected constant-pressure +3 K thermal bubble (physics initial condition, no upward velocity forced).');}
function step(n=1){try{for(let i=0;i<n;i++){const c=cpu.step(state,0.5);if(c.maxHorizontalCfl>0.8||c.maxVerticalCfl>0.8){running=false;log(`CFL guard: h=${c.maxHorizontalCfl.toFixed(3)}, v=${c.maxVerticalCfl.toFixed(3)}`);break;}}}catch(e){running=false;log(`PAUSED: ${String(e)}`);}}
function updateUi(){const d=diagnoseState(h,v,state),dr=(d.dryMass-initialMass)/initialMass;$('mass').textContent=dr.toExponential(3);$('wmax').textContent=`${d.maxAbsW.toFixed(4)} m/s`;$('rhomin').textContent=`${d.minRho.toExponential(3)} kg/m³`;$('pmin').textContent=`${d.minP.toFixed(1)} Pa`;$('time').textContent=`${state.time.toFixed(1)} s`;const bad=d.nan||d.minRho<=0||d.minP<=0;$('mass').className=Math.abs(dr)<1e-9?'ok':'bad';$('rhomin').className=bad?'bad':'ok';$('pmin').className=bad?'bad':'ok';viewer.draw(state);}
$('reset').onclick=reset;$('bubble').onclick=bubble;$('step').onclick=()=>step(1);$('run').onclick=()=>{running=!running;$('run').textContent=running?'Pause':'Run';};
function frame(){if(running)step(2);updateUi();requestAnimationFrame(frame);}frame();

(async()=>{
  const gpuEl=$('gpu');
  try{
    const gpu=await GpuDryCorePrototype.create(h,v,ref,createHydrostaticState(h,v,ref));gpu.step(0.25);const gs=await gpu.downloadState(0.25),d=diagnoseState(h,v,gs);
    if(d.nan||d.minRho<=0||d.minP<=0)throw new Error('GPU smoke produced invalid state');
    gpuEl.textContent='compute smoke OK';gpuEl.className='ok';log(`WebGPU pipelines compiled; hydrostatic smoke max|w|=${d.maxAbsW.toExponential(3)} m/s.`);gpu.destroy();
  }catch(e){gpuEl.textContent='unavailable / failed';gpuEl.className='bad';log(`WebGPU smoke not available: ${String(e)}`);}
})();
