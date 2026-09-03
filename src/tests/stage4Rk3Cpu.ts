declare const process:{exitCode?:number};
import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { addHeldSuarezWavePerturbation, buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { diagnoseState } from '../solver/diagnostics.js';
import { Stage4Rk3SplitCpu } from '../solver/stage4Rk3SplitCpu.js';
import { createHydrostaticState, w3DIndex } from '../solver/state.js';

function assert(cond:unknown,msg:string):asserts cond{if(!cond)throw new Error(msg)}
function maxAbs(a:ArrayLike<number>):number{let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]!));return m}
interface Test{name:string;fn:()=>void}
const tests:Test[]=[];const test=(name:string,fn:()=>void)=>tests.push({name,fn});

test('CPU RK3 split keeps hydrostatic rest well balanced',()=>{
  const h=buildCubedSphere(2),v=buildStretchedVerticalGrid(16,30000,1.4),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref),core=new Stage4Rk3SplitCpu(h,v,ref,4),m0=diagnoseState(h,v,s).dryMass;
  for(let n=0;n<40;n++)core.step(s,10,{heldSuarez:false,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true});
  const d=diagnoseState(h,v,s),mass=Math.abs((d.dryMass-m0)/m0);
  assert(!d.nan&&d.minRho>0&&d.minP>0,'hydrostatic RK3 state invalid');
  assert(mass<5e-13,`hydrostatic RK3 mass drift=${mass}`);
  assert(d.maxAbsW<2e-9,`hydrostatic RK3 max|w|=${d.maxAbsW}`);
  assert(maxAbs(s.uEdge)<2e-9,`hydrostatic RK3 max|u|=${maxAbs(s.uEdge)}`);
});

test('CPU RK3 split conserves global mass with 3-D transport and no sources',()=>{
  const h=buildCubedSphere(2),v=buildStretchedVerticalGrid(10,16000,1.2),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref),core=new Stage4Rk3SplitCpu(h,v,ref,4);
  for(let q=0;q<s.rhoD.length;q++){const f=1+5e-4*Math.sin((q+1)*.29);s.rhoD[q]=s.rhoD[q]!*f;s.rhoThetaM[q]=s.rhoThetaM[q]!*f;}
  for(let q=0;q<s.uEdge.length;q++)s.uEdge[q]=.4*Math.sin((q+1)*.17);
  for(let c=0;c<h.cellCount;c++)for(let i=1;i<v.nz;i++)s.wInterface[w3DIndex(c,i,v.nz)]=.02*Math.sin(Math.PI*v.zInterface[i]!/v.top)*Math.sin((c+1)*.31);
  const m0=diagnoseState(h,v,s).dryMass;
  for(let n=0;n<100;n++)core.step(s,2,{heldSuarez:false,momentumTransport:true,coriolis:false,divergenceDamping:true,topAbsorber:false});
  const d=diagnoseState(h,v,s),mass=Math.abs((d.dryMass-m0)/m0);
  assert(!d.nan&&d.minRho>0&&d.minP>0,'3-D RK3 mass test invalid');
  assert(mass<2e-12,`3-D RK3 global mass drift=${mass}`);
});

test('CPU RK3 split keeps a small vertically standing mode bounded',()=>{
  const h=buildCubedSphere(2),v=buildStretchedVerticalGrid(20,20000,1.15),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref),core=new Stage4Rk3SplitCpu(h,v,ref,4);
  for(let c=0;c<h.cellCount;c++)for(let i=1;i<v.nz;i++)s.wInterface[w3DIndex(c,i,v.nz)]=.01*Math.sin(Math.PI*v.zInterface[i]!/v.top);
  const initial=maxAbs(s.wInterface);let peak=initial;
  for(let n=0;n<120;n++){core.step(s,4,{heldSuarez:false,momentumTransport:false,coriolis:false,divergenceDamping:false,topAbsorber:false});peak=Math.max(peak,maxAbs(s.wInterface));}
  const d=diagnoseState(h,v,s);
  assert(!d.nan&&d.minRho>0&&d.minP>0,'standing RK3 mode invalid');
  assert(peak<initial*8,`standing RK3 mode unbounded initial=${initial}, peak=${peak}`);
});

test('CPU RK3 split Held-Suarez forcing develops a finite circulation without mass loss',()=>{
  const h=buildCubedSphere(2),v=buildStretchedVerticalGrid(10,24000,1.3),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref),core=new Stage4Rk3SplitCpu(h,v,ref,4);addHeldSuarezWavePerturbation(h,v,ref,s,.05);
  const m0=diagnoseState(h,v,s).dryMass,dt=10,steps=Math.round(.05*86400/dt);let peakU=0,peakW=0;
  for(let n=0;n<steps;n++){
    core.step(s,dt,{heldSuarez:true,momentumTransport:true,coriolis:true,divergenceDamping:true,topAbsorber:true});
    if(n%40===0){peakU=Math.max(peakU,maxAbs(s.uEdge));peakW=Math.max(peakW,maxAbs(s.wInterface));}
  }
  const d=diagnoseState(h,v,s),mass=Math.abs((d.dryMass-m0)/m0);
  assert(!d.nan&&d.minRho>0&&d.minP>0,'Held-Suarez RK3 state invalid');
  assert(mass<2e-12,`Held-Suarez RK3 mass drift=${mass}`);
  assert(peakU>1e-4&&Number.isFinite(peakU),`Held-Suarez RK3 failed to develop wind peak=${peakU}`);
  assert(peakW<20&&Number.isFinite(peakW),`Held-Suarez RK3 vertical velocity unreasonable peak=${peakW}`);
});

let passed=0;for(const t of tests){try{t.fn();console.log(`PASS ${t.name}`);passed++}catch(e){console.error(`FAIL ${t.name}`);console.error(e);process.exitCode=1}}
console.log(`${passed}/${tests.length} Stage 4 CPU RK3 split tests passed`);
