declare const process:{exitCode?:number};
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { buildModelTopSpongeRates } from '../physics/modelTopSponge.js';
import { AcousticColumnState, AcousticColumnTendency, predictorRelativeVerticalAcousticStep, predictorVerticalPressureBuoyancyAcceleration } from '../solver/stage4AcousticColumnCpu.js';

function assert(cond:unknown,msg:string):asserts cond{if(!cond)throw new Error(msg)}
function maxAbs(a:ArrayLike<number>):number{let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]!));return m}
function cloneColumn(s:AcousticColumnState):AcousticColumnState{return{rho:s.rho.slice(),rhoTheta:s.rhoTheta.slice(),w:s.w.slice()}}
function zeroRhs(nz:number):AcousticColumnTendency{return{rho:new Float64Array(nz),rhoTheta:new Float64Array(nz),w:new Float64Array(nz+1)}}
interface Test{name:string;fn:()=>void}
const tests:Test[]=[];const test=(name:string,fn:()=>void)=>tests.push({name,fn});

test('Predictor-relative acoustic column preserves hydrostatic rest',()=>{
  const v=buildStretchedVerticalGrid(48,40000,1.4),ref=buildHeldSuarezReference(v),predictor:AcousticColumnState={rho:Float64Array.from(ref.rhoCenter),rhoTheta:Float64Array.from(ref.rhoThetaCenter),w:new Float64Array(v.nz+1)},state=cloneColumn(predictor),rhs=zeroRhs(v.nz);
  rhs.w.set(predictorVerticalPressureBuoyancyAcceleration(v,ref,predictor));
  predictorRelativeVerticalAcousticStep(v,ref,predictor,state,rhs,2.5,.1,buildModelTopSpongeRates(v));
  let dr=0,dx=0;for(let k=0;k<v.nz;k++){dr=Math.max(dr,Math.abs(state.rho[k]!-predictor.rho[k]!));dx=Math.max(dx,Math.abs(state.rhoTheta[k]!-predictor.rhoTheta[k]!));}
  assert(dr<1e-13,`hydrostatic acoustic rho drift=${dr}`);assert(dx<1e-11,`hydrostatic acoustic rhoTheta drift=${dx}`);assert(maxAbs(state.w)<1e-10,`hydrostatic acoustic max|w|=${maxAbs(state.w)}`);
});

test('Predictor-relative acoustic standing mode remains bounded with off-centering',()=>{
  const v=buildStretchedVerticalGrid(32,24000,1.2),ref=buildHeldSuarezReference(v),predictor:AcousticColumnState={rho:Float64Array.from(ref.rhoCenter),rhoTheta:Float64Array.from(ref.rhoThetaCenter),w:new Float64Array(v.nz+1)},state=cloneColumn(predictor),rhs=zeroRhs(v.nz);
  for(let i=1;i<v.nz;i++)state.w[i]=.05*Math.sin(Math.PI*v.zInterface[i]!/v.top);
  const initial=maxAbs(state.w);let peak=initial;
  for(let n=0;n<400;n++){predictorRelativeVerticalAcousticStep(v,ref,predictor,state,rhs,.5,.1);peak=Math.max(peak,maxAbs(state.w));for(let k=0;k<v.nz;k++)assert(state.rho[k]!>0&&state.rhoTheta[k]!>0&&Number.isFinite(state.rho[k]!)&&Number.isFinite(state.rhoTheta[k]!),'standing acoustic state invalid');}
  assert(Number.isFinite(peak)&&peak<initial*4,`standing acoustic mode unbounded: initial=${initial}, peak=${peak}`);
});

test('Predictor-relative implicit Rayleigh damps full predictor plus perturbation w',()=>{
  const v=buildStretchedVerticalGrid(48,40000,1.4),ref=buildHeldSuarezReference(v),rates=buildModelTopSpongeRates(v),predictor:AcousticColumnState={rho:Float64Array.from(ref.rhoCenter),rhoTheta:Float64Array.from(ref.rhoThetaCenter),w:new Float64Array(v.nz+1)},i=v.nz-1,dt=1;
  predictor.w[i]=1;
  const free=cloneColumn(predictor),damped=cloneColumn(predictor),rhs=zeroRhs(v.nz);
  predictorRelativeVerticalAcousticStep(v,ref,predictor,free,rhs,dt,.1);
  predictorRelativeVerticalAcousticStep(v,ref,predictor,damped,rhs,dt,.1,rates);
  const isolated=free.w[i]!/(1+rates[i]!*dt),relativeToIsolated=Math.abs(damped.w[i]!-isolated)/Math.max(Math.abs(isolated),1e-12);
  assert(Math.abs(free.w[i]!-1)<2e-12,`undamped predictor w changed unexpectedly: ${free.w[i]}`);
  assert(damped.w[i]!<free.w[i]!&&damped.w[i]!>0,`full predictor w was not damped: free=${free.w[i]}, damped=${damped.w[i]}`);
  assert(relativeToIsolated<.01,`coupled Rayleigh response too far from isolated implicit limit: damped=${damped.w[i]}, isolated=${isolated}, rel=${relativeToIsolated}`);
});

test('Acoustic reference mass flux reconstructs density update when predictor w and frozen RHS are zero',()=>{
  const v=buildStretchedVerticalGrid(20,16000,1.1),ref=buildHeldSuarezReference(v),predictor:AcousticColumnState={rho:Float64Array.from(ref.rhoCenter),rhoTheta:Float64Array.from(ref.rhoThetaCenter),w:new Float64Array(v.nz+1)},state=cloneColumn(predictor),rhs=zeroRhs(v.nz),dt=.25;
  for(let i=1;i<v.nz;i++)state.w[i]=.03*Math.sin(Math.PI*v.zInterface[i]!/v.top);
  const before=state.rho.slice(),result=predictorRelativeVerticalAcousticStep(v,ref,predictor,state,rhs,dt,0);
  let maxRel=0;
  for(let k=0;k<v.nz;k++){
    const predicted=before[k]!-dt*(result.referenceMassFlux[k+1]!-result.referenceMassFlux[k]!)/v.dz[k]!;
    maxRel=Math.max(maxRel,Math.abs(predicted-state.rho[k]!)/Math.max(Math.abs(state.rho[k]!),1e-12));
  }
  assert(maxRel<2e-13,`acoustic reference-flux continuity mismatch=${maxRel}`);
});

let passed=0;for(const t of tests){try{t.fn();console.log(`PASS ${t.name}`);passed++}catch(e){console.error(`FAIL ${t.name}`);console.error(e);process.exitCode=1}}
console.log(`${passed}/${tests.length} Stage 4 acoustic-column tests passed`);
