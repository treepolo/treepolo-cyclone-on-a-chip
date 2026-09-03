declare const process:{exitCode?:number};
import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { buildRotationGeometry, setAnalyticCellWind } from '../physics/rotation.js';
import { computeStage4SlowTendencies } from '../solver/stage4SlowTendenciesCpu.js';
import { cloneState, createHydrostaticState, w3DIndex } from '../solver/state.js';

function assert(cond:unknown,msg:string):asserts cond{if(!cond)throw new Error(msg)}
function maxAbs(a:ArrayLike<number>):number{let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]!));return m}
function same(a:ArrayLike<number>,b:ArrayLike<number>):boolean{if(a.length!==b.length)return false;for(let i=0;i<a.length;i++)if(a[i]!==b[i])return false;return true}
interface Test{name:string;fn:()=>void}
const tests:Test[]=[];const test=(name:string,fn:()=>void)=>tests.push({name,fn});

test('Stage4 slow RHS leaves resting hydrostatic atmosphere exactly at rest',()=>{
  const h=buildCubedSphere(4),v=buildStretchedVerticalGrid(16,30000,1.4),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref);
  const t=computeStage4SlowTendencies(h,v,ref,s,{heldSuarez:false,momentumTransport:true,coriolis:true});
  assert(maxAbs(t.rhoD)===0,`hydrostatic slow rho tendency=${maxAbs(t.rhoD)}`);
  assert(maxAbs(t.rhoThetaM)===0,`hydrostatic slow rhoTheta tendency=${maxAbs(t.rhoThetaM)}`);
  assert(maxAbs(t.uEdge)===0,`hydrostatic slow u tendency=${maxAbs(t.uEdge)}`);
  assert(maxAbs(t.wInterface)===0,`hydrostatic slow w tendency=${maxAbs(t.wInterface)}`);
});

test('Stage4 slow RHS is pure and does not mutate predictor state',()=>{
  const h=buildCubedSphere(2),v=buildStretchedVerticalGrid(8,12000,1.2),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref),before=cloneState(s);
  for(let q=0;q<s.uEdge.length;q++)s.uEdge[q]=.8*Math.sin((q+1)*.37);
  for(let c=0;c<h.cellCount;c++)for(let i=1;i<v.nz;i++)s.wInterface[w3DIndex(c,i,v.nz)]=.05*Math.sin((c+1)*.21+i*.7);
  const frozen=cloneState(s);
  computeStage4SlowTendencies(h,v,ref,s,{heldSuarez:true,momentumTransport:true,coriolis:true});
  assert(same(s.rhoD,frozen.rhoD)&&same(s.rhoThetaM,frozen.rhoThetaM)&&same(s.uEdge,frozen.uEdge)&&same(s.wInterface,frozen.wInterface),'slow RHS mutated predictor state');
  assert(same(before.rhoD,s.rhoD),'test unexpectedly changed rho before RHS evaluation');
});

test('Stage4 slow scalar advection conserves global dry mass',()=>{
  const h=buildCubedSphere(4),v=buildStretchedVerticalGrid(12,20000,1.3),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref);
  for(let q=0;q<s.rhoD.length;q++){const f=1+2e-3*Math.sin((q+1)*.173);s.rhoD[q]*=f;s.rhoThetaM[q]*=f;}
  for(let q=0;q<s.uEdge.length;q++)s.uEdge[q]=4*Math.sin((q+1)*.113);
  for(let c=0;c<h.cellCount;c++)for(let i=1;i<v.nz;i++)s.wInterface[w3DIndex(c,i,v.nz)]=.2*Math.sin((c+1)*.31+i*.47);
  const t=computeStage4SlowTendencies(h,v,ref,s,{heldSuarez:false,momentumTransport:false,coriolis:false});
  let sum=0,scale=0;
  for(let c=0;c<h.cellCount;c++)for(let k=0;k<v.nz;k++){
    const q=c*v.nz+k,vol=h.cellAreaUnit[c]!*EARTH.radius*EARTH.radius*v.dz[k]!,x=t.rhoD[q]!*vol;sum+=x;scale+=Math.abs(x);
  }
  assert(Math.abs(sum)<=Math.max(1e-6,scale*2e-14),`global slow mass tendency not conservative: sum=${sum}, scale=${scale}`);
});

test('Vertical mass transport preserves a horizontally moving column that is uniform with height',()=>{
  const h=buildCubedSphere(4),v=buildStretchedVerticalGrid(10,15000,1.2),ref=buildHeldSuarezReference(v),g=buildRotationGeometry(h),calm=createHydrostaticState(h,v,ref);
  setAnalyticCellWind(h,g,calm,(r)=>[-18*r[1],18*r[0],0]);
  const moving=cloneState(calm);
  for(let c=0;c<h.cellCount;c++)for(let i=1;i<v.nz;i++)moving.wInterface[w3DIndex(c,i,v.nz)]=.25*Math.sin(Math.PI*v.zInterface[i]!/v.top)*(1+.1*Math.sin((c+1)*.29));
  const a=computeStage4SlowTendencies(h,v,ref,calm,{heldSuarez:false,momentumTransport:true,coriolis:false},g);
  const b=computeStage4SlowTendencies(h,v,ref,moving,{heldSuarez:false,momentumTransport:true,coriolis:false},g);
  let diff=0,scale=0;
  for(let q=0;q<a.uEdge.length;q++){diff=Math.max(diff,Math.abs(b.uEdge[q]!-a.uEdge[q]!));scale=Math.max(scale,Math.abs(a.uEdge[q]!),Math.abs(b.uEdge[q]!));}
  assert(diff<2e-12*Math.max(1,scale),`vertical carrier changed height-uniform horizontal wind: diff=${diff}, scale=${scale}`);
});

let passed=0;for(const t of tests){try{t.fn();console.log(`PASS ${t.name}`);passed++}catch(e){console.error(`FAIL ${t.name}`);console.error(e);process.exitCode=1}}
console.log(`${passed}/${tests.length} Stage 4 slow-tendency tests passed`);
