declare const process:{exitCode?:number};
import { DRY_AIR } from '../core/constants.js';
import { buildCubedSphere,type CubedSphereGrid } from '../grid/cubedSphere.js';
import { buildModifiedCubedSphere } from '../grid/modifiedCubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { buildRotationGeometry,setAnalyticCellWind } from '../physics/rotation.js';
import { diagnoseAxialAngularMomentum,diagnoseAxialAngularMomentumTendency } from '../solver/stage4CirculationDiagnostics.js';
import { computeStage4FrozenRhs } from '../solver/stage4Rk3SplitCpu.js';
import { computeStage4SlowTendencies } from '../solver/stage4SlowTendenciesCpu.js';
import { createHydrostaticState } from '../solver/state.js';

function assert(cond:unknown,msg:string):asserts cond{if(!cond)throw new Error(msg);}
function inversePressure(p:number):number{return DRY_AIR.pRef/DRY_AIR.rd*Math.pow(p/DRY_AIR.pRef,1/DRY_AIR.gamma);}
function accelPerDay(torque:number,leverMass:number):number{return torque/leverMass*86400;}
interface Terms{pressure:number;pc:number;rm:number;sum:number;}
function evaluate(h:CubedSphereGrid):Terms{
  const v=buildStretchedVerticalGrid(4,12000,1.15),ref=buildHeldSuarezReference(v),g=buildRotationGeometry(h),zeroRho=new Float64Array(h.cellCount*v.nz),zeroU=new Float64Array(h.edgeCount*v.nz);
  const pState=createHydrostaticState(h,v,ref);
  for(let c=0;c<h.cellCount;c++){const x=h.cellCenters[c*3]!,y=h.cellCenters[c*3+1]!,z=h.cellCenters[c*3+2]!,shape=.55*(x*x-y*y)+.35*x*y+.20*y*z;for(let k=0;k<v.nz;k++){const q=c*v.nz+k,p=ref.pCenter[k]!*(1+.025*shape);pState.rhoThetaM[q]=inversePressure(p);}}
  const pFrozen=computeStage4FrozenRhs(h,v,ref,pState,{momentumTransport:false,coriolis:false,heldSuarez:false},g),pTorque=diagnoseAxialAngularMomentumTendency(h,v,pState,zeroRho,pFrozen.uEdge,g).velocityTorque,pLever=diagnoseAxialAngularMomentum(h,v,pState,g).torqueLeverMass;
  const s=createHydrostaticState(h,v,ref);setAnalyticCellWind(h,g,s,(r,east,north,k)=>{const vertical=1+.08*k,ue=vertical*(11+4*r[0]-3*r[1]+2*r[2]),vn=vertical*(6*r[0]+5*r[1]-4*r[2]+2*r[0]*r[1]);return[ue*east[0]+vn*north[0],ue*east[1]+vn*north[1],ue*east[2]+vn*north[2]];});
  const continuity=computeStage4FrozenRhs(h,v,ref,s,{momentumTransport:false,coriolis:false,heldSuarez:false},g),momentum=computeStage4SlowTendencies(h,v,ref,s,{momentumTransport:true,coriolis:false,heldSuarez:false},g),coriolis=computeStage4SlowTendencies(h,v,ref,s,{momentumTransport:false,coriolis:true,heldSuarez:false},g),mass=diagnoseAxialAngularMomentumTendency(h,v,s,continuity.rhoD,zeroU,g),mom=diagnoseAxialAngularMomentumTendency(h,v,s,zeroRho,momentum.uEdge,g).velocityTorque,cor=diagnoseAxialAngularMomentumTendency(h,v,s,zeroRho,coriolis.uEdge,g).velocityTorque,lever=diagnoseAxialAngularMomentum(h,v,s,g).torqueLeverMass;
  const pc=mass.planetaryMassRedistributionTorque+cor,rm=mass.relativeMassRedistributionTorque+mom,sum=pc+rm;
  assert([pTorque,pLever,pc,rm,sum,lever].every(Number.isFinite),'non-finite modified-grid AAM diagnostic');assert(pLever>0&&lever>0,'invalid AAM lever mass');
  return{pressure:accelPerDay(pTorque,pLever),pc:accelPerDay(pc,lever),rm:accelPerDay(rm,lever),sum:accelPerDay(sum,lever)};
}
function run(n:number){const raw=buildCubedSphere(n),mod=buildModifiedCubedSphere(n);assert(Math.abs(mod.totalAreaUnit-4*Math.PI)<2e-10,`N${n} modified area closure failed`);for(const e of mod.edges){const lo=e.leftCell*3,ro=e.rightCell*3,dx=mod.cellCenters[ro]!-mod.cellCenters[lo]!,dy=mod.cellCenters[ro+1]!-mod.cellCenters[lo+1]!,dz=mod.cellCenters[ro+2]!-mod.cellCenters[lo+2]!;assert(e.normal[0]*dx+e.normal[1]*dy+e.normal[2]*dz>0,'modified edge normal orientation failed');}return{n,raw:evaluate(raw),mod:evaluate(mod)};}
try{
  const rows=[4,8,16,32].map(run),ratio=(a:number,b:number)=>Math.abs(a)/Math.max(Math.abs(b),1e-30);
  console.log('Stage4 raw-vs-modified cubed-sphere AAM diagnostic (equivalent m/s/day)');
  console.log('N\traw P\tmod P\traw P+C\tmod P+C\traw R+M\tmod R+M\traw sum\tmod sum');
  for(const r of rows)console.log(`${r.n}\t${r.raw.pressure.toExponential(6)}\t${r.mod.pressure.toExponential(6)}\t${r.raw.pc.toExponential(6)}\t${r.mod.pc.toExponential(6)}\t${r.raw.rm.toExponential(6)}\t${r.mod.rm.toExponential(6)}\t${r.raw.sum.toExponential(6)}\t${r.mod.sum.toExponential(6)}`);
  const a=rows[1]!,b=rows[2]!,c=rows[3]!;console.log(`modified refine 8->16: P=${ratio(a.mod.pressure,b.mod.pressure).toFixed(3)} P+C=${ratio(a.mod.pc,b.mod.pc).toFixed(3)} R+M=${ratio(a.mod.rm,b.mod.rm).toFixed(3)} sum=${ratio(a.mod.sum,b.mod.sum).toFixed(3)}`);console.log(`modified refine 16->32: P=${ratio(b.mod.pressure,c.mod.pressure).toFixed(3)} P+C=${ratio(b.mod.pc,c.mod.pc).toFixed(3)} R+M=${ratio(b.mod.rm,c.mod.rm).toFixed(3)} sum=${ratio(b.mod.sum,c.mod.sum).toFixed(3)}`);
}catch(e){console.error('FAIL Stage4 modified-grid AAM diagnostic');console.error(e);process.exitCode=1;}
