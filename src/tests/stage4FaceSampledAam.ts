declare const process:{exitCode?:number};
import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { buildRotationGeometry, setAnalyticCellWind } from '../physics/rotation.js';
import { diagnoseAxialAngularMomentum, diagnoseAxialAngularMomentumTendency } from '../solver/stage4CirculationDiagnostics.js';
import { computeStage4FrozenRhs } from '../solver/stage4Rk3SplitCpu.js';
import { computeStage4SlowTendencies } from '../solver/stage4SlowTendenciesCpu.js';
import { createHydrostaticState, edge3DIndex } from '../solver/state.js';

type V3=readonly[number,number,number];
function basis(r:V3):{east:V3;north:V3}{const xy=Math.hypot(r[0],r[1]),e:V3=xy>1e-14?[-r[1]/xy,r[0]/xy,0]:[0,1,0],n:V3=[-r[2]*e[1],r[2]*e[0],xy];return{east:e,north:n};}
function analytic(r:V3,e:V3,n:V3,k:number):V3{const vertical=1+0.08*k,ue=vertical*(11+4*r[0]-3*r[1]+2*r[2]),vn=vertical*(6*r[0]+5*r[1]-4*r[2]+2*r[0]*r[1]);return[ue*e[0]+vn*n[0],ue*e[1]+vn*n[1],ue*e[2]+vn*n[2]];}
function setFaceMidpoint(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,s:ReturnType<typeof createHydrostaticState>):void{
  for(let eid=0;eid<h.edgeCount;eid++){const edge=h.edges[eid]!,b=basis(edge.midpoint);for(let k=0;k<v.nz;k++){const w=analytic(edge.midpoint,b.east,b.north,k);s.uEdge[edge3DIndex(eid,k,v.nz)]=w[0]*edge.normal[0]+w[1]*edge.normal[1]+w[2]*edge.normal[2];}}
}
function pair(n:number,mode:'projected'|'face-midpoint'):{pc:number;rm:number}{
  const h=buildCubedSphere(n),v=buildStretchedVerticalGrid(4,12000,1.15),ref=buildHeldSuarezReference(v),g=buildRotationGeometry(h),s=createHydrostaticState(h,v,ref);
  if(mode==='projected')setAnalyticCellWind(h,g,s,(r,e,n,k)=>analytic(r,e,n,k));else setFaceMidpoint(h,v,s);
  const continuity=computeStage4FrozenRhs(h,v,ref,s,{momentumTransport:false,coriolis:false,heldSuarez:false},g),mom=computeStage4SlowTendencies(h,v,ref,s,{momentumTransport:true,coriolis:false,heldSuarez:false},g),cor=computeStage4SlowTendencies(h,v,ref,s,{momentumTransport:false,coriolis:true,heldSuarez:false},g),zeroU=new Float64Array(s.uEdge.length),zeroRho=new Float64Array(s.rhoD.length),mass=diagnoseAxialAngularMomentumTendency(h,v,s,continuity.rhoD,zeroU,g),mt=diagnoseAxialAngularMomentumTendency(h,v,s,zeroRho,mom.uEdge,g).velocityTorque,ct=diagnoseAxialAngularMomentumTendency(h,v,s,zeroRho,cor.uEdge,g).velocityTorque,lever=diagnoseAxialAngularMomentum(h,v,s,g).torqueLeverMass;
  return{pc:(mass.planetaryMassRedistributionTorque+ct)/lever*86400,rm:(mass.relativeMassRedistributionTorque+mt)/lever*86400};
}
try{
  const ns=[4,8,16,32],p=ns.map(n=>pair(n,'projected')),f=ns.map(n=>pair(n,'face-midpoint')),ratio=(a:number,b:number)=>Math.abs(a)/Math.max(Math.abs(b),1e-30);
  console.log('Stage4 AAM pair convergence: projected-cell analytic initializer vs direct face-midpoint DOFs (m/s/day)');
  console.log('N\tprojected P+C\tface P+C\tprojected R+M\tface R+M');
  for(let i=0;i<ns.length;i++)console.log(`${ns[i]}\t${p[i]!.pc.toExponential(8)}\t${f[i]!.pc.toExponential(8)}\t${p[i]!.rm.toExponential(8)}\t${f[i]!.rm.toExponential(8)}`);
  console.log(`face P+C refine 8->16=${ratio(f[1]!.pc,f[2]!.pc).toFixed(3)} 16->32=${ratio(f[2]!.pc,f[3]!.pc).toFixed(3)}`);
  console.log(`face R+M refine 8->16=${ratio(f[1]!.rm,f[2]!.rm).toFixed(3)} 16->32=${ratio(f[2]!.rm,f[3]!.rm).toFixed(3)}`);
  if([...p,...f].some(x=>!Number.isFinite(x.pc)||!Number.isFinite(x.rm)))throw new Error('non-finite face-sampled AAM diagnostic');
}catch(e){console.error('FAIL Stage4 face-sampled AAM diagnostic');console.error(e);process.exitCode=1;}
