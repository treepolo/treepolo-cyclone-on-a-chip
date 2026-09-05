declare const process:{exitCode?:number};
import { DRY_AIR } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { buildRotationGeometry, setAnalyticCellWind } from '../physics/rotation.js';
import { diagnoseAxialAngularMomentum, diagnoseAxialAngularMomentumTendency } from '../solver/stage4CirculationDiagnostics.js';
import { computeStage4FrozenRhs } from '../solver/stage4Rk3SplitCpu.js';
import { computeStage4SlowTendencies } from '../solver/stage4SlowTendenciesCpu.js';
import { createHydrostaticState } from '../solver/state.js';

function assert(cond:unknown,msg:string):asserts cond{if(!cond)throw new Error(msg)}
function inversePressure(p:number):number{return DRY_AIR.pRef/DRY_AIR.rd*Math.pow(p/DRY_AIR.pRef,1/DRY_AIR.gamma)}
function accelPerDay(torque:number,leverMass:number):number{return torque/leverMass*86400}

interface Row{n:number;pressure:number;planetaryCoriolis:number;relativeMomentum:number;inviscid:number;}
function run(n:number):Row{
  const h=buildCubedSphere(n),v=buildStretchedVerticalGrid(4,12000,1.15),ref=buildHeldSuarezReference(v),g=buildRotationGeometry(h);
  const zeroRho=new Float64Array(h.cellCount*v.nz),zeroU=new Float64Array(h.edgeCount*v.nz);

  // Production geometry invariant: uEdge is the normal component through the
  // actual great-circle face, oriented left -> right.
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,nn=Math.hypot(...ge.normal),d0=ge.normal[0]*ge.p0[0]+ge.normal[1]*ge.p0[1]+ge.normal[2]*ge.p0[2],d1=ge.normal[0]*ge.p1[0]+ge.normal[1]*ge.p1[1]+ge.normal[2]*ge.p1[2];
    const lo=ge.leftCell*3,ro=ge.rightCell*3,dx=h.cellCenters[ro]!-h.cellCenters[lo]!,dy=h.cellCenters[ro+1]!-h.cellCenters[lo+1]!,dz=h.cellCenters[ro+2]!-h.cellCenters[lo+2]!;
    assert(Math.abs(nn-1)<2e-12&&Math.abs(d0)<2e-12&&Math.abs(d1)<2e-12&&ge.normal[0]*dx+ge.normal[1]*dy+ge.normal[2]*dz>0,`N=${n} edge ${e} is not an oriented unit face conormal`);
  }

  // Smooth horizontally varying pressure field crossing every panel seam.
  const pState=createHydrostaticState(h,v,ref);
  for(let c=0;c<h.cellCount;c++){
    const x=h.cellCenters[c*3]!,y=h.cellCenters[c*3+1]!,z=h.cellCenters[c*3+2]!,shape=0.55*(x*x-y*y)+0.35*x*y+0.20*y*z;
    for(let k=0;k<v.nz;k++){const q=c*v.nz+k,p=ref.pCenter[k]!*(1+0.025*shape);pState.rhoThetaM[q]=inversePressure(p);}
  }
  const pFrozen=computeStage4FrozenRhs(h,v,ref,pState,{momentumTransport:false,coriolis:false,heldSuarez:false},g);
  const pTorque=diagnoseAxialAngularMomentumTendency(h,v,pState,zeroRho,pFrozen.uEdge,g).velocityTorque,pLever=diagnoseAxialAngularMomentum(h,v,pState,g).torqueLeverMass;

  // Smooth generic tangential wind.  In the continuum these paired identities
  // vanish independently:
  //   planetary mass redistribution + Coriolis
  //   relative-AAM mass redistribution + material momentum transport
  const s=createHydrostaticState(h,v,ref);
  setAnalyticCellWind(h,g,s,(r,east,north,k)=>{
    const vertical=1+0.08*k,ue=vertical*(11+4*r[0]-3*r[1]+2*r[2]),vn=vertical*(6*r[0]+5*r[1]-4*r[2]+2*r[0]*r[1]);
    return[ue*east[0]+vn*north[0],ue*east[1]+vn*north[1],ue*east[2]+vn*north[2]];
  });
  const continuity=computeStage4FrozenRhs(h,v,ref,s,{momentumTransport:false,coriolis:false,heldSuarez:false},g),momentum=computeStage4SlowTendencies(h,v,ref,s,{momentumTransport:true,coriolis:false,heldSuarez:false},g),coriolis=computeStage4SlowTendencies(h,v,ref,s,{momentumTransport:false,coriolis:true,heldSuarez:false},g);
  const massParts=diagnoseAxialAngularMomentumTendency(h,v,s,continuity.rhoD,zeroU,g),momTorque=diagnoseAxialAngularMomentumTendency(h,v,s,zeroRho,momentum.uEdge,g).velocityTorque,corTorque=diagnoseAxialAngularMomentumTendency(h,v,s,zeroRho,coriolis.uEdge,g).velocityTorque,lever=diagnoseAxialAngularMomentum(h,v,s,g).torqueLeverMass;
  const planetaryCoriolis=massParts.planetaryMassRedistributionTorque+corTorque,relativeMomentum=massParts.relativeMassRedistributionTorque+momTorque,inviscid=planetaryCoriolis+relativeMomentum;
  const vals=[pTorque,planetaryCoriolis,relativeMomentum,inviscid,pLever,lever];
  assert(vals.every(Number.isFinite),`N=${n} AAM convergence diagnostic produced non-finite value`);assert(pLever>0&&lever>0,`N=${n} invalid torque lever mass`);
  return{n,pressure:accelPerDay(pTorque,pLever),planetaryCoriolis:accelPerDay(planetaryCoriolis,lever),relativeMomentum:accelPerDay(relativeMomentum,lever),inviscid:accelPerDay(inviscid,lever)};
}

try{
  const rows=[4,8,16,32].map(run);
  console.log('Stage4 production AAM spatial-convergence regression (equivalent m/s/day; continuum target = 0)');
  console.log('N\tpressure\tplanetary+coriolis\trelative+momentum\tpair-sum');
  for(const r of rows)console.log(`${r.n}\t${r.pressure.toExponential(6)}\t${r.planetaryCoriolis.toExponential(6)}\t${r.relativeMomentum.toExponential(6)}\t${r.inviscid.toExponential(6)}`);
  const n8=rows[1]!,n16=rows[2]!,n32=rows[3]!,ratio=(a:number,b:number)=>Math.abs(a)/Math.max(Math.abs(b),1e-30);
  const pc816=ratio(n8.planetaryCoriolis,n16.planetaryCoriolis),pc1632=ratio(n16.planetaryCoriolis,n32.planetaryCoriolis),rm816=ratio(n8.relativeMomentum,n16.relativeMomentum),rm1632=ratio(n16.relativeMomentum,n32.relativeMomentum);
  console.log(`refine 8->16: planetary+coriolis=${pc816.toFixed(3)} relative+momentum=${rm816.toFixed(3)}`);
  console.log(`refine 16->32: planetary+coriolis=${pc1632.toFixed(3)} relative+momentum=${rm1632.toFixed(3)}`);
  assert(pc816>3&&pc1632>3,`planetary+Coriolis lost near-second-order convergence: ${pc816}, ${pc1632}`);
  assert(rm816>1.5&&rm1632>1.5,`material momentum pair lost first-order convergence: ${rm816}, ${rm1632}`);
  assert(Math.abs(n32.pressure)<0.02,`smooth closed-sphere pressure torque too large at N32: ${n32.pressure} m/s/day`);
}catch(e){console.error('FAIL Stage4 production AAM spatial-convergence regression');console.error(e);process.exitCode=1;}
