declare const process:{exitCode?:number};
import { DRY_AIR, EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { buildRotationGeometry, setAnalyticCellWind } from '../physics/rotation.js';
import { diagnoseAxialAngularMomentum, diagnoseAxialAngularMomentumTendency } from '../solver/stage4CirculationDiagnostics.js';
import { computeStage4FrozenRhs } from '../solver/stage4Rk3SplitCpu.js';
import { computeStage4SlowTendencies } from '../solver/stage4SlowTendenciesCpu.js';
import { cloneState, createHydrostaticState } from '../solver/state.js';

function assert(cond:unknown,msg:string):asserts cond{if(!cond)throw new Error(msg)}
function inversePressure(p:number):number{return DRY_AIR.pRef/DRY_AIR.rd*Math.pow(p/DRY_AIR.pRef,1/DRY_AIR.gamma)}
function accelPerDay(torque:number,leverMass:number):number{return torque/leverMass*86400}

interface Row{n:number;pressure:number;planetaryCoriolis:number;relativeMomentum:number;inviscid:number;}
function run(n:number):Row{
  const h=buildCubedSphere(n),v=buildStretchedVerticalGrid(4,12000,1.15),ref=buildHeldSuarezReference(v),g=buildRotationGeometry(h);
  const zeroRho=new Float64Array(h.cellCount*v.nz),zeroU=new Float64Array(h.edgeCount*v.nz);

  // Smooth, horizontally varying pressure field with zero wind.  On a closed
  // spherical surface pressure is an internal force and its global axial torque
  // should vanish in the continuum.  The m=2 structure deliberately crosses
  // every cubed-sphere seam instead of aligning with a panel.
  const pState=createHydrostaticState(h,v,ref);
  for(let c=0;c<h.cellCount;c++){
    const x=h.cellCenters[c*3]!,y=h.cellCenters[c*3+1]!,z=h.cellCenters[c*3+2]!;
    const shape=0.55*(x*x-y*y)+0.35*x*y+0.20*y*z;
    for(let k=0;k<v.nz;k++){
      const q=c*v.nz+k,p=ref.pCenter[k]!*(1+0.025*shape);
      pState.rhoThetaM[q]=inversePressure(p);
    }
  }
  const pFrozen=computeStage4FrozenRhs(h,v,ref,pState,{momentumTransport:false,coriolis:false,heldSuarez:false},g);
  const pTorque=diagnoseAxialAngularMomentumTendency(h,v,pState,zeroRho,pFrozen.uEdge,g).velocityTorque;
  const pLever=diagnoseAxialAngularMomentum(h,v,pState,g).torqueLeverMass;

  // Smooth generic tangential wind on a horizontally uniform hydrostatic state.
  // It contains zonal + meridional flow and crosses panel seams.  These two
  // continuum identities should independently vanish:
  //   planetary mass redistribution + Coriolis
  //   relative-AAM mass redistribution + material momentum transport
  const s=createHydrostaticState(h,v,ref);
  setAnalyticCellWind(h,g,s,(r,east,north,k)=>{
    const vertical=1+0.08*k;
    const ue=vertical*(11+4*r[0]-3*r[1]+2*r[2]);
    const vn=vertical*(6*r[0]+5*r[1]-4*r[2]+2*r[0]*r[1]);
    return [ue*east[0]+vn*north[0],ue*east[1]+vn*north[1],ue*east[2]+vn*north[2]];
  });
  const continuity=computeStage4FrozenRhs(h,v,ref,s,{momentumTransport:false,coriolis:false,heldSuarez:false},g);
  const momentum=computeStage4SlowTendencies(h,v,ref,s,{momentumTransport:true,coriolis:false,heldSuarez:false},g);
  const coriolis=computeStage4SlowTendencies(h,v,ref,s,{momentumTransport:false,coriolis:true,heldSuarez:false},g);
  const massParts=diagnoseAxialAngularMomentumTendency(h,v,s,continuity.rhoD,zeroU,g);
  const momTorque=diagnoseAxialAngularMomentumTendency(h,v,s,zeroRho,momentum.uEdge,g).velocityTorque;
  const corTorque=diagnoseAxialAngularMomentumTendency(h,v,s,zeroRho,coriolis.uEdge,g).velocityTorque;
  const lever=diagnoseAxialAngularMomentum(h,v,s,g).torqueLeverMass;
  const planetaryCoriolis=massParts.planetaryMassRedistributionTorque+corTorque;
  const relativeMomentum=massParts.relativeMassRedistributionTorque+momTorque;
  const inviscid=planetaryCoriolis+relativeMomentum;

  const vals=[pTorque,planetaryCoriolis,relativeMomentum,inviscid,pLever,lever];
  assert(vals.every(Number.isFinite),`N=${n} AAM convergence diagnostic produced non-finite value`);
  assert(pLever>0&&lever>0,`N=${n} invalid torque lever mass`);
  return{n,pressure:accelPerDay(pTorque,pLever),planetaryCoriolis:accelPerDay(planetaryCoriolis,lever),relativeMomentum:accelPerDay(relativeMomentum,lever),inviscid:accelPerDay(inviscid,lever)};
}

try{
  const rows=[4,8,16,32].map(run);
  console.log('Stage4 analytic AAM spatial-convergence diagnostic (equivalent m/s/day; continuum target = 0)');
  console.log('N\tpressure\tplanetary+coriolis\trelative+momentum\tpair-sum');
  for(const r of rows)console.log(`${r.n}\t${r.pressure.toExponential(6)}\t${r.planetaryCoriolis.toExponential(6)}\t${r.relativeMomentum.toExponential(6)}\t${r.inviscid.toExponential(6)}`);
  for(let i=1;i<rows.length;i++){
    const a=rows[i-1]!,b=rows[i]!;
    const ratio=(x:number,y:number)=>Math.abs(y)>0?Math.abs(x/y):Infinity;
    console.log(`refine ${a.n}->${b.n}: |old/new| pressure=${ratio(a.pressure,b.pressure).toFixed(3)} planetary+coriolis=${ratio(a.planetaryCoriolis,b.planetaryCoriolis).toFixed(3)} relative+momentum=${ratio(a.relativeMomentum,b.relativeMomentum).toFixed(3)}`);
  }
}catch(e){console.error('FAIL Stage4 analytic AAM spatial-convergence diagnostic');console.error(e);process.exitCode=1;}
