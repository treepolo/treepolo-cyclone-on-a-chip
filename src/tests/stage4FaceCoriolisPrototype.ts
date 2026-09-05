declare const process:{exitCode?:number};
import { EARTH } from '../core/constants.js';
import { cross3, dot3, normalize3, type Vec3 } from '../core/math.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { buildRotationGeometry, reconstructCellHorizontalWind, type RotationGeometry } from '../physics/rotation.js';
import { diagnoseAxialAngularMomentum, diagnoseAxialAngularMomentumTendency } from '../solver/stage4CirculationDiagnostics.js';
import { computeStage4FrozenRhs } from '../solver/stage4Rk3SplitCpu.js';
import { computeStage4SlowTendencies } from '../solver/stage4SlowTendenciesCpu.js';
import { createHydrostaticState, edge3DIndex, type DryState } from '../solver/state.js';

function basis(r:Vec3):{east:Vec3;north:Vec3}{
  const xy=Math.hypot(r[0],r[1]);
  const east:Vec3=xy>1e-14?[-r[1]/xy,r[0]/xy,0]:[0,1,0];
  return{east,north:normalize3(cross3(r,east))};
}

function analyticWind(r:Vec3,k:number,nz:number):Vec3{
  const b=basis(r),z=(k+.5)/nz;
  const ue=(13+4*r[0]-2.5*r[1]+1.8*r[2])*(1+.22*z);
  const vn=(5.5*r[0]+4.2*r[1]-3.1*r[2]+1.5*r[0]*r[1])*(1-.12*z);
  return[
    ue*b.east[0]+vn*b.north[0],
    ue*b.east[1]+vn*b.north[1],
    ue*b.east[2]+vn*b.north[2],
  ];
}

function seedFaceNormalState(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,s:DryState):void{
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!;
    for(let k=0;k<v.nz;k++){
      const w=analyticWind(ge.midpoint,k,v.nz);
      s.uEdge[edge3DIndex(e,k,v.nz)]=dot3(w,ge.normal);
    }
  }
}

/** Face-native traditional Coriolis candidate for the C-grid normal-velocity DOF.
 * For an oriented face normal n and t=r x n, the physical normal tendency is
 * a_n = f u_t.  Tangential wind is reconstructed symmetrically from the two
 * adjacent cell winds and sampled at the actual great-circle face midpoint.
 * Diagnostic only; no AAM correction or climate target is applied. */
function faceCoriolis(
  h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,s:DryState,g:RotationGeometry,
):Float64Array{
  const out=new Float64Array(s.uEdge.length);
  const windByK=Array.from({length:v.nz},(_,k)=>reconstructCellHorizontalWind(h,g,s,k));
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,t=normalize3(cross3(ge.midpoint,ge.normal)),f=2*EARTH.omega*ge.midpoint[2],l=ge.leftCell,r=ge.rightCell;
    for(let k=0;k<v.nz;k++){
      const w=windByK[k]!,lo=l*3,ro=r*3;
      const ut=.5*((w[lo]!+w[ro]!)*t[0]+(w[lo+1]!+w[ro+1]!)*t[1]+(w[lo+2]!+w[ro+2]!)*t[2]);
      out[edge3DIndex(e,k,v.nz)]=f*ut;
    }
  }
  return out;
}

function kineticWork(
  h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,s:DryState,g:RotationGeometry,uT:Float64Array,
):number{
  const accelState:DryState={rhoD:s.rhoD,rhoThetaM:s.rhoThetaM,uEdge:uT,wInterface:s.wInterface,time:s.time};
  let work=0,mass=0;
  for(let k=0;k<v.nz;k++){
    const w=reconstructCellHorizontalWind(h,g,s,k),a=reconstructCellHorizontalWind(h,g,accelState,k);
    for(let c=0;c<h.cellCount;c++){
      const q=c*v.nz+k,o=c*3,vol=h.cellAreaUnit[c]!*EARTH.radius*EARTH.radius*v.dz[k]!,m=s.rhoD[q]!*vol;
      work+=m*(w[o]!*a[o]!+w[o+1]!*a[o+1]!+w[o+2]!*a[o+2]!);mass+=m;
    }
  }
  return work/Math.max(mass,1e-30)*86400;
}

function run(n:number){
  const h=buildCubedSphere(n),v=buildStretchedVerticalGrid(6,16000,1.15),ref=buildHeldSuarezReference(v),g=buildRotationGeometry(h),s=createHydrostaticState(h,v,ref);
  seedFaceNormalState(h,v,s);
  const continuity=computeStage4FrozenRhs(h,v,ref,s,{momentumTransport:false,coriolis:false,heldSuarez:false},g);
  const current=computeStage4SlowTendencies(h,v,ref,s,{momentumTransport:false,coriolis:true,heldSuarez:false},g).uEdge;
  const face=faceCoriolis(h,v,s,g),zeroU=new Float64Array(s.uEdge.length),zeroRho=new Float64Array(s.rhoD.length);
  const mass=diagnoseAxialAngularMomentumTendency(h,v,s,continuity.rhoD,zeroU,g).planetaryMassRedistributionTorque;
  const torque=(uT:Float64Array)=>diagnoseAxialAngularMomentumTendency(h,v,s,zeroRho,uT,g).velocityTorque;
  const lever=diagnoseAxialAngularMomentum(h,v,s,g).torqueLeverMass,scale=86400/lever;
  return{n,current:(mass+torque(current))*scale,face:(mass+torque(face))*scale,currentWork:kineticWork(h,v,s,g,current),faceWork:kineticWork(h,v,s,g,face)};
}

try{
  const rows=[4,8,16,32].map(run),ratio=(a:number,b:number)=>Math.abs(a)/Math.max(Math.abs(b),1e-30);
  console.log('Stage4 face-native Coriolis prototype (P-mass+C residual m/s/day; KE work m^2/s^2/day; diagnostic only)');
  console.log('N\tcurrent P+C\tface P+C\tcurrent work\tface work');
  for(const r of rows)console.log(`${r.n}\t${r.current.toExponential(8)}\t${r.face.toExponential(8)}\t${r.currentWork.toExponential(8)}\t${r.faceWork.toExponential(8)}`);
  console.log(`current refine 8->16=${ratio(rows[1]!.current,rows[2]!.current).toFixed(3)} 16->32=${ratio(rows[2]!.current,rows[3]!.current).toFixed(3)}`);
  console.log(`face refine 8->16=${ratio(rows[1]!.face,rows[2]!.face).toFixed(3)} 16->32=${ratio(rows[2]!.face,rows[3]!.face).toFixed(3)}`);
  if(rows.some(r=>![r.current,r.face,r.currentWork,r.faceWork].every(Number.isFinite)))throw new Error('non-finite face-Coriolis prototype');
}catch(e){console.error('FAIL Stage4 face-native Coriolis prototype');console.error(e);process.exitCode=1;}
