declare const process:{exitCode?:number};
import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { acousticDivergenceCoefficientForDt, acousticDivergenceRms, applyAcousticDivergenceDamping, computeAcousticDivergence } from '../physics/acousticDivergenceDamping.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { reconstructEdgeNormalScalarGradient } from '../physics/horizontalGradient.js';
import { buildRotationGeometry } from '../physics/rotation.js';
import { diagnoseAxialAngularMomentum } from '../solver/stage4CirculationDiagnostics.js';
import { cloneState, createHydrostaticState, edge3DIndex } from '../solver/state.js';

type V3=readonly[number,number,number];
function basis(r:V3):{east:V3;north:V3}{
  const xy=Math.hypot(r[0],r[1]);
  const e:V3=xy>1e-14?[-r[1]/xy,r[0]/xy,0]:[0,1,0];
  const n:V3=[-r[2]*e[1],r[2]*e[0],xy];
  return{east:e,north:n};
}
function analytic(r:V3,e:V3,n:V3,k:number):V3{
  const s=1+.05*k;
  // Smooth field with both divergent and rotational parts.  No grid-aligned
  // construction is used, so the diagnostic exercises the full cubed sphere.
  const ue=s*(9+5*r[0]-4*r[1]+3*r[0]*r[2]+2*r[1]*r[2]);
  const vn=s*(-4+6*r[2]+3*r[0]*r[1]-2*r[0]*r[0]+r[1]*r[2]);
  return[ue*e[0]+vn*n[0],ue*e[1]+vn*n[1],ue*e[2]+vn*n[2]];
}
function setFaceMidpointWind(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,s:ReturnType<typeof createHydrostaticState>):void{
  for(let eid=0;eid<h.edgeCount;eid++){
    const edge=h.edges[eid]!,b=basis(edge.midpoint);
    for(let k=0;k<v.nz;k++){
      const w=analytic(edge.midpoint,b.east,b.north,k);
      s.uEdge[edge3DIndex(eid,k,v.nz)]=w[0]*edge.normal[0]+w[1]*edge.normal[1]+w[2]*edge.normal[2];
    }
  }
}
function applyNonorthCandidate(
  h:ReturnType<typeof buildCubedSphere>,
  v:ReturnType<typeof buildStretchedVerticalGrid>,
  ref:ReturnType<typeof buildHeldSuarezReference>,
  s:ReturnType<typeof createHydrostaticState>,
  coefficient:number,
):void{
  const div=computeAcousticDivergence(h,v,ref,s),g=buildRotationGeometry(h),R=EARTH.radius;
  for(let e=0;e<h.edgeCount;e++){
    const L=h.edges[e]!.centerDistanceAngle*R,L2=L*L;
    for(let k=0;k<v.nz;k++){
      const q=edge3DIndex(e,k,v.nz);
      const gradN=reconstructEdgeNormalScalarGradient(h,g,e,c=>div[c*v.nz+k]!);
      s.uEdge[q]=s.uEdge[q]!+coefficient*L2*gradN;
    }
  }
}
interface Row{n:number;oldTorque:number;newTorque:number;oldDivRatio:number;newDivRatio:number;}
function run(n:number):Row{
  const h=buildCubedSphere(n),v=buildStretchedVerticalGrid(4,12000,1.15),ref=buildHeldSuarezReference(v),g=buildRotationGeometry(h),s=createHydrostaticState(h,v,ref);
  setFaceMidpointWind(h,v,s);
  const dt=2.5,coef=acousticDivergenceCoefficientForDt(dt),before=diagnoseAxialAngularMomentum(h,v,s,g),div0=acousticDivergenceRms(h,v,ref,s);
  const oldState=cloneState(s),newState=cloneState(s);
  applyAcousticDivergenceDamping(h,v,ref,oldState,coef);
  applyNonorthCandidate(h,v,ref,newState,coef);
  const oldA=diagnoseAxialAngularMomentum(h,v,oldState,g),newA=diagnoseAxialAngularMomentum(h,v,newState,g);
  const scale=86400/(dt*before.torqueLeverMass);
  return{
    n,
    oldTorque:(oldA.absolute-before.absolute)*scale,
    newTorque:(newA.absolute-before.absolute)*scale,
    oldDivRatio:acousticDivergenceRms(h,v,ref,oldState)/div0,
    newDivRatio:acousticDivergenceRms(h,v,ref,newState)/div0,
  };
}
try{
  const rows=[4,8,16,32].map(run),ratio=(a:number,b:number)=>Math.abs(a)/Math.max(Math.abs(b),1e-30);
  console.log('Stage4 horizontal acoustic divergence-damping AAM diagnostic');
  console.log('horizontal density uniform; AAM equivalent m/s/day; continuum torque target = 0');
  console.log('N\tlegacy torque\tnonorth torque\tlegacy div ratio\tnonorth div ratio');
  for(const r of rows)console.log(`${r.n}\t${r.oldTorque.toExponential(8)}\t${r.newTorque.toExponential(8)}\t${r.oldDivRatio.toFixed(8)}\t${r.newDivRatio.toFixed(8)}`);
  console.log(`legacy refine 8->16=${ratio(rows[1]!.oldTorque,rows[2]!.oldTorque).toFixed(3)} 16->32=${ratio(rows[2]!.oldTorque,rows[3]!.oldTorque).toFixed(3)}`);
  console.log(`nonorth refine 8->16=${ratio(rows[1]!.newTorque,rows[2]!.newTorque).toFixed(3)} 16->32=${ratio(rows[2]!.newTorque,rows[3]!.newTorque).toFixed(3)}`);
  if(rows.some(r=>![r.oldTorque,r.newTorque,r.oldDivRatio,r.newDivRatio].every(Number.isFinite)))throw new Error('non-finite divergence-damping AAM diagnostic');
}catch(e){console.error('FAIL Stage4 divergence-damping AAM diagnostic');console.error(e);process.exitCode=1;}
