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
import { cell3DIndex, createHydrostaticState, edge3DIndex, type DryState } from '../solver/state.js';

function basis(r:Vec3):{east:Vec3;north:Vec3}{const xy=Math.hypot(r[0],r[1]),east:Vec3=xy>1e-14?[-r[1]/xy,r[0]/xy,0]:[0,1,0];return{east,north:normalize3(cross3(r,east))};}
function analyticWind(r:Vec3,k:number,nz:number):Vec3{const b=basis(r),z=(k+.5)/nz,ue=(13+4*r[0]-2.5*r[1]+1.8*r[2])*(1+.22*z),vn=(5.5*r[0]+4.2*r[1]-3.1*r[2]+1.5*r[0]*r[1])*(1-.12*z);return[ue*b.east[0]+vn*b.north[0],ue*b.east[1]+vn*b.north[1],ue*b.east[2]+vn*b.north[2]];}
function seedFaceNormalState(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,s:DryState,varyDensity:boolean):void{
  for(let e=0;e<h.edgeCount;e++){const ge=h.edges[e]!;for(let k=0;k<v.nz;k++){const w=analyticWind(ge.midpoint,k,v.nz);s.uEdge[edge3DIndex(e,k,v.nz)]=dot3(w,ge.normal);}}
  if(varyDensity)for(let c=0;c<h.cellCount;c++){const x=h.cellCenters[c*3]!,y=h.cellCenters[c*3+1]!,z=h.cellCenters[c*3+2]!,fac=1+.22*(.5*x-.3*y+.25*z+.18*x*y);for(let k=0;k<v.nz;k++){const q=cell3DIndex(c,k,v.nz);s.rhoD[q]=s.rhoD[q]!*fac;s.rhoThetaM[q]=s.rhoThetaM[q]!*fac;}}
}

/** Face-native traditional Coriolis candidate for the C-grid normal-velocity DOF. */
function faceCoriolis(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,s:DryState,g:RotationGeometry):Float64Array{
  const out=new Float64Array(s.uEdge.length),windByK=Array.from({length:v.nz},(_,k)=>reconstructCellHorizontalWind(h,g,s,k));
  for(let e=0;e<h.edgeCount;e++){const ge=h.edges[e]!,t=normalize3(cross3(ge.midpoint,ge.normal)),f=2*EARTH.omega*ge.midpoint[2],l=ge.leftCell,r=ge.rightCell;for(let k=0;k<v.nz;k++){const w=windByK[k]!,lo=l*3,ro=r*3,ut=.5*((w[lo]!+w[ro]!)*t[0]+(w[lo+1]!+w[ro+1]!)*t[1]+(w[lo+2]!+w[ro+2]!)*t[2]);out[edge3DIndex(e,k,v.nz)]=f*ut;}}
  return out;
}

/**
 * Local lumped Galerkin edge-space Coriolis. R is the production edge->cell
 * least-squares reconstruction and J is the exact local 90-degree rotation.
 * a=D^-1 R^T M f J R u, with diagonal D=diag(R^T M R). Therefore the
 * edge-metric work u^T D a is identically zero (up to roundoff), without any
 * global solve, torque correction, or climate target.
 */
function skewCoriolis(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,s:DryState,g:RotationGeometry,massWeighted:boolean):{a:Float64Array;edgeWork:number}{
  const num=new Float64Array(s.uEdge.length),den=new Float64Array(s.uEdge.length),windByK=Array.from({length:v.nz},(_,k)=>reconstructCellHorizontalWind(h,g,s,k));
  for(let k=0;k<v.nz;k++)for(let c=0;c<h.cellCount;c++){
    const o=c*3,w=windByK[k]!,ex=g.east[o]!,ey=g.east[o+1]!,ez=g.east[o+2]!,nx=g.north[o]!,ny=g.north[o+1]!,nzv=g.north[o+2]!,ue=w[o]!*ex+w[o+1]!*ey+w[o+2]!*ez,vn=w[o]!*nx+w[o+1]!*ny+w[o+2]!*nzv,f=2*EARTH.omega*g.radial[o+2]!,q=cell3DIndex(c,k,v.nz),base=h.cellAreaUnit[c]!*v.dz[k]!,mc=massWeighted?base*s.rhoD[q]!:base;
    for(let slot=0;slot<4;slot++){const eid=h.cellEdges[c*4+slot]!,re=g.reconstruction[(c*4+slot)*2]!,rn=g.reconstruction[(c*4+slot)*2+1]!,qe=edge3DIndex(eid,k,v.nz);num[qe]=num[qe]!+mc*f*(re*vn-rn*ue);den[qe]=den[qe]!+mc*(re*re+rn*rn);}
  }
  const a=new Float64Array(num.length);let work=0,norm=0;
  for(let i=0;i<a.length;i++){a[i]=num[i]!/Math.max(den[i]!,1e-30);work+=den[i]!*s.uEdge[i]!*a[i]!;norm+=den[i]!*s.uEdge[i]!*s.uEdge[i]!;}
  return{a,edgeWork:work/Math.max(norm,1e-30)*86400};
}

function cellKineticWork(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,s:DryState,g:RotationGeometry,uT:Float64Array):number{
  const accelState:DryState={rhoD:s.rhoD,rhoThetaM:s.rhoThetaM,uEdge:uT,wInterface:s.wInterface,time:s.time};let work=0,mass=0;
  for(let k=0;k<v.nz;k++){const w=reconstructCellHorizontalWind(h,g,s,k),a=reconstructCellHorizontalWind(h,g,accelState,k);for(let c=0;c<h.cellCount;c++){const q=cell3DIndex(c,k,v.nz),o=c*3,vol=h.cellAreaUnit[c]!*EARTH.radius*EARTH.radius*v.dz[k]!,m=s.rhoD[q]!*vol;work+=m*(w[o]!*a[o]!+w[o+1]!*a[o+1]!+w[o+2]!*a[o+2]!);mass+=m;}}
  return work/Math.max(mass,1e-30)*86400;
}

function run(n:number,varyDensity:boolean){
  const h=buildCubedSphere(n),v=buildStretchedVerticalGrid(6,16000,1.15),ref=buildHeldSuarezReference(v),g=buildRotationGeometry(h),s=createHydrostaticState(h,v,ref);seedFaceNormalState(h,v,s,varyDensity);
  const continuity=computeStage4FrozenRhs(h,v,ref,s,{momentumTransport:false,coriolis:false,heldSuarez:false},g),current=computeStage4SlowTendencies(h,v,ref,s,{momentumTransport:false,coriolis:true,heldSuarez:false},g).uEdge,face=faceCoriolis(h,v,s,g),skewG=skewCoriolis(h,v,s,g,false),skewM=skewCoriolis(h,v,s,g,true),zeroU=new Float64Array(s.uEdge.length),zeroRho=new Float64Array(s.rhoD.length);
  const mass=diagnoseAxialAngularMomentumTendency(h,v,s,continuity.rhoD,zeroU,g).planetaryMassRedistributionTorque,torque=(uT:Float64Array)=>diagnoseAxialAngularMomentumTendency(h,v,s,zeroRho,uT,g).velocityTorque,lever=diagnoseAxialAngularMomentum(h,v,s,g).torqueLeverMass,scale=86400/lever;
  return{n,current:(mass+torque(current))*scale,face:(mass+torque(face))*scale,skewG:(mass+torque(skewG.a))*scale,skewM:(mass+torque(skewM.a))*scale,currentWork:cellKineticWork(h,v,s,g,current),faceWork:cellKineticWork(h,v,s,g,face),skewGCellWork:cellKineticWork(h,v,s,g,skewG.a),skewMCellWork:cellKineticWork(h,v,s,g,skewM.a),skewGEdgeWork:skewG.edgeWork,skewMEdgeWork:skewM.edgeWork};
}

try{
  const ratio=(a:number,b:number)=>Math.abs(a)/Math.max(Math.abs(b),1e-30);
  for(const varying of [false,true]){
    const rows=[4,8,16,32].map(n=>run(n,varying));
    console.log(`Stage4 Coriolis prototypes; density ${varying?'VARIES':'uniform'} (P+C m/s/day; work m2/s2/day)`);
    console.log('N\tcurrent P+C\tface P+C\tskewG P+C\tskewM P+C\tcurrent cellW\tface cellW\tskewG cellW\tskewM cellW\tskewG edgeW\tskewM edgeW');
    for(const r of rows)console.log(`${r.n}\t${r.current.toExponential(7)}\t${r.face.toExponential(7)}\t${r.skewG.toExponential(7)}\t${r.skewM.toExponential(7)}\t${r.currentWork.toExponential(7)}\t${r.faceWork.toExponential(7)}\t${r.skewGCellWork.toExponential(7)}\t${r.skewMCellWork.toExponential(7)}\t${r.skewGEdgeWork.toExponential(3)}\t${r.skewMEdgeWork.toExponential(3)}`);
    console.log(`skewG refine 8->16=${ratio(rows[1]!.skewG,rows[2]!.skewG).toFixed(3)} 16->32=${ratio(rows[2]!.skewG,rows[3]!.skewG).toFixed(3)}; skewM=${ratio(rows[1]!.skewM,rows[2]!.skewM).toFixed(3)},${ratio(rows[2]!.skewM,rows[3]!.skewM).toFixed(3)}`);
    if(rows.some(r=>Object.values(r).some(x=>typeof x==='number'&&!Number.isFinite(x))))throw new Error('non-finite Coriolis prototype');
  }
}catch(e){console.error('FAIL Stage4 Coriolis prototype');console.error(e);process.exitCode=1;}
