declare const process:{exitCode?:number};
import { DRY_AIR, EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { reconstructEdgeNormalScalarGradient } from '../physics/horizontalGradient.js';
import { buildRotationGeometry } from '../physics/rotation.js';
import { diagnoseAxialAngularMomentum, diagnoseAxialAngularMomentumTendency } from '../solver/stage4CirculationDiagnostics.js';
import { cell3DIndex, createHydrostaticState, edge3DIndex, type DryState } from '../solver/state.js';

function inversePressure(p:number):number{return DRY_AIR.pRef/DRY_AIR.rd*Math.pow(p/DRY_AIR.pRef,1/DRY_AIR.gamma);}
function accelPerDay(torque:number,lever:number):number{return torque/lever*86400;}

function smoothMode(x:number,y:number,z:number,m:number):number{
  let re=1,im=0;
  for(let i=0;i<m;i++){const nr=re*x-im*y,ni=re*y+im*x;re=nr;im=ni;}
  return re*(1+.35*z+.15*z*z);
}

function densityShape(x:number,y:number,z:number):number{
  return .48*x-.31*y+.27*z+.22*x*y-.17*y*z+.09*(x*x-z*z);
}

function currentLsPressureAcceleration(
  h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,s:DryState,g:ReturnType<typeof buildRotationGeometry>,pressure:Float64Array,
):Float64Array{
  const uT=new Float64Array(h.edgeCount*v.nz);
  for(let e=0;e<h.edgeCount;e++)for(let k=0;k<v.nz;k++){
    const ge=h.edges[e]!,l=cell3DIndex(ge.leftCell,k,v.nz),r=cell3DIndex(ge.rightCell,k,v.nz),rho=.5*(s.rhoD[l]!+s.rhoD[r]!),grad=reconstructEdgeNormalScalarGradient(h,g,e,c=>pressure[cell3DIndex(c,k,v.nz)]!);
    uT[edge3DIndex(e,k,v.nz)]=-grad/Math.max(rho,1e-12);
  }
  return uT;
}

/**
 * Diagnostic Green-Gauss candidate, not production. A single face pressure is
 * shared by both cells; each cell integrates local pressure traction first and
 * divides by its own mass metric only afterwards. No global torque correction
 * or climate target is used.
 */
function greenGaussPressureAcceleration(
  h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,s:DryState,g:ReturnType<typeof buildRotationGeometry>,pressure:Float64Array,
):Float64Array{
  const cellA=Array.from({length:v.nz},()=>new Float64Array(h.cellCount*3)),R=EARTH.radius;
  for(let k=0;k<v.nz;k++){
    const a=cellA[k]!;
    for(let c=0;c<h.cellCount;c++){
      const o=c*3,q=cell3DIndex(c,k,v.nz),rho=Math.max(s.rhoD[q]!,1e-12),area=h.cellAreaUnit[c]!*R*R;
      let fx=0,fy=0,fz=0;
      for(let slot=0;slot<4;slot++){
        const eid=h.cellEdges[c*4+slot]!,sign=h.cellEdgeSigns[c*4+slot]!,edge=h.edges[eid]!,nb=edge.leftCell===c?edge.rightCell:edge.leftCell,pf=.5*(pressure[q]!+pressure[cell3DIndex(nb,k,v.nz)]!),L=edge.angularLength*R;
        fx-=pf*sign*edge.normal[0]*L;fy-=pf*sign*edge.normal[1]*L;fz-=pf*sign*edge.normal[2]*L;
      }
      let ax=fx/(rho*area),ay=fy/(rho*area),az=fz/(rho*area);
      const rx=g.radial[o]!,ry=g.radial[o+1]!,rz=g.radial[o+2]!,rad=ax*rx+ay*ry+az*rz;ax-=rad*rx;ay-=rad*ry;az-=rad*rz;
      a[o]=ax;a[o+1]=ay;a[o+2]=az;
    }
  }
  const uT=new Float64Array(h.edgeCount*v.nz);
  for(let e=0;e<h.edgeCount;e++){
    const edge=h.edges[e]!,l=edge.leftCell,r=edge.rightCell,n=edge.normal;
    for(let k=0;k<v.nz;k++){
      const a=cellA[k]!,lo=l*3,ro=r*3;uT[edge3DIndex(e,k,v.nz)]=.5*((a[lo]!+a[ro]!)*n[0]+(a[lo+1]!+a[ro+1]!)*n[1]+(a[lo+2]!+a[ro+2]!)*n[2]);
    }
  }
  return uT;
}

function run(n:number,m:number,varyDensity:boolean){
  const h=buildCubedSphere(n),v=buildStretchedVerticalGrid(4,12000,1.15),ref=buildHeldSuarezReference(v),g=buildRotationGeometry(h),s=createHydrostaticState(h,v,ref),pressure=new Float64Array(s.rhoThetaM.length);
  for(let c=0;c<h.cellCount;c++){
    const x=h.cellCenters[c*3]!,y=h.cellCenters[c*3+1]!,z=h.cellCenters[c*3+2]!,shape=smoothMode(x,y,z,m),rf=varyDensity?1+.28*densityShape(x,y,z):1;
    for(let k=0;k<v.nz;k++){
      const q=cell3DIndex(c,k,v.nz),p=ref.pCenter[k]!*(1+.018*shape);pressure[q]=p;s.rhoThetaM[q]=inversePressure(p);s.rhoD[q]=s.rhoD[q]!*rf;
    }
  }
  const zeroRho=new Float64Array(s.rhoD.length),lever=diagnoseAxialAngularMomentum(h,v,s,g).torqueLeverMass;
  const torque=(uT:Float64Array)=>accelPerDay(diagnoseAxialAngularMomentumTendency(h,v,s,zeroRho,uT,g).velocityTorque,lever);
  return{n,m,ls:torque(currentLsPressureAcceleration(h,v,s,g,pressure)),greenGauss:torque(greenGaussPressureAcceleration(h,v,s,g,pressure))};
}

try{
  console.log('Stage4 closed-sphere pressure-torque stress test (equivalent m/s/day; continuum pressure-force target = 0; Green-Gauss diagnostic only)');
  for(const varying of [false,true]){
    console.log(`horizontal density ${varying?'VARIES':'uniform'}`);
    console.log('mode\tN8 LS\tN8 GG\tN16 LS\tN16 GG\tN32 LS\tN32 GG');
    for(const m of [1,2,3,4,5,6,7,8]){
      const r8=run(8,m,varying),r16=run(16,m,varying),r32=run(32,m,varying);
      console.log(`${m}\t${r8.ls.toExponential(7)}\t${r8.greenGauss.toExponential(7)}\t${r16.ls.toExponential(7)}\t${r16.greenGauss.toExponential(7)}\t${r32.ls.toExponential(7)}\t${r32.greenGauss.toExponential(7)}`);
      if(![r8.ls,r8.greenGauss,r16.ls,r16.greenGauss,r32.ls,r32.greenGauss].every(Number.isFinite))throw new Error(`non-finite pressure torque mode ${m}`);
    }
  }
}catch(e){console.error('FAIL Stage4 pressure-torque prototype');console.error(e);process.exitCode=1;}
