declare const process:{exitCode?:number};
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { buildRotationGeometry } from '../physics/rotation.js';
import { diagnoseAxialAngularMomentum } from '../solver/stage4CirculationDiagnostics.js';
import { diagnoseStage4DirectionalAamBreakdown } from '../solver/stage4MomentumBudgetDiagnostics.js';
import { createHydrostaticState, edge3DIndex, w3DIndex } from '../solver/state.js';

type V3=readonly[number,number,number];
function basis(r:V3):{east:V3;north:V3}{const xy=Math.hypot(r[0],r[1]),e:V3=xy>1e-14?[-r[1]/xy,r[0]/xy,0]:[0,1,0],n:V3=[-r[2]*e[1],r[2]*e[0],xy];return{east:e,north:n};}
function setSmooth3dState(n:number){
  const h=buildCubedSphere(n),v=buildStretchedVerticalGrid(10,18000,1.2),ref=buildHeldSuarezReference(v),g=buildRotationGeometry(h),s=createHydrostaticState(h,v,ref);
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,r=ge.midpoint,b=basis(r);
    for(let k=0;k<v.nz;k++){
      const z=v.zCenter[k]!/v.top,ue=(12+5*r[0]-3*r[1]+2*r[2])*(1+.35*z),vn=(4*r[0]+6*r[1]-3*r[2])*(1-.2*z);
      const wx=ue*b.east[0]+vn*b.north[0],wy=ue*b.east[1]+vn*b.north[1],wz=ue*b.east[2]+vn*b.north[2];
      s.uEdge[edge3DIndex(e,k,v.nz)]=wx*ge.normal[0]+wy*ge.normal[1]+wz*ge.normal[2];
    }
  }
  for(let c=0;c<h.cellCount;c++){
    const x=h.cellCenters[c*3]!,y=h.cellCenters[c*3+1]!,zeta=h.cellCenters[c*3+2]!;
    for(let i=1;i<v.nz;i++){
      const zz=v.zInterface[i]!/v.top;
      s.wInterface[w3DIndex(c,i,v.nz)]=.18*Math.sin(Math.PI*zz)*(x-.6*y+.25*zeta);
    }
    for(let k=0;k<v.nz;k++){
      const q=c*v.nz+k,fac=1+.025*(x*y+.4*zeta)*(1-.3*v.zCenter[k]!/v.top);
      s.rhoD[q]*=fac;s.rhoThetaM[q]*=fac;
    }
  }
  return{h,v,ref,g,s};
}
try{
  console.log('Stage4 directional material-AAM diagnostic (equivalent m/s/day)');
  console.log('N\thoriz pair\tvertical pair\ttotal pair\tclosure');
  const rows=[] as {n:number;h:number;v:number;t:number;c:number}[];
  for(const n of [4,8,16,32]){
    const q=setSmooth3dState(n),d=diagnoseStage4DirectionalAamBreakdown(q.h,q.v,q.ref,q.s,q.g),lever=diagnoseAxialAngularMomentum(q.h,q.v,q.s,q.g).torqueLeverMass,scale=86400/lever;
    const row={n,h:d.horizontalPairResidual*scale,v:d.verticalPairResidual*scale,t:d.totalPairResidual*scale,c:d.closure*scale};rows.push(row);
    console.log(`${n}\t${row.h.toExponential(8)}\t${row.v.toExponential(8)}\t${row.t.toExponential(8)}\t${row.c.toExponential(3)}`);
    if(![row.h,row.v,row.t,row.c].every(Number.isFinite))throw new Error(`non-finite directional AAM at N${n}`);
    if(Math.abs(row.c)>1e-10)throw new Error(`directional decomposition closure failed at N${n}: ${row.c}`);
  }
  const ratio=(a:number,b:number)=>Math.abs(a)/Math.max(Math.abs(b),1e-30);
  console.log(`horizontal refine 8->16=${ratio(rows[1]!.h,rows[2]!.h).toFixed(3)} 16->32=${ratio(rows[2]!.h,rows[3]!.h).toFixed(3)}`);
  console.log(`vertical refine 8->16=${ratio(rows[1]!.v,rows[2]!.v).toFixed(3)} 16->32=${ratio(rows[2]!.v,rows[3]!.v).toFixed(3)}`);
}catch(e){console.error('FAIL Stage4 directional material-AAM diagnostic');console.error(e);process.exitCode=1;}
