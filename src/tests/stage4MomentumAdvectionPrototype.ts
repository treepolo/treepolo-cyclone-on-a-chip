declare const process:{exitCode?:number};
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { reconstructCellScalarGradient } from '../physics/horizontalGradient.js';
import { buildRotationGeometry, reconstructCellHorizontalWind, setAnalyticCellWind } from '../physics/rotation.js';
import { diagnoseAxialAngularMomentum, diagnoseAxialAngularMomentumTendency } from '../solver/stage4CirculationDiagnostics.js';
import { computeStage4FrozenRhs } from '../solver/stage4Rk3SplitCpu.js';
import { createHydrostaticState, edge3DIndex } from '../solver/state.js';

function accelPerDay(torque:number,leverMass:number):number{return torque/leverMass*86400;}

/**
 * Smooth-flow experiment only.  This is deliberately NOT wired into the
 * production solver.  It evaluates -u·grad(u) by taking least-squares
 * gradients of the reconstructed global Cartesian wind components on each
 * cell tangent plane, then projects the resulting covariant acceleration back
 * to the C-grid face normal.
 */
function centeredCovariantMomentum(n:number){
  const h=buildCubedSphere(n),v=buildStretchedVerticalGrid(4,12000,1.15),ref=buildHeldSuarezReference(v),g=buildRotationGeometry(h);
  const s=createHydrostaticState(h,v,ref);
  setAnalyticCellWind(h,g,s,(r,east,north,k)=>{
    const vertical=1+0.08*k,ue=vertical*(11+4*r[0]-3*r[1]+2*r[2]),vn=vertical*(6*r[0]+5*r[1]-4*r[2]+2*r[0]*r[1]);
    return[ue*east[0]+vn*north[0],ue*east[1]+vn*north[1],ue*east[2]+vn*north[2]];
  });

  const continuity=computeStage4FrozenRhs(h,v,ref,s,{momentumTransport:false,coriolis:false,heldSuarez:false},g);
  const zeroU=new Float64Array(s.uEdge.length),zeroRho=new Float64Array(s.rhoD.length),uT=new Float64Array(s.uEdge.length);
  const windByK=Array.from({length:v.nz},(_,k)=>reconstructCellHorizontalWind(h,g,s,k));
  const cellA=Array.from({length:v.nz},()=>new Float64Array(h.cellCount*3));

  for(let k=0;k<v.nz;k++){
    const wind=windByK[k]!,a=cellA[k]!;
    for(let c=0;c<h.cellCount;c++){
      const o=c*3,wx=wind[o]!,wy=wind[o+1]!,wz=wind[o+2]!;
      const gx=reconstructCellScalarGradient(h,g,c,cc=>wind[cc*3]!);
      const gy=reconstructCellScalarGradient(h,g,c,cc=>wind[cc*3+1]!);
      const gz=reconstructCellScalarGradient(h,g,c,cc=>wind[cc*3+2]!);
      let ax=-(wx*gx[0]+wy*gx[1]+wz*gx[2]);
      let ay=-(wx*gy[0]+wy*gy[1]+wz*gy[2]);
      let az=-(wx*gz[0]+wy*gz[1]+wz*gz[2]);
      const rx=g.radial[o]!,ry=g.radial[o+1]!,rz=g.radial[o+2]!,rd=ax*rx+ay*ry+az*rz;
      ax-=rd*rx;ay-=rd*ry;az-=rd*rz;
      a[o]=ax;a[o+1]=ay;a[o+2]=az;
    }
  }

  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,l=ge.leftCell,r=ge.rightCell,norm=ge.normal;
    for(let k=0;k<v.nz;k++){
      const a=cellA[k]!,lo=l*3,ro=r*3;
      uT[edge3DIndex(e,k,v.nz)]=.5*((a[lo]!+a[ro]!)*norm[0]+(a[lo+1]!+a[ro+1]!)*norm[1]+(a[lo+2]!+a[ro+2]!)*norm[2]);
    }
  }

  const mass=diagnoseAxialAngularMomentumTendency(h,v,s,continuity.rhoD,zeroU,g);
  const mom=diagnoseAxialAngularMomentumTendency(h,v,s,zeroRho,uT,g).velocityTorque;
  const lever=diagnoseAxialAngularMomentum(h,v,s,g).torqueLeverMass;
  return accelPerDay(mass.relativeMassRedistributionTorque+mom,lever);
}

try{
  const rows=[4,8,16,32].map(n=>({n,residual:centeredCovariantMomentum(n)}));
  console.log('Stage4 experimental centered-covariant material-momentum AAM residual (m/s/day; NOT production)');
  for(const r of rows)console.log(`N=${r.n}\t${r.residual.toExponential(8)}`);
  const ratio=(a:number,b:number)=>Math.abs(a)/Math.max(Math.abs(b),1e-30);
  console.log(`refine 8->16=${ratio(rows[1]!.residual,rows[2]!.residual).toFixed(3)} 16->32=${ratio(rows[2]!.residual,rows[3]!.residual).toFixed(3)}`);
  if(!rows.every(r=>Number.isFinite(r.residual)))throw new Error('non-finite prototype result');
}catch(e){console.error('FAIL Stage4 momentum-advection prototype');console.error(e);process.exitCode=1;}
