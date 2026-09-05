declare const process:{exitCode?:number};
import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { reconstructCellScalarGradient } from '../physics/horizontalGradient.js';
import { buildRotationGeometry, reconstructCellHorizontalWind, setAnalyticCellWind } from '../physics/rotation.js';
import { diagnoseAxialAngularMomentum, diagnoseAxialAngularMomentumTendency } from '../solver/stage4CirculationDiagnostics.js';
import { computeStage4FrozenRhs } from '../solver/stage4Rk3SplitCpu.js';
import { computeStage4SlowTendencies } from '../solver/stage4SlowTendenciesCpu.js';
import { createHydrostaticState, edge3DIndex, type DryState } from '../solver/state.js';

interface Metrics{aamResidual:number;kineticResidual:number;}
function accelPerDay(torque:number,leverMass:number):number{return torque/leverMass*86400;}
function clampDot(x:number):number{return Math.max(-1,Math.min(1,x));}

function setup(n:number){
  const h=buildCubedSphere(n),v=buildStretchedVerticalGrid(4,12000,1.15),ref=buildHeldSuarezReference(v),g=buildRotationGeometry(h);
  const s=createHydrostaticState(h,v,ref);
  setAnalyticCellWind(h,g,s,(r,east,north,k)=>{
    const vertical=1+0.08*k,ue=vertical*(11+4*r[0]-3*r[1]+2*r[2]),vn=vertical*(6*r[0]+5*r[1]-4*r[2]+2*r[0]*r[1]);
    return[ue*east[0]+vn*north[0],ue*east[1]+vn*north[1],ue*east[2]+vn*north[2]];
  });
  const continuity=computeStage4FrozenRhs(h,v,ref,s,{momentumTransport:false,coriolis:false,heldSuarez:false},g);
  const windByK=Array.from({length:v.nz},(_,k)=>reconstructCellHorizontalWind(h,g,s,k));
  return{h,v,ref,g,s,continuity,windByK};
}

function metricsFromEdgeTendency(ctx:ReturnType<typeof setup>,uT:Float64Array):Metrics{
  const{h,v,g,s,continuity,windByK}=ctx;
  const zeroU=new Float64Array(s.uEdge.length),zeroRho=new Float64Array(s.rhoD.length);
  const mass=diagnoseAxialAngularMomentumTendency(h,v,s,continuity.rhoD,zeroU,g);
  const mom=diagnoseAxialAngularMomentumTendency(h,v,s,zeroRho,uT,g).velocityTorque;
  const lever=diagnoseAxialAngularMomentum(h,v,s,g).torqueLeverMass;

  const accelState:DryState={rhoD:s.rhoD,rhoThetaM:s.rhoThetaM,uEdge:uT,wInterface:s.wInterface,time:s.time};
  const accelByK=Array.from({length:v.nz},(_,k)=>reconstructCellHorizontalWind(h,g,accelState,k));
  let dke=0,totalMass=0;
  for(let c=0;c<h.cellCount;c++)for(let k=0;k<v.nz;k++){
    const q=c*v.nz+k,o=c*3,vol=h.cellAreaUnit[c]!*EARTH.radius*EARTH.radius*v.dz[k]!,rho=s.rhoD[q]!,rhoT=continuity.rhoD[q]!,w=windByK[k]!,a=accelByK[k]!;
    const speed2=w[o]!*w[o]!+w[o+1]!*w[o+1]!+w[o+2]!*w[o+2]!;
    const work=w[o]!*a[o]!+w[o+1]!*a[o+1]!+w[o+2]!*a[o+2]!;
    dke+=vol*(.5*rhoT*speed2+rho*work);
    totalMass+=vol*rho;
  }
  return{
    aamResidual:accelPerDay(mass.relativeMassRedistributionTorque+mom,lever),
    kineticResidual:dke/totalMass*86400,
  };
}

function finish(ctx:ReturnType<typeof setup>,cellA:Float64Array[]):Metrics{
  const{h,v,s}=ctx,uT=new Float64Array(s.uEdge.length);
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,l=ge.leftCell,r=ge.rightCell,norm=ge.normal;
    for(let k=0;k<v.nz;k++){
      const a=cellA[k]!,lo=l*3,ro=r*3;
      uT[edge3DIndex(e,k,v.nz)]=.5*((a[lo]!+a[ro]!)*norm[0]+(a[lo+1]!+a[ro+1]!)*norm[1]+(a[lo+2]!+a[ro+2]!)*norm[2]);
    }
  }
  return metricsFromEdgeTendency(ctx,uT);
}

function donorCellProduction(n:number):Metrics{
  const ctx=setup(n),{h,v,ref,g,s}=ctx;
  const t=computeStage4SlowTendencies(h,v,ref,s,{momentumTransport:true,coriolis:false,heldSuarez:false},g);
  return metricsFromEdgeTendency(ctx,t.uEdge);
}

/** Smooth-flow experiment only. NOT production. */
function centeredCovariantMomentum(n:number):Metrics{
  const ctx=setup(n),{h,v,g,windByK}=ctx,cellA=Array.from({length:v.nz},()=>new Float64Array(h.cellCount*3));
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
  return finish(ctx,cellA);
}

/**
 * Unlimited second-order upwind finite-volume experiment. Face wind is
 * reconstructed from the upwind cell to the actual great-circle face midpoint
 * using tangent-plane least-squares gradients. The material derivative is
 * -div(u q)+q div(u), so its first-order limit is the existing donor operator.
 * No limiter is used yet: this is a consistency experiment, not production.
 */
function musclCovariantMomentum(n:number):Metrics{
  const ctx=setup(n),{h,v,g,s,windByK}=ctx,R=EARTH.radius,cellA=Array.from({length:v.nz},()=>new Float64Array(h.cellCount*3));
  for(let k=0;k<v.nz;k++){
    const wind=windByK[k]!,a=cellA[k]!;
    const gx=new Float64Array(h.cellCount*3),gy=new Float64Array(h.cellCount*3),gz=new Float64Array(h.cellCount*3);
    for(let c=0;c<h.cellCount;c++){
      gx.set(reconstructCellScalarGradient(h,g,c,cc=>wind[cc*3]!),c*3);
      gy.set(reconstructCellScalarGradient(h,g,c,cc=>wind[cc*3+1]!),c*3);
      gz.set(reconstructCellScalarGradient(h,g,c,cc=>wind[cc*3+2]!),c*3);
    }
    for(let c=0;c<h.cellCount;c++){
      const o=c*3,curx=wind[o]!,cury=wind[o+1]!,curz=wind[o+2]!,area=h.cellAreaUnit[c]!*R*R;
      let sumF=0,sumX=0,sumY=0,sumZ=0;
      for(let slot=0;slot<4;slot++){
        const eid=h.cellEdges[c*4+slot]!,sgn=h.cellEdgeSigns[c*4+slot]!,ge=h.edges[eid]!,outward=sgn*s.uEdge[edge3DIndex(eid,k,v.nz)]!,F=outward*ge.angularLength*R;
        const nb=ge.leftCell===c?ge.rightCell:ge.leftCell,up=outward>=0?c:nb,uo=up*3;
        const rx=g.radial[uo]!,ry=g.radial[uo+1]!,rz=g.radial[uo+2]!,m=ge.midpoint,mu=clampDot(rx*m[0]+ry*m[1]+rz*m[2]),ang=Math.acos(mu);
        let dx=0,dy=0,dz=0;
        const tx=m[0]-mu*rx,ty=m[1]-mu*ry,tz=m[2]-mu*rz,tm=Math.hypot(tx,ty,tz);
        if(ang>0&&tm>0){const scale=R*ang/tm;dx=tx*scale;dy=ty*scale;dz=tz*scale;}
        let qx=wind[uo]!+gx[uo]!*dx+gx[uo+1]!*dy+gx[uo+2]!*dz;
        let qy=wind[uo+1]!+gy[uo]!*dx+gy[uo+1]!*dy+gy[uo+2]!*dz;
        let qz=wind[uo+2]!+gz[uo]!*dx+gz[uo+1]!*dy+gz[uo+2]!*dz;
        const fr=qx*m[0]+qy*m[1]+qz*m[2];qx-=fr*m[0];qy-=fr*m[1];qz-=fr*m[2];
        sumF+=F;sumX+=F*qx;sumY+=F*qy;sumZ+=F*qz;
      }
      let ax=-sumX/area+curx*sumF/area,ay=-sumY/area+cury*sumF/area,az=-sumZ/area+curz*sumF/area;
      const rx=g.radial[o]!,ry=g.radial[o+1]!,rz=g.radial[o+2]!,rd=ax*rx+ay*ry+az*rz;
      ax-=rd*rx;ay-=rd*ry;az-=rd*rz;
      a[o]=ax;a[o+1]=ay;a[o+2]=az;
    }
  }
  return finish(ctx,cellA);
}

try{
  const ns=[4,8,16,32];
  const donor=ns.map(n=>({n,...donorCellProduction(n)}));
  const centered=ns.map(n=>({n,...centeredCovariantMomentum(n)}));
  const muscl=ns.map(n=>({n,...musclCovariantMomentum(n)}));
  const ratio=(a:number,b:number)=>Math.abs(a)/Math.max(Math.abs(b),1e-30);
  console.log('Stage4 material-momentum smooth-flow diagnostics (NOT production except donor)');
  console.log('AAM residual: equivalent m/s/day; kinetic residual: specific KE m^2/s^2/day; continuum advection targets = 0');
  console.log('N\tdonor AAM\tcentered AAM\tMUSCL AAM\tdonor KE\tcentered KE\tMUSCL KE');
  for(let i=0;i<ns.length;i++)console.log(`${ns[i]}\t${donor[i]!.aamResidual.toExponential(8)}\t${centered[i]!.aamResidual.toExponential(8)}\t${muscl[i]!.aamResidual.toExponential(8)}\t${donor[i]!.kineticResidual.toExponential(8)}\t${centered[i]!.kineticResidual.toExponential(8)}\t${muscl[i]!.kineticResidual.toExponential(8)}`);
  console.log(`AAM centered refine 8->16=${ratio(centered[1]!.aamResidual,centered[2]!.aamResidual).toFixed(3)} 16->32=${ratio(centered[2]!.aamResidual,centered[3]!.aamResidual).toFixed(3)}`);
  console.log(`AAM MUSCL refine 8->16=${ratio(muscl[1]!.aamResidual,muscl[2]!.aamResidual).toFixed(3)} 16->32=${ratio(muscl[2]!.aamResidual,muscl[3]!.aamResidual).toFixed(3)}`);
  if(![...donor,...centered,...muscl].every(r=>Number.isFinite(r.aamResidual)&&Number.isFinite(r.kineticResidual)))throw new Error('non-finite prototype result');
}catch(e){console.error('FAIL Stage4 momentum-advection prototype');console.error(e);process.exitCode=1;}
