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
    dke+=vol*(.5*rhoT*speed2+rho*(w[o]!*a[o]!+w[o+1]!*a[o+1]!+w[o+2]!*a[o+2]!));
    totalMass+=vol*rho;
  }
  return{aamResidual:accelPerDay(mass.relativeMassRedistributionTorque+mom,lever),kineticResidual:dke/totalMass*86400};
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
  return metricsFromEdgeTendency(ctx,computeStage4SlowTendencies(h,v,ref,s,{momentumTransport:true,coriolis:false,heldSuarez:false},g).uEdge);
}

function centeredCovariantMomentum(n:number):Metrics{
  const ctx=setup(n),{h,v,g,windByK}=ctx,cellA=Array.from({length:v.nz},()=>new Float64Array(h.cellCount*3));
  for(let k=0;k<v.nz;k++){
    const wind=windByK[k]!,a=cellA[k]!;
    for(let c=0;c<h.cellCount;c++){
      const o=c*3,wx=wind[o]!,wy=wind[o+1]!,wz=wind[o+2]!;
      const gx=reconstructCellScalarGradient(h,g,c,cc=>wind[cc*3]!),gy=reconstructCellScalarGradient(h,g,c,cc=>wind[cc*3+1]!),gz=reconstructCellScalarGradient(h,g,c,cc=>wind[cc*3+2]!);
      let ax=-(wx*gx[0]+wy*gx[1]+wz*gx[2]),ay=-(wx*gy[0]+wy*gy[1]+wz*gy[2]),az=-(wx*gz[0]+wy*gz[1]+wz*gz[2]);
      const rx=g.radial[o]!,ry=g.radial[o+1]!,rz=g.radial[o+2]!,rd=ax*rx+ay*ry+az*rz;ax-=rd*rx;ay-=rd*ry;az-=rd*rz;
      a[o]=ax;a[o+1]=ay;a[o+2]=az;
    }
  }
  return finish(ctx,cellA);
}

function displacementToFace(ctx:ReturnType<typeof setup>,cell:number,edge:number):[number,number,number]{
  const{h,g}=ctx,o=cell*3,ge=h.edges[edge]!,rx=g.radial[o]!,ry=g.radial[o+1]!,rz=g.radial[o+2]!,m=ge.midpoint,mu=clampDot(rx*m[0]+ry*m[1]+rz*m[2]),ang=Math.acos(mu),tx=m[0]-mu*rx,ty=m[1]-mu*ry,tz=m[2]-mu*rz,tm=Math.hypot(tx,ty,tz);
  if(!(ang>0)||!(tm>0))return[0,0,0];
  const scale=EARTH.radius*ang/tm;return[tx*scale,ty*scale,tz*scale];
}

/**
 * One scalar Barth-Jespersen-style factor per cell is shared by all three
 * Cartesian wind components. This preserves the vector-gradient direction
 * while preventing any reconstructed component from creating a new local
 * extremum relative to the four face-neighbor cells.
 */
function vectorLimiter(
  ctx:ReturnType<typeof setup>,
  wind:Float64Array,
  gx:Float64Array,gy:Float64Array,gz:Float64Array,
):Float64Array{
  const{h}=ctx,phi=new Float64Array(h.cellCount);phi.fill(1);
  for(let c=0;c<h.cellCount;c++){
    const o=c*3,base=[wind[o]!,wind[o+1]!,wind[o+2]!],lo=[...base],hi=[...base];
    for(let slot=0;slot<4;slot++){
      const e=h.edges[h.cellEdges[c*4+slot]!]!,nb=e.leftCell===c?e.rightCell:e.leftCell,no=nb*3;
      for(let d=0;d<3;d++){const q=wind[no+d]!;lo[d]=Math.min(lo[d]!,q);hi[d]=Math.max(hi[d]!,q);}
    }
    let p=1;
    for(let slot=0;slot<4;slot++){
      const eid=h.cellEdges[c*4+slot]!,d=displacementToFace(ctx,c,eid),grads=[gx,gy,gz];
      for(let comp=0;comp<3;comp++){
        const gg=grads[comp]!,delta=gg[o]!*d[0]+gg[o+1]!*d[1]+gg[o+2]!*d[2];
        if(delta>1e-15)p=Math.min(p,(hi[comp]!-base[comp]!)/delta);
        else if(delta<-1e-15)p=Math.min(p,(lo[comp]!-base[comp]!)/delta);
      }
    }
    phi[c]=Math.max(0,Math.min(1,p));
  }
  return phi;
}

/**
 * Second-order upwind finite-volume experiment. Face wind is reconstructed
 * from the upwind cell to the actual great-circle face midpoint. The material
 * derivative is -div(u q)+q div(u), so zero-gradient reconstruction is the
 * existing donor scheme. If limited=true a local monotonic vector limiter is
 * applied; it uses no climate target or preferred wind direction.
 */
function musclCovariantMomentum(n:number,limited:boolean):Metrics{
  const ctx=setup(n),{h,v,g,s,windByK}=ctx,R=EARTH.radius,cellA=Array.from({length:v.nz},()=>new Float64Array(h.cellCount*3));
  for(let k=0;k<v.nz;k++){
    const wind=windByK[k]!,a=cellA[k]!,gx=new Float64Array(h.cellCount*3),gy=new Float64Array(h.cellCount*3),gz=new Float64Array(h.cellCount*3);
    for(let c=0;c<h.cellCount;c++){
      gx.set(reconstructCellScalarGradient(h,g,c,cc=>wind[cc*3]!),c*3);gy.set(reconstructCellScalarGradient(h,g,c,cc=>wind[cc*3+1]!),c*3);gz.set(reconstructCellScalarGradient(h,g,c,cc=>wind[cc*3+2]!),c*3);
    }
    const phi=limited?vectorLimiter(ctx,wind,gx,gy,gz):null;
    for(let c=0;c<h.cellCount;c++){
      const o=c*3,curx=wind[o]!,cury=wind[o+1]!,curz=wind[o+2]!,area=h.cellAreaUnit[c]!*R*R;let sumF=0,sumX=0,sumY=0,sumZ=0;
      for(let slot=0;slot<4;slot++){
        const eid=h.cellEdges[c*4+slot]!,sgn=h.cellEdgeSigns[c*4+slot]!,ge=h.edges[eid]!,outward=sgn*s.uEdge[edge3DIndex(eid,k,v.nz)]!,F=outward*ge.angularLength*R,nb=ge.leftCell===c?ge.rightCell:ge.leftCell,up=outward>=0?c:nb,uo=up*3,d=displacementToFace(ctx,up,eid),fac=phi?phi[up]!:1;
        let qx=wind[uo]!+fac*(gx[uo]!*d[0]+gx[uo+1]!*d[1]+gx[uo+2]!*d[2]),qy=wind[uo+1]!+fac*(gy[uo]!*d[0]+gy[uo+1]!*d[1]+gy[uo+2]!*d[2]),qz=wind[uo+2]!+fac*(gz[uo]!*d[0]+gz[uo+1]!*d[1]+gz[uo+2]!*d[2]);
        const m=ge.midpoint,fr=qx*m[0]+qy*m[1]+qz*m[2];qx-=fr*m[0];qy-=fr*m[1];qz-=fr*m[2];
        sumF+=F;sumX+=F*qx;sumY+=F*qy;sumZ+=F*qz;
      }
      let ax=-sumX/area+curx*sumF/area,ay=-sumY/area+cury*sumF/area,az=-sumZ/area+curz*sumF/area;
      const rx=g.radial[o]!,ry=g.radial[o+1]!,rz=g.radial[o+2]!,rd=ax*rx+ay*ry+az*rz;ax-=rd*rx;ay-=rd*ry;az-=rd*rz;a[o]=ax;a[o+1]=ay;a[o+2]=az;
    }
  }
  return finish(ctx,cellA);
}

try{
  const ns=[4,8,16,32],donor=ns.map(n=>({n,...donorCellProduction(n)})),centered=ns.map(n=>({n,...centeredCovariantMomentum(n)})),muscl=ns.map(n=>({n,...musclCovariantMomentum(n,false)})),limited=ns.map(n=>({n,...musclCovariantMomentum(n,true)})),ratio=(a:number,b:number)=>Math.abs(a)/Math.max(Math.abs(b),1e-30);
  console.log('Stage4 material-momentum smooth-flow diagnostics (NOT production except donor)');
  console.log('AAM: equivalent m/s/day; KE: specific kinetic-energy tendency m^2/s^2/day; continuum advection targets = 0');
  console.log('N\tdonor AAM\tcentered AAM\tMUSCL AAM\tlimited AAM\tdonor KE\tcentered KE\tMUSCL KE\tlimited KE');
  for(let i=0;i<ns.length;i++)console.log(`${ns[i]}\t${donor[i]!.aamResidual.toExponential(8)}\t${centered[i]!.aamResidual.toExponential(8)}\t${muscl[i]!.aamResidual.toExponential(8)}\t${limited[i]!.aamResidual.toExponential(8)}\t${donor[i]!.kineticResidual.toExponential(8)}\t${centered[i]!.kineticResidual.toExponential(8)}\t${muscl[i]!.kineticResidual.toExponential(8)}\t${limited[i]!.kineticResidual.toExponential(8)}`);
  console.log(`AAM centered refine 8->16=${ratio(centered[1]!.aamResidual,centered[2]!.aamResidual).toFixed(3)} 16->32=${ratio(centered[2]!.aamResidual,centered[3]!.aamResidual).toFixed(3)}`);
  console.log(`AAM MUSCL refine 8->16=${ratio(muscl[1]!.aamResidual,muscl[2]!.aamResidual).toFixed(3)} 16->32=${ratio(muscl[2]!.aamResidual,muscl[3]!.aamResidual).toFixed(3)}`);
  console.log(`AAM limited refine 8->16=${ratio(limited[1]!.aamResidual,limited[2]!.aamResidual).toFixed(3)} 16->32=${ratio(limited[2]!.aamResidual,limited[3]!.aamResidual).toFixed(3)}`);
  if(![...donor,...centered,...muscl,...limited].every(r=>Number.isFinite(r.aamResidual)&&Number.isFinite(r.kineticResidual)))throw new Error('non-finite prototype result');
}catch(e){console.error('FAIL Stage4 momentum-advection prototype');console.error(e);process.exitCode=1;}
