import { EARTH } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { pressureFromRhoTheta, temperatureFromThetaP } from '../physics/thermodynamics.js';
import { buildRotationGeometry, reconstructCellHorizontalWind, RotationGeometry } from '../physics/rotation.js';
import { computeStage4SlowTendencies } from './stage4SlowTendenciesCpu.js';
import { DryState, cell3DIndex } from './state.js';

export interface AxialAngularMomentumDiagnostics {
  absolute:number;
  planetary:number;
  relative:number;
  dragTorque:number;
  torqueLeverMass:number;
}

export interface EddyDiagnostics {
  midlatitudeEke:number;
  midlatitudePolewardHeatFlux:number;
  midlatitudePolewardMomentumFlux:number;
  sampledMass:number;
}

function cellBasis(h:CubedSphereGrid,g:RotationGeometry,c:number):{east:[number,number,number];north:[number,number,number];lat:number;cosLat:number}{
  const o=c*3;
  const east:[number,number,number]=[g.east[o]!,g.east[o+1]!,g.east[o+2]!];
  const north:[number,number,number]=[g.north[o]!,g.north[o+1]!,g.north[o+2]!];
  const z=Math.max(-1,Math.min(1,g.radial[o+2]!));
  return{east,north,lat:Math.asin(z),cosLat:Math.sqrt(Math.max(0,1-z*z))};
}

/**
 * Global axial angular momentum in the same shallow-sphere geometry used by
 * Stage 4. In the rotating frame the conserved quantity (absent external
 * torque) is
 *
 *   M = integral rho [Omega R^2 cos^2(phi) + R cos(phi) u] dV.
 *
 * dragTorque is evaluated from the actual discrete Held-Suarez edge tendency,
 * then reconstructed to cell-centered eastward acceleration. This makes the
 * budget compare against the torque the production slow RHS really applies.
 */
export function diagnoseAxialAngularMomentum(h:CubedSphereGrid,v:VerticalGrid,s:DryState,rotation:RotationGeometry=buildRotationGeometry(h)):AxialAngularMomentumDiagnostics{
  const R=EARTH.radius,ref=buildHeldSuarezReference(v);
  const drag=computeStage4SlowTendencies(h,v,ref,s,{momentumTransport:false,coriolis:false,heldSuarez:true},rotation);
  const dragState:DryState={rhoD:s.rhoD,rhoThetaM:s.rhoThetaM,uEdge:drag.uEdge,wInterface:s.wInterface,time:s.time};
  let absolute=0,planetary=0,relative=0,dragTorque=0,torqueLeverMass=0;
  for(let k=0;k<v.nz;k++){
    const wind=reconstructCellHorizontalWind(h,rotation,s,k);
    const accel=reconstructCellHorizontalWind(h,rotation,dragState,k);
    for(let c=0;c<h.cellCount;c++){
      const q=cell3DIndex(c,k,v.nz),o=c*3,b=cellBasis(h,rotation,c),vol=h.cellAreaUnit[c]!*R*R*v.dz[k]!,mass=s.rhoD[q]!*vol;
      const u=wind[o]!*b.east[0]+wind[o+1]!*b.east[1]+wind[o+2]!*b.east[2];
      const au=accel[o]!*b.east[0]+accel[o+1]!*b.east[1]+accel[o+2]!*b.east[2];
      const lever=R*b.cosLat;
      const mp=mass*EARTH.omega*lever*lever,mr=mass*lever*u;
      planetary+=mp;relative+=mr;absolute+=mp+mr;dragTorque+=mass*lever*au;torqueLeverMass+=mass*lever;
    }
  }
  return{absolute,planetary,relative,dragTorque,torqueLeverMass};
}

/** Instantaneous zonal-eddy diagnostics. Heat and momentum fluxes are signed
 * poleward in both hemispheres, so a physically active baroclinic field adds
 * rather than cancelling across the equator. */
export function diagnoseEddies(h:CubedSphereGrid,v:VerticalGrid,s:DryState,bins=24,rotation:RotationGeometry=buildRotationGeometry(h)):EddyDiagnostics{
  const n=bins*v.nz,R=EARTH.radius;
  const meanU=new Float64Array(n),meanV=new Float64Array(n),meanT=new Float64Array(n),weight=new Float64Array(n);
  const winds:Float64Array[]=Array.from({length:v.nz},(_,k)=>reconstructCellHorizontalWind(h,rotation,s,k));
  for(let c=0;c<h.cellCount;c++){
    const b=cellBasis(h,rotation,c),bi=Math.max(0,Math.min(bins-1,Math.floor((b.lat+Math.PI/2)/Math.PI*bins)));
    for(let k=0;k<v.nz;k++){
      const q=cell3DIndex(c,k,v.nz),o=c*3,id=bi*v.nz+k,w=s.rhoD[q]!*h.cellAreaUnit[c]!*R*R*v.dz[k]!,wind=winds[k]!;
      const u=wind[o]!*b.east[0]+wind[o+1]!*b.east[1]+wind[o+2]!*b.east[2];
      const vn=wind[o]!*b.north[0]+wind[o+1]!*b.north[1]+wind[o+2]!*b.north[2];
      const p=pressureFromRhoTheta(s.rhoThetaM[q]!),theta=s.rhoThetaM[q]!/Math.max(s.rhoD[q]!,1e-12),T=temperatureFromThetaP(theta,p);
      meanU[id]+=w*u;meanV[id]+=w*vn;meanT[id]+=w*T;weight[id]+=w;
    }
  }
  for(let i=0;i<n;i++)if(weight[i]!>0){meanU[i]/=weight[i]!;meanV[i]/=weight[i]!;meanT[i]/=weight[i]!;}
  let eke=0,heat=0,mom=0,massSum=0;
  for(let c=0;c<h.cellCount;c++){
    const b=cellBasis(h,rotation,c),latDeg=b.lat*180/Math.PI,absLat=Math.abs(latDeg),bi=Math.max(0,Math.min(bins-1,Math.floor((b.lat+Math.PI/2)/Math.PI*bins))),poleSign=latDeg>=0?1:-1;
    if(absLat<30||absLat>70)continue;
    for(let k=0;k<v.nz;k++){
      if(v.zCenter[k]!>15000)continue;
      const q=cell3DIndex(c,k,v.nz),o=c*3,id=bi*v.nz+k,w=s.rhoD[q]!*h.cellAreaUnit[c]!*R*R*v.dz[k]!,wind=winds[k]!;
      const u=wind[o]!*b.east[0]+wind[o+1]!*b.east[1]+wind[o+2]!*b.east[2];
      const vn=wind[o]!*b.north[0]+wind[o+1]!*b.north[1]+wind[o+2]!*b.north[2];
      const p=pressureFromRhoTheta(s.rhoThetaM[q]!),theta=s.rhoThetaM[q]!/Math.max(s.rhoD[q]!,1e-12),T=temperatureFromThetaP(theta,p);
      const up=u-meanU[id]!,vp=vn-meanV[id]!,tp=T-meanT[id]!;
      eke+=w*.5*(up*up+vp*vp);heat+=w*poleSign*vp*tp;mom+=w*poleSign*up*vp;massSum+=w;
    }
  }
  return{midlatitudeEke:massSum?eke/massSum:NaN,midlatitudePolewardHeatFlux:massSum?heat/massSum:NaN,midlatitudePolewardMomentumFlux:massSum?mom/massSum:NaN,sampledMass:massSum};
}
