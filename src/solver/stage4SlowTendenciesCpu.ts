import { DRY_AIR, EARTH } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { heldSuarezDragRate, heldSuarezTeq, heldSuarezThermalRate } from '../physics/heldSuarez.js';
import { computeHorizontalMaterialMomentumTendency } from '../physics/horizontalMomentumTransport.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { buildRotationGeometry, reconstructCellHorizontalWind, RotationGeometry } from '../physics/rotation.js';
import { pressureFromRhoTheta, thetaFromTP } from '../physics/thermodynamics.js';
import { computeVerticalHorizontalMomentumTendency } from '../physics/verticalHorizontalMomentumTransport.js';
import { computeVerticalVelocityAdvectionTendency } from '../physics/verticalMomentumAdvection.js';
import { DryState, cell3DIndex, edge3DIndex, w3DIndex } from './state.js';

export interface Stage4SlowOptions { momentumTransport?:boolean;coriolis?:boolean;heldSuarez?:boolean; }
export interface Stage4SlowTendencies {
  rhoD:Float64Array;rhoThetaM:Float64Array;uEdge:Float64Array;wInterface:Float64Array;
  /** Slow scalar carriers: perturbation mass/thermodynamic flux only. */
  hMassFlux:Float64Array;vMassFlux:Float64Array;
}

/**
 * Pure frozen slow RHS for the split-explicit RK3 reference.
 *
 * Scalar continuity uses a symmetric reference/perturbation split in all three
 * dimensions. With the hydrostatic reference depending only on height,
 *
 *   rho*u = rho0*u + (rho-rho0)*u,
 *   X*u   = X0*u   + (X-X0)*u.
 *
 * Only the perturbation fluxes are included here. Reference flux divergence is
 * a fast acoustic term and is inserted by computeStage4FrozenRhs(). Thus the
 * acoustic correction owns compression of the reference atmosphere while the
 * large-step RHS owns nonlinear transport of departures from that reference.
 *
 * Horizontal material momentum uses the same total donor mass carrier as the
 * discrete continuity equation and reconstructs the upwind horizontal wind to
 * the actual great-circle face midpoint with tangent least-squares gradients
 * plus a local Barth-Jespersen-style vector limiter. Vertical transport likewise
 * uses the exact total vertical carrier produced by reference+perturbation
 * continuity and converts conservative donor momentum flux back to a material
 * velocity tendency with the discrete product rule. Neither operator contains a
 * climatological target, preferred wind sign, or global AAM correction.
 */
export function computeStage4SlowTendencies(h:CubedSphereGrid,v:VerticalGrid,ref:ReferenceAtmosphere,s:DryState,options:Stage4SlowOptions={},rotation?:RotationGeometry):Stage4SlowTendencies {
  const momentum=options.momentumTransport!==false,coriolis=options.coriolis!==false,heldSuarez=options.heldSuarez!==false,nz=v.nz,R=EARTH.radius,g=rotation??buildRotationGeometry(h);
  const rhoT=new Float64Array(s.rhoD.length),xT=new Float64Array(s.rhoThetaM.length),uT=new Float64Array(s.uEdge.length),wT=momentum?computeVerticalVelocityAdvectionTendency(h,v,s):new Float64Array(s.wInterface.length),hMass=new Float64Array(s.uEdge.length),vMass=new Float64Array(s.wInterface.length);

  // Horizontal perturbation scalar flux.
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,edgeLength=ge.angularLength*R;
    for(let k=0;k<nz;k++){
      const qe=edge3DIndex(e,k,nz),vel=s.uEdge[qe]!,l=cell3DIndex(ge.leftCell,k,nz),r=cell3DIndex(ge.rightCell,k,nz),up=vel>=0?l:r,A=edgeLength*v.dz[k]!,fm=(s.rhoD[up]!-ref.rhoCenter[k]!)*vel*A,fx=(s.rhoThetaM[up]!-ref.rhoThetaCenter[k]!)*vel*A;
      hMass[qe]=fm;rhoT[l]=rhoT[l]!-fm;rhoT[r]=rhoT[r]!+fm;xT[l]=xT[l]!-fx;xT[r]=xT[r]!+fx;
    }
  }
  // Vertical perturbation scalar flux.
  for(let c=0;c<h.cellCount;c++){
    const area=h.cellAreaUnit[c]!*R*R;
    for(let i=1;i<nz;i++){
      const qi=w3DIndex(c,i,nz),vel=s.wInterface[qi]!,srcK=vel>=0?i-1:i,src=cell3DIndex(c,srcK,nz),fm=(s.rhoD[src]!-ref.rhoCenter[srcK]!)*vel*area,fx=(s.rhoThetaM[src]!-ref.rhoThetaCenter[srcK]!)*vel*area;
      vMass[qi]=fm;const l=cell3DIndex(c,i-1,nz),u=cell3DIndex(c,i,nz);rhoT[l]=rhoT[l]!-fm;rhoT[u]=rhoT[u]!+fm;xT[l]=xT[l]!-fx;xT[u]=xT[u]!+fx;
    }
  }
  for(let c=0;c<h.cellCount;c++)for(let k=0;k<nz;k++){
    const q=cell3DIndex(c,k,nz),vol=h.cellAreaUnit[c]!*R*R*v.dz[k]!;rhoT[q]=rhoT[q]!/vol;xT[q]=xT[q]!/vol;
  }

  if(momentum||coriolis){
    const windByK:Float64Array[]=Array.from({length:nz},(_,k)=>reconstructCellHorizontalWind(h,g,s,k)),cellVT:Float64Array[]=Array.from({length:nz},()=>new Float64Array(h.cellCount*3));
    if(momentum){
      const horizontal=computeHorizontalMaterialMomentumTendency(h,v,s,g,'muscl-bj',windByK);
      const vertical=computeVerticalHorizontalMomentumTendency(h,v,ref,s,g,windByK);
      for(let k=0;k<nz;k++){
        const dst=cellVT[k]!,ha=horizontal[k]!,va=vertical[k]!;
        for(let i=0;i<dst.length;i++)dst[i]=ha[i]!+va[i]!;
      }
    }
    if(coriolis){
      for(let k=0;k<nz;k++){const wind=windByK[k]!,dv=cellVT[k]!;for(let c=0;c<h.cellCount;c++){
        const o=c*3,ex=g.east[o]!,ey=g.east[o+1]!,ez=g.east[o+2]!,nx=g.north[o]!,ny=g.north[o+1]!,nzv=g.north[o+2]!,wx=wind[o]!,wy=wind[o+1]!,wz=wind[o+2]!,ue=wx*ex+wy*ey+wz*ez,vn=wx*nx+wy*ny+wz*nzv,f=2*EARTH.omega*g.radial[o+2]!;
        dv[o]=dv[o]!+f*vn*ex-f*ue*nx;dv[o+1]=dv[o+1]!+f*vn*ey-f*ue*ny;dv[o+2]=dv[o+2]!+f*vn*ez-f*ue*nzv;
      }}
    }
    // Covariant/tangent projection at each source cell before averaging to the
    // staggered edge. The global-vector derivative may contain a radial piece
    // solely because the tangent basis rotates across the sphere; that piece is
    // not a physical horizontal acceleration.
    for(let e=0;e<h.edgeCount;e++){
      const ge=h.edges[e]!,l=ge.leftCell,r=ge.rightCell,n=ge.normal;for(let k=0;k<nz;k++){
        const dv=cellVT[k]!,lo=l*3,ro=r*3,lrx=g.radial[lo]!,lry=g.radial[lo+1]!,lrz=g.radial[lo+2]!,rrx=g.radial[ro]!,rry=g.radial[ro+1]!,rrz=g.radial[ro+2]!,ldot=dv[lo]!*lrx+dv[lo+1]!*lry+dv[lo+2]!*lrz,rdot=dv[ro]!*rrx+dv[ro+1]!*rry+dv[ro+2]!*rrz;
        const lx=dv[lo]!-ldot*lrx,ly=dv[lo+1]!-ldot*lry,lz=dv[lo+2]!-ldot*lrz,rx=dv[ro]!-rdot*rrx,ry=dv[ro+1]!-rdot*rry,rz=dv[ro+2]!-rdot*rrz;
        uT[edge3DIndex(e,k,nz)]=uT[edge3DIndex(e,k,nz)]!+.5*((lx+rx)*n[0]+(ly+ry)*n[1]+(lz+rz)*n[2]);
      }
    }
  }

  if(heldSuarez){
    for(let c=0;c<h.cellCount;c++){
      const lat=Math.asin(h.cellCenters[c*3+2]!);for(let k=0;k<nz;k++){
        const q=cell3DIndex(c,k,nz),rho=Math.max(s.rhoD[q]!,1e-12),x=s.rhoThetaM[q]!,p=pressureFromRhoTheta(x),theta=x/rho,sigma=p/DRY_AIR.pRef,thetaEq=thetaFromTP(heldSuarezTeq(lat,p),p),rate=heldSuarezThermalRate(lat,sigma);xT[q]=xT[q]!+rho*rate*(thetaEq-theta);
      }
    }
    for(let e=0;e<h.edgeCount;e++){
      const ge=h.edges[e]!;for(let k=0;k<nz;k++){
        const l=cell3DIndex(ge.leftCell,k,nz),r=cell3DIndex(ge.rightCell,k,nz),sigma=.5*(pressureFromRhoTheta(s.rhoThetaM[l]!)+pressureFromRhoTheta(s.rhoThetaM[r]!))/DRY_AIR.pRef,rate=heldSuarezDragRate(sigma),q=edge3DIndex(e,k,nz);uT[q]=uT[q]!-rate*s.uEdge[q]!;
      }
    }
  }
  return{rhoD:rhoT,rhoThetaM:xT,uEdge:uT,wInterface:wT,hMassFlux:hMass,vMassFlux:vMass};
}
