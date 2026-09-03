import { DRY_AIR, EARTH } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { heldSuarezDragRate, heldSuarezTeq, heldSuarezThermalRate } from '../physics/heldSuarez.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { buildRotationGeometry, reconstructCellHorizontalWind, RotationGeometry } from '../physics/rotation.js';
import { pressureFromRhoTheta, thetaFromTP } from '../physics/thermodynamics.js';
import { computeVerticalVelocityAdvectionTendency } from '../physics/verticalMomentumAdvection.js';
import { DryState, cell3DIndex, edge3DIndex, w3DIndex } from './state.js';

export interface Stage4SlowOptions { momentumTransport?:boolean;coriolis?:boolean;heldSuarez?:boolean; }
export interface Stage4SlowTendencies {
  rhoD:Float64Array;rhoThetaM:Float64Array;uEdge:Float64Array;wInterface:Float64Array;
  /** Slow carriers only: h is full advective flux; v is perturbation Fpert. */
  hMassFlux:Float64Array;vMassFlux:Float64Array;
}

/**
 * Pure frozen slow RHS for Stage-4 RK3.
 *
 * Vertical scalar transport deliberately contains ONLY the perturbation flux
 *   Fpert=(rho-rho0)w, (X-X0)w.
 * The reference flux Fref=rho0*w / X0*w belongs to the acoustic/gravity small
 * step and is inserted into the frozen fast RHS by computeStage4FrozenRhs().
 * This matches the split that passed the real-device 4x2.5 s experiment and
 * prevents reference vertical mass transport from being counted twice.
 *
 * Horizontal-velocity vertical transport is evaluated directly in material
 * form -w du/dz. It therefore transports the full velocity without coupling
 * the velocity tendency to the slow scalar Fpert bookkeeping, and exactly
 * preserves a horizontal wind that is uniform with height.
 */
export function computeStage4SlowTendencies(h:CubedSphereGrid,v:VerticalGrid,ref:ReferenceAtmosphere,s:DryState,options:Stage4SlowOptions={},rotation?:RotationGeometry):Stage4SlowTendencies {
  const momentum=options.momentumTransport!==false,coriolis=options.coriolis!==false,heldSuarez=options.heldSuarez!==false,nz=v.nz,R=EARTH.radius,g=rotation??buildRotationGeometry(h);
  const rhoT=new Float64Array(s.rhoD.length),xT=new Float64Array(s.rhoThetaM.length),rhoHorizontalT=new Float64Array(s.rhoD.length),uT=new Float64Array(s.uEdge.length);
  const wT=momentum?computeVerticalVelocityAdvectionTendency(h,v,s):new Float64Array(s.wInterface.length),hMass=new Float64Array(s.uEdge.length),vMass=new Float64Array(s.wInterface.length);
  const hTheta=new Float64Array(s.uEdge.length),vTheta=new Float64Array(s.wInterface.length);

  // Full horizontal advective scalar flux is a slow term in the current split.
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,edgeLength=ge.angularLength*R;
    for(let k=0;k<nz;k++){
      const qe=edge3DIndex(e,k,nz),vel=s.uEdge[qe]!,l=cell3DIndex(ge.leftCell,k,nz),r=cell3DIndex(ge.rightCell,k,nz),up=vel>=0?l:r,A=edgeLength*v.dz[k]!,fm=s.rhoD[up]!*vel*A,fx=s.rhoThetaM[up]!*vel*A;
      hMass[qe]=fm;hTheta[qe]=fx;
      rhoT[l]=rhoT[l]!-fm;rhoT[r]=rhoT[r]!+fm;rhoHorizontalT[l]=rhoHorizontalT[l]!-fm;rhoHorizontalT[r]=rhoHorizontalT[r]!+fm;xT[l]=xT[l]!-fx;xT[r]=xT[r]!+fx;
    }
  }

  // Slow vertical scalar flux = perturbation only. Reference-state transport
  // is a fast acoustic/gravity term and is intentionally absent here.
  for(let c=0;c<h.cellCount;c++){
    const area=h.cellAreaUnit[c]!*R*R;
    for(let i=1;i<nz;i++){
      const qi=w3DIndex(c,i,nz),vel=s.wInterface[qi]!,srcK=vel>=0?i-1:i,src=cell3DIndex(c,srcK,nz),fm=(s.rhoD[src]!-ref.rhoCenter[srcK]!)*vel*area,fx=(s.rhoThetaM[src]!-ref.rhoThetaCenter[srcK]!)*vel*area;
      vMass[qi]=fm;vTheta[qi]=fx;const l=cell3DIndex(c,i-1,nz),u=cell3DIndex(c,i,nz);
      rhoT[l]=rhoT[l]!-fm;rhoT[u]=rhoT[u]!+fm;xT[l]=xT[l]!-fx;xT[u]=xT[u]!+fx;
    }
  }
  for(let c=0;c<h.cellCount;c++)for(let k=0;k<nz;k++){
    const q=cell3DIndex(c,k,nz),vol=h.cellAreaUnit[c]!*R*R*v.dz[k]!;rhoT[q]=rhoT[q]!/vol;xT[q]=xT[q]!/vol;rhoHorizontalT[q]=rhoHorizontalT[q]!/vol;
  }

  if(momentum||coriolis){
    const windByK:Float64Array[]=Array.from({length:nz},(_,k)=>reconstructCellHorizontalWind(h,g,s,k)),cellVT:Float64Array[]=Array.from({length:nz},()=>new Float64Array(h.cellCount*3));
    if(momentum){
      // Horizontal conservative momentum advection paired with the SAME
      // horizontal continuity flux, converted to velocity tendency.
      const hmx=new Float64Array(hMass.length),hmy=new Float64Array(hMass.length),hmz=new Float64Array(hMass.length);
      for(let e=0;e<h.edgeCount;e++)for(let k=0;k<nz;k++){
        const q=edge3DIndex(e,k,nz),m=hMass[q]!,src=m>=0?h.edges[e]!.leftCell:h.edges[e]!.rightCell,w=windByK[k]!;hmx[q]=m*w[src*3]!;hmy[q]=m*w[src*3+1]!;hmz[q]=m*w[src*3+2]!;
      }
      for(let c=0;c<h.cellCount;c++)for(let k=0;k<nz;k++){
        const q=cell3DIndex(c,k,nz),vol=h.cellAreaUnit[c]!*R*R*v.dz[k]!,rho=Math.max(s.rhoD[q]!,1e-12),wd=windByK[k]!,o=c*3,dv=cellVT[k]!;let mx=0,my=0,mz=0;
        for(let slot=0;slot<4;slot++){const e=h.cellEdges[c*4+slot]!,sgn=h.cellEdgeSigns[c*4+slot]!,qe=edge3DIndex(e,k,nz);mx-=sgn*hmx[qe]!;my-=sgn*hmy[qe]!;mz-=sgn*hmz[qe]!;}
        const rh=rhoHorizontalT[q]!;dv[o]=dv[o]!+(mx/vol-wd[o]!*rh)/rho;dv[o+1]=dv[o+1]!+(my/vol-wd[o+1]!*rh)/rho;dv[o+2]=dv[o+2]!+(mz/vol-wd[o+2]!*rh)/rho;
      }
      // Full vertical material advection of horizontal velocity. A height-
      // uniform wind has zero gradient and is therefore preserved exactly.
      for(let c=0;c<h.cellCount;c++)for(let k=0;k<nz;k++){
        const o=c*3,dv=cellVT[k]!,wc=.5*(s.wInterface[w3DIndex(c,k,nz)]!+s.wInterface[w3DIndex(c,k+1,nz)]!);
        if(wc>0&&k>0){const below=windByK[k-1]!,cur=windByK[k]!,dz=v.zCenter[k]!-v.zCenter[k-1]!;dv[o]=dv[o]!-wc*(cur[o]!-below[o]!)/dz;dv[o+1]=dv[o+1]!-wc*(cur[o+1]!-below[o+1]!)/dz;dv[o+2]=dv[o+2]!-wc*(cur[o+2]!-below[o+2]!)/dz;}
        else if(wc<0&&k<nz-1){const above=windByK[k+1]!,cur=windByK[k]!,dz=v.zCenter[k+1]!-v.zCenter[k]!;dv[o]=dv[o]!-wc*(above[o]!-cur[o]!)/dz;dv[o+1]=dv[o+1]!-wc*(above[o+1]!-cur[o+1]!)/dz;dv[o+2]=dv[o+2]!-wc*(above[o+2]!-cur[o+2]!)/dz;}
      }
    }
    if(coriolis){
      for(let k=0;k<nz;k++){const wind=windByK[k]!,dv=cellVT[k]!;for(let c=0;c<h.cellCount;c++){
        const o=c*3,ex=g.east[o]!,ey=g.east[o+1]!,ez=g.east[o+2]!,nx=g.north[o]!,ny=g.north[o+1]!,nzv=g.north[o+2]!,wx=wind[o]!,wy=wind[o+1]!,wz=wind[o+2]!,ue=wx*ex+wy*ey+wz*ez,vn=wx*nx+wy*ny+wz*nzv,f=2*EARTH.omega*g.radial[o+2]!;
        dv[o]=dv[o]!+f*vn*ex-f*ue*nx;dv[o+1]=dv[o+1]!+f*vn*ey-f*ue*ny;dv[o+2]=dv[o+2]!+f*vn*ez-f*ue*nzv;
      }}
    }
    for(let e=0;e<h.edgeCount;e++){
      const ge=h.edges[e]!,l=ge.leftCell,r=ge.rightCell,n=ge.normal;for(let k=0;k<nz;k++){
        const dv=cellVT[k]!,lx=l*3,rx=r*3,q=edge3DIndex(e,k,nz),ax=.5*(dv[lx]!+dv[rx]!),ay=.5*(dv[lx+1]!+dv[rx+1]!),az=.5*(dv[lx+2]!+dv[rx+2]!);uT[q]=uT[q]!+ax*n[0]+ay*n[1]+az*n[2];
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
