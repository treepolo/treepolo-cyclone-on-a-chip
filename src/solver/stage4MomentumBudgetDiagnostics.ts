import { EARTH } from '../core/constants.js';
import type { CubedSphereGrid } from '../grid/cubedSphere.js';
import type { VerticalGrid } from '../grid/vertical.js';
import { computeHorizontalMaterialMomentumTendency } from '../physics/horizontalMomentumTransport.js';
import type { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { reconstructCellHorizontalWind, type RotationGeometry } from '../physics/rotation.js';
import { diagnoseAxialAngularMomentumTendency } from './stage4CirculationDiagnostics.js';
import { computeStage4FrozenRhs } from './stage4Rk3SplitCpu.js';
import { computeStage4SlowTendencies } from './stage4SlowTendenciesCpu.js';
import { cell3DIndex, edge3DIndex, w3DIndex, type DryState } from './state.js';

export interface Stage4InstantAamBreakdown {
  mass:number;
  massPlanetary:number;
  massRelative:number;
  pressure:number;
  pressureInterior:number;
  pressureSeam:number;
  pressureSplitClosure:number;
  momentum:number;
  coriolis:number;
  planetaryCoriolisResidual:number;
  relativeMomentumResidual:number;
  drag:number;
  inviscid:number;
  pairClosure:number;
  withDrag:number;
  full:number;
  closure:number;
}

export interface Stage4DirectionalAamBreakdown {
  horizontalRelativeMass:number;
  horizontalMomentum:number;
  horizontalPairResidual:number;
  verticalRelativeMass:number;
  verticalMomentum:number;
  verticalPairResidual:number;
  totalRelativeMass:number;
  totalMomentum:number;
  totalPairResidual:number;
  closure:number;
}

function projectCellVectorTendencyToEdges(
  h:CubedSphereGrid,
  v:VerticalGrid,
  rotation:RotationGeometry,
  cellByK:Float64Array[],
):Float64Array{
  const out=new Float64Array(h.edgeCount*v.nz);
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,l=ge.leftCell,r=ge.rightCell,n=ge.normal;
    for(let k=0;k<v.nz;k++){
      const dv=cellByK[k]!,lo=l*3,ro=r*3;
      const lrx=rotation.radial[lo]!,lry=rotation.radial[lo+1]!,lrz=rotation.radial[lo+2]!;
      const rrx=rotation.radial[ro]!,rry=rotation.radial[ro+1]!,rrz=rotation.radial[ro+2]!;
      const ldot=dv[lo]!*lrx+dv[lo+1]!*lry+dv[lo+2]!*lrz;
      const rdot=dv[ro]!*rrx+dv[ro+1]!*rry+dv[ro+2]!*rrz;
      const lx=dv[lo]!-ldot*lrx,ly=dv[lo+1]!-ldot*lry,lz=dv[lo+2]!-ldot*lrz;
      const rx=dv[ro]!-rdot*rrx,ry=dv[ro+1]!-rdot*rry,rz=dv[ro+2]!-rdot*rrz;
      out[edge3DIndex(e,k,v.nz)]=.5*((lx+rx)*n[0]+(ly+ry)*n[1]+(lz+rz)*n[2]);
    }
  }
  return out;
}

/**
 * Split the material AAM cancellation into horizontal and vertical pieces using
 * exactly the production Stage-4 discrete flux definitions.  Diagnostic only.
 *
 * Horizontal continuity combines rho0*u with the perturbation donor flux, which
 * is exactly the total donor rho*u face flux.  Vertical continuity deliberately
 * mirrors the production reference/perturbation split: rho0(interface)*w plus
 * (rho_donor-rho0(donor-center))*w.  Horizontal momentum uses production
 * MUSCL-BJ; vertical horizontal-momentum transport is the current donor-cell
 * material form.  Their sums must reproduce the existing relative+momentum pair.
 */
export function diagnoseStage4DirectionalAamBreakdown(
  h:CubedSphereGrid,
  v:VerticalGrid,
  ref:ReferenceAtmosphere,
  state:DryState,
  rotation:RotationGeometry,
):Stage4DirectionalAamBreakdown{
  const R=EARTH.radius,nz=v.nz,zeroU=new Float64Array(state.uEdge.length),zeroRho=new Float64Array(state.rhoD.length);
  const rhoH=new Float64Array(state.rhoD.length),rhoV=new Float64Array(state.rhoD.length);

  // Exact production horizontal total continuity carrier.
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,edgeLength=ge.angularLength*R;
    for(let k=0;k<nz;k++){
      const q=edge3DIndex(e,k,nz),vel=state.uEdge[q]!,lc=ge.leftCell,rc=ge.rightCell,l=cell3DIndex(lc,k,nz),r=cell3DIndex(rc,k,nz),up=vel>=0?l:r;
      const fm=state.rhoD[up]!*vel*edgeLength*v.dz[k]!;
      const vl=h.cellAreaUnit[lc]!*R*R*v.dz[k]!,vr=h.cellAreaUnit[rc]!*R*R*v.dz[k]!;
      rhoH[l]-=fm/vl;rhoH[r]+=fm/vr;
    }
  }

  // Exact production vertical reference + perturbation continuity carrier.
  for(let c=0;c<h.cellCount;c++){
    const area=h.cellAreaUnit[c]!*R*R;
    for(let i=1;i<nz;i++){
      const qi=w3DIndex(c,i,nz),vel=state.wInterface[qi]!,srcK=vel>=0?i-1:i,src=cell3DIndex(c,srcK,nz);
      const fm=(ref.rhoInterface[i]!+state.rhoD[src]!-ref.rhoCenter[srcK]!)*vel*area;
      const lower=cell3DIndex(c,i-1,nz),upper=cell3DIndex(c,i,nz);
      rhoV[lower]-=fm/(area*v.dz[i-1]!);rhoV[upper]+=fm/(area*v.dz[i]!);
    }
  }

  const windByK:Float64Array[]=Array.from({length:nz},(_,k)=>reconstructCellHorizontalWind(h,rotation,state,k));
  const hCell=computeHorizontalMaterialMomentumTendency(h,v,state,rotation,'muscl-bj',windByK);
  const hU=projectCellVectorTendencyToEdges(h,v,rotation,hCell);

  const vCell:Float64Array[]=Array.from({length:nz},()=>new Float64Array(h.cellCount*3));
  for(let c=0;c<h.cellCount;c++)for(let k=0;k<nz;k++){
    const o=c*3,dv=vCell[k]!,wc=.5*(state.wInterface[w3DIndex(c,k,nz)]!+state.wInterface[w3DIndex(c,k+1,nz)]!),cur=windByK[k]!;
    if(wc>0&&k>0){
      const below=windByK[k-1]!,dz=v.zCenter[k]!-v.zCenter[k-1]!;
      dv[o]-=wc*(cur[o]!-below[o]!)/dz;dv[o+1]-=wc*(cur[o+1]!-below[o+1]!)/dz;dv[o+2]-=wc*(cur[o+2]!-below[o+2]!)/dz;
    }else if(wc<0&&k<nz-1){
      const above=windByK[k+1]!,dz=v.zCenter[k+1]!-v.zCenter[k]!;
      dv[o]-=wc*(above[o]!-cur[o]!)/dz;dv[o+1]-=wc*(above[o+1]!-cur[o+1]!)/dz;dv[o+2]-=wc*(above[o+2]!-cur[o+2]!)/dz;
    }
  }
  const vU=projectCellVectorTendencyToEdges(h,v,rotation,vCell);

  const hMass=diagnoseAxialAngularMomentumTendency(h,v,state,rhoH,zeroU,rotation).relativeMassRedistributionTorque;
  const vMass=diagnoseAxialAngularMomentumTendency(h,v,state,rhoV,zeroU,rotation).relativeMassRedistributionTorque;
  const hMom=diagnoseAxialAngularMomentumTendency(h,v,state,zeroRho,hU,rotation).velocityTorque;
  const vMom=diagnoseAxialAngularMomentumTendency(h,v,state,zeroRho,vU,rotation).velocityTorque;
  const totalRelativeMass=hMass+vMass,totalMomentum=hMom+vMom,totalPairResidual=totalRelativeMass+totalMomentum;

  const fullContinuity=computeStage4FrozenRhs(h,v,ref,state,{momentumTransport:false,coriolis:false,heldSuarez:false},rotation);
  const fullMomentum=computeStage4SlowTendencies(h,v,ref,state,{momentumTransport:true,coriolis:false,heldSuarez:false},rotation);
  const checkMass=diagnoseAxialAngularMomentumTendency(h,v,state,fullContinuity.rhoD,zeroU,rotation).relativeMassRedistributionTorque;
  const checkMom=diagnoseAxialAngularMomentumTendency(h,v,state,zeroRho,fullMomentum.uEdge,rotation).velocityTorque;
  const closure=(totalRelativeMass-checkMass)+(totalMomentum-checkMom);
  return{
    horizontalRelativeMass:hMass,horizontalMomentum:hMom,horizontalPairResidual:hMass+hMom,
    verticalRelativeMass:vMass,verticalMomentum:vMom,verticalPairResidual:vMass+vMom,
    totalRelativeMass,totalMomentum,totalPairResidual,closure,
  };
}

/**
 * Decompose the frozen Stage-4 RHS into axial-angular-momentum torque pairs.
 *
 * This is diagnostic only: it does not alter the state or apply any AAM fixer.
 * The two physically coupled pairs are
 *
 *   planetary mass redistribution + Coriolis
 *   relative mass redistribution + material momentum transport
 *
 * and pressure is reported independently, including cubed-sphere seam/interior
 * pieces.  A non-zero pair residual therefore identifies a discrete operator
 * mismatch without prescribing what the circulation should look like.
 */
export function diagnoseStage4InstantAamBreakdown(
  h:CubedSphereGrid,
  v:VerticalGrid,
  ref:ReferenceAtmosphere,
  state:DryState,
  rotation:RotationGeometry,
):Stage4InstantAamBreakdown {
  const zeroRho=new Float64Array(state.rhoD.length);
  const zeroU=new Float64Array(state.uEdge.length);
  const pressureAndContinuity=computeStage4FrozenRhs(h,v,ref,state,{momentumTransport:false,coriolis:false,heldSuarez:false},rotation);
  const momentumT=computeStage4SlowTendencies(h,v,ref,state,{momentumTransport:true,coriolis:false,heldSuarez:false},rotation);
  const coriolisT=computeStage4SlowTendencies(h,v,ref,state,{momentumTransport:false,coriolis:true,heldSuarez:false},rotation);
  const dragT=computeStage4SlowTendencies(h,v,ref,state,{momentumTransport:false,coriolis:false,heldSuarez:true},rotation);
  const fullT=computeStage4FrozenRhs(h,v,ref,state,{momentumTransport:true,coriolis:true,heldSuarez:true},rotation);

  const massDiag=diagnoseAxialAngularMomentumTendency(h,v,state,pressureAndContinuity.rhoD,zeroU,rotation);
  const pressure=diagnoseAxialAngularMomentumTendency(h,v,state,zeroRho,pressureAndContinuity.uEdge,rotation).totalTorque;
  const momentum=diagnoseAxialAngularMomentumTendency(h,v,state,zeroRho,momentumT.uEdge,rotation).totalTorque;
  const coriolis=diagnoseAxialAngularMomentumTendency(h,v,state,zeroRho,coriolisT.uEdge,rotation).totalTorque;
  const drag=diagnoseAxialAngularMomentumTendency(h,v,state,zeroRho,dragT.uEdge,rotation).totalTorque;
  const full=diagnoseAxialAngularMomentumTendency(h,v,state,fullT.rhoD,fullT.uEdge,rotation).totalTorque;

  const pressureInteriorU=new Float64Array(state.uEdge.length);
  const pressureSeamU=new Float64Array(state.uEdge.length);
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!;
    const seam=h.cellPanel[ge.leftCell]!==h.cellPanel[ge.rightCell];
    for(let k=0;k<v.nz;k++){
      const q=e*v.nz+k;
      if(seam)pressureSeamU[q]=pressureAndContinuity.uEdge[q]!;
      else pressureInteriorU[q]=pressureAndContinuity.uEdge[q]!;
    }
  }
  const pressureInterior=diagnoseAxialAngularMomentumTendency(h,v,state,zeroRho,pressureInteriorU,rotation).totalTorque;
  const pressureSeam=diagnoseAxialAngularMomentumTendency(h,v,state,zeroRho,pressureSeamU,rotation).totalTorque;
  const pressureSplitClosure=pressure-pressureInterior-pressureSeam;

  const mass=massDiag.massRedistributionTorque;
  const massPlanetary=massDiag.planetaryMassRedistributionTorque;
  const massRelative=massDiag.relativeMassRedistributionTorque;
  const planetaryCoriolisResidual=massPlanetary+coriolis;
  const relativeMomentumResidual=massRelative+momentum;
  const inviscid=mass+pressure+momentum+coriolis;
  const pairClosure=inviscid-(planetaryCoriolisResidual+relativeMomentumResidual+pressure);
  const withDrag=inviscid+drag;

  return{
    mass,massPlanetary,massRelative,
    pressure,pressureInterior,pressureSeam,pressureSplitClosure,
    momentum,coriolis,planetaryCoriolisResidual,relativeMomentumResidual,
    drag,inviscid,pairClosure,withDrag,full,closure:full-withDrag,
  };
}
