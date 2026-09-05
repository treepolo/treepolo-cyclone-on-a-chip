import type { CubedSphereGrid } from '../grid/cubedSphere.js';
import type { VerticalGrid } from '../grid/vertical.js';
import type { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import type { RotationGeometry } from '../physics/rotation.js';
import { diagnoseAxialAngularMomentumTendency } from './stage4CirculationDiagnostics.js';
import { computeStage4FrozenRhs } from './stage4Rk3SplitCpu.js';
import { computeStage4SlowTendencies } from './stage4SlowTendenciesCpu.js';
import type { DryState } from './state.js';

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
