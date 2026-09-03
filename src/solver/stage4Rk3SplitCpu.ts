import { DRY_AIR, EARTH } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { acousticDivergenceCoefficientForDt, applyAcousticDivergenceDamping } from '../physics/acousticDivergenceDamping.js';
import { buildModelTopSpongeRates } from '../physics/modelTopSponge.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { buildRotationGeometry, RotationGeometry } from '../physics/rotation.js';
import { pressureFromRhoTheta } from '../physics/thermodynamics.js';
import { AcousticColumnState, AcousticColumnTendency, predictorRelativeVerticalAcousticStep, predictorVerticalPressureBuoyancyAcceleration } from './stage4AcousticColumnCpu.js';
import { buildRk3SplitSchedule } from './rk3SplitSchedule.js';
import { STAGE4_HEVI_OFFCENTERING } from './stage4Config.js';
import { computeStage4SlowTendencies, Stage4SlowOptions } from './stage4SlowTendenciesCpu.js';
import { cloneState, DryState, cell3DIndex, edge3DIndex, w3DIndex } from './state.js';

export interface Stage4FrozenRhs {
  rhoD:Float64Array;
  rhoThetaM:Float64Array;
  uEdge:Float64Array;
  wInterface:Float64Array;
  pressure:Float64Array;
  dpdRhoTheta:Float64Array;
}
export interface Stage4Rk3SplitOptions extends Stage4SlowOptions {
  divergenceDamping?:boolean;
  topAbsorber?:boolean;
}

function copyStateInto(dst:DryState,src:DryState):void{
  dst.rhoD.set(src.rhoD);dst.rhoThetaM.set(src.rhoThetaM);dst.uEdge.set(src.uEdge);dst.wInterface.set(src.wInterface);dst.time=src.time;
}

/**
 * Frozen full predictor RHS used inside one RK acoustic loop.
 * Slow transport/rotation/forcing is evaluated once. Predictor pressure and
 * well-balanced vertical pressure+buoyancy are then added as the zero-delta
 * part of the acoustic correction equations. Changes away from the predictor
 * are handled by linearized fast operators during each small step.
 */
export function computeStage4FrozenRhs(
  h:CubedSphereGrid,
  v:VerticalGrid,
  ref:ReferenceAtmosphere,
  predictor:DryState,
  options:Stage4SlowOptions={},
  rotation?:RotationGeometry,
):Stage4FrozenRhs {
  const g=rotation??buildRotationGeometry(h),slow=computeStage4SlowTendencies(h,v,ref,predictor,options,g),nz=v.nz;
  const u=slow.uEdge.slice(),w=slow.wInterface.slice(),pressure=new Float64Array(predictor.rhoThetaM.length),dpdRhoTheta=new Float64Array(predictor.rhoThetaM.length);
  for(let q=0;q<pressure.length;q++){
    const p=pressureFromRhoTheta(predictor.rhoThetaM[q]!);pressure[q]=p;dpdRhoTheta[q]=DRY_AIR.gamma*p/Math.max(predictor.rhoThetaM[q]!,1e-12);
  }
  // Predictor horizontal pressure acceleration: the full current Stage-4 C-grid
  // pressure gradient. Hydrostatic reference pressure has no horizontal gradient.
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,dist=Math.max(ge.centerDistanceAngle*EARTH.radius,1);
    for(let k=0;k<nz;k++){
      const l=cell3DIndex(ge.leftCell,k,nz),r=cell3DIndex(ge.rightCell,k,nz),q=edge3DIndex(e,k,nz),rhoAvg=Math.max(.5*(predictor.rhoD[l]!+predictor.rhoD[r]!),1e-12);
      u[q]=u[q]!-(pressure[r]!-pressure[l]!)/(rhoAvg*dist);
    }
  }
  // Predictor well-balanced vertical pressure + buoyancy acceleration.
  const pr=new Float64Array(nz),px=new Float64Array(nz),pw=new Float64Array(nz+1);
  for(let c=0;c<h.cellCount;c++){
    for(let k=0;k<nz;k++){const q=cell3DIndex(c,k,nz);pr[k]=predictor.rhoD[q]!;px[k]=predictor.rhoThetaM[q]!;}
    for(let i=0;i<=nz;i++)pw[i]=predictor.wInterface[w3DIndex(c,i,nz)]!;
    const acc=predictorVerticalPressureBuoyancyAcceleration(v,ref,{rho:pr,rhoTheta:px,w:pw});
    for(let i=1;i<nz;i++){const q=w3DIndex(c,i,nz);w[q]=w[q]!+acc[i]!;}
  }
  return{rhoD:slow.rhoD,rhoThetaM:slow.rhoThetaM,uEdge:u,wInterface:w,pressure,dpdRhoTheta};
}

/**
 * One predictor-relative acoustic small step for the global height-coordinate
 * Stage-4 state. Horizontal acoustics use a forward-backward C-grid update:
 * pressure changes edge-normal velocity first, then the updated velocity
 * perturbation enters centered linearized mass/thermodynamic fluxes. Vertical
 * acoustic/gravity coupling is solved implicitly column by column.
 */
export function advanceStage4AcousticSmallStep(
  h:CubedSphereGrid,
  v:VerticalGrid,
  ref:ReferenceAtmosphere,
  predictor:DryState,
  acoustic:DryState,
  frozen:Stage4FrozenRhs,
  dt:number,
  options:Stage4Rk3SplitOptions={},
  rayleighRates?:ArrayLike<number>,
):void {
  if(!(dt>0))throw new Error('Stage4 acoustic dt must be positive');
  const nz=v.nz,R=EARTH.radius;
  // 1) Forward horizontal momentum perturbation using predictor-frozen RHS and
  // the first derivative of -(1/rho) grad(p) about the RK predictor.
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,dist=Math.max(ge.centerDistanceAngle*R,1);
    for(let k=0;k<nz;k++){
      const l=cell3DIndex(ge.leftCell,k,nz),r=cell3DIndex(ge.rightCell,k,nz),q=edge3DIndex(e,k,nz);
      const drL=acoustic.rhoD[l]!-predictor.rhoD[l]!,drR=acoustic.rhoD[r]!-predictor.rhoD[r]!,dxL=acoustic.rhoThetaM[l]!-predictor.rhoThetaM[l]!,dxR=acoustic.rhoThetaM[r]!-predictor.rhoThetaM[r]!;
      const rhoAvg=Math.max(.5*(predictor.rhoD[l]!+predictor.rhoD[r]!),1e-12),dRhoAvg=.5*(drL+drR),dpPred=frozen.pressure[r]!-frozen.pressure[l]!,dDp=frozen.dpdRhoTheta[r]!*dxR-frozen.dpdRhoTheta[l]!*dxL;
      const linearPressure=-dDp/(rhoAvg*dist)+dpPred*dRhoAvg/(rhoAvg*rhoAvg*dist);
      acoustic.uEdge[q]=acoustic.uEdge[q]!+dt*(frozen.uEdge[q]!+linearPressure);
    }
  }

  // 2) Backward scalar part of the horizontal acoustic pair.  Centered
  // predictor-linearized flux perturbations are used; nonlinear/upwind fluxes
  // are already represented in the frozen large-step RHS.
  const hRhoCorr=new Float64Array(acoustic.rhoD.length),hXCorr=new Float64Array(acoustic.rhoThetaM.length);
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,edgeLength=ge.angularLength*R;
    for(let k=0;k<nz;k++){
      const l=cell3DIndex(ge.leftCell,k,nz),r=cell3DIndex(ge.rightCell,k,nz),q=edge3DIndex(e,k,nz),du=acoustic.uEdge[q]!-predictor.uEdge[q]!,uP=predictor.uEdge[q]!,drAvg=.5*((acoustic.rhoD[l]!-predictor.rhoD[l]!)+(acoustic.rhoD[r]!-predictor.rhoD[r]!)),dxAvg=.5*((acoustic.rhoThetaM[l]!-predictor.rhoThetaM[l]!)+(acoustic.rhoThetaM[r]!-predictor.rhoThetaM[r]!)),rhoAvg=.5*(predictor.rhoD[l]!+predictor.rhoD[r]!),xAvg=.5*(predictor.rhoThetaM[l]!+predictor.rhoThetaM[r]!),A=edgeLength*v.dz[k]!;
      const fm=(rhoAvg*du+uP*drAvg)*A,fx=(xAvg*du+uP*dxAvg)*A;
      hRhoCorr[l]=hRhoCorr[l]!-fm;hRhoCorr[r]=hRhoCorr[r]!+fm;hXCorr[l]=hXCorr[l]!-fx;hXCorr[r]=hXCorr[r]!+fx;
    }
  }
  for(let c=0;c<h.cellCount;c++)for(let k=0;k<nz;k++){
    const q=cell3DIndex(c,k,nz),vol=h.cellAreaUnit[c]!*R*R*v.dz[k]!;hRhoCorr[q]=hRhoCorr[q]!/vol;hXCorr[q]=hXCorr[q]!/vol;
  }

  // 3) Vertically implicit predictor-relative acoustic/gravity correction.
  const pcol:AcousticColumnState={rho:new Float64Array(nz),rhoTheta:new Float64Array(nz),w:new Float64Array(nz+1)};
  const acol:AcousticColumnState={rho:new Float64Array(nz),rhoTheta:new Float64Array(nz),w:new Float64Array(nz+1)};
  const rcol:AcousticColumnTendency={rho:new Float64Array(nz),rhoTheta:new Float64Array(nz),w:new Float64Array(nz+1)};
  for(let c=0;c<h.cellCount;c++){
    for(let k=0;k<nz;k++){
      const q=cell3DIndex(c,k,nz);pcol.rho[k]=predictor.rhoD[q]!;pcol.rhoTheta[k]=predictor.rhoThetaM[q]!;acol.rho[k]=acoustic.rhoD[q]!;acol.rhoTheta[k]=acoustic.rhoThetaM[q]!;rcol.rho[k]=frozen.rhoD[q]!+hRhoCorr[q]!;rcol.rhoTheta[k]=frozen.rhoThetaM[q]!+hXCorr[q]!;
    }
    for(let i=0;i<=nz;i++){
      const q=w3DIndex(c,i,nz);pcol.w[i]=predictor.wInterface[q]!;acol.w[i]=acoustic.wInterface[q]!;rcol.w[i]=frozen.wInterface[q]!;
    }
    predictorRelativeVerticalAcousticStep(v,ref,pcol,acol,rcol,dt,STAGE4_HEVI_OFFCENTERING,options.topAbsorber===false?undefined:rayleighRates);
    for(let k=0;k<nz;k++){const q=cell3DIndex(c,k,nz);acoustic.rhoD[q]=acol.rho[k]!;acoustic.rhoThetaM[q]=acol.rhoTheta[k]!;}
    for(let i=0;i<=nz;i++)acoustic.wInterface[w3DIndex(c,i,nz)]=acol.w[i]!;
  }

  if(options.divergenceDamping!==false)applyAcousticDivergenceDamping(h,v,ref,acoustic,acousticDivergenceCoefficientForDt(dt));
}

/** CPU Float64 reference for the locked Wicker-Skamarock/ARW-style RK3 split. */
export class Stage4Rk3SplitCpu {
  readonly rotation:RotationGeometry;
  readonly rayleighRates:Float64Array;
  constructor(public readonly h:CubedSphereGrid,public readonly v:VerticalGrid,public readonly ref:ReferenceAtmosphere,public readonly acousticRatio=4){
    this.rotation=buildRotationGeometry(h);this.rayleighRates=buildModelTopSpongeRates(v);buildRk3SplitSchedule(acousticRatio);
  }
  step(state:DryState,dt:number,options:Stage4Rk3SplitOptions={}):void{
    if(!(dt>0))throw new Error('Stage4 RK3 dt must be positive');
    const base=cloneState(state),schedule=buildRk3SplitSchedule(this.acousticRatio);let predictor=cloneState(base);
    for(const stage of schedule){
      const frozen=computeStage4FrozenRhs(this.h,this.v,this.ref,predictor,options,this.rotation),acoustic=cloneState(base),dtFast=dt*stage.acousticDtFraction;
      for(let n=0;n<stage.acousticSteps;n++)advanceStage4AcousticSmallStep(this.h,this.v,this.ref,predictor,acoustic,frozen,dtFast,options,this.rayleighRates);
      predictor=acoustic;
    }
    copyStateInto(state,predictor);state.time=base.time+dt;
  }
}
