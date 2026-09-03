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

export interface Stage4FrozenRhs {rhoD:Float64Array;rhoThetaM:Float64Array;uEdge:Float64Array;wInterface:Float64Array;pressure:Float64Array;dpdRhoTheta:Float64Array;}
export interface Stage4Rk3SplitOptions extends Stage4SlowOptions {divergenceDamping?:boolean;topAbsorber?:boolean;}
function copyStateInto(dst:DryState,src:DryState):void{dst.rhoD.set(src.rhoD);dst.rhoThetaM.set(src.rhoThetaM);dst.uEdge.set(src.uEdge);dst.wInterface.set(src.wInterface);dst.time=src.time;}

/** Frozen full predictor RHS = slow Fpert/advection + predictor-value fast Fref/pressure/gravity. */
export function computeStage4FrozenRhs(h:CubedSphereGrid,v:VerticalGrid,ref:ReferenceAtmosphere,predictor:DryState,options:Stage4SlowOptions={},rotation?:RotationGeometry):Stage4FrozenRhs{
  const g=rotation??buildRotationGeometry(h),slow=computeStage4SlowTendencies(h,v,ref,predictor,options,g),nz=v.nz,R=EARTH.radius;
  const rho=slow.rhoD.slice(),x=slow.rhoThetaM.slice(),u=slow.uEdge.slice(),w=slow.wInterface.slice(),pressure=new Float64Array(predictor.rhoThetaM.length),dpdRhoTheta=new Float64Array(predictor.rhoThetaM.length);
  for(let q=0;q<pressure.length;q++){const p=pressureFromRhoTheta(predictor.rhoThetaM[q]!);pressure[q]=p;dpdRhoTheta[q]=DRY_AIR.gamma*p/Math.max(predictor.rhoThetaM[q]!,1e-12);}

  // Horizontal reference-state continuity at the predictor. Combined with the
  // slow perturbation flux this exactly reconstructs full rho/X upwind flux at
  // delta=0 because rho0 and X0 are horizontally uniform at each level.
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,edgeLength=ge.angularLength*R;
    for(let k=0;k<nz;k++){
      const qe=edge3DIndex(e,k,nz),vel=predictor.uEdge[qe]!,fm=ref.rhoCenter[k]!*vel*edgeLength*v.dz[k]!,fx=ref.rhoThetaCenter[k]!*vel*edgeLength*v.dz[k]!,l=ge.leftCell,r=ge.rightCell,vl=h.cellAreaUnit[l]!*R*R*v.dz[k]!,vr=h.cellAreaUnit[r]!*R*R*v.dz[k]!,ql=cell3DIndex(l,k,nz),qr=cell3DIndex(r,k,nz);
      rho[ql]=rho[ql]!-fm/vl;rho[qr]=rho[qr]!+fm/vr;x[ql]=x[ql]!-fx/vl;x[qr]=x[qr]!+fx/vr;
    }
  }
  // Vertical reference-state continuity at the predictor.
  for(let c=0;c<h.cellCount;c++)for(let k=0;k<nz;k++){
    const q=cell3DIndex(c,k,nz),wb=predictor.wInterface[w3DIndex(c,k,nz)]!,wt=predictor.wInterface[w3DIndex(c,k+1,nz)]!,dz=v.dz[k]!;
    rho[q]=rho[q]!-(ref.rhoInterface[k+1]!*wt-ref.rhoInterface[k]!*wb)/dz;x[q]=x[q]!-(ref.rhoThetaInterface[k+1]!*wt-ref.rhoThetaInterface[k]!*wb)/dz;
  }

  // Predictor pressure accelerations.
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,dist=Math.max(ge.centerDistanceAngle*R,1);
    for(let k=0;k<nz;k++){
      const l=cell3DIndex(ge.leftCell,k,nz),r=cell3DIndex(ge.rightCell,k,nz),q=edge3DIndex(e,k,nz),rhoAvg=Math.max(.5*(predictor.rhoD[l]!+predictor.rhoD[r]!),1e-12);u[q]=u[q]!-(pressure[r]!-pressure[l]!)/(rhoAvg*dist);
    }
  }
  const pr=new Float64Array(nz),px=new Float64Array(nz),pw=new Float64Array(nz+1);
  for(let c=0;c<h.cellCount;c++){
    for(let k=0;k<nz;k++){const q=cell3DIndex(c,k,nz);pr[k]=predictor.rhoD[q]!;px[k]=predictor.rhoThetaM[q]!;}
    for(let i=0;i<=nz;i++)pw[i]=predictor.wInterface[w3DIndex(c,i,nz)]!;
    const acc=predictorVerticalPressureBuoyancyAcceleration(v,ref,{rho:pr,rhoTheta:px,w:pw});for(let i=1;i<nz;i++){const q=w3DIndex(c,i,nz);w[q]=w[q]!+acc[i]!;}
  }
  return{rhoD:rho,rhoThetaM:x,uEdge:u,wInterface:w,pressure,dpdRhoTheta};
}

/** One predictor-relative global acoustic small step. */
export function advanceStage4AcousticSmallStep(h:CubedSphereGrid,v:VerticalGrid,ref:ReferenceAtmosphere,predictor:DryState,acoustic:DryState,frozen:Stage4FrozenRhs,dt:number,options:Stage4Rk3SplitOptions={},rayleighRates?:ArrayLike<number>):void{
  if(!(dt>0))throw new Error('Stage4 acoustic dt must be positive');const nz=v.nz,R=EARTH.radius;
  // Forward pressure correction for horizontal velocity.
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,dist=Math.max(ge.centerDistanceAngle*R,1);
    for(let k=0;k<nz;k++){
      const l=cell3DIndex(ge.leftCell,k,nz),r=cell3DIndex(ge.rightCell,k,nz),q=edge3DIndex(e,k,nz),drL=acoustic.rhoD[l]!-predictor.rhoD[l]!,drR=acoustic.rhoD[r]!-predictor.rhoD[r]!,dxL=acoustic.rhoThetaM[l]!-predictor.rhoThetaM[l]!,dxR=acoustic.rhoThetaM[r]!-predictor.rhoThetaM[r]!,rhoAvg=Math.max(.5*(predictor.rhoD[l]!+predictor.rhoD[r]!),1e-12),dRhoAvg=.5*(drL+drR),dpPred=frozen.pressure[r]!-frozen.pressure[l]!,dDp=frozen.dpdRhoTheta[r]!*dxR-frozen.dpdRhoTheta[l]!*dxL,linearPressure=-dDp/(rhoAvg*dist)+dpPred*dRhoAvg/(rhoAvg*rhoAvg*dist);
      acoustic.uEdge[q]=acoustic.uEdge[q]!+dt*(frozen.uEdge[q]!+linearPressure);
    }
  }

  // Backward reference-continuity correction. Nonlinear perturbation transport
  // is frozen in the RK slow RHS; only rho0*delta-u and X0*delta-u are fast.
  const hRhoCorr=new Float64Array(acoustic.rhoD.length),hXCorr=new Float64Array(acoustic.rhoThetaM.length);
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,edgeLength=ge.angularLength*R;
    for(let k=0;k<nz;k++){
      const l=cell3DIndex(ge.leftCell,k,nz),r=cell3DIndex(ge.rightCell,k,nz),q=edge3DIndex(e,k,nz),du=acoustic.uEdge[q]!-predictor.uEdge[q]!,A=edgeLength*v.dz[k]!,fm=ref.rhoCenter[k]!*du*A,fx=ref.rhoThetaCenter[k]!*du*A;hRhoCorr[l]=hRhoCorr[l]!-fm;hRhoCorr[r]=hRhoCorr[r]!+fm;hXCorr[l]=hXCorr[l]!-fx;hXCorr[r]=hXCorr[r]!+fx;
    }
  }
  for(let c=0;c<h.cellCount;c++)for(let k=0;k<nz;k++){const q=cell3DIndex(c,k,nz),vol=h.cellAreaUnit[c]!*R*R*v.dz[k]!;hRhoCorr[q]=hRhoCorr[q]!/vol;hXCorr[q]=hXCorr[q]!/vol;}

  // Vertically implicit predictor-relative acoustic/gravity correction.
  const pcol:AcousticColumnState={rho:new Float64Array(nz),rhoTheta:new Float64Array(nz),w:new Float64Array(nz+1)},acol:AcousticColumnState={rho:new Float64Array(nz),rhoTheta:new Float64Array(nz),w:new Float64Array(nz+1)},rcol:AcousticColumnTendency={rho:new Float64Array(nz),rhoTheta:new Float64Array(nz),w:new Float64Array(nz+1)};
  for(let c=0;c<h.cellCount;c++){
    for(let k=0;k<nz;k++){const q=cell3DIndex(c,k,nz);pcol.rho[k]=predictor.rhoD[q]!;pcol.rhoTheta[k]=predictor.rhoThetaM[q]!;acol.rho[k]=acoustic.rhoD[q]!;acol.rhoTheta[k]=acoustic.rhoThetaM[q]!;rcol.rho[k]=frozen.rhoD[q]!+hRhoCorr[q]!;rcol.rhoTheta[k]=frozen.rhoThetaM[q]!+hXCorr[q]!;}
    for(let i=0;i<=nz;i++){const q=w3DIndex(c,i,nz);pcol.w[i]=predictor.wInterface[q]!;acol.w[i]=acoustic.wInterface[q]!;rcol.w[i]=frozen.wInterface[q]!;}
    predictorRelativeVerticalAcousticStep(v,ref,pcol,acol,rcol,dt,STAGE4_HEVI_OFFCENTERING,options.topAbsorber===false?undefined:rayleighRates);
    for(let k=0;k<nz;k++){const q=cell3DIndex(c,k,nz);acoustic.rhoD[q]=acol.rho[k]!;acoustic.rhoThetaM[q]=acol.rhoTheta[k]!;}for(let i=0;i<=nz;i++)acoustic.wInterface[w3DIndex(c,i,nz)]=acol.w[i]!;
  }
  if(options.divergenceDamping!==false)applyAcousticDivergenceDamping(h,v,ref,acoustic,acousticDivergenceCoefficientForDt(dt));
}

export class Stage4Rk3SplitCpu{
  readonly rotation:RotationGeometry;readonly rayleighRates:Float64Array;
  constructor(public readonly h:CubedSphereGrid,public readonly v:VerticalGrid,public readonly ref:ReferenceAtmosphere,public readonly acousticRatio=4){this.rotation=buildRotationGeometry(h);this.rayleighRates=Float64Array.from(buildModelTopSpongeRates(v));buildRk3SplitSchedule(acousticRatio);}
  step(state:DryState,dt:number,options:Stage4Rk3SplitOptions={}):void{
    if(!(dt>0))throw new Error('Stage4 RK3 dt must be positive');const base=cloneState(state),schedule=buildRk3SplitSchedule(this.acousticRatio);let predictor=cloneState(base);
    for(const stage of schedule){const frozen=computeStage4FrozenRhs(this.h,this.v,this.ref,predictor,options,this.rotation),acoustic=cloneState(base),dtFast=dt*stage.acousticDtFraction;for(let n=0;n<stage.acousticSteps;n++)advanceStage4AcousticSmallStep(this.h,this.v,this.ref,predictor,acoustic,frozen,dtFast,options,this.rayleighRates);predictor=acoustic;}
    copyStateInto(state,predictor);state.time=base.time+dt;
  }
}
