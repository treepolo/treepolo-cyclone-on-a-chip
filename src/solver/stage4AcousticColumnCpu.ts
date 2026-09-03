import { DRY_AIR, EARTH } from '../core/constants.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { pressureFromRhoTheta } from '../physics/thermodynamics.js';
import { solveTridiagonal } from './tridiagonal.js';

export interface AcousticColumnState {
  rho:Float64Array;
  rhoTheta:Float64Array;
  w:Float64Array;
}
export interface AcousticColumnTendency {
  rho:Float64Array;
  rhoTheta:Float64Array;
  w:Float64Array;
}
export interface AcousticColumnStepResult {
  /** Full predictor + perturbation reference-state mass flux, per unit area. */
  referenceMassFlux:Float64Array;
  /** Full predictor + perturbation reference rhoTheta flux, per unit area. */
  referenceRhoThetaFlux:Float64Array;
}

function checkShapes(v:VerticalGrid,p:AcousticColumnState,a:AcousticColumnState,r:AcousticColumnTendency):void{
  const nz=v.nz;
  if(p.rho.length!==nz||p.rhoTheta.length!==nz||p.w.length!==nz+1||a.rho.length!==nz||a.rhoTheta.length!==nz||a.w.length!==nz+1||r.rho.length!==nz||r.rhoTheta.length!==nz||r.w.length!==nz+1)throw new Error('acoustic column shape mismatch');
}

/**
 * Frozen full-state vertical pressure + buoyancy acceleration at an RK predictor.
 * This is the differential-form analogue of the existing well-balanced Stage-4
 * pressure-perturbation plus explicit-buoyancy ordering, and is exactly zero for
 * the hydrostatic reference atmosphere.
 */
export function predictorVerticalPressureBuoyancyAcceleration(
  v:VerticalGrid,
  ref:ReferenceAtmosphere,
  predictor:AcousticColumnState,
):Float64Array {
  const nz=v.nz,out=new Float64Array(nz+1),p=new Float64Array(nz);
  for(let k=0;k<nz;k++)p[k]=pressureFromRhoTheta(predictor.rhoTheta[k]!);
  for(let i=1;i<nz;i++){
    const l=i-1,u=i,dzc=v.zCenter[u]!-v.zCenter[l]!,rr=.5*(predictor.rho[l]!+predictor.rho[u]!),r0=.5*(ref.rhoCenter[l]!+ref.rhoCenter[u]!),den=Math.max(.5*(r0+rr),1e-12);
    const dpPrime=(p[u]!-ref.pCenter[u]!)-(p[l]!-ref.pCenter[l]!);
    out[i]=-dpPrime/(den*dzc)-EARTH.gravity*(rr-r0)/Math.max(rr,1e-12);
  }
  return out;
}

/**
 * Predictor-relative vertically implicit acoustic/gravity correction.
 *
 * Let delta = acousticState - predictor. During an RK stage the expensive RHS
 * evaluated at the predictor is frozen. The small-step equations integrated
 * here are
 *
 *   d(delta rho)/dt = R_rho* - div_z(rho0 delta w)
 *   d(delta X  )/dt = R_X*   - div_z(X0   delta w)
 *   d(delta w  )/dt = R_w* + L*_pgb(delta rho,delta X) - r(z)(w*+delta w)
 *
 * where X=rho*theta and L*_pgb is the exact first derivative, about the latest
 * RK predictor, of this project's well-balanced pressure-perturbation plus
 * buoyancy vertical acceleration. The coupled terms are theta-off-centred and
 * reduced to a tridiagonal solve for delta w. This is a new derivation; it does
 * not reuse the rejected old-density-buoyancy-in-HEVI forcing experiment.
 */
export function predictorRelativeVerticalAcousticStep(
  v:VerticalGrid,
  ref:ReferenceAtmosphere,
  predictor:AcousticColumnState,
  acousticState:AcousticColumnState,
  frozenRhs:AcousticColumnTendency,
  dt:number,
  offCentering=0,
  rayleighRates?:ArrayLike<number>,
):AcousticColumnStepResult {
  checkShapes(v,predictor,acousticState,frozenRhs);
  if(!(dt>0))throw new Error('acoustic dt must be positive');
  if(!(offCentering>=0&&offCentering<1))throw new Error('acoustic offCentering must be in [0,1)');
  if(rayleighRates&&rayleighRates.length!==v.nz+1)throw new Error('acoustic Rayleigh profile shape mismatch');
  const nz=v.nz,theta=.5*(1+offCentering),oldWeight=1-theta;
  const drOld=new Float64Array(nz),dxOld=new Float64Array(nz),dwOld=new Float64Array(nz+1),pPred=new Float64Array(nz),dpdX=new Float64Array(nz);
  for(let k=0;k<nz;k++){
    drOld[k]=acousticState.rho[k]!-predictor.rho[k]!;
    dxOld[k]=acousticState.rhoTheta[k]!-predictor.rhoTheta[k]!;
    pPred[k]=pressureFromRhoTheta(predictor.rhoTheta[k]!);
    dpdX[k]=DRY_AIR.gamma*pPred[k]!/Math.max(predictor.rhoTheta[k]!,1e-12);
  }
  for(let i=0;i<=nz;i++)dwOld[i]=acousticState.w[i]!-predictor.w[i]!;

  const divergence=(profile:ArrayLike<number>,w:ArrayLike<number>,k:number)=>(profile[k+1]!*w[k+1]!-profile[k]!*w[k]!)/v.dz[k]!;
  const drBase=new Float64Array(nz),dxBase=new Float64Array(nz);
  for(let k=0;k<nz;k++){
    drBase[k]=drOld[k]!+dt*frozenRhs.rho[k]!-dt*oldWeight*divergence(ref.rhoInterface,dwOld,k);
    dxBase[k]=dxOld[k]!+dt*frozenRhs.rhoTheta[k]!-dt*oldWeight*divergence(ref.rhoThetaInterface,dwOld,k);
  }

  // Linearized change in this project's vertical pressure+buoyancy acceleration.
  const linearAccel=(i:number,drL:number,drU:number,dxL:number,dxU:number):number=>{
    const l=i-1,u=i,dzc=v.zCenter[u]!-v.zCenter[l]!,rr=.5*(predictor.rho[l]!+predictor.rho[u]!),r0=.5*(ref.rhoCenter[l]!+ref.rhoCenter[u]!),den=Math.max(.5*(r0+rr),1e-12),drr=.5*(drL+drU),dden=.25*(drL+drU);
    const dpPrime=(pPred[u]!-ref.pCenter[u]!)-(pPred[l]!-ref.pCenter[l]!);
    const dDp=dpdX[u]!*dxU-dpdX[l]!*dxL;
    const pressure=-dDp/(den*dzc)+dpPrime*dden/(den*den*dzc);
    const buoyancy=-EARTH.gravity*r0*drr/Math.max(rr*rr,1e-24);
    return pressure+buoyancy;
  };
  const lOld=new Float64Array(nz+1);
  for(let i=1;i<nz;i++)lOld[i]=linearAccel(i,drOld[i-1]!,drOld[i]!,dxOld[i-1]!,dxOld[i]!);

  // Given the three local unknown interface perturbation velocities around i,
  // reconstruct the new-time scalar perturbations in layers i-1 and i, then L.
  const lNewLocal=(i:number,wm:number,wi:number,wp:number):number=>{
    const l=i-1,u=i;
    const drL=drBase[l]!-dt*theta*(ref.rhoInterface[i]!*wi-ref.rhoInterface[i-1]!*wm)/v.dz[l]!;
    const drU=drBase[u]!-dt*theta*(ref.rhoInterface[i+1]!*wp-ref.rhoInterface[i]!*wi)/v.dz[u]!;
    const dxL=dxBase[l]!-dt*theta*(ref.rhoThetaInterface[i]!*wi-ref.rhoThetaInterface[i-1]!*wm)/v.dz[l]!;
    const dxU=dxBase[u]!-dt*theta*(ref.rhoThetaInterface[i+1]!*wp-ref.rhoThetaInterface[i]!*wi)/v.dz[u]!;
    return linearAccel(i,drL,drU,dxL,dxU);
  };

  const n=nz-1,lo=new Float64Array(n),di=new Float64Array(n),up=new Float64Array(n),rhs=new Float64Array(n),sol=new Float64Array(n);
  for(let ii=0;ii<n;ii++){
    const i=ii+1,l0=lNewLocal(i,0,0,0),alm=lNewLocal(i,1,0,0)-l0,ali=lNewLocal(i,0,1,0)-l0,alp=lNewLocal(i,0,0,1)-l0,rate=rayleighRates?Math.max(0,Number(rayleighRates[i]!)):0;
    lo[ii]=-dt*theta*alm;
    di[ii]=1+dt*rate-dt*theta*ali;
    up[ii]=-dt*theta*alp;
    rhs[ii]=dwOld[i]!+dt*frozenRhs.w[i]!+dt*oldWeight*lOld[i]!+dt*theta*l0-dt*rate*predictor.w[i]!;
  }
  if(n>0){lo[0]=0;up[n-1]=0;solveTridiagonal(lo,di,up,rhs,sol);}
  const dwNew=new Float64Array(nz+1);
  for(let i=1;i<nz;i++)dwNew[i]=sol[i-1]!;

  const drNew=new Float64Array(nz),dxNew=new Float64Array(nz);
  for(let k=0;k<nz;k++){
    drNew[k]=drBase[k]!-dt*theta*divergence(ref.rhoInterface,dwNew,k);
    dxNew[k]=dxBase[k]!-dt*theta*divergence(ref.rhoThetaInterface,dwNew,k);
    acousticState.rho[k]=predictor.rho[k]!+drNew[k]!;
    acousticState.rhoTheta[k]=predictor.rhoTheta[k]!+dxNew[k]!;
  }
  acousticState.w[0]=0;acousticState.w[nz]=0;
  for(let i=1;i<nz;i++)acousticState.w[i]=predictor.w[i]!+dwNew[i]!;

  const referenceMassFlux=new Float64Array(nz+1),referenceRhoThetaFlux=new Float64Array(nz+1);
  for(let i=0;i<=nz;i++){
    const weighted=predictor.w[i]!+oldWeight*dwOld[i]!+theta*dwNew[i]!;
    referenceMassFlux[i]=ref.rhoInterface[i]!*weighted;
    referenceRhoThetaFlux[i]=ref.rhoThetaInterface[i]!*weighted;
  }
  return{referenceMassFlux,referenceRhoThetaFlux};
}
