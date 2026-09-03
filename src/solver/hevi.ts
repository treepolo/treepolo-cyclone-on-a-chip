import { DRY_AIR } from '../core/constants.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { pressureFromRhoTheta } from '../physics/thermodynamics.js';
import { solveTridiagonal } from './tridiagonal.js';

export interface ColumnState { rho:Float64Array; rhoTheta:Float64Array; w:Float64Array; }

/**
 * Linearized HEVI acoustic step around the hydrostatic reference state.
 * The implicit operator advances the base-state vertical mass and rho*theta fluxes;
 * nonlinear perturbation transport is handled by the outer finite-volume step.
 */
export function heviColumnStep(v:VerticalGrid,ref:ReferenceAtmosphere,state:ColumnState,dt:number):void {
  const nz=v.nz; if(state.rho.length!==nz||state.rhoTheta.length!==nz||state.w.length!==nz+1) throw new Error('column shape mismatch');
  if(nz<2) return;
  const n=nz-1, lo=new Float64Array(n), di=new Float64Array(n), up=new Float64Array(n), rhs=new Float64Array(n), sol=new Float64Array(n);
  const wOld=state.w.slice(); const pOld=new Float64Array(nz);
  for(let k=0;k<nz;k++) pOld[k]=pressureFromRhoTheta(state.rhoTheta[k]!)-ref.pCenter[k]!;
  const Lold=new Float64Array(nz);
  for(let k=0;k<nz;k++) Lold[k]=(ref.rhoThetaInterface[k+1]!*wOld[k+1]!-ref.rhoThetaInterface[k]!*wOld[k]!)/v.dz[k]!;
  for(let ii=0;ii<n;ii++){
    const i=ii+1,l=i-1,u=i, dzc=v.zCenter[u]!-v.zCenter[l]!;
    const rho0i=0.5*(ref.rhoCenter[l]!+ref.rhoCenter[u]!); const rhoi=0.5*(rho0i+0.5*(state.rho[l]!+state.rho[u]!));
    const Al=DRY_AIR.gamma*ref.pCenter[l]!/ref.rhoThetaCenter[l]!, Au=DRY_AIR.gamma*ref.pCenter[u]!/ref.rhoThetaCenter[u]!;
    const fac=0.25*dt*dt/(rhoi*dzc);
    const xim=ref.rhoThetaInterface[i-1]!, xi=ref.rhoThetaInterface[i]!, xip=ref.rhoThetaInterface[i+1]!;
    lo[ii]=-fac*Al*xim/v.dz[l]!;
    di[ii]=1+fac*(Au*xi/v.dz[u]!+Al*xi/v.dz[l]!);
    up[ii]=-fac*Au*xip/v.dz[u]!;
    rhs[ii]=wOld[i]!-dt*(pOld[u]!-pOld[l]!)/(rhoi*dzc)+fac*(Au*Lold[u]!-Al*Lold[l]!);
  }
  lo[0]=0; up[n-1]=0; solveTridiagonal(lo,di,up,rhs,sol);
  state.w[0]=0; state.w[nz]=0; for(let i=1;i<nz;i++) state.w[i]=sol[i-1]!;
  for(let k=0;k<nz;k++){
    const Lnew=(ref.rhoThetaInterface[k+1]!*state.w[k+1]!-ref.rhoThetaInterface[k]!*state.w[k]!)/v.dz[k]!;
    state.rhoTheta[k] = state.rhoTheta[k]! - 0.5*dt*(Lold[k]!+Lnew);
    const Rold=(ref.rhoInterface[k+1]!*wOld[k+1]!-ref.rhoInterface[k]!*wOld[k]!)/v.dz[k]!;
    const Rnew=(ref.rhoInterface[k+1]!*state.w[k+1]!-ref.rhoInterface[k]!*state.w[k]!)/v.dz[k]!;
    state.rho[k] = state.rho[k]! - 0.5*dt*(Rold+Rnew);
  }

}
