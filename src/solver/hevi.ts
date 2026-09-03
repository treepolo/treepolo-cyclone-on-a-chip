import { DRY_AIR } from '../core/constants.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { pressureFromRhoTheta } from '../physics/thermodynamics.js';
import { solveTridiagonal } from './tridiagonal.js';

export interface ColumnState { rho:Float64Array; rhoTheta:Float64Array; w:Float64Array; }

/**
 * Linearized HEVI acoustic step around the hydrostatic reference state.
 *
 * offCentering uses theta = 0.5 * (1 + epsilon). epsilon=0 recovers the
 * centered Crank-Nicolson form used by the Stage 3 reference tests; epsilon>0
 * forward-centers the coupled vertical acoustic solve and selectively damps
 * the fast vertically propagating computational mode. Both vertical mass and
 * rho*theta flux updates use the same theta, so the column flux divergence
 * remains conservative.
 *
 * Optional rayleighRates are interface-centred inverse time scales. They are
 * applied after the tridiagonal vertical acoustic solve and BEFORE the new-time
 * mass/thermodynamic fluxes are formed:
 *
 *   w^(n+1) = w_tilde^(n+1) / (1 + tau(z) dt)
 *
 * Buoyancy remains a slow explicit vertical-momentum forcing outside this
 * acoustic column solve. A real-device Stage-4 ablation showed that inserting
 * the old-time density buoyancy directly into this linear acoustic RHS creates
 * a severe imbalance; the coupled formulation must be re-derived before it is
 * reconsidered.
 *
 * The returned array is the effective HEVI reference-state vertical mass flux
 * per unit horizontal area:
 *
 *   F_ref(i) = rho0(i) * [(1-theta) w_old(i) + theta w_new(i)]
 *
 * This is the exact flux used by the HEVI rho update. Stage 4 combines it with
 * the outer perturbation mass flux before transporting horizontal momentum, so
 * momentum and continuity use the same split mass carrier.
 */
export function heviColumnStep(
  v:VerticalGrid,
  ref:ReferenceAtmosphere,
  state:ColumnState,
  dt:number,
  offCentering=0,
  rayleighRates?:ArrayLike<number>,
):Float64Array {
  const nz=v.nz; if(state.rho.length!==nz||state.rhoTheta.length!==nz||state.w.length!==nz+1) throw new Error('column shape mismatch');
  if(!(offCentering>=0&&offCentering<1))throw new Error('HEVI offCentering must be in [0,1)');
  if(rayleighRates&&rayleighRates.length!==nz+1)throw new Error('HEVI Rayleigh profile shape mismatch');
  const referenceMassFlux=new Float64Array(nz+1);
  if(nz<2) return referenceMassFlux;
  const theta=0.5*(1+offCentering), oldWeight=1-theta;
  const n=nz-1, lo=new Float64Array(n), di=new Float64Array(n), up=new Float64Array(n), rhs=new Float64Array(n), sol=new Float64Array(n);
  const wOld=state.w.slice(); const pOld=new Float64Array(nz);
  for(let k=0;k<nz;k++) pOld[k]=pressureFromRhoTheta(state.rhoTheta[k]!)-ref.pCenter[k]!;
  const Lold=new Float64Array(nz);
  for(let k=0;k<nz;k++) Lold[k]=(ref.rhoThetaInterface[k+1]!*wOld[k+1]!-ref.rhoThetaInterface[k]!*wOld[k]!)/v.dz[k]!;
  for(let ii=0;ii<n;ii++){
    const i=ii+1,l=i-1,u=i, dzc=v.zCenter[u]!-v.zCenter[l]!;
    const rho0i=0.5*(ref.rhoCenter[l]!+ref.rhoCenter[u]!); const rhoi=0.5*(rho0i+0.5*(state.rho[l]!+state.rho[u]!));
    const Al=DRY_AIR.gamma*ref.pCenter[l]!/ref.rhoThetaCenter[l]!, Au=DRY_AIR.gamma*ref.pCenter[u]!/ref.rhoThetaCenter[u]!;
    const base=dt*dt/(rhoi*dzc), facNew=theta*theta*base, facOld=theta*oldWeight*base;
    const xim=ref.rhoThetaInterface[i-1]!, xi=ref.rhoThetaInterface[i]!, xip=ref.rhoThetaInterface[i+1]!;
    lo[ii]=-facNew*Al*xim/v.dz[l]!;
    di[ii]=1+facNew*(Au*xi/v.dz[u]!+Al*xi/v.dz[l]!);
    up[ii]=-facNew*Au*xip/v.dz[u]!;
    rhs[ii]=wOld[i]!-dt*(pOld[u]!-pOld[l]!)/(rhoi*dzc)+facOld*(Au*Lold[u]!-Al*Lold[l]!);
  }
  lo[0]=0; up[n-1]=0; solveTridiagonal(lo,di,up,rhs,sol);
  state.w[0]=0; state.w[nz]=0;
  for(let i=1;i<nz;i++){
    const rate=rayleighRates?Math.max(0,Number(rayleighRates[i]!)):0;
    state.w[i]=sol[i-1]!/(1+rate*dt);
  }
  for(let i=0;i<=nz;i++) referenceMassFlux[i]=ref.rhoInterface[i]!*(oldWeight*wOld[i]!+theta*state.w[i]!);
  for(let k=0;k<nz;k++){
    const Lnew=(ref.rhoThetaInterface[k+1]!*state.w[k+1]!-ref.rhoThetaInterface[k]!*state.w[k]!)/v.dz[k]!;
    state.rhoTheta[k] = state.rhoTheta[k]! - dt*(oldWeight*Lold[k]!+theta*Lnew);
    const Rold=(ref.rhoInterface[k+1]!*wOld[k+1]!-ref.rhoInterface[k]!*wOld[k]!)/v.dz[k]!;
    const Rnew=(ref.rhoInterface[k+1]!*state.w[k+1]!-ref.rhoInterface[k]!*state.w[k]!)/v.dz[k]!;
    state.rho[k] = state.rho[k]! - dt*(oldWeight*Rold+theta*Rnew);
  }
  return referenceMassFlux;
}
