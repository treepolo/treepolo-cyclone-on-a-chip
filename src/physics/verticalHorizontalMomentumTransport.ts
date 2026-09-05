import { EARTH } from '../core/constants.js';
import type { CubedSphereGrid } from '../grid/cubedSphere.js';
import type { VerticalGrid } from '../grid/vertical.js';
import type { ReferenceAtmosphere } from './referenceAtmosphere.js';
import { reconstructCellHorizontalWind, type RotationGeometry } from './rotation.js';
import { cell3DIndex, w3DIndex, type DryState } from '../solver/state.js';

/**
 * Mass-flux-consistent vertical transport of horizontal velocity.
 *
 * Stage 4 splits vertical continuity into a reference acoustic carrier and a
 * slow perturbation carrier.  At an interior interface their sum is
 *
 *   M_i = [rho0_i + rho_up - rho0_up] w_i A,
 *
 * where rho0_i is the reference density at the interface and rho0_up is the
 * reference density at the upwind layer centre.  Horizontal momentum must use
 * that same total carrier.  For each fixed horizontal column we therefore form
 * the conservative donor momentum flux M_i u_up and convert it back to a
 * material velocity tendency with the exact discrete continuity identity
 *
 *   du/dt = [-div(M u) + u div(M)] / (rho V).
 *
 * This removes the previous mismatch between vertical density transport and
 * the velocity-only -w du/dz donor formula.  The operation contains no climate
 * target or angular-momentum correction; it is simply the conservative/product
 * rule written with the production continuity flux.
 */
export function computeVerticalHorizontalMomentumTendency(
  h:CubedSphereGrid,
  v:VerticalGrid,
  ref:ReferenceAtmosphere,
  s:DryState,
  g:RotationGeometry,
  windByK?:Float64Array[],
):Float64Array[]{
  const nz=v.nz,R=EARTH.radius,winds=windByK??Array.from({length:nz},(_,k)=>reconstructCellHorizontalWind(h,g,s,k)),out=Array.from({length:nz},()=>new Float64Array(h.cellCount*3));
  for(let c=0;c<h.cellCount;c++){
    const area=h.cellAreaUnit[c]!*R*R;
    for(let k=0;k<nz;k++){
      const q=cell3DIndex(c,k,nz),o=c*3,cur=winds[k]!,rho=Math.max(s.rhoD[q]!,1e-12),volume=area*v.dz[k]!;
      let sumM=0,sumX=0,sumY=0,sumZ=0;
      // Outward sign from this layer: lower face = -1, upper face = +1.
      for(const [i,sign] of [[k,-1],[k+1,1]] as const){
        if(i<=0||i>=nz)continue;
        const w=s.wInterface[w3DIndex(c,i,nz)]!,srcK=w>=0?i-1:i,src=cell3DIndex(c,srcK,nz);
        const M=(ref.rhoInterface[i]!+s.rhoD[src]!-ref.rhoCenter[srcK]!)*w*area;
        const outward=sign*M,donor=winds[srcK]!;
        sumM+=outward;
        sumX+=outward*donor[o]!;
        sumY+=outward*donor[o+1]!;
        sumZ+=outward*donor[o+2]!;
      }
      const a=out[k]!;
      a[o]=(-sumX+cur[o]!*sumM)/(rho*volume);
      a[o+1]=(-sumY+cur[o+1]!*sumM)/(rho*volume);
      a[o+2]=(-sumZ+cur[o+2]!*sumM)/(rho*volume);
    }
  }
  return out;
}
