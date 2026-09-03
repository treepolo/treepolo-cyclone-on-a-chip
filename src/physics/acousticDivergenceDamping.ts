import { EARTH } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from './referenceAtmosphere.js';
import { DryState, edge3DIndex, w3DIndex } from '../solver/state.js';

/**
 * Dimensionless grid-scale acoustic-divergence filter coefficient.
 * 0.1 is the Stage 4 correctness value; it is fixed before the long-run rerun.
 */
export const ACOUSTIC_DIVERGENCE_DAMPING = 0.1;

/**
 * Computes the base-state-mass-weighted 3-D velocity divergence
 *   div_h(u) + (1/rho0) d(rho0 w)/dz
 * on cell centres.  The horizontal part uses exactly the canonical
 * cubed-sphere shared edges used by the finite-volume transport.
 */
export function computeAcousticDivergence(
  h:CubedSphereGrid,
  v:VerticalGrid,
  ref:ReferenceAtmosphere,
  s:DryState,
  out:Float64Array=new Float64Array(h.cellCount*v.nz),
):Float64Array {
  if(out.length!==h.cellCount*v.nz)throw new Error('acoustic divergence output shape mismatch');
  const R=EARTH.radius;
  for(let c=0;c<h.cellCount;c++){
    const area=h.cellAreaUnit[c]!*R*R;
    for(let k=0;k<v.nz;k++){
      let horizontalFlux=0;
      for(let slot=0;slot<4;slot++){
        const e=h.cellEdges[c*4+slot]!,sign=h.cellEdgeSigns[c*4+slot]!,ge=h.edges[e]!;
        horizontalFlux+=sign*s.uEdge[edge3DIndex(e,k,v.nz)]!*ge.angularLength*R;
      }
      const wb=s.wInterface[w3DIndex(c,k,v.nz)]!,wt=s.wInterface[w3DIndex(c,k+1,v.nz)]!;
      const rho0=Math.max(ref.rhoCenter[k]!,1e-12);
      const vertical=(ref.rhoInterface[k+1]!*wt-ref.rhoInterface[k]!*wb)/(rho0*v.dz[k]!);
      out[c*v.nz+k]=horizontalFlux/area+vertical;
    }
  }
  return out;
}

export function acousticDivergenceRms(h:CubedSphereGrid,v:VerticalGrid,ref:ReferenceAtmosphere,s:DryState):number{
  const d=computeAcousticDivergence(h,v,ref,s);let sum=0,weight=0;
  for(let c=0;c<h.cellCount;c++)for(let k=0;k<v.nz;k++){
    const w=h.cellAreaUnit[c]!*v.dz[k]!;sum+=w*d[c*v.nz+k]!*d[c*v.nz+k]!;weight+=w;
  }
  return Math.sqrt(sum/Math.max(weight,Number.MIN_VALUE));
}

/**
 * Horizontal acoustic filter:
 *   u <- u + gamma * L^2 grad(div)
 * represented on each canonical edge as
 *   du = gamma * d_edge * (D_R - D_L).
 *
 * A Fourier mode therefore receives Laplacian damping of its divergent
 * component.  No mass or thermodynamic state is altered and no velocity
 * clipping is used.
 */
export function applyAcousticDivergenceDamping(
  h:CubedSphereGrid,
  v:VerticalGrid,
  ref:ReferenceAtmosphere,
  s:DryState,
  coefficient=ACOUSTIC_DIVERGENCE_DAMPING,
):void {
  if(!(coefficient>=0&&coefficient<=0.25))throw new Error('acoustic divergence coefficient must be in [0,0.25]');
  if(coefficient===0)return;
  const div=computeAcousticDivergence(h,v,ref,s),R=EARTH.radius;
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,dist=ge.centerDistanceAngle*R;
    for(let k=0;k<v.nz;k++){
      const l=ge.leftCell*v.nz+k,r=ge.rightCell*v.nz+k,q=edge3DIndex(e,k,v.nz);
      s.uEdge[q]=s.uEdge[q]!+coefficient*dist*(div[r]!-div[l]!);
    }
  }
}
