import { EARTH } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from './referenceAtmosphere.js';
import { DryState, edge3DIndex } from '../solver/state.js';

/**
 * Dimensionless grid-scale horizontal acoustic-divergence filter coefficient.
 * 0.1 is the Stage 4 correctness value; it is fixed before the long-run rerun.
 */
export const ACOUSTIC_DIVERGENCE_DAMPING = 0.1;

/**
 * Computes horizontal velocity divergence on cell centres:
 *   div_h(u)
 * using exactly the canonical cubed-sphere shared edges used by the
 * finite-volume transport.
 *
 * Important: vertical divergence is intentionally excluded here.  HEVI owns
 * the vertically propagating acoustic mode.  Feeding d(rho0*w)/dz into a
 * horizontal Laplacian filter on a global grid multiplies tiny Float32 w
 * differences by the enormous horizontal/vertical aspect ratio and can turn
 * round-off into spurious horizontal wind.
 */
export function computeAcousticDivergence(
  h:CubedSphereGrid,
  v:VerticalGrid,
  _ref:ReferenceAtmosphere,
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
      out[c*v.nz+k]=horizontalFlux/area;
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
 *   u <- u + gamma * L^2 grad(div_h u)
 * represented on each canonical edge as
 *   du = gamma * d_edge * (D_R - D_L).
 *
 * A horizontal Fourier mode therefore receives Laplacian damping of its
 * divergent component.  No mass, thermodynamic state, vertical velocity, or
 * rotational component is directly clipped or Rayleigh-damped.
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
