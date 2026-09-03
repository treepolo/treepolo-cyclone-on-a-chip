import { EARTH } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from './referenceAtmosphere.js';
import { DryState, edge3DIndex } from '../solver/state.js';

/**
 * Historical Stage 4 reference: coefficient 0.1 applied once per 100 s.
 * The old implementation treated this as a fixed per-application filter, so
 * applying it every 10 s accidentally made the damping about ten times stronger
 * per unit physical time and distorted the Hadley circulation.
 */
export const ACOUSTIC_DIVERGENCE_REFERENCE_COEFFICIENT = 0.1;
export const ACOUSTIC_DIVERGENCE_REFERENCE_INTERVAL_SECONDS = 100;

/** Continuous-time relaxation timescale equivalent to 10% damping per 100 s. */
export const ACOUSTIC_DIVERGENCE_DAMPING_TIMESCALE_SECONDS =
  -ACOUSTIC_DIVERGENCE_REFERENCE_INTERVAL_SECONDS / Math.log(1 - ACOUSTIC_DIVERGENCE_REFERENCE_COEFFICIENT);

/**
 * Convert a physical timestep into the dimensionless explicit filter strength.
 * By construction:
 *   dt = 100 s -> coefficient = 0.1
 *   dt = 10 s  -> coefficient ~= 0.01048
 * so changing timestep/cadence no longer silently changes damping per unit time.
 */
export function acousticDivergenceCoefficientForDt(dt:number):number {
  if(!(dt>0) || !Number.isFinite(dt))throw new Error('acoustic divergence dt must be positive and finite');
  const coefficient=1-Math.exp(-dt/ACOUSTIC_DIVERGENCE_DAMPING_TIMESCALE_SECONDS);
  if(coefficient>0.25)throw new Error(`acoustic divergence timestep too large for explicit filter: coefficient=${coefficient}`);
  return coefficient;
}

/**
 * Computes horizontal velocity divergence on cell centres:
 *   div_h(u)
 * using exactly the canonical cubed-sphere shared edges used by the
 * finite-volume transport.
 *
 * Vertical divergence is intentionally excluded. HEVI owns vertically
 * propagating acoustic modes; mixing vertical divergence into this horizontal
 * operator amplifies tiny w differences by the global grid aspect ratio.
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
 * Horizontal divergent-mode filter:
 *   u <- u + gamma * L^2 grad(div_h u)
 * represented on each canonical edge as
 *   du = gamma * d_edge * (D_R - D_L).
 *
 * `coefficient` is a per-application dimensionless strength. Production Stage 4
 * callers must obtain it from acousticDivergenceCoefficientForDt(dt), so the
 * physical damping rate is independent of integration cadence.
 */
export function applyAcousticDivergenceDamping(
  h:CubedSphereGrid,
  v:VerticalGrid,
  ref:ReferenceAtmosphere,
  s:DryState,
  coefficient:number,
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
