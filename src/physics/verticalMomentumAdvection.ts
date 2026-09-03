import { EARTH } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { DryState, edge3DIndex, w3DIndex } from '../solver/state.js';

/**
 * First-order donor-cell material-advection tendency for prognostic w:
 *
 *   dw/dt = -u_h · grad_h(w) - w dw/dz.
 *
 * The returned array is a pure tendency evaluated from one immutable state.
 * Boundary interfaces are zero.  Keeping this operator in tendency form is
 * required by the Stage-4 RK3 split-explicit integrator, where the slow RHS is
 * evaluated once per RK stage and then frozen during acoustic substeps.
 */
export function computeVerticalVelocityAdvectionTendency(
  h:CubedSphereGrid,
  v:VerticalGrid,
  s:DryState,
):Float64Array {
  const nz=v.nz,R=EARTH.radius,old=s.wInterface,out=new Float64Array(old.length);
  for(let c=0;c<h.cellCount;c++){
    const area=h.cellAreaUnit[c]!*R*R;
    for(let i=1;i<nz;i++){
      const q=w3DIndex(c,i,nz),wi=old[q]!;
      let tendency=0;
      for(let slot=0;slot<4;slot++){
        const e=h.cellEdges[c*4+slot]!,sgn=h.cellEdgeSigns[c*4+slot]!,ge=h.edges[e]!;
        const ue=.5*(s.uEdge[edge3DIndex(e,i-1,nz)]!+s.uEdge[edge3DIndex(e,i,nz)]!);
        const outward=sgn*ue;
        if(outward<0){
          const n=ge.leftCell===c?ge.rightCell:ge.leftCell;
          const wn=old[w3DIndex(n,i,nz)]!;
          tendency-=outward*(ge.angularLength*R)*(wn-wi)/area;
        }
      }
      if(wi>0){
        tendency-=wi*(wi-old[w3DIndex(c,i-1,nz)]!)/v.dz[i-1]!;
      }else if(wi<0){
        tendency-=wi*(old[w3DIndex(c,i+1,nz)]!-wi)/v.dz[i]!;
      }
      out[q]=tendency;
    }
  }
  return out;
}

/**
 * Compatibility finite-step wrapper used by the existing Stage-4 production
 * path. It is deliberately implemented from the pure tendency above so the
 * old path and the future RK3 stage RHS cannot silently diverge.
 */
export function computeAdvectedVerticalVelocity(
  h:CubedSphereGrid,
  v:VerticalGrid,
  s:DryState,
  dt:number,
):Float64Array {
  if(!(dt>0))throw new Error('vertical-momentum advection dt must be positive');
  const tendency=computeVerticalVelocityAdvectionTendency(h,v,s),out=s.wInterface.slice(),nz=v.nz;
  for(let c=0;c<h.cellCount;c++){
    for(let i=1;i<nz;i++){
      const q=w3DIndex(c,i,nz);out[q]=out[q]!+dt*tendency[q]!;
    }
    out[w3DIndex(c,0,nz)]=0;
    out[w3DIndex(c,nz,nz)]=0;
  }
  return out;
}

export function advectVerticalVelocity(h:CubedSphereGrid,v:VerticalGrid,s:DryState,dt:number):void{
  s.wInterface.set(computeAdvectedVerticalVelocity(h,v,s,dt));
}
