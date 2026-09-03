import { EARTH } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { DryState, edge3DIndex, w3DIndex } from '../solver/state.js';

/**
 * First-order donor-cell advection for the prognostic vertical velocity.
 *
 * The dry core already advances pressure/acoustic and buoyancy tendencies on
 * the staggered w interfaces. This operator supplies the previously missing
 * material-advection terms
 *
 *   - u_h · grad_h(w) - w d(w)/dz
 *
 * without touching mass or thermodynamics. Horizontal advection uses the
 * cubed-sphere edge-normal wind averaged to the vertical interface. Vertical
 * advection uses the local interface w as the advecting velocity. The update is
 * out-of-place so every tendency is formed from the same old w field.
 */
export function advectVerticalVelocity(
  h:CubedSphereGrid,
  v:VerticalGrid,
  s:DryState,
  dt:number,
):void {
  if(!(dt>0))throw new Error('vertical-momentum advection dt must be positive');
  const nz=v.nz,R=EARTH.radius,old=s.wInterface,out=old.slice();
  for(let c=0;c<h.cellCount;c++){
    const area=h.cellAreaUnit[c]!*R*R;
    for(let i=1;i<nz;i++){
      const q=w3DIndex(c,i,nz),wi=old[q]!;
      let tendency=0;
      // Advective-form horizontal donor-cell term. Outflow faces use the local
      // value and therefore contribute zero to (w_face-w_cell); only inflow
      // carries a neighbour difference into the cell.
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
      // Vertical donor-cell derivative on the interface grid.
      if(wi>0){
        tendency-=wi*(wi-old[w3DIndex(c,i-1,nz)]!)/v.dz[i-1]!;
      }else if(wi<0){
        tendency-=wi*(old[w3DIndex(c,i+1,nz)]!-wi)/v.dz[i]!;
      }
      out[q]=wi+dt*tendency;
    }
    out[w3DIndex(c,0,nz)]=0;
    out[w3DIndex(c,nz,nz)]=0;
  }
  s.wInterface.set(out);
}
