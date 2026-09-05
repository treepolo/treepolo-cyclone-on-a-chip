import { EARTH } from '../core/constants.js';
import type { CubedSphereGrid } from '../grid/cubedSphere.js';
import type { VerticalGrid } from '../grid/vertical.js';
import { reconstructCellScalarGradient } from './horizontalGradient.js';
import { reconstructCellHorizontalWind, type RotationGeometry } from './rotation.js';
import { cell3DIndex, edge3DIndex, type DryState } from '../solver/state.js';

export type HorizontalMomentumScheme='mass-donor'|'muscl-bj';

function clampDot(x:number):number{return Math.max(-1,Math.min(1,x));}

/** Great-circle tangent displacement from a cell center to a shared face midpoint, in metres. */
function displacementToFace(h:CubedSphereGrid,g:RotationGeometry,c:number,eid:number):readonly[number,number,number]{
  const o=c*3,edge=h.edges[eid]!,m=edge.midpoint,rx=g.radial[o]!,ry=g.radial[o+1]!,rz=g.radial[o+2]!,mu=clampDot(rx*m[0]+ry*m[1]+rz*m[2]),ang=Math.acos(mu),tx=m[0]-mu*rx,ty=m[1]-mu*ry,tz=m[2]-mu*rz,tm=Math.hypot(tx,ty,tz);
  if(!(ang>0)||!(tm>0))return[0,0,0];
  const s=EARTH.radius*ang/tm;return[tx*s,ty*s,tz*s];
}

/**
 * A single Barth-Jespersen-style limiter shared by the three fixed Cartesian
 * wind components.  Sharing one factor avoids rotating the vector-gradient
 * direction while still preventing any component reconstructed to one of the
 * four face midpoints from leaving the local face-neighbor envelope.
 */
function buildVectorLimiter(
  h:CubedSphereGrid,g:RotationGeometry,wind:Float64Array,
  gx:Float64Array,gy:Float64Array,gz:Float64Array,
):Float64Array{
  const phi=new Float64Array(h.cellCount);phi.fill(1);
  const grads=[gx,gy,gz];
  for(let c=0;c<h.cellCount;c++){
    const o=c*3,base=[wind[o]!,wind[o+1]!,wind[o+2]!],lo=[...base],hi=[...base];
    for(let slot=0;slot<4;slot++){
      const eid=h.cellEdges[c*4+slot]!,edge=h.edges[eid]!,nb=edge.leftCell===c?edge.rightCell:edge.leftCell,no=nb*3;
      for(let d=0;d<3;d++){const q=wind[no+d]!;lo[d]=Math.min(lo[d]!,q);hi[d]=Math.max(hi[d]!,q);}
    }
    let p=1;
    for(let slot=0;slot<4;slot++){
      const d=displacementToFace(h,g,c,h.cellEdges[c*4+slot]!);
      for(let comp=0;comp<3;comp++){
        const gg=grads[comp]!,delta=gg[o]!*d[0]+gg[o+1]!*d[1]+gg[o+2]!*d[2];
        if(delta>1e-15)p=Math.min(p,(hi[comp]!-base[comp]!)/delta);
        else if(delta<-1e-15)p=Math.min(p,(lo[comp]!-base[comp]!)/delta);
      }
    }
    phi[c]=Math.max(0,Math.min(1,p));
  }
  return phi;
}

/**
 * Compute only the horizontal material acceleration of the horizontal wind.
 * This function is intentionally separate from Stage4SlowTendencies while the
 * higher-order transport is being validated.
 *
 * Crucially, both schemes use exactly the same donor total mass flux as the
 * discrete continuity equation:
 *
 *   M_f = rho_upwind * u_n * A_f .
 *
 * The cell velocity tendency is the discrete conservative-to-material identity
 *
 *   du/dt = [-div(M u_face) + u_cell div(M)] / rho_cell,
 *
 * so density transport and momentum transport cannot silently use different
 * face carriers.  `mass-donor` uses the upwind cell wind at the face.
 * `muscl-bj` reconstructs the upwind wind to the real great-circle face
 * midpoint with least-squares tangent gradients and a local monotonic limiter.
 * Neither scheme contains climate-target information or a preferred wind sign.
 */
export function computeHorizontalMaterialMomentumTendency(
  h:CubedSphereGrid,
  v:VerticalGrid,
  s:DryState,
  g:RotationGeometry,
  scheme:HorizontalMomentumScheme,
  windByK?:Float64Array[],
):Float64Array[]{
  const nz=v.nz,R=EARTH.radius,winds=windByK??Array.from({length:nz},(_,k)=>reconstructCellHorizontalWind(h,g,s,k)),out=Array.from({length:nz},()=>new Float64Array(h.cellCount*3));
  for(let k=0;k<nz;k++){
    const wind=winds[k]!,a=out[k]!;
    let gx:Float64Array|undefined,gy:Float64Array|undefined,gz:Float64Array|undefined,phi:Float64Array|undefined;
    if(scheme==='muscl-bj'){
      gx=new Float64Array(h.cellCount*3);gy=new Float64Array(h.cellCount*3);gz=new Float64Array(h.cellCount*3);
      for(let c=0;c<h.cellCount;c++){
        gx.set(reconstructCellScalarGradient(h,g,c,cc=>wind[cc*3]!),c*3);
        gy.set(reconstructCellScalarGradient(h,g,c,cc=>wind[cc*3+1]!),c*3);
        gz.set(reconstructCellScalarGradient(h,g,c,cc=>wind[cc*3+2]!),c*3);
      }
      phi=buildVectorLimiter(h,g,wind,gx,gy,gz);
    }
    for(let c=0;c<h.cellCount;c++){
      const o=c*3,curx=wind[o]!,cury=wind[o+1]!,curz=wind[o+2]!,rho=Math.max(s.rhoD[cell3DIndex(c,k,nz)]!,1e-12),volume=h.cellAreaUnit[c]!*R*R*v.dz[k]!;
      let sumM=0,sumX=0,sumY=0,sumZ=0;
      for(let slot=0;slot<4;slot++){
        const eid=h.cellEdges[c*4+slot]!,edge=h.edges[eid]!,sign=h.cellEdgeSigns[c*4+slot]!,ue=s.uEdge[edge3DIndex(eid,k,nz)]!,up=ue>=0?edge.leftCell:edge.rightCell,upq=cell3DIndex(up,k,nz),M=sign*s.rhoD[upq]!*ue*(edge.angularLength*R)*v.dz[k]!;
        const uo=up*3;let qx=wind[uo]!,qy=wind[uo+1]!,qz=wind[uo+2]!;
        if(scheme==='muscl-bj'){
          const d=displacementToFace(h,g,up,eid),fac=phi![up]!;
          qx+=fac*(gx![uo]!*d[0]+gx![uo+1]!*d[1]+gx![uo+2]!*d[2]);
          qy+=fac*(gy![uo]!*d[0]+gy![uo+1]!*d[1]+gy![uo+2]!*d[2]);
          qz+=fac*(gz![uo]!*d[0]+gz![uo+1]!*d[1]+gz![uo+2]!*d[2]);
          // A transported horizontal vector belongs to the face tangent plane.
          const m=edge.midpoint,rad=qx*m[0]+qy*m[1]+qz*m[2];qx-=rad*m[0];qy-=rad*m[1];qz-=rad*m[2];
        }
        sumM+=M;sumX+=M*qx;sumY+=M*qy;sumZ+=M*qz;
      }
      let ax=(-sumX+curx*sumM)/(rho*volume),ay=(-sumY+cury*sumM)/(rho*volume),az=(-sumZ+curz*sumM)/(rho*volume);
      const rx=g.radial[o]!,ry=g.radial[o+1]!,rz=g.radial[o+2]!,rad=ax*rx+ay*ry+az*rz;ax-=rad*rx;ay-=rad*ry;az-=rad*rz;
      a[o]=ax;a[o+1]=ay;a[o+2]=az;
    }
  }
  return out;
}
