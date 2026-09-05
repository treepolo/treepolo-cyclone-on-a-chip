import { EARTH } from '../core/constants.js';
import { Vec3, dot3, normalize3, scale3, sub3 } from '../core/math.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { RotationGeometry } from './rotation.js';

function radial(g:RotationGeometry,c:number):Vec3{return[g.radial[c*3]!,g.radial[c*3+1]!,g.radial[c*3+2]!]}
function east(g:RotationGeometry,c:number):Vec3{return[g.east[c*3]!,g.east[c*3+1]!,g.east[c*3+2]!]}
function north(g:RotationGeometry,c:number):Vec3{return[g.north[c*3]!,g.north[c*3+1]!,g.north[c*3+2]!]}
function neighbor(h:CubedSphereGrid,c:number,slot:number):number{
  const e=h.edges[h.cellEdges[c*4+slot]!]!;
  return e.leftCell===c?e.rightCell:e.leftCell;
}

/**
 * Least-squares tangent-plane gradient on the non-orthogonal cubed sphere.
 * Neighbor displacements are great-circle center-to-center vectors expressed
 * in the target cell's local east/north basis.  The returned vector is a
 * physical gradient in units value / metre and is tangent at the cell center.
 */
export function reconstructCellScalarGradient(
  h:CubedSphereGrid,
  g:RotationGeometry,
  c:number,
  value:(cell:number)=>number,
  radius=EARTH.radius,
):Vec3{
  const rc=radial(g,c),ec=east(g,c),nc=north(g,c),vc=value(c);
  let aa=0,ab=0,bb=0,ba=0,bbv=0;
  for(let slot=0;slot<4;slot++){
    const nb=neighbor(h,c,slot),rn=radial(g,nb),mu=Math.max(-1,Math.min(1,dot3(rc,rn))),ang=Math.acos(mu);
    const tangRaw=sub3(rn,scale3(rc,mu)),tmag=Math.hypot(tangRaw[0],tangRaw[1],tangRaw[2]);
    if(!(ang>0)||!(tmag>0))continue;
    const t=normalize3(tangRaw),dist=radius*ang,de=dist*dot3(t,ec),dn=dist*dot3(t,nc),dv=value(nb)-vc;
    aa+=de*de;ab+=de*dn;bb+=dn*dn;ba+=de*dv;bbv+=dn*dv;
  }
  const det=aa*bb-ab*ab;
  if(Math.abs(det)<1e-24)throw new Error(`singular scalar-gradient reconstruction at cell ${c}`);
  const ge=(bb*ba-ab*bbv)/det,gn=(-ab*ba+aa*bbv)/det;
  return[ge*ec[0]+gn*nc[0],ge*ec[1]+gn*nc[1],ge*ec[2]+gn*nc[2]];
}

/**
 * Face-normal scalar derivative.  Gradients reconstructed independently in the
 * two adjacent tangent planes are averaged in global Cartesian components and
 * projected onto the exact shared-face conormal.  This avoids the invalid
 * orthogonal-grid approximation (phi_R-phi_L)/centerDistance on a gnomonic
 * cubed sphere.
 */
export function reconstructEdgeNormalScalarGradient(
  h:CubedSphereGrid,
  g:RotationGeometry,
  edge:number,
  value:(cell:number)=>number,
  radius=EARTH.radius,
):number{
  const e=h.edges[edge]!,gl=reconstructCellScalarGradient(h,g,e.leftCell,value,radius),gr=reconstructCellScalarGradient(h,g,e.rightCell,value,radius);
  return .5*((gl[0]+gr[0])*e.normal[0]+(gl[1]+gr[1])*e.normal[1]+(gl[2]+gr[2])*e.normal[2]);
}
