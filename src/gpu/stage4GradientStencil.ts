import { EARTH } from '../core/constants.js';
import { dot3, normalize3, scale3, sub3, Vec3 } from '../core/math.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { buildRotationGeometry } from '../physics/rotation.js';

export interface Stage4GradientStencilData {
  /** Four neighbor cell ids per cell, packed for WGSL vec4<u32>. */
  neighbors: Uint32Array;
  /** Two vec4<f32> per cell: east-gradient weights then north-gradient weights. */
  weights: Float32Array;
}

function vec(a:Float64Array,c:number):Vec3{return[a[c*3]!,a[c*3+1]!,a[c*3+2]!]}

/**
 * Precompute the geometry-only part of the Stage 4 non-orthogonal least-squares
 * tangent gradient. For scalar neighbor differences dv_s,
 *
 *   grad_east  = sum_s weightsEast_s  * dv_s
 *   grad_north = sum_s weightsNorth_s * dv_s
 *
 * This is algebraically the same 2x2 normal-equation solve used by
 * reconstructCellScalarGradient; only the state-independent geometry is moved
 * out of the repeated GPU pressure-gradient kernels.
 */
export function buildStage4GradientStencilData(h:CubedSphereGrid,radius=EARTH.radius):Stage4GradientStencilData{
  const g=buildRotationGeometry(h),neighbors=new Uint32Array(h.cellCount*4),weights=new Float32Array(h.cellCount*8);
  for(let c=0;c<h.cellCount;c++){
    const rc=vec(g.radial,c),ec=vec(g.east,c),nc=vec(g.north,c);
    const de=new Float64Array(4),dn=new Float64Array(4);
    let aa=0,ab=0,bb=0;
    for(let s=0;s<4;s++){
      const eid=h.cellEdges[c*4+s]!,edge=h.edges[eid]!,nb=edge.leftCell===c?edge.rightCell:edge.leftCell;
      neighbors[c*4+s]=nb;
      const rn=vec(g.radial,nb),mu=Math.max(-1,Math.min(1,dot3(rc,rn))),ang=Math.acos(mu),tangRaw=sub3(rn,scale3(rc,mu));
      const tmag=Math.hypot(tangRaw[0],tangRaw[1],tangRaw[2]);
      if(!(ang>0)||!(tmag>0))throw new Error(`invalid scalar-gradient neighbor geometry at cell ${c} slot ${s}`);
      const t=normalize3(tangRaw),dist=radius*ang;
      de[s]=dist*dot3(t,ec);dn[s]=dist*dot3(t,nc);
      aa+=de[s]!*de[s]!;ab+=de[s]!*dn[s]!;bb+=dn[s]!*dn[s]!;
    }
    const det=aa*bb-ab*ab;
    if(Math.abs(det)<1e-24)throw new Error(`singular scalar-gradient stencil at cell ${c}`);
    for(let s=0;s<4;s++){
      weights[c*8+s]=(bb*de[s]!-ab*dn[s]!)/det;
      weights[c*8+4+s]=(-ab*de[s]!+aa*dn[s]!)/det;
    }
  }
  return{neighbors,weights};
}
