import { EARTH } from '../core/constants.js';
import { angle3,cross3,dot3,normalize3,type Vec3 } from '../core/math.js';
import type { CubedSphereGrid,HorizontalEdge } from './cubedSphere.js';

export interface EdgeHodgeStencil {
  /** CSR row offsets, length edgeCount+1. */
  rowOffsets:Int32Array;
  /** Input edge index for every nonzero. */
  columns:Int32Array;
  /** Dimensionless H coefficient mapping dual circulation V to primal flux U. */
  weights:Float64Array;
}

function key(p:Vec3):string{const q=1e12;return `${Math.round(p[0]*q)},${Math.round(p[1]*q)},${Math.round(p[2]*q)}`;}
function center(h:CubedSphereGrid,c:number):Vec3{return[h.cellCenters[c*3]!,h.cellCenters[c*3+1]!,h.cellCenters[c*3+2]!];}
function tangentToward(a:Vec3,b:Vec3):Vec3{const ab=dot3(a,b),d:Vec3=[b[0]-ab*a[0],b[1]-ab*a[1],b[2]-ab*a[2]];return normalize3(d);}
function sharedVertex(a:HorizontalEdge,b:HorizontalEdge):Vec3{const a0=key(a.p0),a1=key(a.p1),b0=key(b.p0),b1=key(b.p1);if(a0===b0||a0===b1)return a.p0;if(a1===b0||a1===b1)return a.p1;throw new Error('adjacent cell edges do not share a primal vertex');}
function vertexValence(h:CubedSphereGrid):Map<string,number>{const cells=new Map<string,Set<number>>();for(let c=0;c<h.cellCount;c++)for(let s=0;s<4;s++){const e=h.edges[h.cellEdges[c*4+s]!]!;for(const p of[e.p0,e.p1]){const k=key(p),set=cells.get(k)??new Set<number>();set.add(c);cells.set(k,set);}}const out=new Map<string,number>();for(const [k,set] of cells)out.set(k,set.size);return out;}
function dualVectorAtCell(h:CubedSphereGrid,eid:number,c:number):Vec3{const e=h.edges[eid]!,cc=center(h,c),nb=e.leftCell===c?e.rightCell:e.leftCell,cn=center(h,nb),dir=tangentToward(cc,cn),len=angle3(cc,cn)*EARTH.radius,sign=e.leftCell===c?1:-1;return[sign*len*dir[0],sign*len*dir[1],sign*len*dir[2]];}
function add(row:Map<number,number>,col:number,w:number):void{row.set(col,(row.get(col)??0)+w);}

/**
 * Build the symmetric non-diagonal H of Thuburn, Cotter & Dubos (2014),
 * Appendix A, for a quadrilateral cubed-sphere primal grid whose eight cube
 * corners have triangular dual cells. The map is
 *
 *   U = H V,
 *
 * where V is velocity circulation integrated along dual edges and U is normal
 * velocity flux integrated across primal edges. H is purely geometrical and
 * dimensionless. The cubed-sphere consistency condition is that each primal
 * vertex be at the barycentre of its surrounding dual vertices; callers that
 * need a consistent H should therefore use buildModifiedCubedSphere().
 */
export function buildNonorthogonalHodgeStencil(h:CubedSphereGrid):EdgeHodgeStencil{
  const rows:Array<Map<number,number>>=Array.from({length:h.edgeCount},()=>new Map<number,number>()),valence=vertexValence(h);
  for(let c=0;c<h.cellCount;c++)for(let s=0;s<4;s++){
    const e0=h.cellEdges[c*4+s]!,e1=h.cellEdges[c*4+((s+1)%4)]!,g0=h.edges[e0]!,g1=h.edges[e1]!,v=sharedVertex(g0,g1),nv=valence.get(key(v))??0;
    if(nv!==3&&nv!==4)throw new Error(`unsupported dual-cell valence ${nv}`);
    const sc=nv===4?4:6,d0=dualVectorAtCell(h,e0,c),d1=dualVectorAtCell(h,e1,c),cross=cross3(d0,d1),area=Math.hypot(cross[0],cross[1],cross[2]);
    if(!(area>0)||!Number.isFinite(area))throw new Error(`degenerate Hodge corner at cell ${c}`);
    const inv=1/(sc*area),d00=dot3(d0,d0),d11=dot3(d1,d1),d01=dot3(d0,d1);
    add(rows[e0]!,e0,d11*inv);add(rows[e0]!,e1,-d01*inv);
    add(rows[e1]!,e1,d00*inv);add(rows[e1]!,e0,-d01*inv);
  }
  const rowOffsets=new Int32Array(h.edgeCount+1);let nnz=0;for(let e=0;e<h.edgeCount;e++){rowOffsets[e]=nnz;nnz+=rows[e]!.size;}rowOffsets[h.edgeCount]=nnz;
  const columns=new Int32Array(nnz),weights=new Float64Array(nnz);let q=0;
  for(let e=0;e<h.edgeCount;e++){const entries=[...rows[e]!.entries()].sort((a,b)=>a[0]-b[0]);for(const [col,w] of entries){columns[q]=col;weights[q]=w;q++;}}
  return{rowOffsets,columns,weights};
}

export function applyNonorthogonalHodge(stencil:EdgeHodgeStencil,V:ArrayLike<number>,out?:Float64Array):Float64Array{
  const edgeCount=stencil.rowOffsets.length-1;if(V.length!==edgeCount)throw new Error(`Hodge input length ${V.length} != edgeCount ${edgeCount}`);const U=out??new Float64Array(edgeCount);if(U.length!==edgeCount)throw new Error('Hodge output length mismatch');
  for(let e=0;e<edgeCount;e++){let sum=0;for(let q=stencil.rowOffsets[e]!;q<stencil.rowOffsets[e+1]!;q++)sum+=stencil.weights[q]!*V[stencil.columns[q]!]!;U[e]=sum;}return U;
}

export function hodgeSymmetryDefect(stencil:EdgeHodgeStencil):number{
  const rows:Array<Map<number,number>>=Array.from({length:stencil.rowOffsets.length-1},()=>new Map<number,number>());for(let e=0;e<rows.length;e++)for(let q=stencil.rowOffsets[e]!;q<stencil.rowOffsets[e+1]!;q++)rows[e]!.set(stencil.columns[q]!,stencil.weights[q]!);
  let max=0,scale=0;for(let e=0;e<rows.length;e++)for(const [j,w] of rows[e]!){const wt=rows[j]?.get(e)??0;max=Math.max(max,Math.abs(w-wt));scale=Math.max(scale,Math.abs(w),Math.abs(wt));}return max/Math.max(scale,1e-30);
}
