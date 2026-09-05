import { angle3,cross3,dot3,normalize3,scale3,sphericalTriangleAreaUnit,type Vec3 } from '../core/math.js';
import { buildCubedSphere,type CubedSphereGrid,type HorizontalEdge } from './cubedSphere.js';

function pointKey(p:Vec3):string{const q=1e12;return `${Math.round(p[0]*q)},${Math.round(p[1]*q)},${Math.round(p[2]*q)}`;}
function averageUnit(points:readonly Vec3[]):Vec3{let x=0,y=0,z=0;for(const p of points){x+=p[0];y+=p[1];z+=p[2];}return normalize3([x,y,z]);}
function cellCenter(arr:Float64Array,c:number):Vec3{return[arr[c*3]!,arr[c*3+1]!,arr[c*3+2]!];}
function sharedEndpointKey(a:HorizontalEdge,b:HorizontalEdge):string{const a0=pointKey(a.p0),a1=pointKey(a.p1),b0=pointKey(b.p0),b1=pointKey(b.p1);if(a0===b0||a0===b1)return a0;if(a1===b0||a1===b1)return a1;throw new Error('adjacent cell edges do not share a vertex');}

/**
 * Build the one-pass barycentrically modified equiangular cubed sphere used by
 * Thuburn, Cotter & Dubos (2014) for a consistent nonorthogonal H operator:
 *
 *  1. start from equiangular cubed-sphere primal vertices;
 *  2. place dual vertices (cell centres) at barycentres of surrounding primal vertices;
 *  3. relocate primal vertices to barycentres of surrounding dual vertices;
 *  4. do not iterate steps 2-3.
 *
 * All barycentres are formed in Cartesian embedding space and radially
 * projected back to the unit sphere. Topology and panel indexing are inherited
 * from buildCubedSphere; all geometry affected by the relocation is recomputed.
 */
export function buildModifiedCubedSphere(n:number):CubedSphereGrid{
  const raw=buildCubedSphere(n);
  const rawVertexPosition=new Map<string,Vec3>();
  const rawVertexCells=new Map<string,Set<number>>();
  for(let c=0;c<raw.cellCount;c++)for(let s=0;s<4;s++){
    const e=raw.edges[raw.cellEdges[c*4+s]!]!;
    for(const p of[e.p0,e.p1]){
      const k=pointKey(p);rawVertexPosition.set(k,p);
      const cells=rawVertexCells.get(k)??new Set<number>();cells.add(c);rawVertexCells.set(k,cells);
    }
  }

  // cellEdges are stored in cyclic perimeter order by buildCubedSphere.  The
  // shared vertex of consecutive edges therefore gives a cyclic corner list.
  const cellCornerKeys:string[][]=Array.from({length:raw.cellCount},()=>[]);
  for(let c=0;c<raw.cellCount;c++)for(let s=0;s<4;s++){
    const e0=raw.edges[raw.cellEdges[c*4+s]!]!,e1=raw.edges[raw.cellEdges[c*4+((s+1)%4)]!]!;
    cellCornerKeys[c]!.push(sharedEndpointKey(e0,e1));
  }

  const centers=new Float64Array(raw.cellCount*3);
  for(let c=0;c<raw.cellCount;c++){
    const corners=cellCornerKeys[c]!.map(k=>{const p=rawVertexPosition.get(k);if(!p)throw new Error('missing raw corner');return p;});
    centers.set(averageUnit(corners),c*3);
  }

  const movedVertex=new Map<string,Vec3>();
  for(const [k,cells] of rawVertexCells){
    const surrounding:Vec3[]=[];for(const c of cells)surrounding.push(cellCenter(centers,c));
    movedVertex.set(k,averageUnit(surrounding));
  }

  const edges:HorizontalEdge[]=raw.edges.map(e=>{
    const p0=movedVertex.get(pointKey(e.p0)),p1=movedVertex.get(pointKey(e.p1));if(!p0||!p1)throw new Error('missing relocated endpoint');
    const cl=cellCenter(centers,e.leftCell),cr=cellCenter(centers,e.rightCell),delta:Vec3=[cr[0]-cl[0],cr[1]-cl[1],cr[2]-cl[2]];
    let normal=normalize3(cross3(p0,p1));if(dot3(normal,delta)<0)normal=scale3(normal,-1);
    return{...e,p0,p1,midpoint:averageUnit([p0,p1]),normal,angularLength:angle3(p0,p1),centerDistanceAngle:angle3(cl,cr)};
  });

  const areas=new Float64Array(raw.cellCount);let totalAreaUnit=0;
  for(let c=0;c<raw.cellCount;c++){
    const q=cellCornerKeys[c]!.map(k=>{const p=movedVertex.get(k);if(!p)throw new Error('missing relocated cell corner');return p;});
    const area=sphericalTriangleAreaUnit(q[0]!,q[1]!,q[2]!)+sphericalTriangleAreaUnit(q[0]!,q[2]!,q[3]!);
    if(!(area>0)&&Number.isFinite(area))throw new Error(`invalid modified cubed-sphere cell area at ${c}: ${area}`);
    areas[c]=area;totalAreaUnit+=area;
  }

  return{...raw,cellCenters:centers,cellAreaUnit:areas,edges,totalAreaUnit};
}
