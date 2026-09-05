import { Vec3, add3, angle3, cross3, normalize3, scale3, sphericalTriangleAreaUnit, sub3, dot3 } from '../core/math.js';

interface FaceBasis { name:string; c:Vec3; a:Vec3; b:Vec3; }
const FACES:FaceBasis[] = [
  {name:'+X',c:[1,0,0],a:[0,1,0],b:[0,0,1]},
  {name:'-X',c:[-1,0,0],a:[0,-1,0],b:[0,0,1]},
  {name:'+Y',c:[0,1,0],a:[-1,0,0],b:[0,0,1]},
  {name:'-Y',c:[0,-1,0],a:[1,0,0],b:[0,0,1]},
  {name:'+Z',c:[0,0,1],a:[0,1,0],b:[-1,0,0]},
  {name:'-Z',c:[0,0,-1],a:[0,1,0],b:[1,0,0]},
];

export interface HorizontalEdge {
  leftCell:number;
  rightCell:number;
  p0:Vec3;
  p1:Vec3;
  midpoint:Vec3;
  /** Unit spherical conormal to the shared great-circle face, oriented left -> right. */
  normal:Vec3;
  angularLength:number;
  centerDistanceAngle:number;
}

export interface CubedSphereGrid {
  n:number;
  cellCount:number;
  edgeCount:number;
  cellCenters:Float64Array;
  cellAreaUnit:Float64Array;
  cellPanel:Uint8Array;
  cellIJ:Int32Array;
  edges:HorizontalEdge[];
  cellEdges:Int32Array;
  cellEdgeSigns:Int8Array;
  totalAreaUnit:number;
}

function point(face:FaceBasis, alpha:number,beta:number):Vec3 {
  const x=Math.tan(alpha), y=Math.tan(beta);
  return normalize3(add3(face.c, add3(scale3(face.a,x),scale3(face.b,y))));
}
function keyPoint(p:Vec3):string { const q=1e12; return `${Math.round(p[0]*q)},${Math.round(p[1]*q)},${Math.round(p[2]*q)}`; }
function keyEdge(a:Vec3,b:Vec3):string { const ka=keyPoint(a), kb=keyPoint(b); return ka<kb?`${ka}|${kb}`:`${kb}|${ka}`; }
function getCenter(arr:Float64Array, id:number):Vec3 { return [arr[id*3]!,arr[id*3+1]!,arr[id*3+2]!]; }

export function buildCubedSphere(n:number):CubedSphereGrid {
  if (!Number.isInteger(n) || n<2) throw new Error('cubed-sphere N must be integer >=2');
  const cellCount=6*n*n;
  const centers=new Float64Array(cellCount*3), areas=new Float64Array(cellCount), panels=new Uint8Array(cellCount), ij=new Int32Array(cellCount*2);
  const edgeTemp=new Map<string,{p0:Vec3;p1:Vec3;cells:number[]}>();
  const cellEdgeKeys:string[][]=Array.from({length:cellCount},()=>[]);
  const da=(Math.PI/2)/n;
  let totalArea=0;
  for(let f=0;f<6;f++) {
    const face=FACES[f]!;
    for(let j=0;j<n;j++) for(let i=0;i<n;i++) {
      const id=f*n*n+j*n+i;
      const a0=-Math.PI/4+i*da, a1=a0+da, b0=-Math.PI/4+j*da, b1=b0+da;
      const p00=point(face,a0,b0), p10=point(face,a1,b0), p11=point(face,a1,b1), p01=point(face,a0,b1);
      const c=point(face,(a0+a1)/2,(b0+b1)/2);
      centers.set(c,id*3); panels[id]=f; ij[id*2]=i; ij[id*2+1]=j;
      const area=sphericalTriangleAreaUnit(p00,p10,p11)+sphericalTriangleAreaUnit(p00,p11,p01); areas[id]=area; totalArea+=area;
      const segs:[[Vec3,Vec3],[Vec3,Vec3],[Vec3,Vec3],[Vec3,Vec3]]=[[p00,p10],[p10,p11],[p11,p01],[p01,p00]];
      for(const [p0,p1] of segs) {
        const k=keyEdge(p0,p1); cellEdgeKeys[id]!.push(k);
        const e=edgeTemp.get(k); if(e) e.cells.push(id); else edgeTemp.set(k,{p0,p1,cells:[id]});
      }
    }
  }
  const edges:HorizontalEdge[]=[]; const edgeIdByKey=new Map<string,number>();
  for(const [key,e] of edgeTemp) {
    if(e.cells.length!==2) throw new Error(`edge ${key} has ${e.cells.length} neighbors`);
    const left=e.cells[0]!, right=e.cells[1]!; const cl=getCenter(centers,left), cr=getCenter(centers,right);
    const mid=normalize3(add3(e.p0,e.p1)),delta=sub3(cr,cl);
    // A C-grid face-normal velocity must be normal to the actual shared face,
    // not to the line joining the two cell centers. Equal-angle gnomonic
    // cubed-sphere cells are non-orthogonal in panel interiors, so the two
    // directions can differ substantially. For a great-circle edge, p0 x p1
    // is the exact constant spherical conormal along that edge. Orient it from
    // left cell to right cell to preserve the existing canonical flux sign.
    let normal=normalize3(cross3(e.p0,e.p1));
    if(dot3(normal,delta)<0)normal=scale3(normal,-1);
    const edge:HorizontalEdge={leftCell:left,rightCell:right,p0:e.p0,p1:e.p1,midpoint:mid,normal,angularLength:angle3(e.p0,e.p1),centerDistanceAngle:angle3(cl,cr)};
    edgeIdByKey.set(key,edges.length); edges.push(edge);
  }
  const cellEdges=new Int32Array(cellCount*4), signs=new Int8Array(cellCount*4);
  for(let c=0;c<cellCount;c++) {
    const keys=cellEdgeKeys[c]!; if(keys.length!==4) throw new Error('cell edge count !=4');
    for(let s=0;s<4;s++) { const eid=edgeIdByKey.get(keys[s]!); if(eid===undefined) throw new Error('missing edge'); cellEdges[c*4+s]=eid; const e=edges[eid]!; signs[c*4+s]=e.leftCell===c?1:-1; }
  }
  return {n,cellCount,edgeCount:edges.length,cellCenters:centers,cellAreaUnit:areas,cellPanel:panels,cellIJ:ij,edges,cellEdges,cellEdgeSigns:signs,totalAreaUnit:totalArea};
}
