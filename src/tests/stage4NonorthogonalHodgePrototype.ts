declare const process:{exitCode?:number};
import { EARTH } from '../core/constants.js';
import { angle3,cross3,dot3,normalize3,scale3,sub3,type Vec3 } from '../core/math.js';
import { buildCubedSphere,type CubedSphereGrid,type HorizontalEdge } from '../grid/cubedSphere.js';

function pkey(p:Vec3):string{const q=1e11;return `${Math.round(p[0]*q)},${Math.round(p[1]*q)},${Math.round(p[2]*q)}`;}
function center(h:CubedSphereGrid,c:number):Vec3{return[h.cellCenters[c*3]!,h.cellCenters[c*3+1]!,h.cellCenters[c*3+2]!];}
function tangentToward(a:Vec3,b:Vec3):Vec3{const d:Vec3=[b[0]-dot3(a,b)*a[0],b[1]-dot3(a,b)*a[1],b[2]-dot3(a,b)*a[2]];return normalize3(d);}
function solidWind(r:Vec3):Vec3{
  // Smooth tilted solid-body rotation; arbitrary amplitude, no climate target.
  const axis=normalize3([.37,-.21,.905] as Vec3),amp=27;
  return scale3(cross3(axis,r),amp);
}
function sharedVertex(a:HorizontalEdge,b:HorizontalEdge):Vec3{
  const ka0=pkey(a.p0),ka1=pkey(a.p1),kb0=pkey(b.p0),kb1=pkey(b.p1);
  if(ka0===kb0||ka0===kb1)return a.p0;
  if(ka1===kb0||ka1===kb1)return a.p1;
  throw new Error('adjacent cell edges do not share a primal vertex');
}
function dualVectorAtCell(h:CubedSphereGrid,eid:number,c:number):Vec3{
  const e=h.edges[eid]!,cc=center(h,c),nb=e.leftCell===c?e.rightCell:e.leftCell,cn=center(h,nb),out=tangentToward(cc,cn),len=angle3(cc,cn)*EARTH.radius,sgn=e.leftCell===c?1:-1;
  return scale3(out,sgn*len);
}
function dualCirculation(h:CubedSphereGrid,eid:number):number{
  const e=h.edges[eid]!,l=center(h,e.leftCell),r=center(h,e.rightCell),m=normalize3([l[0]+r[0],l[1]+r[1],l[2]+r[2]]),t=tangentToward(m,r),len=angle3(l,r)*EARTH.radius;
  return dot3(solidWind(m),t)*len;
}
function exactFaceFlux(h:CubedSphereGrid,eid:number):number{
  const e=h.edges[eid]!;return dot3(solidWind(e.midpoint),e.normal)*e.angularLength*EARTH.radius;
}

/**
 * Thuburn-Cotter-Dubos (2014) Appendix-A symmetric nonorthogonal H map,
 * specialized to this spherical cubed-sphere topology.  Each corner of a
 * dual cell is represented by a pair of adjacent primal-cell edges.  The
 * spherical use of tangent dual-edge vectors is expected to introduce only
 * the documented O(dx^2) geometric error. Diagnostic only.
 */
function applySymmetricH(h:CubedSphereGrid,V:Float64Array):Float64Array{
  const U=new Float64Array(h.edgeCount);
  const vertexCells=new Map<string,Set<number>>();
  for(let c=0;c<h.cellCount;c++)for(let s=0;s<4;s++){
    const e=h.edges[h.cellEdges[c*4+s]!]!;
    for(const p of[e.p0,e.p1]){const k=pkey(p),set=vertexCells.get(k)??new Set<number>();set.add(c);vertexCells.set(k,set);}
  }
  // For each dual-cell corner (a primal cell centre + one primal vertex), the
  // two boundary dual edges are the two cyclically adjacent primal edges.
  for(let c=0;c<h.cellCount;c++)for(let s=0;s<4;s++){
    const e0=h.cellEdges[c*4+s]!,e1=h.cellEdges[c*4+((s+1)%4)]!,a=h.edges[e0]!,b=h.edges[e1]!,vertex=sharedVertex(a,b),valence=vertexCells.get(pkey(vertex))?.size??0;
    if(valence!==3&&valence!==4)throw new Error(`unexpected dual-cell valence ${valence}`);
    const d0=dualVectorAtCell(h,e0,c),d1=dualVectorAtCell(h,e1,c),area=Math.hypot(...cross3(d0,d1));
    if(!(area>0))throw new Error('degenerate dual corner');
    // A8 contribution to U_e from the other edge e'. Evaluating both members
    // of the pair explicitly preserves the symmetric quadratic construction.
    U[e0]=U[e0]!+((V[e0]!*d1[0]-V[e1]!*d0[0])*d1[0]+(V[e0]!*d1[1]-V[e1]!*d0[1])*d1[1]+(V[e0]!*d1[2]-V[e1]!*d0[2])*d1[2])/(valence*area);
    U[e1]=U[e1]!+((V[e1]!*d0[0]-V[e0]!*d1[0])*d0[0]+(V[e1]!*d0[1]-V[e0]!*d1[1])*d0[1]+(V[e1]!*d0[2]-V[e0]!*d1[2])*d0[2])/(valence*area);
  }
  return U;
}

function run(n:number){
  const h=buildCubedSphere(n),V=new Float64Array(h.edgeCount),target=new Float64Array(h.edgeCount);
  for(let e=0;e<h.edgeCount;e++){V[e]=dualCirculation(h,e);target[e]=exactFaceFlux(h,e);}
  const U=applySymmetricH(h,V);let err2=0,ref2=0,max=0,energy=0;
  for(let e=0;e<h.edgeCount;e++){const d=U[e]!-target[e]!;err2+=d*d;ref2+=target[e]!*target[e]!;max=Math.max(max,Math.abs(d));energy+=.5*V[e]!*U[e]!;}
  return{n,relL2:Math.sqrt(err2/Math.max(ref2,1e-60)),maxOverRms:max/Math.sqrt(ref2/h.edgeCount),energy};
}

try{
  const rows=[4,8,16,32].map(run),ratio=(a:number,b:number)=>a/Math.max(b,1e-30);
  console.log('Stage4 nonorthogonal symmetric Hodge prototype: dual circulation V -> primal normal flux U');
  console.log('N\trelL2 flux error\tmaxAbs/refRMS\t0.5 V^T H V');
  for(const r of rows)console.log(`${r.n}\t${r.relL2.toExponential(8)}\t${r.maxOverRms.toExponential(8)}\t${r.energy.toExponential(8)}`);
  console.log(`L2 refine 8->16=${ratio(rows[1]!.relL2,rows[2]!.relL2).toFixed(3)} 16->32=${ratio(rows[2]!.relL2,rows[3]!.relL2).toFixed(3)}`);
  if(rows.some(r=>![r.relL2,r.maxOverRms,r.energy].every(Number.isFinite)))throw new Error('non-finite Hodge result');
  if(rows.some(r=>!(r.energy>0)))throw new Error('symmetric H quadratic energy is not positive for test wind');
}catch(e){console.error('FAIL Stage4 nonorthogonal Hodge prototype');console.error(e);process.exitCode=1;}
