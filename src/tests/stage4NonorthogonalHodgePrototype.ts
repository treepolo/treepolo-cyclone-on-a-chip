declare const process:{exitCode?:number};
import { EARTH } from '../core/constants.js';
import { angle3,cross3,dot3,normalize3,scale3,type Vec3 } from '../core/math.js';
import { buildCubedSphere,type CubedSphereGrid,type HorizontalEdge } from '../grid/cubedSphere.js';

function pkey(p:Vec3):string{const q=1e11;return `${Math.round(p[0]*q)},${Math.round(p[1]*q)},${Math.round(p[2]*q)}`;}
function center(h:CubedSphereGrid,c:number):Vec3{return[h.cellCenters[c*3]!,h.cellCenters[c*3+1]!,h.cellCenters[c*3+2]!];}
function tangentToward(a:Vec3,b:Vec3):Vec3{const d:Vec3=[b[0]-dot3(a,b)*a[0],b[1]-dot3(a,b)*a[1],b[2]-dot3(a,b)*a[2]];return normalize3(d);}
function solidWind(r:Vec3):Vec3{const axis=normalize3([.37,-.21,.905] as Vec3),amp=27;return scale3(cross3(axis,r),amp);}
function sharedVertex(a:HorizontalEdge,b:HorizontalEdge):Vec3{
  const ka0=pkey(a.p0),ka1=pkey(a.p1),kb0=pkey(b.p0),kb1=pkey(b.p1);
  if(ka0===kb0||ka0===kb1)return a.p0;if(ka1===kb0||ka1===kb1)return a.p1;
  throw new Error('adjacent cell edges do not share a primal vertex');
}
function buildVertexCells(h:CubedSphereGrid):Map<string,Set<number>>{
  const out=new Map<string,Set<number>>();
  for(let c=0;c<h.cellCount;c++)for(let s=0;s<4;s++){
    const e=h.edges[h.cellEdges[c*4+s]!]!;
    for(const p of[e.p0,e.p1]){const k=pkey(p),set=out.get(k)??new Set<number>();set.add(c);out.set(k,set);}
  }
  return out;
}
function dualVectorAtCell(h:CubedSphereGrid,eid:number,c:number):Vec3{
  const e=h.edges[eid]!,cc=center(h,c),nb=e.leftCell===c?e.rightCell:e.leftCell,cn=center(h,nb),out=tangentToward(cc,cn),len=angle3(cc,cn)*EARTH.radius,sgn=e.leftCell===c?1:-1;
  return scale3(out,sgn*len);
}
function dualCirculation(h:CubedSphereGrid,eid:number):number{
  const e=h.edges[eid]!,l=center(h,e.leftCell),r=center(h,e.rightCell),m=normalize3([l[0]+r[0],l[1]+r[1],l[2]+r[2]]),t=tangentToward(m,r),len=angle3(l,r)*EARTH.radius;
  return dot3(solidWind(m),t)*len;
}
function exactFaceFlux(h:CubedSphereGrid,eid:number):number{const e=h.edges[eid]!;return dot3(solidWind(e.midpoint),e.normal)*e.angularLength*EARTH.radius;}

/** Thuburn-Cotter-Dubos (2014) Appendix-A symmetric H; diagnostic only. */
function applySymmetricH(h:CubedSphereGrid,V:Float64Array,vertexCells=buildVertexCells(h)):Float64Array{
  const U=new Float64Array(h.edgeCount);
  for(let c=0;c<h.cellCount;c++)for(let s=0;s<4;s++){
    const e0=h.cellEdges[c*4+s]!,e1=h.cellEdges[c*4+((s+1)%4)]!,a=h.edges[e0]!,b=h.edges[e1]!,vertex=sharedVertex(a,b),valence=vertexCells.get(pkey(vertex))?.size??0;
    if(valence!==3&&valence!==4)throw new Error(`unexpected dual-cell valence ${valence}`);
    const sc=valence===4?4:6;
    const d0=dualVectorAtCell(h,e0,c),d1=dualVectorAtCell(h,e1,c),area=Math.hypot(...cross3(d0,d1));
    if(!(area>0))throw new Error('degenerate dual corner');
    U[e0]=U[e0]!+((V[e0]!*d1[0]-V[e1]!*d0[0])*d1[0]+(V[e0]!*d1[1]-V[e1]!*d0[1])*d1[1]+(V[e0]!*d1[2]-V[e1]!*d0[2])*d1[2])/(sc*area);
    U[e1]=U[e1]!+((V[e1]!*d0[0]-V[e0]!*d1[0])*d0[0]+(V[e1]!*d0[1]-V[e0]!*d1[1])*d0[1]+(V[e1]!*d0[2]-V[e0]!*d1[2])*d0[2])/(sc*area);
  }
  return U;
}
interface Stats{count:number;err2:number;ref2:number;max:number;}
function add(s:Stats,d:number,r:number):void{s.count++;s.err2+=d*d;s.ref2+=r*r;s.max=Math.max(s.max,Math.abs(d));}
function rel(s:Stats):number{return Math.sqrt(s.err2/Math.max(s.ref2,1e-60));}
function run(n:number){
  const h=buildCubedSphere(n),vertexCells=buildVertexCells(h),V=new Float64Array(h.edgeCount),target=new Float64Array(h.edgeCount);
  for(let e=0;e<h.edgeCount;e++){V[e]=dualCirculation(h,e);target[e]=exactFaceFlux(h,e);}
  const U=applySymmetricH(h,V,vertexCells),all:Stats={count:0,err2:0,ref2:0,max:0},interior:Stats={count:0,err2:0,ref2:0,max:0},seam:Stats={count:0,err2:0,ref2:0,max:0},seamRegular:Stats={count:0,err2:0,ref2:0,max:0},corner:Stats={count:0,err2:0,ref2:0,max:0};let energy=0;
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,d=U[e]!-target[e]!,r=target[e]!,isSeam=h.cellPanel[ge.leftCell]!==h.cellPanel[ge.rightCell],isCorner=(vertexCells.get(pkey(ge.p0))?.size===3)||(vertexCells.get(pkey(ge.p1))?.size===3);
    add(all,d,r);if(isSeam)add(seam,d,r);else add(interior,d,r);if(isCorner)add(corner,d,r);else if(isSeam)add(seamRegular,d,r);energy+=.5*V[e]!*U[e]!;
  }
  // Symmetry check on deterministic non-physical edge vectors. H should be
  // symmetric independent of the flow field if the local assembly is right.
  const A=new Float64Array(h.edgeCount),B=new Float64Array(h.edgeCount);for(let e=0;e<h.edgeCount;e++){A[e]=Math.sin(.71*(e+1));B[e]=Math.cos(.37*(e+2));}
  const HA=applySymmetricH(h,A,vertexCells),HB=applySymmetricH(h,B,vertexCells);let aHb=0,bHa=0,norm=0;for(let e=0;e<h.edgeCount;e++){aHb+=A[e]!*HB[e]!;bHa+=B[e]!*HA[e]!;norm+=Math.abs(A[e]!*HB[e]!)+Math.abs(B[e]!*HA[e]!);}
  return{n,relL2:rel(all),interior:rel(interior),seam:rel(seam),seamRegular:rel(seamRegular),corner:rel(corner),counts:[interior.count,seamRegular.count,corner.count],maxOverRms:all.max/Math.sqrt(all.ref2/all.count),energy,symmetry:Math.abs(aHb-bHa)/Math.max(norm,1e-30)};
}

try{
  const rows=[4,8,16,32].map(run),ratio=(a:number,b:number)=>a/Math.max(b,1e-30);
  console.log('Stage4 nonorthogonal symmetric Hodge prototype: dual circulation V -> primal normal flux U');
  console.log('N\tall L2\tinterior L2\tregular-seam L2\tcorner L2\tmax/refRMS\tcounts i/s/c\tsymmetry\t0.5VHU');
  for(const r of rows)console.log(`${r.n}\t${r.relL2.toExponential(7)}\t${r.interior.toExponential(7)}\t${r.seamRegular.toExponential(7)}\t${r.corner.toExponential(7)}\t${r.maxOverRms.toExponential(7)}\t${r.counts.join('/')}\t${r.symmetry.toExponential(3)}\t${r.energy.toExponential(7)}`);
  console.log(`all refine 8->16=${ratio(rows[1]!.relL2,rows[2]!.relL2).toFixed(3)} 16->32=${ratio(rows[2]!.relL2,rows[3]!.relL2).toFixed(3)}`);
  console.log(`interior refine=${ratio(rows[1]!.interior,rows[2]!.interior).toFixed(3)},${ratio(rows[2]!.interior,rows[3]!.interior).toFixed(3)} regular-seam=${ratio(rows[1]!.seamRegular,rows[2]!.seamRegular).toFixed(3)},${ratio(rows[2]!.seamRegular,rows[3]!.seamRegular).toFixed(3)}`);
  if(rows.some(r=>![r.relL2,r.interior,r.seam,r.seamRegular,r.corner,r.maxOverRms,r.energy,r.symmetry].every(Number.isFinite)))throw new Error('non-finite Hodge result');
  if(rows.some(r=>!(r.energy>0)))throw new Error('symmetric H quadratic energy is not positive for test wind');
  if(rows.some(r=>r.symmetry>5e-13))throw new Error(`assembled H is not symmetric: ${rows.map(r=>r.symmetry).join(',')}`);
}catch(e){console.error('FAIL Stage4 nonorthogonal Hodge prototype');console.error(e);process.exitCode=1;}
