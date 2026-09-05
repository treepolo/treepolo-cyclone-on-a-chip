declare const process:{exitCode?:number};
import { EARTH } from '../core/constants.js';
import { cross3,dot3,normalize3,type Vec3 } from '../core/math.js';
import type { CubedSphereGrid } from '../grid/cubedSphere.js';
import { applyNonorthogonalHodge,buildNonorthogonalHodgeStencil } from '../grid/nonorthogonalHodge.js';
import { buildModifiedCubedSphere } from '../grid/modifiedCubedSphere.js';

const GLX=[-0.9602898564975363,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956498,0.5255324099163290,0.7966664774136267,0.9602898564975363];
const GLW=[0.1012285362903763,0.2223810344533745,0.3137066458778873,0.3626837833783620,0.3626837833783620,0.3137066458778873,0.2223810344533745,0.1012285362903763];
const AXIS=normalize3([.31,-.47,.826] as Vec3),ROT=11;
function center(h:CubedSphereGrid,c:number):Vec3{return[h.cellCenters[c*3]!,h.cellCenters[c*3+1]!,h.cellCenters[c*3+2]!];}
function tangentToward(a:Vec3,b:Vec3):Vec3{const ab=dot3(a,b),d:Vec3=[b[0]-ab*a[0],b[1]-ab*a[1],b[2]-ab*a[2]];return normalize3(d);}
function basis(r:Vec3):{east:Vec3;north:Vec3}{const q=Math.hypot(r[0],r[1]),east:Vec3=q>1e-14?[-r[1]/q,r[0]/q,0]:[0,1,0];return{east,north:normalize3(cross3(r,east))};}
function wind(r:Vec3):Vec3{const [x,y,z]=r,g:Vec3=[5*z+3*y,3*x,24*z+5*x],gd=dot3(g,r),rot=cross3(AXIS,r);return[g[0]-gd*x+ROT*rot[0],g[1]-gd*y+ROT*rot[1],g[2]-gd*z+ROT*rot[2]];}
function coriolis(r:Vec3,u:Vec3):Vec3{const rxu=cross3(r,u),f=2*EARTH.omega*r[2];return[-f*rxu[0],-f*rxu[1],-f*rxu[2]];}
function arcSetup(a:Vec3,b:Vec3):{delta:number;t0:Vec3}{const ab=Math.max(-1,Math.min(1,dot3(a,b))),delta=Math.acos(ab),sd=Math.sin(delta);if(!(delta>0)||Math.abs(sd)<1e-14)throw new Error('degenerate arc');return{delta,t0:[(b[0]-ab*a[0])/sd,(b[1]-ab*a[1])/sd,(b[2]-ab*a[2])/sd]};}
function integrateArc(a:Vec3,b:Vec3,fn:(r:Vec3,t:Vec3)=>number):number{const {delta,t0}=arcSetup(a,b),half=.5*delta;let sum=0;for(let i=0;i<8;i++){const th=half*(1+GLX[i]!),co=Math.cos(th),si=Math.sin(th),r:Vec3=[co*a[0]+si*t0[0],co*a[1]+si*t0[1],co*a[2]+si*t0[2]],t:Vec3=[-si*a[0]+co*t0[0],-si*a[1]+co*t0[1],-si*a[2]+co*t0[2]];sum+=GLW[i]!*fn(r,t);}return EARTH.radius*half*sum;}
function analyticV(h:CubedSphereGrid):Float64Array{const a=new Float64Array(h.edgeCount);for(let e=0;e<h.edgeCount;e++){const ge=h.edges[e]!,l=center(h,ge.leftCell),r=center(h,ge.rightCell);a[e]=integrateArc(l,r,(x,t)=>dot3(wind(x),t));}return a;}
function analyticDV(h:CubedSphereGrid):Float64Array{const a=new Float64Array(h.edgeCount);for(let e=0;e<h.edgeCount;e++){const ge=h.edges[e]!,l=center(h,ge.leftCell),r=center(h,ge.rightCell);a[e]=integrateArc(l,r,(x,t)=>dot3(coriolis(x,wind(x)),t));}return a;}

interface SparseOp{rows:Array<Map<number,number>>;fitRms:number;maxFitRms:number;minPivot:number;}
const PAIRS:[[number,number],[number,number],[number,number],[number,number],[number,number],[number,number]]=[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]];
function solve6(G:number[][],b:number[]):{x:number[];minPivot:number}{const a=G.map((r,i)=>[...r,b[i]!]),n=6;let minPivot=Infinity;for(let k=0;k<n;k++){let p=k;for(let i=k+1;i<n;i++)if(Math.abs(a[i]![k]!)>Math.abs(a[p]![k]!))p=i;if(p!==k)[a[k],a[p]]=[a[p]!,a[k]!];const piv=a[k]![k]!;minPivot=Math.min(minPivot,Math.abs(piv));if(Math.abs(piv)<1e-20)throw new Error(`rank-deficient local skew fit pivot=${piv}`);for(let j=k;j<=n;j++)a[k]![j]=a[k]![j]!/piv;for(let i=0;i<n;i++)if(i!==k){const f=a[i]![k]!;if(f===0)continue;for(let j=k;j<=n;j++)a[i]![j]=a[i]![j]!-f*a[k]![j]!;}}return{x:Array.from({length:n},(_,i)=>a[i]![n]!),minPivot};}
function buildSkewFit(h:CubedSphereGrid):SparseOp{
  const rows:Array<Map<number,number>>=Array.from({length:h.edgeCount},()=>new Map<number,number>());let fit2=0,fitN=0,maxFitRms=0,minPivot=Infinity;
  for(let c=0;c<h.cellCount;c++){
    const cc=center(h,c),b=basis(cc),f=2*EARTH.omega*cc[2],edges=Array.from({length:4},(_,s)=>h.cellEdges[c*4+s]!);
    // N maps local east/north constant velocity to globally-oriented primal
    // normal fluxes on the four cell faces.
    const N=edges.map(eid=>{const e=h.edges[eid]!,L=e.angularLength*EARTH.radius;return[L*dot3(e.normal,b.east),L*dot3(e.normal,b.north)] as [number,number];});
    // B is this cell's half-contribution to the four globally-oriented dual
    // circulation tendencies for the same two constant basis winds.
    const B:number[][]=[];
    for(const eid of edges){const e=h.edges[eid]!,l=center(h,e.leftCell),r=center(h,e.rightCell),mid=normalize3([l[0]+r[0],l[1]+r[1],l[2]+r[2]] as Vec3),t=tangentToward(mid,r),Ld=Math.acos(Math.max(-1,Math.min(1,dot3(l,r))))*EARTH.radius;B.push([-.5*Ld*f*dot3(t,b.north),.5*Ld*f*dot3(t,b.east)]);}
    // Eight physical consistency equations (4 target edges x 2 tangent basis
    // winds) for the six independent entries of a 4x4 skew matrix.
    const M:number[][]=[],rhs:number[]=[];
    for(let i=0;i<4;i++)for(let q=0;q<2;q++){const row=new Array<number>(6).fill(0);for(let p=0;p<6;p++){const [a,z]=PAIRS[p]!;if(i===a)row[p]=N[z]![q]!;else if(i===z)row[p]=-N[a]![q]!;}M.push(row);rhs.push(B[i]![q]!);}
    const G=Array.from({length:6},()=>new Array<number>(6).fill(0)),g=new Array<number>(6).fill(0);for(let m=0;m<8;m++)for(let p=0;p<6;p++){g[p]+=M[m]![p]!*rhs[m]!;for(let q=0;q<6;q++)G[p]![q]+=M[m]![p]!*M[m]![q]!;}
    const sol=solve6(G,g);minPivot=Math.min(minPivot,sol.minPivot);let local2=0,ref2=0;for(let m=0;m<8;m++){let y=0;for(let p=0;p<6;p++)y+=M[m]![p]!*sol.x[p]!;const d=y-rhs[m]!;local2+=d*d;ref2+=rhs[m]!*rhs[m]!;}const local=Math.sqrt(local2/Math.max(ref2,1e-60));fit2+=local2;fitN+=ref2;maxFitRms=Math.max(maxFitRms,local);
    for(let p=0;p<6;p++){const [i,j]=PAIRS[p]!,ei=edges[i]!,ej=edges[j]!,x=sol.x[p]!,ri=rows[ei]!,rj=rows[ej]!;ri.set(ej,(ri.get(ej)??0)+x);rj.set(ei,(rj.get(ei)??0)-x);}
  }
  return{rows,fitRms:Math.sqrt(fit2/Math.max(fitN,1e-60)),maxFitRms,minPivot};
}
function apply(A:SparseOp,x:ArrayLike<number>):Float64Array{const y=new Float64Array(A.rows.length);for(let i=0;i<A.rows.length;i++){let s=0;for(const[j,w]of A.rows[i]!)s+=w*x[j]!;y[i]=s;}return y;}
function skewDefect(A:SparseOp):number{let e=0,s=0;for(let i=0;i<A.rows.length;i++)for(const[j,w]of A.rows[i]!){const z=A.rows[j]?.get(i)??0;e=Math.max(e,Math.abs(w+z));s=Math.max(s,Math.abs(w),Math.abs(z));}return e/Math.max(s,1e-30);}
function relError(a:ArrayLike<number>,b:ArrayLike<number>):number{let e=0,r=0;for(let i=0;i<a.length;i++){const d=a[i]!-b[i]!;e+=d*d;r+=b[i]!*b[i]!;}return Math.sqrt(e/Math.max(r,1e-60));}
function energyRate(V:ArrayLike<number>,U:ArrayLike<number>,dV:ArrayLike<number>):number{let vu=0,w=0;for(let e=0;e<V.length;e++){vu+=V[e]!*U[e]!;w+=U[e]!*dV[e]!;}return 2*w/Math.max(Math.abs(vu),1e-60)*86400;}

interface Rec{east:Float64Array;north:Float64Array;weights:Float64Array;dualLength:Float64Array;}
function buildRec(h:CubedSphereGrid):Rec{const east=new Float64Array(h.cellCount*3),north=new Float64Array(h.cellCount*3),weights=new Float64Array(h.cellCount*8),dualLength=new Float64Array(h.edgeCount);for(let e=0;e<h.edgeCount;e++){const ge=h.edges[e]!,l=center(h,ge.leftCell),r=center(h,ge.rightCell);dualLength[e]=Math.acos(Math.max(-1,Math.min(1,dot3(l,r))))*EARTH.radius;}for(let c=0;c<h.cellCount;c++){const cc=center(h,c),b=basis(cc);east.set(b.east,c*3);north.set(b.north,c*3);let aa=0,ab=0,bb=0;const v:Array<[number,number]>=[];for(let s=0;s<4;s++){const eid=h.cellEdges[c*4+s]!,ge=h.edges[eid]!,nb=ge.leftCell===c?ge.rightCell:ge.leftCell,t=tangentToward(cc,center(h,nb)),a=dot3(t,b.east),q=dot3(t,b.north);v.push([a,q]);aa+=a*a;ab+=a*q;bb+=q*q;}const det=aa*bb-ab*ab,i00=bb/det,i01=-ab/det,i11=aa/det;for(let s=0;s<4;s++){const[a,q]=v[s]!;weights[(c*4+s)*2]=i00*a+i01*q;weights[(c*4+s)*2+1]=i01*a+i11*q;}}return{east,north,weights,dualLength};}
function reconstruct(h:CubedSphereGrid,g:Rec,V:ArrayLike<number>):Float64Array{const out=new Float64Array(h.cellCount*3);for(let c=0;c<h.cellCount;c++){let ue=0,vn=0;for(let s=0;s<4;s++){const eid=h.cellEdges[c*4+s]!,ge=h.edges[eid]!,sgn=ge.leftCell===c?1:-1,q=sgn*V[eid]!/g.dualLength[eid]!;ue+=g.weights[(c*4+s)*2]!*q;vn+=g.weights[(c*4+s)*2+1]!*q;}const o=c*3;out[o]=ue*g.east[o]!+vn*g.north[o]!;out[o+1]=ue*g.east[o+1]!+vn*g.north[o+1]!;out[o+2]=ue*g.east[o+2]!+vn*g.north[o+2]!;}return out;}
function planet(h:CubedSphereGrid,U:ArrayLike<number>):number{let z=0;for(let c=0;c<h.cellCount;c++){let flux=0;for(let s=0;s<4;s++)flux+=h.cellEdgeSigns[c*4+s]!*U[h.cellEdges[c*4+s]!]!;const r=center(h,c);z-=flux*EARTH.omega*EARTH.radius*EARTH.radius*(1-r[2]*r[2]);}return z;}
function corTorque(h:CubedSphereGrid,g:Rec,dV:ArrayLike<number>):number{const a=reconstruct(h,g,dV);let z=0;for(let c=0;c<h.cellCount;c++){const r=center(h,c),b=basis(r),o=c*3,ae=a[o]!*b.east[0]+a[o+1]!*b.east[1]+a[o+2]!*b.east[2],area=h.cellAreaUnit[c]!*EARTH.radius*EARTH.radius;z+=area*EARTH.radius*Math.sqrt(Math.max(0,1-r[2]*r[2]))*ae;}return z;}
function lever(h:CubedSphereGrid):number{let z=0;for(let c=0;c<h.cellCount;c++){const r=center(h,c);z+=h.cellAreaUnit[c]!*EARTH.radius**3*Math.sqrt(Math.max(0,1-r[2]*r[2]));}return z;}
function run(n:number){const h=buildModifiedCubedSphere(n),V=analyticV(h),U=applyNonorthogonalHodge(buildNonorthogonalHodgeStencil(h),V),exact=analyticDV(h),C=buildSkewFit(h),d=apply(C,U),rec=buildRec(h),scale=86400/lever(h);return{n,fit:C.fitRms,maxFit:C.maxFitRms,minPivot:C.minPivot,dv:relError(d,exact),pair:(planet(h,U)+corTorque(h,rec,d))*scale,analyticPair:(planet(h,U)+corTorque(h,rec,exact))*scale,work:energyRate(V,U,d),skew:skewDefect(C)};}
try{const rows=[4,8,16,32].map(run),ratio=(a:number,b:number)=>Math.abs(a)/Math.max(Math.abs(b),1e-30);console.log('Stage4 modified-grid local skew-fit Coriolis; each cell fits constant tangent winds inside skew(4)');console.log('N\tcell fit RMS\tmax cell fit\tC dV L2\tanalytic pair\tC pair\twork/day\tskew defect\tmin normal pivot');for(const r of rows)console.log(`${r.n}\t${r.fit.toExponential(6)}\t${r.maxFit.toExponential(6)}\t${r.dv.toExponential(6)}\t${r.analyticPair.toExponential(7)}\t${r.pair.toExponential(7)}\t${r.work.toExponential(3)}\t${r.skew.toExponential(2)}\t${r.minPivot.toExponential(3)}`);console.log(`dV refine=${ratio(rows[1]!.dv,rows[2]!.dv).toFixed(3)},${ratio(rows[2]!.dv,rows[3]!.dv).toFixed(3)} pair refine=${ratio(rows[1]!.pair,rows[2]!.pair).toFixed(3)},${ratio(rows[2]!.pair,rows[3]!.pair).toFixed(3)}`);if(rows.some(r=>![r.fit,r.maxFit,r.minPivot,r.dv,r.pair,r.analyticPair,r.work,r.skew].every(Number.isFinite)))throw new Error('non-finite skew-fit result');if(rows.some(r=>r.skew>2e-13||Math.abs(r.work)>2e-11))throw new Error('skew-fit lost energy neutrality');}catch(e){console.error('FAIL Stage4 local skew-fit Coriolis');console.error(e);process.exitCode=1;}
