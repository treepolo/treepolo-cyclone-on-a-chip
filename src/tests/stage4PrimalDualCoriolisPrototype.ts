declare const process:{exitCode?:number};
import { EARTH } from '../core/constants.js';
import { cross3,dot3,normalize3,type Vec3 } from '../core/math.js';
import { buildCubedSphere,type CubedSphereGrid } from '../grid/cubedSphere.js';
import { applyNonorthogonalHodge,buildNonorthogonalHodgeStencil } from '../grid/nonorthogonalHodge.js';
import { buildModifiedCubedSphere } from '../grid/modifiedCubedSphere.js';
import { buildRotationGeometry } from '../physics/rotation.js';

const GLX=[-0.9602898564975363,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956498,0.5255324099163290,0.7966664774136267,0.9602898564975363];
const GLW=[0.1012285362903763,0.2223810344533745,0.3137066458778873,0.3626837833783620,0.3626837833783620,0.3137066458778873,0.2223810344533745,0.1012285362903763];
const AXIS=normalize3([.31,-.47,.826] as Vec3),ROT=11;
function center(h:CubedSphereGrid,c:number):Vec3{return[h.cellCenters[c*3]!,h.cellCenters[c*3+1]!,h.cellCenters[c*3+2]!];}
function tangentToward(a:Vec3,b:Vec3):Vec3{const ab=dot3(a,b),d:Vec3=[b[0]-ab*a[0],b[1]-ab*a[1],b[2]-ab*a[2]];return normalize3(d);}
function basis(r:Vec3):{east:Vec3;north:Vec3}{const q=Math.hypot(r[0],r[1]),east:Vec3=q>1e-14?[-r[1]/q,r[0]/q,0]:[0,1,0];return{east,north:normalize3(cross3(r,east))};}
function wind(r:Vec3):Vec3{const [x,y,z]=r,g:Vec3=[5*z+3*y,3*x,24*z+5*x],gd=dot3(g,r),rot=cross3(AXIS,r);return[g[0]-gd*x+ROT*rot[0],g[1]-gd*y+ROT*rot[1],g[2]-gd*z+ROT*rot[2]];}
function coriolis(r:Vec3,u:Vec3):Vec3{const rxu=cross3(r,u),f=2*EARTH.omega*r[2];return[-f*rxu[0],-f*rxu[1],-f*rxu[2]];}
function arcSetup(a:Vec3,b:Vec3):{delta:number;t0:Vec3}{const ab=Math.max(-1,Math.min(1,dot3(a,b))),delta=Math.acos(ab),sd=Math.sin(delta);if(!(delta>0)||Math.abs(sd)<1e-14)throw new Error('degenerate geodesic arc');return{delta,t0:[(b[0]-ab*a[0])/sd,(b[1]-ab*a[1])/sd,(b[2]-ab*a[2])/sd]};}
function integrateArc(a:Vec3,b:Vec3,fn:(r:Vec3,t:Vec3)=>number):number{const {delta,t0}=arcSetup(a,b),half=.5*delta;let sum=0;for(let i=0;i<GLX.length;i++){const th=half+half*GLX[i]!,co=Math.cos(th),si=Math.sin(th),r:Vec3=[co*a[0]+si*t0[0],co*a[1]+si*t0[1],co*a[2]+si*t0[2]],t:Vec3=[-si*a[0]+co*t0[0],-si*a[1]+co*t0[1],-si*a[2]+co*t0[2]];sum+=GLW[i]!*fn(r,t);}return EARTH.radius*half*sum;}
function analyticV(h:CubedSphereGrid):Float64Array{const V=new Float64Array(h.edgeCount);for(let e=0;e<h.edgeCount;e++){const ge=h.edges[e]!,l=center(h,ge.leftCell),r=center(h,ge.rightCell);V[e]=integrateArc(l,r,(x,t)=>dot3(wind(x),t));}return V;}
function analyticDV(h:CubedSphereGrid):Float64Array{const A=new Float64Array(h.edgeCount);for(let e=0;e<h.edgeCount;e++){const ge=h.edges[e]!,l=center(h,ge.leftCell),r=center(h,ge.rightCell);A[e]=integrateArc(l,r,(x,t)=>dot3(coriolis(x,wind(x)),t));}return A;}
function exactFaceU(h:CubedSphereGrid):Float64Array{const U=new Float64Array(h.edgeCount);for(let e=0;e<h.edgeCount;e++){const ge=h.edges[e]!;U[e]=integrateArc(ge.p0,ge.p1,x=>dot3(wind(x),ge.normal));}return U;}

interface SparseOp{rows:Array<Map<number,number>>;}
function add(row:Map<number,number>,col:number,w:number):void{row.set(col,(row.get(col)??0)+w);}
/**
 * Consistent local candidate A: primal normal flux U -> dual-circulation
 * Coriolis tendency Vdot. It reconstructs each adjacent cell velocity from
 * U/faceLength using the existing 4-face LS geometry, applies the traditional
 * cell Coriolis acceleration, then projects the symmetric two-cell average
 * onto the target dual edge. A has at most eight source edges per row.
 */
function buildLocalA(h:CubedSphereGrid):SparseOp{
  const rg=buildRotationGeometry(h),rows:Array<Map<number,number>>=Array.from({length:h.edgeCount},()=>new Map<number,number>());
  for(let target=0;target<h.edgeCount;target++){
    const te=h.edges[target]!,l=center(h,te.leftCell),r=center(h,te.rightCell),mid=normalize3([l[0]+r[0],l[1]+r[1],l[2]+r[2]] as Vec3),t=tangentToward(mid,r),dualLength=Math.acos(Math.max(-1,Math.min(1,dot3(l,r))))*EARTH.radius;
    for(const c of[te.leftCell,te.rightCell]){
      const o=c*3,east:Vec3=[rg.east[o]!,rg.east[o+1]!,rg.east[o+2]!],north:Vec3=[rg.north[o]!,rg.north[o+1]!,rg.north[o+2]!],f=2*EARTH.omega*rg.radial[o+2]!,teast=dot3(t,east),tnorth=dot3(t,north);
      for(let s=0;s<4;s++){
        const src=h.cellEdges[c*4+s]!,re=rg.reconstruction[(c*4+s)*2]!,rn=rg.reconstruction[(c*4+s)*2+1]!,faceLength=h.edges[src]!.angularLength*EARTH.radius;
        // a_e=f*v_n, a_n=-f*u_e; target contribution is half the
        // two-cell average projected on the target dual tangent.
        add(rows[target]!,src,.5*dualLength*f*(rn*teast-re*tnorth)/faceLength);
      }
    }
  }
  return{rows};
}
function skewPart(A:SparseOp):SparseOp{const n=A.rows.length,rows:Array<Map<number,number>>=Array.from({length:n},()=>new Map<number,number>());for(let i=0;i<n;i++)for(const [j,aij] of A.rows[i]!){const aji=A.rows[j]?.get(i)??0;const c=.5*(aij-aji);if(c!==0){rows[i]!.set(j,c);rows[j]!.set(i,-c);}}return{rows};}
function apply(A:SparseOp,x:ArrayLike<number>):Float64Array{const y=new Float64Array(A.rows.length);for(let i=0;i<A.rows.length;i++){let s=0;for(const [j,w] of A.rows[i]!)s+=w*x[j]!;y[i]=s;}return y;}
function skewDefect(A:SparseOp):number{let err=0,scale=0;for(let i=0;i<A.rows.length;i++)for(const [j,w] of A.rows[i]!){const wt=A.rows[j]?.get(i)??0;err=Math.max(err,Math.abs(w+wt));scale=Math.max(scale,Math.abs(w),Math.abs(wt));}return err/Math.max(scale,1e-30);}

interface DualReconstruction{east:Float64Array;north:Float64Array;weights:Float64Array;dualLength:Float64Array;}
function buildDualReconstruction(h:CubedSphereGrid):DualReconstruction{const east=new Float64Array(h.cellCount*3),north=new Float64Array(h.cellCount*3),weights=new Float64Array(h.cellCount*8),dualLength=new Float64Array(h.edgeCount);for(let e=0;e<h.edgeCount;e++){const ge=h.edges[e]!,l=center(h,ge.leftCell),r=center(h,ge.rightCell);dualLength[e]=Math.acos(Math.max(-1,Math.min(1,dot3(l,r))))*EARTH.radius;}for(let c=0;c<h.cellCount;c++){const cc=center(h,c),b=basis(cc);east.set(b.east,c*3);north.set(b.north,c*3);let aa=0,ab=0,bb=0;const comps:Array<[number,number]>=[];for(let s=0;s<4;s++){const eid=h.cellEdges[c*4+s]!,ge=h.edges[eid]!,nb=ge.leftCell===c?ge.rightCell:ge.leftCell,dir=tangentToward(cc,center(h,nb)),a=dot3(dir,b.east),q=dot3(dir,b.north);comps.push([a,q]);aa+=a*a;ab+=a*q;bb+=q*q;}const det=aa*bb-ab*ab;if(Math.abs(det)<1e-14)throw new Error(`singular dual reconstruction at cell ${c}`);const i00=bb/det,i01=-ab/det,i11=aa/det;for(let s=0;s<4;s++){const [a,q]=comps[s]!;weights[(c*4+s)*2]=i00*a+i01*q;weights[(c*4+s)*2+1]=i01*a+i11*q;}}return{east,north,weights,dualLength};}
function reconstruct(h:CubedSphereGrid,g:DualReconstruction,V:ArrayLike<number>):Float64Array{const out=new Float64Array(h.cellCount*3);for(let c=0;c<h.cellCount;c++){let ue=0,vn=0;for(let s=0;s<4;s++){const eid=h.cellEdges[c*4+s]!,ge=h.edges[eid]!,sign=ge.leftCell===c?1:-1,q=sign*V[eid]!/g.dualLength[eid]!;ue+=g.weights[(c*4+s)*2]!*q;vn+=g.weights[(c*4+s)*2+1]!*q;}const o=c*3;out[o]=ue*g.east[o]!+vn*g.north[o]!;out[o+1]=ue*g.east[o+1]!+vn*g.north[o+1]!;out[o+2]=ue*g.east[o+2]!+vn*g.north[o+2]!;}return out;}
function planetaryMassTorque(h:CubedSphereGrid,U:ArrayLike<number>):number{let torque=0;for(let c=0;c<h.cellCount;c++){let flux=0;for(let s=0;s<4;s++)flux+=h.cellEdgeSigns[c*4+s]!*U[h.cellEdges[c*4+s]!]!;const r=center(h,c),specific=EARTH.omega*EARTH.radius*EARTH.radius*Math.max(0,1-r[2]*r[2]);torque+=-flux*specific;}return torque;}
function relativeCoriolisTorque(h:CubedSphereGrid,g:DualReconstruction,dV:ArrayLike<number>):number{const a=reconstruct(h,g,dV);let torque=0;for(let c=0;c<h.cellCount;c++){const r=center(h,c),b=basis(r),o=c*3,ae=a[o]!*b.east[0]+a[o+1]!*b.east[1]+a[o+2]!*b.east[2],area=h.cellAreaUnit[c]!*EARTH.radius*EARTH.radius,lever=EARTH.radius*Math.sqrt(Math.max(0,1-r[2]*r[2]));torque+=area*lever*ae;}return torque;}
function leverMass(h:CubedSphereGrid):number{let m=0;for(let c=0;c<h.cellCount;c++){const r=center(h,c),area=h.cellAreaUnit[c]!*EARTH.radius*EARTH.radius,lever=EARTH.radius*Math.sqrt(Math.max(0,1-r[2]*r[2]));m+=area*lever;}return m;}
function relError(a:ArrayLike<number>,b:ArrayLike<number>):number{let e=0,r=0;for(let i=0;i<a.length;i++){const d=a[i]!-b[i]!;e+=d*d;r+=b[i]!*b[i]!;}return Math.sqrt(e/Math.max(r,1e-60));}
function energyRate(V:ArrayLike<number>,U:ArrayLike<number>,dV:ArrayLike<number>):number{let vu=0,work=0;for(let e=0;e<V.length;e++){vu+=V[e]!*U[e]!;work+=U[e]!*dV[e]!;}return 2*work/Math.max(Math.abs(vu),1e-60)*86400;}
interface Result{n:number;fluxErr:number;aErr:number;cErr:number;analyticPair:number;aPair:number;cPair:number;aWork:number;cWork:number;skew:number;}
function evaluate(h:CubedSphereGrid):Result{const V=analyticV(h),dExact=analyticDV(h),U=applyNonorthogonalHodge(buildNonorthogonalHodgeStencil(h),V),A=buildLocalA(h),C=skewPart(A),dA=apply(A,U),dC=apply(C,U),rec=buildDualReconstruction(h),scale=86400/leverMass(h),p=planetaryMassTorque(h,U);return{n:h.n,fluxErr:relError(U,exactFaceU(h)),aErr:relError(dA,dExact),cErr:relError(dC,dExact),analyticPair:(p+relativeCoriolisTorque(h,rec,dExact))*scale,aPair:(p+relativeCoriolisTorque(h,rec,dA))*scale,cPair:(p+relativeCoriolisTorque(h,rec,dC))*scale,aWork:energyRate(V,U,dA),cWork:energyRate(V,U,dC),skew:skewDefect(C)};}
try{console.log('Stage4 local primal-dual Coriolis prototype; U=HV, A=local reconstructed Coriolis, C=(A-A^T)/2');console.log('pair in equivalent m/s/day; work is relative dK/K per day.');console.log('grid N\tHflux L2\tA dV L2\tC dV L2\tanalytic pair\tA pair\tC pair\tA work/day\tC work/day\tskew defect');for(const kind of['raw','modified'] as const){const rows=[4,8,16,32].map(n=>evaluate(kind==='raw'?buildCubedSphere(n):buildModifiedCubedSphere(n)));for(const r of rows)console.log(`${kind} ${r.n}\t${r.fluxErr.toExponential(6)}\t${r.aErr.toExponential(6)}\t${r.cErr.toExponential(6)}\t${r.analyticPair.toExponential(7)}\t${r.aPair.toExponential(7)}\t${r.cPair.toExponential(7)}\t${r.aWork.toExponential(6)}\t${r.cWork.toExponential(6)}\t${r.skew.toExponential(2)}`);const ratio=(x:number,y:number)=>Math.abs(x)/Math.max(Math.abs(y),1e-30);console.log(`${kind} C-error refine=${ratio(rows[1]!.cErr,rows[2]!.cErr).toFixed(3)},${ratio(rows[2]!.cErr,rows[3]!.cErr).toFixed(3)} C-pair refine=${ratio(rows[1]!.cPair,rows[2]!.cPair).toFixed(3)},${ratio(rows[2]!.cPair,rows[3]!.cPair).toFixed(3)}`);if(rows.some(r=>![r.fluxErr,r.aErr,r.cErr,r.analyticPair,r.aPair,r.cPair,r.aWork,r.cWork,r.skew].every(Number.isFinite)))throw new Error(`${kind} non-finite Coriolis candidate`);if(rows.some(r=>r.skew>2e-13||Math.abs(r.cWork)>2e-11))throw new Error(`${kind} skew Coriolis lost exact energy neutrality`);}}
catch(e){console.error('FAIL Stage4 local primal-dual Coriolis prototype');console.error(e);process.exitCode=1;}
