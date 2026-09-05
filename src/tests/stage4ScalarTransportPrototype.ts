declare const process:{exitCode?:number};
import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { reconstructCellScalarGradient } from '../physics/horizontalGradient.js';
import { buildRotationGeometry, setAnalyticCellWind } from '../physics/rotation.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { createHydrostaticState, edge3DIndex } from '../solver/state.js';

function clampDot(x:number):number{return Math.max(-1,Math.min(1,x));}
function faceDelta(h:ReturnType<typeof buildCubedSphere>,g:ReturnType<typeof buildRotationGeometry>,c:number,eid:number):readonly[number,number,number]{
  const o=c*3,m=h.edges[eid]!.midpoint,rx=g.radial[o]!,ry=g.radial[o+1]!,rz=g.radial[o+2]!,mu=clampDot(rx*m[0]+ry*m[1]+rz*m[2]),ang=Math.acos(mu),tx=m[0]-mu*rx,ty=m[1]-mu*ry,tz=m[2]-mu*rz,tm=Math.hypot(tx,ty,tz);
  if(!(ang>0)||!(tm>0))return[0,0,0];const s=EARTH.radius*ang/tm;return[tx*s,ty*s,tz*s];
}
function limiter(h:ReturnType<typeof buildCubedSphere>,g:ReturnType<typeof buildRotationGeometry>,q:Float64Array,grad:Float64Array):Float64Array{
  const phi=new Float64Array(h.cellCount);phi.fill(1);
  for(let c=0;c<h.cellCount;c++){
    let lo=q[c]!,hi=q[c]!;for(let s=0;s<4;s++){const e=h.edges[h.cellEdges[c*4+s]!]!,nb=e.leftCell===c?e.rightCell:e.leftCell;lo=Math.min(lo,q[nb]!);hi=Math.max(hi,q[nb]!);}
    let p=1;const o=c*3;
    for(let s=0;s<4;s++){const d=faceDelta(h,g,c,h.cellEdges[c*4+s]!),delta=grad[o]!*d[0]+grad[o+1]!*d[1]+grad[o+2]!*d[2];if(delta>1e-15)p=Math.min(p,(hi-q[c]!)/delta);else if(delta<-1e-15)p=Math.min(p,(lo-q[c]!)/delta);}
    phi[c]=Math.max(0,Math.min(1,p));
  }
  return phi;
}
interface Row{n:number;donorL2:number;musclL2:number;donorAnomL2:number;musclAnomL2:number;divConstRel:number;donorCons:number;musclCons:number;faceViolation:number;}
function run(n:number):Row{
  const h=buildCubedSphere(n),v=buildStretchedVerticalGrid(4,4000,1.05),ref=buildHeldSuarezReference(v),g=buildRotationGeometry(h),s=createHydrostaticState(h,v,ref),V=22,a=.17,b=-.11,c0=.07;
  setAnalyticCellWind(h,g,s,r=>[-V*r[1],V*r[0],0]);
  const q=new Float64Array(h.cellCount),qa=new Float64Array(h.cellCount),exact=new Float64Array(h.cellCount),grad=new Float64Array(h.cellCount*3);
  for(let c=0;c<h.cellCount;c++){
    const o=c*3,x=h.cellCenters[o]!,y=h.cellCenters[o+1]!,z=h.cellCenters[o+2]!;qa[c]=a*x+b*y+c0*z;q[c]=1+qa[c]!;exact[c]=-V*(-a*y+b*x)/EARTH.radius;
  }
  for(let cc=0;cc<h.cellCount;cc++)grad.set(reconstructCellScalarGradient(h,g,cc,j=>q[j]!),cc*3);
  const phi=limiter(h,g,q,grad),donor=new Float64Array(h.cellCount),muscl=new Float64Array(h.cellCount),donorA=new Float64Array(h.cellCount),musclA=new Float64Array(h.cellCount),divConst=new Float64Array(h.cellCount);let faceViolation=0;
  for(let eid=0;eid<h.edgeCount;eid++){
    const e=h.edges[eid]!,ue=s.uEdge[edge3DIndex(eid,0,v.nz)]!,up=ue>=0?e.leftCell:e.rightCell,L=e.angularLength*EARTH.radius,d=faceDelta(h,g,up,eid),o=up*3,recon=phi[up]!*(grad[o]!*d[0]+grad[o+1]!*d[1]+grad[o+2]!*d[2]);
    const donorFace=q[up]!,musclFace=donorFace+recon,donorAFace=qa[up]!,musclAFace=donorAFace+recon;
    let lo=q[up]!,hi=q[up]!;for(let ss=0;ss<4;ss++){const ee=h.edges[h.cellEdges[up*4+ss]!]!,nb=ee.leftCell===up?ee.rightCell:ee.leftCell;lo=Math.min(lo,q[nb]!);hi=Math.max(hi,q[nb]!);}faceViolation=Math.max(faceViolation,lo-musclFace,musclFace-hi,0);
    const fd=ue*L*donorFace,fm=ue*L*musclFace,fda=ue*L*donorAFace,fma=ue*L*musclAFace,fc=ue*L,al=h.cellAreaUnit[e.leftCell]!*EARTH.radius*EARTH.radius,ar=h.cellAreaUnit[e.rightCell]!*EARTH.radius*EARTH.radius;
    donor[e.leftCell]=donor[e.leftCell]!-fd/al;donor[e.rightCell]=donor[e.rightCell]!+fd/ar;muscl[e.leftCell]=muscl[e.leftCell]!-fm/al;muscl[e.rightCell]=muscl[e.rightCell]!+fm/ar;
    donorA[e.leftCell]=donorA[e.leftCell]!-fda/al;donorA[e.rightCell]=donorA[e.rightCell]!+fda/ar;musclA[e.leftCell]=musclA[e.leftCell]!-fma/al;musclA[e.rightCell]=musclA[e.rightCell]!+fma/ar;
    divConst[e.leftCell]=divConst[e.leftCell]!-fc/al;divConst[e.rightCell]=divConst[e.rightCell]!+fc/ar;
  }
  let ed=0,em=0,eda=0,ema=0,ediv=0,scale=0,cd=0,cm=0,fluxScale=0;
  for(let cell=0;cell<h.cellCount;cell++){
    const A=h.cellAreaUnit[cell]!*EARTH.radius*EARTH.radius,ex=exact[cell]!;ed+=A*(donor[cell]!-ex)**2;em+=A*(muscl[cell]!-ex)**2;eda+=A*(donorA[cell]!-ex)**2;ema+=A*(musclA[cell]!-ex)**2;ediv+=A*divConst[cell]!**2;scale+=A*ex*ex;cd+=A*donor[cell]!;cm+=A*muscl[cell]!;fluxScale+=A*(Math.abs(donor[cell]!)+Math.abs(muscl[cell]!));
  }
  return{n,donorL2:Math.sqrt(ed/scale),musclL2:Math.sqrt(em/scale),donorAnomL2:Math.sqrt(eda/scale),musclAnomL2:Math.sqrt(ema/scale),divConstRel:Math.sqrt(ediv/scale),donorCons:Math.abs(cd)/Math.max(fluxScale,1e-30),musclCons:Math.abs(cm)/Math.max(fluxScale,1e-30),faceViolation};
}
try{
  const rows=[4,8,16,32].map(run),ratio=(a:number,b:number)=>a/Math.max(b,1e-30);
  console.log('Stage4 horizontal scalar transport manufactured solid-body rotation (diagnostic only)');
  console.log('N\tdonor total\tMUSCL total\tdonor anomaly\tMUSCL anomaly\tconstant div/error scale\tdonor cons\tMUSCL cons\tface violation');
  for(const r of rows)console.log(`${r.n}\t${r.donorL2.toExponential(7)}\t${r.musclL2.toExponential(7)}\t${r.donorAnomL2.toExponential(7)}\t${r.musclAnomL2.toExponential(7)}\t${r.divConstRel.toExponential(7)}\t${r.donorCons.toExponential(3)}\t${r.musclCons.toExponential(3)}\t${r.faceViolation.toExponential(3)}`);
  console.log(`MUSCL anomaly refine 8->16=${ratio(rows[1]!.musclAnomL2,rows[2]!.musclAnomL2).toFixed(3)} 16->32=${ratio(rows[2]!.musclAnomL2,rows[3]!.musclAnomL2).toFixed(3)}`);
  console.log(`constant-div refine 8->16=${ratio(rows[1]!.divConstRel,rows[2]!.divConstRel).toFixed(3)} 16->32=${ratio(rows[2]!.divConstRel,rows[3]!.divConstRel).toFixed(3)}`);
  if(rows.some(r=>![r.donorL2,r.musclL2,r.donorAnomL2,r.musclAnomL2,r.divConstRel,r.donorCons,r.musclCons,r.faceViolation].every(Number.isFinite)))throw new Error('non-finite scalar transport diagnostic');
  if(rows.some(r=>r.donorCons>2e-14||r.musclCons>2e-14))throw new Error('closed-sphere scalar transport lost global conservation');
  if(rows.some(r=>r.faceViolation>2e-14))throw new Error('limited scalar face reconstruction escaped local envelope');
}catch(e){console.error('FAIL Stage4 scalar transport prototype');console.error(e);process.exitCode=1;}
