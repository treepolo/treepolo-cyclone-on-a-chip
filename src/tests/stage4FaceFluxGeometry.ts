declare const process:{exitCode?:number};
import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { buildRotationGeometry, setAnalyticCellWind } from '../physics/rotation.js';
import { createHydrostaticState, edge3DIndex } from '../solver/state.js';

function divergenceRel(n:number,kind:'projected'|'midpoint'|'face-average'):number{
  const h=buildCubedSphere(n),v=buildStretchedVerticalGrid(4,4000,1.05),ref=buildHeldSuarezReference(v),g=buildRotationGeometry(h),s=createHydrostaticState(h,v,ref),V=22;
  if(kind==='projected')setAnalyticCellWind(h,g,s,r=>[-V*r[1],V*r[0],0]);
  else for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,m=ge.midpoint,nn=ge.normal,a=ge.angularLength,f=kind==='face-average'?2*Math.sin(a/2)/a:1;
    const ux=-V*f*m[1],uy=V*f*m[0],ue=ux*nn[0]+uy*nn[1];
    for(let k=0;k<v.nz;k++)s.uEdge[edge3DIndex(e,k,v.nz)]=ue;
  }
  let err=0,areaSum=0,global=0;
  for(let c=0;c<h.cellCount;c++){
    const area=h.cellAreaUnit[c]!*EARTH.radius*EARTH.radius;let flux=0;
    for(let slot=0;slot<4;slot++){
      const e=h.cellEdges[c*4+slot]!,sign=h.cellEdgeSigns[c*4+slot]!,L=h.edges[e]!.angularLength*EARTH.radius;
      flux+=sign*s.uEdge[edge3DIndex(e,0,v.nz)]!*L;
    }
    const div=flux/area;err+=area*div*div;areaSum+=area;global+=flux;
  }
  const rms=Math.sqrt(err/areaSum),scale=V/EARTH.radius;
  if(Math.abs(global)>1e-5*Math.max(1,V*EARTH.radius))throw new Error(`N=${n} ${kind} global closed-sphere flux closure=${global}`);
  return rms/scale;
}
try{
  const ns=[4,8,16,32],rows=ns.map(n=>({n,projected:divergenceRel(n,'projected'),midpoint:divergenceRel(n,'midpoint'),faceAverage:divergenceRel(n,'face-average')})),ratio=(a:number,b:number)=>a/Math.max(b,1e-30);
  console.log('Stage4 divergence-free solid-body face-flux geometry diagnostic (RMS div normalized by V/R)');
  console.log('N\tcell->edge projected\texact midpoint\texact face-average');
  for(const r of rows)console.log(`${r.n}\t${r.projected.toExponential(8)}\t${r.midpoint.toExponential(8)}\t${r.faceAverage.toExponential(8)}`);
  console.log(`midpoint refine 8->16=${ratio(rows[1]!.midpoint,rows[2]!.midpoint).toFixed(3)} 16->32=${ratio(rows[2]!.midpoint,rows[3]!.midpoint).toFixed(3)}`);
  if(rows.some(r=>![r.projected,r.midpoint,r.faceAverage].every(Number.isFinite)))throw new Error('non-finite face-flux geometry diagnostic');
  if(rows[3]!.faceAverage>2e-10)throw new Error(`exact face-average solid-body flux should close to roundoff; N32 rel=${rows[3]!.faceAverage}`);
}catch(e){console.error('FAIL Stage4 face-flux geometry diagnostic');console.error(e);process.exitCode=1;}
