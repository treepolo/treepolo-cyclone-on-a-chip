declare const process:{exitCode?:number};
import { dot3 } from '../core/math.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStage4GradientStencilData } from '../gpu/stage4GradientStencil.js';
import { reconstructCellScalarGradient, reconstructEdgeNormalScalarGradient } from '../physics/horizontalGradient.js';
import { buildRotationGeometry } from '../physics/rotation.js';

function assert(cond:unknown,msg:string):asserts cond{if(!cond)throw new Error(msg)}

try{
  const h=buildCubedSphere(16),g=buildRotationGeometry(h),st=buildStage4GradientStencilData(h);
  const value=(c:number)=>{const i=c*3,x=h.cellCenters[i]!,y=h.cellCenters[i+1]!,z=h.cellCenters[i+2]!;return .7*x-.3*y+.25*z*z+.11*x*y;};
  let err2=0,ref2=0,maxRel=0;
  for(let c=0;c<h.cellCount;c++){
    const vc=value(c),dv=new Float64Array(4);
    for(let s=0;s<4;s++)dv[s]=value(st.neighbors[c*4+s]!)-vc;
    let ge=0,gn=0;
    for(let s=0;s<4;s++){ge+=st.weights[c*8+s]!*dv[s]!;gn+=st.weights[c*8+4+s]!*dv[s]!;}
    const eo=c*3,e=[g.east[eo]!,g.east[eo+1]!,g.east[eo+2]!] as [number,number,number],n=[g.north[eo]!,g.north[eo+1]!,g.north[eo+2]!] as [number,number,number];
    const cached:[number,number,number]=[ge*e[0]+gn*n[0],ge*e[1]+gn*n[1],ge*e[2]+gn*n[2]],direct=reconstructCellScalarGradient(h,g,c,value);
    const dx=cached[0]-direct[0],dy=cached[1]-direct[1],dz=cached[2]-direct[2],er=dx*dx+dy*dy+dz*dz,rr=dot3(direct,direct);
    err2+=er;ref2+=rr;maxRel=Math.max(maxRel,Math.sqrt(er/Math.max(rr,1e-40)));
  }
  const rel=Math.sqrt(err2/ref2);
  assert(rel<2e-7,`cached gradient stencil relative L2=${rel}`);
  assert(maxRel<1e-6,`cached gradient stencil max relative error=${maxRel}`);

  let edgeErr2=0,edgeRef2=0,edgeMaxRel=0;
  for(let eid=0;eid<h.edgeCount;eid++){
    const edge=h.edges[eid]!,l=edge.leftCell,r=edge.rightCell,vl=value(l),vr=value(r);
    let cached=0;
    for(let s=0;s<4;s++)cached+=st.edgeProjectedWeights[eid*8+s]!*(value(st.neighbors[l*4+s]!)-vl);
    for(let s=0;s<4;s++)cached+=st.edgeProjectedWeights[eid*8+4+s]!*(value(st.neighbors[r*4+s]!)-vr);
    const direct=reconstructEdgeNormalScalarGradient(h,g,eid,value),d=cached-direct;
    edgeErr2+=d*d;edgeRef2+=direct*direct;edgeMaxRel=Math.max(edgeMaxRel,Math.abs(d)/Math.max(Math.abs(direct),1e-20));
  }
  const edgeRel=Math.sqrt(edgeErr2/edgeRef2);
  assert(edgeRel<2e-7,`edge-projected gradient stencil relative L2=${edgeRel}`);
  assert(edgeMaxRel<2e-5,`edge-projected gradient stencil max relative error=${edgeMaxRel}`);
  console.log(`PASS cached Stage 4 gradient stencil matches direct least-squares reference relL2=${rel} maxRel=${maxRel} edgeRelL2=${edgeRel} edgeMaxRel=${edgeMaxRel}`);
}catch(e){console.error('FAIL cached Stage 4 gradient stencil regression');console.error(e);process.exitCode=1;}
