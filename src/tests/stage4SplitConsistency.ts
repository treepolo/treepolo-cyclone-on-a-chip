declare const process:{exitCode?:number};
import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { computeStage4FrozenRhs } from '../solver/stage4Rk3SplitCpu.js';
import { cell3DIndex, createHydrostaticState, edge3DIndex, w3DIndex } from '../solver/state.js';

function assert(cond:unknown,msg:string):asserts cond{if(!cond)throw new Error(msg)}
function relErr(a:ArrayLike<number>,b:ArrayLike<number>):number{let n=0,d=0;for(let i=0;i<a.length;i++){const e=a[i]!-b[i]!;n+=e*e;d+=b[i]!*b[i]!;}return Math.sqrt(n/Math.max(d,Number.MIN_VALUE));}
interface Test{name:string;fn:()=>void}
const tests:Test[]=[];const test=(name:string,fn:()=>void)=>tests.push({name,fn});

test('Frozen split scalar RHS reconstructs the intended full discrete continuity flux',()=>{
  const h=buildCubedSphere(3),v=buildStretchedVerticalGrid(9,18000,1.25),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref),nz=v.nz,R=EARTH.radius;
  for(let q=0;q<s.rhoD.length;q++){
    const f=1+1.5e-3*Math.sin((q+1)*.271)+7e-4*Math.cos((q+3)*.119);
    s.rhoD[q]=s.rhoD[q]!*f;s.rhoThetaM[q]=s.rhoThetaM[q]!*(1+8e-4*Math.sin((q+5)*.197));
  }
  for(let q=0;q<s.uEdge.length;q++)s.uEdge[q]=3.2*Math.sin((q+1)*.163)-.7*Math.cos((q+2)*.071);
  for(let c=0;c<h.cellCount;c++)for(let i=1;i<nz;i++)s.wInterface[w3DIndex(c,i,nz)]=.18*Math.sin(Math.PI*v.zInterface[i]!/v.top)*Math.sin((c+1)*.233+i*.41);

  const frozen=computeStage4FrozenRhs(h,v,ref,s,{heldSuarez:false,momentumTransport:false,coriolis:false});
  const rho=new Float64Array(s.rhoD.length),x=new Float64Array(s.rhoThetaM.length);

  // Horizontal: rho0 is horizontally uniform, therefore Fref + Fpert equals
  // the ordinary full upwind scalar flux exactly.
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,edgeLength=ge.angularLength*R;
    for(let k=0;k<nz;k++){
      const qe=edge3DIndex(e,k,nz),vel=s.uEdge[qe]!,l=cell3DIndex(ge.leftCell,k,nz),r=cell3DIndex(ge.rightCell,k,nz),up=vel>=0?l:r,A=edgeLength*v.dz[k]!,fm=s.rhoD[up]!*vel*A,fx=s.rhoThetaM[up]!*vel*A;
      rho[l]=rho[l]!-fm;rho[r]=rho[r]!+fm;x[l]=x[l]!-fx;x[r]=x[r]!+fx;
    }
  }

  // Vertical: the locked HEVI discretization uses interface reference values
  // plus the upwind cell-centred perturbation. Reconstruct that exact effective
  // face carrier rather than substituting a different full-flux rule.
  for(let c=0;c<h.cellCount;c++){
    const area=h.cellAreaUnit[c]!*R*R;
    for(let i=1;i<nz;i++){
      const qi=w3DIndex(c,i,nz),vel=s.wInterface[qi]!,srcK=vel>=0?i-1:i,src=cell3DIndex(c,srcK,nz),faceRho=ref.rhoInterface[i]!+(s.rhoD[src]!-ref.rhoCenter[srcK]!),faceX=ref.rhoThetaInterface[i]!+(s.rhoThetaM[src]!-ref.rhoThetaCenter[srcK]!),fm=faceRho*vel*area,fx=faceX*vel*area,l=cell3DIndex(c,i-1,nz),u=cell3DIndex(c,i,nz);
      rho[l]=rho[l]!-fm;rho[u]=rho[u]!+fm;x[l]=x[l]!-fx;x[u]=x[u]!+fx;
    }
  }
  for(let c=0;c<h.cellCount;c++)for(let k=0;k<nz;k++){
    const q=cell3DIndex(c,k,nz),vol=h.cellAreaUnit[c]!*R*R*v.dz[k]!;rho[q]=rho[q]!/vol;x[q]=x[q]!/vol;
  }

  const er=relErr(frozen.rhoD,rho),ex=relErr(frozen.rhoThetaM,x);
  assert(er<3e-14,`split/full rho continuity mismatch=${er}`);
  assert(ex<3e-14,`split/full rhoTheta continuity mismatch=${ex}`);
});

test('Frozen split continuity has zero global scalar source with rigid vertical boundaries',()=>{
  const h=buildCubedSphere(3),v=buildStretchedVerticalGrid(8,14000,1.2),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref),R=EARTH.radius;
  for(let q=0;q<s.uEdge.length;q++)s.uEdge[q]=2*Math.sin((q+1)*.21);
  for(let c=0;c<h.cellCount;c++)for(let i=1;i<v.nz;i++)s.wInterface[w3DIndex(c,i,v.nz)]=.1*Math.cos((c+1)*.31+i*.4);
  const f=computeStage4FrozenRhs(h,v,ref,s,{heldSuarez:false,momentumTransport:false,coriolis:false});let mr=0,mx=0,sr=0,sx=0;
  for(let c=0;c<h.cellCount;c++)for(let k=0;k<v.nz;k++){
    const q=cell3DIndex(c,k,v.nz),vol=h.cellAreaUnit[c]!*R*R*v.dz[k]!,ar=f.rhoD[q]!*vol,ax=f.rhoThetaM[q]!*vol;mr+=ar;mx+=ax;sr+=Math.abs(ar);sx+=Math.abs(ax);
  }
  assert(Math.abs(mr)<=Math.max(1e-5,sr*3e-14),`global frozen rho source=${mr}`);
  assert(Math.abs(mx)<=Math.max(1e-3,sx*3e-14),`global frozen rhoTheta source=${mx}`);
});

let passed=0;for(const t of tests){try{t.fn();console.log(`PASS ${t.name}`);passed++}catch(e){console.error(`FAIL ${t.name}`);console.error(e);process.exitCode=1}}
console.log(`${passed}/${tests.length} Stage 4 split-consistency tests passed`);
