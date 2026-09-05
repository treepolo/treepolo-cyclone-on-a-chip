declare const process:{exitCode?:number};
import { EARTH } from '../core/constants.js';
import { dot3, Vec3 } from '../core/math.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { ACOUSTIC_DIVERGENCE_REFERENCE_COEFFICIENT, acousticDivergenceCoefficientForDt, acousticDivergenceRms, applyAcousticDivergenceDamping } from '../physics/acousticDivergenceDamping.js';
import { reconstructEdgeNormalScalarGradient } from '../physics/horizontalGradient.js';
import { addHeldSuarezWavePerturbation, buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { buildModelTopSpongeRates, MODEL_TOP_SPONGE } from '../physics/modelTopSponge.js';
import { buildRotationGeometry, reconstructCellHorizontalWind, rotateLocalCoriolis, setAnalyticCellWind } from '../physics/rotation.js';
import { advectVerticalVelocity } from '../physics/verticalMomentumAdvection.js';
import { diagnoseState } from '../solver/diagnostics.js';
import { DryCoreCpu } from '../solver/dryCoreCpu.js';
import { heviColumnStep } from '../solver/hevi.js';
import { RotatingDryCoreCpu } from '../solver/rotatingDryCoreCpu.js';
import { createHydrostaticState, w3DIndex } from '../solver/state.js';

function assert(cond:unknown,msg:string):asserts cond{if(!cond)throw new Error(msg)}
interface Test{name:string;fn:()=>void}
const tests:Test[]=[];
const test=(name:string,fn:()=>void)=>tests.push({name,fn});

function solidBodyReconstructionError(n:number):{rel:number;seamRms:number;interiorRms:number}{
  const h=buildCubedSphere(n),v=buildStretchedVerticalGrid(4,10000,1),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref),g=buildRotationGeometry(h),omega=2e-5;
  setAnalyticCellWind(h,g,s,r=>[-omega*6371000*r[1],omega*6371000*r[0],0]);
  const w=reconstructCellHorizontalWind(h,g,s,0);let num=0,den=0,seam=0,interior=0,ns=0,ni=0;
  for(let c=0;c<h.cellCount;c++){
    const r:Vec3=[h.cellCenters[c*3]!,h.cellCenters[c*3+1]!,h.cellCenters[c*3+2]!],exact:Vec3=[-omega*6371000*r[1],omega*6371000*r[0],0],err=Math.hypot(w[c*3]!-exact[0],w[c*3+1]!-exact[1],w[c*3+2]!-exact[2]);
    num+=err*err;den+=dot3(exact,exact);let isSeam=false;
    for(let slot=0;slot<4;slot++){const e=h.edges[h.cellEdges[c*4+slot]!]!;if(h.cellPanel[e.leftCell]!==h.cellPanel[e.rightCell])isSeam=true}
    if(isSeam){seam+=err*err;ns++}else{interior+=err*err;ni++}
  }
  return{rel:Math.sqrt(num/den),seamRms:Math.sqrt(seam/ns),interiorRms:Math.sqrt(interior/ni)};
}

test('V2 face-normal C-grid reconstruction converges for solid-body wind across cubed-sphere seams',()=>{
  // With true face-normal DOFs, projecting cell winds to faces and reconstructing
  // them back is a spatial interpolation on the non-orthogonal grid. Global and
  // seam-local absolute errors must both fall under refinement. A seam/interior
  // ratio is not a convergence measure because the interior may converge faster.
  const a=solidBodyReconstructionError(16),b=solidBodyReconstructionError(32);
  assert(b.rel<a.rel*.4,`solid-body global reconstruction did not converge: N16=${a.rel}, N32=${b.rel}`);
  assert(b.rel<1e-3,`solid-body N32 reconstruction relL2=${b.rel}`);
  assert(b.seamRms<a.seamRms*.7,`solid-body seam RMS did not converge: N16=${a.seamRms}, N32=${b.seamRms}`);
});

test('V2 inertial oscillation preserves amplitude and period',()=>{
  const lat=45*Math.PI/180,f=2*EARTH.omega*Math.sin(lat),period=2*Math.PI/Math.abs(f),steps=1000,dt=period/steps;let u=12,v=-3;const u0=u,v0=v;
  for(let i=0;i<steps;i++) [u,v]=rotateLocalCoriolis(u,v,f,dt);
  assert(Math.hypot(u-u0,v-v0)/Math.hypot(u0,v0)<2e-13,`inertial cycle error=${Math.hypot(u-u0,v-v0)}`);
});

test('V2 nonorthogonal spherical pressure gradient is geostrophically balanced',()=>{
  const h=buildCubedSphere(32),g=buildRotationGeometry(h),A=2e4,R=EARTH.radius,edgeA=new Float64Array(h.edgeCount);
  const phi=(c:number)=>{const z=h.cellCenters[c*3+2]!;return-A*z*z};
  for(let e=0;e<h.edgeCount;e++)edgeA[e]=-reconstructEdgeNormalScalarGradient(h,g,e,phi,R);
  let num=0,den=0;
  for(let c=0;c<h.cellCount;c++){
    const z=h.cellCenters[c*3+2]!,lat=Math.asin(z);if(Math.abs(lat)<15*Math.PI/180)continue;let an=0;
    for(let slot=0;slot<4;slot++){const eid=h.cellEdges[c*4+slot]!,a=edgeA[eid]!;an+=g.reconstruction[(c*4+slot)*2+1]!*a}
    const f=2*EARTH.omega*z,uAnalytic=(2*A*Math.sin(lat)*Math.cos(lat)/R)/f,res=an-f*uAnalytic;num+=res*res;den+=an*an;
  }
  assert(Math.sqrt(num/den)<5e-3,`geostrophic RMS imbalance=${Math.sqrt(num/den)}`);
});

test('V2 divergence damping cadence is normalized in physical time',()=>{
  const c100=acousticDivergenceCoefficientForDt(100),c10=acousticDivergenceCoefficientForDt(10);
  assert(Math.abs(c100-ACOUSTIC_DIVERGENCE_REFERENCE_COEFFICIENT)<1e-14,`100 s coefficient=${c100}`);
  assert(Math.abs(Math.pow(1-c10,10)-(1-ACOUSTIC_DIVERGENCE_REFERENCE_COEFFICIENT))<1e-12,`10 s cadence mismatch coefficient=${c10}`);
  assert(c10>.01&&c10<.011,`10 s coefficient should be about 0.0105, got ${c10}`);
});

test('V2 horizontal acoustic divergence filter damps grid-scale divergent noise',()=>{
  const h=buildCubedSphere(8),v=buildStretchedVerticalGrid(8,20000,1.2),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref);
  for(let q=0;q<s.uEdge.length;q++)s.uEdge[q]=5*Math.sin((q+1)*12.9898)+2*Math.sin((q+1)*4.1414);
  const before=acousticDivergenceRms(h,v,ref,s);applyAcousticDivergenceDamping(h,v,ref,s,.1);const after=acousticDivergenceRms(h,v,ref,s);
  assert(after<before,`divergence filter did not damp: before=${before}, after=${after}`);assert(after/before<.95,`divergence damping too weak in regression: ratio=${after/before}`);
});

test('V2 horizontal acoustic filter does not convert vertical motion into horizontal wind',()=>{
  const h=buildCubedSphere(4),v=buildStretchedVerticalGrid(8,20000,1.2),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref);
  for(let c=0;c<h.cellCount;c++)for(let i=1;i<v.nz;i++)s.wInterface[w3DIndex(c,i,v.nz)]=0.25*Math.sin((c+1)*0.71+i*1.13);
  applyAcousticDivergenceDamping(h,v,ref,s,.1);
  let maxU=0;for(const u of s.uEdge)maxU=Math.max(maxU,Math.abs(u));
  assert(maxU===0,`vertical motion leaked into horizontal acoustic filter: max|u|=${maxU}`);
});

test('V2 implicit HEVI top absorber applies the configured Rayleigh profile before new-time fluxes',()=>{
  const v=buildStretchedVerticalGrid(48,40000,1.4),ref=buildHeldSuarezReference(v),rates=buildModelTopSpongeRates(v),i=v.nz-1,dt=1;
  const free={rho:Float64Array.from(ref.rhoCenter),rhoTheta:Float64Array.from(ref.rhoThetaCenter),w:new Float64Array(v.nz+1)};
  const damp={rho:Float64Array.from(ref.rhoCenter),rhoTheta:Float64Array.from(ref.rhoThetaCenter),w:new Float64Array(v.nz+1)};
  free.w[i]=1;damp.w[i]=1;
  heviColumnStep(v,ref,free,dt,.1);
  heviColumnStep(v,ref,damp,dt,.1,rates);
  const rate=rates[i]!,expected=free.w[i]!/(1+rate*dt);
  assert(MODEL_TOP_SPONGE.maxRate===.2,`unexpected top absorber peak rate=${MODEL_TOP_SPONGE.maxRate}`);
  assert(rate>.15,`upper interface Rayleigh rate unexpectedly weak: ${rate}`);
  assert(Math.abs(damp.w[i]!-expected)<1e-12,`implicit Rayleigh mismatch: got=${damp.w[i]}, expected=${expected}`);
  assert(Math.abs(damp.w[i]!)<Math.abs(free.w[i]!),`implicit top absorber did not reduce upper w: free=${free.w[i]}, damp=${damp.w[i]}`);
});

test('V2 split HEVI plus outer vertical mass flux reconstructs the actual density update',()=>{
  const h=buildCubedSphere(2),v=buildStretchedVerticalGrid(12,12000,1.1),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref),core=new DryCoreCpu(h,v,ref,.1),dt=.25;
  for(let c=0;c<h.cellCount;c++)for(let i=1;i<v.nz;i++)s.wInterface[w3DIndex(c,i,v.nz)]=.2*Math.sin(Math.PI*v.zInterface[i]!/v.top);
  const before=s.rhoD.slice();core.captureTransport=true;core.step(s,dt);const t=core.lastTransport;assert(t,'missing transport snapshot');
  let maxRel=0,maxFlux=0;
  for(const f of t.vMassFlux)maxFlux=Math.max(maxFlux,Math.abs(f));
  for(let c=0;c<h.cellCount;c++)for(let k=0;k<v.nz;k++){
    const q=c*v.nz+k,vol=h.cellAreaUnit[c]!*EARTH.radius*EARTH.radius*v.dz[k]!;let mt=0;
    for(let slot=0;slot<4;slot++){const e=h.cellEdges[c*4+slot]!,sgn=h.cellEdgeSigns[c*4+slot]!;mt-=sgn*t.hMassFlux[e*v.nz+k]!;}
    const vb=c*(v.nz+1)+k,vt=vb+1;mt+=t.vMassFlux[vb]!-t.vMassFlux[vt]!;
    const predicted=before[q]!+dt*mt/vol;maxRel=Math.max(maxRel,Math.abs(predicted-s.rhoD[q]!)/Math.max(Math.abs(s.rhoD[q]!),1e-12));
  }
  assert(maxFlux>0,'transport snapshot failed to capture HEVI reference mass flux');
  assert(maxRel<2e-13,`split-flux continuity mismatch: max relative error=${maxRel}`);
});

test('V2 vertical-velocity advection preserves a locally uniform w field',()=>{
  const h=buildCubedSphere(4),v=buildStretchedVerticalGrid(6,6000,1),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref),i=3;
  for(let q=0;q<s.uEdge.length;q++)s.uEdge[q]=3*Math.sin((q+1)*0.37);
  for(let c=0;c<h.cellCount;c++)for(let j=1;j<v.nz;j++)s.wInterface[w3DIndex(c,j,v.nz)]=1;
  advectVerticalVelocity(h,v,s,1);
  let err=0;for(let c=0;c<h.cellCount;c++)err=Math.max(err,Math.abs(s.wInterface[w3DIndex(c,i,v.nz)]!-1));
  assert(err<1e-14,`uniform interior w was changed by advection: max error=${err}`);
});

test('V2 vertical-velocity advection uses the upwind vertical donor',()=>{
  const h=buildCubedSphere(2),v=buildStretchedVerticalGrid(6,6000,1),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref),i=2,dt=1;
  for(let c=0;c<h.cellCount;c++){s.wInterface[w3DIndex(c,1,v.nz)]=.5;s.wInterface[w3DIndex(c,2,v.nz)]=1;s.wInterface[w3DIndex(c,3,v.nz)]=1.5;}
  const expected=1-dt*1*(1-.5)/v.dz[i-1]!;
  advectVerticalVelocity(h,v,s,dt);
  const got=s.wInterface[w3DIndex(0,i,v.nz)]!;
  assert(Math.abs(got-expected)<1e-14,`vertical donor mismatch: got=${got}, expected=${expected}`);
});

test('V2 rotating core keeps resting hydrostatic atmosphere at rest',()=>{
  const h=buildCubedSphere(4),v=buildStretchedVerticalGrid(16,30000,1.4),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref),core=new RotatingDryCoreCpu(h,v,ref),m0=diagnoseState(h,v,s).dryMass;
  for(let i=0;i<50;i++)core.step(s,1,{heldSuarez:false,momentumTransport:true,coriolis:true});
  const d=diagnoseState(h,v,s);assert(Math.abs((d.dryMass-m0)/m0)<1e-12,'rotating hydrostatic mass drift');assert(d.maxAbsW<1e-10,`rotating hydrostatic max|w|=${d.maxAbsW}`);assert(!d.nan&&d.minRho>0&&d.minP>0,'rotating hydrostatic state invalid');
});

test('V2 Held-Suarez one-day dry circulation develops without losing mass',()=>{
  const h=buildCubedSphere(4),v=buildStretchedVerticalGrid(12,30000,1.4),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref),core=new RotatingDryCoreCpu(h,v,ref);addHeldSuarezWavePerturbation(h,v,ref,s,.05);
  const m0=diagnoseState(h,v,s).dryMass,dt=10,steps=Math.round(86400/dt);let peakWind=0;
  for(let i=0;i<steps;i++){const cfl=core.step(s,dt);assert(cfl.maxHorizontalCfl<.8&&cfl.maxVerticalCfl<.8,`Held-Suarez CFL h=${cfl.maxHorizontalCfl},v=${cfl.maxVerticalCfl}`);if(i%200===0)for(const u of s.uEdge)peakWind=Math.max(peakWind,Math.abs(u))}
  const d=diagnoseState(h,v,s);assert(Math.abs((d.dryMass-m0)/m0)<1e-10,'Held-Suarez mass drift');assert(!d.nan&&d.minRho>0&&d.minP>0,'Held-Suarez state invalid');assert(peakWind>1e-3,'Held-Suarez forcing did not develop wind');
});

let passed=0;
for(const t of tests){try{t.fn();console.log(`PASS ${t.name}`);passed++}catch(e){console.error(`FAIL ${t.name}`);console.error(e);process.exitCode=1}}
console.log(`${passed}/${tests.length} Stage 4 CPU tests passed`);
