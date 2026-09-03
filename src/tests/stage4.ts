declare const process:{exitCode?:number};
import { EARTH } from '../core/constants.js';
import { dot3, Vec3 } from '../core/math.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { ACOUSTIC_DIVERGENCE_REFERENCE_COEFFICIENT, acousticDivergenceCoefficientForDt, acousticDivergenceRms, applyAcousticDivergenceDamping } from '../physics/acousticDivergenceDamping.js';
import { addHeldSuarezWavePerturbation, buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { buildModelTopSpongeRates, MODEL_TOP_SPONGE } from '../physics/modelTopSponge.js';
import { buildRotationGeometry, reconstructCellHorizontalWind, rotateLocalCoriolis, setAnalyticCellWind } from '../physics/rotation.js';
import { diagnoseState } from '../solver/diagnostics.js';
import { heviColumnStep } from '../solver/hevi.js';
import { RotatingDryCoreCpu } from '../solver/rotatingDryCoreCpu.js';
import { createHydrostaticState, w3DIndex } from '../solver/state.js';

function assert(cond:unknown,msg:string):asserts cond{if(!cond)throw new Error(msg)}
interface Test{name:string;fn:()=>void}
const tests:Test[]=[];
const test=(name:string,fn:()=>void)=>tests.push({name,fn});

test('V2 rotation geometry reconstructs solid-body wind across cubed-sphere seams',()=>{
  const h=buildCubedSphere(16),v=buildStretchedVerticalGrid(4,10000,1),ref=buildHeldSuarezReference(v),s=createHydrostaticState(h,v,ref),g=buildRotationGeometry(h),omega=2e-5;
  setAnalyticCellWind(h,g,s,r=>[-omega*6371000*r[1],omega*6371000*r[0],0]);
  const w=reconstructCellHorizontalWind(h,g,s,0);let num=0,den=0,seam=0,interior=0,ns=0,ni=0;
  for(let c=0;c<h.cellCount;c++){
    const r:Vec3=[h.cellCenters[c*3]!,h.cellCenters[c*3+1]!,h.cellCenters[c*3+2]!],exact:Vec3=[-omega*6371000*r[1],omega*6371000*r[0],0],err=Math.hypot(w[c*3]!-exact[0],w[c*3+1]!-exact[1],w[c*3+2]!-exact[2]);
    num+=err*err;den+=dot3(exact,exact);let isSeam=false;
    for(let slot=0;slot<4;slot++){const e=h.edges[h.cellEdges[c*4+slot]!]!;if(h.cellPanel[e.leftCell]!==h.cellPanel[e.rightCell])isSeam=true}
    if(isSeam){seam+=err*err;ns++}else{interior+=err*err;ni++}
  }
  const rel=Math.sqrt(num/den),ratio=Math.sqrt(seam/ns)/Math.sqrt(interior/ni);
  assert(rel<2e-3,`solid-body reconstruction relL2=${rel}`);assert(ratio<10,`seam/interior RMS ratio=${ratio}`);
});

test('V2 inertial oscillation preserves amplitude and period',()=>{
  const lat=45*Math.PI/180,f=2*EARTH.omega*Math.sin(lat),period=2*Math.PI/Math.abs(f),steps=1000,dt=period/steps;let u=12,v=-3;const u0=u,v0=v;
  for(let i=0;i<steps;i++) [u,v]=rotateLocalCoriolis(u,v,f,dt);
  assert(Math.hypot(u-u0,v-v0)/Math.hypot(u0,v0)<2e-13,`inertial cycle error=${Math.hypot(u-u0,v-v0)}`);
});

test('V2 discrete spherical pressure gradient is geostrophically balanced',()=>{
  const h=buildCubedSphere(32),g=buildRotationGeometry(h),A=2e4,R=EARTH.radius,edgeA=new Float64Array(h.edgeCount);
  for(let e=0;e<h.edgeCount;e++){const ge=h.edges[e]!,zl=h.cellCenters[ge.leftCell*3+2]!,zr=h.cellCenters[ge.rightCell*3+2]!,phiL=-A*zl*zl,phiR=-A*zr*zr;edgeA[e]=-(phiR-phiL)/(ge.centerDistanceAngle*R)}
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

test('V2 buoyancy forcing is inside the HEVI absorber instead of bypassing it',()=>{
  const v=buildStretchedVerticalGrid(48,40000,1.4),ref=buildHeldSuarezReference(v),rates=buildModelTopSpongeRates(v),i=v.nz-1,dt=1,accel=new Float64Array(v.nz+1);
  accel[i]=2;
  const free={rho:Float64Array.from(ref.rhoCenter),rhoTheta:Float64Array.from(ref.rhoThetaCenter),w:new Float64Array(v.nz+1)};
  const damp={rho:Float64Array.from(ref.rhoCenter),rhoTheta:Float64Array.from(ref.rhoThetaCenter),w:new Float64Array(v.nz+1)};
  heviColumnStep(v,ref,free,dt,.1,undefined,accel);
  heviColumnStep(v,ref,damp,dt,.1,rates,accel);
  const rate=rates[i]!,expected=free.w[i]!/(1+rate*dt);
  assert(Math.abs(free.w[i]!)>1e-6,`vertical forcing failed to produce w: ${free.w[i]}`);
  assert(Math.abs(damp.w[i]!-expected)<1e-12,`buoyancy/absorber coupling mismatch: got=${damp.w[i]}, expected=${expected}`);
  assert(Math.abs(damp.w[i]!)<Math.abs(free.w[i]!),`Rayleigh absorber was bypassed by vertical forcing: free=${free.w[i]}, damp=${damp.w[i]}`);
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
