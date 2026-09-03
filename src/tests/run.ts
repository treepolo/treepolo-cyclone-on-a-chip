declare const process: { exitCode?: number };
import { EARTH, DRY_AIR } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildIsothermalReference } from '../physics/referenceAtmosphere.js';
import { diagnoseDry, pressureFromRhoTheta, rhoFromPT, thetaFromTP } from '../physics/thermodynamics.js';
import { diagnoseState } from '../solver/diagnostics.js';
import { DryCoreCpu } from '../solver/dryCoreCpu.js';
import { addThermalBubble } from '../solver/initialConditions.js';
import { createHydrostaticState } from '../solver/state.js';
import { heviColumnStep } from '../solver/hevi.js';
import { assert, near, relative } from './assert.js';

interface Test {name:string; fn:()=>void}
const tests:Test[]=[]; const test=(name:string,fn:()=>void)=>tests.push({name,fn});

test('V0 cubed-sphere area and topology',()=>{
  for(const n of [4,8,16]){
    const g=buildCubedSphere(n); relative(g.totalAreaUnit,4*Math.PI,2e-13,`area N=${n}`);
    assert(g.edgeCount===12*n*n,`edge count N=${n}`);
    for(const e of g.edges) assert(e.leftCell!==e.rightCell,'edge neighbors unique');
  }
});

test('V1 equation-of-state round trip',()=>{
  for(const T of [220,260,288,310]) for(const p of [20000,50000,100000]){
    const rho=rhoFromPT(p,T),theta=thetaFromTP(T,p),rt=rho*theta,p2=pressureFromRhoTheta(rt),d=diagnoseDry(rho,rt);
    relative(p2,p,3e-15,'EOS pressure'); relative(d.T,T,3e-15,'EOS T');
  }
});

test('V1 HEVI acoustic column stable at large acoustic CFL',()=>{
  const v=buildStretchedVerticalGrid(80,20000,0.000001), ref=buildIsothermalReference(v,288);
  const rho=ref.rhoCenter.slice(),x=ref.rhoThetaCenter.slice(),w=new Float64Array(v.nz+1);
  const mid=Math.floor(v.nz/2); x[mid]=x[mid]!*(1+1e-8);
  const minDz=Math.min(...v.dz), cs=Math.sqrt(DRY_AIR.gamma*DRY_AIR.rd*288), dt=8*minDz/cs;
  for(let i=0;i<300;i++) heviColumnStep(v,ref,{rho,rhoTheta:x,w},dt);
  assert([...rho,...x,...w].every(Number.isFinite),'HEVI finite'); assert(Math.max(...Array.from(w,Math.abs))<100,'HEVI bounded');
});


test('V1 acoustic standing-wave phase/amplitude with HEVI',()=>{
  const v=buildStretchedVerticalGrid(80,20000,1e-6), nz=v.nz, rho0=1, p0=100000, theta=p0/(DRY_AIR.rd*rho0), x0=rho0*theta;
  const fill=(n:number,value:number)=>{const a=new Float64Array(n);a.fill(value);return a;};
  const ref={T0:p0/(DRY_AIR.rd*rho0),pCenter:fill(nz,p0),rhoCenter:fill(nz,rho0),thetaCenter:fill(nz,theta),rhoThetaCenter:fill(nz,x0),pInterface:fill(nz+1,p0),rhoInterface:fill(nz+1,rho0),thetaInterface:fill(nz+1,theta),rhoThetaInterface:fill(nz+1,x0)};
  const rho=fill(nz,rho0),x=fill(nz,x0),w=new Float64Array(nz+1),w0=new Float64Array(nz+1),H=v.top,A=0.1;
  for(let i=0;i<=nz;i++){w[i]=A*Math.sin(Math.PI*v.zInterface[i]!/H);w0[i]=w[i]!;}
  const dz=H/nz,c=Math.sqrt(DRY_AIR.gamma*p0/rho0),keff=2/dz*Math.sin(Math.PI/(2*nz)),omega=c*keff,dt=5*dz/c,omegaCN=2/dt*Math.atan(omega*dt/2),period=2*Math.PI/omegaCN,steps=Math.round(period/dt);
  for(let i=0;i<steps;i++)heviColumnStep(v,ref,{rho,rhoTheta:x,w},dt);
  let num=0,den=0;for(let i=0;i<=nz;i++){num+=(w[i]!-w0[i]!)**2;den+=w0[i]!**2;}
  const relL2=Math.sqrt(num/den); assert(relL2<1e-3,`acoustic phase/amplitude relL2=${relL2}`);
});

test('V1 stratified gravity-wave response is bounded and conservative',()=>{
  const h=buildCubedSphere(6),v=buildStretchedVerticalGrid(28,24000,1.3),ref=buildIsothermalReference(v,288),s=createHydrostaticState(h,v,ref),core=new DryCoreCpu(h,v,ref);
  for(let c=0;c<h.cellCount;c++){
    const x=h.cellCenters[c*3]!, y=h.cellCenters[c*3+1]!, phase=Math.atan2(y,x);
    for(let k=0;k<v.nz;k++){
      const shape=Math.sin(Math.PI*v.zCenter[k]!/v.top)*Math.cos(2*phase), dtheta=0.02*shape, q=c*v.nz+k, theta=ref.thetaCenter[k]!+dtheta;
      s.rhoThetaM[q]=ref.rhoThetaCenter[k]!; s.rhoD[q]=s.rhoThetaM[q]!/theta;
    }
  }
  const m0=diagnoseState(h,v,s).dryMass; let peak=0;
  for(let i=0;i<80;i++){core.step(s,0.5);peak=Math.max(peak,diagnoseState(h,v,s).maxAbsW);}
  const d=diagnoseState(h,v,s); assert(peak>1e-4&&peak<10,`gravity-wave response peak=${peak}`);relative(d.dryMass,m0,5e-11,'gravity-wave mass');assert(!d.nan&&d.minRho>0&&d.minP>0,'gravity wave bounded');
});

test('V1 global hydrostatic rest remains at rest',()=>{
  const h=buildCubedSphere(4),v=buildStretchedVerticalGrid(24,40000,1.6),ref=buildIsothermalReference(v,288),s=createHydrostaticState(h,v,ref),core=new DryCoreCpu(h,v,ref);
  const m0=diagnoseState(h,v,s).dryMass;
  for(let i=0;i<20;i++) core.step(s,1.0);
  const d=diagnoseState(h,v,s); relative(d.dryMass,m0,5e-14,'hydrostatic mass'); assert(d.maxAbsW<1e-12,`hydrostatic w=${d.maxAbsW}`); assert(!d.nan,'hydrostatic finite');
});

test('V1 thermal bubble produces upward motion without pressure seeding',()=>{
  const h=buildCubedSphere(8),v=buildStretchedVerticalGrid(32,30000,1.5),ref=buildIsothermalReference(v,288),s=createHydrostaticState(h,v,ref),core=new DryCoreCpu(h,v,ref);
  addThermalBubble(h,v,ref,s,{lonDeg:0,latDeg:0,altitude:3000,horizontalRadius:1.2e6,verticalRadius:1800,deltaTheta:3});
  const m0=diagnoseState(h,v,s).dryMass; for(let i=0;i<30;i++)core.step(s,0.5); const d=diagnoseState(h,v,s);
  assert(d.maxAbsW>0.01,`bubble should move, maxw=${d.maxAbsW}`); relative(d.dryMass,m0,3e-12,'bubble mass conservation'); assert(d.minRho>0&&d.minP>0&&!d.nan,'bubble physical state');
});

let passed=0;
for(const t of tests){ try{t.fn();console.log(`PASS ${t.name}`);passed++;}catch(e){console.error(`FAIL ${t.name}`);console.error(e);process.exitCode=1;} }
console.log(`${passed}/${tests.length} tests passed`);
