declare const process:{exitCode?:number};
import { buildRk3SplitSchedule, rkPredictorFromBase, wickerSkamarockRk3Scalar } from '../solver/rk3SplitSchedule.js';

function assert(cond:unknown,msg:string):asserts cond{if(!cond)throw new Error(msg)}
interface Test{name:string;fn:()=>void}
const tests:Test[]=[];
const test=(name:string,fn:()=>void)=>tests.push({name,fn});

test('RK3 split ns=4 schedule is 1xdt/3, 2xdt/4, 4xdt/4',()=>{
  const s=buildRk3SplitSchedule(4);
  assert(s.length===3,'expected three RK stages');
  assert(Math.abs(s[0]!.targetFraction-1/3)<1e-15&&s[0]!.acousticSteps===1&&Math.abs(s[0]!.acousticDtFraction-1/3)<1e-15,'stage 1 schedule mismatch');
  assert(s[1]!.targetFraction===.5&&s[1]!.acousticSteps===2&&s[1]!.acousticDtFraction===.25,'stage 2 schedule mismatch');
  assert(s[2]!.targetFraction===1&&s[2]!.acousticSteps===4&&s[2]!.acousticDtFraction===.25,'stage 3 schedule mismatch');
  for(const st of s)assert(Math.abs(st.acousticSteps*st.acousticDtFraction-st.targetFraction)<1e-15,`acoustic duration does not reach stage target ${st.targetFraction}`);
});

test('RK3 split rejects odd acoustic ratios',()=>{
  let threw=false;try{buildRk3SplitSchedule(3)}catch{threw=true}assert(threw,'odd ns must be rejected');
});

test('RK predictors restart from the immutable large-step base',()=>{
  const base=10,dt=6;
  const stage1=rkPredictorFromBase(base,dt,1/3,2);
  const stage2=rkPredictorFromBase(base,dt,1/2,3);
  const stage3=rkPredictorFromBase(base,dt,1,4);
  assert(stage1===14,`stage1=${stage1}`);
  assert(stage2===19,`stage2=${stage2}`);
  assert(stage3===34,`stage3=${stage3}`);
  assert(stage2!==stage1+dt*.5*3,'stage2 was accidentally chained from stage1');
});

test('Wicker-Skamarock RK3 is exact for constant RHS',()=>{
  const x=wickerSkamarockRk3Scalar(7,2.5,()=>-4);
  assert(Math.abs(x-(7-10))<1e-15,`constant-RHS RK result=${x}`);
});

test('Wicker-Skamarock RK3 has cubic linear amplification polynomial',()=>{
  const x0=1,lambda=.2,dt=.5,z=lambda*dt;
  const got=wickerSkamarockRk3Scalar(x0,dt,x=>lambda*x);
  const expected=1+z+z*z/2+z*z*z/6;
  assert(Math.abs(got-expected)<1e-15,`linear amplification mismatch got=${got}, expected=${expected}`);
});

let passed=0;
for(const t of tests){try{t.fn();console.log(`PASS ${t.name}`);passed++}catch(e){console.error(`FAIL ${t.name}`);console.error(e);process.exitCode=1}}
console.log(`${passed}/${tests.length} RK3 split tests passed`);
