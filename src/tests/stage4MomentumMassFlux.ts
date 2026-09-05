declare const process:{exitCode?:number};
import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { computeHorizontalMaterialMomentumTendency, type HorizontalMomentumScheme } from '../physics/horizontalMomentumTransport.js';
import { buildRotationGeometry, reconstructCellHorizontalWind, setAnalyticCellWind } from '../physics/rotation.js';
import { diagnoseAxialAngularMomentum, diagnoseAxialAngularMomentumTendency } from '../solver/stage4CirculationDiagnostics.js';
import { computeStage4FrozenRhs } from '../solver/stage4Rk3SplitCpu.js';
import { computeStage4SlowTendencies } from '../solver/stage4SlowTendenciesCpu.js';
import { cell3DIndex, createHydrostaticState, edge3DIndex, type DryState } from '../solver/state.js';

interface Metric{aam:number;ke:number;}

function project(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,g:ReturnType<typeof buildRotationGeometry>,cellA:Float64Array[]):Float64Array{
  const uT=new Float64Array(h.edgeCount*v.nz);
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,l=ge.leftCell,r=ge.rightCell,n=ge.normal;
    for(let k=0;k<v.nz;k++){
      const a=cellA[k]!,lo=l*3,ro=r*3;
      uT[edge3DIndex(e,k,v.nz)]=.5*((a[lo]!+a[ro]!)*n[0]+(a[lo+1]!+a[ro+1]!)*n[1]+(a[lo+2]!+a[ro+2]!)*n[2]);
    }
  }
  return uT;
}

function metric(
  h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,g:ReturnType<typeof buildRotationGeometry>,s:DryState,rhoT:Float64Array,uT:Float64Array,
):Metric{
  const zeroU=new Float64Array(uT.length),zeroRho=new Float64Array(rhoT.length),mass=diagnoseAxialAngularMomentumTendency(h,v,s,rhoT,zeroU,g),mom=diagnoseAxialAngularMomentumTendency(h,v,s,zeroRho,uT,g).velocityTorque,lever=diagnoseAxialAngularMomentum(h,v,s,g).torqueLeverMass;
  const accelState:DryState={rhoD:s.rhoD,rhoThetaM:s.rhoThetaM,uEdge:uT,wInterface:s.wInterface,time:s.time};
  let dke=0,totalMass=0;
  for(let k=0;k<v.nz;k++){
    const w=reconstructCellHorizontalWind(h,g,s,k),a=reconstructCellHorizontalWind(h,g,accelState,k);
    for(let c=0;c<h.cellCount;c++){
      const q=cell3DIndex(c,k,v.nz),o=c*3,vol=h.cellAreaUnit[c]!*EARTH.radius*EARTH.radius*v.dz[k]!,rho=s.rhoD[q]!,speed2=w[o]!*w[o]!+w[o+1]!*w[o+1]!+w[o+2]!*w[o+2]!;
      dke+=vol*(.5*rhoT[q]!*speed2+rho*(w[o]!*a[o]!+w[o+1]!*a[o+1]!+w[o+2]!*a[o+2]!));totalMass+=rho*vol;
    }
  }
  return{aam:(mass.relativeMassRedistributionTorque+mom)/lever*86400,ke:dke/totalMass*86400};
}

function run(n:number,varyDensity:boolean){
  const h=buildCubedSphere(n),v=buildStretchedVerticalGrid(4,12000,1.15),ref=buildHeldSuarezReference(v),g=buildRotationGeometry(h),s=createHydrostaticState(h,v,ref);
  if(varyDensity){
    for(let c=0;c<h.cellCount;c++){
      const x=h.cellCenters[c*3]!,y=h.cellCenters[c*3+1]!,z=h.cellCenters[c*3+2]!,f=1+.14*(.55*x-.35*y+.25*z+.20*x*y);
      for(let k=0;k<v.nz;k++){const q=cell3DIndex(c,k,v.nz);s.rhoD[q]=s.rhoD[q]!*f;s.rhoThetaM[q]=s.rhoThetaM[q]!*f;}
    }
  }
  setAnalyticCellWind(h,g,s,(r,east,north,k)=>{
    const zz=1+.08*k,ue=zz*(11+4*r[0]-3*r[1]+2*r[2]),vn=zz*(6*r[0]+5*r[1]-4*r[2]+2*r[0]*r[1]);
    return[ue*east[0]+vn*north[0],ue*east[1]+vn*north[1],ue*east[2]+vn*north[2]];
  });
  const full=computeStage4FrozenRhs(h,v,ref,s,{momentumTransport:false,coriolis:false,heldSuarez:false},g),legacy=computeStage4SlowTendencies(h,v,ref,s,{momentumTransport:true,coriolis:false,heldSuarez:false},g).uEdge,winds=Array.from({length:v.nz},(_,k)=>reconstructCellHorizontalWind(h,g,s,k));
  const candidate=(scheme:HorizontalMomentumScheme)=>project(h,v,g,computeHorizontalMaterialMomentumTendency(h,v,s,g,scheme,winds));
  return{
    n,
    legacy:metric(h,v,g,s,full.rhoD,legacy),
    massDonor:metric(h,v,g,s,full.rhoD,candidate('mass-donor')),
    muscl:metric(h,v,g,s,full.rhoD,candidate('muscl-bj')),
  };
}

try{
  for(const varying of [false,true]){
    const rows=[4,8,16,32].map(n=>run(n,varying));
    console.log(`Stage4 mass-flux momentum diagnostic; horizontal density ${varying?'VARIES':'uniform'} (AAM m/s/day, KE m^2/s^2/day)`);
    console.log('N\tlegacy AAM\tmass-donor AAM\tMUSCL-BJ AAM\tlegacy KE\tmass-donor KE\tMUSCL-BJ KE');
    for(const r of rows)console.log(`${r.n}\t${r.legacy.aam.toExponential(8)}\t${r.massDonor.aam.toExponential(8)}\t${r.muscl.aam.toExponential(8)}\t${r.legacy.ke.toExponential(8)}\t${r.massDonor.ke.toExponential(8)}\t${r.muscl.ke.toExponential(8)}`);
    const all=rows.flatMap(r=>[r.legacy.aam,r.massDonor.aam,r.muscl.aam,r.legacy.ke,r.massDonor.ke,r.muscl.ke]);if(!all.every(Number.isFinite))throw new Error('non-finite mass-flux momentum diagnostic');
  }
}catch(e){console.error('FAIL Stage4 mass-flux momentum diagnostic');console.error(e);process.exitCode=1;}
