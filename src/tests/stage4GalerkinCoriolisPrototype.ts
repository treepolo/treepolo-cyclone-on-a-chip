declare const process:{exitCode?:number};
import { EARTH } from '../core/constants.js';
import { cross3,dot3,normalize3,type Vec3 } from '../core/math.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { buildRotationGeometry,type RotationGeometry } from '../physics/rotation.js';
import { diagnoseAxialAngularMomentum,diagnoseAxialAngularMomentumTendency } from '../solver/stage4CirculationDiagnostics.js';
import { computeStage4FrozenRhs } from '../solver/stage4Rk3SplitCpu.js';
import { cell3DIndex,createHydrostaticState,edge3DIndex,type DryState } from '../solver/state.js';

function basis(r:Vec3):{east:Vec3;north:Vec3}{const q=Math.hypot(r[0],r[1]),e:Vec3=q>1e-14?[-r[1]/q,r[0]/q,0]:[0,1,0];return{east:e,north:normalize3(cross3(r,e))};}
function analyticWind(r:Vec3,k:number,nz:number):Vec3{const b=basis(r),z=(k+.5)/nz,ue=(13+4*r[0]-2.5*r[1]+1.8*r[2])*(1+.22*z),vn=(5.5*r[0]+4.2*r[1]-3.1*r[2]+1.5*r[0]*r[1])*(1-.12*z);return[ue*b.east[0]+vn*b.north[0],ue*b.east[1]+vn*b.north[1],ue*b.east[2]+vn*b.north[2]];}
function seed(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,s:DryState,varyDensity:boolean){for(let e=0;e<h.edgeCount;e++){const ge=h.edges[e]!;for(let k=0;k<v.nz;k++)s.uEdge[edge3DIndex(e,k,v.nz)]=dot3(analyticWind(ge.midpoint,k,v.nz),ge.normal);}if(varyDensity)for(let c=0;c<h.cellCount;c++){const x=h.cellCenters[c*3]!,y=h.cellCenters[c*3+1]!,z=h.cellCenters[c*3+2]!,fac=1+.22*(.5*x-.3*y+.25*z+.18*x*y);for(let k=0;k<v.nz;k++){const q=cell3DIndex(c,k,v.nz);s.rhoD[q]*=fac;s.rhoThetaM[q]*=fac;}}}

function dot(a:Float64Array,b:Float64Array){let s=0;for(let i=0;i<a.length;i++)s+=a[i]!*b[i]!;return s;}
function axpy(y:Float64Array,a:number,x:Float64Array){for(let i=0;i<y.length;i++)y[i]=y[i]!+a*x[i]!;}

function applyR(h:ReturnType<typeof buildCubedSphere>,g:RotationGeometry,x:Float64Array):Float64Array{
  const out=new Float64Array(h.cellCount*2);
  for(let c=0;c<h.cellCount;c++){let ue=0,vn=0;for(let s=0;s<4;s++){const e=h.cellEdges[c*4+s]!,re=g.reconstruction[(c*4+s)*2]!,rn=g.reconstruction[(c*4+s)*2+1]!,q=x[e]!;ue+=re*q;vn+=rn*q;}out[c*2]=ue;out[c*2+1]=vn;}
  return out;
}
function applyRtM(h:ReturnType<typeof buildCubedSphere>,g:RotationGeometry,y:Float64Array,w:Float64Array):Float64Array{
  const out=new Float64Array(h.edgeCount);
  for(let c=0;c<h.cellCount;c++){const ue=y[c*2]!,vn=y[c*2+1]!,wc=w[c]!;for(let s=0;s<4;s++){const e=h.cellEdges[c*4+s]!,re=g.reconstruction[(c*4+s)*2]!,rn=g.reconstruction[(c*4+s)*2+1]!;out[e]=out[e]!+wc*(re*ue+rn*vn);}}
  return out;
}
function applyA(h:ReturnType<typeof buildCubedSphere>,g:RotationGeometry,x:Float64Array,w:Float64Array){return applyRtM(h,g,applyR(h,g,x),w);}
function diagonal(h:ReturnType<typeof buildCubedSphere>,g:RotationGeometry,w:Float64Array){const d=new Float64Array(h.edgeCount);for(let c=0;c<h.cellCount;c++)for(let s=0;s<4;s++){const e=h.cellEdges[c*4+s]!,re=g.reconstruction[(c*4+s)*2]!,rn=g.reconstruction[(c*4+s)*2+1]!;d[e]+=w[c]!*(re*re+rn*rn);}return d;}

function pcg(h:ReturnType<typeof buildCubedSphere>,g:RotationGeometry,b:Float64Array,w:Float64Array):{x:Float64Array;iterations:number;relResidual:number}{
  const n=b.length,x=new Float64Array(n),r=b.slice(),d=diagonal(h,g,w),z=new Float64Array(n),p=new Float64Array(n);for(let i=0;i<n;i++)z[i]=r[i]!/Math.max(d[i]!,1e-30);p.set(z);let rz=dot(r,z),bNorm=Math.sqrt(Math.max(dot(b,b),1e-60)),rel=Math.sqrt(dot(r,r))/bNorm,it=0;
  for(;it<1200&&rel>2e-12;it++){
    const Ap=applyA(h,g,p,w),den=dot(p,Ap);if(!(den>1e-28*Math.max(dot(p,p),1)))break;const alpha=rz/den;axpy(x,alpha,p);axpy(r,-alpha,Ap);rel=Math.sqrt(dot(r,r))/bNorm;if(rel<=2e-12){it++;break;}for(let i=0;i<n;i++)z[i]=r[i]!/Math.max(d[i]!,1e-30);const rz2=dot(r,z),beta=rz2/rz;for(let i=0;i<n;i++)p[i]=z[i]!+beta*p[i]!;rz=rz2;
  }
  return{x,iterations:it,relResidual:rel};
}

function galerkin(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,s:DryState,g:RotationGeometry,massWeighted:boolean):{a:Float64Array;maxIter:number;maxResidual:number;work:number}{
  const out=new Float64Array(s.uEdge.length);let maxIter=0,maxResidual=0,work=0,energy=0;
  for(let k=0;k<v.nz;k++){
    const u=new Float64Array(h.edgeCount);for(let e=0;e<h.edgeCount;e++)u[e]=s.uEdge[edge3DIndex(e,k,v.nz)]!;const uv=applyR(h,g,u),w=new Float64Array(h.cellCount),j=new Float64Array(h.cellCount*2);
    for(let c=0;c<h.cellCount;c++){const o=c*3,q=cell3DIndex(c,k,v.nz),mc=h.cellAreaUnit[c]!*v.dz[k]!*(massWeighted?s.rhoD[q]!:1),f=2*EARTH.omega*g.radial[o+2]!,ue=uv[c*2]!,vn=uv[c*2+1]!;w[c]=mc;j[c*2]=f*vn;j[c*2+1]=-f*ue;work+=mc*(ue*j[c*2]!+vn*j[c*2+1]!);energy+=mc*(ue*ue+vn*vn);}
    const rhs=applyRtM(h,g,j,w),sol=pcg(h,g,rhs,w);maxIter=Math.max(maxIter,sol.iterations);maxResidual=Math.max(maxResidual,sol.relResidual);for(let e=0;e<h.edgeCount;e++)out[edge3DIndex(e,k,v.nz)]=sol.x[e]!;
  }
  return{a:out,maxIter,maxResidual,work:work/Math.max(energy,1e-30)*86400};
}

function reconstructedWork(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,s:DryState,g:RotationGeometry,a:Float64Array):number{
  let work=0,mass=0;for(let k=0;k<v.nz;k++){const u=new Float64Array(h.edgeCount),aa=new Float64Array(h.edgeCount);for(let e=0;e<h.edgeCount;e++){u[e]=s.uEdge[edge3DIndex(e,k,v.nz)]!;aa[e]=a[edge3DIndex(e,k,v.nz)]!;}const ru=applyR(h,g,u),ra=applyR(h,g,aa);for(let c=0;c<h.cellCount;c++){const q=cell3DIndex(c,k,v.nz),m=h.cellAreaUnit[c]!*EARTH.radius*EARTH.radius*v.dz[k]!*s.rhoD[q]!;work+=m*(ru[c*2]!*ra[c*2]!+ru[c*2+1]!*ra[c*2+1]!);mass+=m;}}
  return work/Math.max(mass,1e-30)*86400;
}

function run(n:number,varyDensity:boolean){const h=buildCubedSphere(n),v=buildStretchedVerticalGrid(6,16000,1.15),ref=buildHeldSuarezReference(v),g=buildRotationGeometry(h),s=createHydrostaticState(h,v,ref);seed(h,v,s,varyDensity);const cont=computeStage4FrozenRhs(h,v,ref,s,{momentumTransport:false,coriolis:false,heldSuarez:false},g),zeroU=new Float64Array(s.uEdge.length),zeroR=new Float64Array(s.rhoD.length),mass=diagnoseAxialAngularMomentumTendency(h,v,s,cont.rhoD,zeroU,g).planetaryMassRedistributionTorque,lever=diagnoseAxialAngularMomentum(h,v,s,g).torqueLeverMass,scale=86400/lever,torque=(a:Float64Array)=>diagnoseAxialAngularMomentumTendency(h,v,s,zeroR,a,g).velocityTorque;const geom=galerkin(h,v,s,g,false),rho=galerkin(h,v,s,g,true);return{n,geom:(mass+torque(geom.a))*scale,rho:(mass+torque(rho.a))*scale,geomWork:reconstructedWork(h,v,s,g,geom.a),rhoWork:reconstructedWork(h,v,s,g,rho.a),geomIdentityWork:geom.work,rhoIdentityWork:rho.work,geomIter:geom.maxIter,rhoIter:rho.maxIter,geomRes:geom.maxResidual,rhoRes:rho.maxResidual};}

try{for(const varying of [false,true]){console.log(`Stage4 full-mass Galerkin Coriolis; density ${varying?'VARIES':'uniform'}`);console.log('N\tgeom P+C\trho P+C\tgeom cellWork\trho cellWork\tgeom identityW\trho identityW\titer(g/r)\tres(g/r)');for(const n of [4,8,16,32]){const r=run(n,varying);console.log(`${n}\t${r.geom.toExponential(8)}\t${r.rho.toExponential(8)}\t${r.geomWork.toExponential(8)}\t${r.rhoWork.toExponential(8)}\t${r.geomIdentityWork.toExponential(3)}\t${r.rhoIdentityWork.toExponential(3)}\t${r.geomIter}/${r.rhoIter}\t${r.geomRes.toExponential(2)}/${r.rhoRes.toExponential(2)}`);if(!Object.values(r).every(x=>typeof x!=='number'||Number.isFinite(x)))throw new Error(`non-finite N${n}`);if(r.geomRes>1e-8||r.rhoRes>1e-8)throw new Error(`PCG did not converge N${n}: ${r.geomRes}, ${r.rhoRes}`);}}
}catch(e){console.error('FAIL Stage4 full-mass Galerkin Coriolis prototype');console.error(e);process.exitCode=1;}
