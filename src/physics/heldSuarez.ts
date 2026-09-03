import { DRY_AIR, EARTH } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from './referenceAtmosphere.js';
import { pressureFromRhoTheta, thetaFromTP } from './thermodynamics.js';
import { DryState, cell3DIndex, edge3DIndex } from '../solver/state.js';

const DAY=86400;
export const HELD_SUAREZ={T0:315,deltaTy:60,deltaThetaZ:10,Tmin:200,sigmaB:.7,ka:1/(40*DAY),ks:1/(4*DAY),kf:1/DAY};
export function heldSuarezTeq(lat:number,p:number):number{
  const sigma=Math.max(1e-6,p/DRY_AIR.pRef),sn=Math.sin(lat),cs=Math.cos(lat);
  return Math.max(HELD_SUAREZ.Tmin,(HELD_SUAREZ.T0-HELD_SUAREZ.deltaTy*sn*sn-HELD_SUAREZ.deltaThetaZ*Math.log(sigma)*cs*cs)*Math.pow(sigma,DRY_AIR.kappa));
}
export function heldSuarezThermalRate(lat:number,sigma:number):number{
  const sfc=Math.max(0,(sigma-HELD_SUAREZ.sigmaB)/(1-HELD_SUAREZ.sigmaB));
  return HELD_SUAREZ.ka+(HELD_SUAREZ.ks-HELD_SUAREZ.ka)*sfc*Math.pow(Math.cos(lat),4);
}
export function heldSuarezDragRate(sigma:number):number{
  return HELD_SUAREZ.kf*Math.max(0,(sigma-HELD_SUAREZ.sigmaB)/(1-HELD_SUAREZ.sigmaB));
}
export function applyHeldSuarezForcing(h:CubedSphereGrid,v:VerticalGrid,s:DryState,dt:number):void{
  for(let c=0;c<h.cellCount;c++){const lat=Math.asin(h.cellCenters[c*3+2]!);for(let k=0;k<v.nz;k++){const q=cell3DIndex(c,k,v.nz),rho=s.rhoD[q]!,x=s.rhoThetaM[q]!,p=pressureFromRhoTheta(x),theta=x/rho,sigma=p/DRY_AIR.pRef,thetaEq=thetaFromTP(heldSuarezTeq(lat,p),p),rate=heldSuarezThermalRate(lat,sigma),thetaNew=thetaEq+(theta-thetaEq)*Math.exp(-rate*dt);s.rhoThetaM[q]=rho*thetaNew;}}
  for(let e=0;e<h.edgeCount;e++){const ge=h.edges[e]!;for(let k=0;k<v.nz;k++){const l=cell3DIndex(ge.leftCell,k,v.nz),r=cell3DIndex(ge.rightCell,k,v.nz),sigma=.5*(pressureFromRhoTheta(s.rhoThetaM[l]!)+pressureFromRhoTheta(s.rhoThetaM[r]!))/DRY_AIR.pRef,rate=heldSuarezDragRate(sigma);if(rate>0){const q=edge3DIndex(e,k,v.nz);s.uEdge[q]=s.uEdge[q]!*Math.exp(-rate*dt);}}}
}
function meanTeq(p:number):number{const sigma=Math.max(1e-6,p/DRY_AIR.pRef);return Math.max(HELD_SUAREZ.Tmin,(HELD_SUAREZ.T0-HELD_SUAREZ.deltaTy/3-HELD_SUAREZ.deltaThetaZ*Math.log(sigma)*2/3)*Math.pow(sigma,DRY_AIR.kappa))}
function dpdz(p:number):number{return-EARTH.gravity*p/(DRY_AIR.rd*meanTeq(p))}
function integrateP(p0:number,dz:number):number{const n=Math.max(1,Math.ceil(Math.abs(dz)/100)),h=dz/n;let p=p0;for(let i=0;i<n;i++){const k1=dpdz(p),k2=dpdz(p+.5*h*k1),k3=dpdz(p+.5*h*k2),k4=dpdz(p+h*k3);p+=h*(k1+2*k2+2*k3+k4)/6}return p}
export function buildHeldSuarezReference(v:VerticalGrid):ReferenceAtmosphere{
  const pc=new Float64Array(v.nz),rc=new Float64Array(v.nz),tc=new Float64Array(v.nz),xc=new Float64Array(v.nz),pi=new Float64Array(v.nz+1),ri=new Float64Array(v.nz+1),ti=new Float64Array(v.nz+1),xi=new Float64Array(v.nz+1);pi[0]=DRY_AIR.pRef;
  for(let i=1;i<=v.nz;i++)pi[i]=integrateP(pi[i-1]!,v.zInterface[i]!-v.zInterface[i-1]!);for(let k=0;k<v.nz;k++)pc[k]=integrateP(pi[k]!,v.zCenter[k]!-v.zInterface[k]!);
  for(let k=0;k<v.nz;k++){const p=pc[k]!,T=meanTeq(p),rho=p/(DRY_AIR.rd*T),th=thetaFromTP(T,p);rc[k]=rho;tc[k]=th;xc[k]=rho*th}for(let i=0;i<=v.nz;i++){const p=pi[i]!,T=meanTeq(p),rho=p/(DRY_AIR.rd*T),th=thetaFromTP(T,p);ri[i]=rho;ti[i]=th;xi[i]=rho*th}
  return{T0:meanTeq(DRY_AIR.pRef),pCenter:pc,rhoCenter:rc,thetaCenter:tc,rhoThetaCenter:xc,pInterface:pi,rhoInterface:ri,thetaInterface:ti,rhoThetaInterface:xi};
}
export function addHeldSuarezWavePerturbation(h:CubedSphereGrid,v:VerticalGrid,ref:ReferenceAtmosphere,s:DryState,ampK=.1):void{
  for(let c=0;c<h.cellCount;c++){const x=h.cellCenters[c*3]!,y=h.cellCenters[c*3+1]!,z=h.cellCenters[c*3+2]!,lon=Math.atan2(y,x),lat=Math.asin(z);for(let k=0;k<v.nz;k++){const shape=Math.cos(lat)**2*Math.cos(4*lon)*Math.sin(Math.PI*v.zCenter[k]!/v.top),q=cell3DIndex(c,k,v.nz),theta=ref.thetaCenter[k]!+ampK*shape;s.rhoThetaM[q]=ref.rhoThetaCenter[k]!;s.rhoD[q]=s.rhoThetaM[q]!/theta;}}
}
