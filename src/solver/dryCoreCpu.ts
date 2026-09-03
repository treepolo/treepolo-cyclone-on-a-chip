import { DRY_AIR, EARTH } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { pressureFromRhoTheta } from '../physics/thermodynamics.js';
import { DryState, cell3DIndex, edge3DIndex, w3DIndex } from './state.js';
import { heviColumnStep } from './hevi.js';

export interface StepDiagnostics { maxHorizontalCfl:number; maxVerticalCfl:number; }
export interface TransportSnapshot { hMassFlux:Float64Array; vMassFlux:Float64Array; }
export class DryCoreCpu {
  captureTransport=false;lastTransport?:TransportSnapshot;
  readonly heviRayleighRates?:Float64Array;
  constructor(
    public readonly h:CubedSphereGrid,
    public readonly v:VerticalGrid,
    public readonly ref:ReferenceAtmosphere,
    public readonly heviOffCentering=0,
    heviRayleighRates?:ArrayLike<number>,
  ){
    if(heviRayleighRates){
      if(heviRayleighRates.length!==v.nz+1)throw new Error('HEVI Rayleigh profile shape mismatch');
      this.heviRayleighRates=Float64Array.from(heviRayleighRates);
    }
  }
  step(s:DryState,dt:number,heviRayleighEnabled=true):StepDiagnostics{
    if(!(dt>0))throw new Error('dt must be positive');const{h,v,ref}=this,nz=v.nz,R=EARTH.radius,p=new Float64Array(h.cellCount*nz);for(let q=0;q<p.length;q++)p[q]=pressureFromRhoTheta(s.rhoThetaM[q]!);
    let maxHCfl=0;for(let e=0;e<h.edgeCount;e++){const ge=h.edges[e]!,dist=ge.centerDistanceAngle*R;for(let k=0;k<nz;k++){const l=cell3DIndex(ge.leftCell,k,nz),r=cell3DIndex(ge.rightCell,k,nz),q=edge3DIndex(e,k,nz),rho=.5*(s.rhoD[l]!+s.rhoD[r]!);s.uEdge[q]=s.uEdge[q]!-dt*(p[r]!-p[l]!)/(Math.max(rho,1e-12)*dist);maxHCfl=Math.max(maxHCfl,Math.abs(s.uEdge[q]!)*dt/dist);}}
    const cr=new Float64Array(nz),cx=new Float64Array(nz),cw=new Float64Array(nz+1),ca=new Float64Array(nz+1);
    for(let c=0;c<h.cellCount;c++){
      for(let k=0;k<nz;k++){cr[k]=s.rhoD[cell3DIndex(c,k,nz)]!;cx[k]=s.rhoThetaM[cell3DIndex(c,k,nz)]!}
      for(let k=0;k<=nz;k++){cw[k]=s.wInterface[w3DIndex(c,k,nz)]!;ca[k]=0}
      for(let i=1;i<nz;i++){
        const rho=.5*(cr[i-1]!+cr[i]!),rho0=.5*(ref.rhoCenter[i-1]!+ref.rhoCenter[i]!);
        ca[i]=-EARTH.gravity*(rho-rho0)/Math.max(rho,1e-12);
      }
      heviColumnStep(v,ref,{rho:cr,rhoTheta:cx,w:cw},dt,this.heviOffCentering,heviRayleighEnabled?this.heviRayleighRates:undefined,ca);
      for(let k=0;k<nz;k++){s.rhoD[cell3DIndex(c,k,nz)]=cr[k]!;s.rhoThetaM[cell3DIndex(c,k,nz)]=cx[k]!}
      for(let k=0;k<=nz;k++)s.wInterface[w3DIndex(c,k,nz)]=cw[k]!;
    }
    const dR=new Float64Array(s.rhoD.length),dX=new Float64Array(s.rhoThetaM.length),hMass=this.captureTransport?new Float64Array(s.uEdge.length):undefined,vMass=this.captureTransport?new Float64Array(s.wInterface.length):undefined;
    for(let e=0;e<h.edgeCount;e++){const ge=h.edges[e]!,edgeLen=ge.angularLength*R;for(let k=0;k<nz;k++){const ql=cell3DIndex(ge.leftCell,k,nz),qr=cell3DIndex(ge.rightCell,k,nz),qe=edge3DIndex(e,k,nz),u=s.uEdge[qe]!,up=u>=0?ql:qr,A=edgeLen*v.dz[k]!,fr=s.rhoD[up]!*u*A,fx=s.rhoThetaM[up]!*u*A;if(hMass)hMass[qe]=fr;dR[ql]=dR[ql]!-fr;dR[qr]=dR[qr]!+fr;dX[ql]=dX[ql]!-fx;dX[qr]=dX[qr]!+fx;}}
    let maxVCfl=0;for(let c=0;c<h.cellCount;c++)for(let i=1;i<nz;i++){const wi=w3DIndex(c,i,nz),w=s.wInterface[wi]!,ql=cell3DIndex(c,i-1,nz),qu=cell3DIndex(c,i,nz),up=w>=0?ql:qu,A=h.cellAreaUnit[c]!*R*R,sk=w>=0?i-1:i,fr=(s.rhoD[up]!-ref.rhoCenter[sk]!)*w*A,fx=(s.rhoThetaM[up]!-ref.rhoThetaCenter[sk]!)*w*A;if(vMass)vMass[wi]=fr;dR[ql]=dR[ql]!-fr;dR[qu]=dR[qu]!+fr;dX[ql]=dX[ql]!-fx;dX[qu]=dX[qu]!+fx;maxVCfl=Math.max(maxVCfl,Math.abs(w)*dt/Math.min(v.dz[i-1]!,v.dz[i]!));}
    for(let c=0;c<h.cellCount;c++)for(let k=0;k<nz;k++){const q=cell3DIndex(c,k,nz),vol=h.cellAreaUnit[c]!*R*R*v.dz[k]!;s.rhoD[q]=s.rhoD[q]!+dt*dR[q]!/vol;s.rhoThetaM[q]=s.rhoThetaM[q]!+dt*dX[q]!/vol;if(!(s.rhoD[q]!>0)||!(s.rhoThetaM[q]!>0)||!Number.isFinite(s.rhoD[q]!)||!Number.isFinite(s.rhoThetaM[q]!))throw new Error(`invalid state at cell=${c},k=${k}`)}
    if(hMass&&vMass)this.lastTransport={hMassFlux:hMass,vMassFlux:vMass};s.time+=dt;return{maxHorizontalCfl:maxHCfl,maxVerticalCfl:maxVCfl};
  }
}
