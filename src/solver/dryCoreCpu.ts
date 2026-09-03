import { DRY_AIR, EARTH } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { pressureFromRhoTheta } from '../physics/thermodynamics.js';
import { DryState, cell3DIndex, edge3DIndex, w3DIndex } from './state.js';
import { heviColumnStep } from './hevi.js';

export interface StepDiagnostics { maxHorizontalCfl:number; maxVerticalCfl:number; }

export class DryCoreCpu {
  constructor(public readonly h:CubedSphereGrid, public readonly v:VerticalGrid, public readonly ref:ReferenceAtmosphere) {}

  step(s:DryState,dt:number):StepDiagnostics {
    if(!(dt>0)) throw new Error('dt must be positive');
    const {h,v,ref}=this; const nz=v.nz, R=EARTH.radius;
    const p=new Float64Array(h.cellCount*nz);
    for(let q=0;q<p.length;q++) p[q]=pressureFromRhoTheta(s.rhoThetaM[q]!);

    // Horizontal pressure-gradient acceleration on canonical shared edges.
    let maxHCfl=0;
    for(let e=0;e<h.edgeCount;e++){
      const ge=h.edges[e]!, dist=ge.centerDistanceAngle*R;
      for(let k=0;k<nz;k++){
        const l=cell3DIndex(ge.leftCell,k,nz), r=cell3DIndex(ge.rightCell,k,nz), q=edge3DIndex(e,k,nz);
        const rho=0.5*(s.rhoD[l]!+s.rhoD[r]!); s.uEdge[q] = s.uEdge[q]! - dt*(p[r]!-p[l]!)/(Math.max(rho,1e-12)*dist);
        maxHCfl=Math.max(maxHCfl,Math.abs(s.uEdge[q]!)*dt/dist);
      }
    }

    // Per-column HEVI vertical acoustic solve.
    const cr=new Float64Array(nz), cx=new Float64Array(nz), cw=new Float64Array(nz+1);
    for(let c=0;c<h.cellCount;c++){
      for(let k=0;k<nz;k++){cr[k]=s.rhoD[cell3DIndex(c,k,nz)]!;cx[k]=s.rhoThetaM[cell3DIndex(c,k,nz)]!;}
      for(let k=0;k<=nz;k++)cw[k]=s.wInterface[w3DIndex(c,k,nz)]!;
      heviColumnStep(v,ref,{rho:cr,rhoTheta:cx,w:cw},dt);
      for(let k=0;k<nz;k++){s.rhoD[cell3DIndex(c,k,nz)]=cr[k]!;s.rhoThetaM[cell3DIndex(c,k,nz)]=cx[k]!;}
      for(let k=0;k<=nz;k++)s.wInterface[w3DIndex(c,k,nz)]=cw[k]!;
    }

    // Slow buoyancy term is intentionally outside the implicit acoustic solve.
    for(let c=0;c<h.cellCount;c++) for(let ki=1;ki<nz;ki++){
      const ql=cell3DIndex(c,ki-1,nz), qu=cell3DIndex(c,ki,nz), wi=w3DIndex(c,ki,nz);
      const rho=0.5*(s.rhoD[ql]!+s.rhoD[qu]!); const rho0=0.5*(ref.rhoCenter[ki-1]!+ref.rhoCenter[ki]!);
      s.wInterface[wi] = s.wInterface[wi]! - dt*EARTH.gravity*(rho-rho0)/Math.max(rho,1e-12);
    }

    // Conservative outer transport. Horizontal: full fields. Vertical: perturbation flux only;
    // base-state vertical flux was already advanced by HEVI.
    const dR=new Float64Array(s.rhoD.length), dX=new Float64Array(s.rhoThetaM.length);
    for(let e=0;e<h.edgeCount;e++){
      const ge=h.edges[e]!, edgeLen=ge.angularLength*R;
      for(let k=0;k<nz;k++){
        const ql=cell3DIndex(ge.leftCell,k,nz), qr=cell3DIndex(ge.rightCell,k,nz), qe=edge3DIndex(e,k,nz), u=s.uEdge[qe]!;
        const up=u>=0?ql:qr, sideArea=edgeLen*v.dz[k]!, fr=s.rhoD[up]!*u*sideArea, fx=s.rhoThetaM[up]!*u*sideArea;
        dR[ql]=dR[ql]!-fr; dR[qr]=dR[qr]!+fr; dX[ql]=dX[ql]!-fx; dX[qr]=dX[qr]!+fx;
      }
    }
    let maxVCfl=0;
    for(let c=0;c<h.cellCount;c++) for(let ki=1;ki<nz;ki++){
      const w=s.wInterface[w3DIndex(c,ki,nz)]!, ql=cell3DIndex(c,ki-1,nz), qu=cell3DIndex(c,ki,nz), up=w>=0?ql:qu;
      const area=h.cellAreaUnit[c]!*R*R;
      const rp=s.rhoD[up]!-ref.rhoCenter[w>=0?ki-1:ki]!;
      const xp=s.rhoThetaM[up]!-ref.rhoThetaCenter[w>=0?ki-1:ki]!;
      const fr=rp*w*area, fx=xp*w*area; dR[ql]=dR[ql]!-fr;dR[qu]=dR[qu]!+fr;dX[ql]=dX[ql]!-fx;dX[qu]=dX[qu]!+fx;
      const dzMin=Math.min(v.dz[ki-1]!,v.dz[ki]!); maxVCfl=Math.max(maxVCfl,Math.abs(w)*dt/dzMin);
    }
    for(let c=0;c<h.cellCount;c++) for(let k=0;k<nz;k++){
      const q=cell3DIndex(c,k,nz), vol=h.cellAreaUnit[c]!*R*R*v.dz[k]!;
      s.rhoD[q] = s.rhoD[q]! + dt*dR[q]!/vol; s.rhoThetaM[q] = s.rhoThetaM[q]! + dt*dX[q]!/vol;
      if(!(s.rhoD[q]!>0)||!(s.rhoThetaM[q]!>0)||!Number.isFinite(s.rhoD[q]!)||!Number.isFinite(s.rhoThetaM[q]!)) throw new Error(`invalid state at cell=${c},k=${k}`);
    }
    s.time+=dt; return {maxHorizontalCfl:maxHCfl,maxVerticalCfl:maxVCfl};
  }
}
