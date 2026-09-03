import { EARTH } from '../core/constants.js';
import { Vec3, dot3, scale3, sub3 } from '../core/math.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { applyAcousticDivergenceDamping } from '../physics/acousticDivergenceDamping.js';
import { applyHeldSuarezForcing } from '../physics/heldSuarez.js';
import { applyModelTopSponge } from '../physics/modelTopSponge.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { RotationGeometry, addCellWindDeltaToEdges, applyTraditionalCoriolis, buildRotationGeometry, reconstructCellHorizontalWind } from '../physics/rotation.js';
import { DryCoreCpu, StepDiagnostics, TransportSnapshot } from './dryCoreCpu.js';
import { DryState, cell3DIndex, edge3DIndex, w3DIndex } from './state.js';

export interface RotatingStepOptions{heldSuarez?:boolean;momentumTransport?:boolean;coriolis?:boolean;topSponge?:boolean;divergenceDamping?:boolean}
export class RotatingDryCoreCpu{
  readonly dry:DryCoreCpu;readonly rotation:RotationGeometry;
  constructor(public readonly h:CubedSphereGrid,public readonly v:VerticalGrid,public readonly ref:ReferenceAtmosphere){this.dry=new DryCoreCpu(h,v,ref);this.dry.captureTransport=true;this.rotation=buildRotationGeometry(h)}
  step(s:DryState,dt:number,opt:RotatingStepOptions={}):StepDiagnostics{
    const cor=opt.coriolis!==false,mom=opt.momentumTransport!==false,hs=opt.heldSuarez!==false,sponge=opt.topSponge!==false,divDamp=opt.divergenceDamping!==false;
    if(cor)applyTraditionalCoriolis(this.h,this.rotation,s,.5*dt);
    const d=this.dry.step(s,dt);
    if(mom){if(!this.dry.lastTransport)throw new Error('missing transport snapshot');this.advectMomentum(s,dt,this.dry.lastTransport)}
    if(hs)applyHeldSuarezForcing(this.h,this.v,s,dt);
    if(cor)applyTraditionalCoriolis(this.h,this.rotation,s,.5*dt);
    if(divDamp)applyAcousticDivergenceDamping(this.h,this.v,this.ref,s);
    if(sponge)applyModelTopSponge(this.v,s,dt);
    return d;
  }
  private advectMomentum(s:DryState,dt:number,t:TransportSnapshot):void{
    const{h,v}=this,nz=v.nz,R=EARTH.radius,windByK:Float64Array[]=Array.from({length:nz},(_,k)=>reconstructCellHorizontalWind(h,this.rotation,s,k)),hmx=new Float64Array(s.uEdge.length),hmy=new Float64Array(s.uEdge.length),hmz=new Float64Array(s.uEdge.length);
    for(let e=0;e<h.edgeCount;e++)for(let k=0;k<nz;k++){const q=edge3DIndex(e,k,nz),m=t.hMassFlux[q]!,src=m>=0?h.edges[e]!.leftCell:h.edges[e]!.rightCell,w=windByK[k]!;hmx[q]=m*w[src*3]!;hmy[q]=m*w[src*3+1]!;hmz[q]=m*w[src*3+2]!}
    const vmx=new Float64Array(s.wInterface.length),vmy=new Float64Array(s.wInterface.length),vmz=new Float64Array(s.wInterface.length);for(let c=0;c<h.cellCount;c++)for(let i=1;i<nz;i++){const q=w3DIndex(c,i,nz),m=t.vMassFlux[q]!,src=m>=0?i-1:i,w=windByK[src]!;vmx[q]=m*w[c*3]!;vmy[q]=m*w[c*3+1]!;vmz[q]=m*w[c*3+2]!}
    const deltas:Float64Array[]=Array.from({length:nz},()=>new Float64Array(h.cellCount*3));for(let c=0;c<h.cellCount;c++)for(let k=0;k<nz;k++){const q=cell3DIndex(c,k,nz),vol=h.cellAreaUnit[c]!*R*R*v.dz[k]!;let massT=0,mxT=0,myT=0,mzT=0;for(let slot=0;slot<4;slot++){const eid=h.cellEdges[c*4+slot]!,sgn=h.cellEdgeSigns[c*4+slot]!,qe=edge3DIndex(eid,k,nz);massT-=sgn*t.hMassFlux[qe]!;mxT-=sgn*hmx[qe]!;myT-=sgn*hmy[qe]!;mzT-=sgn*hmz[qe]!}const qb=w3DIndex(c,k,nz),qt=w3DIndex(c,k+1,nz);massT+=t.vMassFlux[qb]!-t.vMassFlux[qt]!;mxT+=vmx[qb]!-vmx[qt]!;myT+=vmy[qb]!-vmy[qt]!;mzT+=vmz[qb]!-vmz[qt]!;const rhoNew=s.rhoD[q]!,rhoOld=rhoNew-dt*massT/vol,wold=windByK[k]!,old:Vec3=[wold[c*3]!,wold[c*3+1]!,wold[c*3+2]!],mom:Vec3=[rhoOld*old[0]+dt*mxT/vol,rhoOld*old[1]+dt*myT/vol,rhoOld*old[2]+dt*mzT/vol],radial:Vec3=[h.cellCenters[c*3]!,h.cellCenters[c*3+1]!,h.cellCenters[c*3+2]!],raw:Vec3=[mom[0]/rhoNew,mom[1]/rhoNew,mom[2]/rhoNew],nw=sub3(raw,scale3(radial,dot3(raw,radial))),delta:Vec3=[nw[0]-old[0],nw[1]-old[1],nw[2]-old[2]];deltas[k]!.set(delta,c*3)}for(let k=0;k<nz;k++)addCellWindDeltaToEdges(h,s,k,deltas[k]!);
  }
}
