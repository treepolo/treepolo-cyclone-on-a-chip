import { angle3 } from '../core/math.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { DryState, cell3DIndex } from './state.js';

export interface ThermalBubbleOptions { lonDeg:number; latDeg:number; altitude:number; horizontalRadius:number; verticalRadius:number; deltaTheta:number; }
function lonLatUnit(lonDeg:number,latDeg:number):readonly[number,number,number]{ const lo=lonDeg*Math.PI/180,la=latDeg*Math.PI/180,c=Math.cos(la); return [c*Math.cos(lo),c*Math.sin(lo),Math.sin(la)]; }

/** Constant-pressure warm bubble: rhoTheta stays on the reference pressure surface and density is reduced. */
export function addThermalBubble(h:CubedSphereGrid,v:VerticalGrid,ref:ReferenceAtmosphere,s:DryState,opt:ThermalBubbleOptions):void {
  const target=lonLatUnit(opt.lonDeg,opt.latDeg);
  for(let c=0;c<h.cellCount;c++){
    const center:[number,number,number]=[h.cellCenters[c*3]!,h.cellCenters[c*3+1]!,h.cellCenters[c*3+2]!];
    const horizontal=angle3(center,target)*6.371e6;
    if(horizontal>opt.horizontalRadius) continue;
    for(let k=0;k<v.nz;k++){
      const dz=v.zCenter[k]!-opt.altitude; if(Math.abs(dz)>opt.verticalRadius) continue;
      const rr=Math.sqrt((horizontal/opt.horizontalRadius)**2+(dz/opt.verticalRadius)**2); if(rr>=1)continue;
      const shape=0.5*(1+Math.cos(Math.PI*rr)); const dtheta=opt.deltaTheta*shape;
      const q=cell3DIndex(c,k,v.nz); const theta=ref.thetaCenter[k]!+dtheta;
      s.rhoThetaM[q]=ref.rhoThetaCenter[k]!; s.rhoD[q]=s.rhoThetaM[q]!/theta;
    }
  }
}
