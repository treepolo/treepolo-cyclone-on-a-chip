import { EARTH, DRY_AIR } from '../core/constants.js';
import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { pressureFromRhoTheta } from '../physics/thermodynamics.js';
import { DryState, cell3DIndex } from './state.js';

export interface CoreDiagnostics { dryMass:number; minRho:number; minP:number; maxAbsW:number; nan:boolean; }
export function diagnoseState(h:CubedSphereGrid,v:VerticalGrid,s:DryState):CoreDiagnostics {
  let mass=0,minRho=Infinity,minP=Infinity,maxW=0,nan=false;
  const r2=EARTH.radius*EARTH.radius;
  for(let c=0;c<h.cellCount;c++) for(let k=0;k<v.nz;k++){
    const q=cell3DIndex(c,k,v.nz),rho=s.rhoD[q]!, rt=s.rhoThetaM[q]!, p=pressureFromRhoTheta(rt,DRY_AIR);
    mass += rho*h.cellAreaUnit[c]!*r2*v.dz[k]!; minRho=Math.min(minRho,rho);minP=Math.min(minP,p);if(!Number.isFinite(rho)||!Number.isFinite(p))nan=true;
  }
  for(const w of s.wInterface){maxW=Math.max(maxW,Math.abs(w));if(!Number.isFinite(w))nan=true;}
  return {dryMass:mass,minRho,minP,maxAbsW:maxW,nan};
}
