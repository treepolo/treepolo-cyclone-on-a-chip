import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';

export interface DryState {
  rhoD:Float64Array; rhoThetaM:Float64Array; uEdge:Float64Array; wInterface:Float64Array; time:number;
}
export function cell3DIndex(cell:number,k:number,nz:number):number { return cell*nz+k; }
export function edge3DIndex(edge:number,k:number,nz:number):number { return edge*nz+k; }
export function w3DIndex(cell:number,kInterface:number,nz:number):number { return cell*(nz+1)+kInterface; }

export function createHydrostaticState(h:CubedSphereGrid,v:VerticalGrid,ref:ReferenceAtmosphere):DryState {
  const rhoD=new Float64Array(h.cellCount*v.nz), rhoThetaM=new Float64Array(h.cellCount*v.nz);
  for(let c=0;c<h.cellCount;c++) for(let k=0;k<v.nz;k++){ const q=cell3DIndex(c,k,v.nz);rhoD[q]=ref.rhoCenter[k]!;rhoThetaM[q]=ref.rhoThetaCenter[k]!; }
  return {rhoD,rhoThetaM,uEdge:new Float64Array(h.edgeCount*v.nz),wInterface:new Float64Array(h.cellCount*(v.nz+1)),time:0};
}
export function cloneState(s:DryState):DryState { return {rhoD:s.rhoD.slice(),rhoThetaM:s.rhoThetaM.slice(),uEdge:s.uEdge.slice(),wInterface:s.wInterface.slice(),time:s.time}; }
