import { AtmosphereConfig, DRY_AIR, EARTH, PlanetConfig } from '../core/constants.js';
import { VerticalGrid } from '../grid/vertical.js';
import { thetaFromTP } from './thermodynamics.js';

export interface ReferenceAtmosphere {
  T0:number;
  pCenter:Float64Array; rhoCenter:Float64Array; thetaCenter:Float64Array; rhoThetaCenter:Float64Array;
  pInterface:Float64Array; rhoInterface:Float64Array; thetaInterface:Float64Array; rhoThetaInterface:Float64Array;
}
export function buildIsothermalReference(v:VerticalGrid,T0=288,planet:PlanetConfig=EARTH,a:AtmosphereConfig=DRY_AIR):ReferenceAtmosphere {
  const pc=new Float64Array(v.nz), rc=new Float64Array(v.nz), tc=new Float64Array(v.nz), rtc=new Float64Array(v.nz);
  const pi=new Float64Array(v.nz+1), ri=new Float64Array(v.nz+1), ti=new Float64Array(v.nz+1), rti=new Float64Array(v.nz+1);
  const H=a.rd*T0/planet.gravity;
  for(let k=0;k<v.nz;k++){ const p=a.pRef*Math.exp(-v.zCenter[k]!/H), rho=p/(a.rd*T0), th=thetaFromTP(T0,p,a); pc[k]=p;rc[k]=rho;tc[k]=th;rtc[k]=rho*th; }
  for(let k=0;k<=v.nz;k++){ const p=a.pRef*Math.exp(-v.zInterface[k]!/H), rho=p/(a.rd*T0), th=thetaFromTP(T0,p,a); pi[k]=p;ri[k]=rho;ti[k]=th;rti[k]=rho*th; }
  return {T0,pCenter:pc,rhoCenter:rc,thetaCenter:tc,rhoThetaCenter:rtc,pInterface:pi,rhoInterface:ri,thetaInterface:ti,rhoThetaInterface:rti};
}
