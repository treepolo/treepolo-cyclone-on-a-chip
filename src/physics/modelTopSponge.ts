import { VerticalGrid } from '../grid/vertical.js';
import { DryState, w3DIndex } from '../solver/state.js';

export interface ModelTopSpongeConfig {
  startFraction:number;
  maxRate:number;
}

export const MODEL_TOP_SPONGE:ModelTopSpongeConfig={
  startFraction:0.75,
  maxRate:1/600,
};

export function modelTopSpongeRate(z:number,top:number,config:ModelTopSpongeConfig=MODEL_TOP_SPONGE):number{
  const start=config.startFraction*top;
  if(z<=start)return 0;
  const s=Math.min(1,Math.max(0,(z-start)/(top-start)));
  const ramp=Math.sin(0.5*Math.PI*s);
  return config.maxRate*ramp*ramp;
}

export function buildModelTopSpongeRates(v:VerticalGrid,config:ModelTopSpongeConfig=MODEL_TOP_SPONGE):Float32Array{
  const out=new Float32Array(v.nz+1);
  for(let i=0;i<=v.nz;i++)out[i]=modelTopSpongeRate(v.zInterface[i]!,v.top,config);
  return out;
}

/**
 * Absorbing-layer treatment for the artificial rigid model top.
 * Only vertical velocity is damped and only in the uppermost part of the domain.
 * Mass and thermodynamic state are untouched, so this does not repair conservation
 * by normalization or clamp an invalid state.
 */
export function applyModelTopSponge(v:VerticalGrid,s:DryState,dt:number,config:ModelTopSpongeConfig=MODEL_TOP_SPONGE):void{
  if(!(dt>0))throw new Error('sponge dt must be positive');
  for(let c=0;c<s.wInterface.length/(v.nz+1);c++)for(let i=1;i<v.nz;i++){
    const rate=modelTopSpongeRate(v.zInterface[i]!,v.top,config);
    if(rate<=0)continue;
    const q=w3DIndex(c,i,v.nz);
    s.wInterface[q]=s.wInterface[q]!*Math.exp(-rate*dt);
  }
}
