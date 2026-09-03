import { VerticalGrid } from '../grid/vertical.js';
import { DryState, w3DIndex } from '../solver/state.js';

export interface ModelTopSpongeConfig {
  startFraction:number;
  maxRate:number;
}

/**
 * Stage-4 upper absorbing layer.
 *
 * The production path applies this profile implicitly inside the HEVI vertical
 * acoustic solve.  The 0.2 s^-1 peak follows the Klemp et al. (2008) / WRF
 * implicit gravity-wave absorber convention; it is intentionally much
 * stronger than the old post-step 1/600 s^-1 w-only sponge because a thin
 * acoustic/gravity-wave buffer must remove upward-propagating energy before it
 * reaches the rigid lid.
 */
export const MODEL_TOP_SPONGE:ModelTopSpongeConfig={
  startFraction:0.75,
  maxRate:0.2,
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
 * Legacy explicit helper retained for isolated experiments only.  Stage 4 does
 * NOT call this function: the production absorber is applied implicitly inside
 * HEVI before the new-time vertical mass/thermodynamic fluxes are evaluated.
 */
export function applyModelTopSponge(v:VerticalGrid,s:DryState,dt:number,config:ModelTopSpongeConfig=MODEL_TOP_SPONGE):void{
  if(!(dt>0))throw new Error('sponge dt must be positive');
  for(let c=0;c<s.wInterface.length/(v.nz+1);c++)for(let i=1;i<v.nz;i++){
    const rate=modelTopSpongeRate(v.zInterface[i]!,v.top,config);
    if(rate<=0)continue;
    const q=w3DIndex(c,i,v.nz);
    s.wInterface[q]=s.wInterface[q]!/(1+rate*dt);
  }
}
