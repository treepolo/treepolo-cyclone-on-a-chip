export interface VerticalGrid { nz:number; top:number; zInterface:Float64Array; zCenter:Float64Array; dz:Float64Array; }
export function buildStretchedVerticalGrid(nz:number, top=40000, stretch=2.2):VerticalGrid {
  if(!Number.isInteger(nz)||nz<4) throw new Error('nz must be >=4');
  const zi=new Float64Array(nz+1), zc=new Float64Array(nz), dz=new Float64Array(nz);
  const denom=Math.expm1(stretch);
  for(let k=0;k<=nz;k++){ const s=k/nz; zi[k]=top*Math.expm1(stretch*s)/denom; }
  for(let k=0;k<nz;k++){ dz[k]=zi[k+1]!-zi[k]!; zc[k]=0.5*(zi[k+1]!+zi[k]!); }
  return {nz,top,zInterface:zi,zCenter:zc,dz};
}
