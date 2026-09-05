declare const process:{exitCode?:number};
import { EARTH } from '../core/constants.js';
import { buildCubedSphere } from '../grid/cubedSphere.js';
import { buildStretchedVerticalGrid } from '../grid/vertical.js';
import { buildHeldSuarezReference } from '../physics/heldSuarez.js';
import { buildRotationGeometry, reconstructCellHorizontalWind, type RotationGeometry } from '../physics/rotation.js';
import { diagnoseAxialAngularMomentum, diagnoseAxialAngularMomentumTendency } from '../solver/stage4CirculationDiagnostics.js';
import { diagnoseStage4DirectionalAamBreakdown } from '../solver/stage4MomentumBudgetDiagnostics.js';
import { cell3DIndex, createHydrostaticState, edge3DIndex, w3DIndex } from '../solver/state.js';

type V3=readonly[number,number,number];
function basis(r:V3):{east:V3;north:V3}{const xy=Math.hypot(r[0],r[1]),e:V3=xy>1e-14?[-r[1]/xy,r[0]/xy,0]:[0,1,0],n:V3=[-r[2]*e[1],r[2]*e[0],xy];return{east:e,north:n};}
function project(h:ReturnType<typeof buildCubedSphere>,v:ReturnType<typeof buildStretchedVerticalGrid>,g:RotationGeometry,cell:Float64Array[]):Float64Array{
  const out=new Float64Array(h.edgeCount*v.nz);
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,l=ge.leftCell,r=ge.rightCell,n=ge.normal;
    for(let k=0;k<v.nz;k++){
      const a=cell[k]!,lo=l*3,ro=r*3,lx=g.radial[lo]!,ly=g.radial[lo+1]!,lz=g.radial[lo+2]!,rx=g.radial[ro]!,ry=g.radial[ro+1]!,rz=g.radial[ro+2]!;
      const ld=a[lo]!*lx+a[lo+1]!*ly+a[lo+2]!*lz,rd=a[ro]!*rx+a[ro+1]!*ry+a[ro+2]!*rz;
      const ax=.5*((a[lo]!-ld*lx)+(a[ro]!-rd*rx)),ay=.5*((a[lo+1]!-ld*ly)+(a[ro+1]!-rd*ry)),az=.5*((a[lo+2]!-ld*lz)+(a[ro+2]!-rd*rz));
      out[edge3DIndex(e,k,v.nz)]=ax*n[0]+ay*n[1]+az*n[2];
    }
  }
  return out;
}
function state(nz:number){
  const h=buildCubedSphere(8),v=buildStretchedVerticalGrid(nz,18000,1),ref=buildHeldSuarezReference(v),g=buildRotationGeometry(h),s=createHydrostaticState(h,v,ref);
  for(let e=0;e<h.edgeCount;e++){
    const ge=h.edges[e]!,r=ge.midpoint,b=basis(r);
    for(let k=0;k<nz;k++){
      const z=v.zCenter[k]!/v.top,ue=(14+4*r[0]-2*r[1]+3*r[2])*(1+.5*z+.15*z*z),vn=(3+5*r[0]+2*r[1]-4*r[2])*(1-.25*z+.1*z*z),wx=ue*b.east[0]+vn*b.north[0],wy=ue*b.east[1]+vn*b.north[1],wz=ue*b.east[2]+vn*b.north[2];
      s.uEdge[edge3DIndex(e,k,nz)]=wx*ge.normal[0]+wy*ge.normal[1]+wz*ge.normal[2];
    }
  }
  for(let c=0;c<h.cellCount;c++){
    const x=h.cellCenters[c*3]!,y=h.cellCenters[c*3+1]!,z=h.cellCenters[c*3+2]!;
    for(let i=1;i<nz;i++){const zz=v.zInterface[i]!/v.top;s.wInterface[w3DIndex(c,i,nz)]=.22*Math.sin(Math.PI*zz)*(x-.7*y+.3*z);}
    for(let k=0;k<nz;k++){const q=cell3DIndex(c,k,nz),fac=1+.03*(x*y+.35*z)*(1-.4*v.zCenter[k]!/v.top);s.rhoD[q]=s.rhoD[q]!*fac;s.rhoThetaM[q]=s.rhoThetaM[q]!*fac;}
  }
  return{h,v,ref,g,s};
}
function legacyVerticalU(q:ReturnType<typeof state>):Float64Array{
  const {h,v,g,s}=q,nz=v.nz,winds=Array.from({length:nz},(_,k)=>reconstructCellHorizontalWind(h,g,s,k)),cell=Array.from({length:nz},()=>new Float64Array(h.cellCount*3));
  for(let c=0;c<h.cellCount;c++)for(let k=0;k<nz;k++){
    const o=c*3,dv=cell[k]!,wc=.5*(s.wInterface[w3DIndex(c,k,nz)]!+s.wInterface[w3DIndex(c,k+1,nz)]!),cur=winds[k]!;
    if(wc>0&&k>0){const below=winds[k-1]!,dz=v.zCenter[k]!-v.zCenter[k-1]!;dv[o]=-wc*(cur[o]!-below[o]!)/dz;dv[o+1]=-wc*(cur[o+1]!-below[o+1]!)/dz;dv[o+2]=-wc*(cur[o+2]!-below[o+2]!)/dz;}
    else if(wc<0&&k<nz-1){const above=winds[k+1]!,dz=v.zCenter[k+1]!-v.zCenter[k]!;dv[o]=-wc*(above[o]!-cur[o]!)/dz;dv[o+1]=-wc*(above[o+1]!-cur[o+1]!)/dz;dv[o+2]=-wc*(above[o+2]!-cur[o+2]!)/dz;}
  }
  return project(h,v,g,cell);
}
function independentMassConsistentU(q:ReturnType<typeof state>):Float64Array{
  const {h,v,ref,g,s}=q,nz=v.nz,R=EARTH.radius,winds=Array.from({length:nz},(_,k)=>reconstructCellHorizontalWind(h,g,s,k)),cell=Array.from({length:nz},()=>new Float64Array(h.cellCount*3));
  for(let c=0;c<h.cellCount;c++){
    const area=h.cellAreaUnit[c]!*R*R;
    for(let k=0;k<nz;k++){
      const cur=winds[k]!,o=c*3,rho=Math.max(s.rhoD[cell3DIndex(c,k,nz)]!,1e-12),vol=area*v.dz[k]!;
      let sumM=0,sumX=0,sumY=0,sumZ=0;
      for(const [i,sign] of [[k,-1],[k+1,1]] as const){
        if(i<=0||i>=nz)continue;
        const w=s.wInterface[w3DIndex(c,i,nz)]!,srcK=w>=0?i-1:i,src=cell3DIndex(c,srcK,nz),M=(ref.rhoInterface[i]!+s.rhoD[src]!-ref.rhoCenter[srcK]!)*w*area,faceSign=sign*M,donor=winds[srcK]!;
        sumM+=faceSign;sumX+=faceSign*donor[o]!;sumY+=faceSign*donor[o+1]!;sumZ+=faceSign*donor[o+2]!;
      }
      const a=cell[k]!;a[o]=(-sumX+cur[o]!*sumM)/(rho*vol);a[o+1]=(-sumY+cur[o+1]!*sumM)/(rho*vol);a[o+2]=(-sumZ+cur[o+2]!*sumM)/(rho*vol);
    }
  }
  return project(h,v,g,cell);
}
try{
  console.log('Stage4 vertical horizontal-momentum AAM regression; N8 horizontal grid');
  console.log('nz\tlegacy pair\tproduction pair\timprovement\tindependent delta\tdirectional closure');
  const rows=[] as {nz:number;legacy:number;production:number}[];
  for(const nz of [6,12,24,48]){
    const q=state(nz),dir=diagnoseStage4DirectionalAamBreakdown(q.h,q.v,q.ref,q.s,q.g),legacyU=legacyVerticalU(q),indU=independentMassConsistentU(q),zeroRho=new Float64Array(q.s.rhoD.length),legacyMom=diagnoseAxialAngularMomentumTendency(q.h,q.v,q.s,zeroRho,legacyU,q.g).velocityTorque,indMom=diagnoseAxialAngularMomentumTendency(q.h,q.v,q.s,zeroRho,indU,q.g).velocityTorque,lever=diagnoseAxialAngularMomentum(q.h,q.v,q.s,q.g).torqueLeverMass,scale=86400/lever,legacy=(dir.verticalRelativeMass+legacyMom)*scale,production=dir.verticalPairResidual*scale,independent=(dir.verticalRelativeMass+indMom)*scale,delta=production-independent;
    rows.push({nz,legacy,production});
    console.log(`${nz}\t${legacy.toExponential(8)}\t${production.toExponential(8)}\t${(Math.abs(legacy)/Math.max(Math.abs(production),1e-30)).toFixed(2)}x\t${delta.toExponential(3)}\t${(dir.closure*scale).toExponential(3)}`);
    if(![legacy,production,independent,delta].every(Number.isFinite))throw new Error(`non-finite vertical regression at nz=${nz}`);
    if(Math.abs(delta)>1e-11)throw new Error(`production vertical operator disagrees with independent reference at nz=${nz}: ${delta}`);
    if(nz===48&&Math.abs(production)>Math.abs(legacy)/100)throw new Error(`production vertical AAM pairing did not improve by 100x at nz=48: legacy=${legacy} production=${production}`);
  }
  const ratio=(a:number,b:number)=>Math.abs(a)/Math.max(Math.abs(b),1e-30);
  console.log(`legacy refine 12->24=${ratio(rows[1]!.legacy,rows[2]!.legacy).toFixed(3)} 24->48=${ratio(rows[2]!.legacy,rows[3]!.legacy).toFixed(3)}`);
  console.log(`production fixed-N8 floor 12->24=${ratio(rows[1]!.production,rows[2]!.production).toFixed(3)} 24->48=${ratio(rows[2]!.production,rows[3]!.production).toFixed(3)}`);
}catch(e){console.error('FAIL Stage4 vertical momentum regression');console.error(e);process.exitCode=1;}
