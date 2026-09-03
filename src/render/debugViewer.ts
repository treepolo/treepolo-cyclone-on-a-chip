import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { DryState, cell3DIndex, w3DIndex } from '../solver/state.js';

export class DebugViewer {
  private ctx:CanvasRenderingContext2D; private yaw=0.6; private pitch=-0.25; private zoom=1; private dragging=false; private px=0;private py=0;
  constructor(private canvas:HTMLCanvasElement,private h:CubedSphereGrid,private v:VerticalGrid,private ref:ReferenceAtmosphere){
    const ctx=canvas.getContext('2d');if(!ctx)throw new Error('Canvas2D unavailable');this.ctx=ctx;this.attach();new ResizeObserver(()=>this.resize()).observe(canvas);this.resize();
  }
  private resize(){const dpr=Math.min(devicePixelRatio||1,2);const r=this.canvas.getBoundingClientRect();this.canvas.width=Math.max(1,Math.floor(r.width*dpr));this.canvas.height=Math.max(1,Math.floor(r.height*dpr));}
  private attach(){
    this.canvas.addEventListener('pointerdown',e=>{this.dragging=true;this.px=e.clientX;this.py=e.clientY;this.canvas.setPointerCapture(e.pointerId);});
    this.canvas.addEventListener('pointermove',e=>{if(!this.dragging)return;this.yaw+=(e.clientX-this.px)*0.008;this.pitch=Math.max(-1.45,Math.min(1.45,this.pitch+(e.clientY-this.py)*0.008));this.px=e.clientX;this.py=e.clientY;});
    this.canvas.addEventListener('pointerup',()=>this.dragging=false);this.canvas.addEventListener('pointercancel',()=>this.dragging=false);
    this.canvas.addEventListener('wheel',e=>{e.preventDefault();this.zoom=Math.max(0.55,Math.min(2.5,this.zoom*Math.exp(-e.deltaY*0.001)));},{passive:false});
  }
  private rotate(x:number,y:number,z:number):[number,number,number]{
    const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);const x1=cy*x-sy*y,y1=sy*x+cy*y;return [x1,cp*y1-sp*z,sp*y1+cp*z];
  }
  draw(s:DryState):void{
    const ctx=this.ctx,W=this.canvas.width,H=this.canvas.height;ctx.clearRect(0,0,W,H);ctx.fillStyle='#05070a';ctx.fillRect(0,0,W,H);
    const scale=Math.min(W,H)*0.37*this.zoom,cx=W*0.5,cy=H*0.5;
    ctx.beginPath();ctx.arc(cx,cy,scale,0,Math.PI*2);ctx.fillStyle='#09131c';ctx.fill();ctx.strokeStyle='#31506a';ctx.lineWidth=Math.max(1,devicePixelRatio);ctx.stroke();
    const pts:{x:number;y:number;z:number;r:number;w:number;thp:number}[]=[];const kStride=this.v.nz>40?3:2;
    for(let c=0;c<this.h.cellCount;c++) for(let k=0;k<this.v.nz;k+=kStride){
      const alt=this.v.zCenter[k]!,rr=1+0.42*alt/this.v.top;const x=this.h.cellCenters[c*3]!*rr,y=this.h.cellCenters[c*3+1]!*rr,z=this.h.cellCenters[c*3+2]!*rr;const [xr,yr,zr]=this.rotate(x,y,z);if(zr<-0.15)continue;
      const persp=1/(1.6-0.22*zr),q=cell3DIndex(c,k,this.v.nz),wi=w3DIndex(c,Math.min(k+1,this.v.nz),this.v.nz);const w=s.wInterface[wi]!,theta=s.rhoThetaM[q]!/s.rhoD[q]!,thp=theta-this.ref.thetaCenter[k]!;
      pts.push({x:cx+xr*scale*persp,y:cy-yr*scale*persp,z:zr,r:Math.max(0.7,1.45*devicePixelRatio*persp),w,thp});
    }
    pts.sort((a,b)=>a.z-b.z);for(const p of pts){const a=Math.min(1,0.22+0.18*Math.abs(p.w)+0.3*Math.abs(p.thp));const hue=p.w>0?25:205;ctx.fillStyle=`hsla(${hue},80%,65%,${a})`;ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fill();}
    ctx.fillStyle='#9db1c4';ctx.font=`${11*Math.max(1,devicePixelRatio)}px ui-monospace,monospace`;ctx.fillText(`t=${s.time.toFixed(1)} s`,12*devicePixelRatio,20*devicePixelRatio);
  }
}
