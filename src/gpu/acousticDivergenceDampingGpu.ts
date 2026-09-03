import { ACOUSTIC_DIVERGENCE_DAMPING } from '../physics/acousticDivergenceDamping.js';
import { GpuRotatingDryCore } from './rotatingDryCoreGpu.js';

type GPUAny=any;

const DIVERGENCE=/* wgsl */`
struct Params{nz:u32,edgeCount:u32,cellCount:u32,_pad0:u32};
@group(0)@binding(0)var<uniform>P:Params;
@group(0)@binding(1)var<storage,read>cellArea:array<f32>;
@group(0)@binding(2)var<storage,read>edgeMetric:array<vec2<f32>>;
@group(0)@binding(3)var<storage,read>cellEdges:array<vec4<u32>>;
@group(0)@binding(4)var<storage,read>cellSigns:array<vec4<i32>>;
@group(0)@binding(5)var<storage,read>verticalMetric:array<vec2<f32>>;
@group(0)@binding(6)var<storage,read>u:array<f32>;
@group(0)@binding(7)var<storage,read>w:array<f32>;
@group(0)@binding(8)var<storage,read_write>divergence:array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){
  let q=gid.x;if(q>=P.cellCount*P.nz){return;}
  let c=q/P.nz;let k=q-c*P.nz;let ee=cellEdges[c];let ss=cellSigns[c];
  var hf=0.0;
  for(var s:u32=0u;s<4u;s++){
    let e=ee[s];hf+=f32(ss[s])*u[e*P.nz+k]*edgeMetric[e].x;
  }
  let wb=w[c*(P.nz+1u)+k];let wt=w[c*(P.nz+1u)+k+1u];let vm=verticalMetric[k];
  divergence[q]=hf/cellArea[c]+vm.y*wt-vm.x*wb;
}`;

const ADJUST=/* wgsl */`
struct Params{nz:u32,edgeCount:u32,cellCount:u32,_pad0:u32,coefficient:f32,_pad1:vec3<f32>};
@group(0)@binding(0)var<uniform>P:Params;
@group(0)@binding(1)var<storage,read>edgeCells:array<vec2<u32>>;
@group(0)@binding(2)var<storage,read>edgeMetric:array<vec2<f32>>;
@group(0)@binding(3)var<storage,read>divergence:array<f32>;
@group(0)@binding(4)var<storage,read_write>u:array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){
  let q=gid.x;if(q>=P.edgeCount*P.nz){return;}
  let e=q/P.nz;let k=q-e*P.nz;let cc=edgeCells[e];
  let dl=divergence[cc.x*P.nz+k];let dr=divergence[cc.y*P.nz+k];
  u[q]=u[q]+P.coefficient*edgeMetric[e].y*(dr-dl);
}`;

function makeBuffer(device:GPUAny,data:ArrayBufferView,usage:number,label:string):GPUAny{
  const size=Math.max(4,Math.ceil(data.byteLength/4)*4),b=device.createBuffer({label,size,usage,mappedAtCreation:true});
  new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));b.unmap();return b;
}

export class GpuAcousticDivergenceDamping{
  readonly device:GPUAny;
  private readonly params:GPUAny;
  private readonly verticalMetric:GPUAny;
  private readonly divergence:GPUAny;
  private readonly divergencePipeline:GPUAny;
  private readonly adjustPipeline:GPUAny;
  private readonly divergenceGroup:GPUAny;
  private readonly adjustGroup:GPUAny;
  constructor(public readonly gpu:GpuRotatingDryCore){
    this.device=gpu.device;const U=(globalThis as any).GPUBufferUsage,storage=U.STORAGE|U.COPY_DST|U.COPY_SRC;
    this.params=this.device.createBuffer({label:'acoustic divergence params',size:32,usage:U.UNIFORM|U.COPY_DST});
    const vm=new Float32Array(gpu.v.nz*2);
    for(let k=0;k<gpu.v.nz;k++){
      const rho=Math.max(gpu.ref.rhoCenter[k]!,1e-12),dz=gpu.v.dz[k]!;
      vm[k*2]=gpu.ref.rhoInterface[k]!/(rho*dz);vm[k*2+1]=gpu.ref.rhoInterface[k+1]!/(rho*dz);
    }
    this.verticalMetric=makeBuffer(this.device,vm,storage,'acoustic vertical metric');
    this.divergence=this.device.createBuffer({label:'acoustic 3D divergence',size:gpu.h.cellCount*gpu.v.nz*4,usage:storage});
    this.divergencePipeline=this.device.createComputePipeline({label:'acoustic divergence',layout:'auto',compute:{module:this.device.createShaderModule({label:'acoustic divergence shader',code:DIVERGENCE}),entryPoint:'main'}});
    this.adjustPipeline=this.device.createComputePipeline({label:'acoustic divergence adjust',layout:'auto',compute:{module:this.device.createShaderModule({label:'acoustic divergence adjust shader',code:ADJUST}),entryPoint:'main'}});
    const b=gpu.core.buffers;
    this.divergenceGroup=this.device.createBindGroup({layout:this.divergencePipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:this.params}},{binding:1,resource:{buffer:b.cellArea}},{binding:2,resource:{buffer:b.edgeMetric}},{binding:3,resource:{buffer:b.cellEdges}},{binding:4,resource:{buffer:b.cellSigns}},{binding:5,resource:{buffer:this.verticalMetric}},{binding:6,resource:{buffer:b.u}},{binding:7,resource:{buffer:b.w}},{binding:8,resource:{buffer:this.divergence}},
    ]});
    this.adjustGroup=this.device.createBindGroup({layout:this.adjustPipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:this.params}},{binding:1,resource:{buffer:b.edgeCells}},{binding:2,resource:{buffer:b.edgeMetric}},{binding:3,resource:{buffer:this.divergence}},{binding:4,resource:{buffer:b.u}},
    ]});
  }
  apply(coefficient=ACOUSTIC_DIVERGENCE_DAMPING):void{
    if(!(coefficient>=0&&coefficient<=0.25))throw new Error('acoustic divergence coefficient must be in [0,0.25]');
    const ab=new ArrayBuffer(32),u=new Uint32Array(ab),f=new Float32Array(ab);u[0]=this.gpu.v.nz;u[1]=this.gpu.h.edgeCount;u[2]=this.gpu.h.cellCount;f[4]=coefficient;this.device.queue.writeBuffer(this.params,0,ab);
    const enc=this.device.createCommandEncoder({label:'acoustic divergence damping'});
    let p=enc.beginComputePass();p.setPipeline(this.divergencePipeline);p.setBindGroup(0,this.divergenceGroup);p.dispatchWorkgroups(Math.ceil(this.gpu.h.cellCount*this.gpu.v.nz/128));p.end();
    p=enc.beginComputePass();p.setPipeline(this.adjustPipeline);p.setBindGroup(0,this.adjustGroup);p.dispatchWorkgroups(Math.ceil(this.gpu.h.edgeCount*this.gpu.v.nz/128));p.end();
    this.device.queue.submit([enc.finish()]);
  }
  destroy():void{this.params.destroy?.();this.verticalMetric.destroy?.();this.divergence.destroy?.();}
}
