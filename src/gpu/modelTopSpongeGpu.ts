import { buildModelTopSpongeRates } from '../physics/modelTopSponge.js';
import { GpuRotatingDryCore } from './rotatingDryCoreGpu.js';

type GPUAny=any;

const SHADER=/* wgsl */`
struct Params{nz:u32,cellCount:u32,dt:f32,_pad:f32};
@group(0)@binding(0)var<uniform>P:Params;
@group(0)@binding(1)var<storage,read>rate:array<f32>;
@group(0)@binding(2)var<storage,read_write>w:array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){
  let q=gid.x;
  let stride=P.nz+1u;
  let count=P.cellCount*stride;
  if(q>=count){return;}
  let i=q%stride;
  let r=rate[i];
  if(r>0.0){w[q]=w[q]*exp(-r*P.dt);}
}`;

function makeBuffer(device:GPUAny,data:ArrayBufferView,usage:number,label:string):GPUAny{
  const size=Math.max(4,Math.ceil(data.byteLength/4)*4);
  const b=device.createBuffer({label,size,usage,mappedAtCreation:true});
  new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));
  b.unmap();return b;
}

export class GpuModelTopSponge{
  readonly device:GPUAny;
  private readonly params:GPUAny;
  private readonly rates:GPUAny;
  private readonly pipeline:GPUAny;
  private readonly group:GPUAny;
  constructor(public readonly gpu:GpuRotatingDryCore){
    this.device=gpu.device;
    const U=(globalThis as any).GPUBufferUsage;
    this.params=this.device.createBuffer({label:'model-top sponge params',size:16,usage:U.UNIFORM|U.COPY_DST});
    this.rates=makeBuffer(this.device,buildModelTopSpongeRates(gpu.v),U.STORAGE|U.COPY_DST,'model-top sponge rates');
    this.pipeline=this.device.createComputePipeline({label:'model-top sponge',layout:'auto',compute:{module:this.device.createShaderModule({label:'model-top sponge shader',code:SHADER}),entryPoint:'main'}});
    this.group=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.params}},{binding:1,resource:{buffer:this.rates}},{binding:2,resource:{buffer:gpu.core.buffers.w}}]});
  }
  prepare(dt:number):void{
    const ab=new ArrayBuffer(16),u=new Uint32Array(ab),f=new Float32Array(ab);
    u[0]=this.gpu.v.nz;u[1]=this.gpu.h.cellCount;f[2]=dt;this.device.queue.writeBuffer(this.params,0,ab);
  }
  encode(enc:GPUAny):void{
    const pass=enc.beginComputePass();pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.group);pass.dispatchWorkgroups(Math.ceil(this.gpu.h.cellCount*(this.gpu.v.nz+1)/128));pass.end();
  }
  apply(dt:number):void{
    this.prepare(dt);const enc=this.device.createCommandEncoder({label:'model-top sponge apply'});this.encode(enc);this.device.queue.submit([enc.finish()]);
  }
  destroy():void{this.params.destroy?.();this.rates.destroy?.();}
}
