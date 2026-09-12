export interface CoreV2GpuBlockTridiagonalBatch {
  columnCount: number;
  nz: number;
  lower: Float32Array;
  diagonal: Float32Array;
  upper: Float32Array;
  rhs: Float32Array;
}

export interface CoreV2GpuBlockTridiagonalResult {
  solution: Float32Array;
  status: Uint32Array;
}

type GPUAny = any;
const BLOCK = 5;
const BLOCK2 = 25;
const USAGE = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 } as const;
const MAP_READ = 1;

function aligned(bytes: number): number {
  return Math.max(4, Math.ceil(bytes / 4) * 4);
}

function upload(device: GPUAny, data: ArrayBufferView, usage: number, label: string): GPUAny {
  const buffer = device.createBuffer({
    label,
    size: aligned(data.byteLength),
    usage: usage | USAGE.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  );
  buffer.unmap();
  return buffer;
}

function empty(device: GPUAny, bytes: number, usage: number, label: string): GPUAny {
  return device.createBuffer({ label, size: aligned(bytes), usage });
}

async function readF32(device: GPUAny, source: GPUAny, count: number): Promise<Float32Array> {
  const bytes = count * 4;
  const staging = empty(device, bytes, USAGE.MAP_READ | USAGE.COPY_DST, 'core-v2-block5-read-f32');
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(MAP_READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

async function readU32(device: GPUAny, source: GPUAny, count: number): Promise<Uint32Array> {
  const bytes = count * 4;
  const staging = empty(device, bytes, USAGE.MAP_READ | USAGE.COPY_DST, 'core-v2-block5-read-u32');
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(MAP_READ);
  const out = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

function validate(batch: CoreV2GpuBlockTridiagonalBatch): void {
  if (!Number.isInteger(batch.columnCount) || batch.columnCount < 1) {
    throw new Error('Core v2 GPU block solver requires positive columnCount');
  }
  if (!Number.isInteger(batch.nz) || batch.nz < 1) {
    throw new Error('Core v2 GPU block solver requires positive nz');
  }
  const blocks = batch.columnCount * batch.nz;
  if (
    batch.lower.length !== blocks * BLOCK2 ||
    batch.diagonal.length !== blocks * BLOCK2 ||
    batch.upper.length !== blocks * BLOCK2 ||
    batch.rhs.length !== blocks * BLOCK
  ) throw new Error('Core v2 GPU block solver input lengths mismatch');
}

const SHADER = /* wgsl */`
struct Params{columnCount:u32,nz:u32,_p0:u32,_p1:u32};
@group(0) @binding(0) var<uniform> P:Params;
@group(0) @binding(1) var<storage,read> lower:array<f32>;
@group(0) @binding(2) var<storage,read> diagonal:array<f32>;
@group(0) @binding(3) var<storage,read> upper:array<f32>;
@group(0) @binding(4) var<storage,read> rhsInput:array<f32>;
@group(0) @binding(5) var<storage,read_write> modifiedUpper:array<f32>;
@group(0) @binding(6) var<storage,read_write> modifiedRhsAndSolution:array<f32>;
@group(0) @binding(7) var<storage,read_write> status:array<u32>;

struct DenseSolveResult{
  matrix:array<f32,25>,
  rhs:array<f32,25>,
  ok:u32,
};

fn dense5_solve(matrixInput:array<f32,25>,rhsInput:array<f32,25>,columnCount:u32)->DenseSolveResult{
  var matrix=matrixInput;
  var rhs=rhsInput;
  var result:DenseSolveResult;
  var matrixScale=0.0;
  for(var i:u32=0u;i<25u;i++){matrixScale=max(matrixScale,abs(matrix[i]));}
  if(!(matrixScale>0.0) || !isFinite(matrixScale)){
    result.matrix=matrix;result.rhs=rhs;result.ok=0u;return result;
  }
  for(var pivotColumn:u32=0u;pivotColumn<5u;pivotColumn++){
    var pivotRow=pivotColumn;
    var pivotMagnitude=abs(matrix[pivotColumn*5u+pivotColumn]);
    for(var row:u32=pivotColumn+1u;row<5u;row++){
      let magnitude=abs(matrix[row*5u+pivotColumn]);
      if(magnitude>pivotMagnitude){pivotMagnitude=magnitude;pivotRow=row;}
    }
    if(!(pivotMagnitude>1e-6*matrixScale) || !isFinite(pivotMagnitude)){
      result.matrix=matrix;result.rhs=rhs;result.ok=0u;return result;
    }
    if(pivotRow!=pivotColumn){
      for(var c:u32=0u;c<5u;c++){
        let a=pivotColumn*5u+c;let b=pivotRow*5u+c;let temp=matrix[a];matrix[a]=matrix[b];matrix[b]=temp;
      }
      for(var c:u32=0u;c<columnCount;c++){
        let a=pivotColumn*columnCount+c;let b=pivotRow*columnCount+c;let temp=rhs[a];rhs[a]=rhs[b];rhs[b]=temp;
      }
    }
    let pivot=matrix[pivotColumn*5u+pivotColumn];
    for(var row:u32=pivotColumn+1u;row<5u;row++){
      let factor=matrix[row*5u+pivotColumn]/pivot;
      matrix[row*5u+pivotColumn]=0.0;
      for(var c:u32=pivotColumn+1u;c<5u;c++){
        matrix[row*5u+c]-=factor*matrix[pivotColumn*5u+c];
      }
      for(var c:u32=0u;c<columnCount;c++){
        rhs[row*columnCount+c]-=factor*rhs[pivotColumn*columnCount+c];
      }
    }
  }
  for(var rr:i32=4;rr>=0;rr--){
    let row=u32(rr);let pivot=matrix[row*5u+row];
    for(var c:u32=0u;c<columnCount;c++){
      var value=rhs[row*columnCount+c];
      for(var k:u32=row+1u;k<5u;k++){value-=matrix[row*5u+k]*rhs[k*columnCount+c];}
      rhs[row*columnCount+c]=value/pivot;
    }
  }
  result.matrix=matrix;result.rhs=rhs;result.ok=1u;return result;
}

fn block_offset(column:u32,k:u32)->u32{return (column*P.nz+k)*25u;}
fn rhs_offset(column:u32,k:u32)->u32{return (column*P.nz+k)*5u;}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let column=gid.x;if(column>=P.columnCount){return;}
  status[column]=0u;
  for(var k:u32=0u;k<P.nz;k++){
    let bo=block_offset(column,k);let ro=rhs_offset(column,k);
    var b:array<f32,25>;var rv:array<f32,5>;
    for(var i:u32=0u;i<25u;i++){b[i]=diagonal[bo+i];}
    for(var i:u32=0u;i<5u;i++){rv[i]=rhsInput[ro+i];}
    if(k>0u){
      var a:array<f32,25>;var productBlock:array<f32,25>;var productVector:array<f32,5>;
      for(var i:u32=0u;i<25u;i++){a[i]=lower[bo+i];}
      let pbo=block_offset(column,k-1u);let pro=rhs_offset(column,k-1u);
      for(var r:u32=0u;r<5u;r++){
        for(var c:u32=0u;c<5u;c++){
          var value=0.0;
          for(var j:u32=0u;j<5u;j++){value+=a[r*5u+j]*modifiedUpper[pbo+j*5u+c];}
          productBlock[r*5u+c]=value;
        }
        var vv=0.0;
        for(var j:u32=0u;j<5u;j++){vv+=a[r*5u+j]*modifiedRhsAndSolution[pro+j];}
        productVector[r]=vv;
      }
      for(var i:u32=0u;i<25u;i++){b[i]-=productBlock[i];}
      for(var i:u32=0u;i<5u;i++){rv[i]-=productVector[i];}
    }

    if(k+1u<P.nz){
      var rhsMatrix:array<f32,25>;
      for(var r:u32=0u;r<5u;r++){for(var c:u32=0u;c<5u;c++){rhsMatrix[r*5u+c]=upper[bo+r*5u+c];}}
      let upperSolve=dense5_solve(b,rhsMatrix,5u);
      if(upperSolve.ok==0u){status[column]=1u;return;}
      for(var i:u32=0u;i<25u;i++){modifiedUpper[bo+i]=upperSolve.rhs[i];}
      var rhsColumn:array<f32,25>;
      for(var i:u32=0u;i<25u;i++){rhsColumn[i]=0.0;}
      for(var i:u32=0u;i<5u;i++){rhsColumn[i]=rv[i];}
      let rhsSolve=dense5_solve(b,rhsColumn,1u);
      if(rhsSolve.ok==0u){status[column]=2u;return;}
      for(var i:u32=0u;i<5u;i++){modifiedRhsAndSolution[ro+i]=rhsSolve.rhs[i];}
    }else{
      var rhsColumn:array<f32,25>;
      for(var i:u32=0u;i<25u;i++){rhsColumn[i]=0.0;}
      for(var i:u32=0u;i<5u;i++){rhsColumn[i]=rv[i];}
      let rhsSolve=dense5_solve(b,rhsColumn,1u);
      if(rhsSolve.ok==0u){status[column]=3u;return;}
      for(var i:u32=0u;i<5u;i++){modifiedRhsAndSolution[ro+i]=rhsSolve.rhs[i];}
    }
  }

  let last=rhs_offset(column,P.nz-1u);
  for(var i:u32=0u;i<5u;i++){modifiedRhsAndSolution[last+i]=modifiedRhsAndSolution[last+i];}
  for(var kk:i32=i32(P.nz)-2;kk>=0;kk--){
    let k=u32(kk);let bo=block_offset(column,k);let ro=rhs_offset(column,k);let next=rhs_offset(column,k+1u);
    var solved:array<f32,5>;
    for(var r:u32=0u;r<5u;r++){
      var value=modifiedRhsAndSolution[ro+r];
      for(var j:u32=0u;j<5u;j++){value-=modifiedUpper[bo+r*5u+j]*modifiedRhsAndSolution[next+j];}
      solved[r]=value;
    }
    for(var i:u32=0u;i<5u;i++){modifiedRhsAndSolution[ro+i]=solved[i];}
  }
}
`;

export class CoreV2GpuBlockTridiagonal5 {
  private readonly device: GPUAny;
  private readonly pipeline: GPUAny;
  private readonly pipelineValidation: Promise<any>;
  private readonly compilationInfo: Promise<any>;

  constructor(device: GPUAny) {
    this.device = device;
    this.device.pushErrorScope('validation');
    const module = device.createShaderModule({ label: 'core-v2-block5-module', code: SHADER });
    this.compilationInfo = module.getCompilationInfo();
    this.pipeline = device.createComputePipeline({
      label: 'core-v2-block5-pipeline',
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    this.pipelineValidation = this.device.popErrorScope();
  }

  async solve(batch: CoreV2GpuBlockTridiagonalBatch): Promise<CoreV2GpuBlockTridiagonalResult> {
    validate(batch);
    const setupError = await this.pipelineValidation;
    if (setupError) {
      const info = await this.compilationInfo;
      const messages = Array.from(info.messages ?? [])
        .map((message: any) => `${message.type ?? 'message'} ${message.lineNum ?? '?'}:${message.linePos ?? '?'} ${message.message ?? String(message)}`)
        .join(' | ');
      throw new Error(`Core v2 GPU block5 pipeline error: ${setupError.message}${messages ? `; shader: ${messages}` : ''}`);
    }

    const raw = new ArrayBuffer(16);
    const u = new Uint32Array(raw);
    u[0] = batch.columnCount;
    u[1] = batch.nz;
    const params = upload(this.device, new Uint8Array(raw), USAGE.UNIFORM, 'core-v2-block5-params');
    const lower = upload(this.device, batch.lower, USAGE.STORAGE, 'core-v2-block5-lower');
    const diagonal = upload(this.device, batch.diagonal, USAGE.STORAGE, 'core-v2-block5-diagonal');
    const upper = upload(this.device, batch.upper, USAGE.STORAGE, 'core-v2-block5-upper');
    const rhs = upload(this.device, batch.rhs, USAGE.STORAGE, 'core-v2-block5-rhs');
    const modifiedUpper = empty(this.device, batch.upper.byteLength, USAGE.STORAGE, 'core-v2-block5-modified-upper');
    const solution = empty(this.device, batch.rhs.byteLength, USAGE.STORAGE | USAGE.COPY_SRC, 'core-v2-block5-solution');
    const status = empty(this.device, batch.columnCount * 4, USAGE.STORAGE | USAGE.COPY_SRC, 'core-v2-block5-status');

    this.device.pushErrorScope('validation');
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: lower } },
        { binding: 2, resource: { buffer: diagonal } },
        { binding: 3, resource: { buffer: upper } },
        { binding: 4, resource: { buffer: rhs } },
        { binding: 5, resource: { buffer: modifiedUpper } },
        { binding: 6, resource: { buffer: solution } },
        { binding: 7, resource: { buffer: status } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(batch.columnCount / 64));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const validationError = await this.device.popErrorScope();
    if (validationError) {
      params.destroy();lower.destroy();diagonal.destroy();upper.destroy();rhs.destroy();modifiedUpper.destroy();solution.destroy();status.destroy();
      throw new Error(`Core v2 GPU block5 dispatch validation error: ${validationError.message}`);
    }

    const [solved, statusOut] = await Promise.all([
      readF32(this.device, solution, batch.rhs.length),
      readU32(this.device, status, batch.columnCount),
    ]);
    params.destroy();lower.destroy();diagonal.destroy();upper.destroy();rhs.destroy();modifiedUpper.destroy();solution.destroy();status.destroy();
    return { solution: solved, status: statusOut };
  }
}
