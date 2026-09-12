import { DRY_AIR } from '../core/constants.js';

type GPUAny = any;

const GPU_BUFFER_USAGE = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
} as const;
const GPU_MAP_MODE_READ = 0x0001;

function alignedSize(bytes: number): number {
  return Math.max(4, Math.ceil(bytes / 4) * 4);
}

function createUploadBuffer(
  device: GPUAny,
  data: ArrayBufferView,
  usage: number,
  label: string,
): GPUAny {
  const buffer = device.createBuffer({
    label,
    size: alignedSize(data.byteLength),
    usage: usage | GPU_BUFFER_USAGE.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  );
  buffer.unmap();
  return buffer;
}

function createEmptyBuffer(
  device: GPUAny,
  bytes: number,
  usage: number,
  label: string,
): GPUAny {
  return device.createBuffer({
    label,
    size: alignedSize(bytes),
    usage,
  });
}

async function readF32Buffer(
  device: GPUAny,
  source: GPUAny,
  floatCount: number,
): Promise<Float32Array> {
  const bytes = floatCount * 4;
  const staging = createEmptyBuffer(
    device,
    bytes,
    GPU_BUFFER_USAGE.MAP_READ | GPU_BUFFER_USAGE.COPY_DST,
    'core-v2-readback',
  );
  const encoder = device.createCommandEncoder({ label: 'core-v2-readback-encoder' });
  encoder.copyBufferToBuffer(source, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPU_MAP_MODE_READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return result;
}

/**
 * GPU storage layout for one conservative Core v2 cell:
 *
 *   state[2*q+0] = vec4(rho, rho*ux, rho*uy, rho*uz)
 *   state[2*q+1] = vec4(rhoE, 0, 0, 0)
 *
 * The extra padding is intentional: the same 16-byte aligned layout will be
 * used by the later face and column kernels instead of maintaining a second GPU
 * physics state representation.
 */
export const CORE_V2_GPU_FLOATS_PER_CELL = 8;

const COMMON = /* wgsl */`
struct Params {
  count:u32,
  _pad0:u32,
  _pad1:u32,
  _pad2:u32,
  omegaDt:vec4<f32>,
  gas:vec4<f32>,
};
@group(0) @binding(0) var<uniform> P:Params;

fn beta_plus(m:f32)->f32 {
  if (abs(m) < 1.0) {
    return 0.25 * (2.0-m) * (m+1.0) * (m+1.0);
  }
  return select(0.0, 1.0, m >= 0.0);
}

fn beta_minus(m:f32)->f32 {
  if (abs(m) < 1.0) {
    return 0.25 * (2.0+m) * (m-1.0) * (m-1.0);
  }
  return select(1.0, 0.0, m >= 0.0);
}
`;

const CORIOLIS_SHADER = COMMON + /* wgsl */`
@group(0) @binding(1) var<storage,read_write> state:array<vec4<f32>>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let q=gid.x;
  if(q>=P.count){return;}
  let omega=P.omegaDt.xyz;
  let dt=P.omegaDt.w;
  let omegaMagnitude=length(omega);
  if(omegaMagnitude==0.0 || dt==0.0){return;}

  let axis=omega/omegaMagnitude;
  let angle=-2.0*omegaMagnitude*dt;
  let c=cos(angle);
  let s=sin(angle);
  let packed=state[2u*q];
  let m=packed.yzw;
  let rotated=m*c + cross(axis,m)*s + axis*dot(axis,m)*(1.0-c);
  state[2u*q]=vec4<f32>(packed.x,rotated);
}
`;

/**
 * One face occupies two vec4 in left/right primitive buffers:
 *   [rho, ux, uy, uz], [pressure, geopotential, 0, 0]
 * and one geometry vec4:
 *   [A_x, A_y, A_z, p_ref]
 * Output is two vec4:
 *   [integrated mass, integrated mx, integrated my, integrated mz]
 *   [integrated total energy, 0, 0, 0]
 */
const SLAU2_REFERENCE_SUBTRACTED_SHADER = COMMON + /* wgsl */`
@group(0) @binding(1) var<storage,read> leftState:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> rightState:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> faceGeometry:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> outFlux:array<vec4<f32>>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let q=gid.x;
  if(q>=P.count){return;}

  let l0=leftState[2u*q];
  let l1=leftState[2u*q+1u];
  let r0=rightState[2u*q];
  let r1=rightState[2u*q+1u];
  let geom=faceGeometry[q];
  let vectorArea=geom.xyz;
  let area=length(vectorArea);
  let n=vectorArea/area;
  let pRef=geom.w;

  let rhoL=l0.x;
  let rhoR=r0.x;
  let velL=l0.yzw;
  let velR=r0.yzw;
  let pL=l1.x;
  let pR=r1.x;
  let phiL=l1.y;
  let phiR=r1.y;

  let gamma=P.gas.x;
  let aL=sqrt(gamma*pL/rhoL);
  let aR=sqrt(gamma*pR/rhoR);
  let aFace=0.5*(aL+aR);
  let unL=dot(velL,n);
  let unR=dot(velR,n);
  let speed2L=dot(velL,velL);
  let speed2R=dot(velR,velR);
  let mL=unL/aFace;
  let mR=unR/aFace;
  let velocityRms=sqrt(0.5*(speed2L+speed2R));
  let machHat=min(1.0,velocityRms/aFace);
  let chi=(1.0-machHat)*(1.0-machHat);
  let expansionSwitch=-max(min(mL,0.0),-1.0)*min(max(mR,0.0),1.0);
  let weightedAbs=(rhoL*abs(unL)+rhoR*abs(unR))/(rhoL+rhoR);
  let absLeft=(1.0-expansionSwitch)*weightedAbs+expansionSwitch*abs(unL);
  let absRight=(1.0-expansionSwitch)*weightedAbs+expansionSwitch*abs(unR);

  let pPrimeL=pL-pRef;
  let pPrimeR=pR-pRef;
  let pressureDifference=pPrimeR-pPrimeL;
  let massFlux=0.5*(
    rhoL*(unL+absLeft)+
    rhoR*(unR-absRight)-
    (chi/aFace)*pressureDifference
  );

  let bp=beta_plus(mL);
  let bm=beta_minus(mR);
  let rhoMean=0.5*(rhoL+rhoR);
  let velocityPressureTerm=velocityRms*(bp+bm-1.0)*aFace*rhoMean;
  let pressurePerturbationFlux=
    0.5*(pPrimeL+pPrimeR)+
    0.5*(bp-bm)*(pPrimeL-pPrimeR)+
    velocityPressureTerm;

  let useLeft=massFlux>=0.0;
  let upwindVelocity=select(velR,velL,useLeft);
  let upwindPressure=select(pR,pL,useLeft);
  let upwindDensity=select(rhoR,rhoL,useLeft);
  let upwindSpeed2=select(speed2R,speed2L,useLeft);
  let upwindPhi=select(phiR,phiL,useLeft);
  let hTotal=gamma/(gamma-1.0)*upwindPressure/upwindDensity+
    0.5*upwindSpeed2+upwindPhi;

  let momentumFlux=massFlux*upwindVelocity+pressurePerturbationFlux*n;
  outFlux[2u*q]=vec4<f32>(
    area*massFlux,
    area*momentumFlux.x,
    area*momentumFlux.y,
    area*momentumFlux.z
  );
  outFlux[2u*q+1u]=vec4<f32>(area*massFlux*hTotal,0.0,0.0,0.0);
}
`;

function makeParams(
  count: number,
  omega: readonly [number, number, number],
  dt: number,
): Uint8Array {
  const raw = new ArrayBuffer(48);
  const u32 = new Uint32Array(raw);
  const f32 = new Float32Array(raw);
  u32[0] = count;
  f32[4] = omega[0];
  f32[5] = omega[1];
  f32[6] = omega[2];
  f32[7] = dt;
  f32[8] = DRY_AIR.gamma;
  f32[9] = DRY_AIR.rd;
  f32[10] = DRY_AIR.cvd;
  f32[11] = DRY_AIR.pRef;
  return new Uint8Array(raw);
}

function validatePackedCells(packed: Float32Array): number {
  if (packed.length % CORE_V2_GPU_FLOATS_PER_CELL !== 0) {
    throw new Error('Core v2 GPU packed state length must be a multiple of 8');
  }
  return packed.length / CORE_V2_GPU_FLOATS_PER_CELL;
}

function validatePackedPrimitiveFaces(
  left: Float32Array,
  right: Float32Array,
  geometry: Float32Array,
): number {
  if (left.length !== right.length || left.length % 8 !== 0) {
    throw new Error('Core v2 GPU primitive face buffers must have matching 8-float records');
  }
  const count = left.length / 8;
  if (geometry.length !== count * 4) {
    throw new Error('Core v2 GPU face geometry must have four floats per face');
  }
  return count;
}

export class CoreV2GpuFoundation {
  readonly device: GPUAny;
  private readonly coriolisPipeline: GPUAny;
  private readonly slau2Pipeline: GPUAny;

  constructor(device: GPUAny) {
    this.device = device;
    const coriolisModule = device.createShaderModule({
      label: 'core-v2-coriolis-module',
      code: CORIOLIS_SHADER,
    });
    const slau2Module = device.createShaderModule({
      label: 'core-v2-slau2-reference-subtracted-module',
      code: SLAU2_REFERENCE_SUBTRACTED_SHADER,
    });
    this.coriolisPipeline = device.createComputePipeline({
      label: 'core-v2-coriolis-pipeline',
      layout: 'auto',
      compute: { module: coriolisModule, entryPoint: 'main' },
    });
    this.slau2Pipeline = device.createComputePipeline({
      label: 'core-v2-slau2-reference-subtracted-pipeline',
      layout: 'auto',
      compute: { module: slau2Module, entryPoint: 'main' },
    });
  }

  /** Hardware implementation of the same exact Coriolis flow used by CPU Core v2. */
  async applyExactCoriolis(
    packedState: Float32Array,
    omega: readonly [number, number, number],
    dt: number,
  ): Promise<Float32Array> {
    const count = validatePackedCells(packedState);
    if (!(dt >= 0) || !Number.isFinite(dt)) {
      throw new Error(`invalid Core v2 GPU Coriolis dt=${dt}`);
    }
    const params = createUploadBuffer(
      this.device,
      makeParams(count, omega, dt),
      GPU_BUFFER_USAGE.UNIFORM,
      'core-v2-coriolis-params',
    );
    const state = createUploadBuffer(
      this.device,
      packedState,
      GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_SRC,
      'core-v2-coriolis-state',
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.coriolisPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: state } },
      ],
    });
    const encoder = this.device.createCommandEncoder({ label: 'core-v2-coriolis-encoder' });
    const pass = encoder.beginComputePass({ label: 'core-v2-coriolis-pass' });
    pass.setPipeline(this.coriolisPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 128));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const result = await readF32Buffer(this.device, state, packedState.length);
    params.destroy();
    state.destroy();
    return result;
  }

  /**
   * Hardware evaluation of the exact Core v2 hydrostatic-reference-subtracted
   * SLAU2 face flux. This is the same mathematical face flux used by CPU Core v2.
   */
  async referenceSubtractedSlau2Faces(
    leftPrimitive: Float32Array,
    rightPrimitive: Float32Array,
    faceGeometry: Float32Array,
  ): Promise<Float32Array> {
    const count = validatePackedPrimitiveFaces(leftPrimitive, rightPrimitive, faceGeometry);
    const params = createUploadBuffer(
      this.device,
      makeParams(count, [0, 0, 0], 0),
      GPU_BUFFER_USAGE.UNIFORM,
      'core-v2-slau2-params',
    );
    const left = createUploadBuffer(
      this.device,
      leftPrimitive,
      GPU_BUFFER_USAGE.STORAGE,
      'core-v2-slau2-left',
    );
    const right = createUploadBuffer(
      this.device,
      rightPrimitive,
      GPU_BUFFER_USAGE.STORAGE,
      'core-v2-slau2-right',
    );
    const geometry = createUploadBuffer(
      this.device,
      faceGeometry,
      GPU_BUFFER_USAGE.STORAGE,
      'core-v2-slau2-geometry',
    );
    const output = createEmptyBuffer(
      this.device,
      count * 8 * 4,
      GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_SRC,
      'core-v2-slau2-output',
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.slau2Pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: left } },
        { binding: 2, resource: { buffer: right } },
        { binding: 3, resource: { buffer: geometry } },
        { binding: 4, resource: { buffer: output } },
      ],
    });
    const encoder = this.device.createCommandEncoder({ label: 'core-v2-slau2-encoder' });
    const pass = encoder.beginComputePass({ label: 'core-v2-slau2-pass' });
    pass.setPipeline(this.slau2Pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 128));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const result = await readF32Buffer(this.device, output, count * 8);
    params.destroy();
    left.destroy();
    right.destroy();
    geometry.destroy();
    output.destroy();
    return result;
  }
}
