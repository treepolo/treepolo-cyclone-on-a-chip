// Browser-side GPU test harness: results are printed to the console as "GPUTEST ..." lines,
// which tools/gpuTest.mjs (Playwright, SwiftShader WebGPU) collects.

let passed = 0, failed = 0;
export function gcheck(name: string, ok: boolean, value?: number | string): void {
  const v = value === undefined ? '' : `  (${typeof value === 'number' ? value.toExponential(3) : value})`;
  if (ok) passed++; else failed++;
  console.log(`GPUTEST ${ok ? 'PASS' : 'FAIL'} ${name}${v}`);
}
export function gdone(): void { console.log(`GPUTEST DONE ${passed}/${passed + failed}`); }

export async function getDevice(): Promise<GPUDevice> {
  if (!navigator.gpu) throw new Error('WebGPU unavailable');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('no WebGPU adapter');
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: Math.min(16, adapter.limits.maxStorageBuffersPerShaderStage), maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
  device.addEventListener('uncapturederror', (e) => { console.log(`GPUTEST FAIL gpu-error ${(e as GPUUncapturedErrorEvent).error.message.slice(0, 600)}`); });
  // a shader that does not compile says why (the pipeline made from it is only reported as invalid)
  const create = device.createShaderModule.bind(device);
  device.createShaderModule = (d: GPUShaderModuleDescriptor): GPUShaderModule => {
    const mod = create(d);
    void mod.getCompilationInfo().then((info) => { for (const m of info.messages) if (m.type === 'error') console.log(`GPUTEST FAIL shader ${m.lineNum}:${m.linePos} ${m.message.slice(0, 300)} | ${d.code.split('\n')[Math.max(0, m.lineNum - 1)]?.slice(0, 160)}`); });
    return mod;
  };
  return device;
}
