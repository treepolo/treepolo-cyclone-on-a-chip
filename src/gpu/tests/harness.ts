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
  return device;
}
