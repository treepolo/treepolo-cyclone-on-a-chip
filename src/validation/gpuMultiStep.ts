import { CubedSphereGrid } from '../grid/cubedSphere.js';
import { VerticalGrid } from '../grid/vertical.js';
import { ReferenceAtmosphere } from '../physics/referenceAtmosphere.js';
import { GpuDryCorePrototype } from '../gpu/dryCoreGpu.js';
import { diagnoseState } from '../solver/diagnostics.js';
import { DryCoreCpu } from '../solver/dryCoreCpu.js';
import { createHydrostaticState, DryState } from '../solver/state.js';

export const GPU_VALIDATION_DEFAULTS = {
  steps: 1000,
  dt: 0.25,
  checkpoints: [1, 10, 100, 250, 500, 1000] as const,
  thresholds: {
    massDrift: 1e-6,
    maxAbsW: 1e-3,
    rhoRelL2: 2e-5,
    rhoThetaRelL2: 2e-5,
    maxAbsUDiff: 1e-4,
    maxAbsWDiff: 1e-3,
  },
};

export interface GpuValidationSample {
  step: number;
  simTime: number;
  gpuMassDrift: number;
  cpuMassDrift: number;
  gpuMaxAbsW: number;
  cpuMaxAbsW: number;
  rhoRelL2: number;
  rhoThetaRelL2: number;
  maxAbsUDiff: number;
  maxAbsWDiff: number;
  minRho: number;
  minP: number;
  invalid: boolean;
}

export interface GpuMultiStepResult {
  passed: boolean;
  steps: number;
  dt: number;
  elapsedMs: number;
  samples: GpuValidationSample[];
  failures: string[];
}

export type GpuValidationProgress = (sample: GpuValidationSample, totalSteps: number) => void;

function relativeL2(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return Infinity;
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i++) {
    const av = a[i]!;
    const bv = b[i]!;
    const d = av - bv;
    num += d * d;
    den += bv * bv;
  }
  return Math.sqrt(num / Math.max(den, Number.MIN_VALUE));
}

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return Infinity;
  let out = 0;
  for (let i = 0; i < a.length; i++) out = Math.max(out, Math.abs(a[i]! - b[i]!));
  return out;
}

function compareStates(
  h: CubedSphereGrid,
  v: VerticalGrid,
  gpuState: DryState,
  cpuState: DryState,
  gpuInitialMass: number,
  cpuInitialMass: number,
  step: number,
  dt: number,
): GpuValidationSample {
  const gd = diagnoseState(h, v, gpuState);
  const cd = diagnoseState(h, v, cpuState);
  return {
    step,
    simTime: step * dt,
    gpuMassDrift: (gd.dryMass - gpuInitialMass) / gpuInitialMass,
    cpuMassDrift: (cd.dryMass - cpuInitialMass) / cpuInitialMass,
    gpuMaxAbsW: gd.maxAbsW,
    cpuMaxAbsW: cd.maxAbsW,
    rhoRelL2: relativeL2(gpuState.rhoD, cpuState.rhoD),
    rhoThetaRelL2: relativeL2(gpuState.rhoThetaM, cpuState.rhoThetaM),
    maxAbsUDiff: maxAbsDiff(gpuState.uEdge, cpuState.uEdge),
    maxAbsWDiff: maxAbsDiff(gpuState.wInterface, cpuState.wInterface),
    minRho: gd.minRho,
    minP: gd.minP,
    invalid: gd.nan || gd.minRho <= 0 || gd.minP <= 0,
  };
}

export async function runHydrostaticGpuValidation(
  h: CubedSphereGrid,
  v: VerticalGrid,
  ref: ReferenceAtmosphere,
  onProgress?: GpuValidationProgress,
): Promise<GpuMultiStepResult> {
  const { steps, dt, thresholds } = GPU_VALIDATION_DEFAULTS;
  const checkpointSet = new Set<number>(GPU_VALIDATION_DEFAULTS.checkpoints.filter((x) => x <= steps));
  checkpointSet.add(steps);

  const cpu = new DryCoreCpu(h, v, ref);
  const cpuState = createHydrostaticState(h, v, ref);
  const gpuSeed = createHydrostaticState(h, v, ref);
  const cpuInitialMass = diagnoseState(h, v, cpuState).dryMass;
  const gpu = await GpuDryCorePrototype.create(h, v, ref, gpuSeed);
  const samples: GpuValidationSample[] = [];
  const failures: string[] = [];
  const started = performance.now();

  try {
    const gpuInitial = await gpu.downloadState(0);
    const gpuInitialMass = diagnoseState(h, v, gpuInitial).dryMass;

    for (let step = 1; step <= steps; step++) {
      cpu.step(cpuState, dt);
      gpu.step(dt);

      if (checkpointSet.has(step)) {
        await gpu.device.queue.onSubmittedWorkDone();
        const gpuState = await gpu.downloadState(step * dt);
        const sample = compareStates(h, v, gpuState, cpuState, gpuInitialMass, cpuInitialMass, step, dt);
        samples.push(sample);
        onProgress?.(sample, steps);
        if (sample.invalid) break;
      }

      if (step % 25 === 0) {
        await gpu.device.queue.onSubmittedWorkDone();
        await new Promise<void>((resolve) => setTimeout(resolve, 0));
      }
    }

    for (const s of samples) {
      if (s.invalid) failures.push(`step ${s.step}: invalid density/pressure/NaN state`);
      if (Math.abs(s.gpuMassDrift) > thresholds.massDrift) failures.push(`step ${s.step}: |GPU mass drift|=${Math.abs(s.gpuMassDrift).toExponential(3)} > ${thresholds.massDrift}`);
      if (s.gpuMaxAbsW > thresholds.maxAbsW) failures.push(`step ${s.step}: GPU max|w|=${s.gpuMaxAbsW.toExponential(3)} m/s > ${thresholds.maxAbsW}`);
      if (s.rhoRelL2 > thresholds.rhoRelL2) failures.push(`step ${s.step}: rho relative L2=${s.rhoRelL2.toExponential(3)} > ${thresholds.rhoRelL2}`);
      if (s.rhoThetaRelL2 > thresholds.rhoThetaRelL2) failures.push(`step ${s.step}: rhoTheta relative L2=${s.rhoThetaRelL2.toExponential(3)} > ${thresholds.rhoThetaRelL2}`);
      if (s.maxAbsUDiff > thresholds.maxAbsUDiff) failures.push(`step ${s.step}: max|GPU-CPU u|=${s.maxAbsUDiff.toExponential(3)} m/s > ${thresholds.maxAbsUDiff}`);
      if (s.maxAbsWDiff > thresholds.maxAbsWDiff) failures.push(`step ${s.step}: max|GPU-CPU w|=${s.maxAbsWDiff.toExponential(3)} m/s > ${thresholds.maxAbsWDiff}`);
    }

    if (samples.length === 0 || samples[samples.length - 1]!.step !== steps) {
      failures.push(`validation stopped before ${steps} steps`);
    }

    return {
      passed: failures.length === 0,
      steps,
      dt,
      elapsedMs: performance.now() - started,
      samples,
      failures,
    };
  } finally {
    gpu.destroy();
  }
}
