// The automatic step size of the regional model, in one place (worker.ts for the CPU and GPU runs, and the tests).
//
// dt = min(acoustic limit, target Courant number / max(|u|/dx + |v|/dy + |w|/dz)), at most the set-up's own step (the CPU) or
// three times it (the GPU); it shrinks at once, grows by at most 10 % per check, and a Courant number beyond 1.375 times the
// target cuts it back hard. The acoustic limit keeps the horizontal sound Courant number of the split steps
// c dt / (n_s dx) <= 0.45 with the speed of sound of the hottest air there is now (c = 350 m/s is what the set-up's step
// was made for; much hotter air - an interaction that heats it by thousands of K - shortens the step in proportion).
// Nothing keeps the step from getting as small as the flow needs: a limit on it would be a limit on what can be done to the model.

export const CFL_DEFAULT = 0.8;
/** the smallest step, as a fraction of the set-up's own (only there so that the step is never zero) */
export const DT_MIN_FRAC = 1e-4;
/** speed of sound the set-up's step is made for (m/s) */
export const C_SOUND = 350;
/** air whose sound speed is more than this factor above C_SOUND is hot enough to shorten the step */
const HOT_FACTOR = 1.3;

export interface StepInput {
  /** the step now (s) */
  cur: number;
  /** the set-up's own step (s) */
  dt0: number;
  /** max(|u|/dx + |v|/dy + |w|/dz) (1/s) */
  rate: number;
  /** speed of sound of the hottest air (m/s) */
  cmax: number;
  /** target Courant number */
  cfl: number;
  /** smaller horizontal grid spacing (m) */
  dxMin: number;
  nsound: number;
  /** the GPU may take steps up to three times the set-up's, within the acoustic limit */
  gpu: boolean;
}

export function nextDt(i: StepInput): number {
  const { cur, dt0, rate, cmax, cfl, dxMin, nsound, gpu } = i;
  const ac350 = nsound * 0.45 * dxMin / C_SOUND;
  // (air of an ordinary run is at most about 330 K, 364 m/s: the step is only shortened for air hotter than that by a margin)
  const hot = Number.isFinite(cmax) && cmax > HOT_FACTOR * C_SOUND ? HOT_FACTOR * C_SOUND / cmax : 1;
  const hi = (gpu ? Math.max(dt0, Math.min(3 * dt0, ac350)) : dt0) * hot;
  let next = Math.min(hi, cfl / Math.max(rate, 1e-9));
  if (rate * cur > 1.375 * cfl) next = Math.min(next, 0.875 * cfl / rate);       // overshoot: cut back hard
  else if (next > cur) next = Math.min(next, 1.1 * cur);
  return Math.max(DT_MIN_FRAC * dt0, next);
}

/** Speed of sound (m/s) of air of potential temperature th (K) at Exner function pi. */
export const soundSpeed = (th: number, pi: number): number => Math.sqrt(1.4 * 287.05 * Math.max(th * pi, 0));

/**
 * The step that an interaction which warms the air by `dT` K (buoyancy a = g dT / theta over the step: a dt^2 / dz <= `ramp`
 * Courant numbers) can start with, before any wind exists to measure.
 */
export function bubbleDt(dT: number, theta: number, dz: number, ramp = 0.3): number {
  const a = 9.80665 * Math.abs(dT) / Math.max(theta, 100);
  return a > 0 ? Math.sqrt(ramp * dz / a) : Infinity;
}
