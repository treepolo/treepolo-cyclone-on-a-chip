// Physical constants (SI). All model code reads planet / gas properties from here.

export interface Planet {
  radius: number;      // m
  omega: number;       // s^-1
  gravity: number;     // m s^-2
}

export interface DryAir {
  rd: number;          // J kg^-1 K^-1
  cp: number;          // J kg^-1 K^-1
  kappa: number;       // R_d / c_p
  pRef: number;        // Pa
}

export const EARTH: Planet = {
  radius: 6.371e6,
  omega: 7.292115e-5,
  gravity: 9.80665,
};

const rd = 287.05;
const cp = 1004.5;
export const DRY_AIR: DryAir = { rd, cp, kappa: rd / cp, pRef: 1.0e5 };

export const DAY = 86400;

/** Lowest air temperature (K) the regional model keeps: the advection scheme can undershoot beside a very steep
 *  temperature jump (a bubble of thousands of K), and the saturation formulas have a pole near 36 K. Real air
 *  does not get colder than about 180 K; nothing in a run gets near this floor but an interaction that asks for
 *  more cold than there is. */
export const T_FLOOR = 100;
/** Fastest fall speed (m/s) of rain, snow, graupel and ice in the sedimentation: nothing in the air falls faster (hail about 50 m/s). The bound keeps the sub-stepping
 *  finite for a state that is not numbers any more (an infinite fall speed would make the loop endless; on the GPU, trip the watchdog). */
export const VT_MAX = 100;
