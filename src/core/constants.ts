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
