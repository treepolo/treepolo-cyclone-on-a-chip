export interface PlanetConfig {
  radius: number;
  omega: number;
  gravity: number;
}

export interface AtmosphereConfig {
  rd: number;
  rv: number;
  cpd: number;
  cvd: number;
  gamma: number;
  kappa: number;
  pRef: number;
}

export const EARTH: PlanetConfig = {
  radius: 6.371e6,
  omega: 7.292115e-5,
  gravity: 9.80665,
};

export const DRY_AIR: AtmosphereConfig = (() => {
  const rd = 287.05;
  const cpd = 1004.5;
  const cvd = cpd - rd;
  return { rd, rv: 461.5, cpd, cvd, gamma: cpd / cvd, kappa: rd / cpd, pRef: 100000 };
})();
