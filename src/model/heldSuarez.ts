// Held & Suarez (1994) idealised forcing for dry dynamical cores.
//   Newtonian relaxation toward a zonally symmetric radiative-equilibrium temperature and
//   Rayleigh friction in the boundary layer (sigma > 0.7), sigma = p / p_surface (local).
// Reference: Held, I. M. and M. J. Suarez, 1994, BAMS 75, 1825–1830.

import { DAY, DryAir } from '../core/constants.js';
import { GridState, PhysicsForcing, PhysicsTendencies } from './dycore.js';

export const HS94 = {
  kf: 1 / DAY,
  ka: 1 / (40 * DAY),
  ks: 1 / (4 * DAY),
  sigmaB: 0.7,
  deltaTy: 60,     // K
  deltaThetaZ: 10, // K
  T0: 315,         // K
  Tmin: 200,       // K
};

export function heldSuarezTeq(sinLat: number, p: number, air: DryAir): number {
  const c2 = 1 - sinLat * sinLat;
  const pr = p / air.pRef;
  const t = (HS94.T0 - HS94.deltaTy * sinLat * sinLat - HS94.deltaThetaZ * Math.log(pr) * c2) * Math.pow(pr, air.kappa);
  return Math.max(HS94.Tmin, t);
}

export class HeldSuarezForcing implements PhysicsForcing {
  constructor(private readonly air: DryAir) {}

  compute(g: GridState, t: PhysicsTendencies): void {
    const ng = g.nlat * g.nlon, sb = HS94.sigmaB;
    for (let k = 0; k < g.K; k++) {
      const sig = g.sigma[k]!;
      const bl = Math.max(0, (sig - sb) / (1 - sb));
      const kv = HS94.kf * bl;
      for (let j = 0; j < g.nlat; j++) {
        const mu = g.mu[j]!, c2 = 1 - mu * mu;
        const kt = HS94.ka + (HS94.ks - HS94.ka) * bl * c2 * c2;
        for (let i = 0; i < g.nlon; i++) {
          const p2 = j * g.nlon + i, q = k * ng + p2;
          const p = sig * g.ps[p2]!;
          t.dT[q] = t.dT[q]! - kt * (g.T[q]! - heldSuarezTeq(mu, p, this.air));
          if (kv > 0) {
            t.du[q] = t.du[q]! - kv * g.u[q]!;
            t.dv[q] = t.dv[q]! - kv * g.v[q]!;
          }
        }
      }
    }
  }
}
