// Sigma vertical coordinate with the Simmons & Burridge (1981) energy- and
// angular-momentum-conserving discretisation (pure-sigma special case).
//
// Levels k = 0..K-1 run top -> bottom. Half levels sigmaHalf[0] = 0 (model top), sigmaHalf[K] = 1.

import { DryAir } from '../core/constants.js';

export interface SigmaLevels {
  K: number;
  sigmaHalf: Float64Array;   // K+1
  sigma: Float64Array;       // K, full levels (p_k / p_s for diagnostics)
  dsigma: Float64Array;      // K
  lnRatio: Float64Array;     // ln(sigmaHalf[k+1]/sigmaHalf[k]); lnRatio[0] unused (infinite)
  alpha: Float64Array;       // SB81 alpha_k
  /** Phi_k - Phi_s = sum_j G[k][j] T_j  (row-major K x K) */
  G: Float64Array;
  /** Linearised energy conversion: dT_k/dt = -sum_j tau[k][j] D_j about isothermal T_ref */
  tau: Float64Array;
  tRef: number;
}

export function uniformSigmaHalf(K: number): Float64Array {
  const s = new Float64Array(K + 1);
  for (let k = 0; k <= K; k++) s[k] = k / K;
  return s;
}

export function buildSigmaLevels(sigmaHalf: Float64Array, air: DryAir, tRef: number): SigmaLevels {
  const K = sigmaHalf.length - 1;
  const sigma = new Float64Array(K), dsigma = new Float64Array(K);
  const lnRatio = new Float64Array(K), alpha = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    const a = sigmaHalf[k]!, b = sigmaHalf[k + 1]!;
    dsigma[k] = b - a;
    if (k === 0) {
      lnRatio[k] = Infinity;
      alpha[k] = Math.LN2;
    } else {
      lnRatio[k] = Math.log(b / a);
      alpha[k] = 1 - a / (b - a) * lnRatio[k]!;
    }
    // full-level sigma consistent with Phi_k = Phi_{k+1/2} + alpha_k R T_k:  ln(sigma_k) = ln(sigma_{k+1/2}) - alpha_k
    sigma[k] = b * Math.exp(-alpha[k]!);
  }
  const R = air.rd;
  const G = new Float64Array(K * K);
  for (let k = 0; k < K; k++) {
    G[k * K + k] = R * alpha[k]!;
    for (let j = k + 1; j < K; j++) G[k * K + j] = R * lnRatio[j]!;
  }
  const tau = new Float64Array(K * K);
  const kap = air.kappa;
  for (let k = 0; k < K; k++) {
    for (let j = 0; j < k; j++) tau[k * K + j] = kap * tRef * lnRatio[k]! * dsigma[j]! / dsigma[k]!;
    tau[k * K + k] = kap * tRef * alpha[k]!;
  }
  return { K, sigmaHalf, sigma, dsigma, lnRatio, alpha, G, tau, tRef };
}
