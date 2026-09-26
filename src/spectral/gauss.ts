// Gauss–Legendre quadrature nodes (mu = sin(latitude)) and weights on [-1, 1].
// Nodes are returned north-to-south (mu descending); weights sum to 2.

export interface GaussianLatitudes {
  mu: Float64Array;
  weight: Float64Array;
  lat: Float64Array; // radians
}

export function gaussianLatitudes(nlat: number): GaussianLatitudes {
  if (nlat % 2 !== 0) throw new Error('nlat must be even');
  const mu = new Float64Array(nlat);
  const weight = new Float64Array(nlat);
  const half = nlat / 2;
  for (let j = 0; j < half; j++) {
    let x = Math.cos(Math.PI * (j + 0.75) / (nlat + 0.5));
    let dp = 0;
    for (let iter = 0; iter < 100; iter++) {
      // Legendre P_n(x) by three-term recurrence
      let p0 = 1, p1 = x;
      for (let n = 2; n <= nlat; n++) {
        const p2 = ((2 * n - 1) * x * p1 - (n - 1) * p0) / n;
        p0 = p1; p1 = p2;
      }
      dp = nlat * (x * p1 - p0) / (x * x - 1);
      const dx = p1 / dp;
      x -= dx;
      if (Math.abs(dx) < 1e-15) break;
    }
    // recompute derivative at converged root
    let p0 = 1, p1 = x;
    for (let n = 2; n <= nlat; n++) {
      const p2 = ((2 * n - 1) * x * p1 - (n - 1) * p0) / n;
      p0 = p1; p1 = p2;
    }
    dp = nlat * (x * p1 - p0) / (x * x - 1);
    const w = 2 / ((1 - x * x) * dp * dp);
    mu[j] = x; weight[j] = w;
    mu[nlat - 1 - j] = -x; weight[nlat - 1 - j] = w;
  }
  const lat = new Float64Array(nlat);
  for (let j = 0; j < nlat; j++) lat[j] = Math.asin(mu[j]!);
  return { mu, weight, lat };
}
