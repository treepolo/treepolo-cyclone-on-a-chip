// Spectral transform regressions: quadrature, round trips, vector identities.
import { SpectralTransform, SpectralField } from '../spectral/transform.js';
import { RealFFT } from '../spectral/fft.js';
import { check, summary, rng } from './assert.js';

function randomSpec(t: SpectralTransform, seed: number): SpectralField {
  const r = rng(seed), s = t.newSpec();
  for (let i = 0; i < t.nspec; i++) {
    s.re[i] = r() - 0.5;
    s.im[i] = t.mOf[i] === 0 ? 0 : r() - 0.5;
  }
  return s;
}
function maxDiff(a: SpectralField, b: SpectralField): number {
  let d = 0;
  for (let i = 0; i < a.re.length; i++) d = Math.max(d, Math.abs(a.re[i]! - b.re[i]!), Math.abs(a.im[i]! - b.im[i]!));
  return d;
}

// FFT round trip
{
  const n = 64, f = new RealFFT(n), r = rng(3), x = new Float64Array(n), y = new Float64Array(n);
  const cr = new Float64Array(n / 2), ci = new Float64Array(n / 2);
  // band-limited random signal with m <= 20
  const ar = new Float64Array(21), ai = new Float64Array(21);
  for (let m = 0; m <= 20; m++) { ar[m] = r() - 0.5; ai[m] = m ? r() - 0.5 : 0; }
  for (let i = 0; i < n; i++) {
    let v = ar[0]!;
    for (let m = 1; m <= 20; m++) { const th = 2 * Math.PI * m * i / n; v += 2 * (ar[m]! * Math.cos(th) - ai[m]! * Math.sin(th)); }
    x[i] = v;
  }
  f.forward(x, 0, cr, ci, 20);
  let e = 0;
  for (let m = 0; m <= 20; m++) e = Math.max(e, Math.abs(cr[m]! - ar[m]!), Math.abs(ci[m]! - ai[m]!));
  check('FFT forward matches analytic coefficients', e < 1e-13, e);
  f.inverse(cr, ci, 20, y, 0);
  let e2 = 0;
  for (let i = 0; i < n; i++) e2 = Math.max(e2, Math.abs(x[i]! - y[i]!));
  check('FFT inverse round trip', e2 < 1e-12, e2);
}

for (const T of [21, 42]) {
  const t = new SpectralTransform(T);
  let wsum = 0;
  for (let j = 0; j < t.nlat; j++) wsum += t.weight[j]!;
  check(`T${T}: Gaussian weights sum to 2`, Math.abs(wsum - 2) < 1e-13, wsum - 2);

  const a = randomSpec(t, 11), g = t.newGrid(), b = t.newSpec();
  t.synth(a, g);
  t.anal(g, b);
  check(`T${T}: scalar spectral->grid->spectral round trip`, maxDiff(a, b) < 1e-12, maxDiff(a, b));

  // vorticity/divergence -> (U,V) -> vorticity/divergence
  const radius = 6.371e6;
  const vor = randomSpec(t, 5), div = randomSpec(t, 7);
  vor.re[0] = 0; div.re[0] = 0;
  const U = t.newGrid(), V = t.newGrid();
  t.synthUV(vor, div, radius, U, V);
  const d2 = t.newSpec(), z2 = t.newSpec();
  t.analDivCurl(U, V, radius, d2, z2);
  check(`T${T}: div(U,V) recovers divergence`, maxDiff(div, d2) < 1e-11, maxDiff(div, d2));
  check(`T${T}: curl(U,V) recovers vorticity`, maxDiff(vor, z2) < 1e-11, maxDiff(vor, z2));

  // gradient identity: div(grad X) = -n(n+1)/a^2 X
  const X = randomSpec(t, 13), dl = t.newGrid(), dm = t.newGrid();
  t.synthGrad(X, dl, dm);
  // (A,B) = (dX/dlambda, (1-mu^2) dX/dmu)/a  are cos(phi) * grad components
  for (let i = 0; i < dl.length; i++) { dl[i] = dl[i]! / radius; dm[i] = dm[i]! / radius; }
  const lap = t.newSpec();
  t.analDivCurl(dl, dm, radius, lap, null);
  let e = 0;
  for (let s = 0; s < t.nspec; s++) {
    const f = -t.nn1[s]! / (radius * radius);
    e = Math.max(e, Math.abs(lap.re[s]! - f * X.re[s]!) * radius * radius, Math.abs(lap.im[s]! - f * X.im[s]!) * radius * radius);
  }
  check(`T${T}: div(grad X) = Laplacian eigenvalues`, e < 1e-9, e);

  // solid-body rotation: U = u0 cos^2(phi) has vorticity 2 u0 mu / a
  const u0 = 20;
  for (let j = 0; j < t.nlat; j++) for (let i = 0; i < t.nlon; i++) { U[j * t.nlon + i] = u0 * (1 - t.mu[j]! ** 2); V[j * t.nlon + i] = 0; }
  const z3 = t.newSpec();
  t.analDivCurl(U, V, radius, null, z3);
  const expect = 2 * u0 / radius / Math.sqrt(1.5); // P_1^0 = sqrt(3/2) mu
  check(`T${T}: solid-body vorticity coefficient`, Math.abs(z3.re[1]! - expect) < 1e-18, z3.re[1]! - expect);
}

summary('spectral');
