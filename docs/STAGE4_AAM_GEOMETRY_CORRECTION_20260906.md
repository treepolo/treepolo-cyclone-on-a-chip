# Stage 4 AAM geometry correction — 2026-09-06

## Trigger

A conservation-pair AAM diagnostic continued the pre-correction Held–Suarez state from diagnostic day 30.75 to 31.75. The instantaneous decomposition at day 30.75 showed large nonphysical inviscid torques despite exact algebraic decomposition closure:

- planetary-AAM mass redistribution: `+2.8429e19 N m` (`+0.09388 m/s/day`);
- Coriolis acceleration: `+9.4593e20 N m` (`+3.12378 m/s/day`);
- planetary mass + Coriolis residual: `+9.7436e20 N m` (`+3.21766 m/s/day`);
- relative-AAM mass redistribution: `+2.6560e17 N m` (`+0.00088 m/s/day`);
- material momentum advection: `-8.1328e19 N m` (`-0.26857 m/s/day`);
- relative mass + material momentum residual: `-8.1063e19 N m` (`-0.26770 m/s/day`);
- global pressure-gradient torque: `-5.1152e20 N m` (`-1.68921 m/s/day`), split as panel interior `+6.0965e20 N m` and panel seam `-1.1212e21 N m`;
- full frozen RHS AAM tendency: `+1.7025e20 N m` (`+0.56223 m/s/day`).

The one-day integrated budget from day 30.75 to 31.75 gave observed AAM tendency `+1.1932e20 N m`, mean physical drag torque `-2.1466e20 N m`, and residual numerical/filter torque `+3.3398e20 N m` (`+1.10284 m/s/day`).

These values established that the earlier low-level tropical westerly drift could not be interpreted only as Held–Suarez spin-up: the discrete spatial operators contained a large artificial axial-AAM source.

## Root cause: edge velocity was not actually face-normal

`HorizontalEdge.normal` had been constructed from the line joining the two neighboring cell centers, projected into the edge-midpoint tangent plane. That direction is a center-to-center connector, not the true conormal of the shared great-circle face on an equiangular gnomonic cubed sphere.

The prognostic `uEdge` was therefore described and used as a C-grid face-normal velocity while geometrically representing a different direction. Continuity, Coriolis projection, pressure acceleration, wind reconstruction, and momentum transport were consequently not operating on a common discrete velocity geometry.

The production grid now defines each edge normal from the great-circle plane through the edge endpoints and orients it from `leftCell` to `rightCell`. The pre-correction prognostic states are intentionally checkpoint-incompatible because the meaning of every `uEdge` value changed.

## Analytic AAM convergence result

A smooth analytic spherical-flow regression was added to test the two conservation pairs independently. With the old connector normal, the planetary-mass + Coriolis residual did not converge under refinement and approached an O(1) error. A diagnostic A/B using the true face conormal changed that behavior to approximately second-order convergence.

The production regression after the correction reports equivalent acceleration in `m/s/day`:

| N | pressure torque | planetary mass + Coriolis | relative mass + momentum | pair sum |
|---:|---:|---:|---:|---:|
| 4 | `-3.419231e-15` | `+3.176704e-1` | `-6.443015e-1` | `-3.266312e-1` |
| 8 | `+4.115755e-15` | `+9.399249e-2` | `-3.544648e-1` | `-2.604724e-1` |
| 16 | `+3.324862e-15` | `+2.483113e-2` | `-1.845025e-1` | `-1.596713e-1` |
| 32 | `+1.122901e-14` | `+6.321569e-3` | `-9.398726e-2` | `-8.766569e-2` |

Refinement ratios are:

- planetary mass + Coriolis: `3.785` for N=8→16 and `3.928` for N=16→32, approximately second order;
- relative mass + material momentum: `1.921` and `1.963`, approximately first order, consistent with the present donor-cell material-momentum transport;
- smooth closed-sphere pressure torque remains at numerical zero, O(`1e-14 m/s/day`).

Thus the nonconvergent Coriolis/continuity defect is fixed. The remaining material-momentum AAM error is convergent but low-order and remains a later accuracy target.

## Non-orthogonal pressure gradient

Changing only the edge normal exposed a second invalid assumption: the previous horizontal pressure acceleration used a two-point center difference divided by center distance. That is only a face-normal derivative on an orthogonal grid.

The Stage 4 CPU and GPU RK3 split operators now use a non-orthogonal reconstruction:

1. reconstruct a cell-centered tangent gradient from the four neighboring cells with a 2-D least-squares fit in the local east/north tangent basis;
2. average left and right cell gradients at a shared face;
3. project the averaged gradient onto the true shared-face conormal;
4. use the same operator for predictor pressure acceleration and for the linearized acoustic pressure correction.

The geostrophic-balance CPU regression, which failed after switching to the true face normal while retaining the old center-difference stencil, passes again with the non-orthogonal gradient.

## Validation after the correction

The full CPU CI is green, including Stage 0/1, Stage 4 CPU, RK3 schedule, slow tendency, acoustic column, RK3 CPU, split consistency, and the production AAM convergence regression.

A browser WebGPU production gate on N=8 × 48, H=40 km, dt=10 s also passed all gates:

### Predictor-relative acoustic

- rho relative L2 `5.451e-8`;
- rho-theta relative L2 `4.250e-8`;
- max `|Delta w| = 3.037e-5 m/s`;
- hydrostatic max `|w| = 7.312e-14 m/s`.

### Frozen slow RHS

- raw rho tendency relative L2 `3.463e-4`;
- raw rho-theta tendency relative L2 `4.296e-5`;
- max `|Delta du/dt| = 4.629e-11 m/s^2`;
- max `|Delta dw/dt| = 5.796e-13 m/s^2`;
- hydrostatic rest max tendency `0`.

### Full RK3 and 40-step GPU batch

One-step:

- rho relative L2 `1.107e-7`;
- rho-theta relative L2 `8.306e-8`;
- max `|Delta u| = 2.758e-7 m/s`;
- max `|Delta w| = 8.749e-5 m/s`.

Forty-step single GPU batch:

- rho relative L2 `4.932e-7`;
- rho-theta relative L2 `1.846e-7`;
- max `|Delta u| = 2.030e-6 m/s`;
- max `|Delta w| = 2.680e-4 m/s`;
- GPU mass drift `4.593e-8` over 400 simulated seconds.

### Two-day production-scale Held–Suarez stability

- final day `2.00`;
- mass drift `7.851e-7`;
- max `|w| = 7e-4 m/s`;
- max `|u_edge| = 3.139 m/s`;
- horizontal divergence RMS `3.300e-8 s^-1`;
- max horizontal CFL `2.680e-5`;
- max vertical CFL `1.210e-5`;
- elapsed `44.44 s`.

No numerical gate was relaxed.

## Performance implication and next step

The corrected GPU pressure-gradient implementation is deliberately direct and redundant for correctness first: each edge invocation reconstructs left and right least-squares cell gradients, including spherical geometry terms, and the acoustic `hvel` kernel repeats this for every acoustic substep. It is therefore expected to cost more than the old two-point stencil.

The existing performance page measures production 40-step batch throughput, and the timestamp profiler already reports `prep.prepU` and `acoustic.hvel` separately. Those two pages are sufficient to measure the cost before changing the implementation.

If profiling confirms pressure reconstruction is now a major hotspot, the preferred low-risk optimization is to precompute reusable cell-gradient geometry/stencil coefficients and then, if necessary, precompute cell pressure/delta-pressure gradients into buffers before edge projection. This must preserve exactly the corrected non-orthogonal operator and must be followed by CPU/GPU agreement and production-scale stability gates.

Only after performance is characterized should a fresh long Held–Suarez run be started. Old spin-up and production checkpoints must not be reused.