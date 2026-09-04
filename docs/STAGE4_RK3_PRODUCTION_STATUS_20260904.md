# Stage 4 RK3 Production Status — 2026-09-04

## Current production path

Stage 4 production has been promoted from the legacy one-step HEVI/outer split to the verified three-stage RK3 split-explicit integrator.

- outer timestep: `dt = 10 s`
- RK stages: `1/3 -> 1/2 -> 1`
- acoustic schedule for `ns=4`:
  - stage 1: `1 x dt/3`
  - stage 2: `2 x dt/4`
  - stage 3: `4 x dt/4`
- slow RHS is frozen per RK stage
- acoustic variables are predictor-relative
- each RK stage acoustic integration restarts from the immutable large-step base
- horizontal acoustic correction uses a forward/backward C-grid update
- vertical acoustic/gravity coupling is off-centered implicit (`epsilon=0.10`, `theta=0.55`)
- model-top Rayleigh absorption is applied inside the vertical implicit acoustic/gravity solve
- continuity uses a symmetric reference/perturbation split: `Fref` in the fast subsystem and `Fpert` in the slow subsystem
- horizontal and vertical momentum advection are material tendencies
- Held-Suarez forcing, Coriolis, 3-D momentum transport, and acoustic divergence damping are enabled in production climate runs

The previous `GpuStage4Integrator` implementation remains in the repository only as a legacy comparison path; `stage4.html` now uses the RK3 split-explicit path.

## Hardware validation completed

All following tests passed on the user's real WebGPU device.

### Predictor-relative acoustic column

- `rho` relative L2: `5.451e-8`
- `rhoTheta` relative L2: `4.250e-8`
- GPU Fref mass self-identity relative L2: `3.816e-8`
- GPU Fref rhoTheta self-identity relative L2: `3.808e-8`
- max `|Delta w|`: `3.037e-5 m/s`
- hydrostatic max `|w|`: `7.312e-14 m/s`

### Frozen slow RHS

- GPU horizontal Fpert self-identity: `0`
- GPU vertical Fpert self-identity: `0`
- GPU rho divergence self-identity: `7.201e-8`
- GPU rhoTheta divergence self-identity: `7.697e-8`
- Held-Suarez thermal-only CPU/GPU relative L2: `2.264e-6`
- max `|Delta du/dt|`: `4.629e-11 m/s^2`
- max `|Delta dw/dt|`: `5.796e-13 m/s^2`
- hydrostatic rest max tendency: `0`

### Full one-outer-step RK3 GPU/CPU agreement

- `rho` relative L2: `1.107e-7`
- `rhoTheta` relative L2: `8.306e-8`
- max `|Delta u|`: `3.260e-7 m/s`
- max `|Delta w|`: `8.749e-5 m/s`
- GPU one-step mass drift: `-1.894e-9`

### 40-step batched GPU/CPU agreement

The 40 outer steps are encoded into one GPU command buffer.

- `rho` relative L2: `5.396e-7`
- `rhoTheta` relative L2: `1.857e-7`
- max `|Delta u|`: `3.960e-6 m/s`
- max `|Delta w|`: `2.503e-4 m/s`
- GPU mass drift: `1.557e-9`

## Production-scale two-day Held-Suarez gate

Grid and runtime configuration:

- cubed sphere `N=8`
- 384 horizontal columns
- `Nz=48`
- `Htop=40 km`
- 18,432 3-D cells
- outer `dt=10 s`
- RK3 split-explicit production integrator

The two-day gate passed without relaxing any stability thresholds.

Day 2 diagnostics:

- mass drift: `8.974e-7`
- max `|w|`: `6.218e-4 m/s`
- max `|w|` below absorber: `6.218e-4 m/s`
- max `|w|` in absorber: `6.008e-4 m/s`
- max `|u_edge|`: `2.819 m/s`
- horizontal divergence RMS: `2.567e-8 s^-1`
- horizontal CFL: `2.418e-5`
- vertical CFL: `1.019e-5`

For comparison, the legacy one-step split integrator produced approximately `13.64 m/s` vertical velocity in the upper absorber by day 2. The RK3 split-explicit path therefore removes the previously observed upper-level fast-mode instability over the two-day production-scale test.

## Remaining Stage 4 milestone

The remaining Stage 4 closure gate is the 30-day Held-Suarez development run on the same `N=8 x Nz=48 x 40 km` configuration.

The gates remain unchanged:

- `|mass drift| <= 5e-5`
- upper-midlatitude max westerly `> 0.5 m/s`
- tropical low-level zonal wind `< 0`
- max overturning streamfunction `> 1e9 kg/s`
- NH and SH dominant overturning signs opposite
- `max |w| < 10 m/s` throughout
- no invalid density or pressure

No state clamping, hard-coded circulation, or relaxed stability gate is permitted as a closure fix.
