# Stage 4 Held–Suarez spin-up diagnosis — 2026-09-05

## Trigger

After the HEVI vertical private-scratch optimization, a fresh 30-day `N=8 × Nz=48 × 40 km`, `dt=10 s` RK3 split-explicit run reproduced the pre-optimization day-30 diagnostics to displayed precision:

- mass drift `1.125e-5`;
- upper-midlatitude max westerly `29.655 m/s`;
- tropical low-level mean zonal wind `+2.762 m/s`;
- max overturning streamfunction `2.356e12 kg/s`;
- max `|w| = 4.599e-3 m/s`;
- max edge wind `59.061 m/s`.

The only legacy endpoint failure is the tropical low-level wind sign (`trade < 0` required by the development gate). This confirms the GPU performance optimization did not create the circulation result.

## Important reinterpretation of the 30-day gate

A 30-day instantaneous endpoint is not the canonical Held–Suarez climate statistic. Held & Suarez (1994) and common implementations use a long spin-up before climatological averaging; a widely used setup discards the first 200 days and averages the following 1000 days. Therefore day-30 `trade < 0` remains useful as a development/transient diagnostic, but it must not be treated as sufficient evidence that the long-term Held–Suarez climate is correct or incorrect.

The present run itself is strongly transient in this metric:

- day 11.25: trade approximately `-1.663 m/s`;
- day 17.50: trade approximately `-0.039 m/s`;
- day 17.75: trade approximately `+0.046 m/s`;
- day 30.00: trade approximately `+2.762 m/s`.

Thus the immediate next question is whether the sign reversal continues, saturates, or reverses during longer spin-up.

## Current code checks

The implemented Held–Suarez analytic equilibrium temperature, Newtonian relaxation-rate profile, and near-surface Rayleigh-drag rate match the standard parameter form. The current tropical-low-level diagnostic is the area-weighted zonal-mean zonal wind over `5–30 deg` absolute latitude and model height `<=2.5 km`; there is no obvious east/west sign inversion in that diagnostic.

## Diagnostic continuation path

A new diagnostic-only page is added at:

`stage4-spinup.html`

It:

- reads the latest compatible production or spin-up diagnostic checkpoint;
- can extend the same numerical model to day 60, 120, or 200;
- retags only checkpoint target metadata; it does not alter prognostic state or numerical equations;
- stores progress under a separate IndexedDB key (`stage4-rk3-spinup-diagnostic`), leaving the production checkpoint untouched;
- reports the legacy endpoint diagnostics and the linear slope of tropical low-level zonal wind over the latest five simulated days.

## Next diagnosis after spin-up separation

If the tropical low-level wind returns to a persistent easterly during longer spin-up, the old 30-day sign gate should be reclassified or replaced by a post-spin-up time-mean climate statistic.

If the wind remains persistently westerly or continues to accelerate eastward through longer spin-up, do not tune Held–Suarez forcing merely to force the sign. The next structural checks should be:

1. axial angular-momentum budget / numerical torque of the spherical momentum discretization;
2. resolved baroclinic-eddy activity and eddy momentum/heat fluxes;
3. sensitivity to the very coarse `N=8` development grid and first-order donor-cell material advection;
4. only then, any question about mapping the Held–Suarez thermal relaxation into the fully compressible thermodynamic variable.

The numerical stability and mass gates remain unchanged.