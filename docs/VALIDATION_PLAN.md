# Validation Plan v1

原則：每一層物理在加入下一層以前先通過自己的 quantitative tests；畫面「像天氣」不能當驗收。

## V0 — geometry / transport infrastructure

1. cubed-sphere total area = `4 pi R^2`。
2. 所有 interior/seam edges exactly two neighbors。
3. uniform scalar field 的離散 flux divergence = 0 within tolerance。
4. solid-body rotation passive tracer 一圈後 global mass drift：CPU Float64 ≤ `1e-12`；GPU Float32 初始 target ≤ `1e-6 relative`，之後依 reduction/flux method 收緊。
5. positive-definite transport：若 initial `q>=0`，不得產生有物理量級的負值。
6. seam error 不得比同解析度 panel interior error 出現 order-of-magnitude jump。

## V1 — dry non-rotating 3D core

- equation-of-state round trip
- isothermal hydrostatic rest
- stratified hydrostatic rest
- acoustic wave phase/amplitude
- internal gravity wave
- rising thermal bubble
- density current
- vertical-column acoustic CFL test

硬門檻：hydrostatic-rest case 不得持續生成系統性垂直風；dry-mass conservation 沒過就停止後續 physics。

## V2 — rotating sphere

- inertial oscillation
- geostrophic balance
- solid-body atmosphere rotation
- standard spherical-wave / Rossby test where appropriate
- Held–Suarez dry dynamical-core climate

Held–Suarez 驗收要比較 long-run zonal-mean temperature、zonal wind、meridional overturning；不能只截一張看似三胞環流的粒子圖。

## V3 — baroclinic weather

- Jablonowski–Williamson / DCMIP-like baroclinic wave
- frontogenesis diagnostics
- cyclone intensification / track convergence with resolution
- Rossby long-/short-wave propagation

要求：不施加 cyclone generator，微小 perturbation 能自行成長為合理 baroclinic eddies、鋒面與溫帶氣旋。

## V4 — terrain

- resting stratified atmosphere over terrain
- Schär mountain wave
- reduced-radius nonhydrostatic mountain-wave test
- terrain pressure-gradient spurious-wind metric

## V5 — moist thermodynamics

- closed-box phase-change water conservation
- latent-heating consistency
- saturation-adjustment equilibrium
- warm-rain column
- moist thermal bubble / deep convection
- Kessler-like benchmark

## V6 — tropical cyclone

分兩級：

1. seeded-vortex idealized test：只驗證已有 vortex 時，模式能否維持合理暖心、眼牆、低層入流、高層外流；不能拿它證明自然生成能力。
2. spontaneous-genesis test：在暖海面、充足水汽、弱垂直風切並允許內部擾動的理想化熱帶環境中，禁止直接指定成熟 cyclone circulation，觀察是否能由擾動發展出暖心系統。

診斷至少包含 minimum SLP、maximum tangential wind、warm-core anomaly、radius-height circulation、surface enthalpy flux、precipitation、angular-momentum budget。

## V7 — land / seasonal / monsoon

- sea-breeze idealized case
- land thermal-inertia day/night cycle
- seasonal-insolation geometry
- idealized continent/ocean monsoon reversal
- topography sensitivity

## Continuous diagnostics

simulation 執行時必須可取得：
- CFL maxima by mode
- min/max `rho,p,T,q_j`
- NaN/Inf flag
- dry-mass drift
- water-mass drift
- energy diagnostic drift
- max `|w|` / wind speed
- minimum layer thickness / Jacobian

遇到 NaN、負密度、負壓力、嚴重負水物質或 layer inversion 時，立即 pause 並保存 offending cell/step；禁止 clamp 後繼續裝作正常。

## Reference suites

- FV3 idealized tests: https://www.gfdl.noaa.gov/fv3/fv3-idealized-tests/
- CESM Held–Suarez dry core: https://www.cesm.ucar.edu/models/simple/dry-dynamical-core
- CESM moist Held–Suarez: https://www.cesm.ucar.edu/models/simple/moist-held-suarez
- CM1: https://www2.mmm.ucar.edu/people/bryan/cm1/