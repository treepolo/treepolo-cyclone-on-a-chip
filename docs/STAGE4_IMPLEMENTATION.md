# Stage 4 — 旋轉全球乾大氣 / Rotating Global Dry Atmosphere

狀態：**實作中，尚未封關。** CPU V2 reference path 已擴充至 6 項回歸；real-device rotating WebGPU smoke 與 short-term CPU/GPU agreement 已通過。30-day Held–Suarez development gate 目前正在處理長時間 acoustic/divergence instability，詳見 `STAGE4_LONGRUN_FAILURE_20260904.md`。

## 1. 本階段範圍

Stage 4 在 Stage 3 的三維 fully-compressible / nonhydrostatic dry core 上加入：

- Earth rotation / traditional Coriolis；
- cubed-sphere 局地 east/north tangent basis；
- C-grid edge-normal velocity ↔ cell horizontal vector wind reconstruction；
- 與 mass flux 共用通量的三維水平 momentum transport；
- Held–Suarez Newtonian thermal relaxation；
- near-surface Rayleigh drag；
- model-top absorbing sponge；
- 3-D acoustic divergence damping；
- zonal-mean temperature / zonal wind / meridional wind / overturning diagnostics；
- WebGPU rotating-core path 與真機 validation harness。

本階段仍無水汽、雲、微物理、地形、海氣交互作用，也沒有任何 hard-coded jet、Hadley cell、Rossby wave 或 cyclone generator。

## 2. Rotation / Coriolis

每個 cubed-sphere cell 由 unit radial vector 建立局地 east/north 正交 tangent basis。四條 C-grid edge-normal velocity 以最小平方法重建為 cell-centered `(u_east, v_north)`，更新後再投影回 shared-edge normal velocity。

Stage 4 reference path 使用 traditional shallow-atmosphere Coriolis：

`f = 2 Ω sin(phi)`

`du/dt = f v`

`dv/dt = -f u`

Coriolis 子步使用解析旋轉矩陣，不以 forward Euler 人為改變純 inertial-oscillation 振幅。

## 3. 三維水平動量輸送

Stage 4 沒有建立與 mass transport 脫鉤的獨立風場 advection：

- horizontal mass flux 攜帶 upwind cell 的 global 3-D tangent wind；
- vertical perturbation mass flux 同樣攜帶 horizontal momentum；
- momentum divergence 與 mass divergence 共用 canonical shared faces；
- 更新後的 global vector 投影回 local tangent plane 與 C-grid edge velocity。

目前 correctness core 仍使用 donor-cell / first-order upwind transport。Production high-order monotone transport 尚未完成，因此現階段不能把數值擴散程度當作最終天氣精度。

## 4. Held–Suarez forcing

核心 benchmark 參數：

- `T0 = 315 K`
- equator-to-pole contrast `ΔTy = 60 K`
- vertical stability term `Δθz = 10 K`
- `Tmin = 200 K`
- `sigma_b = 0.7`
- free-atmosphere thermal relaxation `1/40 day`
- near-surface thermal relaxation `1/4 day`
- surface Rayleigh drag `1/day`

`Teq = max(200 K, [315 - 60 sin²(phi) - 10 ln(sigma) cos²(phi)] sigma^kappa)`

Newtonian relaxation 作用於 potential temperature；surface drag 使用 exponential decay。初始狀態只加入最大約 `0.05 K` 的平滑 wave perturbation 作 symmetry breaker。

## 5. Model-top absorbing layer

30 km rigid model top 會反射非靜力重力／聲波，因此加入只作用在最上方 25% 高度的 vertical-velocity Rayleigh sponge：

- start = `0.75 H_top`
- `sin²` ramp
- top e-folding time = `600 s`
- 只修改 `w`
- 不修改 mass / pressure / thermodynamic variables

此 sponge 專門處理人工上界反射，不用來壓對流層 circulation。

## 6. Acoustic divergence damping

第二次 long-run 真機測試顯示：即使 `dt=10 s` 並有 top sponge，`max |w|` 仍從約 `0.1 m/s` 持續增長至 day 9.5 的 `10.05 m/s`。因此正式補上 HEVI/split-explicit compressible core 常用的 acoustic-divergence filter。

Stage 4 使用：

`D = div_h(u) + (1/rho0) d(rho0 w)/dz`

並在 canonical horizontal edge 上做：

`u_e <- u_e + gamma_d * d_e * (D_R - D_L)`

其中：

- `gamma_d = 0.1`，在看到下一次 30-day 結果前固定；
- `d_e` 是 edge 相鄰 cell-center distance；
- 只修改 horizontal velocity 的 divergent component；
- 不修改 mass / `rhoTheta`；
- 不做 `w` clamp 或 state normalization。

CPU 與 WebGPU 使用相同離散幾何。GPU 最重的 divergence pass 維持 8 個 storage buffers，符合本專案 WebGPU baseline target。

## 7. Reference timestep / long-run cadence

第一次 long-run 使用 `dt=20 s`，day 4 進入 NaN；因此 long-run reference timestep 已修正為：

- `dt = 10 s`
- 每 `0.25 simulated day` readback
- invalid state、`|mass drift| > 5e-5` 或 `max |w| >= 10 m/s` 立即提前 FAIL

降低 timestep 是根據長期穩定性證據修正 correctness configuration，並非放寬驗收。

## 8. CPU V2 regressions

`npm test` 會先跑 Stage 3 7/7，再跑 Stage 4 V2 tests。Stage 4 目前包含：

1. cubed-sphere solid-body wind reconstruction across seams；
2. inertial oscillation amplitude/period；
3. discrete spherical geostrophic balance；
4. acoustic-divergence filter 對 grid-scale divergent noise 的衰減；
5. rotating hydrostatic rest；
6. 1-day Held–Suarez dry-circulation sanity（`dt=10 s`）。

新增 acoustic filter 後需要由本機重新執行 `npm test`，確認 6/6 才進下一次 30-day 真機 gate。

## 9. Real-device Gate A — GPU/CPU short-term agreement

先前真機結果已 PASS。加入 acoustic filter 後必須重新跑一次，確保 CPU Float64 與 GPU Float32 的新 filter 一致。

原先鎖定 threshold 維持：

| Metric | Gate |
|---|---:|
| GPU dry-mass drift | `<= 2e-6` |
| `rhoD` CPU/GPU relative L2 | `<= 1e-4` |
| `rhoThetaM` CPU/GPU relative L2 | `<= 1e-4` |
| max `|Δu|` | `<= 0.05 m/s` |
| max `|Δw|` | `<= 0.02 m/s` |
| invalid state | forbidden |

## 10. Real-device Gate B — 30-day Held–Suarez development

Grid：`6 × 4 × 4 × 12`，top `30 km`，`dt=10 s`。

每 0.25 日檢查：

- dry-mass drift；
- max vertical velocity；
- upper-midlatitude maximum zonal westerly；
- tropical low-level mean zonal wind；
- meridional overturning streamfunction；
- NH / SH dominant overturning sign。

最終 development gates 不放寬：

- `|mass drift| <= 5e-5`
- upper-midlatitude max westerly `> 0.5 m/s`
- tropical low-level mean zonal wind `< 0`
- max overturning streamfunction `> 1e9 kg/s`
- NH / SH dominant overturning signs opposite
- `max |w| < 10 m/s` throughout development run
- no invalid state

此 30-day coarse-grid test 只作 Stage 4 development gate。正式 Held–Suarez climatology 仍需更高解析與數百日 spin-up / averaging，再與標準 zonal-mean climate 統計定量比較。

## 11. 封關條件

Stage 4 只有在以下全部通過後才標記 COMPLETE：

1. Stage 3 regressions — PASS；
2. Stage 4 CPU V2 6/6 — PASS；
3. rotating + acoustic filter + top-sponge WebGPU smoke — PASS；
4. GPU/CPU short-term agreement — PASS；
5. 30-day Held–Suarez development gate — PASS。

任何 long-run instability 都先修數值核心，不以提高 `max |w|` gate、clamp state 或強制指定 circulation 來取得 PASS。
