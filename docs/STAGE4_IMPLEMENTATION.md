# Stage 4 — 旋轉全球乾大氣 / Rotating Global Dry Atmosphere

狀態：**實作中，尚未封關。** Stage 3 CPU regressions 目前為 8 項；本次補齊三維動量輸送後，Stage 4 CPU regressions 預期為 11 項，需由使用者本機重新執行 `npm test` 驗證。WebGPU smoke、GPU/CPU agreement 與 30-day Held–Suarez gate 也必須重跑。

## 1. 本階段範圍

Stage 4 在 Stage 3 fully-compressible / nonhydrostatic 3-D dry core 上加入：

- Earth rotation / traditional Coriolis reference mode；
- cubed-sphere east/north tangent basis；
- C-grid edge-normal velocity ↔ cell horizontal wind reconstruction；
- 三維動量輸送：horizontal wind 的 horizontal/vertical transport，以及 prognostic `w` 的 horizontal/vertical advection；
- Held–Suarez Newtonian thermal relaxation；
- near-surface Rayleigh drag；
- HEVI vertical acoustic off-centering；
- horizontal acoustic-divergence damping；
- HEVI-integrated model-top implicit Rayleigh absorbing layer；
- zonal-mean wind / temperature / overturning diagnostics；
- WebGPU rotating-core path 與 real-device validation harness。

本階段仍無水汽、雲、微物理、地形、海氣交互作用；沒有 hard-coded jet、Hadley cell、Rossby wave 或 cyclone generator。

## 2. Rotation / Coriolis

每個 cubed-sphere cell 由 radial vector 建立局地 east/north tangent basis。C-grid edge-normal velocity 重建為 cell horizontal vector，更新後再投影回 shared-edge normal velocity。

Stage 4 reference path 使用 `f = 2 Ω sin(phi)`；Coriolis half-step 用解析旋轉矩陣，不以 forward Euler 人為改變 inertial-oscillation 振幅。

## 3. 三維動量輸送

Physics spec 鎖定的連續動量方程含完整 `div(rho u⊗u)`，而 `w` 是真正 prognostic vertical velocity。檢查 long-run instability 時發現舊 Stage 4 實作只有 horizontal wind 的部分 transport，存在兩個結構缺口：

1. `w` 沒有 `u_h·grad_h(w)` 與 `w dw/dz` advection；
2. horizontal momentum 的 vertical carrier 直接重用了 scalar core 的 `vMassFlux`，但該 scalar flux 因 HEVI reference-state splitting 只儲存 `(rho-rho0)w` perturbation flux，不能代表完整的 vertical momentum carrier。

本版修正：

- `w` 新增 first-order donor-cell horizontal + vertical advective transport；
- horizontal advection of `w` 以 w-interface 上左右 layer 的 edge-normal wind 平均為 advecting velocity；
- vertical advection of `w` 依 `w` 符號選 lower/upper donor；
- horizontal momentum 的 vertical transport 改以 full `rho*w*area` carrier 搬運 upwind horizontal cell wind；
- CPU/GPU 都以同一個 pre-advection velocity state 計算 horizontal / vertical momentum tendencies，再提交新 `u` / `w`，避免 operator ordering 分岔；
- GPU 新 `wAdvect` pass 使用 7 個 storage buffers，仍低於既有 8-buffer baseline。

目前仍是 correctness-grade first-order donor-cell transport；production high-order monotone reconstruction 尚未完成。

## 4. Held–Suarez forcing

主要參數：`T0=315 K`, `ΔTy=60 K`, `Δθz=10 K`, `Tmin=200 K`, `sigma_b=0.7`, free-atmosphere thermal relaxation `1/40 day`, near-surface thermal relaxation `1/4 day`, surface drag `1/day`。初始狀態只加入約 `0.05 K` 平滑 zonal-wave perturbation 作 symmetry breaker。

## 5. HEVI vertical acoustic treatment

Stage 3 centered reference：`epsilon=0 -> theta=0.5`。

Stage 4：`epsilon=0.10 -> theta=0.55`。

new-time acoustic coupling coefficient 為 `theta^2 dt^2`，old-time coupling 為 `theta(1-theta) dt^2`。vertical `rho` / `rhoTheta` flux 使用同一 theta。

### 5.1 被否決的 buoyancy-in-HEVI 實驗

曾根據 upper-absorber 診斷，把 old-time density buoyancy 直接加入 linear HEVI acoustic RHS。CPU/GPU short agreement 雖 PASS，但真機 30-day gate 在 **day 0.25** 即出現：

- global `max|w| = 43.78 m/s`；
- below-absorber `max|w| = 43.07 m/s`；
- peak location `30.20 km, -5.6°`；
- vertical CFL `0.374`。

相較前一版 day-2 `14.10 m/s`，這是明確惡化。因此該 coupling **已完整回退**；目前恢復已驗證過的配置：HEVI pressure/acoustic solve 後再做 explicit slow buoyancy forcing。若未來要重新把 buoyancy 納入 implicit gravity-wave solve，必須重新推導一致離散式，不能直接把 old-density buoyancy 塞入 acoustic RHS。

## 6. Horizontal acoustic-divergence damping

只使用 `D = div_h(u)`，避免把 vertical divergence 經巨大 horizontal/vertical aspect ratio 耦合進 horizontal correction。

filter strength 以物理時間正規化：歷史 reference 為 `100 s` 作用一次 `gamma=0.1`；`dt=10 s` 時 `gamma≈0.01048`。

## 7. Model-top absorbing layer：HEVI implicit Rayleigh

舊 post-step `w` sponge 已淘汰。現在 HEVI tridiagonal acoustic solve 得到 `w_tilde` 後，在形成 new-time vertical flux 前套用：

`w_new = w_tilde / (1 + tau(z) dt)`

配置：

- absorber start = `0.75 H_top`；
- `sin^2` ramp；
- peak `tau = 0.2 s^-1`；
- long-run grid `H_top=40 km`, `Nz=48`，absorber 約 `30–40 km`；
- active interior interfaces 必須 `>=6`；
- Stage 4 不再另跑 post-step sponge；Stage 3 無 absorber。

真機比較：舊 post-step sponge 在 `40 km × 48` 於 day 2 得 `18.95 m/s @ 38.47 km`；implicit HEVI absorber 降為 `14.10 m/s @ 38.47 km`，證明 absorber 有作用但不是全部根因。

## 8. Long-run development grid

目前 30-day gate：

- cubed-sphere `N=8` → 384 horizontal columns；
- `Nz=48`；
- `H_top=40 km`；
- 18,432 3-D cells；
- `dt=10 s`；
- checkpoint 每 `0.25 day`。

這符合 Stage 2 第一版全球 dry-core 垂直候選範圍（40 km、48–72 layers）；水平 N=8 仍只是 development resolution。

## 9. CPU regressions

本版 Stage 4 預期 11 項：

1. cubed-sphere solid-body wind reconstruction；
2. inertial oscillation；
3. spherical geostrophic balance；
4. time-normalized divergence-damping cadence；
5. grid-scale horizontal divergent-noise damping；
6. vertical motion 不得被 horizontal filter 轉成 horizontal wind；
7. implicit HEVI top absorber profile / rate；
8. locally uniform interior `w` 經 advection 不得被改變；
9. vertical `w` donor-cell direction/value；
10. rotating hydrostatic rest；
11. one-day Held–Suarez sanity。

**本文件不宣稱本版已通過；需本機 `npm test` 實測。**

## 10. Real-device Gate A — GPU/CPU short-term agreement

threshold 維持：GPU dry-mass drift `<=2e-6`；`rhoD` / `rhoThetaM` relative L2 `<=1e-4`；max `|Δu|<=0.05 m/s`；max `|Δw|<=0.02 m/s`；invalid state forbidden。

本版 CPU/GPU momentum transport 都有實質改動，agreement 必須重跑。

## 11. Real-device Gate B — 30-day Held–Suarez development

每 0.25 日仍診斷 mass drift、global/below/inside-absorber `max|w|`、peak location、max edge wind、horizontal divergence RMS、horizontal/vertical CFL、upper-midlatitude westerly、tropical low-level zonal wind、overturning streamfunction。

Gates 不放寬：`|mass drift|<=5e-5`、upper-midlatitude max westerly `>0.5 m/s`、tropical low-level zonal wind `<0`、max overturning `>1e9 kg/s`、NH/SH dominant overturning signs opposite、`max|w|<10 m/s` throughout、no invalid state。

若完整三維 momentum transport 後仍只在 model top 快速失穩且 CFL 很低，下一個結構性工作將是 Stage 2 已鎖定但尚未完成的 **outer RK3 + split acoustic substeps**，而不是繼續提高 Rayleigh peak 或放寬 gate。

## 12. 封關條件

Stage 4 只有以下全部通過才 COMPLETE：Stage 3 regressions、Stage 4 CPU regressions、WebGPU smoke、GPU/CPU agreement、30-day Held–Suarez development gate。任何 long-run instability 都先修 numerical core；不提高 `max|w|` gate、不 clamp state、不 hard-code circulation。
