# Stage 4 — 旋轉全球乾大氣 / Rotating Global Dry Atmosphere

狀態：**實作中，尚未封關。** Stage 3 CPU regressions 目前為 8 項；Stage 4 CPU regressions 在本次 pressure+buoyancy HEVI coupling 修改後應為 10 項，需由本機重新執行 `npm test` 驗證。real-device WebGPU smoke、GPU/CPU agreement 與 30-day Held–Suarez gate 都必須在這一版重新跑。

## 1. 本階段範圍

Stage 4 在 Stage 3 fully-compressible / nonhydrostatic 3-D dry core 上加入：

- Earth rotation / traditional Coriolis reference mode；
- cubed-sphere 局地 east/north tangent basis；
- C-grid edge-normal velocity ↔ cell horizontal wind reconstruction；
- 與 mass flux 共用通量的 3-D horizontal momentum transport；
- Held–Suarez Newtonian thermal relaxation；
- near-surface Rayleigh drag；
- HEVI vertical acoustic off-centering；
- pressure + buoyancy coupled HEVI vertical-momentum RHS；
- horizontal acoustic-divergence damping；
- HEVI-integrated model-top implicit Rayleigh absorbing layer；
- zonal-mean wind / temperature / overturning diagnostics；
- WebGPU rotating-core path 與 real-device validation harness。

本階段仍無水汽、雲、微物理、地形、海氣交互作用；沒有任何 hard-coded jet、Hadley cell、Rossby wave 或 cyclone generator。

## 2. Rotation / Coriolis

每個 cubed-sphere cell 由 radial vector 建立局地 east/north tangent basis。C-grid edge-normal velocity 重建為 cell horizontal vector，更新後再投影回 shared-edge normal velocity。

Stage 4 reference path 使用：

`f = 2 Ω sin(phi)`

Coriolis half-step 使用解析旋轉，不以 forward Euler 人為改變 inertial-oscillation 振幅。

## 3. 三維水平動量輸送

horizontal mass flux 攜帶 upwind cell horizontal momentum；vertical perturbation mass flux 同樣攜帶 horizontal momentum。momentum divergence 與 mass divergence 共用 canonical faces。

目前 correctness core 仍使用 first-order donor-cell transport；production high-order monotone reconstruction 尚未完成，因此 Stage 4 development circulation 不能當成最終 climatology 精度。

## 4. Held–Suarez forcing

主要參數：

- `T0 = 315 K`
- `ΔTy = 60 K`
- `Δθz = 10 K`
- `Tmin = 200 K`
- `sigma_b = 0.7`
- free-atmosphere thermal relaxation `1/40 day`
- near-surface thermal relaxation `1/4 day`
- surface drag `1/day`

potential temperature 使用 exponential Newtonian relaxation；surface drag 亦使用 exponential decay。初始狀態只加入約 `0.05 K` 平滑 zonal-wave perturbation 作 symmetry breaker。

## 5. HEVI vertical acoustic / vertical-momentum treatment

Stage 3 centered reference：

`epsilon = 0` → `theta = 0.5`

Stage 4：

`epsilon = 0.10` → `theta = 0.55`

new-time acoustic coupling coefficient 為 `theta^2 dt^2`，old-time coupling 為 `theta(1-theta) dt^2`。vertical `rho` / `rhoTheta` flux 使用同一 theta，維持 conservative column flux-divergence form。

### 5.1 pressure + buoyancy 共用 vertical-momentum RHS

早期 dry core 的 operator split 是：

`HEVI pressure/acoustic -> explicit buoyancy -> vertical transport`

這在加入 HEVI-integrated upper absorber 後產生明確漏洞：Rayleigh absorber 先處理完 `w`，隨後 buoyancy pass 又可直接把垂直動量加回去，繞過 absorber。

現改為在進 HEVI 前由 interface density anomaly 計算：

`b_i = -g (rho_i-rho0_i)/rho_i`

並把它直接加入 HEVI RHS：

`RHS_i = w_old - dt * grad(p')/rho + dt * b_i + old-time acoustic coupling`

因此 pressure-gradient、buoyancy、acoustic coupling 與 model-top Rayleigh absorption 都在同一條 vertical-momentum operator path。獨立 CPU buoyancy update 與 GPU `buoyancy` pipeline 已移除。

Stage 3 standalone acoustic tests 呼叫 `heviColumnStep` 時不傳 vertical acceleration，因此 centered acoustic reference benchmark 本身不受 buoyancy forcing 污染；實際 `DryCoreCpu` / GPU dry core 則會提供 buoyancy forcing。

## 6. Horizontal acoustic-divergence damping

目前只使用 horizontal divergence：

`D = div_h(u)`

`u_e <- u_e + gamma(dt) d_e (D_R - D_L)`

不再把 `(1/rho0)d(rho0 w)/dz` 送入 horizontal filter，避免全球網格巨大 horizontal/vertical aspect ratio 將 Float32 `w` 差異放大成 spurious horizontal wind。

filter strength 以物理時間正規化：歷史 reference 為 `100 s` 作用一次 `gamma=0.1`；任意 timestep 使用等效 exponential cadence。`dt=10 s` 時約 `gamma=0.01048`。

## 7. Model-top absorbing layer：HEVI implicit Rayleigh

### 7.1 為何淘汰舊 post-step sponge

早期 Stage 4 使用完整 timestep 結束後的：

`w <- w * exp(-tau(z) dt)`

peak `tau = 1/600 s^-1`。真機定位顯示：

- `30 km × 20` 版長期最大 `w` 長時間集中在上層，day 14.75 才污染到 20.28 km；
- 將垂直域提升到 `40 km × 48`、令 absorber 有約 7 個 active interfaces 後，舊 post-step sponge 在 day 2 就於 `38.47 km` 出現 `18.95 m/s`；
- 同時 below-absorber max `|w|` 只有 `1.82 m/s`、vertical CFL `0.128`、horizontal CFL `6.2e-5`。

這證明問題不是單純 absorber 層數不足，而是舊 formulation 對 artificial rigid top reflection 太弱。

### 7.2 implicit formulation

Stage 4 改採 HEVI-integrated implicit Rayleigh absorber。HEVI tridiagonal solve 得到 `w_tilde` 後，在形成 new-time vertical flux 前：

`w_new = w_tilde / (1 + tau(z) dt)`

其中：

- absorber start = `0.75 H_top`；
- profile = `sin^2` ramp；
- peak `tau = 0.2 s^-1`；
- production long-run grid `H_top=40 km`, `Nz=48`，所以 absorber 約 `30–40 km`；
- active interior absorber interfaces 必須 `>=6`，否則 long-run gate 拒絕啟動；
- Stage 4 不再另外執行 post-step GPU/CPU `w` sponge；
- Stage 3 無 absorber，reference benchmarks 不變。

首次真機 implicit-absorber run 將 day-2 peak 從 `18.95` 降到 `14.10 m/s`，但爆點仍在 `38.47 km`，below-absorber 仍約 `1.82 m/s`。這證明 absorber 有效但當時仍有 post-HEVI buoyancy bypass，因此才進一步完成 §5.1 的 pressure+buoyancy coupling。

## 8. Long-run development grid

目前 30-day gate：

- cubed-sphere `N=8` → 384 horizontal columns；
- `Nz=48`；
- `H_top=40 km`；
- 18,432 3-D cells；
- `dt=10 s`；
- checkpoint 每 0.25 simulated day。

這符合 Stage 2 原先的第一版全球 dry-core 垂直候選範圍（40 km、48–72 layers），但水平 N=8 仍只是 development resolution。

## 9. CPU regressions

`npm test` 先跑 Stage 3 tests，再跑 Stage 4 V2 tests。

Stage 4 本版預期 10 項：

1. solid-body wind reconstruction across cubed-sphere seams；
2. inertial oscillation amplitude/period；
3. discrete spherical geostrophic balance；
4. time-normalized divergence-damping cadence；
5. grid-scale horizontal divergent-noise damping；
6. vertical motion 不得被 horizontal filter 轉成 horizontal wind；
7. HEVI implicit top absorber 必須符合 `w_tilde/(1+tau dt)` 且 peak rate=`0.2 s^-1`；
8. 明確 vertical/buoyancy forcing 也必須先通過同一個 HEVI Rayleigh absorber，禁止 bypass；
9. rotating hydrostatic rest；
10. one-day Held–Suarez dry-circulation sanity。

**本文件不宣稱本版已 10/10；需使用者本機 `npm test` 實測。**

## 10. Real-device Gate A — GPU/CPU short-term agreement

threshold：

| Metric | Gate |
|---|---:|
| GPU dry-mass drift | `<= 2e-6` |
| `rhoD` CPU/GPU relative L2 | `<= 1e-4` |
| `rhoThetaM` CPU/GPU relative L2 | `<= 1e-4` |
| max `|Δu|` | `<= 0.05 m/s` |
| max `|Δw|` | `<= 0.02 m/s` |
| invalid state | forbidden |

CPU 與 WGSL 的 vertical-momentum operator 都已改動，因此必須重新跑 agreement gate。

## 11. Real-device Gate B — 30-day Held–Suarez development

每 0.25 日診斷：

- dry-mass drift；
- global max `|w|`；
- max `|w|` below / inside absorber；
- max-`w` altitude / latitude；
- max edge wind；
- horizontal divergence RMS；
- horizontal / vertical advective CFL；
- upper-midlatitude westerly；
- tropical low-level mean zonal wind；
- overturning streamfunction 與 hemispheric sign。

Gates 不放寬：

- `|mass drift| <= 5e-5`
- upper-midlatitude max westerly `> 0.5 m/s`
- tropical low-level mean zonal wind `< 0`
- max overturning `> 1e9 kg/s`
- NH / SH dominant overturning signs opposite
- `max |w| < 10 m/s` throughout development run
- no invalid / NaN / non-positive density or pressure

此 30-day run 只是 development gate。正式 Held–Suarez climatology 仍需要更高水平解析度與更長 spin-up / averaging。

## 12. 封關條件

Stage 4 只有以下全部通過才 COMPLETE：

1. Stage 3 regressions PASS；
2. Stage 4 CPU V2 regressions PASS；
3. rotating WebGPU smoke（含 buoyancy-coupled HEVI + implicit absorber）PASS；
4. GPU/CPU short-term agreement PASS；
5. 30-day Held–Suarez development gate PASS。

任何 long-run instability 都先修 numerical core；不提高 `max|w|` gate、不 clamp state、不 hard-code circulation。
