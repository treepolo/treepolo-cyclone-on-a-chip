# Stage 4 上界／sponge 定位診斷 / Upper-boundary and sponge diagnosis

日期 / Date: 2026-09-04

## 真機定位結果 A：30 km × 20 / Real-device localization A

在 `N=8 × Nz=20`, `H_top=30 km`, `dt=10 s`, HEVI `epsilon=0.10`, time-normalized horizontal divergence damping 與既有 top sponge 下，30-day Held–Suarez gate 於 day 14.75 失敗：

- global `max |w| = 11.7864 m/s`;
- max-|w| location = `z=20.28 km`, latitude `82.1 deg`;
- max `|w|` below sponge = `11.7864 m/s`;
- max `|w|` in sponge = `8.3876 m/s`;
- max edge wind = `87.268 m/s`;
- horizontal divergence RMS = `5.043e-7 s^-1`;
- max horizontal CFL = `9.013e-4`;
- max vertical CFL = `6.404e-2`;
- dry-mass drift = `-3.068e-5`.

從約 day 1 到 day 14.5，全域最大垂直速度大多位於 `z≈27.31 km` 的 sponge 區；失敗 checkpoint 才跳到 `20.28 km`、高緯度，位於原 sponge 起點 `22.5 km` 下方。

因此目前證據不支持 advective CFL 崩潰，也不支持 horizontal divergence 突然爆炸。更符合資料的是模式頂／上層波動長時間累積與吸收層解析不足。

## 發現：30 km × 20 層的 sponge 數值上只有約兩個有效 interior interface

`buildStretchedVerticalGrid(20, 30000, 1.4)` 的上層 interface 約為：

- `20.28 km`
- `22.46 km`
- `24.80 km`
- `27.31 km`
- `30.00 km` model top

原 sponge 從 `0.75 H_top = 22.5 km` 開始；`22.46 km` 尚未進入 sponge，`30 km` 為剛性 `w=0` 邊界，所以真正有非零 damping 且仍是 interior `w` DOF 的主要 interface 只有約 `24.80` 與 `27.31 km`。這不足以構成良好解析的 absorbing layer。

## 與既有數值規格的落差

`docs/NUMERICAL_CORE_SPEC.md` 已指定第一版全球 dry core 候選：

- `H_top = 40 km`;
- `Nz = 48–72`;
- model-top 上方約 `20–25%` depth 為 sponge layer。

先前 long-run gate 的 `30 km × 20` 僅為早期低成本 development grid，現在已證明不足以作為上界穩定性驗收網格。

## 修正 A：回到 40 km × 48 層

長期 Stage 4 gate 改為：

- horizontal cubed-sphere `N=8` 不變；
- `Nz: 20 -> 48`;
- `H_top: 30 km -> 40 km`;
- stretch `1.4` 不變；
- absorber start fraction `0.75` 不變，因此上界吸收區為約 `30–40 km`；
- `dt=10 s` 不變；
- HEVI `epsilon=0.10` 不變；
- time-normalized horizontal divergence damping 不變；
- all physical and numerical pass/fail gates unchanged。

在 `Nz=48, H_top=40 km, stretch=1.4` 下，上方 25% 區域有約 7 個 active interior interfaces。驗證程式新增 invariant：active absorber interior interfaces 必須 `>= 6`，否則 long-run gate 拒絕啟動。

## 真機定位結果 B：40 km × 48 + 舊 post-step sponge

提高垂直域與層數後，舊 post-step sponge 反而在 day 2 更清楚地暴露上界問題：

- global `max |w| = 18.9506 m/s`;
- location `z=38.47 km`, latitude `-82.1 deg`;
- below-sponge `max |w| = 1.8200 m/s`;
- sponge `max |w| = 18.9506 m/s`;
- max edge wind `6.878 m/s`;
- horizontal divergence RMS `1.909e-7 s^-1`;
- horizontal CFL `6.18e-5`;
- vertical CFL `1.278e-1`;
- mass drift `-4.355e-6`.

這證明較高解析度本身不是解法；真正問題集中在 rigid top 附近的波吸收，而不是對流層或 CFL。

## 修正 B：把 Rayleigh absorber 移入 HEVI acoustic solve

舊路徑是在完整 timestep 後才對 `w` 做弱的 post-step damping，頂層 rate 約 `1/600 s^-1`。Stage 4 已改成在 HEVI 垂直 acoustic tridiagonal solve 之後、形成 new-time vertical mass/thermodynamic flux 之前套用 implicit Rayleigh：

`w_new = w_tilde / (1 + rate(z) * dt)`

配置：

- start `0.75 H_top = 30 km`;
- `sin^2` ramp；
- peak rate `0.2 s^-1`；
- old post-step production sponge removed；
- Stage 3 defaults to no Rayleigh profile。

## 真機定位結果 C：HEVI implicit absorber，但 buoyancy 仍 post-HEVI

新的 implicit HEVI absorber 通過 CPU/GPU agreement，但 30-day gate 仍在 day 2 失敗：

- global `max |w| = 14.0981 m/s`;
- location `z=38.47 km`, latitude `82.1 deg`;
- below-absorber `max |w| = 1.8202 m/s`;
- absorber `max |w| = 14.0981 m/s`;
- max edge wind `15.793 m/s`;
- horizontal divergence RMS `2.794e-7 s^-1`;
- horizontal CFL `1.418e-4`;
- vertical CFL `9.511e-2`;
- mass drift `-2.588e-6`.

相較舊 post-step sponge，day-2 peak 從 `18.95` 降到 `14.10 m/s`，證明 implicit absorber 有效但不完整。爆點仍固定在 `38.47 km`，而 absorber 以下仍只有 `1.82 m/s`，所以主要問題仍是上層垂直動量路徑。

程式檢查發現 operator split 為：

`HEVI pressure/acoustic + Rayleigh -> explicit buoyancy -> vertical transport`

也就是 HEVI absorber 先吸收一次後，獨立 buoyancy pass 又能直接把垂直加速度加回 `w`，這筆新垂直動量不再經過 Rayleigh absorber。

## 修正 C：pressure + buoyancy 共用 HEVI vertical-momentum RHS

Stage 3/4 dry core 現改為先由當下 interface density anomaly 計算 explicit buoyancy acceleration：

`b_i = -g (rho_i - rho0_i) / rho_i`

然後將它與 perturbation pressure-gradient 一起放進 HEVI vertical-momentum RHS：

`RHS_i = w_old - dt * grad(p')/rho + dt * b_i + acoustic_old_time_terms`

tridiagonal solve 完成後，再套用同一個 implicit Rayleigh profile，最後才以 damped `w_new` 形成 new-time `rho` / `rhoTheta` vertical flux。

因此新的單一垂直動量路徑為：

`pressure + buoyancy -> HEVI acoustic solve -> implicit Rayleigh absorber -> vertical mass/thermodynamic flux`

獨立 CPU buoyancy update 與 GPU `buoyancy` compute pipeline 已移除。這不是新增第二層 damping，而是消除會繞過 absorber 的 operator split。

新增 regression：給上層 interface 一個明確 vertical acceleration，比較 free HEVI 與 Rayleigh HEVI，要求 forcing 產生的 `w` 也必須滿足同一個 `1/(1+rate*dt)` 衰減，防止未來重新出現 buoyancy bypass。

## 下一個判讀 / Next interpretation

下一輪必須先重新通過 Stage 3 regressions、Stage 4 CPU regressions、WebGPU smoke 與 GPU/CPU short-term agreement，因為 CPU/GPU vertical momentum operator 都已改動。

若 30-day gate 中 `30–40 km` 的 `max |w|` 因此大幅降低，表示 buoyancy bypass 是上界能量注入的重要來源。

若仍然只在 absorber 內快速增長，下一步要檢查 Rayleigh damping 是否需真正進入 tridiagonal matrix coupling／上界 pressure treatment，而不是繼續提高 peak rate。

若上層穩定後才在 `30 km` 以下長時間失穩，則離開 upper-boundary 問題，進入原數值規格預留的 scale-selective hyperdiffusion/filter 與 outer RK3 / split-explicit acoustic integration。
