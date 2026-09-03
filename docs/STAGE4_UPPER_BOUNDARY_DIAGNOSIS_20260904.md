# Stage 4 上界／sponge 定位診斷 / Upper-boundary and sponge diagnosis

日期 / Date: 2026-09-04

## 真機定位結果 A：30 km × 20 層

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

原 sponge 從 `0.75 H_top = 22.5 km` 開始；`22.46 km` 尚未進入 sponge，`30 km` 為剛性 `w=0` 邊界，所以真正有非零 damping 且仍是 interior `w` DOF 的主要 interface 只有約 `24.80` 與 `27.31 km`。

## 與既有數值規格的落差

`docs/NUMERICAL_CORE_SPEC.md` 已指定第一版全球 dry core 候選：

- `H_top = 40 km`;
- `Nz = 48–72`;
- model-top 上方約 `20–25%` depth 為 sponge layer。

因此長期 gate 先提升為 `N=8 × Nz=48`, `H_top=40 km`，其餘設定和 gates 不變；30–40 km 的上方 25% 吸收層約有 7 個 active interior interfaces，並新增 invariant：active absorber interfaces 必須 `>=6`。

## 真機定位結果 B：40 km × 48 層仍更快於 model top 失穩

提高垂直域與 absorber 解析度後，30-day gate 反而在 day 2 失敗：

- global `max |w| = 18.9506 m/s`;
- max-|w| location = `38.47 km`, latitude `-82.1 deg`;
- max `|w|` below absorber = `1.8200 m/s`;
- max `|w|` inside absorber = `18.9506 m/s`;
- max edge wind = `6.878 m/s`;
- horizontal divergence RMS = `1.909e-7 s^-1`;
- max horizontal CFL = `6.18e-5`;
- max vertical CFL = `0.128`;
- dry-mass drift = `-4.355e-6`.

這個結果非常關鍵：對流層／中層仍相對平穩，失穩幾乎完全侷限在 artificial model top 附近；也不是 advective CFL 問題。單純增加 sponge 層數沒有解決反射，表示瓶頸是 sponge formulation 本身，而不是只缺解析度。

## 根因：舊 absorber 是過弱的 post-step w damping

舊 Stage 4 treatment 在完整 timestep 結束後才對 `w` 做：

`w <- w * exp(-tau(z) dt)`

且 peak rate 只有：

`tau_max = 1/600 s^-1 ≈ 0.00167 s^-1`。

對 30–40 km 只有約 10 km 厚度的上界 buffer，這對快速垂直聲學／重力波反射過弱；而且它在 HEVI acoustic solve、new-time mass/thermodynamic flux 計算都完成之後才作用，並沒有真正進入垂直快模態求解。

WRF 技術文件與 Klemp et al. (2008) 對 nonhydrostatic split-explicit dynamics 推薦的是 implicit vertical-velocity Rayleigh absorber：在 vertically implicit acoustic solve 得到未阻尼 `w_tilde` 後、更新其餘垂直聲學變數之前，使用：

`w_new = w_tilde / (1 + tau(z) dt)`

其中 `tau(z)` 使用平滑的 sin² profile，典型 peak `gamma_r = 0.2 s^-1`。這種做法直接作用於 vertically propagating fast mode，且不需要 reference-state horizontal wind/temperature relaxation。

## 修正：HEVI 內生 implicit Rayleigh absorber

Stage 4 已改為：

- `N=8 × Nz=48`, `H_top=40 km` 保持；
- absorber 仍為 upper 25% (`30–40 km`)；
- `tau(z)` 使用原本 sin² ramp；
- peak rate 改為 `0.2 s^-1`；
- Rayleigh adjustment 直接放進 CPU 與 GPU HEVI vertical acoustic solve；
- damped new-time `w` 隨即用於 HEVI 的 new-time `rho` / `rhoTheta` vertical flux；
- 移除 Stage 4 production path 的 separate post-step `w` sponge，避免雙重阻尼；
- Stage 3 保持無 top absorber，原 HEVI benchmarks 不變；
- `dt=10 s`, HEVI `epsilon=0.10`, time-normalized horizontal divergence damping 與所有 gates 不變。

新增 CPU regression，直接比較同一個 HEVI column solve 在 absorber 開／關時的上層 interface `w`，要求 damped result 精確符合 `w_tilde/(1+tau dt)`，並鎖定 peak rate `0.2 s^-1`。

## 下一個判讀 / Next interpretation

下一輪先跑 CPU tests，再跑 GPU/CPU agreement。若 agreement PASS，才進 30-day gate。

若 implicit HEVI absorber 後 top-region `w` 被控制、但 30 km 以下仍在長時間發展中失穩，下一步轉向規格已預留的 scale-selective hyperdiffusion/filter 與 outer RK3/split-explicit acoustic integration。

若 top-region 仍快速增長，則要檢查目前 explicit buoyancy substep 與 HEVI absorber 的 operator splitting，因為 buoyancy 現在是在 HEVI 之後施加，可能需要把 slow vertical forcing 納入 damped vertical momentum update，而不是再加強 post-step damping。
