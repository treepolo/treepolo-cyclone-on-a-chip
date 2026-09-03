# Stage 4 上界／長期失穩定位診斷 / Upper-boundary and long-run instability diagnosis

日期 / Date: 2026-09-04

## A. 30 km × 20：長期上層累積

`N=8 × Nz=20`, `H_top=30 km`, `dt=10 s`, HEVI `epsilon=0.10`, time-normalized horizontal divergence damping 下，day 14.75 失敗：

- global `max|w| = 11.7864 m/s`；
- location `20.28 km, 82.1°`；
- below-sponge `11.7864 m/s`；sponge `8.3876 m/s`；
- horizontal CFL `9.013e-4`；vertical CFL `6.404e-2`；
- divergence RMS `5.043e-7 s^-1`；mass drift `-3.068e-5`。

前十多天 peak 大多在約 `27.31 km` 的 sponge 區，最後才向下污染。舊 `30 km × 20` stretched grid 的上方 25% sponge 實際只有約兩個主要 active interior interfaces，因此不足以判讀正式 upper-boundary stability。

## B. 40 km × 48 + 舊 post-step sponge

依 Stage 2 原規格改為 `H_top=40 km`, `Nz=48`，30–40 km 約有 7 個 active absorber interfaces；驗證程式要求至少 6 個。

舊 post-step sponge 在 day 2 失敗：

- global `max|w| = 18.9506 m/s`；
- location `38.47 km, -82.1°`；
- below absorber `1.8200 m/s`；absorber `18.9506 m/s`；
- hCFL `6.18e-5`；vCFL `0.1278`；
- divRMS `1.909e-7 s^-1`；mass drift `-4.355e-6`。

較高垂直解析度沒有自動解決問題，反而把 rigid-top 附近的波反射／累積暴露得更清楚。

## C. HEVI-integrated implicit Rayleigh absorber

舊 post-step `w` damping 改為 HEVI acoustic solve 內的 implicit Rayleigh：

`w_new = w_tilde / (1 + rate(z) dt)`

配置：start `0.75 H_top=30 km`、`sin^2` ramp、peak `0.2 s^-1`。舊 production post-step sponge 移除。

真機 day-2 結果：

- global `max|w| = 14.0981 m/s`；
- location `38.47 km, 82.1°`；
- below absorber `1.8202 m/s`；absorber `14.0981 m/s`；
- hCFL `1.418e-4`；vCFL `9.511e-2`；
- divRMS `2.794e-7 s^-1`；mass drift `-2.588e-6`。

相較 B，peak `18.95 -> 14.10 m/s`，所以 implicit absorber **有效但不是根因的全部**。

## D. 被否決的 pressure+buoyancy HEVI coupling

因為當時程式是 `HEVI + absorber -> explicit buoyancy`，曾假設 buoyancy 在 absorber 後重新注入 `w`。實驗把 old-time density buoyancy 直接塞入 linear HEVI vertical-momentum RHS，CPU/GPU short agreement 仍 PASS，但長期結果立即惡化：

- failure `day 0.25`；
- global `max|w| = 43.7810 m/s`；
- below absorber `43.0716 m/s`；absorber `43.7810 m/s`；
- location `30.20 km, -5.6°`；
- max edge wind `0.628 m/s`；
- divRMS `1.292e-7 s^-1`；
- hCFL `6.27e-6`；vCFL `0.374`；
- mass drift `-1.839e-6`。

因此「buoyancy bypass 是主要根因」這個假說被真機結果否決。該變更已完整回退；目前恢復 explicit slow buoyancy after HEVI。未來若要做 implicit gravity-wave/buoyancy coupling，必須從一致的離散方程重新推導，不能直接把 old-density buoyancy tendency 加進 acoustic RHS。

## E. 新發現：Stage 4 三維動量 transport 原本不完整

重新對照 `PHYSICS_SPEC.md` 的完整 3-D Euler momentum `div(rho u⊗u)` 與程式後，找到比 absorber tuning 更根本的缺項：

1. prognostic interface `w` 完全沒有 horizontal / vertical advective transport；
2. horizontal momentum 的 vertical transport 直接重用 scalar `vMassFlux`，但 scalar core 的 vertical outer flux 因 HEVI reference-state splitting 只包含 `(rho-rho0)w` perturbation part，不是完整 vertical momentum carrier。

這表示舊 Stage 4 並沒有真正完成規格要求的 3-D momentum transport。上傳重力波進入低密度高空後，缺少 `u_h·grad_h(w) + w dw/dz` 搬運／非線性調整，可能使固定格點的 `w` 不合理累積。

## F. 當前修正：完整三維 momentum transport

本版已：

- 回退 D 的 buoyancy-in-HEVI；
- 為 `w` 加入 donor-cell horizontal + vertical advection；
- horizontal `w` advection 使用 interface 上相鄰兩 layer edge wind 的平均；
- vertical `w` advection 依 `w` 符號選 lower/upper donor；
- horizontal momentum 的 vertical carrier 改用 full `rho*w*area`；
- CPU/GPU 都從同一個 pre-advection velocity state 計算 horizontal / vertical momentum tendencies，再一起提交；
- GPU `wAdvect` pass 維持在 8-storage-buffer baseline 以下；
- 新增 regression：uniform local `w` preservation 與 vertical upwind donor direction/value。

解析度、`dt=10 s`、HEVI `epsilon=0.10`、implicit Rayleigh peak `0.2 s^-1`、time-normalized horizontal divergence damping 與所有 long-run gates 均未放寬。

## 下一個判讀 / Next interpretation

本版必須先重新通過：Stage 3 CPU regressions、Stage 4 CPU regressions、WebGPU smoke、GPU/CPU short-term agreement，才可跑 30-day gate。

若完整 3-D momentum transport 明顯降低／延後 high-altitude `w` accumulation，代表長期失穩的重要來源確實是缺失的 nonlinear momentum transport。

若仍在 model top 附近快速失穩、absorber 以下保持安定且 CFL 很低，下一個結構性工作不再是調大 sponge，而是回到 Stage 2 已鎖定但尚未實作完整的 **outer RK3 + split-explicit acoustic substeps**；必要時再加入規格允許的 scale-selective hyperdiffusion/filter。
