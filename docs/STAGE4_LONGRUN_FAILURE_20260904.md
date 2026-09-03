# Stage 4 長時間 Held–Suarez 真機失敗與修正紀錄 / Long-run real-device failures and corrections

日期 / Date: 2026-09-04

## Gate A：短期 GPU/CPU 一致性 / Short-term agreement

真機 PASS（500 steps）：

- GPU mass drift `6.138e-8`（加入 top sponge 後重跑為 `3.588e-8`）；
- `rhoD` relative L2 約 `1.4e-5`；
- `rhoThetaM` relative L2 約 `3.9e-6`；
- max `|Δu|` 約 `2.6e-3 m/s`；
- max `|Δw|` 約 `2.3e-3 m/s`，top-sponge 版約 `1.68e-3 m/s`。

因此目前沒有證據顯示 Stage 4 GPU WGSL 與 CPU Float64 在短期內有符號、投影或 forcing 實作分歧。

## Failure A：原始 `dt=20 s`，第 4 日 NaN

| Day | mass drift | upper-midlatitude westerly | tropical low-level zonal wind | max overturning | max `|w|` |
|---:|---:|---:|---:|---:|---:|
| 1 | `2.885e-7` | `2.068 m/s` | `-0.491 m/s` | `5.119e10 kg/s` | `0.6785 m/s` |
| 2 | `2.352e-7` | `3.608 m/s` | `-0.255 m/s` | `2.576e10 kg/s` | `2.644 m/s` |
| 3 | `3.601e-7` | `4.364 m/s` | `-0.304 m/s` | `3.478e10 kg/s` | `7.208 m/s` |
| 4 | `NaN` | `NaN` | `NaN` | `NaN` | `NaN` |

修正 A：補上 30 km rigid model top 的 Rayleigh sponge，作用於上方 25% 高度且只阻尼 `w`；長期 reference timestep 改為 `10 s`；checkpoint 改成每 0.25 日。

## Failure B：`dt=10 s` + top sponge，第 9.5 日 stability guard

第二次真機測試沒有 NaN，但 `max |w|` 仍持續長大並在 day 9.5 超過事先設定的 `10 m/s` stability guard：

- day 0.25: `max|w| = 0.1004 m/s`
- day 1.00: `0.2952 m/s`
- day 2.00: `0.8485 m/s`
- day 3.00: `2.680 m/s`
- day 4.00: `4.835 m/s`
- day 6.00: `7.284 m/s`
- day 8.00: `9.050 m/s`
- day 9.25: `9.932 m/s`
- day 9.50: `10.049 m/s` → FAIL

同期間：

- mass drift 仍只有約 `10^-6`；
- upper-midlatitude westerly 已發展至約 `6.6 m/s`；
- tropical low-level zonal wind 持續為 easterly（約 `-0.47 m/s`）；
- overturning 約 `10^10 kg/s`。

這表示 model-top reflection 確實是第一個問題的一部分，但 top sponge 並不足以處理整個可壓縮快模態噪音。不能用加強整層 Rayleigh drag、clamp `w` 或把 10 m/s gate 往上調來掩蓋它。

## 修正 B：3-D acoustic divergence damping

Stage 2 的數值規格原先就保留 numerical divergence / acoustic filtering。現在正式加入：

- CPU: `src/physics/acousticDivergenceDamping.ts`
- WebGPU: `src/gpu/acousticDivergenceDampingGpu.ts`
- dimensionless coefficient 固定 `gamma_d = 0.1`
- 使用與 cubed-sphere finite-volume transport 相同 canonical shared-edge geometry；
- cell divergence 採 base-state-mass-weighted 3-D form：
  `div_h(u) + (1/rho0) d(rho0 w)/dz`
- filter 只修改 horizontal edge velocity 的 divergent component；
- 不修改 mass、`rhoTheta`，不做 velocity clamp；
- model-top sponge 繼續保留，因為兩者處理不同來源：sponge 處理人工頂界反射，divergence filter 處理可壓縮 acoustic divergence。

此方向與 MPAS / WRF 類 HEVI、split-explicit fully-compressible dynamical core 的常見 acoustic filtering 原理一致。

## Gate 不變 / Gates are not relaxed

長期 development gate 仍要求：

- `|mass drift| <= 5e-5`
- upper-midlatitude westerly `> 0.5 m/s`
- tropical low-level mean zonal wind `< 0`
- max overturning `> 1e9 kg/s`
- NH / SH dominant overturning signs opposite
- no invalid / NaN / non-positive density or pressure
- development-run numerical stability guard `max |w| < 10 m/s`

加入 filter 後先重跑 CPU regressions、GPU/CPU agreement，再重跑同一個 30-day gate。結果出來以前 Stage 4 仍未封關。
