# Stage 4 長時間 Held–Suarez 真機失敗與修正紀錄 / Long-run real-device failures and corrections

日期 / Date: 2026-09-04

## Gate A：短期 GPU/CPU 一致性 / Short-term agreement

最初 rotating-core 真機 PASS（500 steps）：

- GPU mass drift `6.138e-8`（加入 top sponge 後重跑為 `3.588e-8`）；
- `rhoD` relative L2 約 `1.4e-5`；
- `rhoThetaM` relative L2 約 `3.9e-6`；
- max `|Δu|` 約 `2.6e-3 m/s`；
- max `|Δw|` 約 `2.3e-3 m/s`，top-sponge 版約 `1.68e-3 m/s`。

因此原始 Stage 4 GPU WGSL 與 CPU Float64 在短期內沒有符號、投影或 forcing 實作分歧。

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

同期間：mass drift 仍只有約 `10^-6`；upper-midlatitude westerly 已發展至約 `6.6 m/s`；tropical low-level zonal wind 持續為 easterly（約 `-0.47 m/s`）；overturning 約 `10^10 kg/s`。

這表示 model-top reflection 確實是第一個問題的一部分，但 top sponge 並不足以處理整個可壓縮快模態噪音。不能用加強整層 Rayleigh drag、clamp `w` 或把 10 m/s gate 往上調來掩蓋它。

## Attempt B：初版 3-D acoustic divergence damping

依 Stage 2 原先保留的 acoustic/divergence stabilization，加入 coefficient `gamma_d = 0.1` 的 divergence filter。第一版 cell divergence 採：

`div_h(u) + (1/rho0) d(rho0 w)/dz`

並用此 divergence 的水平梯度修正 horizontal edge velocity。CPU regression 對人工水平 divergent noise 可降低 RMS，因此初步單元測試通過。

### Failure C：3-D filter 造成 GPU/CPU 快速分岔

真機 short-term agreement 立即暴露跨尺度問題：

| Step | max `|Δu|` | max `|Δw|` |
|---:|---:|---:|
| 1 | `5.891e-4 m/s` | `2.209e-5 m/s` |
| 10 | `6.159e-2 m/s` | `2.383e-4 m/s` |
| 100 | `1.469 m/s` | `2.836e-3 m/s` |
| 250 | `6.190 m/s` | `9.375e-3 m/s` |
| 500 | `178.1 m/s` | `0.8857 m/s` |

質量漂移在 step 500 仍只有 `-6.974e-9`，所以不是 mass conservation 爆掉；差異由 velocity operator 快速放大。

根因是全球粗網格具有巨大的 horizontal/vertical aspect ratio。CPU Float64 與 GPU Float32 在 HEVI 後只有約 `10^-5 m/s` 級的 `w` 差異，但 `(1/rho0)d(rho0 w)/dz` 被送入以水平 grid length 尺度化的 velocity correction，會把微小垂直差異直接放大成水平速度擾動，再透過動力核心回饋。這個離散形式對本專案的 anisotropic global grid 不合適。

## 修正 C：horizontal-divergence-only acoustic damping

Acoustic filter 改為只計算：

`D = div_h(u)`

並維持：

`u_edge <- u_edge + gamma_d * d_edge * (D_R - D_L)`

理由與責任分工：

- HEVI 負責 vertically propagating acoustic mode；
- model-top sponge 吸收人工 rigid-top 的垂直波反射；
- horizontal divergence damping 只處理 horizontally propagating acoustic/divergent grid-scale mode；
- vertical velocity `w` 不再作為 horizontal filter 的直接輸入；
- 不修改 mass、`rhoTheta`，不做 velocity clamp；
- `gamma_d = 0.1` 與所有 long-run physical gates 維持不變。

新增 regression：建立 `u=0`、`w!=0` 的狀態後執行 acoustic filter，要求 `u` 必須保持精確為 0，以防 vertical-to-horizontal aspect-ratio leakage 再次出現。

GPU divergence pass 也移除 `w` 與 vertical metric storage buffers，使 filter 的 CPU/GPU 離散形式重新一一對應。

## Gate 不變 / Gates are not relaxed

長期 development gate 仍要求：

- `|mass drift| <= 5e-5`
- upper-midlatitude westerly `> 0.5 m/s`
- tropical low-level mean zonal wind `< 0`
- max overturning `> 1e9 kg/s`
- NH / SH dominant overturning signs opposite
- no invalid / NaN / non-positive density or pressure
- development-run numerical stability guard `max |w| < 10 m/s`

修正後仍必須依序重跑 CPU regressions、GPU/CPU agreement，再重跑同一個 30-day gate。結果出來以前 Stage 4 仍未封關。
