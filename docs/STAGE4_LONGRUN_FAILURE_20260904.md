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

責任分工：HEVI 負責 vertically propagating acoustic mode；model-top sponge 吸收 rigid-top 垂直波反射；horizontal divergence damping 只處理 horizontally propagating acoustic/divergent grid-scale mode。`w` 不再作為 horizontal filter 的直接輸入；不修改 mass、`rhoTheta`，不做 velocity clamp；`gamma_d = 0.1` 維持不變。

新增 regression：建立 `u=0`、`w!=0` 的狀態後執行 acoustic filter，要求 `u` 必須保持精確為 0，以防 vertical-to-horizontal aspect-ratio leakage 再次出現。GPU divergence pass 也移除 `w` 與 vertical metric storage buffers，使 filter 的 CPU/GPU 離散形式重新一一對應。

### 修正 C 真機結果：短期一致性重新 PASS

500-step GPU/CPU agreement：

- mass drift `9.172e-7`
- `rhoD` relative L2 `1.716e-5`
- `rhoThetaM` relative L2 `8.915e-6`
- max `|Δu| = 1.169e-3 m/s`
- max `|Δw| = 2.241e-3 m/s`

因此 3-D filter 引入的 CPU/GPU 分岔已被消除。

## Failure D：horizontal-only filter，N=4 × Nz=12 在 day 10.25 仍超過 `w` guard

同一個 `dt=10 s`、top sponge、horizontal acoustic filter 下，極粗長期網格仍在 day 10.25 超過 `max |w| < 10 m/s` gate：

- day 0.25: `0.0938 m/s`
- day 1.00: `0.3886 m/s`
- day 2.00: `1.714 m/s`
- day 3.00: `4.863 m/s`
- day 4.00: `5.956 m/s`
- day 6.00: `6.985 m/s`
- day 8.00: `6.549 m/s`
- day 9.00: `6.570 m/s`
- day 10.00: `9.301 m/s`
- day 10.25: `11.587 m/s` → FAIL

但大尺度方向仍合理：day 10.25 upper-midlatitude westerly `4.397 m/s`、tropical low-level zonal wind `-0.067 m/s`、overturning `4.417e10 kg/s`；mass drift `-1.844e-5` 仍在 `5e-5` gate 內。

這表示目前剩餘問題更集中於長時間垂直快模態／解析度依賴，而非 GPU/CPU 分歧或大尺度 circulation 完全缺失。

## 解析度敏感性實驗 / Resolution-sensitivity experiment

早期 long-run grid `N=4 × Nz=12` 僅有 96 個水平 column、1,152 個 3-D cells，適合 correctness smoke，但太粗，不應作為唯一長時間穩定性證據。

30-day development gate 在**不改變物理、damping coefficient、`dt=10 s` 或驗收門檻**的前提下提升為：

- cubed-sphere `N=8`：384 horizontal columns；
- `Nz=20`；
- 30 km model top；
- total 3-D cells `7,680`；
- zonal diagnostics 由 12 bins 提升到 24 bins。

### Failure E：提高解析度後反而在 day 4.25 更快失穩

N=8 × Nz=20 真機結果：

- day 0.25: mass `-1.578e-6`, jet `0.086 m/s`, trade `+0.004 m/s`, psi `1.633e10 kg/s`, `max|w|=0.1135 m/s`
- day 1.00: mass `-3.194e-6`, jet `0.793 m/s`, trade `-0.108 m/s`, psi `3.189e10 kg/s`, `max|w|=0.4196 m/s`
- day 2.00: mass `-3.621e-6`, jet `2.656 m/s`, trade `-0.260 m/s`, psi `8.950e10 kg/s`, `max|w|=2.255 m/s`
- day 3.00: mass `-2.904e-6`, jet `4.793 m/s`, trade `-0.326 m/s`, psi `2.347e11 kg/s`, `max|w|=4.699 m/s`
- day 4.00: mass `-2.772e-6`, jet `10.194 m/s`, trade `-0.336 m/s`, psi `4.841e11 kg/s`, `max|w|=8.938 m/s`
- day 4.25: mass `-2.810e-6`, jet `12.024 m/s`, trade `-0.333 m/s`, psi `5.554e11 kg/s`, `max|w|=20.071 m/s` → FAIL

解析度提高後 stability failure 從 N=4 的 day 10.25 提前到 day 4.25。mass drift 仍只有 `10^-6` 級，CPU/GPU 500-step agreement 仍 PASS，因此「只是 N=4 網格太粗」已被排除為主要解釋。較細的 vertical grid 反而讓未充分阻尼的垂直快模態／高垂直波數更容易被解析與成長，是目前更符合數據的方向。

## Attempt E：HEVI vertical time off-centering

Stage 4 現在採用可配置的 HEVI forward-centering：

`theta = 0.5 * (1 + epsilon)`

其中 Stage 3 reference 維持 `epsilon = 0`、`theta = 0.5`，保留 centered Crank-Nicolson acoustic benchmark；Stage 4 設定：

`epsilon = 0.1` → `theta = 0.55`

垂直 acoustic pair 同時使用同一個 `theta`：

- pressure-gradient implicit coupling 的新時間層權重由 `0.5` 改為 `theta`；
- vertical `rho` 與 `rhoTheta` base-state flux 更新改為 `(1-theta) F^n + theta F^(n+1)`；
- tridiagonal operator 的 new-time coupling 係數為 `theta^2 dt^2`；
- RHS old-time coupling 係數為 `theta(1-theta) dt^2`。

這保持垂直 mass / scalar flux divergence 的 conservative form，但讓 fast vertical acoustic mode 具有小幅數值耗散。沒有修改 Held–Suarez forcing、horizontal divergence damping、top sponge、`dt=10 s`、mass gate 或 `max|w| < 10 m/s` gate。

CPU 與 WebGPU 使用同一個共享 Stage 4 `epsilon` 設定，避免再次出現兩套離散式分岔。Stage 3 GPU 預設仍為 `epsilon=0`。

新增 regression：

1. 原 centered HEVI standing-wave phase/amplitude test 明確固定 `epsilon=0`；
2. 新增 acoustic-energy test，比較相同 standing mode 的 centered 與 `epsilon=0.1` HEVI，要求 centered energy 近似守恆、off-centered case 明顯衰減。

下一次真機驗證仍依序執行 `npm test`、Stage 4 GPU/CPU agreement、N=8 × Nz=20 30-day gate。若 short agreement 失敗，先修 CPU/GPU 離散一致性；若 agreement PASS 但 30-day 仍失穩，再根據 `w` growth curve 判斷是否需要調整垂直時間離散、timestep 或更完整的 split-explicit acoustic treatment，而不放寬物理 gate。

## Gate 不變 / Gates are not relaxed

長期 development gate 仍要求：

- `|mass drift| <= 5e-5`
- upper-midlatitude westerly `> 0.5 m/s`
- tropical low-level mean zonal wind `< 0`
- max overturning `> 1e9 kg/s`
- NH / SH dominant overturning signs opposite
- no invalid / NaN / non-positive density or pressure
- development-run numerical stability guard `max |w| < 10 m/s`

Stage 4 在新的 HEVI off-centering + N=8 × 20、30-day gate 通過以前仍不封關。
