# Stage 3 GPU 多步靜力／CPU-GPU 一致性驗證 / GPU Multi-step Hydrostatic & CPU-GPU Agreement Validation

## 目的 / Purpose

Stage 3 的 one-step WebGPU smoke 已在 Windows + Chrome 真機通過，但 one-step 只能證明 pipeline 能建立、執行與 readback，不能證明 Float32 GPU 解在長時間積分下不會累積假垂直風或守恆漂移。

The Stage 3 one-step WebGPU smoke passed on a real Windows + Chrome device, but a one-step smoke only proves pipeline creation, execution, and readback. It does not prove that the Float32 GPU solution remains bounded over many timesteps.

## 測試案例 / Test case

- 初始條件：完全靜止、等溫、靜力平衡乾大氣。
- 全球網格：Stage 3 debug grid，`6 × 8 × 8 × 32`。
- GPU：Float32 WebGPU dry core。
- CPU reference：同一個 Stage 3 solver 的 Float64 路徑。
- timestep：`dt = 0.25 s`。
- 總步數：`1000`，即 250 s 模擬時間。
- checkpoints：`1, 10, 100, 250, 500, 1000` steps。

The GPU and CPU start from the same hydrostatic state and advance independently. The test does not use the interactive CPU Run button.

## 每個 checkpoint 的量測 / Metrics at every checkpoint

1. GPU dry-air mass drift relative to its own Float32 initial state.
2. CPU dry-air mass drift relative to the Float64 initial state.
3. GPU / CPU `max |w|`.
4. `rhoD` CPU-vs-GPU relative L2 error.
5. `rhoThetaM` CPU-vs-GPU relative L2 error.
6. maximum absolute `uEdge` difference.
7. maximum absolute `wInterface` difference.
8. minimum GPU density and pressure.
9. NaN / Inf / non-positive density / pressure flag.

GPU mass drift uses the uploaded-and-read-back Float32 initial state as its baseline, so the test measures time-integration drift rather than counting one-time Float64→Float32 quantization as conservation loss.

## Stage 3 gate thresholds / 門檻

第一版 gate 使用下列明確門檻；若真機結果失敗，先分析誤差來源，不以放寬門檻掩蓋 solver 問題。

| Metric | Threshold |
|---|---:|
| `|GPU dry mass drift|` | `≤ 1e-6` |
| GPU `max |w|` in resting atmosphere | `≤ 1e-3 m/s` |
| `rhoD` CPU/GPU relative L2 | `≤ 2e-5` |
| `rhoThetaM` CPU/GPU relative L2 | `≤ 2e-5` |
| max `|Δu|` | `≤ 1e-4 m/s` |
| max `|Δw|` | `≤ 1e-3 m/s` |
| density / pressure / NaN validity | must remain valid |

這些門檻屬 Stage 3 debug-grid correctness gate，不是未來 production weather accuracy 的最終誤差規範。

These thresholds are correctness gates for the Stage 3 debug grid, not final production-weather accuracy requirements.

## UI / 執行方式

更新本機後：

```bash
npm test
npm run serve
```

開啟 `http://127.0.0.1:5173`。one-step WebGPU smoke 通過後，按：

`執行 1000 步驗證 / Run 1000-step validation`

頁面會顯示 checkpoint 數值與最終 `通過 / PASS` 或 `失敗 / FAIL`；完整失敗原因同時寫入中英雙語 Log。

## 狀態 / Status

驗證 harness 已實作；仍需在真實 WebGPU 裝置執行。只有真機 1000-step gate 通過後，Stage 3 才正式封關並進入 Stage 4。

The validation harness is implemented and awaits a real-device run. Stage 3 closes only after the real-device 1000-step gate passes.
