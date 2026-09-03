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

以下門檻在真機結果產生前已先鎖定；不依結果事後放寬。

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

## 真機結果 / Real-device result — PASS

Windows + Chrome 真機執行 1000-step validation：**PASS**。

| Step | GPU mass drift | GPU max `|w|` (m/s) | `rhoD` rel. L2 | `rhoThetaM` rel. L2 |
|---:|---:|---:|---:|---:|
| 1 | `0.000e+0` | `6.843e-6` | `2.033e-8` | `2.848e-8` |
| 10 | `-6.751e-9` | `5.893e-5` | `4.519e-8` | `2.848e-8` |
| 100 | `-1.260e-7` | `2.513e-4` | `7.522e-7` | `1.142e-7` |
| 250 | `6.664e-8` | `3.496e-4` | `1.019e-6` | `2.580e-7` |
| 500 | `1.649e-7` | `3.107e-4` | `1.316e-6` | `3.931e-7` |
| 1000 | `4.341e-7` | `9.828e-4` | `1.605e-6` | `8.172e-7` |

1000-step final checkpoint：

- GPU dry-mass drift = `4.341e-7` ≤ `1e-6` → PASS。
- GPU max `|w| = 9.828e-4 m/s` ≤ `1e-3 m/s` → PASS。
- `rhoD` CPU/GPU relative L2 = `1.605e-6` ≤ `2e-5` → PASS。
- `rhoThetaM` CPU/GPU relative L2 = `8.172e-7` ≤ `2e-5` → PASS。
- max `|Δu| = 0` ≤ `1e-4 m/s` → PASS。
- max `|Δw| = 9.828e-4 m/s` ≤ `1e-3 m/s` → PASS。
- 全程未觸發 NaN、負密度或負壓力 validation failure。
- 1000 步共 250 s 模擬時間，真機驗證耗時約 `5.28 s`。

## 判讀 / Interpretation

Stage 3 的多步 GPU hydrostatic-rest、守恆與 CPU/GPU agreement gate **正式通過**。Float32 GPU 解在 1000 步內沒有出現爆炸、質量失控或 CPU/GPU 場量快速發散。

需要持續監控的一點：1000-step `max |w| = 9.828e-4 m/s` 已達目前 `1e-3 m/s` gate 的約 98.3%。它仍符合事前門檻，因此 Stage 3 可以封關；但 Stage 4 加入旋轉與長時間積分後，hydrostatic residual / balance error 必須繼續列入 regression diagnostics，若繼續成長則需改善 reference-state、Float32 conditioning 或時間積分，而不能把它當成真實環流。

## 狀態 / Status

**PASS — Stage 3 final GPU gate complete. Stage 3 is closed and Stage 4 may begin.**
