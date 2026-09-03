# Stage 3 真實 WebGPU smoke 記錄 / Real-device WebGPU smoke log — 2026-09-03

## 第一次真機結果 / First real-device result

Windows Chrome 成功取得 WebGPU adapter，但 pipeline 建立失敗。錯誤指出 HEVI compute stage 使用 10 個 storage buffers，而 requestDevice 後的 device 使用 WebGPU baseline `maxStorageBuffersPerShaderStage = 8`。該實際 adapter 本身支援 16。

Windows Chrome successfully obtained a WebGPU adapter, but pipeline creation failed. The HEVI compute stage used 10 storage buffers while the requested device exposed the WebGPU baseline `maxStorageBuffersPerShaderStage = 8`. The physical adapter itself supports 16.

## 根因 / Root cause

Stage 3 第一版把垂直參考場 `zCenter`, `dz`, `refP`, `refRho`, `refX`, `refRhoI`, `refXI` 分散成太多獨立 storage bindings。這會超過 WebGPU baseline，而且即使只在桌機上 request 10 或 16，也會降低手機與較保守 adapter 的相容性。

The first Stage 3 GPU layout split the vertical reference state across too many independent storage bindings. Requesting 10 or 16 only on capable desktop adapters would hide the problem while reducing mobile/baseline compatibility.

## 修正 / Fix

不採用「直接向 adapter 要 10 個 binding」作為最終修法。GPU data layout 已重新打包：

The final fix does not simply request 10 bindings. GPU data is repacked instead:

- HEVI：10 → **5 storage buffers**。
- horizontal flux：兩個 scalar flux buffers 合併成一個 `vec2<f32>` flux buffer。
- vertical flux：同上。
- divergence：10 → **8 storage buffers**，符合 WebGPU baseline。
- layer reference fields 打包成 `layerRef = [z, dz, p0, rho0, rhoTheta0]`。
- interface reference fields 打包成 `interfaceRef = [rho0, rhoTheta0]`。

The maximum Stage 3 compute-stage requirement is now **8 storage buffers**, matching the WebGPU baseline target used by the desktop/mobile architecture.

## 第二次真機結果：通過 / Second real-device result: PASS

使用者在 Windows + Chrome、更新到 commit `0b95a239ef85fe8b1ac6ba826b0e5aff9390721c` 後重新執行：

- `npm test`：CPU Float64 **7/7 passed**。
- WebGPU compute pipelines：**全部成功編譯**。
- hydrostatic one-step smoke：**PASS**。
- GPU hydrostatic smoke `max |w| = 6.843e-6 m/s`。
- density / pressure / NaN validation：未觸發錯誤。
- Stage 3 core 的 storage-buffer requirement = **8**，因此已不依賴桌機 adapter 額外提供的 16-buffer capability。

After updating to commit `0b95a239ef85fe8b1ac6ba826b0e5aff9390721c`, the real Windows + Chrome run produced:

- CPU Float64 tests: **7/7 passed**.
- All WebGPU compute pipelines: **compiled successfully**.
- One-step hydrostatic GPU smoke: **PASS**.
- Hydrostatic GPU smoke `max |w| = 6.843e-6 m/s`.
- No density / pressure / NaN validation failure.
- Stage 3 core storage-buffer requirement = **8**, so the implementation no longer depends on the desktop adapter's optional 16-buffer capability.

## 判定 / Assessment

這個結果足以通過 **Stage 3 真機 WebGPU pipeline/smoke gate**：GPU 核心可在實際瀏覽器裝置建立、執行與 readback，而且沒有結構性 hydrostatic 爆炸。

This result passes the **Stage 3 real-device WebGPU pipeline/smoke gate**: the GPU core can compile, execute and read back on a real browser device without a structural hydrostatic instability.

但 one-step smoke 不等於長時間數值可信度驗證。Stage 3 正式封關前仍要增加 multi-step GPU hydrostatic-rest 與 CPU-vs-GPU comparison，確認 `f32` hydrostatic residual 不會單向累積，並量測 mass drift / max `|w|` 隨時間的演化。

A one-step smoke is not a long-duration numerical-fidelity test. Before final Stage 3 closure, add a multi-step GPU hydrostatic-rest and CPU-vs-GPU comparison to verify that the `f32` hydrostatic residual does not accumulate monotonically and to measure mass drift / max `|w|` over time.
