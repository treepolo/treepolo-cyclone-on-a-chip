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

## 狀態 / Status

CPU Float64 的 7/7 Stage 3 tests 不受此 binding-layout 修正影響。更新後的 WGSL 仍需要在實際 WebGPU 裝置重新執行 smoke test；在重新測試成功以前，不宣稱 Stage 3 GPU gate 已通過。

The CPU Float64 7/7 Stage 3 tests are unaffected by this binding-layout refactor. The updated WGSL still requires another real-device smoke run; the Stage 3 GPU gate remains pending until that test succeeds.
