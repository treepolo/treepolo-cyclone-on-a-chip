# Stage 2 Prototype Results

## P1 — WebGPU 3D stencil / flux benchmark

目的：讀取實際瀏覽器的 WebGPU limits，對六 panel、三維 SoA field 做 repeated compute passes，量測 cell-updates/s、buffer limits 與基本 ping-pong data path。

本次執行環境的 headless Chromium 無法初始化可用 GPU/EGL/Vulkan backend，因此沒有把 software fallback 或 CPU 數字冒充真實 GPU 效能。這個限制不改變架構決策：正式程式啟動時對使用者裝置做 capability probing + short throughput benchmark，再選 `N`、`Nz`、粒子數與 diagnostic cadence。

## P2 — cubed-sphere seam finite-volume transport

原型建立六面 gnomonic cubed sphere。每一條 cell edge 由全球三維端點配對成唯一 shared edge；solid-body rotation 的被動 Gaussian tracer 使用一階迎風 finite-volume flux。每條 edge flux 只計算一次，對左右 cells 等量異號更新，因此 seam 也遵守同一 conservation rule。

Float64 實測：

| 每面 N | 全球 cells | edges | 一圈 steps | 相對總質量誤差 | tracer 最小值 | relative L2 error |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 864 | 1,728 | 266 | -1.47e-16 | 1.78e-5 | 0.790 |
| 24 | 3,456 | 6,912 | 553 | 0 | 1.86e-8 | 0.664 |
| 48 | 13,824 | 27,648 | 1,128 | 0 | 5.87e-13 | 0.512 |

結論：

1. 六個 panel seam 可以用 shared-edge topology 做到機器精度的全球質量守恆。
2. 一階迎風保持 positivity，但擴散非常大；誤差隨解析度下降，production 必須升級高階 monotone / positive-definite reconstruction。
3. 另做 repeated Float32 panel remap：如果每一步把 global Cartesian position 重新轉 face/local coordinate 並量化，1000 次後最差角誤差約 `5e-4 rad`。因此正式 PDE topology 不使用每步浮點 face-search/remap；neighbor/sign/geometry 必須預建。

## P3 — vertical acoustic explicit vs implicit

線性一維聲波柱：高度 20 km、80 層、聲速 340 m/s，pressure 在 cell center，vertical velocity 在 layer interface；上下 rigid wall。比較 staggered explicit forward-backward 與 Crank–Nicolson implicit column solve。

Float64、300 steps 實測：

| acoustic CFL | explicit | explicit energy ratio | implicit energy ratio |
|---:|---|---:|---:|
| 0.25 | stable | 1.0093 | 1.00000000000003 |
| 1 | stable but less accurate | 1.0505 | 1.00000000000003 |
| 3 | blows up around 43 steps | >5e98 | 0.99999999999993 |
| 10 | blows up around 24 steps | >5e96 | 1.00000000000006 |

結論：細垂直層若全顯式處理聲波，整個全球模式的 timestep 會被最小 `dz/c_s` 綁死；per-column vertically implicit solve 可解除這個限制，而且各 column 可大量 GPU 平行。因此正式時間積分鎖定 HEVI / split-explicit family。

## 對 Stage 2 決策的直接影響

- `cubed sphere` 保留。
- panel seam 使用固定 shared-edge topology，不使用 runtime face guessing。
- transport 採 conservative pairwise edge flux。
- production advection 不能停在一階迎風。
- vertical acoustic fast mode 必須 implicit。
- GPU throughput 不寫死成某張顯卡的數字，而由 runtime probe 決定裝置 preset。