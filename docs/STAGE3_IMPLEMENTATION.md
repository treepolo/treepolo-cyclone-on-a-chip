# Stage 3 — 最小三維物理核心

狀態：**完成 / COMPLETE。CPU Float64 reference core 已通過 7/7 V0/V1 回歸；WebGPU compute core 已在真實 Windows + Chrome 裝置成功編譯並通過 one-step hydrostatic smoke，以及 1000-step multi-step hydrostatic-rest / conservation / CPU-vs-GPU agreement gate。Stage 3 正式封關，可進入 Stage 4。**

## 1. 本階段範圍

Stage 3 固定為 dry、non-rotating、flat-surface core。計算域已經是六面 gnomonic cubed-sphere × 多垂直層的真正 3D spherical shell；旋轉、Held–Suarez forcing、baroclinic instability 與天氣系統留到 Stage 4。

本階段沒有加入水汽、微物理、PBL、地形或任何 cyclone/front generator。

## 2. 已實作

### 2.1 Cubed-sphere geometry / topology

- equiangular gnomonic cubed sphere；每面 `N×N`。
- CPU Float64 建立 cell centers、spherical cell area、canonical shared edges、edge length、neighbor cells、cell-edge sign。
- panel seam 在 build 階段配對成和 panel interior 相同的 shared-edge topology；time step 不做 face guessing。
- 每個 cell 恰有四個 horizontal edges，每個 global edge 恰有兩個 neighbor cells。

### 2.2 垂直網格

- flat-surface stretched geometric-height grid。
- `zInterface[Nz+1]`、`zCenter[Nz]`、`dz[Nz]`。
- Stage 3 debug 預設 top = 40 km；SLEVE terrain deformation 到 terrain stage 再打開。

### 2.3 Hydrostatic reference atmosphere

- CPU Float64 isothermal hydrostatic state `p0(z), rho0(z), theta0(z), rhoTheta0(z)`。
- interface reference arrays 同時建立，供 HEVI base-state flux 使用。
- pressure-gradient 以 perturbation `p' = p - p0` 進入 vertical acoustic solve；背景 hydrostatic pressure gradient 不重複加速空氣。

### 2.4 Dry prognostic state

正式資料模型已落成：

- `rhoD[cell,k]`
- `rhoThetaM[cell,k]`
- `uEdge[edge,k]`
- `wInterface[cell,k+1/2]`

CPU reference 使用 `Float64Array`；GPU production prototype 使用 `f32` storage buffers。

### 2.5 Equation of state

Dry Stage 3：

`p = p_ref * (R_d * rhoThetaM / p_ref)^gamma`

`theta = rhoThetaM / rhoD`

`T = theta * (p/p_ref)^kappa`

### 2.6 Conservative finite-volume transport

- horizontal flux 以 canonical shared edge 計算；正 flux 定義為 left cell → right cell。
- cell divergence 由同一 edge flux 以相反 sign 更新兩側 cell。
- vertical base-state acoustic flux 由 HEVI 推進；outer explicit transport 只輸送 vertical perturbation flux，避免 base flux double count。
- Stage 3 reconstruction 仍是 donor-cell/upwind，只用於建立守恆骨架與 debug；production high-order monotone transport 依 roadmap 後續升級。

### 2.7 HEVI vertical acoustic solve

CPU 與 WGSL prototype 都採 per-column implicit tridiagonal solve：

- pressure perturbation / base-state `rhoTheta` acoustic coupling 隱式處理；
- rigid lower/model-top `w=0` for Stage 3 tests；
- buoyancy 作為 slow term，在 implicit acoustic solve 之外顯式加入；
- CPU tridiagonal 使用 Thomas algorithm；GPU Stage 3 prototype 為 one invocation per column、`Nz<=128` 的 correctness-first Thomas implementation，未做效能最佳化。

本階段曾抓到並修正一個重要 bug：如果用 total pressure gradient 而未先減掉 hydrostatic reference，靜止大氣會自行產生巨大假 `w`。現在 hydrostatic-rest regression 會防止此錯誤回歸。

### 2.8 WebGPU compute core

`src/gpu/dryCoreGpu.ts` 已包含：

1. pressure diagnostic；
2. horizontal pressure-gradient edge velocity update；
3. per-column HEVI solve；
4. buoyancy；
5. horizontal shared-edge mass / rhoTheta flux；
6. vertical perturbation flux；
7. cell-centric conservative divergence update；
8. state upload / readback；
9. WebGPU validation error scope 與 hydrostatic smoke-test entry。

第一版真機測試發現 compute stage 使用 10 個 storage buffers，超過 WebGPU baseline 的 8。之後已將垂直 reference fields 與 flux buffers 重新打包，使 Stage 3 所有 compute pipeline 的最高需求降為 **8 storage buffers/stage**；第二次真機測試成功建立所有 pipelines 並執行 hydrostatic smoke。

One-step real-device result：Windows + Chrome，hydrostatic smoke `max |w| = 6.843e-6 m/s`，無 NaN、負密度或負壓力錯誤。詳細紀錄見 `STAGE3_GPU_SMOKE_20260903.md`。

### 2.9 3D Debug Viewer

- 瀏覽器可旋轉／縮放的 3D spherical-atmosphere point viewer。
- 大氣厚度為便於除錯而做 radial exaggeration；solver 的物理高度仍用真實公尺。
- 可 reset、單步、連續跑、插入 constant-pressure warm thermal bubble。
- 顯示 dry-mass drift、max `|w|`、min density、min pressure、simulation time。
- thermal bubble 只改 thermodynamic initial condition，不直接指定 upward velocity。
- 所有使用者介面依 `UI_SPEC.md` 同時顯示繁體中文與英文。

## 3. CPU Float64 validation results

`npm test`：**7/7 passed**。

| Test | Result |
|---|---|
| V0 cubed-sphere area/topology | PASS；`ΣA = 4π` within Float64 tolerance，global edge count `12 N²` |
| EOS round trip | PASS |
| HEVI acoustic CFL≈8 stability | PASS |
| standing acoustic wave phase/amplitude | PASS；one numerical period relative L2 < `1e-3` |
| stratified internal-gravity-wave response | PASS；bounded + mass-conservative |
| global isothermal hydrostatic rest | PASS；20 s mass drift = `0`，max `|w| = 1.23e-13 m/s` |
| constant-pressure +3 K thermal bubble | PASS；15 s mass drift = `2.03e-16`，max `|w| = 3.83e-2 m/s` |

## 4. 真機 WebGPU 驗收 / Real-device WebGPU validation

### 4.1 One-step smoke — PASS

- WebGPU adapter / device 建立：PASS。
- 所有 Stage 3 WGSL compute pipeline 編譯：PASS。
- storage-buffer baseline compatibility：最高需求 8。
- hydrostatic state GPU integration + readback：PASS。
- density / pressure / NaN validation：PASS。
- one-step hydrostatic residual：`max |w| = 6.843e-6 m/s`。

### 4.2 1000-step multi-step gate — PASS

設定：`6 × 8 × 8 × 32`、`dt = 0.25 s`、1000 steps = 250 s simulation time；CPU Float64 與 GPU Float32 從同一靜力初始條件獨立積分。

1000-step checkpoint：

- GPU dry mass drift = `4.341e-7`，threshold `1e-6` → PASS。
- GPU max `|w| = 9.828e-4 m/s`，threshold `1e-3 m/s` → PASS。
- `rhoD` relative L2 = `1.605e-6`，threshold `2e-5` → PASS。
- `rhoThetaM` relative L2 = `8.172e-7`，threshold `2e-5` → PASS。
- max `|Δu| = 0`，threshold `1e-4 m/s` → PASS。
- max `|Δw| = 9.828e-4 m/s`，threshold `1e-3 m/s` → PASS。
- 無 NaN、負密度或負壓力 failure。
- 真機驗證耗時約 `5.28 s`。

詳細 checkpoint 結果見 `STAGE3_GPU_MULTISTEP_VALIDATION.md`。

注意：`max |w|` 在 1000-step 時已達 gate 約 98.3%，因此 Stage 4 及更長時間積分必須繼續監控 hydrostatic/balance residual。此項目前通過事前規定的 correctness gate，但不得被解讀成可以忽略長時間 Float32 誤差。

## 5. 執行

```bash
npm install
npm test
npm run serve
```

瀏覽器開啟：

`http://127.0.0.1:5173`

## 6. Stage 3 後續 refinement，不阻擋 Stage 4

- GPU HEVI 效能與更平行的 tridiagonal algorithm；
- RK3 outer integrator 完整 production 化；目前 CPU/GPU debug step 是 Stage 3 split operator；
- horizontal momentum nonlinear advection 的 production high-order discretization；
- density-current benchmark；
- model-top sponge；
- SLEVE terrain geometry。

其中 production horizontal momentum transport 會在 Stage 4 為旋轉全球乾大氣與 baroclinic dynamics 實際補齊；terrain/SLEVE 仍留在後續地表階段。

## 7. Stage 4 入口

**Stage 3 gate 已全部通過。Stage 4 可以開始。**

Stage 4 首批工作：

- planet rotation / Coriolis；
- inertial oscillation / geostrophic balance tests；
- production horizontal momentum transport；
- Held–Suarez thermal relaxation + near-surface drag；
- long-run zonal-mean circulation；
- Rossby / baroclinic development 所需的 dry-core dynamics。

Stage 4 的每個新增物理層仍需保留 Stage 3 hydrostatic-rest、mass-conservation 與 GPU regression tests，避免新增功能破壞已通過的核心。
