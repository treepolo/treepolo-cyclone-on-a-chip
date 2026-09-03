# Stage 3 — 最小三維物理核心

狀態：**CPU Float64 reference core 已完成並通過第一批 V0/V1 回歸；WebGPU compute core 已在真實 Windows + Chrome 裝置成功編譯並通過 one-step hydrostatic smoke。Stage 3 現在只剩 multi-step GPU hydrostatic-rest / CPU-vs-GPU 數值一致性驗證，通過後正式封關。**

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
- Stage 3 debug 預設 top = 40 km；SLEVE terrain deformation 到 Stage 6/terrain work 再打開。

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
- Stage 3 reconstruction 仍是 donor-cell/upwind，只用於建立守恆骨架與 debug；production high-order monotone transport 仍按 Stage 2 roadmap 延後選型。

### 2.7 HEVI vertical acoustic solve

CPU 與 WGSL prototype 都採 per-column implicit tridiagonal solve：

- pressure perturbation / base-state `rhoTheta` acoustic coupling隱式處理；
- rigid lower/model-top `w=0` for Stage 3 tests；
- buoyancy 作為 slow term，在 implicit acoustic solve 之外顯式加入；
- CPU tridiagonal 使用 Thomas algorithm；GPU Stage 3 prototype 為 one invocation per column、`Nz<=128` 的 correctness-first Thomas implementation，未做效能最佳化。

這裡曾抓到並修正一個重要 bug：如果用 total pressure gradient 而未先減掉 hydrostatic reference，靜止大氣會自行產生巨大假 `w`。現在 hydrostatic-rest regression 會防止此錯誤回歸。

### 2.8 WebGPU compute prototype

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

Real-device result：Windows + Chrome，one-step hydrostatic smoke `max |w| = 6.843e-6 m/s`，無 NaN、負密度或負壓力錯誤。詳細紀錄見 `STAGE3_GPU_SMOKE_20260903.md`。

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

這些測試主要驗證 Stage 3 的 reference-state、conservation、acoustic/buoyancy coupling 與三維資料流，還不能證明 production weather fidelity。

## 4. 真機 WebGPU 驗收 / Real-device WebGPU validation

目前已通過：

- WebGPU adapter / device 建立。
- 所有 Stage 3 WGSL compute pipeline 編譯。
- Stage 3 storage-buffer baseline compatibility：最高需求 8。
- one-step hydrostatic state GPU integration + readback。
- density / pressure / NaN smoke validation。
- one-step hydrostatic residual：`max |w| = 6.843e-6 m/s`。

尚需補：

- multi-step hydrostatic-rest，觀察 `f32` residual 是否累積。
- GPU dry-mass drift。
- CPU Float64 vs GPU Float32 state comparison over the same short integration。

## 5. 執行

```bash
npm install
npm test
npm run serve
```

瀏覽器開啟：

`http://127.0.0.1:5173`

## 6. Stage 3 尚未宣稱完成的 production 項目

在進 Stage 4 前必須完成：

- multi-step real-GPU numerical agreement / conservation tolerance。

可在後續 refinement 持續優化、但不阻擋 Stage 4 的項目：

- GPU HEVI 效能與更平行的 tridiagonal algorithm；
- RK3 outer integrator 完整 production 化；目前 CPU/GPU debug step 是 Stage 3 split operator；
- horizontal momentum nonlinear advection 的 production high-order discretization；
- density-current benchmark；
- model-top sponge；
- SLEVE terrain geometry。

## 7. Stage 4 入口

在 multi-step GPU hydrostatic-rest / CPU-vs-GPU comparison 通過後，Stage 4 才加入：

- planet rotation / Coriolis；
- geostrophic/inertial tests；
- Held–Suarez thermal relaxation + near-surface drag；
- long-run zonal-mean circulation；
- Rossby / baroclinic development 所需的 production horizontal momentum transport。

在 hydrostatic rest、mass conservation 或 GPU core validation 未通過的裝置上，不進 Stage 4。
