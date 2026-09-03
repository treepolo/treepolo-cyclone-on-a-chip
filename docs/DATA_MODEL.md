# Data Model and GPU Layout v1

## 1. 原則

- solver state、diagnostics、visual particles 分離。
- 使用 Structure of Arrays，不建立巨大的 `Cell {rho,p,T,u,v,w,...}` array。
- field 可依 panel / vertical chunk 分 buffer，避免單一 WebGPU buffer/binding limit。
- geometry 大多 immutable。
- diagnostics 能即時計算就不永久保存完整 3D volume。
- renderer 直接讀 solver/derived GPU buffers，避免每 frame GPU→CPU→GPU readback。

## 2. Grid identifiers

### Cell
邏輯 ID 為 `(panel,i,j,k)`；CPU topology builder 同時產生 flattened integer index。GPU hot path 只用 integer index tables，不使用字串，也不 runtime face-search。

### Horizontal edge
每條物理 edge 有唯一 canonical `edgeId`：
- leftCell / rightCell
- canonical normal direction
- edge length
- midpoint direction
- orientation/sign metadata

panel seam edge 和 panel 內 edge 使用同一資料結構；seam 差異在 topology build 階段消化。

### Vertical interface
每 column 有 `Nz+1` interfaces；`w[k]` 定義在 interface，scalar layer `k` 位於 `w[k]` 與 `w[k+1]` 之間。

## 3. Core GPU arrays

### Immutable / slow geometry
- `cellArea[]: f32`
- `cellCenterXYZ[]: vec4<f32>`
- `edgeCells[]: vec2<u32>`
- `edgeMetric[]: vec4<f32>`
- `edgeNormalXYZ[]: vec4<f32>`
- `layerZeta[]: f32`
- terrain-enabled 時增加 column Jacobian / height geometry arrays

### Prognostic dry core
- `rhoD[]: f32`
- `rhoThetaM[]: f32`
- `uEdge[]: f32`
- `wInterface[]: f32`

RK stage / ping-pong 只替 prognostic arrays 配置必要工作 buffer；derived fields 不自動 double-buffer。

### Moist extension
- `rhoQv[]`
- `rhoQc[]`
- `rhoQr[]`
- `rhoQi[]`
- `rhoQs[]`
- optional `rhoQg[]`

手機可以使用較簡 microphysics tier 減少 species 與記憶體，但不能換成不同的核心動力方程。

### Surface arrays
- terrain height
- land/sea mask
- SST / skin temperature
- soil moisture
- albedo
- aerodynamic roughness
- surface heat capacity

## 4. Derived fields

依 UI/diagnostic 需求產生：pressure、temperature、relative humidity、wind speed、vertical velocity view、divergence、relative/absolute vorticity、potential vorticity、cloud condensate、precipitation rate。只有打開對應圖層或 benchmark 需要時才建立／更新完整 diagnostic buffer。

## 5. Lagrangian particles

粒子使用獨立 storage buffers。位置優先表示成 normalized direction + altitude，或 local-cell coordinate，避免 global Cartesian 大數精度問題。基本欄位：position、age、source tag；trail 可使用 ring buffer / compact history。

GPU particle step：locate current cell → interpolate 3D velocity → RK2/RK3 trajectory step → optional diagnostic sample。粒子刪除、重生、LOD 變化不得修改 Eulerian mass field。

## 6. Conservation ledger

CPU 低頻率收集：
- dry-air mass
- each water species / total water
- axial angular momentum diagnostic
- kinetic/internal/potential energy diagnostic
- user-injected heat/water/momentum
- precipitation leaving atmosphere / surface exchanges

ledger 同時是除錯工具與玩家實驗介面。

## 7. WebGPU device adaptation

啟動時讀取 WebGPU limits 並跑短 P1 benchmark。裝置 preset 可調：
- horizontal `N`
- `Nz`
- particle count
- trail history
- diagnostic refresh rate
- optional physics tier
- render resolution / DPR

核心 mass、momentum、thermodynamic、vertical-momentum equations 不因手機版而關掉。