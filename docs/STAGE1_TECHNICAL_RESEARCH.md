# Stage 1 — 技術研究與核心選型

本文件記錄 treepolo Cyclone on a Chip 在正式撰寫完整規格前的技術研究與第一版核心選型。目標是建立一個可在瀏覽器執行、桌機與手機皆可操作、具有真正三維厚度與垂直運動的全球大氣模擬器；天氣系統必須由物理方程自然演化，不以「生成颱風／生成鋒面」之類的規則硬編。

> 狀態：Stage 1 第一版結論。Stage 2 寫完整規格時仍可根據 prototype benchmark 修改局部選型，但若要推翻本文件的核心決策，需要記錄理由。

## 1. 現有模式研究

### 1.1 MPAS-Atmosphere

可取用的設計：

- fully compressible、non-hydrostatic 動力核心。
- centroidal Voronoi 非結構網格與 C-grid staggering。
- 可平滑改變解析度的全球網格。
- generalized terrain-following geometric-height vertical coordinate。
- split-explicit third-order Runge–Kutta；氣象尺度模式使用大時間步，聲波使用小時間步。
- exact dry-air mass / scalar mass conservation、positive-definite transport 等概念很適合當驗收參考。

不直接採用的原因：

- MPAS 的 Voronoi 非結構鄰接對全球可變解析度很漂亮，但在瀏覽器 WebGPU 上會帶來大量間接索引與不規則記憶體存取。
- 我們需要同一 GPU 同時跑物理、粒子與 3D 繪圖；規則化資料布局的工程優勢比「最漂亮的任意可變解析度網格」重要。

參考：
- https://mpas-dev.github.io/atmosphere/atmosphere.html
- https://www2.mmm.ucar.edu/projects/mpas/site/documentation/users_guide/overview.html
- https://www2.mmm.ucar.edu/projects/mpas/mpas_website/build/html/documentation/users_guide/appC_grid.html

### 1.2 WRF-ARW

可取用的設計：

- fully compressible、non-hydrostatic。
- Arakawa C-grid。
- RK2/RK3 + acoustic/gravity-wave time splitting。
- hybrid terrain-following hydrostatic-pressure coordinate。
- 多種 monotonic / positive-definite scalar transport、microphysics、PBL 與 surface physics 可作日後參數化參考。

不直接採用的原因：

- WRF 的主要強項是區域模式與 nesting；它不是我們要的「從第一天就是全球球面、瀏覽器 GPU、互動 3D」架構。
- pressure-coordinate 對傳統 NWP 很成熟，但本專案的 3D 幾何高度視覺化與非靜力垂直結構更適合以 height-based coordinate 為主。

參考：
- https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/dynamics.html
- https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/v4.4/users_guide_chap5.html

### 1.3 ICON

可取用的設計：

- fully compressible、non-hydrostatic Euler equations。
- icosahedral triangular C-grid，沒有經緯網格極點奇異。
- height-based terrain-following SLEVE coordinate，上層逐漸轉成固定高度面。
- 2026 已有 operational GPU NWP 的公開成果，顯示完整氣象模式在 GPU 上跑有實際工程可行性。

不直接採用的原因：

- icosahedral triangular grid 仍屬非結構拓撲；對本專案 WebGPU 的資料布局與粒子查值不是最簡單方案。

參考：
- https://gmd.copernicus.org/articles/15/7153/2022/
- https://gmd.copernicus.org/articles/19/755/2026/

### 1.4 FV3

這是本專案目前最重要的「水平網格與全球架構」參考。

可取用的設計：

- finite-volume cubed-sphere dynamical core。
- 全球沒有經緯網格的極點奇異，又能把六個面維持成近似規則的 2D 陣列。
- 支援 non-hydrostatic simulation。
- 支援 stretched grid、nesting 與 variable-resolution 使用情境。
- 有 DCMIP baroclinic wave、orographic wave、idealized tropical cyclone、supercell 等公開理想化測試案例。
- finite-volume conservation 與 pressure-gradient treatment 很值得研究。

需要避免照搬的部分：

- FV3 的 vertically Lagrangian coordinate、D-grid/C-grid 混合與完整 remapping 系統工程量很大，Stage 1 不決定整套複製。
- cubed-sphere panel edge 仍有 seam interpolation / grid imprinting 問題，Stage 2 必須把跨面通量處理寫成明確規格並列入測試。

參考：
- https://www.gfdl.noaa.gov/fv3/
- https://www.gfdl.noaa.gov/fv3/fv3-key-components/
- https://www.gfdl.noaa.gov/fv3/fv3-grids/
- https://www.gfdl.noaa.gov/fv3/fv3-idealized-tests/

### 1.5 CM1

可取用的設計：

- 3D non-hydrostatic idealized atmosphere model。
- 對 thunderstorm、deep precipitating convection、LES 等小至中尺度問題非常適合作為物理與 benchmark 參考。
- 支援多種 equation sets；compressible、anelastic 等模式可協助我們比較時間步與低 Mach number 問題。

不直接採用的原因：

- 主要是 Cartesian / idealized small-scale model，不是全球球面架構。

參考：
- https://www2.mmm.ucar.edu/people/bryan/cm1/
- https://www2.mmm.ucar.edu/people/bryan/cm1/cm1_equations.pdf

## 2. 第一版核心決策

### D1 — 不 fork MPAS / WRF / ICON / FV3 作為產品本體

**決定：自製 solver；成熟模式用作演算法、參數化與 benchmark 來源。**

理由：本專案最特殊的條件是「browser + mobile + GPU compute + 3D interactive visualization」。直接塞入大型 Fortran/MPI NWP codebase，最後很可能花更多工作處理跨語言、I/O、GPU/瀏覽器與即時互動，而不是處理物理。

### D2 — 正式平台：WebGPU；WebGL 不承擔物理 solver

**決定：正式高效能路徑使用 WebGPU compute shader + WebGPU renderer。**

理由：

- WebGPU 同時提供 compute 與 rendering，可讓大氣場、粒子與繪圖資料留在同一張 GPU 上，避免每幀 GPU→CPU→GPU 搬運。
- Chrome 已在 Android 上提供 WebGPU，Safari 26 已在 macOS/iOS/iPadOS/visionOS shipping WebGPU。
- 2026 Chrome 也已加入 WebGPU compatibility mode，能在部分較舊 Android/OpenGL ES 3.1 裝置上擴大覆蓋。

限制：

- WebGPU 仍不是所有瀏覽器／裝置都保證可用，因此啟動時必須做 capability probing。
- WGSL 沒有可直接拿來跑 shader 的 concrete `f64`；正式 GPU solver 不能依賴 double precision。

參考：
- https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API
- https://developer.chrome.com/blog/new-in-webgpu-121
- https://developer.chrome.com/blog/new-in-webgpu-146
- https://webkit.org/blog/17333/webkit-features-in-safari-26-0/
- https://gpuweb.github.io/gpuweb/wgsl/

### D3 — 軟體技術棧：TypeScript host + raw WebGPU + WGSL

**決定：Stage 3 起主要執行環境使用 TypeScript + WebGPU + WGSL；不把 Three.js 當物理核心。**

理由：

- WebGPU browser API 的新功能與 compatibility mode 能直接使用，不受中介 library 支援時程限制。
- solver 的每一個 storage buffer、compute pass、同步點與數值格式都能明確控制。
- JavaScript `Number` / `Float64Array` 可在 CPU 建立小網格 double-precision reference solver，用於驗證 GPU `f32` 解。
- 不需要一開始引入 Rust/WASM 或大型 engine；若 CPU profiling 證明 TypeScript host 成為瓶頸，再局部導入 WASM。

3D renderer 也直接共用同一個 WebGPU device。UI 使用 HTML/CSS/TypeScript overlay；攝影機、選取、剖面與粒子繪圖自行建立，不讓 render framework 決定 solver 的資料模型。

### D4 — 水平網格：gnomonic cubed-sphere

**決定：正式全球水平網格採六面 cubed-sphere；Stage 3/4 首先做 uniform grid。**

淘汰：

- latitude-longitude grid：極區網格收斂造成 CFL 與數值奇異，不採用。
- MPAS Voronoi / ICON triangle：科學上可行，但對本專案 WebGPU 記憶體規律性與查值複雜度不划算。

預定配置：

- 六個 panel，各自為 `N × N × Nz` column array。
- scalar state 以 cell center 為主。
- horizontal velocity 採 face-normal staggered arrangement（C-grid family）。
- vertical velocity 放在 layer interface。
- panel edge 建立固定 halo / neighbor transform，不用經緯度極點特判。

### D5 — 垂直座標：hybrid height / SLEVE-like terrain-following coordinate

**決定：幾何高度是垂直座標的核心。**

需求：

- 大氣域是真正有厚度的 3D spherical shell。
- 地表附近層面跟隨地形。
- 地形影響隨高度快速衰減，上層逐漸轉為近似固定幾何高度面。
- 垂直網格允許 stretching，近地面與對流層可較密，上層較疏。
- 模式頂部使用固定高度 + sponge / damping layer，避免上界反射污染對流層。

這個選擇比純 pressure coordinate 更符合本專案「切開地球直接看垂直對流」的視覺需求，也保留 non-hydrostatic mountain-wave / convection 的物理意義。

### D6 — 方程組：fully compressible、non-hydrostatic、rotating Euler atmosphere

**決定：正式核心不使用 2D shallow-water，也不使用 hydrostatic primitive-equation solver。**

最低核心必須包含：

- dry-air mass conservation
- 3D momentum conservation
- thermodynamic prognostic equation
- equation of state
- gravity
- Earth rotation / Coriolis
- true vertical velocity
- later: water species mass conservation + latent heating

第一版採傳統 Earth-weather 使用的 shallow-atmosphere approximation，但計算域仍是完整三維、有垂直厚度且可 non-hydrostatic 對流。`shallow-atmosphere approximation` 只代表忽略部分 O(z/R) 的 deep-atmosphere metric effects，不代表把大氣壓成 2D。若未來要模擬高度占行星半徑顯著比例的行星大氣，再增加 deep-atmosphere option。

### D7 — 空間離散：conservative finite-volume + staggered fluxes

**決定：質量與 scalar transport 使用 flux-form finite-volume；速度採 C-grid family stagger。**

要求：

- 任何跨 cell 質量流都用同一 face flux 更新兩側 cell，避免「粒子、密度、氣壓各算各的」。
- moisture / cloud species 加入後必須有 positive-definite transport，禁止出現負水汽。
- pressure-gradient scheme 必須有 hydrostatic-rest benchmark；山區不得因離散誤差自己產生巨大假風。
- 高階 advection 可以晚於最小 2nd-order conservative core 加入，但不能為了畫面平滑犧牲守恆。

### D8 — 時間積分：RK3 large step + split acoustic substeps；垂直快模態優先 implicit

**決定：第一正式候選採 WRF/MPAS 類 split-explicit RK3 family。**

理由：

- 氣象尺度 motion 不應因聲速 CFL 被迫使用同樣小時間步。
- 大時間步積分 meteorological modes；小步處理 acoustic modes。
- 垂直聲波對細垂直層特別苛刻，因此 Stage 2 規格優先採 per-column vertically implicit acoustic treatment；這種 column solve 很適合 GPU 平行處理多個 column。

保留實驗選項：

- reduced-speed-of-sound / modified compressible acceleration 只能當「加速模式」研究，不作 reference physics default。
- reference mode 必須保留未人為降低聲速的可壓縮解，以便驗證加速模式有沒有扭曲天氣。

### D9 — GPU 精度策略：f32 state + reference-state/perturbation formulation

WGSL GPU shader 目前不能依賴 concrete `f64`，所以不能把「地心座標 6,371,000 m」與「幾十公尺 cell displacement」直接用同一組 global `f32` Cartesian number 做差。

**決定：**

- GPU prognostic state 以 `f32` 為主。
- cubed-sphere geometry 使用 normalized sphere / local panel coordinate，避免大數減大數。
- hydrostatic base state 與 perturbation 分離，pressure / temperature 等容易出現 cancellation 的量優先採 reference-state formulation。
- CPU validation path 使用 `Float64Array` / JavaScript double precision。
- global reductions（總質量、總能量誤差等）採分層 reduction，必要時使用 compensated summation / pairwise reduction；不可只把數百萬個 `f32` 直接任意相加。

### D10 — variable resolution：架構預留，但不在第一個 solver 版本啟用

**決定：**

1. uniform cubed-sphere 先通過所有 dry-core benchmark。
2. 之後加入 stretched cubed-sphere 或 block-structured nested patch。
3. dynamic AMR / moving nest 暫不列為初始要求。

理由：熱帶氣旋與中尺度系統最終需要遠比全球背景網格細的解析度，但一開始把 coarse/fine interface 也塞進 solver，會讓任何守恆 bug 都更難定位。

### D11 — 粒子不是物理 solver

**決定：可見粒子是 Lagrangian tracer / air-parcel visualization layer。**

- 粒子從 Eulerian 3D velocity field 插值取得速度。
- 可攜帶 age、source、sampled temperature、humidity、pressure、vorticity 等顯示資訊。
- 粒子數量、trail 長度與 point size 可依桌機／手機效能調整。
- 粒子不能另外擁有一套「碰撞／慣性／氣壓」系統來反過來決定大氣狀態。

## 3. WebGPU 對網格尺寸的直接限制

WebGPU 裝置能力必須在 runtime 讀取，不假設所有 GPU 一樣。常見保守限制包含約 256 MiB `maxBufferSize`、256 compute invocations/workgroup 與 16 KiB workgroup storage 的 baseline tier；實際硬體可能更高。

因此資料布局採：

- Structure of Arrays（SoA），不要一個 cell struct 塞所有變數。
- field / panel / vertical chunk 分 buffer，避免單一超大 buffer。
- ping-pong state buffer 只對真正需要的 prognostic fields 使用。
- diagnostic field 儘量即算即用，不永久存整個 3D volume。
- renderer 直接讀 solver storage buffers 或由 GPU compute 產生 render buffer，不做每幀 CPU readback。

參考：
- https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits

## 4. 桌機與手機採同一物理核心，不同解析度預設

手機版不另寫一套假物理。差異只允許出現在：

- horizontal resolution
- vertical layer count
- particle count
- trail length
- output / diagnostic frequency
- optional physics cost tier
- render resolution / DPR

不能在手機版把 Coriolis、pressure-gradient、vertical momentum、latent heating 等核心方程關掉來換 FPS。

## 5. Stage 2 必須精確規格化的尚未鎖定項目

Stage 1 已選定架構方向，但下列項目要在 Stage 2 用公式與 benchmark 鎖死：

1. dry-core prognostic thermodynamic variable：`rho*theta` family vs total-energy family 的最終形式。
2. exact cubed-sphere metric terms、panel-edge vector transform 與 halo exchange。
3. C-grid momentum operator / pressure-gradient operator 的離散公式。
4. vertical acoustic implicit solve 的離散形式與 boundary conditions。
5. numerical diffusion / hyperdiffusion / divergence damping 的最小必要配置。
6. model top height、sponge-layer 起始高度與垂直層 spacing function。
7. hydrostatic reference-state definition 與 GPU perturbation variable scaling。
8. first-order/second-order prototype 到 production higher-order advection 的升級路徑。
9. stretched grid vs nested grid 的第一個 variable-resolution 實作。
10. mobile / desktop 的具體 memory budget、resolution preset 與最低 WebGPU limits。

## 6. Stage 2 前的最小 prototype 建議

雖然正式 solver 在 Stage 3 才開始，Stage 2 規格定稿前應做三個極小 benchmark prototype，避免在紙上選錯：

### P1 — WebGPU 3D stencil / flux benchmark

- six-panel SoA buffer layout
- 典型 `N × N × Nz` neighbor access
- 多個 compute pass ping-pong
- 量測 desktop 與 mobile 的 cells/s、bandwidth、GPU frame time

### P2 — cubed-sphere seam transport prototype

- passive tracer 繞全球平流
- 多次跨越六面 seam
- 檢查 mass conservation、shape error、grid imprinting

### P3 — vertical column acoustic prototype

- compressible hydrostatic column
- acoustic / gravity perturbation
- explicit vs vertically implicit small-step 比較
- 確認 WebGPU column solve 的效能與穩定性

這三個 prototype 不需要雲、地圖、粒子特效，也不算正式模擬器功能；它們的目的只是讓 Stage 2 的數值規格有實測依據。

## 7. Stage 1 結論摘要

目前推薦正式方向：

```text
Browser / mobile / desktop
        │
TypeScript host + HTML/CSS UI
        │
WebGPU device
   ├─ WGSL atmospheric compute
   │    ├─ cubed-sphere horizontal grid
   │    ├─ hybrid height vertical grid
   │    ├─ fully compressible non-hydrostatic dynamics
   │    └─ split-explicit RK3 family
   │
   └─ WGSL/WebGPU rendering
        ├─ Earth / terrain
        ├─ atmospheric particles
        ├─ trails
        ├─ slices / isosurfaces
        └─ diagnostic overlays

CPU Float64 reference / validation path
        └─ small grids only
```

核心參考來源分工：

- **FV3**：cubed-sphere、finite-volume、global / stretched / nested grid 思想。
- **MPAS-A**：fully compressible non-hydrostatic、height coordinate、split-explicit、守恆與 variable-resolution benchmark 思想。
- **WRF-ARW**：RK3 / acoustic splitting、C-grid、microphysics / PBL / surface physics 參考。
- **ICON**：SLEVE vertical coordinate、global nonhydrostatic、GPU NWP 工程參考。
- **CM1**：deep convection、small-scale nonhydrostatic benchmark 與 later moist-convection 驗證。

Stage 1 到此可以視為完成第一版；下一階段是 Stage 2：把以上選型轉成可直接照著實作與驗收的完整 physics / numerical / data / visualization specification。
