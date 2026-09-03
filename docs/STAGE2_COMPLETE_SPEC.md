# Stage 2 — 完整規格索引

Stage 2 的目的，是把 Stage 3 開始實作前不能再臨場改動的架構、變數定義、數值方法、資料模型與驗收原則鎖定。後期才有足夠資訊選擇的微物理、邊界層、輻射與可變解析度細節保留到對應階段做 benchmark，不為了文件表面完整而提前亂選。

## Stage 2 已鎖定的核心決策

1. **平台**：TypeScript host + raw WebGPU/WGSL；CPU Float64 reference path。
2. **幾何**：真正有厚度的 3D spherical shell，正式動力核心直接預報垂直速度 `w`。
3. **水平網格**：uniform gnomonic equiangular cubed sphere 起步。
4. **水平拓撲**：每條物理 edge 有唯一 canonical identity；panel seam 在建網格時轉成固定鄰接關係，time step 不重新猜 face。
5. **垂直座標**：SLEVE-like hybrid geometric-height coordinate，小尺度地形影響比大尺度地形更快隨高度消失。
6. **動力方程**：fully compressible、non-hydrostatic、rotating Euler atmosphere；Earth reference 採 traditional shallow-atmosphere approximation，但仍完整保留垂直維度與 non-hydrostatic vertical momentum。
7. **熱力預報量**：dry-density-coupled moist potential temperature family；`theta_m = theta * (1 + R_v/R_d * q_v)`。
8. **水物質**：以相對乾空氣的 mixing ratio 定義，預報 `rho_d q_j`，輸送必須 positive-definite。
9. **空間離散**：conservative finite volume；scalar 在 cell center、水平法向速度／通量在 edge、`w` 在 vertical interface。
10. **時間積分**：RK3 / split-explicit family + HEVI per-column vertical acoustic implicit solve。
11. **GPU 精度**：prognostic state 用 `f32`，搭配 hydrostatic reference/perturbation 與 unit-sphere/local geometry；CPU 參考解用 `f64`。
12. **粒子**：純 Lagrangian tracer / visualization layer，不另建一套粒子氣體物理。
13. **可變解析度**：先讓 uniform core 通過 dry/moist benchmark，再比較 stretched cubed sphere 與 block-structured nesting。
14. **玩家干預**：只能透過 heat/moisture/momentum source 或 surface/boundary change；禁止直接操縱「颱風」「鋒面」「高壓」等天氣類型。
15. **驗證制度**：每加一層物理立即做 quantitative benchmark；mass、positivity、hydrostatic-rest 等基礎測試沒過，不進下一層。

## Stage 2 prototype 結論

### P1 — WebGPU 3D stencil / flux benchmark

benchmark 程式已設計完成，但目前執行環境的 headless Chromium 無法初始化 GPU/EGL，因此不填入假的 throughput 數字。正式程式啟動時必須讀取 WebGPU limits 並在真實裝置跑短 benchmark，再選 `N`、`Nz`、粒子數與診斷更新率。

### P2 — cubed-sphere shared-edge finite-volume transport

Float64 參考實作已實際跑過 solid-body rotation passive tracer。每條 edge 通量只計算一次，對兩側 cell 等量異號更新；跨六個 panel seam 仍能把全球 tracer mass drift 壓到機器精度。一階迎風法維持 positivity，但擴散過大，因此 production transport 必須升級高階 monotone / positive-definite reconstruction。

### P3 — vertical acoustic explicit vs implicit

20 km 高、80 層線性聲波柱測試中，完全顯式步進在 acoustic CFL 3 與 10 迅速爆炸；Crank–Nicolson column implicit solve 在相同 CFL 下保持穩定，300 steps 的 Float64 能量比維持在約 `1 ± 1e-13`。因此正式架構鎖定 HEVI / vertically implicit fast mode。

## 本階段文件

- `STAGE2_COMPLETE_SPEC.md`：本索引與鎖定決策。
- `PHYSICS_SPEC.md`：方程、狀態變數、濕物理接口與人為物理干預規則。
- `NUMERICAL_CORE_SPEC.md`：cubed sphere、垂直座標、有限體積、HEVI、precision。
- `DATA_MODEL.md`：GPU buffer、grid topology、particles 與 conservation ledger。
- `VALIDATION_PLAN.md`：從 geometry 到三胞環流、斜壓氣旋、濕對流、熱帶氣旋、季風的 benchmark。
- `PROTOTYPE_RESULTS.md`：三個 prototype 的詳細結果。

## Stage 3 的明確入口

Stage 3 不再討論要不要 3D、要不要 non-hydrostatic、用哪種全球網格；直接實作：

1. cubed-sphere geometry/topology builder；
2. stretched vertical reference grid，先 flat surface；
3. CPU Float64 hydrostatic reference atmosphere；
4. GPU state buffers：`rhoD`, `rhoThetaM`, `uEdge`, `wInterface`；
5. equation of state / pressure diagnostic；
6. conservative mass-flux divergence；
7. vertical acoustic HEVI column solver；
8. minimal 3D debug renderer：大氣球殼、slice、velocity/particle diagnostics；
9. V0/V1 tests，先過 hydrostatic rest、acoustics、gravity wave、thermal bubble，再加入旋轉。

## 尚未寫死、且不阻擋 Stage 3 的項目

- production 高階 scalar limiter 的具體演算法；
- microphysics scheme；
- PBL scheme；
- radiation scheme；
- variable-resolution 第一版採 stretch 或 nesting；
- 真實地形 SLEVE filter cutoff 與 decay heights 的最終值。

上述項目都有固定介面與驗收位置，但應在相應階段依 benchmark 選擇。