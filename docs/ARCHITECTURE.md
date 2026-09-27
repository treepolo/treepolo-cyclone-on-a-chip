# 架構與路線圖 / Architecture and Roadmap（v0.1，2026-09-26 重新設計 / redesign）

## 0. 為什麼重寫 / Why the rewrite

v0.0.x 嘗試以 **全可壓縮、非靜力、立方球 C 網格（只存邊法向風）** 的全球動力核心，直接在 WebGPU 上實作。約 11,000 行程式中大部分是 Stage 4 的診斷與 prototype；經過多輪修補，仍無法讓乾燥 Held–Suarez 基準穩定積分 30 天，更遑論三胞環流。卡關處並非偶然：

1. 非正交立方球上的 C 網格，其科氏力／壓力梯度算子要同時做到能量中性與角動量守恆，是已知的困難問題（TRiSK 等方法原本為正交 Voronoi 網格設計）。舊 repo 的 AAM 閉合、Hodge、primal-dual prototype 全都是在與此搏鬥。
2. 在全球解析度（> 10 km），聲波與非靜力效應對天氣沒有物理貢獻，卻主導了數值穩定性（HEVI、split-explicit、聲波濾波、模式頂吸收層……）。
3. 在沒有已驗證的全球環流以前，就把 GPU 移植、CPU/GPU 一致性、真機 gate 一起做，使每一次除錯都要跨三層。

因此 v0.1 保留原專案的 **物理目標與硬限制**（真實單位、真實方程、天氣不可硬編、每層物理先過量化驗收），但更換動力核心的技術路線。舊程式完整保存在 git 歷史中（最後一個舊版 commit：`3cc0c9e`）。

The v0.0.x attempt (fully compressible non-hydrostatic cubed-sphere C-grid in WebGPU) never passed a 30-day dry Held–Suarez run. v0.1 keeps the physical goals and hard constraints but replaces the dynamical-core strategy. The legacy code remains in git history (last legacy commit `3cc0c9e`).

## 1. 尺度與模式分工 / Scales and model hierarchy

目標現象跨越五個數量級，單一全球網格不可能同時解析：

| 現象 / Phenomenon | 需要的水平格距 / Grid spacing needed | 由哪一層模式負責 / Model tier |
|---|---|---|
| 三胞環流、噴流、季風、行星波 / three-cell circulation, jets, monsoons | 100–300 km | 全球譜模式 / global spectral |
| 溫帶氣旋、鋒面、氣團、切離低壓 / extratropical cyclones, fronts, air masses, cut-off lows | 50–150 km | 全球譜模式 / global spectral |
| 熱帶氣旋的生成與路徑 / TC genesis and track | 25–50 km | 全球譜模式（高解析）/ global (high-res) |
| 颱風眼、眼牆、雙眼牆與眼牆置換 / eye, eyewall, concentric eyewalls, ERC | 1–3 km | 區域非靜力巢狀模式 / regional non-hydrostatic nest |
| 中尺度對流系統、飑線 / MCS, squall lines | 1–4 km | 區域非靜力巢狀模式 / regional nest |
| 超大胞與龍捲風 / supercells, tornadoes | 50–250 m | 局地 LES 巢狀模式 / local LES nest |

**靜力近似不是「隨便定式子」**：它是對真實方程在水平尺度 ≫ 垂直尺度時的嚴格尺度近似，ECMWF IFS 直到 9 km 仍採用。非靜力可壓縮方程在需要它的尺度（區域／局地巢狀模式）才使用，那裡幾何是局地笛卡兒網格，數值方法（CM1/WRF 類）成熟且簡單得多。

The hydrostatic approximation is a scale-justified approximation of the real equations, used operationally down to ~9 km. The fully compressible non-hydrostatic equations are used where they matter: in regional and local nests on Cartesian grids.

## 2. 全球動力核心 / Global dynamical core（已完成 / done）

- 方程：旋轉球面上的靜力原始方程（乾空氣質量、水平動量、熱力學第一定律、靜力方程、理想氣體狀態方程），SI 單位。
- 水平離散：球諧函數轉換法（spectral transform），三角截斷 T21 / T42 / T63，Gaussian 網格（無極點問題、無 C 網格科氏力相容性問題）。
- 預報量：渦度 ζ、輻散 D、溫度 T、ln p_s。
- 垂直：σ 座標，Simmons & Burridge (1981) 能量與角動量守恆的離散。
- 時間：半隱式 leapfrog（Hoskins & Simmons 1975，等溫參考態 300 K），Robert–Asselin–Williams 濾波，隱式 ∇⁸ 超擴散。
- 物理強迫介面 `PhysicsForcing`：格點上的 du/dt、dv/dt、dT/dt（之後擴充水物質）。

實作：`src/spectral/*`（轉換）、`src/model/dycore.ts`、`src/model/vertical.ts`。

### 已驗證 / Verified (`npm test`)

- FFT、Gaussian 權重、純量往返、ζ/D ↔ (U,V) 往返、∇·∇ = 拉普拉斯特徵值、剛體旋轉渦度：全部達機器精度。
- 層結靜止大氣 10 天後最大風速 < 1e-11 m/s。
- 無強迫斜壓流 20 天：乾空氣質量漂移 ~8e-7、總能量漂移 ~3e-7、軸向角動量漂移 ~4e-5。
- Δt = 1 h（外重力波 CFL ≫ 1）半隱式穩定。

## 3. 路線圖 / Roadmap

每一步都要先通過量化驗收才進下一步（沿用 `VALIDATION_PLAN.md` 精神）。

### R1 乾大氣氣候 / Dry climate — **完成 / done**（結果見 / results: `RESULTS_R1_DRY.md`）
- Held–Suarez (1994) 強迫，T21 / T42 長期積分（200 天 spin-up + 300–500 天平均）。
- 驗收：緯向平均 [u]、[T]、經圈流函數 ψ、渦動通量與 HS94 / 文獻 dry-core 比較（噴流強度與位置、熱帶地面東風、中緯地面西風、Hadley 與 Ferrel 胞）。
- Jablonowski–Williamson 斜壓波（確定性溫帶氣旋、鋒面發展）。

### R2 濕大氣水球 / Moist aquaplanet — **完成 / done**（結果見 / results: `RESULTS_R2_R5.md`）
- 水汽 q：格點上的 **形狀保持半拉格朗日** 輸送（三維軌跡、跨極點、三次內插 + Bermejo–Staniforth 限制器，q ≥ 0、不產生新極值），全球水量修正（NCAR CCM3 / ECMWF IFS 做法）。
- 物理（Frierson et al. 2006, 2007；參數取自 Isca `frierson` 測試案例）：兩流灰體輻射、簡化 Monin–Obukhov 地表通量、隱式 K-profile 邊界層、2.5 m slab ocean、50 hPa 以上能量守恆 sponge、Simplified Betts–Miller 對流、大尺度凝結與降水再蒸發。物理在每個動力步之後依序分裂作用。
- T21（300 日平均）與 T42（200 日平均）：P = E（4.24 / 4.28 mm/day）、ITCZ、副熱帶乾區、中緯度風暴路徑、Hadley 胞 ±6–8 × 10¹⁰ kg/s、噴流 33–39 m/s。

### R3 真實地表與季節 / Land, seasons, topography — **完成第一版，季風偏弱 / first version done, monsoons weak**
- ERA 地表高度與海陸遮罩、頻譜平滑地形、季節日照、Byrne & O'Gorman (2013) 隨水汽變化的灰體光學厚度、Merlis et al. (2013) q-flux、陸地 bucket 水文。
- **熱力學海冰**（Semtner 1976 零層模式）：混合層在 271.35 K 結冰，冰厚由傳導、表面與底部（q-flux）熱收支決定，能量精確守恆；取代只改反照率的舊作法。
- 結果：大陸季節溫差（西伯利亞夏冬差 41 K）、撒哈拉夏季高溫、西非季風（JJA 1.3 vs DJF 0.0 mm/day）、南美夏季雨季、三胞環流與季節性 ITCZ 移動。
- 已知不足：亞洲季風反向（印度、華南冬雨多於夏雨）。原因是灰體模式在深熱帶的陸地比鄰近海洋冷約 10 °C（陸地反照率 0.42、蒸發冷卻），而 slab 印度洋高達 34–38 °C，季風所需的海陸熱力對比反轉。陸地反照率敏感度實驗進行中；根本改善需要非灰體輻射與雲。

### R4 GPU 加速 / GPU acceleration — **完成 / done**
- WebGPU/WGSL（f32）：批次勒讓德轉換、workgroup Stockham FFT（雙精度 twiddle 表）、格點動力、半隱式求解、RAW 濾波、半拉格朗日水汽、完整灰體柱物理（含 SBM、海冰）、全球水量與質量修正；區域模式全部核心、亂流、地表、Kessler 與冰相微物理、開放側邊界。
- `npm run test:gpu`（headless Chromium + SwiftShader）：每一部分都對 CPU Float64 版本做逐場比較。

### R5 區域非靜力模式 / Regional non-hydrostatic model — **完成 / done**
- `src/regional/core.ts`：全可壓縮非靜力方程（u, v, w, θ, π′），C 網格，Wicker–Skamarock RK3 + 分裂聲波步，垂直隱式 w–π′，散度阻尼，5 階迎風通量型平流 + **Skamarock (2006) 正定通量限制器**（水物質），浮力與氣壓梯度用完整密度位溫 θρ = θ(1 + 0.61qv − Σ凝結物)，週期或 **開放側邊界**（Davies 鬆弛區）。
- 微物理：Kessler 暖雨；**六類冰相**（qv, qc, qr, qi, qs, qg；Lin et al. 1983、Rutledge & Hobbs 1983、Hong et al. 2004）：冰核化、冰／雪／霰的凝華與昇華（Bergeron 過程）、凇附、收集、自動轉換、Bigg 凍結、融化、−40 °C 均質凍結、沉降。0 °C 以上與 Kessler 完全相同。
- `src/regional/physics.ts`：Smagorinsky–Lilly 亂流、逐格點地表溫度與濕度的 bulk 通量、Newtonian 輻射冷卻。
- **單向巢狀**（`src/regional/nest.ts`）：由全球模式狀態初始化，側邊界隨全球模式時間更新；網頁可在地球上點選區域「放大」（1200 km／12 km 或 480 km／4 km 可解析對流）。
- 驗收：Straka 密度流（θ′ −9.66 K、鋒面 15.45 km）、超大胞分裂、15 km 熱帶氣旋快速增強至 ~40 m/s、巢狀區域降水與全球模式一致；GPU 與 CPU 逐場一致。

### R6 高解析颱風與局地 LES / High-resolution TC and local LES — **下一步 / next**
- 2–3 km 颱風（GPU）搭配冰相微物理：眼、眼牆、外圍雨帶，長時間積分觀察次級眼牆形成與眼牆置換是否自然出現。
- 50–250 m 格距的超大胞環境，觀察龍捲風旋生（tornadogenesis）是否自然出現。

### R7 輻射與雲 / Radiation and clouds — **規劃 / planned**
- 以非灰體（多頻帶，含水汽窗區）輻射與診斷雲取代灰體輻射，改善海陸熱力對比與亞洲季風。

## 4. 硬限制（沿用）/ Hard constraints (kept)

- 使用 SI 單位與真實物理方程；參數化方案必須有文獻出處與物理單位。
- 禁止任何「生成颱風／鋒面／三胞環流」之類的程式；只能給初始場、邊界條件、地表條件與物理源項。
- Eulerian 場是唯一的物理狀態；畫面上的粒子只是無質量示蹤粒子。
- 每層物理先通過量化驗收才加入下一層；不以加強阻尼掩蓋錯誤。
- 使用者介面繁體中文 + English 同時顯示（`UI_SPEC.md`）。

## 5. 參考文獻 / References

- Held, I. M., and M. J. Suarez, 1994: A proposal for the intercomparison of the dynamical cores of atmospheric general circulation models. BAMS, 75, 1825–1830.
- Hoskins, B. J., and A. J. Simmons, 1975: A multi-layer spectral model and the semi-implicit method. QJRMS, 101, 637–655.
- Simmons, A. J., and D. M. Burridge, 1981: An energy and angular-momentum conserving vertical finite-difference scheme and hybrid vertical coordinates. MWR, 109, 758–766.
- Williams, P. D., 2009: A proposed modification to the Robert–Asselin time filter. MWR, 137, 2538–2546.
- Frierson, D. M. W., I. M. Held, and P. Zurita-Gotor, 2006: A gray-radiation aquaplanet moist GCM. JAS, 63, 2548–2566.
- Jablonowski, C., and D. L. Williamson, 2006: A baroclinic instability test case for atmospheric model dynamical cores. QJRMS, 132, 2943–2975.
- Bryan, G. H., and J. M. Fritsch, 2002: A benchmark simulation for moist nonhydrostatic numerical models. MWR, 130, 2917–2928.
- Lin, Y.-L., R. D. Farley, and H. D. Orville, 1983: Bulk parameterization of the snow field in a cloud model. JCAM, 22, 1065–1092.
- Rutledge, S. A., and P. V. Hobbs, 1983: The mesoscale and microscale structure and organization of clouds and precipitation in midlatitude cyclones. VIII. JAS, 40, 1185–1206.
- Hong, S.-Y., J. Dudhia, and S.-H. Chen, 2004: A revised approach to ice microphysical processes for the bulk parameterization of clouds and precipitation. MWR, 132, 103–120.
- Skamarock, W. C., 2006: Positive-definite and monotonic limiters for unrestricted-time-step transport schemes. MWR, 134, 2241–2250.
- Semtner, A. J., 1976: A model for the thermodynamic growth of sea ice in numerical investigations of climate. JPO, 6, 379–389.
- Davies, H. C., 1976: A lateral boundary formulation for multi-level prediction models. QJRMS, 102, 405–418.
