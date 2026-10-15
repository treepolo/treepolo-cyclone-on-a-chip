# 模擬器總覽 / Simulator Overview

> 給之後讀這個 repo 的 AI（與人）：這份文件說明**這個模擬器是什麼、有哪些功能、用了哪些技術、程式在哪裡、哪些事有驗證、哪些沒有**。
> 它是地圖，不是進度表：進度與歷史在 `docs/HANDOFF.md`（逐項記錄，含使用者的要求與回饋）、待辦在 `docs/ROADMAP.md`、
> 架構決策在 `docs/ARCHITECTURE.md`、物理規格在 `docs/PHYSICS_SPEC.md`、局部加高／外圈的開發計畫在 `docs/DEV_PLAN_nesting.md`。
> 更新日期：2026-10-15（網頁版本 v58）。行號會漂移，以檔名與函式名為準。

## 0. 一句話

treepolo Cyclone on a Chip（晶片上的旋風）是個人用的**瀏覽器內 3D 大氣物理模擬器**：以真實方程與 SI 單位，
讓三胞環流、噴流、溫帶氣旋、季風、颱風（眼牆、雙眼牆）、中尺度對流、超大胞與龍捲**由方程自己長出來**。
全球譜模式負責行星尺度，區域非靜力模式負責 50 m–15 km 的風暴，兩者都能在 WebGPU 上跑（CPU Float64 版本是對照基準）。

**硬限制（所有改動都要遵守）**
1. 真實方程與 SI 單位；參數化必須有文獻依據。
2. **天氣不可硬編**：只能設定初始場、邊界條件、地表條件與物理源項；不可畫假雲、假渦旋。畫面上的軌跡粒子只是無質量示蹤，雲是由模式的水物質算出來的消光。
3. Eulerian 場是唯一物理狀態。
4. 每層物理先過量化驗收才加下一層；不以加強阻尼掩蓋錯誤。
5. 介面繁體中文＋English 並列（`docs/UI_SPEC.md`）。

**開發慣例（使用者的要求）**：測試只驗證程式（不崩潰、沒寫錯）；**模擬結果由使用者自己玩、自己判斷**，所以不要跑冗長的模擬對照
（近期的工作甚至要求完全不跑時間步，只做無時間步的測試）。對共用程式（尤其眼區細化）的修改只能是加法，並要證明行為不變。

## 1. 兩個網頁

| 頁面 | 檔案 | 內容 |
|---|---|---|
| 全球模式 | `index.html` → `src/app/main.ts`、`worker.ts`、`globe.ts` | WebGL2 地球、全球譜模式（Web Worker，WebGPU 或 CPU）、緯向剖面、點地球上某處開區域預報 |
| 區域模式 | `regional.html` → `src/app/regional/main.ts`、`worker.ts`、`volume.ts` | 3D 雲體積、圖表、互動工具、設定與網格即時調整、細化／外圈／局部加高、存檔與回放 |

發布方式：`tools/buildArtifact.mjs` 打包成 claude.ai Artifact（只有主頁拿得到 `window.claude` runtime；`regional.html` 直接開會轉到 `index.html#regional`）。
本機：`npm run serve`（`tools/serve.mjs`，預設 5173）。

## 2. 全球模式（`src/spectral/`、`src/model/`、`src/gpu/`）

- **動力核心**：旋轉球面上的靜力原始方程，球諧轉換法（Gaussian 網格，T21/T42/T63/T85/T170），預報 ζ、D、T、ln p_s；σ 座標（Simmons–Burridge 垂直離散）；
  半隱式 leapfrog＋Robert–Asselin–Williams 濾波＋隱式 ∇⁸ 超擴散（`dycore.ts`、`vertical.ts`）。
- **強迫與測試案例**：Held–Suarez（`heldSuarez.ts`）、Jablonowski–Williamson 斜壓波（`jablonowski.ts`）。
- **濕物理**（`src/model/moist/`）：形狀保持半拉格朗日水汽輸送（`semiLagrangian.ts`，q ≥ 0、不產生新極值）、灰體輻射、邊界層、Simplified Betts–Miller 對流、
  大尺度凝結、slab 海洋＋熱力學海冰、q-flux、陸地 bucket 水文；真實海陸與地形（`data/earth_t42.json`）、季節日照。
- **起轉狀態**：`data/spinup_earth_t42q.bin`（可載入任何解析度，`src/model/spinup.ts`）。
- **GPU**：`dycoreGpu.ts`、`transformGpu.ts`、`moistGpu.ts`，每部分對 CPU 參考解逐場驗證（`src/gpu/tests/`）。
- **已知限制**：亞洲季風在灰體輻射下反向（`RESULTS_R2_R5.md`）。
- **一向巢狀**（`src/regional/nest.ts`）：在地球上點地點，以當下全球場當區域模式的初始與側邊界（1200 km／12 km，或 480 km／4 km 可解析對流）。

## 3. 區域模式的核心（`src/regional/`）

### 3.1 動力
`core.ts`：全可壓縮、非靜力、笛卡兒 C 網格（CM1/WRF 一族）。預報 u、v、w、θ、π′（Exner 擾動）與水物質純量；水平均勻靜力基準態 θ₀(z)、qv₀、π₀、ρ₀。
Wicker–Skamarock RK3＋聲波分裂（垂直 w–π′ 為 Crank–Nicolson 逐柱隱式、散度阻尼）；5 階上風偏向通量平流（近蓋子／地面降階）；正定通量限制器用於水氣；
側邊界週期或開放（Davies 鬆弛帶 `relaxCells`、`relaxTau`；也可圓柱形鬆弛 `relaxCyl` 給巢狀）；頂部蓋子（w=0）加 Rayleigh 海綿層（`min(6 km, 0.3·頂)`，鬆弛到基準態或巢狀目標）。
**重要假設：垂直格距是均勻的**（`dz` 在 core/GPU/physics/cumulus/diagnostics 等上百處用純量）。因此「加高部分比主網格粗」需要拉伸垂直格距，尚未做（見 `DEV_PLAN_nesting.md` Step 4）。
座標系可 Galilean 移動（`shiftFrame`）、整格平移（`roll`，領域跟著風暴走）。科氏力在 f 平面；有背景風時地轉（科氏力作用在 u−ub）。

### 3.2 微物理與物理
- `ice.ts`：單矩六類（qv、qc、qr、qi、qs、qg；暖雨部分等於 Kessler，冰相依 Lin–Farley–Orville 1983、Rutledge–Hobbs 1983、Hong–Dudhia–Chen 2004），冰過飽和（預設開）與液態水過飽和（預設關）可在執行中開關。`kessler.ts`：Kessler 暖雨與 Weisman–Klemp 探空。
- `physics.ts`：Smagorinsky–Lilly 亂流（水平／垂直混合長度、Richardson 修正）、海面／地面通量（動量、感熱、水氣；最小風速 vmin；Donelan 風格 Cd）、
  陣風（Beljaars 1995 自由對流＋Redelsperger 2000 降雨陣風）、輻射（Newtonian 鬆弛或固定冷卻）、邊界層擾動；地表有海／陸遮罩、地表溫度（可塗）。
- `cumulus.ts`：尺度感知積雲參數化（Frierson 2007 簡化 Betts–Miller；3 km 以下關、12 km 以上全開；熱帶預設開）。
- `tropical.ts`、`supercell.ts`：颱風環境（Rotunno–Emanuel 1987 渦旋、條件不穩定熱帶探空）與龍捲超大胞環境（Weisman–Klemp 1982 探空、四分之一圓風速圖）。
  渦旋、龍捲只能由方程長出來。
- `guard.ts`、`stepControl.ts`：狀態保護（NaN 格放回環境、溫度／水／風／π 上下限；`T_FLOOR=100`、`T_CEIL=1e6`、`Q_CEIL=1`、`V_CEIL=1e5`、`PI_CEIL=20`）與自動步長
  （庫朗數目標 0.8、聲波限制、依最熱空氣縮短、互動後立即重算）。互動不設強度上限，物理上做不到時才在紀錄說明。

### 3.3 診斷與顯示量
`diagnostics.ts`（CPU）與 GPU 顯示核心（`regionalGpu.ts` 的 `readDisplay`、`readColumns`、`readRZ`，彼此逐項驗證）：柱合成（最大回波、紅外雲頂、降水率、UH、CAPE／CIN、風切、螺旋度、LCL、LI、STP／SCP、VIL、回波頂…）、
水平切面、自選垂直剖面、徑向—高度（颱風）、探空（斜溫圖＋風標＋指數）、方位平均、海平面等壓線、Hovmöller。
`display.ts`：光學量（消光係數 β = 1.5ρq/(ρ_p r_e)；Smith 1990 次網格雲量，RHc 隨格距 0.85→0.96），**只供顯示、不回饋模式**。
`storms.ts`：風暴目錄（渦旋＝平滑海平面氣壓極小；對流胞＝柱最大上升連通區），給每個風暴持續的身分、路徑、七／十級暴風半徑；`supercell.ts` 的 `StormTracker` 供領域跟隨。

### 3.4 互動與強迫
`forcing.ts`（風：推送／旋轉／輻合／氣旋／反氣旋，一次或持續；**一律用加的**，不覆寫）、`vortex.ts`（暖心／冷心渦旋，梯度風與靜力平衡）；
`worker.ts` 的 `interact`：暖泡冷池、增濕（乘／加／設相對濕度）、液態水／固態水（設／乘／加）、塗海／陸／海溫（設到指定溫度）、改環境風與濕度。
GPU 上用小核心直接編輯（不讀回），互動前存快照，發散時自動還原並暫停。範圍內強度一致，不向邊緣減弱。

### 3.5 軸對稱快速版
`axisym.ts`＋`axiDriver.ts`：同樣方程與物理，只算半徑–高度（r² 通量形式守恆角動量），CPU，數天只要幾分鐘，供篩選颱風參數；命令列 `node dist/tools/runAxisym.js`。

## 4. 區域網頁：設定、網格與巢狀（`src/app/regional/`）

### 4.1 設定即實驗
`setup.ts` 的 `RegionalSetup`：範圍、格距、頂、Δz、步長、邊界、跟隨風暴、緯度、海／陸、海溫、通量、探空、環境風、輻射、vmin、陣風、積雲、初始擾動…；原本的實驗是 `PRESETS`
（颱風 15／5／3 km、超大胞 2／1 km、龍捲 1 km／250 m、軸對稱）。`build.ts` 的 `buildModel()` 建模式，`setupForm.ts` 依欄位表產生面板（顯示 CAPE/LCL/SRH、格點數，超過 3000 萬格不可套用）。

### 4.2 模擬中改設定與網格（延續工作階段，時間、回放、圖表、持續風、粒子都不斷）
- **設定**（`liveSettings.ts`、`liveForm.ts`）：緯度、邊界、地表、海溫、通量、探空（保留對基準的偏差，近似）、環境風（新舊差加到整個風場）、輻射、物理、積雲…。
- **網格**（worker `regridTo`）：範圍縮小、模式頂加高／降低（新層沒有雲雨，蓋子與海綿層搬到新頂）、垂直格距；`refineInto` 把狀態內插進新網格。
- **細化／粗化**（`refine.ts`）：整區先粗後細（15→5 km 颱風、2→1 km 超大胞、1 km→250 m 龍捲），內插用立方根雲量與盒狀平滑。

### 4.3 三種巢狀（互相獨立，可同時使用）
| 功能 | 檔案 | 說明 |
|---|---|---|
| **眼區細化**（圓柱，雙向） | `twoway.ts`、`nestGpu.ts`、`eyeNest.ts` | 領域中心的圓柱內水平與垂直都更細（水平比 2–32、垂直比 1–8）；圓柱外鬆弛、內部每步回饋（cos² 漸變）；領域跟風暴走所以圓柱停在颱風眼上。**這是使用者最在意、不可被其他功能改壞的功能。** |
| **外圈**（範圍加大，單向） | `extension.ts`、`outerView.ts`，`twoway.ts`/`nestGpu.ts` 的 `x0,y0,ring` 選項 | 現在的模式原封不動當內圈，外面加一圈較粗的大網格（格距為內圈整數倍）；外圈領先步進，內圈邊界向外圈鬆弛；3D 畫面畫出外圈（金線標內圈邊），圖表只看內圈 |
| **局部加高**（方形，雙向） | `tall.ts`、`tallNest.ts`、`tallGpu.ts`、`tallForm.ts` | 選一塊方形範圍把模式頂加高：那一塊用自己的網格（水平／垂直同主網格或更細的整數分之一）從地面算到新頂，蓋子與海綿層在新頂；側邊目標在 Z1 以下取主網格、以上取環境；回饋只進主網格有的層；眼區在其內時也回饋進它。**不能比主網格粗** |

步進順序（`worker.ts` `stepGpu`/`stepCpu`）：外圈準備 → 主網格步＋眼區子步＋眼區回饋 → 局部加高的目標、子步、眼區→加高回饋、加高→主網格回饋。
存檔只存主網格（眼區、局部加高平均進主網格）；回放不含外圈與局部加高。

### 4.4 其他頁面功能
- **3D 雲體積檢視**（`volume.ts`，WebGL2 光線步進）：消光係數著色（雲水 150、冰 55、雪 27、霰 7.5、雨 3 m²/kg），光照在每個新畫面預先算（往太陽一次掃描＋往上一次，`light.ts`），
  佔據格跳過空氣、遮蔽的軌跡粒子、剖面切開、眼區／外圈／局部加高疊層、地圖式相機與自由飛行、指北針、手機版面。
  畫面只畫模式的場（不加程式化雲形），這是使用者明確要求。
- **衛星雲圖**（`satellite.ts`）：可見光（真實色彩）、紅外、水氣，由同一份顯示位元組從正上方光線步進。
- **圖表**（`charts.ts`、`chartDraw.ts`、`scales.ts`）：上述各診斷圖、固定色階（可自訂）、時間序列、風暴路徑；滑鼠工具。
- **工具**（`tools3d.ts`）：3D 地面點選放暖泡、冷池、增濕、水象、風、渦旋、塗地面；持續風畫在 3D。
- **存檔**（`saves.ts`）：IndexedDB＋ZIP 匯出入，列表只讀 meta；頁面內不用 `alert/confirm/prompt`（Artifact 檢視器不支援）。
- **回放 v2**（`replay.ts`、`replayCharts.ts`、`replayFile.ts`、`replayPanel.ts`、`src/regional/replayData.ts`）：依模式時間間隔記錄，三個層級（3D／加全部合成圖／加完整 16 位元狀態），可匯出載入並重算圖表。
- **其他**：步長控制（自動／手動）、速度控制（`pacer.ts`）、無人值守實驗（`runner.ts`）、任務挑戰（`missions.ts`）、效能分析按鈕。

## 5. 目錄地圖

| 路徑 | 內容 |
|---|---|
| `src/spectral/` | FFT、Gaussian 緯度、球諧轉換 |
| `src/model/` | 全球動力核心、強迫、濕物理（`moist/`）、起轉 |
| `src/regional/` | 區域核心與物理（見第 3 節）、巢狀幾何（`twoway.ts`、`tall.ts`、`nest.ts`）、細化（`refine.ts`）、診斷 |
| `src/gpu/` | WebGPU：全球、區域（`regionalGpu.ts` 約 2000 行 WGSL＋驅動）、巢狀耦合（`nestGpu.ts`、`tallGpu.ts`）；`tests/` 為 GPU 驗收 |
| `src/app/` | 全球頁面；`src/app/regional/` 區域頁面（worker、渲染器、面板、圖表） |
| `src/tools/` | Node 長期積分、軸對稱掃描、SVG 圖 |
| `src/tests/` | 驗收測試（`live.ts`、`outerView.ts`、`tall.ts` 無時間步；其餘會時間積分） |
| `tools/` | `serve.mjs`、`buildArtifact.mjs`、`gpuTest.mjs`、`renderCheck/`（合成體積渲染回歸） |
| `data/` | 地表、海洋熱傳輸、起轉狀態 |
| `docs/` | `HANDOFF.md`（交接與歷史）、`ROADMAP.md`、`ARCHITECTURE.md`、`PHYSICS_SPEC.md`、`VALIDATION_PLAN.md`、`UI_SPEC.md`、`RESULTS_*.md`、`DEV_PLAN_nesting.md`、本文件 |

資料流（區域頁）：`main.ts`（UI）⇄ `worker.ts`（模式與 GPU）以 `protocol.ts` 的訊息溝通；worker 每個畫面讀回顯示位元組與診斷，送 `RegionalFrame`；
`volume.ts`／`charts.ts` 只吃畫面，不碰模式。

## 6. 測試與驗證

- `npm run build`（tsc）；`npm test`（CPU 驗收，很久，含時間積分）；`npm run test:gpu`（SwiftShader 上的 WebGPU 驗收，`gpu-test.html?only=...`）。
- **無時間步測試**（快，約一秒）：`node dist/tests/live.js`（風形式、設定即時套用、網格改變、外圈）、`outerView.js`、`tall.js`。
- 渲染回歸：`node tools/renderCheck/render.cjs <前綴> main nest side outerfar outernest outerside tall tallside tallfoot tallnest`（無頭瀏覽器、合成體積；改 `volume.ts` 前後用 `cmp` 比 PNG，沒有新功能時必須逐位元相同）。
- GPU 一次性檢查（只 dispatch 核心，不步進）：頁面放 repo 根目錄、用 `node tools/gpuTest.mjs <頁面>` 跑、跑完刪。
- 環境只有軟體 GPU（比 GTX 1650 慢上百倍），所以長時間物理實驗不在本機跑。

**誠實的驗證狀態**：全球模式、區域核心、微物理、診斷在早期版本有量化驗收（`RESULTS_*.md`）。v50 之後的功能（互動擴充、氣旋／反氣旋風、模擬中改設定與網格、外圈、外圈顯示、
外圈＋眼區並用、局部加高）**只做了無時間步的測試與一次性核心檢查**，整套流程在真實頁面上的行為由使用者實測；`HANDOFF.md` 每一項都寫了「沒驗證」的部分。

## 7. 給 AI 的工作指引

1. 先讀 `docs/HANDOFF.md` 開頭（使用者偏好、溝通方式、發布流程、測試方式）與最後的「下一步」。
2. 用**正體中文**回覆，語氣溫暖簡潔；分開講驗證過什麼、沒驗證什麼；不要把推測說成事實；不丟沒根據的提案。
3. 方向性決定先討論；使用者說「開始」才動手；說「不要跑模擬」就一個時間步都不跑。
4. 不要改眼區細化的程式（`eyeNest.ts`、`twoway.ts` 圓柱行為、`nestGpu.ts` 眼區用法、worker 的 `eye` 分支語意），新功能寫在新檔。
5. 新程式碼不要假設垂直格距均勻之外的事，也盡量用 `m.zc[k]`／`m.zf[k]` 而不是 `(k+0.5)*dz`（為將來的拉伸格距）。
6. 功能必須在 3D 畫面（或該有的圖表）看得到、操作得到才算完成。
7. 提交結尾附 `Co-Authored-By` 與 `Claude-Session` 兩行；推到指定分支，不開 PR。
8. 已知待辦（使用者決定後才做）：垂直拉伸格距（讓加高部分比主網格粗）、外圈向內圈鬆弛、光的連鎖（外圈雲遮蔭內圈）、海洋混合層（海水熱含量）、2–3 km 颱風與龍捲 LES 的進一步驗證。
