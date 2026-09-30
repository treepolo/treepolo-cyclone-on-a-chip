# 交接文件 / Handoff

新對話請先讀這份，再讀 `docs/ROADMAP.md`（待辦與順序）與 `docs/ARCHITECTURE.md`。

## 使用者與溝通
- 用**正體中文**回覆，語氣溫暖、不要冷淡；進度報告簡潔。
- 使用者主要用 PC 玩（NVIDIA GTX 1650，Chrome，WebGPU）；手機 Samsung Note20（Mali GPU，會退回 CPU）。
- 使用者說：**直接一路做下去，不用每階段停下來**，全部做完再一次測試。但大的方向性決定仍要先討論（使用者曾因直接開工而中斷）。
- 使用者**不要**：垂直網格下密上疏、聲波子步 6→4、移動巢狀、GPU 深度優化、在他電腦上裝 Claude Code。
  龍捲項目若要用「地面附近垂直加密」必須先問。
- 原則：**天氣現象必須從方程自己長出來**，不可用程式畫假雲、假渦旋。
- **開發測試只驗證程式**（不崩潰、沒寫錯、參數沒錯）；**模擬結果由使用者自己玩、自己判斷**。每批做完告訴使用者什麼算正常、
  什麼算不正常。不要跑很久的模擬對照；真的需要時只跑一組、用 15 km，不要用軸對稱模式判斷 3D 樣子。
  截圖判斷雲的樣子時雲的不透明度要調到最大，看正側面與正上方，地面選「海陸（只看雲）」。

## 專案
- Repo `treepolo/treepolo-cyclone-on-a-chip`，分支 `claude/3d-earth-atmosphere-simulator-ncy8n4`（在此開發與推送，不要開 PR）。
- Commit 結尾：
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01WuWtj2MPxtKazZuiVe5hfv
  ```
  （新 session 的 Claude-Session 連結依系統提示為準。）
- 已發布的 Artifact（使用者玩的網址）：https://claude.ai/artifact/G2sPhxcsoxZj9twEiAEJhC
  - 更新：`npx tsc -p . && node tools/buildArtifact.mjs <scratchpad>/artifact`，再用 Artifact 工具 publish：
    `file_path=<scratchpad>/artifact/index.html`、`url=上面網址`、`files={...}` 只列有改的檔
    （路徑對應 repo 相對路徑，如 `"dist/app/main.js": "dist/app/main.js"`；起轉狀態用
    `"data/spinup_earth_t42q.b64.txt": "<scratchpad>/artifact/spinup_earth_t42q.b64.txt"`）。
  - capabilities 已宣告 `{downloads: true, db: {}}`（重新發布時省略 capabilities 會沿用）。
  - 若 publish 說某檔「not read as published」：先 `Artifact action=list scope=files url=...` 再發布。
  - Artifact 只有主頁（index.html）拿得到 `window.claude` runtime；regional.html 直接開啟會自動轉到
    `index.html#regional`（主頁內的區域模式疊加視窗），因此 db/downloads 可用。

## 架構速覽
- 全球模式：譜方法原始方程（`src/model/dycore.ts`）＋灰體濕物理（`src/model/moist/aquaplanet.ts`，SBM 對流、
  slab 海洋、海冰、q-flux）。GPU 版 `src/gpu/dycoreGpu.ts`、`moistGpu.ts`。預設 T21/T42/T85/T170。
  起轉狀態 `data/spinup_earth_t42q.bin`（7/8，T42＋觀測海洋熱傳輸，可載入任何解析度：`src/model/spinup.ts`）。
- 區域模式：全可壓縮非靜力 RK3＋聲波分裂（`src/regional/core.ts`），冰相微物理 `ice.ts`，物理 `physics.ts`，
  GPU 版 `src/gpu/regionalGpu.ts`。實驗是「設定」（`src/app/regional/setup.ts` 的 `RegionalSetup`，原本的實驗變成 `PRESETS`），
  由 `build.ts` 的 `buildModel()` 建模式；巢狀 `src/regional/nest.ts`。
- 頁面：`index.html` + `src/app/main.ts`/`worker.ts`/`globe.ts`；`regional.html` + `src/app/regional/main.ts`/`worker.ts`/`volume.ts`。
- 已完成的重要功能：GPU 效能優化（自動時間步長、θρ 預算、亂流只算第一 RK 階段、晴空凝結物跳過、效能分析按鈕）、
  先粗後細細化（`src/regional/refine.ts`，15→5 km 颱風、2→1 km 超大胞、1 km→250 m 龍捲）、
  存檔讀檔（`src/app/saves.ts`，IndexedDB＋ZIP 匯出匯入）、速度控制（`src/app/pacer.ts`）、
  無人值守實驗（`src/app/regional/runner.ts`，報告寫入 Artifact db 集合 `runs`，用 ArtifactData 讀）。

## 測試
- 伺服器：`PORT=5199 node tools/serve.mjs &`（serve 常在指令間被殺，和測試寫在同一條指令）。
- CPU：`npm test`（很久）或 `node dist/tests/regional.js`（38 項）。
- GPU（SwiftShader）：`node tools/gpuTest.mjs "gpu-test.html?only=regional"`（含 adaptive、refine、charts；機器忙時會超過預設
  10 分鐘，可設 `GPU_TEST_TIMEOUT=2400000`），`?only=charts`、`?only=spinup`、`?only=earth`、`?only=perf`。
- 端對端：scratchpad 裡用 playwright（`require(<npm root -g>/playwright)`，chromium 參數
  `--enable-unsafe-webgpu --use-webgpu-adapter=swiftshader --enable-features=Vulkan`，畫面加 `--use-gl=angle --use-angle=swiftshader`）。
- 我方環境只有軟體 GPU（比 GTX 1650 慢上百倍）：長時間物理實驗用 Node CPU 背景跑或請使用者用無人值守實驗。
- 注意：`pkill -f <pattern>` 會殺掉自己的 shell；不要對有未提交修改的檔案用 `git checkout`。

## 區域模式圖表（第 3 節，已完成）
- 協定：`ToRegionalWorker` 的 `{type:'charts', req: ChartRequest|null}`（地圖變數、切面層、剖面線、探空點、r–z 中心）、
  `{type:'volMode'}`、`{type:'tracers', n}`；`RegionalFrame` 帶 `charts`（maps / slice / section / sounding / rz）、`tracers`、
  `origin`（移動座標實驗的區域原點對地位置）與新統計量（dbzMax、uhMax、capeMax、storm 位置、1.5 km vt(r)）。
  全球頁面嵌入的區域模式不送 charts 請求，不增加成本。
- 診斷：`src/regional/diagnostics.ts`（CPU：柱合成 COL=10 值含地面氣塊 CAPE/CIN、切面、剖面、方位平均、合成圖）；GPU 顯示核心在
  `regionalGpu.ts`（`readDisplay` 同定義，`readColumns`、`readRZ`）。`?only=charts` 測 GPU 與 CPU 一致（含軌跡粒子）。
- 軌跡粒子：`src/regional/tracers.ts`（CPU）＋ `GpuRegional.advectTracers`（同步驟、同雜湊亂數）；只供顯示。
  3D 繪製在 `volume.ts`：先把粒子畫到離屏緩衝（顏色＋距離），光線追蹤到該距離時再疊上，雲會遮住後面的粒子。
- 介面：`src/app/regional/charts.ts`（各圖）、`chartDraw.ts`（色階、色標、等值線、箭頭、風標）。顏色：單一色相藍色序列、
  藍–灰–紅發散；雷達回波用 NWS 色階、紅外雲頂用強化灰階（領域慣例）。
- 測試方式：在 Node 用 CPU 先跑出成熟狀態寫成存檔（`packSave`），再用 playwright 匯入存檔截圖各圖（本環境 SwiftShader 太慢，跑不出成熟風暴）。

## 軸對稱快速版颱風模式（ROADMAP 3b，已完成）
- 模式：`src/regional/axisym.ts`（徑向 C 網格，v 用 r² 通量形式以守恆角動量；RK3＋聲波分裂、隱式 w–π′、5 階平流、正定限制器；
  Smagorinsky 圓柱座標混合、海面通量含最小風速 vmin、Newtonian 冷卻；冰相微物理直接沿用 `IceMicrophysics`（`MicroHost` 型別））。
- 網頁：實驗「颱風軸對稱快速版」（`axiDriver.ts`）在 worker 內以 CPU 跑，參數面板（海溫、緯度、Δr、初始風、lh、lv、Ck、vmin、輻射冷卻上限）；
  顯示時把半徑–高度場繞軸旋轉到 160×160 的顯示網格，所以 3D、地圖、切面、剖面、探空都能用；r–z 與 Hovmöller 用原始場。存檔讀檔可用。
- 命令列：`node dist/tools/runAxisym.js days=6 sweep=vmin:1,3,5`（worker_threads，每核心一組，CSV 在 results/axisym）。
- 測試：`node dist/tests/axisym.js`（6 項，已加入 `npm test`）。

## 颱風雨帶與雙眼牆（ROADMAP 第 4、5 節；15 km 與軸對稱已測，待 GPU）
- 選項：`RegionalPhysicsConfig.vmin`（地面通量最小風速）、`radConst`（固定對流層冷卻 K/s，平流層仍鬆弛），GPU 常數 `VMIN`、`RADC`；
  `tropicalSounding(sst, Ttrop, rhTop)`。worker 的 `tcEnv`（頁面面板送來、存檔裡保存）套用到 tc／tc_hr／tc_3。
- 工具：`runAxisym.js ... cases='base|vmin:4|radc:1.5+vmin:4'`（CSV 有核心／外圍雨量與雨環數）；
  `runTropicalCyclone.js 8 15000 1200000 out ice vmin=4 radc=1.5 rh12=0.6`（每 3 小時印核心／外圍雨量與外圍降雨面積比）。
- 結果：軸對稱篩選與雙眼牆見 RESULTS 4.3c，3D 15 km 三組對照見 RESULTS 4.3d。使用者 5 km 也沒有雨帶。
- **真正原因與修正（RESULTS 4.3e）**：RE87 中性探空 CAPE = 0。新預設 `snd: 'unstable'`（`unstableTropicalSounding`：邊界層混合均勻、
  對流層冷 3 K·sin、CAPE 約 1000 J/kg）＋ `radConst` 1.5 K/day；`LEGACY_TC`（RE87＋鬆弛）合併在舊存檔參數底下，舊存檔不變。
  命令列工具的預設仍是 RE87（文件裡的舊結果可重現），加 `snd=unstable radc=1.5`（3D）或 `unstable=1 radc=1.5`（軸對稱）用新環境。
  使用者覺得長時間的背景測試太慢：能用短測試證明的就不要跑幾小時的對照。
- 雨帶統計：颱風實驗的每個畫面帶 `stats.tcRain`（上一個模式小時的核心 < 60 km／外圍 100–300 km 平均降雨率、外圍 > 1 mm/h
  面積比；3D 在 worker 的 `tcRainUpdate`，軸對稱在 `AxiDriver.frame`）；頁面資訊列、時間序列「外圍雨量」、CSV、
  無人值守報告（`rain_core`、`rain_outer`、`wet_outer`）都有。
- 3 km 颱風（tc_3）：CPU 端 RegionalModel／RegionalPhysics 的工作陣列改成第一次 CPU 步進時才配置（GPU 跑時只保留狀態陣列）。
  worker 的 `isTc()` 之前漏了 tc_3（沒有中心氣壓、RMW、眼牆），已修正。

## 龍捲與互動（ROADMAP 第 6、7 節）
- `supercell.ts`：`TornadoEnv`（R、depth、U6、qvMax）、`TORNADO_DEFAULT`（新的強低層風切預設）、`TORNADO_WK82`（舊預設，
  沒有環境紀錄的舊存檔用它）；`kessler.ts` 的 `weismanKlempQ(qvMax)`。`runTornado.js ... env=wk82` 可重現舊結果。
- 龍捲偵測：worker `stats.tornado`（ζ、V、EF、位置），頁面 `tornadoWatch` 記事件；runner 的 tor_ef／tor_v／tor_zeta。
- 互動訊息：`perturb`（CPU 改完上傳 GPU）、`paint`（`RegionalPhysics.surface` 與 `GpuRegional.setSurface`，細化後重設）、
  `environment`（改完重建 GPU，因為阻尼層風與邊界目標在 GPU 表內）。圖表的滑鼠工具在 `charts.ts`（`MapTool`）。
- 自由飛行：`VolumeView.setCamera('fly')`。挑戰任務：`missions.ts`（localStorage 記完成，try/catch）。

## 使用者大需求清單（2026-09，六批，已完成）
1. 安全互動與介面：互動前快照、發散時自動還原並暫停（`worker.ts` `takeUndo`／`blowUp`）；3D 畫面上的執行／暫停／單步；
   相機（拖曳轉、右鍵或雙指平移、雙擊飛到該點、滾輪、WASD）。
2. 陣風（`physics.ts` `gustSpeed`：Beljaars 1995 自由對流＋Redelsperger 2000 降雨陣風，CPU/GPU）、信風（`build.ts` `envWind`；
   有科氏力時背景風是地轉的：`RegionalConfig.geostrophic`，科氏力作用在 u−ub、v−vb）。
3. 設定：`setup.ts`（所有條件＋預設組合、`sanitize`、`autoDt`）、`build.ts`（建模式、舊存檔對應 `setupFromLegacy`、細化 `refinedSetup`）、
   右側面板 `setupForm.ts`（欄位表產生、依條件顯示／隱藏、CAPE/LCL/SRH 與格點數、超過 3000 萬格不能套用）。
   邊界週期／開放、跟隨風暴、海／陸、診斷顯示海陸比例；跟隨時地表（塗的陸地）跟著地面移動，開放邊界新進的格柱取環境場。
4. 多風暴：`src/regional/storms.ts`（渦旋：海平面氣壓平滑 30 km 的極小、比中位數低 ≥ 2 hPa、150 km 內氣旋式環流 ≥ 3 m/s、相距 ≥ 200 km；
   對流胞：柱最大上升 ≥ 10 m/s 的連通區；對地座標配對、看到兩次才顯示），風暴出現前不追蹤、不跟隨。頁面清單點選看單一風暴時間序列、
   3D 標籤、所有路徑。中心氣壓是海平面氣壓（`dpEnv` 也是）。新圖：可見光、水氣、可降水量、參數化對流降水率、水平輻散切面。
5. 3D 互動：`tools3d.ts`（工具列、預覽圈、持續風畫在 3D）、`volume.ts`（`toolActive`、`onToolDrag`、`project`）；
   `src/regional/forcing.ts`（推送／旋轉／輻合的風，一次或持續＝向目標鬆弛 τ 10 分，CPU pre-step 與 GPU kernel 同公式）；
   增濕／變乾、暖泡冷池可指定高度半徑。
6. 積雲參數化：`src/regional/cumulus.ts`（Frierson 2007 簡化 Betts–Miller：τ 2 h、RH 0.7、能量一致；深對流降水變成雲底下的雨水，
   20% 留在雲頂當冰（外流卷雲來源）；太乾時用 Frierson 的淺對流把邊界層水氣往上送；尺度感知 3 km 以下關、12 km 以上全開）。
   GPU kernel 在每步開頭（順序：邊界層擾動→積雲→持續風，與 CPU pre-step 相同）。3D 的次網格積雲只是顯示：每個對流格柱依雲量
   機率（每 30 分鐘重抽）畫一根塔，可在「顯示」關掉。熱帶預設（tc 系列）預設開啟。
7. 使用者回饋（第 27 版後）：次網格對流塔顯示已拿掉（使用者覺得閃爍、很怪）；七級／十級暴風半徑（`storms.ts` `galeRadii`，
   16 方向最外圈平均，最低層對地風）；3D 重播（`replay.ts`，依裝置記憶體上限、大網格存半解析度、滿了就隔一張丟）；
   可見光雲圖（`charts.ts` `drawVisible`：雲反照率＋光學厚度 1 的雲頂高度做坡度光照與陰影，細化插值）；
   次網格雲量（`src/regional/display.ts`，Smith 1990 三角分布，RHc 依格距 15 km 0.85→1 km 0.96，只畫出來）；
   3D 渲染改成依消光係數（雲水 150、冰 55、雪 27 m²/kg…，byte = (β/0.3)^(1/3)），光照在頁面每個新畫面預先算
   （往太陽方向一次掃描＋往上一次；`volume.ts` `computeLight`），不再逐像素往太陽取樣；地圖式相機（左鍵抓地面、右鍵轉向傾斜、
   滾輪往游標縮放，畫面上方跟著方位，正上方不鎖定）、預設北方朝上、指北針（點一下北方朝上）、風暴標籤開關。
8. 使用者回饋（第 28 版後，第 29 版）：
   - 互動強度可選（`tools3d.ts`：暖泡冷池用溫度幅度或熱含量 ρ cp A R² H 4π(1/6−1/π²) 指定，增濕倍數、海溫變化量）。
   - 眼區細化（「眼牆與風眼細化」區塊，`worker.ts` `refineEye`）：以偵測到的渦旋為中心的開放邊界盒子（邊界向細化當下狀態鬆弛、
     跟隨風暴）；渦旋還沒被確認兩次（剛載入存檔）時會先再分析一次。
   - 粗化（`build.ts` `coarsenedSetup`、`refine.ts` `coarsenInto` 盒平均＋邊緣 cos² 混合）：細化過的回到原網格（父網格堆疊最多 3 層，
     細化區貼回去），沒有父網格就平均到 2 或 3 倍格距。
   - 相機：轉向不限角度（可在眼裡往上看）、視野廣度 20–140°、按住滾輪拖曳＝轉向（擋掉 Chrome 自動捲動）。
   - 3D 雲（`volume.ts` `dens`）：格點存平均消光，格內用「雲量 = 平均 / 雲內消光」決定哪裡有雲（雜訊值均勻分布，所以格平均不變）；
     雲內消光依種類：層狀液態 0.03、對流液態 0.09、冰 0.0011 1/m。種類來自模式：上升速度（依格距換算，3 km 以下 1.5 m/s，15 km 0.3 m/s）
     決定花椰菜狀翻滾紋理（越強越細碎、邊緣越銳利），靜止液態雲是團塊，冰（砧、卷雲）是軟邊薄片。顯示位元組多了 w（`wByte`）與
     冰比例（`iceByte`，雲位元組 0 時為 0），CPU/GPU 一致（GPU 測試逐格比對）。雜訊 64³、每 8 格重複、軸轉 27° 避免格線。
     粗步穿過晴空、進雲退一步細步，細步時往太陽取一點算自陰影（含多次散射尾巴）。模式計算量完全沒變。
   - 垂直誇大滑桿（1–15×，大區域預設 8×，原本 12×）；區域外地面依實驗是海或陸，不再霧化，只有很遠的地平線融進天空。
   - 圖表放大：地圖（水平切面、合成圖、可見光）在框內縮放，座標軸跟著變；其他圖（剖面、探空、時間序列…）整張放大。
     滾輪縮放、右鍵或中鍵拖曳平移、手機雙指、雙擊還原。
9. 使用者回饋（第 29 版後，第 30 版）：第 29 版雲太規則（雜訊每 8 格重複）、格與格的雲各自獨立。改成：
   - 雜訊用真實大小（與格距無關）：雲群 2.5–30 km（兩個不相關大小與角度的拼貼 40 km、64.7 km 相加，不會重複）＋單朵雲與泡泡
     0.4–1.5 km（6 km 拼貼）；兩者的加權和用解析的分布函數轉回均勻（`u2`／`invU2`），所以格平均仍是模式值。
   - 比一個像素小的細節換成它的精確平均（遠看是細顆粒，近看是一朵朵雲）；細步長近處是泡泡的 1/4、遠處約一個像素。
   - 花椰菜是光照：上升的塔（液態或正在結冰都算）由泡泡組成，往太陽方向取一點算自陰影，泡泡之間的縫（泡泡值低）少受天空光；
     冰只有在幾乎不上升時才攤成薄片（砧、卷雲）。雲裡的降水（霰、雨）跟著雲的形狀，雲底下是平滑雨幕。
   - 3D 剖面（圖表列「剖面」）：南北向／東西向／水平／沿剖面線 A–B，位置滑桿、切掉另一側；被切掉的部分不擋光（光照重算，
     停手 0.12 秒後）。15 km 的遍地一格一格對流是模式本身（格點對流），不是渲染。
10. 使用者回饋（第 30 版後，第 31 版）：強烈發展的積雲／積雨雲要「飽滿」，破碎只屬於弱、短命、消散中的雲。改成：
    - 頁面在每個新畫面算每格的「對流活躍度」（`volume.ts` `convectionVigour`，取代體積貼圖的 w 通道）：本格或周圍 3×3×3 的上升、
      水平 4 km 內任一格柱的最強上升（塔的側面與過衝雲頂本身幾乎不上升）、或周圍多為液態雲且不下沉；門檻依格距（3 km 以下 1.5 m/s，
      15 km 0.3 m/s）。下沉、稀薄、遠離上升氣流的雲才碎；幾乎不上升、遠離上升氣流的冰雲是薄片（砧）。
    - 活躍的雲畫成實心雲體（模式場超過雲邊值約 0.04 g/m³），表面用兩個不相關大小與角度的氣泡雜訊（約 4 km 雲塔、1.5 km 泡泡）往外鼓，
      只在雲體旁邊、主要往外，所以不會中空、不會有漂浮碎片；雲面用二分法找準，法向量由邊界函數的梯度，光照包覆較軟（雲內散射）。
      試過真的球體泡泡：像一堆球、太假又慢 10 倍，已放棄。
    - 效能（SwiftShader 同畫面）：第 30 版 1.9 s／張，這版約 3.4–3.9 s（約 1.9 倍）；只在畫面改變時重畫，模式本身不變。
      「顯示」區多了「雲的細節」開關（關掉就沒有雲塊與雲面光照，給手機或較慢的顯卡）。
11. 使用者回饋（第 31 版後，第 32 版）：要分形感（大、中、小、迷你團塊），而且每次渲染要連續（團塊往上長、消散也連續），不加效能。
    - 分形：雲塔雜訊（17 km 拼貼）團塊約 4、2、1 km，泡泡雜訊（2.12 km 拼貼）約 0.5、0.25、0.13 km；每層鼓起量與大小成正比
      （自相似，每個大小在光影上一樣明顯）；泡泡層在中間大小約兩個像素以上才出現。查詢次數與第 31 版相同（計時相同）。
    - 連續：worker 的畫面多了 `anchor`（原點減去隨框架速度的漂移：只在網格滾動或細化時跳，雜訊跟著雲，不會因為跟隨風暴整片換掉）；
      頁面 `CloudDetail`（`volume.ts`）：活躍度有記憶（跟著場的跳動搬移，e 折 15 分），停止上升的塔慢慢變碎；活躍雲的團塊隨上升氣流
      往上移（活躍雲格平均 w 的一半，最多 8 m/s，所有團塊一起平移，不花效能）；重播存每張的 [活躍度, 冰] 與 anchor、上升量，所以也連續。
12. 使用者回饋（第 32 版後，第 33 版）：大團塊彼此太連續（一個粗糙表面的大團塊），大中小之間跳太多；團塊大小不該事先決定，要由氣流決定。
    - 團塊大小由氣流：頁面算每格的「羽流上升速度」（本格或周圍的上升、以及 4 km 內最強上升的 0.6 倍，依格距放大；byte sqrt(w/30)），
      最大團塊 ≈ 150 s × 羽流上升（0.25–8.5 km：弱積雲小、強雲塔數公里）。雲的表面本身幾乎不上升，所以用所屬羽流，不用本地 w
      （試過本地 w：整片變成細鱗片）。
    - 每個大小分開加權：新的 `uOct` 雜訊每個通道一個大小（Worley 4、8、16、2 個/拼貼），兩個拼貼（17 km、2.12 km）共 8.5、4.25、2.1、1.06、
      0.53、0.27、0.13 km 七個大小，各自鼓起 = 大小的一半（沒有上限），只保留 ≤ 羽流大小、< 約 2 格（格點解析的由模式自己畫）、> 約 2 像素的。
      大團塊有明顯的縫，不再融成一片。
    - 體積貼圖 b 通道改成「雲的種類」（0.5 + 活躍/2 − 冰片/2：1 活躍、0.5 弱液態、0 冰片），a 通道改成羽流上升；重播存同樣的兩個 byte。
    - 效能（SwiftShader 同時同畫面對照第 32 版）：超大胞 5.8 s 對 6.0 s，整個颱風 7.0 s 對 6.4 s；省下的來自：遠離雲面不查雜訊、只替每條光線
      第一個雲面找準與算法向量、晴空粗步 120→88。
13. 使用者回饋（第 33 版後，第 34 版）：團塊在不同方向的形狀應不同（上升對阻力、重力），大小不同影響不同；向光與背光之間有割裂感、背光缺細節。
    - 團塊形狀：泡泡雜訊用不對稱距離產生（`cloudNoise` 的 `worley(C, hor, below)`）：中心以上是圓頂、以下被壓扁、比高更寬，越大越明顯
      （8.5 km：寬 1/0.8、下方 1/2.4；0.13 km 幾乎是球）。大的一組 `uOct`（17 km 拼貼）、小的一組 `uOctS`（2.12 km）。只在產生時算，執行時沒有成本；
      產生也變快（格點值先算好、Worley 環繞索引先查表）：0.73 s，第 33 版 0.89 s。
    - 光影：雲面光照的觸發不再依前一步（原本有時有、有時沒有，鄰近像素跳動）；光包覆更遠、多次散射尾巴更長（柔和的明暗交界）；
      背光面：天空光依朝上程度、地面反光依朝下程度、團塊之間的縫暗一點（用已算好的團塊值），所以背光面也看得到團塊。
      雲面的光照係數延續到後面的樣本，不會因為有沒有找到雲面而突變。
    - 效能：超大胞 6.0–6.25 s（第 33 版 5.8–6.3），颱風 7.3–7.5 s（7.0–7.1）。
14. 使用者回饋（第 34 版後，第 35 版）：大團塊不是扁的，是渾圓飽滿的（第 34 版的扁帽子是錯的）；破碎的雲要比較透光。
    - 團塊：大的略高於寬（上升中被往上拉長，`worley(C, 1.03–1.08, 1.15–1.3)`），與下方雲體接合處略平；小的是球。鼓起 = 大小的 0.7
      （原 0.5），更偏向外凸（雜訊中位移 0.28，原 0.35）。
    - 弱雲的雲內含水量依周圍雲量：零星碎片約 0.05 g/m³（半透明，混入乾空氣稀釋），成片的層雲約 0.2 g/m³；格平均不變，所以碎片變淡、
      略散開；邊緣也較軟。
    - 效能：超大胞 6.2–6.5 s、颱風 7.5 s（第 34 版 6.0–6.25、7.3–7.5），持平。
15. 使用者回饋（第 35 版後，第 36 版）：用舊存檔看，程式化的雲形（第 29–35 版的雜訊碎片、團塊、薄片）違反物理原則、要關掉；有謎之鱗片、
    高層雲像井字網、積雲發黑有色差、超卡。**已全部移除**：3D 只畫模式算出的格點平均消光（三線性內插），沒有任何雜訊或另外編的形狀；
    次網格雲（Smith）保留，畫成薄的半透明雲。15 km 颱風第 1 分鐘那片「工整的雲」證實是最低層 RH 0.88 > RHc 0.85 的均勻次網格雲被雜訊切成的，
    不是積雲參數化（那時還沒作用）。移除的：`cloudNoise`、`convectionVigour`、`CloudDetail`、雲的細節開關、畫面的 anchor、顯示用的 w／冰
    位元組（worker、axiDriver、GPU 顯示 kernel 只打包雲與降水）。體積貼圖改 RG8。效能（SwiftShader 同畫面）：颱風 7.5 s → 0.9 s，超大胞
    6.3 s → 1.2 s。原則：模式網格解析不了的細節不畫（花椰菜要在 1 km 以下的細化網格才會由模式自己算出來）。
16. 眼區細化面板改成自由輸入（盒子 km、Δx km、Δz m），打開時依最大風半徑建議（約 5 倍 RMW、RMW/15），顯示格點數與「每模式秒計算量約為目前的
    幾倍」（格點數／時間步長），以及格距不夠細、盒子裝不下眼牆的警告。**下一步（使用者已決定）**：眼牆細化要在原本的空間中做雙向巢狀，
    範圍是圓柱，水平與垂直都細化；外圍區域跟隨風暴、圓柱固定在外圍區域中心（不是在網格上移動的巢狀）。
17. 圓柱形雙向巢狀眼區細化（第 37 版）：取代第 16 項的獨立開放盒子。核心 `src/regional/twoway.ts`（`nestGeometry`：水平比 r 2–32、垂直比 rz 1–8，
    細網格正方盒對齊外圍格點、置於外圍區域中心，≤ 1024 格寬、≤ 200 層；`nestTargets`：圓柱外的鬆弛環與頂部海綿層的目標＝外圍場在空間
    三線性、時間在外圍步前後之間內插，以距基本剖面的偏差內插；`nestFeedback`：R − Wf 以內外圍取細網格的格平均，cos² 漸變到 R）；GPU 耦合
    `src/gpu/nestGpu.ts`（`GpuNest.step(n)`：外圍一步、n 個內部步（每步中點的目標）、回饋，全在 GPU）；worker 整合 `src/app/regional/eyeNest.ts`
    （`EyeNest.build`：同一設定、同頂、外圍現在的基本風、狀態內插；`cpuStep`；`gpuUp`；`fit`：內部步數＝外圍 Δt ÷ 內部可用 Δt（CFL 0.8、
    聲波限制）；`roll`：外圍整格平移時細網格跟著平移 r 倍、移入的欄由外圍內插；`surfaceFrom`；`setForcings`；`volume`）。worker：
    `nestStart`／`nestStop` 訊息、`stepGpu`／`stepCpu`、`nestFit`、`moveDomain`（由 `followStorm` 拆出）、`startNest`（沒有追蹤就開始跟隨風暴、
    先整格平移把風暴放到中心）；有巢狀時 `followStorm` 只要風暴偏離中心超過一格或 0.15 R 就整格平移（`StormTracker.shift` 同步追蹤位置）；
    互動（暖泡、增濕、一次風、環境改變、塗地面、持續風）同時作用在細網格；還原互動時細網格由還原後的外圍重建；新實驗、載入、細化、
    粗化時停止；存檔只存外圍。畫面：`frame.nest`（細網格的雲與降水位元組＋幾何），`volume.ts` 第二組 3D 貼圖與光照（光線離開細網格盒子
    時接外圍的光學厚度），圓柱內以回饋的 cos² 權重混合、細網格內用較細的步長；重播也存細網格（大的減半）。面板：半徑 km、Δx km、Δz m，
    顯示實際格距（外圍的 1/r、1/rz）、格點數、內部步數、每模式秒計算量倍數；依 RMW 建議（R ≈ 2.5 RMW、Δx ≈ RMW/20、Δz 外圍一半）。
    內插改良（也用在整區細化 `refineInto`）：(a) 雲與降水粒子以立方根內插再立方（邊緣像外圍畫面一樣收掉，不會沿外圍格面變成方塊），整區
    一個係數保持質量；(b) 所有場再做一個外圍格寬的置中盒狀平滑（三線性內插在外圍格面有摺痕，細網格和它的畫面會看到階梯；以距基本
    剖面偏差平滑、邊緣線性延伸）；(c) `makeSampler` 在最低層中心以下、最高層中心以上半層改成線性延伸（原本夾住）；GPU `samp` 同步。
    已知：剛開始細化時圓柱內薄的次網格雲比外圍少（RHc 依格距，格子越細門檻越高），細網格自己長出變化後才補上。
18. 使用者回饋（第 37 版後，第 38 版）：3D 畫面的雜訊（剖面上、眼區細化圓柱內一片黑白顆粒，還有圓弧花紋）。原因（`volume.ts` 光線步進）：
    (a) 細步數有上限（170／260），颱風大片薄卷雲就用完，進到眼牆時變回約 14 km 的粗步、每個像素隨機起點 → 深入密雲的深度不同、
    光照不同 → 椒鹽雜訊（細網格步長更小，圓柱內最先用完）；(b) 就算細步，第一個落在密雲裡的樣本在半格內的位置隨機／規律 →
    顆粒或等高線花紋；(c) 光照貼圖在體素中心，密雲表面介於亮的空體素與暗的雲體素之間、陰影邊界逐體素是或否 → 網格狀花紋與鋸齒。
    改法：佔據網格 `buildOccupancy`（8×8×Bz 格一塊、含 26 鄰塊，R8 NEAREST 貼圖）讓晴空粗步不會跨過任何雲；拿掉細步上限（上限 640 次，
    薄雲步長依光學厚度最長一格）；進入新的雲段退回上一個晴空點；一步從下方越過固定消光（每半格 0.3 光學厚度）就二分法找出表面、
    從表面起以四分之一步繼續（`refines` 每條光線最多 6 次）；光照在往太陽一格、往上一格處查，並在貼圖第 0、1 層 mipmap 之間取
    （`textureLod` 0.6，GPU `generateMipmap`，陰影邊緣柔化約一格；原本試過 CPU 模糊，每幀 +1.5 s，不用）。退回時連軌跡粒子的疊加一起還原。
    量測（scratchpad `pngnoise.mjs`：像素與四鄰平均差）：合成颱風近看 1.34 → 0.28；軸對稱成熟颱風剖面近看 1.15 → 0.36。
- 測試：`node dist/tests/regional.js` 43 項（含雙向巢狀 4 項、雨柱內插 1 項）；GPU `?only=twoway` 3 項、`?only=rbasic` 15 項、`?only=charts`、
  `?only=refine`；瀏覽器：scratchpad `e2e/nest.cjs`（颱風 15 km 開始／單步／執行／暖泡／停止）、`e2e/nest2.cjs`（超大胞 25 分鐘後開細化、側視）。

19. 使用者回饋（第 38 版後，第 39 版）：
    (a) 互動要好幾個畫面後才在紀錄出現、才生效：互動訊息要等步進迴圈的忙碌鎖（GPU 批次＋畫面讀回）。改成 GPU 上的互動直接用 GPU 核心
    （`GpuRegional.edit` 暖泡／加濕、`windOnce` 一次性風、`snapshot`／`restoreSnapshot` 撤銷），步進迴圈只是在等 GPU 時（`stepping`）不必等鎖，
    紀錄立刻回、下一個畫面立刻送（`frameAsap`）；其他訊息的忙碌等待改成 `waitIdle()`（`waiting` 計數讓步進迴圈先讓位）。GPU 測試 `?only=edit` 2 項。
    (b) 第 38 版的橫紋：我當時誤判為雨帶；真正原因是光線在出口（地面、剖面）前最後一段不滿一步的路徑沒有合成，亮暗隨高度規律變化。改成最後一步
    截到出口 t1；細步改為各向異性半格（依射線跨過最快的軸）。(c) 卡頓：第 39 版先做了「相機移動時畫四分之一像素」，錯了，見第 20 項（已拿掉）。
    (d) 衛星雲圖（`satellite.ts`，獨立 WebGL2）：用 3D 視圖同一份顯示位元組，從正上方逐像素往下光線步進，解析度 = 地圖像素 × dpr（每邊上限 2048）；
    每步四分之一層（巢狀圓柱內用細層），樣本高度各欄相同；雲量與光照都用三次 B 樣條重建（8 次三線性取樣，Sigg & Hadwiger 2005），
    三線性在格面上的折角在螢幕解析度下會變成格子紋（放大時看得到），B 樣條沒有。可見光（真實色彩）：相似性縮放消光 β* = 0.15β（視線與往太陽
    都用，厚度 τ 的雲層反照約 τ/(τ+7.7)，薄卷雲透出海面），實際比例（不誇大垂直）的光照體積（`light.ts` 的 `sunSkyLight`／`nestSunSkyLight`，
    與 `volume.ts` 共用；陰影長度正確），雲光 = 0.8√(往太陽透過率) + 天空光，海陸反照率、瑞利路徑輻射（尺度高 8 km）、sRGB gamma。
    紅外線：吸收光學厚度為可見光一半，溫度用基本態 T(z)（`ChartData.tz`），晴空用最低層氣溫（新地圖 `sfcT`），由 T⁴ 平均反算亮溫，
    16 位元寫進 RGBA8 再解碼；滑鼠讀值加該像素亮溫。水氣圖改平滑內插。沒有 WebGL2 時退回舊的欄位圖。同一畫面、同一視窗不重畫（快取）。
    (e) 新觀測圖（合成圖選單分組：衛星／雷達與降水／地面／劇烈天氣）：地面氣溫與露點、0–6 km 風切、0–1／0–3 km 風暴相對螺旋度（Bunkers 右移胞，
    `windIndices`：0–6 km 每 500 m 取樣平均、螺旋度每 250 m）、LCL、LI（基本態最接近 500 hPa 的層）、STP／SCP（固定層、地面氣塊）、18 dBZ 回波頂、
    VIL（56 dBZ 上限）、UH 軌跡與地面最大風速軌跡（worker `swath`：每個畫面取樣的最大值，跟著網域平移，新 run／細化重設；CPU 後端的 UH 軌跡只在
    顯示圖表時更新）。GPU 欄位核心 COL 14 → 21（`C.shear`…`C.vil`），CPU `columnDiagnostics` 同定義；軸對稱模式也算（徑向／切向分量，指數與軸向無關）。
    水平切面加「925–200 hPa 等壓面圖」：取基本態氣壓最接近的模式層，白色等高線 = 該層氣壓與虛溫用測高公式推到等壓面的高度；切面變數加溫度 T。
    探空的風切／螺旋度改用同一個 `windIndices`，並加 LI。
- 測試（第 39 版）：GPU `?only=charts` 11 項（新增：風切、螺旋度、VIL 與 CPU 相差 < 1e-4／1e-3，LCL、回波頂逐欄相同，LI < 0.02 K）、`?only=edit` 2、
  `?only=rbasic` 15、`?only=twoway` 3、`?only=refine` 4；瀏覽器 scratchpad `e2e/charts39.cjs`（每張合成圖、850／500／200 hPa、探空，無錯誤、無卡在等待）、
  `e2e/latency.cjs`（互動紀錄 8–38 ms 內出現）、軸對稱模式同一腳本（`charts39.cjs tc_axi`：先切到圖表再執行，軟體渲染的 3D 跟不上它的畫面；
  它沒有積雲方案，`cuRain` 給 0，原本會一直等待）、`e2e/sat.cjs`＋`scratch-sat.html`（衛星渲染器單獨測，合成場或 Node 跑的超大胞位元組；測完要刪）。

20. 使用者回饋（第 39 版後，第 40 版）：「開著區域模式不動、什麼都沒跑，電腦就卡死」是 bug，不是效能問題；第 39 版的「移動時降解析度」
    移動時畫面全黑（把 FBO blit 到預設畫布，Windows 上預設畫布是多重取樣的，不允許 → 黑）。原因（推斷，這裡沒有 Windows 可驗證，但與症狀完全吻合）：
    Windows 的 Chrome 把 WebGL 轉成 D3D HLSL（ANGLE），迴圈的呼叫圖裡有需要梯度的取樣（隱含 LOD 的 `texture()`）時整個迴圈標成
    `[unroll]`；第 38 版的光線步進是 640 次迴圈、裡面再包 8 次二分法迴圈和多次取樣，展開後 FXC 編譯要很久、吃大量 CPU 與記憶體，
    頁面一打開（建構時編譯著色器）電腦就卡死；Linux 的 SwiftShader 測試不會遇到。改法：`volume.ts`、`satellite.ts`、`globe.ts`
    所有著色器取樣改成明確 LOD 的 `textureLod(…, 0.0)`（這些貼圖沒有 mipmap，結果逐像素相同；光照貼圖本來就是 `textureLod` 0.6），
    迴圈就能保持 `[loop]`。拿掉移動時降解析度的整段程式。全球頁的「→ 區域模式」連結開啟覆蓋層時，全球模式也暫停（和選點開巢狀一樣）。
    **規則：著色器迴圈裡一律用 `textureLod`／`texelFetch`，不要用 `texture()`。**
- 測試（第 40 版）：同一場景 3D 畫面與第 39 版逐像素相同；衛星渲染器與全球頁地球著色器編譯執行無錯誤；閒置頁面 60 秒內沒有任何重畫與畫面訊息。

## 文件截圖
- `docs/results/charts_*.png`（RESULTS 4.6）：在 Node 用 CPU 跑出成熟狀態存檔（超大胞 `packSave`；軸對稱用 `AxiDriver`＋`packSave`），
  再用 playwright 匯入頁面、切換圖表、只截 `#stage` 區域。

## 下一步
- 等使用者玩第 40 版：卡死修正（見 20，若仍卡，請使用者開 chrome://gpu 看看有無錯誤、或告訴我是哪一步開始卡）；衛星雲圖與新觀測圖（見 19）、互動即時生效、橫紋修正；第 37 版的圓柱雙向巢狀眼區細化（見 17）；「參數化雲」若是指積雲參數化方案，問使用者要不要預設關掉（第 36 版報告已問，待回覆）。
- 等使用者玩完六批的回饋（模擬樣子由使用者判斷）：15 km 颱風＋積雲方案的樣子（遍地對流、外流卷雲）、陣風與信風、多風暴清單、3D 工具。
- 已答應但延後：龍捲的地面附近垂直加密（使用者同意過，排在颱風之後）。
- 「放山」需要地形座標（大改動，使用者說不要）。
