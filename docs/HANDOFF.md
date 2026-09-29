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
- 測試：`node dist/tests/regional.js` 38 項；GPU `?only=rbasic` 15 項（含陣風＋信風、積雲、持續風）、`?only=charts` 10 項（含新雲圖、3D 顯示位元組）。

## 文件截圖
- `docs/results/charts_*.png`（RESULTS 4.6）：在 Node 用 CPU 跑出成熟狀態存檔（超大胞 `packSave`；軸對稱用 `AxiDriver`＋`packSave`），
  再用 playwright 匯入頁面、切換圖表、只截 `#stage` 區域。

## 下一步
- 等使用者玩第 30 版的回饋：雲的形狀（不規則、雲群、近看的泡泡光影）、3D 剖面；第 29 版的垂直誇大、眼區細化與粗化、圖表放大。
- 等使用者玩完六批的回饋（模擬樣子由使用者判斷）：15 km 颱風＋積雲方案的樣子（遍地對流、外流卷雲）、陣風與信風、多風暴清單、3D 工具。
- 已答應但延後：龍捲的地面附近垂直加密（使用者同意過，排在颱風之後）。
- 「放山」需要地形座標（大改動，使用者說不要）。
