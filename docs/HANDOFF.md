# 交接文件 / Handoff

新對話請先讀這份，再讀 `docs/ROADMAP.md`（待辦與順序）與 `docs/ARCHITECTURE.md`。

## 使用者與溝通
- 用**正體中文**回覆，語氣溫暖、不要冷淡；進度報告簡潔。
- 使用者主要用 PC 玩（NVIDIA GTX 1650，Chrome，WebGPU）；手機 Samsung Note20（Mali GPU，會退回 CPU）。
- 使用者說：**直接一路做下去，不用每階段停下來**，全部做完再一次測試。但大的方向性決定仍要先討論（使用者曾因直接開工而中斷）。
- 使用者**不要**：垂直網格下密上疏、聲波子步 6→4、移動巢狀、GPU 深度優化、在他電腦上裝 Claude Code。
  龍捲項目若要用「地面附近垂直加密」必須先問。
- 原則：**天氣現象必須從方程自己長出來**，不可用程式畫假雲、假渦旋。

## 專案
- Repo `treepolo/treepolo-cyclone-on-a-chip`，分支 `claude/3d-earth-atmosphere-simulator-ncy8n4`（在此開發與推送，不要開 PR）。
- Commit 結尾：
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_018CtDoRsoXh6wMbdQ4AXxb9
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
  GPU 版 `src/gpu/regionalGpu.ts`。實驗定義在 `src/app/regional/worker.ts` 的 `build()`；巢狀 `src/regional/nest.ts`。
- 頁面：`index.html` + `src/app/main.ts`/`worker.ts`/`globe.ts`；`regional.html` + `src/app/regional/main.ts`/`worker.ts`/`volume.ts`。
- 已完成的重要功能：GPU 效能優化（自動時間步長、θρ 預算、亂流只算第一 RK 階段、晴空凝結物跳過、效能分析按鈕）、
  先粗後細細化（`src/regional/refine.ts`，15→5 km 颱風、2→1 km 超大胞、1 km→250 m 龍捲）、
  存檔讀檔（`src/app/saves.ts`，IndexedDB＋ZIP 匯出匯入）、速度控制（`src/app/pacer.ts`）、
  無人值守實驗（`src/app/regional/runner.ts`，報告寫入 Artifact db 集合 `runs`，用 ArtifactData 讀）。

## 測試
- 伺服器：`PORT=5199 node tools/serve.mjs &`（serve 常在指令間被殺，和測試寫在同一條指令）。
- CPU：`npm test`（很久）或 `node dist/tests/regional.js`（26 項）。
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

## 颱風雨帶（ROADMAP 第 4 節，進行中）
- 選項：`RegionalPhysicsConfig.vmin`（地面通量最小風速）、`radConst`（固定對流層冷卻 K/s，平流層仍鬆弛），GPU 常數 `VMIN`、`RADC`；
  `tropicalSounding(sst, Ttrop, rhTop)`。worker 的 `tcEnv`（頁面面板送來、存檔裡保存）套用到 tc／tc_hr／tc_3。
- 工具：`runAxisym.js ... cases='base|vmin:4|radc:1.5+vmin:4'`（CSV 有核心／外圍雨量與雨環數）；
  `runTropicalCyclone.js 8 15000 1200000 out ice vmin=4 radc=1.5 rh12=0.6`（每 3 小時印核心／外圍雨量與外圍降雨面積比）。
- 結果見 ROADMAP 第 4 節；3D 15 km 三組對照（A 基準、B vmin 4、C vmin 4＋冷卻 1.5）在 scratchpad 背景跑（下個 session 需重跑）。
- 3 km 颱風（tc_3）：CPU 端 RegionalModel／RegionalPhysics 的工作陣列改成第一次 CPU 步進時才配置（GPU 跑時只保留狀態陣列）。

## 龍捲與互動（ROADMAP 第 6、7 節）
- `supercell.ts`：`TornadoEnv`（R、depth、U6、qvMax）、`TORNADO_DEFAULT`（新的強低層風切預設）、`TORNADO_WK82`（舊預設，
  沒有環境紀錄的舊存檔用它）；`kessler.ts` 的 `weismanKlempQ(qvMax)`。`runTornado.js ... env=wk82` 可重現舊結果。
- 龍捲偵測：worker `stats.tornado`（ζ、V、EF、位置），頁面 `tornadoWatch` 記事件；runner 的 tor_ef／tor_v／tor_zeta。
- 互動訊息：`perturb`（CPU 改完上傳 GPU）、`paint`（`RegionalPhysics.surface` 與 `GpuRegional.setSurface`，細化後重設）、
  `environment`（改完重建 GPU，因為阻尼層風與邊界目標在 GPU 表內）。圖表的滑鼠工具在 `charts.ts`（`MapTool`）。
- 自由飛行：`VolumeView.setCamera('fly')`。挑戰任務：`missions.ts`（localStorage 記完成，try/catch）。

## 下一步
- 颱風雨帶：等 3D 15 km 三組對照結果（A 基準、B vmin 4、C vmin 4＋冷卻 1.5），決定預設要不要改；用 5 km／3 km GPU 讓使用者確認。
- 多重眼牆：3 km 實驗與軸對稱 1–2 km 長時間積分，用 Hovmöller（眼牆點）判斷。
- 要問使用者：龍捲的地面附近垂直加密；「放山」需要地形座標（大改動）。
