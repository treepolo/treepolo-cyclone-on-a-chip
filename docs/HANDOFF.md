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
- CPU：`npm test`（很久）或 `node dist/tests/regional.js`（21 項）。
- GPU（SwiftShader）：`node tools/gpuTest.mjs "gpu-test.html?only=regional"`（32 項，含 adaptive、refine），
  `?only=spinup`、`?only=earth`、`?only=perf`。
- 端對端：scratchpad 裡用 playwright（`require(<npm root -g>/playwright)`，chromium 參數
  `--enable-unsafe-webgpu --use-webgpu-adapter=swiftshader --enable-features=Vulkan`，畫面加 `--use-gl=angle --use-angle=swiftshader`）。
- 我方環境只有軟體 GPU（比 GTX 1650 慢上百倍）：長時間物理實驗用 Node CPU 背景跑或請使用者用無人值守實驗。
- 注意：`pkill -f <pattern>` 會殺掉自己的 shell；不要對有未提交修改的檔案用 `git checkout`。

## 目前進行中：第 2 項「區域模式圖表」（約完成 25%）
已完成（已提交）：
- `regionalGpu.ts`：`readDisplay(levels, volMode)` 現在每欄回傳 `COL=8` 個值
  `[wmax, wmin, cmax, pmax, dBZmax, 雲頂高, 雲頂溫, 2–5 km UH]`，planes 含所有預報場（`DisplayPlanes`），
  volMode 讓 3D 第二通道顯示降水/上升氣流/渦度；新增 `readColumns(points)`（剖面、探空）、`readRZ(xc,yc,dr,nr)`（颱風軸對稱平均）。
- `src/regional/diagnostics.ts`：氣壓、飽和混合比、θe、露點、dBZ、`SLICE_VARS`/`SECTION_VARS`/`MAP_VARS`、`sectionValues()`。
- worker 已改用 `COL` stride（其餘尚未使用新資料）。

待做：
1. 協定：`ToRegionalWorker` 加 `{type:'charts', level, section:{i0,j0,i1,j1}|null, point:{i,j}|null, rz:boolean, volMode}`；
   `RegionalFrame` 加 `maps`（MAP_VARS × nx·ny）、`slice`、`section`、`point`、`rz`。
2. worker `sendFrame`：GPU 用上述讀取；CPU 後端用迴圈算同樣的東西（dBZ、雲頂、UH 等）。
3. `src/app/regional/charts.ts`（canvas 繪圖：色階圖＋色標＋風向箭頭、剖面、r–z、斜溫圖＋風徑圖、時間序列、Hovmöller）；
   `regional.html` 加「主畫面」選單（3D／水平切面／合成圖／垂直剖面／颱風 r–z／探空／時間序列／Hovmöller）、變數與高度選擇、
   在地圖上拖曳畫剖面線、點擊看探空。時間序列與 Hovmöller 在主執行緒累積（颱風：氣壓降、最大風速、RMW、1.5 km 切向風 vs 半徑）。
4. CAPE/CIN、3D 軌跡粒子：之後再做（ROADMAP 有列）。

之後依 ROADMAP 順序：軸對稱快速版颱風模式 → 颱風雨帶（只有眼牆有雲）→ 多重眼牆與置換 → 龍捲 → 互動機制。
