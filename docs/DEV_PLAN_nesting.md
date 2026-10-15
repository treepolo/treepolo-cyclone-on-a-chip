# 開發計畫：外圈看得到、外圈與眼區細化並用、局部加高

> 狀態：**計畫已寫好，尚未開始實作。** 使用者沒說「開始」之前，不改任何程式（寫文件除外）。
> 這份文件是寫給之後的我自己：容器會重開、對話會被摘要，所以所有需要的事實（檔案、函式、約束、驗證方法、發布流程）都寫在這裡。
> 分支：`claude/3d-earth-atmosphere-simulator-ncy8n4`（**不要**推到 `...-m387en`，不開 PR）。行號會漂移，以函式名為準。

## 0. 怎麼用這份計畫

1. 先讀第 1 節（規則）和第 3 節（現況）。
2. 使用者說「開始」→ 從第 6 節的 Step 0 依序做到 Step 3，**一次做完，中途不拆成要使用者逐一確認的版本**（使用者明說過：先討論好再一次做）。Step 4、5 是選配，只有使用者另外說要才做。
3. 每個 Step 結束都要滿足第 1 節的「完成定義」，並更新第 2 節的進度表（把 `[ ]` 改 `[x]` 並提交）。
4. 遇到計畫沒寫到、或與程式現況不符的地方：**停下來、查證、把查到的寫回這份文件**，再繼續。不要憑印象寫程式。
5. 進度另外用 TaskCreate／TaskUpdate 追蹤（每個 Step 一項）。

## 1. 規則（使用者的原話與我自己的紀律）

**R1 沒說開始不改。**「我沒說開始你都不要改」。寫文件、讀程式、跑唯讀檢查可以。

**R2 不跑模擬。**「測試可以跑，只是不要跑模擬」；模擬由使用者自己測。
- 可以跑：`npm run build`（tsc）、`node dist/tests/live.js`、本計畫新增的**無時間步**測試檔、單一 GPU 核心的一次性檢查（只 dispatch 一次，不進時間）、無頭瀏覽器載入頁面截圖（含餵合成體積資料給渲染器）。
- 不要跑：`npm test` 整個（`regional.js`、`dycore.js`、`moist.js`、`vortex.js`、`axisym.js` 會時間積分）、任何 `step()`／`run()` 迴圈、worker 真的跑起來步進。
- 「單一算子求值」（例如算一次 `nestTargets`、一次 GPU 核心）不是模擬，可以。

**R3 眼區細化不碰。**「這些跟眼區細化是不一樣的功能，你改眼區細化幹什麼」。
- 眼區細化的檔案與分支：`src/app/regional/eyeNest.ts`、worker.ts 裡所有 `eye` 分支、`volume.ts` 裡的 `uN*`／`nestW`／`nestQ`、`GpuNest` 與 `twoway.ts` 的圓柱行為。
- 新功能一律**寫在新檔案**。對共用檔（`twoway.ts`、`nestGpu.ts`、`refine.ts`、`light.ts`、`volume.ts`、`worker.ts`、`protocol.ts`、`main.ts`）只允許**加法**，而且沒有新功能時行為要與改前**逐位元相同**，並用 Step 0 / Step 1.1 的回歸檢查證明。
- 新功能必須能和眼區細化同時開著（使用者：「和眼區細化不能同時用是怎樣，完全不合理」）。

**R4 看得到才算完成。**「畫面看不到是三小」。功能做在模型裡但 3D 畫面（或該有的圖表）看不到、操作不到，就是沒做完。

**R5 誠實回報。** 繁體中文、溫暖簡短；分開講「驗證過什麼」「沒驗證什麼（需要使用者跑模擬）」；不把推測說成事實；不丟沒根據的提案（「不要蝦雞巴提案，用點常識」）；遇到做不到的事，在**做之前**就講，不要做完才說。

**R6 一個 Step 的完成定義（DoD）。**
1. `npm run build` 無錯。
2. 該 Step 列出的無時間步測試全過，並貼進 HANDOFF。
3. `docs/HANDOFF.md` 新增項目（從 **34** 起編號）並更新「下一步」。
4. `git commit`（結尾附兩行 trailer：`Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`、`Claude-Session: https://claude.ai/code/session_01WuWtj2MPxtKazZuiVe5hfv`）、`git push -u origin claude/3d-earth-atmosphere-simulator-ncy8n4`。
5. 使用者看得到的改變 → 發布 Artifact（第 9.3 節），版本從 **v56** 起。
6. 回報使用者（第 9.5 節模板）。

**R7 失敗就停。** 測試失敗、型別錯誤修不掉、發現計畫的前提不成立 → 停下來，如實回報，不要硬做完。

**R8 不開 PR。** 只推分支。

## 2. 進度表

- [ ] Step 0 地基：共用檔回歸檢查（不改行為）
- [ ] Step 1 外圈在 3D 畫面看得到（v56）
  - [ ] 1.1 渲染器通用化（射線範圍、佔據格）＋像素回歸
  - [ ] 1.2 協定與 worker：`OuterFrame`
  - [ ] 1.3 頁面：保存、`setOuter`、回放不含外圈
  - [ ] 1.4 著色器：外圈取樣、相接處漸變、地面、光、步長
  - [ ] 1.5 內圈框線、鏡頭範圍、開關
  - [ ] 1.6 無時間步測試＋合成體積截圖
  - [ ] 1.7 HANDOFF、提交、發布、回報
- [ ] Step 2 外圈＋眼區細化並用（v57）
- [ ] Step 3 局部加高（v58）
  - [ ] 3.1 `tall.ts` 幾何／目標／回饋＋測試
  - [ ] 3.2 `tallNest.ts` 建模＋測試
  - [ ] 3.3 `tallGpu.ts`＋GPU 一次性檢查
  - [ ] 3.4 worker 整合（含與眼區、外圈並用的步進順序）
  - [ ] 3.5 協定與頁面面板（足跡預覽）
  - [ ] 3.6 渲染器：局部加高疊層（比主盒高）
  - [ ] 3.7 測試、HANDOFF、發布、回報
- [ ] Step 4（選配、大）垂直拉伸格距——只有使用者說要才做
- [ ] Step 5（選配）外圈向內圈鬆弛（雙向的簡化版）、光的連鎖

## 3. 現況（誠實版）

### 3.1 使用者玩過／我交付過的相關功能
- v54：模擬中改網格（範圍縮小、模式頂、垂直格距）。模式頂加高時**整個範圍**加高、蓋子與海綿層搬到新頂、新層沒有雲雨（`regridTo`、`refineInto`）。
- v55：外圈（範圍加大，增加的部分有自己較粗的格距）。內圈＝現在的模式（原封不動）；外圈是另一個粗網格模型，領先內圈步進，內圈邊界向外圈鬆弛（單向）。

### 3.2 v55 外圈的缺陷（本計畫要解的）
1. **3D 畫面和圖表只看內圈，外圈看不到**——使用者認為這是失敗（「畫面看不到是三小」「難怪我昨天玩的時候按了沒反應」）。
2. **外圈與眼區細化互相拒絕**（`startExt` 的 `if (eye) return …`、`startNest` 的 `if (ext) throw …`）——使用者認為不合理。
3. 單向：外圈框內的風暴只是內圈的粗略版，會自己演變，與內圈風暴可能不一致。（Step 5 選配。）
4. 外圈存在時側邊界不能改；存檔、回放不含外圈。

### 3.3 局部加高（任務 #65）尚未做
使用者要求（原話）：「蓋子跟海綿層放到增加的網格的最上面」「新增高度，原本高度的蓋子要移除」「要能選全部範圍，也要能有自己的解析度（比如設定比較粗）」「我要的就是可以選局部」。

### 3.4 我改過、眼區細化也在用的共用檔（披露；2026-10-15 已用 `git diff` 逐行讀過）
| 檔案 | 我改了什麼 | 對眼區細化的影響（讀 diff 的結論） |
|---|---|---|
| `src/regional/twoway.ts`（421df74） | `NestGeom` 新增選用欄位 `x0,y0,ring`；`nestTargets` 的 `x0/y0` 改成 `g.x0 ?? g.i0*pc.dx`；`used()` 多一行 `if (g.ring !== undefined)` | 眼區的 geom 沒有這三欄 → 與舊版相同 |
| `src/gpu/nestGpu.ts`（421df74） | 常數 `X0/Y0` 改成 uniform `P.y/P.z`（值＝`g.x0 ?? g.i0*pc.dx`）；新常數 `RING`（眼區＝-1）；`used` 的判斷式改寫 | `RING=-1` 時 `used` 與舊式等價；`P.y/P.z` 與舊常數同值（f32） |
| `src/app/regional/eyeNest.ts`（20b5bb4） | `cfg`、`phys` 由 `readonly` 改 `public`；建構子裡的 `preStep` 掛鉤抽成 `hook()`，加 `rehook()` | 建構時行為相同；`rehook()` 只有「模擬中改設定」呼叫 |
| `src/regional/refine.ts`（ec3eaa6） | 新增 `regridPlane`；`refineInto` 的水象在 `z >= 粗模型頂` 時設 0 | 眼區的頂與主網格相同，格心 z 永遠 < 頂 → 不觸發 |
| `src/regional/core.ts`（20b5bb4） | 新增 `changeSounding` | 純加法 |

**這些是「讀 diff 判斷等價」，還沒有數值回歸測試**（Step 0 補上）。

### 3.5 還沒驗證的（需要使用者跑模擬）
v52–v55 的整套流程在真實頁面上都沒跑過（套用設定、改網格、外圈步進／時間內插／跟風暴／移除／設定同步、外圈風暴與內圈風暴不一致造成的邊界影響大小）。這份計畫不改變這個事實；每次回報都要重申。

## 4. 這份計畫做得到／做不到（先講清楚）

| 項目 | 結論 |
|---|---|
| 外圈在 3D 畫面看得到（雲、降水、地面雲影、與內圈相接處不斷層） | 做得到（Step 1） |
| 外圈的圖表 | **不做**：圖表仍只看內圈（外圈的 3D 體積才有顯示）。若使用者要外圈圖表另議 |
| 外圈與眼區細化同時用 | 做得到（Step 2），不改眼區細化 |
| 局部加高：任選方形範圍（含接近全範圍）、自己的新頂、蓋子與海綿層在新頂、解析度**與主網格相同或更細**（整數倍） | 做得到（Step 3） |
| 局部加高與眼區細化、外圈同時用 | 做得到（Step 3.4 定義步進順序） |
| **加高的部分比主網格粗**（使用者舉例「設定比較粗」） | **Step 0–3 做不到**。核心的垂直格距是均勻的（上百處假設），粗的上層需要拉伸格距＝重寫垂直算子（Step 4），風險最高、又不能用模擬驗證。第 8 節請使用者決定 |
| 「完全等於整個範圍」的加高 | 已有：v54 的改模式頂（整個範圍、同格距）。局部加高的範圍上限是主網格扣掉邊緣緩衝 |
| 存檔、回放含外圈／局部加高 | v1 不含（存檔只存主網格；回放時外圈、局部加高隱藏，紀錄說明） |
| 外圈→內圈以外的方向（內圈回饋外圈） | v1 單向；Step 5 選配 |

## 5. 設計決定

### D1 外圈怎麼顯示（Step 1）
- 內圈所有座標（工具點選、鏡頭、軌跡粒子、切面、疊層）維持**內圈局部座標**不動；外圈用**延伸的射線範圍**加一張**外圈體積貼圖**，只在內圈盒子外面（及相接漸變帶）取樣。這樣眼區疊層（`nestW`、`nestQ`）與所有既有程式完全不用改。
- 射線範圍由 `hitBox` 的 `[0, uBox]` 一般化為 `[uLo, uHi]`（uniform；沒有額外圖層時 `uLo=0, uHi=uBox` → 畫面與改前逐像素相同）。Step 3 的局部加高只需把 `uHi.z` 加高。
- 外圈與內圈相接：內圈邊緣有鬆弛帶（`relaxCells + 2` 格）。權重 `wo = smoothstep(-taper, 0, d)`，`d` ＝到內圈盒邊的有號距離（盒外為正）：盒外 `wo=1`（只看外圈）、盒內鬆弛帶外 `wo=0`（只看內圈）、帶內線性混合。混合順序：`s = mix(mix(內圈, 眼區, w), 外圈, wo)`。
- 外圈格距較粗 → 外圈區域的射線步長用外圈格距（`dtO = 0.5/max(ard.xy/uOCell, ard.z/uOCellZ)`），帶內取內圈步長。
- 佔據格（跳過空氣）：外圈另建一張佔據貼圖（用現有 `occupancyGrid` 對外圈體積算，不改函式）；著色器裡「內圈盒內且 `wo==0`」查內圈佔據，`wo>0` 或盒外查外圈佔據；內圈盒邊附近（距離 < 一個佔據塊）也要查外圈佔據，避免一步跨過漸變帶漏掉外圈的雲。
- 地面：內圈盒外、外圈盒內畫純海／陸色（`uOut`）並套外圈光（雲影）；外圈盒外維持現狀。內圈盒邊畫一條細線，使用者才知道內圈在哪。
- 光：外圈體積自己算 `sunSkyLight`（現有函式）。外圈雲對內圈的遮蔭（光的連鎖）列為 Step 1.8/5 選配。
- 外圈體積資料量：外圈格數可到 30 M，不能每幀整個傳。上限 `OUTER_DISPLAY_MAX = 4e6` 個體素，超過就水平降取樣（區塊內取消光量（位元組³）平均再開立方）；傳送頻率限制（至少隔 ≈1 s 牆鐘且外圈模型時間有前進才傳），沒傳的幀不帶 `outer` 欄位（頁面沿用上一份），外圈移除時傳 `null`。
- 工具點選落在內圈盒外：現有 `tools3d.ground()` 已經把盒外視為無效（回傳 null），不用改。

### D2 外圈＋眼區細化並用（Step 2）
- 眼區是**內圈的子網格**；外圈是**內圈的父**。兩者互不相識。只需拿掉兩個拒絕，並在眼區的步進分支前先做外圈的準備（`extPrepare()`／`extPrepareCpu()`，它們只寫內圈的邊界目標緩衝，眼區的 `link.step` 寫的是眼區自己的緩衝）。眼區模組不動。
- 要逐一檢查 worker 裡每個同時出現 `eye` 與 `ext` 的分支（第 6 節 Step 2 的表）。

### D3 局部加高（Step 3）
- **做成一個獨立的子網格 T**（新模組，不是眼區的變體）：
  - 水平：方形足跡，`np × np` 個主網格格子，整格對齊，位置任選（預設風暴／範圍中央），寬度任選（最小 8 個主格；最大＝主網格扣邊緣緩衝）。解析度 `r`（水平，≥1）、`rz`（垂直，≥1）為整數倍，T 的格距＝主網格／倍數。
  - 垂直：T 從地面到新頂 `Z2 > Z1`（`Z1`＝主網格頂），`dzT = dz/rz`。T 自己的蓋子（w=0）與瑞利海綿層在 `Z2`（`dampDepth = min(6000, 0.3·Z2)`），主網格在足跡內的原蓋子由 T 取代（T 的回饋把主網格足跡內的狀態蓋掉）。
  - 側邊界：T 是開放邊界（`lateral:'open'`，邊緣鬆弛帶），鬆弛目標：`z < Z1` 取主網格（空間＋時間內插，偏差法 `makeSampler`），`z ≥ Z1` 取 T 自己的無擾動環境（基準態）。海綿層同樣取這些目標。
  - 回饋：只回饋 `k < 主網格 nz` 的層，足跡邊緣用方形的 cos² 漸變（寬 `Wf`），w 只回饋 k>0（頂面 w 不動）。
- 為什麼不用現有 `GpuNest`／`nestTargets`/`nestFeedback`：它們的回饋權重是圓柱的、目標在 `z≥主網格頂`時延伸最後一層偏差（對加高部分是錯的，應為環境）。為了不動眼區的共用碼，**新寫** `tall.ts`（CPU）與 `tallGpu.ts`（GPU，核心是 `nestGpu.ts` 的改寫版，會有少量重複，這是隔離的代價）。
- 已知限制（要寫進面板說明）：
  1. T 的側邊在 `Z1 − 阻尼深度 … Z1` 這一段吃到的是被**主網格海綿層壓過**的值；風暴的外流碰到 T 的側邊會被壓。足跡要夠大、風暴在中央。
  2. T 的海綿層若與原頂以下重疊（`Z2 < Z1 + 6 km` 左右），那段被主網格拉著走（面板顯示建議的最小 `Z2`）。
  3. 不能比主網格粗（見第 4 節）。
- **與眼區並用**（眼區仍是主網格的子網格，圓柱可以在 T 的足跡內）：每個主網格步的順序
  1. （有外圈）`extPrepare()`
  2. 主網格步＋眼區子步＋眼區回饋主網格（＝現有 `link.step(eye.nsub)`，**原樣呼叫**）
  3. T：用主網格（此時已含眼區的回饋）算目標 → T 子步 →（若眼區圓柱整個在 T 足跡內且眼區倍數是 T 倍數的整數倍）**眼區→T 回饋**（用 `new GpuNest(T.gpu, eye.gpu, 眼區在 T 座標下的 geom)` 的 `feedback()`，公開 API，不改類別）→ T 回饋主網格。
  - 不做第 3 步中的眼區→T 回饋的話，眼區的回饋會在下一步被 T 的回饋蓋掉，眼區等於只是畫面細節。倍數不整除時：不做這個回饋、紀錄說明。
  - CPU 路徑：`const old = parentState(m)`（在 `eye.cpuStep` 之前取）→ `eye.cpuStep(m, mp)` → `tall.cpuStep(m, old)`。
- **跟風暴走**（`moveDomain` 整格滾動）：T 像眼區一樣隨主網格的場一起滾動（`T.m.roll(di·r, dj·r)`），進來的欄位取主網格內插（`refineInto` 的 `cols` 參數）。
- 存檔、回放、圖表：只含主網格（T 在存檔時被平均進主網格，同眼區的作法與紀錄文字）。

### D4 垂直拉伸格距（Step 4，選配）
見 Step 4。結論先寫：這是讓「加高部分比主網格粗」的唯一做法；風險高，使用者說要才做。

## 6. 步驟

每個步驟的欄位：**目標／檔案／做法／無時間步驗證／使用者要測什麼**。

---

### Step 0 地基：共用檔回歸檢查（不改行為）

**目標**：證明第 3.4 節的共用檔修改對眼區細化沒有影響；並建立後續要用的對照基線。

**做法**
1. `git status` 乾淨、`git log -1` 是 421df74 之後；`npm run build`；`node dist/tests/live.js` → 32/32（起點）。
2. 取舊版：`git worktree add $SP/old ec3eaa6`（v54，外圈之前），`ln -s /home/user/treepolo-cyclone-on-a-chip/node_modules $SP/old/node_modules`，在 `$SP/old` 跑 `npx tsc -p tsconfig.json` → `$SP/old/dist`。
3. CPU 比對腳本（放 `$SP/check/nestRegress.mjs`，不進 repo）：載入新舊兩份 `dist/regional/twoway.js`、`core.js`、`refine.js`，用同一個亂數種子造 24×24×12 的父模型（`weismanKlemp` 探空）加隨機偏差場，`nestGeometry` 造眼區 geom，比較：`nestGeometry` 輸出、`nestTargets`（`alpha` 取 0、0.5、1，含 `old` 狀態）輸出陣列、`nestFeedback` 後父模型陣列。期望**逐位元相同**（`Object.is`）。
4. GPU 比對頁（`$SP/check/gpuNestRegress.html`＋`node tools/gpuTest.mjs` 的作法：頁面放 repo 根目錄暫用、跑完刪掉）：用新舊兩份 `GpuNest`（舊版從 `$SP/old/dist` 載入，路徑用 `/../` 不行時把舊 dist 複製到 `dist_old/` 暫用）在同一組隨機狀態上各跑一次 `targets(0.5)` 與 `feedback()`，讀回比較。期望相同或差 ≤ 1e-7。舊的 `nestGpu.js` 搭配**新的** `GpuRegional` 物件即可（它只用 `device`、`cpu`、`S`、`S0`、`B`、`nf`、`nq` 這些屬性；它自己 import 的 `WG/GX/linearGid` 來自舊 dist，沒關係）。**只 dispatch 兩個核心，不步進。**
5. 做完 `git worktree remove --force $SP/old && git worktree prune`（worktree 不進提交）。
6. 結果寫進 HANDOFF 項目 34（含第 3.4 節的披露表）。若有不同 → 停下來，找出差異並還原成舊行為，如實回報使用者。

**完成**：回歸全過（或已修復並說明）。不發布。

---

### Step 1 外圈在 3D 畫面看得到（v56）

#### 1.1 渲染器通用化（不改畫面）
- 檔案：`src/app/regional/volume.ts`（著色器 `FS`、`VolumeView`）、`light.ts`。
- 先讀 `VolumeView` 的建構子，確認能在測試頁用一個 canvas 直接建立。先做**金標準**：在改任何東西之前，用無頭瀏覽器（參數 `--use-gl=angle --use-angle=swiftshader`，見 `$SP/e2e/*.cjs` 的寫法）載入 `dist/app/regional/volume.js` 的 `VolumeView`，餵一份**固定的合成體積**（例如 48×48×24，幾個球形雲加雨，固定亂數種子），固定相機，截圖存 `$SP/golden/base_*.png`（至少三種：無眼區、有合成眼區疊層、切面開）。
- 改動：
  1. `hitBox` 改用 `uLo`、`uHi`（uniform，預設 `0` 與 `uBox`）。
  2. 佔據貼圖座標由 `p/uBox*uOccScale` 改成 `(p-uLo)/uOccExtent`，`uOccExtent` 預設等於原本涵蓋的範圍（`uBox/uOccScale`）。
  3. `VolumeView` 增加 `lo/hi` 欄位與 `clampTarget` 用 `lo/hi`（預設同原式）。
- 驗證：同樣的合成輸入再截圖，與金標準**逐像素相同**（容許 0；如有差異先查原因，不放寬）。若環境無法取得 WebGL2 → 此步改成純讀碼檢查，並在回報中說明「渲染器回歸只靠讀碼」。

#### 1.2 協定與 worker
- 檔案：`protocol.ts`、`worker.ts`、新檔 `src/app/regional/outerView.ts`（純函式）。
- `protocol.ts`：
  ```ts
  export interface OuterFrame {
    nx: number; nz: number;           // 降取樣後的格數（nx × nx × nz）
    f: number;                        // 水平降取樣倍數（≥1）
    dx: number; dz: number;           // 降取樣後的格距 (m)
    x0: number; y0: number;           // 外圈左下角在內圈座標系（m，可為負）
    L: number; top: number;           // 外圈寬、頂 (m)
    taper: number;                    // 內圈邊緣鬆弛帶寬 (m)
    cloud: Uint8Array; rain: Uint8Array;
  }
  ```
  `RegionalFrame` 加 `outer?: OuterFrame | null`（`undefined`＝沿用上一份，`null`＝沒有外圈）。
- `outerView.ts`：`decimateVolume(cloud, rain, nx, nz, f)`（水平 f×f 區塊：消光量＝位元組³的平均再開立方，常數場不變）、`outerFrameOf(ext, bytes…)`（`x0 = -ext.o.geom.x0`，`y0 = -ext.o.geom.y0`，`L = ext.o.model.c.nx * ext.o.model.c.dx`，`taper = ext.o.geom.ring * innerDx`）、`decimationFor(nT, nz, cap)`。
- `worker.ts` 的 `sendFrameRaw`：在 `nestVolume()` 旁新增 `outerVolume()`：有 `ext` 且到了傳送時機才做——GPU 用 `ext.gpu.readDisplay([], volMode, subgrid)` 解 `packed`；CPU 用 `volumeBytes(ext.o.model, …)`；超過 `OUTER_DISPLAY_MAX` 就 `decimateVolume`；buffer 加進 `transfer`；`post({… outer })`。沒有 `ext` 時：若上一幀有送過，這幀送 `outer: null` 一次。外圈 `extDrop`／`extStop` 時重設「已送過」旗標。
- 數值發散檢查：外圈的 `wmax/wmin` 也納入 `W_BLOWUP` 檢查（`volumeBytes` 回傳值已有）。

#### 1.3 頁面
- 檔案：`main.ts`、`volume.ts`、`replayPanel.ts`／`ReplayStore` 建立 `ReplayFrame` 的地方。
- `main.ts` 幀處理：維護 `lastOuter`（收到 `outer` 欄位才更新，`null` 清掉）；`showVolume(f)` 之後呼叫 `view.setOuter(replayIdx !== null ? null : lastOuter)`。回放時外圈隱藏，並在第一次隱藏時寫一行紀錄。`ReplayFrame` **不要存 `outer`**（記憶體）。
- `VolumeView.setOuter(o: OuterFrame | null)`：上傳 `oVol`（RG8 3D 貼圖，用現有 `upload`）、算 `oLight`（`sunSkyLight(o.nx, o.nx, o.nz, o.cloud, [oL, oL, bz], o.top, SUN, kExt, cutPlane, false)`，`oL = o.L / Lx`）、建外圈佔據貼圖、設 `lo/hi`＝內圈盒與外圈盒的聯集（內圈局部座標，單位＝內圈寬為 1）、`dirty=true`。

#### 1.4 著色器
- 新 uniform：`uOOn`、`uOVol`、`uOLight`、`uOOcc`、`uOMin`（外圈盒左下角，箱單位）、`uOSize`、`uOCell`、`uOCellZ`、`uOTaper`、`uOOccExtent`。
- 新函式 `outerW(p)`（見 D1）；`volAt(p, w)`、`lightAt(p, w)` 加入外圈混合；迴圈裡 `dtB`：`wo>=1.0` 用外圈步長、否則內圈；`bThr = 0.3/dtB`；佔據判斷依 D1。
- 地面：內圈盒外、外圈盒內的點用 `uOut` 套外圈光；其餘不變。
- 切面（`uCutOn`）對外圈同樣有效（已經在 `bytesAt` 裡）。

#### 1.5 內圈框線、鏡頭、開關
- 地面著色加一條內圈盒邊細線（寬約 1.5 像素，用 `fwidth`），只在有外圈時顯示。
- `clampTarget`、`dist` 上限、預設鏡頭：有外圈時能看到整個外圈（`dist` 上限隨 `hi-lo` 放大）。
- 面板：「顯示外圈」勾選（預設開）；關掉就 `setOuter(null)` 效果（但仍保留資料）。

#### 1.6 無時間步測試（`src/tests/outerView.ts`，加進 `npm test` 的 script）
- `decimateVolume`：常數場不變；單一亮格 → 區塊平均值正確；`f=1` 回傳原值。
- `outerFrameOf`：用 `buildOuter` 造一個小外圈（不步進），檢查 `x0 = -geom.x0`、`L`、`taper`；框在中央時 `x0 = -(nT-np)/2·sp`。
- 聯集範圍函式 `extentBounds`：內圈在外圈中央、偏一側、有負座標。
- 佔據：外圈雲只在盒外 → 外圈佔據貼圖該處非 0、內圈佔據不受影響。
- 渲染（無頭瀏覽器、合成體積）：(a) 無外圈 → 與金標準逐像素相同；(b) 合成外圈雲條帶在內圈盒外 → 該處像素與背景不同、內圈盒內無外圈雲；(c) 外圈＋合成眼區疊層同時存在 → 兩者都出現；(d) 相接漸變帶中央的像素與兩側無斷層（亮度差 < 閾值）。
- 一次性：`outerFrameOf` 在 GPU 外圈模型上讀回的路徑**需要真的 GPU 外圈模型**（建模型但不步進，`uploadFrom` 後直接 `readDisplay`）——可以做，是單一核心。

#### 1.7 收尾
HANDOFF 項目 35；提交；發布 v56；回報。**使用者要測**：加外圈後 3D 畫面外圈的雲是否出現在內圈外面、相接處是否有明顯斷層、內圈框線、跑起來的速度、移除外圈後畫面是否恢復、回放時外圈是否隱藏。

---

### Step 2 外圈＋眼區細化並用（v57）

**目標**：外圈存在時可以開眼區細化，反之亦然；3D 畫面同時顯示外圈、內圈、眼區。

**要改的地方（逐一檢查 `eye` 與 `ext` 同時存在的分支；以函式名搜尋）**
| 位置 | 現況 | 要做的事 |
|---|---|---|
| `startExt` | `if (eye) return '請先停止眼區細化…'` | 刪掉這行 |
| `startNest` | `if (ext) throw new Error('請先移除外圈…')` | 刪掉這行（外圈的 `extRelink` 已在 `nestGpuUp` 裡） |
| `stepGpu` | `if (e && link) … link.step(e.nsub)`／`else if (ext?.link) {extPrepare(); gpu.step(1)}` | 眼區分支每步前先 `if (ext?.link) extPrepare()` |
| `stepCpu` | `if (eye) eye.cpuStep(…) else { if (ext) extPrepareCpu(); … }` | 眼區分支前先 `if (ext) extPrepareCpu()` |
| `adaptDt`／`extAdapt`／`nestFit` | 兩者各自呼叫 | 確認順序：外圈 `extAdapt` → 內圈 → `nestFit`（眼區的子步數依內圈 dt） |
| `moveDomain` | 先位移 `ext.o.geom.x0/y0`、`extToCpu`、`outerTargets(…)`、`enterEnvironment`，再 `eye?.roll` | 確認 `eye.roll` 用的內圈邊界目標已刷新；確認 GPU 重建後 `nestGpuUp` 先 `extRelink` 再 `eye.gpuUp` |
| `nestGpuUp` | `extRelink(); if (!eye…) eye.gpuUp(gpu)` | 已同時處理兩者；確認 `eye.gpuUp` 失敗停掉眼區時外圈連結不受影響 |
| `restoreUndo` | GPU 快照分支與 CPU 分支都會重啟眼區（`startNest`） | 確認重啟眼區時外圈連結仍在；外圈在還原後比內圈領先一步（`extPrepare` 的 alpha 會被夾在 0 → 以外圈上一步開頭的狀態為目標，近似，紀錄不用說） |
| `applySettings` | 眼區與外圈各自套用 | 已有；確認 `eye.surfaceFrom` 與外圈的順序 |
| `interact`（擾動、增濕、風、渦旋、水象、環境） | 眼區與外圈各自處理（`if (eye)`、`if (ext)`） | 讀一遍確認不互相覆蓋；環境互動同時套到三者 |
| `regridTo` | 停眼區、移除外圈 | 同現狀 |
| `sendFrameRaw` | `nestVolume()`、（Step 1）`outerVolume()` | 兩者都送 |
| 存檔 | 眼區平均進外圍網格，外圈不含 | 紀錄文字合併說明 |

**無時間步驗證**：`npm run build`；`live.ts` 加一項：`nestGeometry` 在「內圈為開放邊界」（`relaxCells+2` 緩衝）下對一個典型內圈能造出眼區 geom，且眼區框與外圈的環帶（`geom.ring`）不重疊；逐項讀碼檢查上表並把結論記在 HANDOFF。渲染器：合成「外圈＋眼區」截圖（已在 1.6 (c)）。

**使用者要測**：先加外圈再開眼區、先開眼區再加外圈、各自移除、跟風暴走時、改設定時，跑一小段看有沒有 NaN／暫停、眼區回饋是否還在、外圈與眼區在畫面上同時出現。

**完成**：HANDOFF 項目 36；提交；發布 v57；回報。

---

### Step 3 局部加高（v58）

**前置確認（讀碼，結果寫回本文件）**
- `GpuRegional` 的 `S0`（步開頭狀態）是否每步都更新（`GpuNest.targets` 用 `parent.S0`/`S` 做時間內插）。
- `buildModel` 以 `top = Z2` 造模型時，探空是否涵蓋到 `Z2`（v54 的改頂已用過，應可）。
- `RegionalModel.roll`、`shiftFrame`、`changeSounding` 對 T 的適用性。
- `makeSampler` 在頂以上的行為（延伸半層後取常數）——T 的目標在 `z≥Z1` 不用它，改取環境。

#### 3.1 `src/regional/tall.ts`（CPU，新檔）
- `export interface TallGeom { r; rz; i0; j0; np; nx; nz; dx; dz; x0; y0; Z1; Z2; Wf; ring }`（`x0,y0`＝足跡左下角在主網格座標系，m；`ring`＝邊緣鬆弛帶寬，T 格；`Wf`＝回饋漸變寬，m）。
- `tallGeometry(host: RegionalConfig, spec: { cx; cy; width; top; r; rz }): TallGeom | string`：整格對齊、邊緣緩衝（主網格開放：`relaxCells+2`；週期：2）、最小寬 8 個主格、`Z2 ≥ Z1 + dz`（建議 `Z1 + 6 km`，不足給警告字串但不拒絕）、`nz·nx²` ≤ `TALL_MAX_CELLS (30e6)`、`r,rz` 整數 ≥1 且 `np·r ≤ 1024`、垂直層數 ≤ 400；失敗回傳繁中＋英文訊息。
- `tallTargets(host, old, T, g, alpha, out)`：同 `nestTargets` 的結構，但 (1) 範圍＝邊緣環（`g.ring`）加海綿層；(2) 每個 T 格若 `z_T < Z1`：取主網格偏差內插（`makeSampler`），若 `z_T ≥ Z1`：目標＝T 的基準態（`T.ub/vb/th0/qv0`，`pp=0`）；(3) `Z1` 上下各半個主格內線性混合，避免跳變。
- `tallFeedback(T, host, g)`：T 的 `r×r×rz` 區塊平均（偏差法，與 `nestFeedback` 相同的公式）回主網格 `k < host.nz`；權重 `wt = taper(到足跡邊的距離)`：距離 ≥ `Wf` 為 1，邊上為 0，cos² 過渡。
- `tallRoll`／`tallSurface` 用現成的 `refineInto(host, T, x0, y0, …, cols)` 與 `nestSurface` 的概念（足跡版）。
- **測試 `src/tests/tall.ts`（無時間步，加進 `npm test` script 於 `live.js` 之後）**：
  1. 幾何：合法／過小／太靠邊／太多格／`Z2` 太低／倍數非整數 各一例；「接近全範圍」的最大寬度。
  2. 目標：主網格為均勻環境且 T 也是環境 → 目標＝基準態；主網格為線性偏差場 → `z<Z1` 的目標等於三線性內插值（1e-12）；`z≥Z1` 的目標偏差為 0；`Z1` 附近連續；`alpha` 內插；環以外的格子不寫。
  3. 回饋：T＝主網格內插（`refineInto`）→ `tallFeedback` 後主網格不變（1e-12）；足跡外不變；`k ≥ host.nz` 的 T 層不影響主網格；頂面 w 不動；邊緣權重為 0、內部為 1。
  4. 建模（3.2 的 `build`）：初始場在 `z<Z1` 等於主網格內插；`z≥Z1` 水象為 0、偏差等於 v54 的延伸規則；頂 w=0；`dampDepth = min(6000, 0.3·Z2)`；基準態在共用層與主網格一致（1e-9）。
  5. 滾動：`roll` 之後進來的欄位等於主網格內插。

#### 3.2 `src/app/regional/tallNest.ts`（新檔，仿 `EyeNest` 的結構但是獨立的類別）
- `TallNest.build(host, hostAcc, setup, frameVel, surface, spec, gpuOk): TallNest | string`：`tallGeometry` → `buildModel({...setup, preset:'custom', L: g.nx·g.dx, dx: g.dx, dz: g.dz, top: g.Z2, boundary:'open', follow:false, initAmp:0, dt:0})`（記得 `m.c.lateral='open'`、`relaxCells`、`relaxTau`＝同 `openInner`；`boundary` 目標用 `emptyTargets`）→ 基準風：共用層取主網格的 `ub/vb`（內插），其上取 `buildModel` 的環境風並做座標系修正（照 `buildOuter` 的 `shiftFrame` 作法）→ 地表取主網格（足跡版 `nestSurface`）→ `refineInto` 填 `z<Z1`（`z≥Z1` 的水象自動為 0）→ `tallTargets(…, alpha=1)` 填邊界目標 → `RegionalPhysics`。
- 方法：`fit(dt, rate, cfl)`（子步數，邏輯同 `EyeNest.fit`，**抽成共用純函式放在 `tallNest.ts` 自己裡**，不改 `EyeNest`）、`courantRate()`、`cpuStep(host, old)`、`gpuUp(hostGpu)`、`gpuDown()`、`sync()`、`upload()`、`shiftFrame`、`roll`、`surfaceFrom`、`setForcings`、`volume(mode, subgrid)`、`hook()/rehook()`（持續風；與 `EyeNest` 相同的寫法）、`destroy()`。
- **測試**：用小模型（例如主網格 24×24×10，T 12×12 主格、`r=2, rz=1`，`Z2=1.6·Z1`）建 `TallNest`（只建、不步進），檢查 3.1 的第 4 項，並檢查 `fit` 在給定 Courant 率下的子步數。

#### 3.3 `src/gpu/tallGpu.ts`（新檔）
- `GpuTall(host: GpuRegional, child: GpuRegional, g: TallGeom)`：複製 `GpuNest` 的 WGSL 並改：(1) `used`＝邊緣環或海綿層；(2) 目標在 `z ≥ Z1` 取子網格基準態、`Z1` 附近線性混合；(3) 回饋權重為方形漸變；(4) 垂直對應用 `rz`；(5) `x0/y0` 為 uniform（足跡跟著風暴時可更新）。公開 `targets(alpha)`、`feedback()`、`step(n)`（`host.step(1)`＋n 個子步＋回饋；與眼區並用時 worker 不呼叫它，自己依 D3 的順序呼叫 `targets`/`feedback`）、`destroy()`。
- 用 `dispatchLinear`／`linearGid`（大網格 workgroup 數會超過 65535）。
- **一次性 GPU 檢查**（頁面暫放 repo 根目錄、跑完刪）：用 3.2 的小模型，把相同的隨機狀態上傳到 GPU，各 dispatch 一次 `targets(0.5)` 與 `feedback()`，讀回與 CPU `tallTargets`／`tallFeedback` 比較（u、v、θ、qv、π′ 差 ≤ 3e-5，w ≤ 1e-6）。**不步進。**

#### 3.4 worker 整合（`worker.ts`，新增 `tall` 分支，平行於 `eye`、`ext`）
- 狀態 `let tall: TallNest | null`。訊息：`{type:'tall', spec}`、`{type:'untall'}`；回 `{type:'tall', info: TallInfo | null}`。
- 分支清單（仿 `eye` 的每一處，以 `grep -n "eye" worker.ts` 列出對照，**每個 `eye` 出現處都要決定 `tall` 要不要同樣處理**，把決定寫進本文件）：`build()` 開頭 `tallDrop()`；`startTall/stopTall`；`stepGpu`／`stepCpu`（D3 的順序）；`adaptDt`（T 的 `fit`）；`nestGpuUp`（`tall.gpuUp`）；`moveDomain`（`shiftFrame`、`roll`、`setForcings`）；`restoreUndo`；`applySettings`（`applyLive(h, next, old, frame, true)`）；`interact`（擾動、增濕、風、渦旋、水象、環境：足跡內同時套到 T；高度 `z ≥ Z1` 只有 T 吃到）；`syncFromGpu`／存檔；`regridTo`（停 T，紀錄說明）；`stateGuard`（`guardModel(tall.m)`）；`sendFrameRaw`（`tallOut`）。
- 與眼區並用：D3 的順序；眼區→T 回饋的條件檢查與 `GpuNest(T.gpu, eye.gpu, geomInT)` 的建立放在 `tallLinkEye()`，眼區開／關、T 開／關、GPU 重建時都要重建或拆除。
- 與外圈並用：T 的父是內圈，外圈是內圈的父；互不干涉。`extPrepare` 在內圈步之前，T 在內圈步之後。

#### 3.5 協定與頁面
- `protocol.ts`：`TallInfo { Z1; Z2; width; r; rz; dx; dz; cells; x0; y0; nsub; Wf }`、`TallFrame extends TallInfo { cloud; rain }`；`RegionalFrame.tall?: TallFrame | null`（同外圈的 `undefined/null` 規則可省略：T 與內圈同頻率傳，體積較小，每幀傳）。
- `regional.html`＋新檔 `src/app/regional/tallForm.ts`（仿 `liveForm.ts` 的 `GridForm`）：「局部加高」區塊——足跡中心（選單：範圍中央／目前風暴位置；或輸入座標）、寬度（km；有「最大」按鈕）、新頂 `Z2`（km，顯示建議最小值）、水平倍數 `r`、垂直倍數 `rz`、顯示算出的 `dx/dz/格數/成本`、「開始」「停止」。**足跡預覽**：輸入時 3D 地面即時畫出方框（用 1.5 的框線機制，第二個矩形）。
- `main.ts`：訊息收發、`view.setTall(frame)`、停止時 `clearTall()`。回放時 T 隱藏。

#### 3.6 渲染器：局部加高疊層
- `volume.ts`：新增 `tall` 狀態與 `uTOn/uTVol/uTLight/uTBox/uTTop/uTCell/uTCellZ/uTWf`；`tallW(p)`（方形足跡的 cos² 漸變：`d` ＝到足跡邊的距離）。`uHi.z` 擴到 T 的頂（箱單位 `Z2/Lx·exag`）；主體積在 `p.z > uBox.z` 時取 0（避免夾邊延伸）；`volAt`：`s = mix(mix(主, T, tw), 眼區, w)`；T 取樣座標 `((p.xy - uTBox.xy)/uTBox.z, p.z/uTTop)`。
- 光：T 的光＝新函式 `tallSunSkyLight(…)`（放 `light.ts`，**新增**函式，不改既有；邏輯同 `nestSunSkyLight`，但 T 比主盒高：太陽掃描通過 T 自己的雲，離開足跡時取主網格的光學厚度；高於主頂的部分外面是空的）。
- 佔據格：聯集範圍含 T 的頂；T 範圍用 `occupancyGrid`（現成）對 T 體積算。
- 垂直放大：主盒 z 對應 `Z1`，T 對應 `Z2`，同一個放大係數（`exag`），箱高＝`max`。
- 切面、軌跡粒子、工具高度刻度（若有）在高於主頂處要確認不出錯。

#### 3.7 測試與收尾
- 無時間步：3.1／3.2／3.3 的所有測試；合成渲染：T 在主頂上方的雲出現、足跡外的上空是空的、T 邊緣漸變無斷層、T＋眼區＋外圈同時存在。
- HANDOFF 項目 37（含限制：不能比主網格粗、側邊海綿層效應、存檔、回放）；提交；發布 v58；回報。
- **使用者要測**：局部加高開始後跑一小段：高層是否出現雲、側邊有沒有假的亂流、蓋子（頂）附近有沒有反射波、與眼區並用時眼區是否還有回饋、成本（跑得多慢）。

---

### Step 4（選配、大）垂直拉伸格距——**只有使用者說要才做**

**目標**：讓加高部分（整個範圍或 T 內）的垂直格距比下層粗。
**為什麼大**：核心假設均勻 `dz` 的地方：`core.ts` 約 33 處、`regionalGpu.ts` 約 48、`physics.ts` 約 13、`cumulus.ts` 約 12、`diagnostics.ts` 約 12，另有 nest／refine／worker／display。用 `grep -n "\bdz\b"` 列清單。
**做法大綱**
1. `RegionalConfig` 加 `dzLevels?: number[]`（各層厚度）；`RegionalModel` 算 `zf/zc/dzc[k]/dzf[k]`，並提供 `uniform` 旗標（`dzLevels` 缺省＝現狀，**位元相同**）。
2. 逐算子改成逐層厚度：平流通量散度 `/(ρ0 dzc[k])`、垂直梯度用 `dzf[k]`、聲波隱式三對角的係數、擴散、海綿層（用高度）、水象沉降（`dz[k]`）、積雲方案、地表層（最低層厚度）、診斷。高階垂直平流（3 階／5 階迎風）在拉伸處降階，並規定相鄰層厚度比 ≤ 1.1。
3. GPU：厚度陣列放 storage buffer，核心逐層讀取。
4. 所有跨格距的程式（`refineInto`／`makeSampler`、`nestTargets`、`regridTo`、`display`、渲染器）都要以高度（`zc`）而不是 `(k+0.5)·dz` 計算；渲染器的 z 貼圖座標在顯示前重取樣成均勻層。
5. **無時間步驗證**：(a) `dzLevels` 缺省時，各算子單次求值與舊版**逐位元相同**（舊版用 worktree 取，沿用 Step 0 的比對方式）；(b) 解析場上的算子收斂率（細化 2 倍誤差降為 1/4）、散度守恆（垂直通量望遠和）、靜力基準態與拉伸格距的一致性。**動力行為（穩定性、聲波、反射）只能由使用者跑模擬驗證**，回報時要明說。
**風險**：最高。不要在使用者沒同意下做。

### Step 5（選配）
- 5.1 外圈向內圈鬆弛：外圈在內圈盒內的格點向 `coarsenInto(內圈)` 的平均鬆弛（時間尺度 τ 可設，預設幾個外圈步）：`x ← x + (mean − x)(1 − exp(−dt/τ))`。無時間步測試：單次套用後的值等於公式。
- 5.2 光的連鎖：外圈遮蔭內圈。用現有 `nestSunSkyLight`（把內圈當 nest、外圈當 outer），需要 `sunSkyLight` 的 `keepOd` 輸出再傳給眼區的 `nestSunSkyLight`（兩層連鎖）。

## 7. 風險表

| 風險 | 影響 | 對策 |
|---|---|---|
| 著色器改動讓既有畫面變了 | 破壞眼區疊層、效能 | 1.1 金標準逐像素回歸；新功能走 uniform 開關，關閉時等於舊路徑 |
| 外圈體積太大拖垮傳輸 | 畫面卡頓 | 4 M 體素上限、降取樣、限頻；使用者回報卡頓時調整 `OUTER_DISPLAY_MAX` |
| 佔據格跳過外圈的雲 | 外圈雲缺塊 | D1 的「內圈盒邊附近也查外圈佔據」；合成截圖測 (b)(d) |
| T 側邊吃到被海綿層壓過的主網格值 | 側邊亂流／外流被壓 | 面板說明＋足跡建議；使用者回報大小 |
| T 與主網格在 `Z1` 附近不協調（w、π′） | 反射波、假重力波 | `Z1` 附近線性混合目標；使用者實測；必要時加寬混合帶 |
| 眼區→T 回饋倍數不整除 | 眼區回饋被 T 蓋掉 | 條件檢查＋紀錄說明；面板提示倍數選擇 |
| 在 worker 裡漏掉某個 `tall`／`eye`／`ext` 分支 | 某操作壞掉（NaN、不同步） | 3.4 的逐處對照表寫進本文件；回報時列出每個分支的決定 |
| 全部都沒跑模擬 | 動力行為沒驗證 | 每次回報明說；列出使用者要測的清單 |
| 共用檔修改破壞眼區 | 使用者最在意的回歸 | Step 0；之後只加法；新功能不放進共用檔 |

## 8. 需要使用者決定的事（附預設）

1. **加高部分要不要能比主網格粗？** 預設：Step 0–3 先不做（做不到，見第 4 節）；做完 Step 3 後問使用者要不要做 Step 4。若使用者的主要需求其實是「整個範圍加高並且上層粗」，那 Step 4 才是主線——這點在「開始」之前最好確認一次。
2. 局部加高的足跡形狀：預設方形（與眼區的圓柱不同）。
3. 外圈體積顯示上限 4 M 體素、至少隔 ≈1 s 傳一次：預設。
4. 回放、存檔不含外圈／局部加高：預設。

## 9. 附錄

### 9.1 程式碼地圖
- **worker**（`src/app/regional/worker.ts`）：`nestInfo`、`stopNest`；外圈：`extDrop`、`extPrepare`、`extPrepareCpu`、`extAdapt`、`outerGpu`、`extRelink`、`rebuildInnerGpu`、`extToCpu`/`extFromCpu`、`startExt`、`extStop`；步進：`stepGpu`、`stepCpu`、`adaptDt`、`stepAfterChange`；`restoreUndo`；`build`；`moveDomain`、`nestGpuUp`、`setFollow`、`applySettings`；`regridTo`；`startNest`；`interact`；`sendFrameRaw`（`nestVolume`、`volumeBytes`/`readDisplay`）。
- **協定**：`protocol.ts`：`NestInfo`、`NestFrame`、`RegionalFrame`（`nest`、`origin`、`tracers`）、`FromRegionalWorker`（`nest`、`ext`、`settings` 訊息）。
- **頁面**：`main.ts`：`showVolume`、`lastVol`、`showNest`、`view.setVolume/setGround/clearNest`；`tools3d.ts`：`ground()`（盒外回傳 null）；`liveForm.ts`：`GridForm`（外圈欄位與按鈕）。
- **渲染器**（`volume.ts`）：著色器 `FS`：`nestW`、`nestQ`、`volAt`、`lightAt`、`hitBox`、`bytesAt`、`extAt`、主迴圈（占據格跳躍、細步、二分法找雲面）；`VolumeView`：`setVolume`、`upload`、`buildOccupancy`、`computeLight`、`computeNestLight`、`clearNest`、`setGround`、`clampTarget`、`pickPlane`/`groundAt`、uniform 設定（`uBox`、`uNOn`…）。`light.ts`：`sunSkyLight`、`nestSunSkyLight`、`occupancyGrid`。
- **外圈**：`extension.ts`：`outerSpacing`、`buildOuter`、`openInner`；`twoway.ts`：`NestGeom`（`x0,y0,ring`）、`nestTargets`、`nestFeedback`、`feedbackWeight`、`nestSurface`；`nestGpu.ts`：`GpuNest`。
- **眼區**：`eyeNest.ts`：`EyeNest`（`build`、`fit`、`cpuStep`、`gpuUp`、`roll`、`volume`…）。
- **共用**：`refine.ts`：`makeSampler`、`refineInto`、`coarsenInto`、`regridPlane`；`build.ts`：`buildModel`、`physicsOf`；`liveSettings.ts`：`applyLive`、`addWind`；`display.ts`：`volumeBytes`；`stepControl.ts`。
- **測試**：`src/tests/live.ts`（32 項、無時間步、`node dist/tests/live.js` 約 1 秒）、`assert.ts`（`check`、`summary`）。

### 9.2 指令
- 建置：`cd /home/user/treepolo-cyclone-on-a-chip && npm run build`
- 無時間步測試：`node dist/tests/live.js`（以及新增的 `outerView.js`、`tall.js`；新增後把它們加進 `package.json` 的 `test` script，放在 `live.js` 之後）
- GPU 一次性檢查：頁面暫放 repo 根目錄（例如 `gpu-scratch-*.html`），用 `node tools/gpuTest.mjs gpu-scratch-xxx.html`（頁面輸出以 `GPUTEST` 開頭，最後一行 `GPUTEST DONE`），**跑完刪掉，不提交**。
- 無頭瀏覽器截圖：`require(<npm root -g>/playwright)`，`chromium.launch({ args: ['--enable-unsafe-webgpu','--use-webgpu-adapter=swiftshader','--enable-features=Vulkan','--use-gl=angle','--use-angle=swiftshader'] })`；範例 `$SP/e2e/phone7.cjs`、`look3.cjs`。
- `SP=/tmp/claude-0/-home-user-treepolo-cyclone-on-a-chip/fee3e23d-02d3-5349-9130-445cb1ee5b1e/scratchpad`（容器重開後路徑可能不同，用系統提示裡的 Scratchpad 目錄）。
- 提交：結尾兩行 trailer（R6 第 4 點）。推送：`git push -u origin claude/3d-earth-atmosphere-simulator-ncy8n4`；網路錯誤才重試（2/4/8/16 秒）。

### 9.3 發布 Artifact（網址 https://claude.ai/artifact/G2sPhxcsoxZj9twEiAEJhC，目前 v55）
1. `npm run build`
2. `node tools/buildArtifact.mjs $SP/artifact`（產生 `index.html`、`files.json`、base64 資料檔）
3. 由 `$SP/artifact/files.json` 去掉 `data/spinup_earth_t42q.b64.txt` 這個鍵，存成 `$SP/artifact/files_pub.json`（共 80 個檔；新增的 `dist/**/*.js` 會自動出現，檔數應增加；`tests/` 與 `tools/` 不會被收）
4. 呼叫 Artifact：`file_path=$SP/artifact/index.html`、`url=https://claude.ai/artifact/G2sPhxcsoxZj9twEiAEJhC`、`label="v56 …"`、`files=<files_pub.json 的內容>`；不要傳 `icon`
5. 發布結果裡確認版本號，再回報。

### 9.4 已踩過的地雷
- `class` 欄位初始化器裡用 `this.g` 會在參數屬性初始化前執行 → 欄位只宣告，在建構子裡賦值（`GpuNest.x0/y0`）。
- `gpu = null` 之後 TS 把型別收窄成 `never` → 不要在這種地方再呼叫方法。
- 持續風掛鉤只在物理被重做時重掛（`hookForcing`／`eye.rehook()`），否則會加兩次。
- 用 bash heredoc 產生含 `${...}` 的 TS 檔會被展開 → 用 Write 工具寫檔。
- 用 python 做錨點替換前先確認錨點文字真的存在。
- 測試預期要貼近物理：蓋子處的偏差、渦旋附近的取樣點會讓期望值失準，先想清楚再寫期望。
- 大網格的 dispatch 超過 65535 個 workgroup → 用 `dispatchLinear`／`linearGid`。
- `outerSpacing` 要求外圈至少 4 格。
- 一律均勻 `dz`：任何新程式碼若用 `(k+0.5)*dz` 算高度，Step 4 時要改成 `zc[k]`；新程式碼盡量直接用 `m.zc[k]`/`m.zf[k]`。

### 9.5 回報模板（繁體中文）
1. 一句話：做完了什麼、在哪裡玩（網址、版本）。
2. **驗證過的**：列測試名稱與數字（無時間步）。
3. **沒驗證的**：哪些需要使用者跑模擬才知道，逐項。
4. **限制／做不到的**：誠實講（例如「加高部分仍不能比主網格粗」）。
5. **請你測的**：具體操作與要看的現象。
6. 若有披露（共用檔修改、行為改變），放最前面。
