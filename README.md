# treepolo Cyclone on a Chip / 晶片上的旋風

個人用的 3D 地球大氣物理模擬器。目標是用**真實物理單位與方程式**，讓三胞環流、噴流、氣團、鋒面、溫帶氣旋、高層冷心低壓、季風、熱帶氣旋（颱風眼、眼牆、雙眼牆與眼牆置換）、中尺度對流系統、龍捲風等現象**由方程自然演化**，而不是寫程式直接生成天氣。

A personal 3D planetary atmosphere simulator. Weather must emerge from real equations in SI units; nothing in the code generates a weather system directly.

## 目前狀態 / Status（v0.1）

v0.1 是一次**完整重新設計**（原因與路線圖見 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)）：

- ✅ 全球譜動力核心：靜力原始方程、球諧轉換、σ 座標（Simmons–Burridge 能量／角動量守恆）、半隱式 leapfrog。
- ✅ 基礎驗收：轉換恆等式達機器精度；靜止大氣保持靜止；20 天無強迫積分能量漂移 ~3e-7、質量 ~8e-7。
- ✅ Held–Suarez 乾大氣氣候（T42，300 日平均）：由靜止大氣自行發展出三胞環流（Hadley／Ferrel／極地胞）、33 m/s @ 43° 的渦動驅動噴流、信風與中緯西風，與 HS94 參考氣候一致（[`docs/RESULTS_R1_DRY.md`](docs/RESULTS_R1_DRY.md)）。
- ✅ Jablonowski–Williamson 斜壓波：1 m/s 小擾動在第 9 日自行發展成 949 hPa 的溫帶氣旋與冷暖鋒。
- ✅ 瀏覽器 3D 地球：模式在 Web Worker 中即時積分，顯示溫度／風／渦度／氣壓、風場示蹤粒子與緯向平均剖面。
- ✅ 濕大氣水球（R2）：形狀保持半拉格朗日水汽輸送、灰體輻射、邊界層、Simplified Betts–Miller 對流、大尺度凝結；水收支逐欄閉合。
- 🔄 地球設定（R3）：真實海陸與地形、季節日照、隨水汽變化的灰體輻射、陸地 bucket 水文、海冰反照率；季風氣候驗收中。
- ✅ WebGPU 後端（R4）：整個模式（動力、水汽輸送、柱物理）可在 GPU 上執行，並對 CPU Float64 參考解驗證；瀏覽器自動使用 WebGPU。
- ✅ 區域非靜力模式（R5）：全可壓縮 RK3 + 聲波分裂步、Kessler 暖雨、Smagorinsky 亂流、海面通量；通過 Straka 密度流基準，Weisman–Klemp 超大胞自然分裂；有 WebGPU 版本與 3D 雲體積檢視（`regional.html`）。
- ⏭ 之後：熱帶氣旋（眼、眼牆、雙眼牆、眼牆置換）驗收 → 冰相微物理 → 由全球模式驅動的巢狀網格 → 龍捲尺度 LES。

## 執行 / Run

```bash
npm install
npm test          # CPU 驗收（轉換、動力核心、水汽物理）/ CPU regressions
npm run test:gpu  # WebGPU 驗收（headless Chromium + SwiftShader）/ GPU regressions
npm run serve     # 全球模式 http://127.0.0.1:5173/ · 區域模式 http://127.0.0.1:5173/regional.html
```

長期氣候積分 / Long climate runs (Node, writes `results/<preset>/`)：

```bash
npm run climate -- T42L20 200 300     # preset, spin-up days, averaging days
node dist/tools/runJablonowski.js 42 26 900 10
npm run aquaplanet -- AQUA_T42 200 200     # 濕水球 / moist aquaplanet
npm run earth -- EARTH_T21 2 1             # 地球：季節、季風 / Earth: seasons, monsoons (years)
```

輸出包含 `summary.txt`、`climate.json` 與緯向平均 [u]、[T]、ψ、渦動通量的 SVG 圖。

## 程式結構 / Layout

| 路徑 / Path | 內容 / Contents |
|---|---|
| `src/spectral/` | Gaussian 緯度、實數 FFT、球諧轉換（純量、梯度、ζ/D ↔ U/V） |
| `src/model/dycore.ts` | 靜力原始方程譜動力核心 |
| `src/model/vertical.ts` | σ 座標與 Simmons–Burridge 垂直離散 |
| `src/model/heldSuarez.ts` | Held–Suarez (1994) 強迫 |
| `src/model/jablonowski.ts` | Jablonowski–Williamson (2006) 斜壓波初始場 |
| `src/model/diagnostics.ts` | 緯向平均、經圈流函數、渦動通量 |
| `src/model/semiLagrangian.ts` | 形狀保持半拉格朗日水汽輸送 |
| `src/model/moist/` | 濕熱力學、灰體輻射柱物理、Simplified Betts–Miller 對流 |
| `src/gpu/` | WebGPU 版本：轉換、動力核心、水汽與柱物理；`src/gpu/tests/` GPU 驗收 |
| `data/earth_t42.json` | ERA 地表高度與海陸遮罩（T42） |
| `src/regional/` | 區域全可壓縮非靜力模式、Kessler 微物理、次網格／海面物理、熱帶氣旋設定 |
| `src/app/` | 瀏覽器介面：WebGL2 地球、Web Worker 模式（WebGPU/CPU）、剖面圖；`src/app/regional/` 3D 雲體積檢視 |
| `src/tools/` | Node 長期積分與 SVG 繪圖 |
| `src/tests/` | 驗收測試 |
| `docs/` | 架構與路線圖、物理規格、驗收計畫、UI 規格 |

## 核心硬限制 / Hard constraints

- 真實物理方程與 SI 單位；參數化必須有文獻依據與物理單位。
- 天氣系統不可硬編生成；只能設定初始場、邊界條件、地表條件與物理源項。
- Eulerian 場是唯一物理狀態；畫面上的粒子只是無質量示蹤粒子。
- 每層物理先通過量化驗收才加入下一層；不以加強阻尼掩蓋錯誤。
- 介面繁體中文 + English 同時顯示（[`docs/UI_SPEC.md`](docs/UI_SPEC.md)）。

舊版（v0.0.x，立方球全可壓縮 WebGPU 核心）保存在 git 歷史，最後一個舊版 commit 為 `3cc0c9e`。
