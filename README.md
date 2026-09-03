# treepolo Cyclone on a Chip

個人用的全球三維大氣物理模擬器。計算域是具有實際厚度的 3D spherical atmosphere，正式動力核心直接計算垂直速度與對流；長期目標是讓三胞環流、西風帶、Rossby 波與槽脊、鋒面、溫帶氣旋、濕對流、暖心熱帶氣旋與季風等現象由物理方程自然演化，而非用規則直接生成天氣系統。

## 目前進度

- Stage 1：技術研究與核心選型 — 完成。
- Stage 2：風險 prototype + 完整物理／數值／資料／驗證規格 — 完成。
- Stage 3：最小三維 dry non-rotating core — CPU reference 7/7 tests 通過；真實 Windows + Chrome WebGPU pipeline/smoke gate 已通過。
- Stage 3 multi-step GPU hydrostatic-rest + CPU-vs-GPU 驗證 harness 已實作，等待真機 1000-step 結果。
- 下一步：multi-step GPU gate 通過後正式封關 Stage 3，再進 Stage 4 旋轉全球乾大氣。

Stage 2 入口：`docs/STAGE2_COMPLETE_SPEC.md`  
Stage 3 實作與結果：`docs/STAGE3_IMPLEMENTATION.md`  
Stage 3 真機 GPU smoke：`docs/STAGE3_GPU_SMOKE_20260903.md`  
Stage 3 GPU 多步驗證：`docs/STAGE3_GPU_MULTISTEP_VALIDATION.md`

## Stage 3 run

```bash
npm install
npm test
npm run serve
```

開啟 `http://127.0.0.1:5173`。Debug Viewer 可以旋轉真正的 3D 大氣球殼、單步／連續積分並插入 constant-pressure thermal bubble；頁面同時會建立 WebGPU compute pipelines 並跑 hydrostatic smoke test。

one-step smoke 通過後，按 `執行 1000 步驗證 / Run 1000-step validation` 執行 Stage 3 最後的真機 GPU gate。

使用者介面依 `docs/UI_SPEC.md` 固定採繁體中文 + English 同時顯示，不使用語言切換作為主要介面模式。

## Stage 2 prototypes

```bash
python prototypes/stage2_reference.py
python prototypes/stage2_reference.py --full
```

P1 WebGPU 裝置 benchmark：由本機 HTTP server 開啟 `prototypes/p1_webgpu.html`，詳細方式見 `prototypes/README.md`。

## 核心硬限制

- 真正 3D、有厚度的大氣球殼。
- fully compressible、non-hydrostatic atmosphere；旋轉從 Stage 4 開始加入。
- 天氣系統不可硬編生成。
- Eulerian field 才是物理狀態；可見粒子只作 Lagrangian tracer / visualization。
- mass / water transport 使用 conservative finite-volume flux。
- 每一層物理在加入下一層以前都要通過 quantitative benchmark。
