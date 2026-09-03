# treepolo Cyclone on a Chip

個人用的全球三維大氣物理模擬器。計算域是具有實際厚度的 3D spherical atmosphere，正式動力核心直接計算垂直速度與對流；長期目標是讓三胞環流、西風帶、Rossby 波與槽脊、鋒面、溫帶氣旋、濕對流、暖心熱帶氣旋與季風等現象由物理方程自然演化，而非用規則直接生成天氣系統。

## 目前進度

- Stage 1：技術研究與核心選型 — 完成。
- Stage 2：風險 prototype + 完整物理／數值／資料／驗證規格 — 完成。
- Stage 3：最小三維 dry non-rotating core — **完成**。CPU Float64 7/7、真機 WebGPU pipeline/smoke、1000-step hydrostatic-rest / conservation / CPU-vs-GPU agreement 全部通過。
- Stage 4：旋轉全球乾大氣 — **CPU V2 reference 5/5 tests 已通過；WebGPU rotating core、GPU/CPU agreement 與 30-day Held–Suarez development gate 已實作，等待真機驗收。**

Stage 2：`docs/STAGE2_COMPLETE_SPEC.md`  
Stage 3：`docs/STAGE3_IMPLEMENTATION.md`  
Stage 4：`docs/STAGE4_IMPLEMENTATION.md`

## 執行 / Run

```bash
npm install
npm test
npm run serve
```

Stage 3 Debug Viewer：`http://127.0.0.1:5173/`  
Stage 4 Validation：`http://127.0.0.1:5173/stage4.html`

Stage 4 頁面啟動時會先建立並執行 rotating WebGPU pipeline smoke。通過後依序執行：

1. `執行一致性驗證 / Run agreement validation`
2. `執行 30 日驗證 / Run 30-day validation`

真機結果未通過以前，Stage 4 不標記 COMPLETE。

使用者介面依 `docs/UI_SPEC.md` 固定採繁體中文 + English 同時顯示，不使用語言切換作為主要介面模式。

## Stage 3 final GPU result

Windows + Chrome，`6 × 8 × 8 × 32` debug grid，`dt = 0.25 s`，1000 steps：

- GPU dry-mass drift：`4.341e-7`。
- GPU resting-atmosphere max `|w|`：`9.828e-4 m/s`。
- `rhoD` CPU/GPU relative L2：`1.605e-6`。
- `rhoThetaM` CPU/GPU relative L2：`8.172e-7`。
- max `|Δu| = 0`。
- max `|Δw| = 9.828e-4 m/s`。
- 無 NaN、負密度或負壓力錯誤。

所有數值均通過事前鎖定的 Stage 3 gate。`max |w|` 已接近目前 gate，因此後續長時間積分持續保留 hydrostatic/balance residual regression。

## Stage 2 prototypes

```bash
python prototypes/stage2_reference.py
python prototypes/stage2_reference.py --full
```

P1 WebGPU 裝置 benchmark：由本機 HTTP server 開啟 `prototypes/p1_webgpu.html`，詳細方式見 `prototypes/README.md`。

## 核心硬限制

- 真正 3D、有厚度的大氣球殼。
- fully compressible、non-hydrostatic atmosphere。
- 天氣系統不可硬編生成。
- Eulerian field 才是物理狀態；可見粒子只作 Lagrangian tracer / visualization。
- mass / water transport 使用 conservative finite-volume flux。
- 每一層物理在加入下一層以前都要通過 quantitative benchmark。
