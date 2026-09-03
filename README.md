# treepolo Cyclone on a Chip

個人用的全球三維大氣物理模擬器。計算域是具有實際厚度的 3D spherical atmosphere，正式動力核心會直接計算垂直速度與對流；目標是讓三胞環流、西風帶、Rossby 波與槽脊、鋒面、溫帶氣旋、濕對流、暖心熱帶氣旋與季風等現象由物理方程自然演化，而非用規則直接生成天氣系統。

## 目前進度

- Stage 1：技術研究與核心選型 — 完成。
- Stage 2：風險 prototype + 完整物理／數值／資料／驗證規格 — 完成。
- 下一步：Stage 3 最小三維物理核心。

Stage 2 入口：`docs/STAGE2_COMPLETE_SPEC.md`

## Stage 2 prototypes

CPU Float64 的 cubed-sphere transport 與 vertical-acoustic reference：

```bash
python prototypes/stage2_reference.py
```

完整解析度 P2 reference：

```bash
python prototypes/stage2_reference.py --full
```

P1 WebGPU 裝置 benchmark 請由本機 HTTP server 開啟 `prototypes/p1_webgpu.html`；詳細方式見 `prototypes/README.md`。

## 核心硬限制

- 真正 3D、有厚度的大氣球殼。
- fully compressible、non-hydrostatic、rotating atmosphere。
- 天氣系統不可硬編生成。
- Eulerian field 才是物理狀態；可見粒子只作 Lagrangian tracer / visualization。
- mass / water transport 使用 conservative finite-volume flux。
- 每一層物理在加入下一層以前都要通過 quantitative benchmark。