# Core v2 — 單一桌機守恆大氣核心

狀態：**已開始實作**。本文件是 Core v2 的設計契約，不是 Stage 4 的延伸補丁清單。

## 1. 已確定的設計決策

1. **只有一套核心。** 不另外維護「參考核心」與「桌機核心」。所有理論測試、收斂測試、CPU/GPU 對照都必須驗證同一套數學演算法。
2. **普通桌機是硬限制。** 演算法從一開始就必須適合局部 stencil、GPU 平行化與有限記憶體；不把「之後上超級電腦」當解法。
3. **守恆量直接作為主要狀態：**
   - `rho`
   - `rho * u_x`
   - `rho * u_y`
   - `rho * u_z`
   - `rho * E`
4. **動量使用地心三維 Cartesian 分量。** 球面上的東／北／垂直分量是診斷或邊界幾何，不再是彼此分離的預報方程。
5. **同一個 shared-face 通量只計算一次。** 一個內部面左格拿走多少，右格就收到同一筆數值；質量、三維動量、總能量均如此。
6. **總能量包含內能、三維動能與 geopotential energy。** 重力交換必須在離散上和總能量一致，禁止靠事後修正總能量。
7. **完整三維 Coriolis 保留。** 基礎旋轉算子必須零作功；目前先以 exact rotation 實作這個不變量。
8. **Base core 不含下列項目：**
   - Held–Suarez 強迫
   - 近地面 Rayleigh drag
   - divergence damping
   - model-top sponge
   - 全域質量／角動量／能量 fixer
   - 人工指定的環流或風向
9. 上述外部物理或邊界機制只能在 base core 的守恆、波動、平衡與收斂測試通過後，作為明確可開關的模組加入。
10. **Core v2 不以 Stage 4 的宏觀結果作為調參目標。** 不得因為信風方向、噴流強度等結果不合期待，就反向調 numerical coefficient。

## 2. 第一批已落地的程式不變量

這一批程式碼建立最底層、之後不可破壞的結構：

- `src/corev2/state.ts`
  - primitive ↔ conservative conversion
  - `rho E` 中明確包含 geopotential
  - 壓力只由保守狀態與 EOS 診斷
- `src/corev2/eulerFlux.ts`
  - 定義單一狀態穿過有向面的物理 Euler flux
  - 當左右狀態相同時，任何後續數值面通量都必須退化到這個定義
- `src/corev2/finiteVolume.ts`
  - 每個內部面只累加一次
  - 左／右格接收完全相反的 integrated flux
  - 這是質量、三維動量、總能量全域守恆的基礎
- `src/corev2/rotation.ts`
  - 對 `dm/dt = -2 Omega × m` 做解析旋轉
  - `rho` 與 `rho E` 不變
  - 動量長度只允許 roundoff 誤差，因此 Coriolis 不可偷偷做功

## 3. 目前刻意尚未實作的部分

以下不是「可有可無」；它們是 Core v2 必須完成的後續核心工作，只是這一筆提交不假裝已經完成：

1. shared-face 左右狀態重建與正式數值面通量；
2. 最終桌機用的全馬赫數時間積分／壓力處理；
3. 與總能量相容的重力離散與 hydrostatic well-balancing；
4. 全球三維球殼控制體與真正面積／體積幾何；
5. GPU production kernel；
6. 封閉系統的質量、總能量、角動量長時間收斂 gate。

在上述項目完成前，不重新啟動 Held–Suarez 氣候調參或長時間 climatology 驗收。

## 4. 這一批的驗收

`src/tests/coreV2.ts` 現在要求：

- primitive / conservative 狀態來回一致；
- 面法向反轉時物理 Euler flux 精確反號；
- 靜止氣體只有壓力動量通量；
- 任意單一內部 face flux 不得改變全域五個守恆量；
- 任意封閉 internal-face network 不得改變全域五個守恆量；
- Coriolis 解析更新不得改變動量長度、總能量或壓力；
- 非正內能狀態必須直接拒絕，不可繼續污染模擬。

這些 gate 之後會留在 production CI 裡，不因後續架構重寫而刪除。
