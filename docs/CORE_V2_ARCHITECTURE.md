# Core v2 — 單一桌機守恆大氣核心

狀態：**持續實作中**。本文件是 Core v2 的設計契約，不是 Stage 4 的延伸補丁清單。

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
7. **完整三維 Coriolis 保留。** 基礎旋轉算子必須零作功；目前以解析旋轉實作這個不變量。
8. **全球幾何採真正三維球殼有限體積。** 水平 cubed-sphere 控制面向徑向擠出；每格體積、每個面積與向量面積直接由幾何積分建立，而不是把平面公式硬套到球面。
9. **正式 Euler 面通量採 SLAU2 all-speed flux。** 它不需要 freestream/reference Mach 的人工調整參數；相同左右狀態必須精確退化成物理 Euler flux。
10. **正式空間重建採 compact weighted least-squares + Barth–Jespersen limiter。** 每格最多六個直接鄰居；重建以真正三維體積中心與面中心為幾何基準。限制器沒有可調係數，且密度與壓力的面值不得超出鄰近格心極值。
11. **Base core 不含下列項目：**
   - Held–Suarez 強迫
   - 近地面 Rayleigh drag
   - divergence damping
   - model-top sponge
   - 全域質量／角動量／能量 fixer
   - 人工指定的環流或風向
12. 上述外部物理或邊界機制只能在 base core 的守恆、波動、平衡與收斂測試通過後，作為明確可開關的模組加入。
13. **Core v2 不以 Stage 4 的宏觀結果作為調參目標。** 不得因為信風方向、噴流強度等結果不合期待，就反向調 numerical coefficient。

## 2. 已落地的核心結構

- `src/corev2/state.ts`
  - primitive ↔ conservative conversion
  - `rho E` 中明確包含 geopotential
  - 壓力只由保守狀態與 EOS 診斷
- `src/corev2/finiteVolume.ts`
  - 每個內部面只累加一次
  - 左／右格接收完全相反的 integrated flux
  - 質量、三維動量、總能量共用同一條 shared-face 守恆路徑
- `src/corev2/eulerFlux.ts`
  - 物理 Euler flux 與向量面積版本
- `src/corev2/sphericalShellGeometry.ts`
  - 三維球殼 control volume
  - 精確球殼總體積與球面面積
  - 每格向量面積閉合 `sum(A_f)=0`
  - 均勻壓力不會因格網幾何產生假力
- `src/corev2/slau2Flux.ts`
  - parameter-free SLAU2 all-speed numerical flux
  - 低馬赫與強可壓縮流共用同一個面通量定義
  - identical-state consistency、方向反轉對稱與靜止 contact 都有 CI gate
- `src/corev2/reconstruction.ts`
  - 真正三維 volume/face centroid
  - 六鄰居 compact weighted least-squares
  - Barth–Jespersen monotonic limiter
  - 正密度／正壓力格心不會因線性重建本身得到負的面值
- `src/corev2/shellEulerOperator.ts`
  - 一階 piecewise-constant 路徑保留作為 monotone regression floor
  - 正式空間路徑為 limited second-order reconstruction + SLAU2
  - 每一 shared face 仍只求值一次並等量反號累加
  - 現階段封閉球殼上下邊界是 stationary slip wall，只交換壓力反作用，不交換質量與能量
- `src/corev2/rotation.ts`
  - 對 `dm/dt = -2 Omega × m` 做解析旋轉
  - `rho` 與 `rho E` 不變
  - 動量長度只允許 roundoff 誤差，因此 Coriolis 不可偷偷做功

## 3. 現在尚未完成、而且必須完成的核心工作

以下不是可選項：

1. **桌機 production 時間積分／聲學壓力處理。** 必須保留完整可壓縮 Euler 物理，同時不能讓全球大氣計算被最短垂直尺度的聲學 CFL 無意義地拖垮；方法本身也必須適合單機 GPU。
2. **重力與靜力平衡閉合。** 動量中的重力、壓力梯度與 `rho E` 中的 geopotential 必須使用同一離散能量帳；靜力平衡不得靠 damping 維持。
3. **GPU production kernel。** GPU 與 CPU 不得成為兩套不同的數學模型。
4. **封閉系統長時間 gate。** 質量、總能量、角動量、靜力／地轉平衡與解析波動必須做時間步與網格收斂。
5. 上述項目通過後，才重新加入乾大氣理想化 forcing 並做氣候驗收。

在這些項目完成前，不重新啟動 Held–Suarez 氣候調參或把宏觀風場當成修數值算子的目標。

## 4. 現行 CI 不變量

Core v2 現在至少要求：

- primitive / conservative 狀態來回一致；
- 物理 Euler flux 的方向反轉對稱；
- 任意 shared internal face 不得改變全域五個守恆量；
- Coriolis 更新不得改變總能量或動量長度；
- 非正內能狀態直接拒絕；
- 全球球殼體積／面積吻合解析值；
- 每一三維 control volume 的向量面積閉合；
- 均勻 Euler state 不得被格網幾何自行加速；
- SLAU2 identical-state consistency、左右／法向反轉 antisymmetry、stationary contact preservation；
- least-squares stencil 對任意 Cartesian 線性場恢復正確梯度；
- limited reconstruction 的面密度／壓力保持在鄰近格心 bounds 內；
- 二階 shared-face operator 在封閉球殼中不得創造全域質量或總能量；
- 二階路徑的均勻靜止狀態逐格保持靜止。

這些 gate 會留在 production CI 裡，不因後續架構重寫而刪除。
