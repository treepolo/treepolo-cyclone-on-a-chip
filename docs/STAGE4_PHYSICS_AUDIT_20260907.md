# Stage 4 物理方程與數值閉合稽核 — 2026-09-07

稽核基準：`main` 於稽核開始時為 `e8691face6a6d695a288412711509a733aed0e8d`。

本文件刻意把三類東西分開：

1. **物理方程／物理源項**：模型聲稱自然界中的空氣遵守什麼。
2. **數值離散／時間積分**：如何在有限網格上近似上述方程。
3. **人工穩定化／人工邊界處理**：不代表自然界額外存在一個力，只用來處理有限解析度與人工模式頂。

本次不修改 production 方程，只稽核「規格聲稱解什麼、目前程式實際解什麼、還有哪些不一致」。

---

## 1. 規格要求的 Stage 4 乾燥物理核心

`docs/PHYSICS_SPEC.md` 的正式目標是：

- 三維有限厚度球殼；
- rotating、fully compressible、non-hydrostatic Euler atmosphere；
- Earth 第一版採 traditional shallow-atmosphere approximation；
- 乾空氣質量守恆；
- 真正三維動量；
- `rho_d theta_m` 熱力預報量；
- ideal-gas / potential-temperature equation of state；
- gravity；
- planetary rotation / Coriolis；
- Stage 4 idealized circulation 額外加入 Held–Suarez Newtonian thermal relaxation 與 near-surface drag。

水汽、雲、微物理、潛熱、PBL 湍流、真實輻射、海洋、地形、土壤等明確屬後續階段，不是 Stage 4 缺陷。

---

## 2. 現行 production 的物理項對照

### 2.1 乾空氣質量守恆 — **有，結構上完整**

現行 RK3 路徑把 continuity 分為：

- fast reference carrier；
- slow perturbation carrier。

在 predictor 上兩者相加回完整質量通量。水平與垂直 shared-face flux 都成對更新左右／上下 cell，沒有事後 mass normalization。

**稽核結論：物理項存在；沒有發現重複計算。**

### 2.2 熱力學 `rho theta` 與狀態方程 — **有**

乾燥時壓力由：

`p = p_ref * (R_d rhoTheta / p_ref)^gamma`

診斷。無外部加熱時，`rhoTheta` 隨質量通量搬運；Held–Suarez thermal relaxation 以 source tendency 加入。

**稽核結論：物理變數選擇符合既定規格。**

### 2.3 水平壓力梯度力 — **有，但離散仍未達規格要求的 compatible / pairwise form**

目前已修正兩個舊錯誤：

- C-grid edge velocity 改用真正 shared great-circle face conormal；
- 非正交 cubed sphere 不再用 `(p_R-p_L)/centerDistance`，改為 cell tangent least-squares gradient → 左右平均 → 投影真正 face conormal。

這比舊版正確很多。

但是 `docs/NUMERICAL_CORE_SPEC.md` 原本要求 pressure force 使用 finite-volume face-force / consistent-metric formulation，shared face force 具 pairwise equal-and-opposite 結構。現行 least-squares edge acceleration 並不是這種離散；smooth test 的 pressure torque 可非常小，但 developed flow 曾出現明顯 pressure AAM torque。

**稽核結論：物理項有；數值閉合尚未完成。**

### 2.4 垂直壓力梯度＋重力／浮力 — **有，但發現一個需要重新推導的核心疑點**

目標 shallow-atmosphere vertical momentum 可寫成：

`Dw/Dt = -(1/rho) dp/dz - g`

若用 hydrostatic reference `dp0/dz = -rho0 g`，代數上可改寫為：

`Dw/Dt = -(1/rho) d(p-p0)/dz - g (rho-rho0)/rho`

但目前 `predictorVerticalPressureBuoyancyAcceleration()` 與 GPU 對應式的 pressure-perturbation denominator 使用：

`den = 0.5 * (rho0_interface_average + rho_interface_average)`

也就是 pressure perturbation 項是：

`-d(p-p0)/dz / den`

而不是 target equation 的 `-d(p-p0)/dz / rho`。

這個 blended denominator 在 reference state 附近近似相同，也維持 hydrostatic rest，但它**不是上述 Euler 方程的直接代數恆等式**。目前 repo 註解稱它為 well-balanced pressure+buoyancy formulation，但沒有看到足以證明這個非線性 denominator 與 target equation 等價的推導。

**稽核判定：P0。先重新推導／證明，不能把它當已驗證物理。**

這一點尤其值得優先檢查，因為最新 N16 late instability 發生在上層、低密度區，正是 reference/actual density 差異與 pressure perturbation treatment 最敏感的位置之一。

### 2.5 水平動量的水平輸送 — **有，且已大幅改善**

目前 production 使用：

- 與 continuity 相同的 donor total mass carrier；
- 面中心二階風場重建；
- Barth–Jespersen 類 limiter；
- conservative momentum flux；
- 再用 discrete product rule 轉成 material velocity tendency。

這一部分已不再是最初的一階 velocity-only donor。

**稽核結論：目前方向正確，已有 AAM refinement regression。**

### 2.6 水平動量的垂直輸送 — **有，且目前與 predictor continuity carrier 對齊**

目前使用：

`M_i = [rho0_interface + rho_upwind - rho0_center_upwind] * w_i * area`

搬運 horizontal momentum，再用離散 product rule 轉回速度 tendency。

這修掉了 RK3 重構後曾重新出現的 `-w du/dz` velocity-form mismatch。

**稽核結論：目前 predictor-stage mass carrier 是一致的；但仍是一階 donor reconstruction。**

### 2.7 垂直動量 `w` 的三維平流 — **有，但仍是明顯未閉合項**

CPU 與 GPU production 都仍直接使用一階 velocity-form：

`dw/dt = -u_h · grad_h(w) - w dw/dz`

它沒有像 horizontal momentum 一樣：

- 使用 staggered vertical-momentum control volume；
- 使用與 continuity 相同的 mass carrier；
- 使用 conservative momentum flux + discrete product rule；
- 建立對應 kinetic-energy / momentum compatibility。

連續方程中 material form 與 conservative form 可等價，但**離散後不會自動等價**。目前 horizontal momentum 已因這個問題改成 mass-flux-consistent；`w` 卻仍保留舊 velocity-form donor。

**稽核判定：P0。這是最新 late-`w` instability 之前應先處理／至少做 manufactured conservation test 的核心候選，而不是先加新 damping。**

### 2.8 科氏力 — **物理近似符合 Stage 4 規格；離散相容性仍未封關**

目前使用 traditional approximation：

`f = 2 Omega sin(phi)`

只作用水平風；忽略 non-traditional Coriolis 與 deep-atmosphere metric terms。這是 `PHYSICS_SPEC` 明確允許的 Stage 4 Earth reference approximation，不算缺項。

但數值上目前是：edge-normal velocity → cell wind reconstruction → cell Coriolis tendency → 相鄰 cell 平均投影回 edge。

它不是已證明 kinetic-energy/AAM-compatible 的 primal-dual Coriolis operator；目前的 synthetic refinement 雖收斂，但 developed-state `planetary mass + Coriolis` residual 仍非零。

**稽核判定：P0（針對 Stage 4 AAM / trade closure），但屬數值幾何問題，不是漏掉物理科氏力。**

### 2.9 Held–Suarez thermal relaxation — **主要公式有，但發現 benchmark 定義不一致**

`Teq(phi,p)` 的 `p/p0` 形式與標準 Held–Suarez 相符。

但 thermal relaxation rate 與 near-surface drag 的 boundary-layer coordinate，目前程式傳入：

`sigma = p / p_ref`

標準 Held–Suarez 實作則使用 local sigma：

`sigma = p / p_surface`

例如 WRF 的 Held–Suarez radiation module 明確以 `sig = p_phy / p8w(surface)` 計算 `sigma_b=0.7` 以下的 thermal-rate enhancement 與 surface drag。

因此現行模型實際上把 boundary-layer forcing 固定在約 700 hPa 以下，而不是跟著每根 column 的 local surface pressure 移動。

在平坦、surface pressure 接近 1000 hPa 時差異可能不大，但它是**真實 benchmark forcing 定義的偏差**，而不是單純數值離散問題。

**稽核判定：P1，應做 A/B，不應在 trade drift 尚未解釋時繼續假設 Held–Suarez forcing 已完全正確。**

### 2.10 近地面摩擦 — **有**

水平 edge velocity 依 Held–Suarez near-surface Rayleigh rate 阻尼。沒有額外真實 PBL turbulence；這符合 Stage 4 idealized benchmark 範圍。

---

## 3. 人工數值項／人工邊界項

這些不是「新的自然物理」。

### 3.1 RK3 + split acoustic substeps

三階外層積分＋快速聲學子步＋垂直隱式解。

**分類：必要數值積分架構，不是補一個自然力。**

已證明舊 10 s one-step ordering 會激發 fast vertical mode；新 RK3 split 在 N8 的 early-run stability 有巨大改善。

### 3.2 HEVI off-centering (`epsilon=0.10`)

**分類：數值阻尼／時間離散。**

### 3.3 Model-top implicit Rayleigh absorber

30–40 km 人工吸收區，最高 rate `0.2 s^-1`，只為避免 40 km rigid model lid 反射波。

**分類：人工邊界處理，不是自然大氣中的物理層。**

目前 production 只在 HEVI implicit solve 使用；legacy post-step sponge 沒有重複套用。

### 3.4 Horizontal divergence damping

只濾 horizontal divergent mode；垂直散度已移除。strength 依實際 small-step dt 轉成固定 physical damping timescale。

**分類：人工數值濾波。**

但現行 correction 仍以 `centerDistance * (D_R-D_L)` 形成 edge correction，沒有使用目前 pressure gradient 已採用的 non-orthogonal face-normal gradient reconstruction。因此它與新 face-conormal geometry 並未完全一致。

另外，`NUMERICAL_CORE_SPEC` 要求每個 stabilization 都有 energy/mass effect diagnostic。現在 mass effect 可追，AAM 也已有 attribution，但**尚未看到正式 total/kinetic energy effect budget 被納入 Stage 4 closure gate**。

**稽核判定：P1。不要再調 coefficient；先做 geometry-consistent form 與 energy budget。**

---

## 4. 目前沒有發現的「重複補丁」

現行 RK3 production path 經逐項檢查，以下舊／新路徑目前沒有同時重複套用：

- Coriolis：沒有再跑 legacy half-step；RK3 slow RHS only。
- top sponge：沒有再跑 legacy post-step sponge；HEVI implicit only。
- vertical pressure/buoyancy：predictor full tendency + predictor-relative linear fast correction，是同一 split linearization，不是把完整 force 加兩次。
- continuity：slow perturbation flux + fast reference flux 是 partition，不是雙重 mass transport。
- Held–Suarez：RK3 slow RHS only；沒有再額外呼叫 legacy exponential forcing path。
- divergence damping：每 acoustic small step 執行，但 coefficient 已按該 small-step dt 正規化，不再是舊版 cadence 增加十倍阻尼的錯誤。

所以現況不是「所有歷史補丁都疊在一起」。真正問題比較像：**幾個核心離散仍未閉合，加上少數人工穩定化仍有非物理 torque/energy effect。**

---

## 5. 另一個已確認的 production 規格缺口：scalar transport 仍是一階

`NUMERICAL_CORE_SPEC` 明確要求 production dry core 至少 2nd/3rd-order monotone reconstruction。

但目前 `computeStage4SlowTendencies()` 的 horizontal / vertical perturbation `rho` 與 `rhoTheta` flux 都仍是 donor-cell upwind。

Repo 已有 `stage4ScalarTransportPrototype.ts` 的 MUSCL + limiter diagnostic，但沒有 promote 到 production。

這會直接影響：

- meridional temperature gradients；
- baroclinic eddies；
- eddy heat flux；
- eddy momentum flux；
- Held–Suarez jet / Hadley / trade climatology。

因此在 N8/N16 粗網格上，用一階 scalar advection 得到的 climatology 不應被當成已完成的 production dry-core climate。

**稽核判定：P1，而且是既定 Numerical Core Spec 尚未完成，不是新需求。**

---

## 6. 物理近似／後續階段項：不是 Stage 4 bug

下列目前未實作，但都符合既定分期：

- non-traditional Coriolis；
- deep-atmosphere metric terms；
- 高度變化的 spherical radius metric；
- 水汽、雲、雨、冰、雪、霰；
- 潛熱；
- 真實長波／短波輻射；
- PBL turbulent mixing；
- surface sensible/latent heat flux；
- SST / slab ocean；
- land/soil/vegetation；
- terrain / SLEVE deformation。

Stage 4 規格本來就採 traditional shallow-atmosphere dry Euler benchmark，所以不能用這些後續物理缺項解釋目前的 AAM drift 或 numerical `w` runaway。

---

## 7. 稽核後的優先順序

### P0-A：重新推導 vertical pressure + buoyancy operator

先從 target Euler equation 出發，逐項推到 reference/perturbation form；特別回答：

- pressure perturbation denominator 為什麼是 `(rho+rho0)/2` 而不是 `rho`？
- 這是可證明的 well-balanced consistent approximation，還是歷史穩定化遺留？
- 對大 density perturbation 的 truncation error 是多少？

未回答前，不繼續調 top damping。

### P0-B：建立 mass-flux-consistent vertical momentum (`w`) manufactured test

比較目前 velocity-form donor 與 conservative/mass-carrier-compatible staggered vertical momentum operator：

- uniform `w` preservation；
- vertical momentum conservation；
- kinetic-energy work；
- resolution convergence；
- N8/N16 late-mode tendency attribution。

未有數據前，不直接 promote 新 operator。

### P0-C：把 AAM 的 pressure/Coriolis compatibility 當 architecture 問題

不要調 Held–Suarez forcing 去抵消 numerical torque。現有 primal-dual/Hodge prototype 可以繼續作架構研究，但 production promotion 必須同時滿足：

- consistency；
- energy neutrality；
- AAM convergence；
- local cost 可接受。

### P1-A：修正／A-B Held–Suarez local sigma

`Teq` 繼續用 `p/p0`；thermal-rate enhancement 與 surface drag 改為 local `p/p_surface` 候選，先 CPU diagnostic A/B 再決定 production。

### P1-B：promote high-order monotone scalar transport

把既有 scalar MUSCL prototype 做成與 reference/perturbation split 相容的 production CPU/GPU operator；重新跑 mass conservation、positivity、CPU/GPU agreement、Held–Suarez。

### P1-C：新增正式能量預算

至少分解：

- horizontal kinetic energy；
- vertical kinetic energy；
- internal/available thermodynamic proxy；
- pressure work；
- buoyancy/gravity conversion；
- Held–Suarez thermal source；
- surface drag sink；
- top absorber sink；
- divergence damping sink/source；
- residual numerical energy tendency。

這是判讀 N16 day-29 `w` growth 是否為真物理轉換、邊界反射或 numerical energy injection 的必要工具。

### P2：更新 Stage 4 文件

`docs/STAGE4_IMPLEMENTATION.md` 仍保留「下一步才做 RK3」等過期描述，已與目前 production 不一致。這會增加下一次重構把舊決策重新引入的風險。

---

## 8. 稽核總結

Stage 4 現在**不是單純靠一堆 patch 才能跑**；核心物理框架仍然清楚：dry fully-compressible non-hydrostatic rotating shallow-atmosphere Euler + Held–Suarez forcing。

但也不能宣稱「物理方程已完整無疑，只剩數值穩定性」：本次稽核至少發現兩個應升為 P0 的核心方程／離散問題：

1. vertical pressure perturbation 使用 blended `(rho+rho0)/2` denominator，尚未證明與 target Euler equation 一致；
2. prognostic `w` 的 nonlinear advection 仍是 velocity-form first-order donor，沒有完成與 continuity 相容的 vertical momentum closure。

另外，Held–Suarez boundary-layer `sigma` 使用 `p/p_ref` 而非 local `p/p_surface`，以及 production scalar transport 仍是一階，都是在繼續解讀 trade climatology 前應處理的實質缺口。

因此下一步不應再加新的 damping patch，而應先完成上述 equation-level audit items，再回到 N16 long-run。