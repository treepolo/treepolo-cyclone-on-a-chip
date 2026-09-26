# Physics Specification v2（2026-09-26 依新架構修訂 / revised for the v0.1 architecture）

## 1. 物理目標與硬限制

Cyclone on a Chip 的大氣計算域是具有有限厚度的三維球殼，每個 air column 有多個垂直層；上升、下沉、深對流、眼牆、高層外流與噴流垂直切變都必須由方程產生。禁止直接寫「生成颱風」「生成溫帶氣旋」「生成鋒面」「生成三胞環流」等現象規則；程式可以指定初始場、邊界條件、地表條件與物理源項，天氣結構必須由方程自行演化。

不同尺度由不同層級的模式負責（見 `ARCHITECTURE.md`）：全球模式用靜力原始方程（對 >10 km 格距是真實方程的嚴格尺度近似）；需要非靜力效應的現象（眼牆內核、對流胞、龍捲）由區域／局地的全可壓縮非靜力巢狀模式計算。

## 2. 單位與地球 preset

全系統使用 SI：m、s、kg、K、Pa、J。基本常數集中在 Planet/Atmosphere configuration，不散落在 shader magic number。

- `R_e = 6.371e6 m`
- `Omega = 7.292115e-5 s^-1`
- `g0 = 9.80665 m s^-2`
- `R_d = 287.05 J kg^-1 K^-1`
- `R_v = 461.5 J kg^-1 K^-1`
- `c_pd = 1004.5 J kg^-1 K^-1`
- `c_vd = c_pd - R_d`
- `gamma = c_pd / c_vd`
- `kappa = R_d / c_pd`
- `p_ref = 100000 Pa`

日後可修改行星半徑、自轉率、重力、日照等做理想化實驗。

## 3. 動力核心 / Dynamical cores

### 3.1 全球模式：靜力原始方程（已實作）

σ = p/p_s 座標，traditional shallow-atmosphere approximation：

- 水平動量（向量不變式）：`∂v/∂t = −(ζ+f) k×v − σ̇ ∂v/∂σ − ∇(Φ + ½|v|²) − R_d T ∇ln p_s + F`
- 熱力學：`∂T/∂t = −v·∇T − σ̇ ∂T/∂σ + κ T ω/p + Q`
- 連續方程：`∂ln p_s/∂t = −∫₀¹ (D + v·∇ln p_s) dσ`
- 靜力方程：`∂Φ/∂ln σ = −R_d T`
- 理想氣體：`p = ρ R_d T`

其中 `f = 2Ω sin φ`，F 與 Q 由物理參數化提供（有明確單位 m s⁻² 與 K s⁻¹）。離散方法見 `ARCHITECTURE.md` §2。

加入水汽後改用虛溫 `T_v = T (1 + (R_v/R_d − 1) q_v)` 於靜力方程與氣壓梯度力，水物質以相對乾空氣或濕空氣的 mixing ratio 預報，輸送必須正定且守恆。

### 3.2 區域／局地模式：全可壓縮非靜力 Euler 方程（R5 起）

`∂ρ_d/∂t + ∇·(ρ_d u) = 0`

`∂(ρ_m u)/∂t + ∇·(ρ_m u⊗u) + ∇p = −ρ_m g k − 2ρ_m Ω×u + F`

熱力預報量採 `ρ_d θ_m`，`θ_m = θ [1 + (R_v/R_d) q_v]`，壓力由狀態方程 `p = p_ref [R_d ρ_d θ_m / p_ref]^γ` 診斷（MPAS／CM1 類做法）。笛卡兒 C 網格、分裂顯式聲波 + 垂直隱式。

## 4. 水物質與濕熱力學

Stage 5 起至少加入：

- `q_v` 水汽
- `q_c` 雲水
- `q_r` 雨水
- `q_i` 雲冰
- `q_s` 雪
- `q_g` 霰，可在較完整 microphysics tier 啟用

每個 `q_j` 單位為 kg water / kg dry air，預報 conserved `rho_d q_j`。所有 advection 必須 positive-definite；相變 source/sink 成對守恆總水量，凝結／蒸發／凍結／融化的潛熱同步回饋熱力方程。

`rho_m = rho_d * (1 + Σ q_j)`。氣體壓力由乾空氣與水汽 partial pressure 貢獻；液態／固態凝結物增加質量，但不直接提供氣體分壓。

## 5. 逐層加入的物理源項

### Dry core
- gravity
- planetary rotation / Coriolis
- scale-selective ∇⁸ hyperdiffusion（只作用於截斷尺度附近，e-folding 0.1–0.25 day）

### Idealized global circulation
- Held–Suarez 類 Newtonian thermal relaxation
- near-surface drag

### Moist atmosphere
- saturation / cloud microphysics
- latent heat
- precipitation sedimentation
- surface sensible / latent heat flux
- boundary-layer turbulent mixing
- longwave / shortwave radiation：先簡化、後升級

### Realistic surface
- SST / slab-ocean boundary
- land heat capacity / skin temperature
- soil moisture
- albedo
- aerodynamic roughness
- topography
- vegetation optional tier
- diurnal/seasonal solar forcing and axial tilt

## 6. 人為改變天氣的合法接口

玩家工具只能產生有單位、可記帳的物理 source 或 boundary change，例如：

- volumetric heating `W m^-3` 或 mass-specific heating `W kg^-1`
- water-vapor source `kg kg^-1 s^-1`
- momentum forcing `m s^-2`
- SST / surface temperature change
- albedo / soil moisture / roughness / terrain change

每次操作必須記錄累積加入或移除的質量、能量與動量。禁止直接修改「颱風強度」「高壓中心」「鋒面位置」等診斷結果。

## 7. 粒子

可見粒子為無質量回饋的 Lagrangian tracers。粒子從 Eulerian 3D velocity field 插值速度並積分 trajectory，可攜帶 sampled `T,p,qv,w,vorticity,theta` 等顯示資訊。粒子數量本身不參與 equation of state，也不能擁有獨立碰撞、獨立氣壓或額外慣性系統。

## 8. 現象與必要物理

- 三胞環流：緯向熱力強迫 + 旋轉 + 摩擦 + eddy transport 的時間／緯向平均結果。
- 西風帶、Rossby 波、槽脊：球面旋轉與 potential-vorticity dynamics 自然結果。
- 溫帶氣旋／鋒面：斜壓不穩定與 frontogenesis 自然結果。
- 熱帶氣旋：暖海面、surface enthalpy flux、水汽、潛熱、旋轉、boundary layer、對流的 coupled feedback；禁止 cyclone generator。
- 季風：陸海熱力差、季節日照、水循環與地形共同結果。

## 9. 技術參考

- MPAS-A technical note: https://www2.mmm.ucar.edu/projects/mpas/mpas_website_linked_files/MPAS-A_tech_note.pdf
- MPAS `theta_m` field definition: https://www2.mmm.ucar.edu/projects/mpas/site/documentation/users_guide/appD_fields.html
- WRF moist Euler formulation: https://www2.mmm.ucar.edu/wrf/users/docs/technote/v2_technote.pdf