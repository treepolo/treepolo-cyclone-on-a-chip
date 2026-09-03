# Physics Specification v1

## 1. 物理目標與硬限制

Cyclone on a Chip 的正式大氣計算域是具有有限厚度的三維球殼；每一個 air column 有多個垂直層，正式動力核心直接預報垂直速度，因此上升、下沉、深對流、眼牆、高層外流與噴流垂直切變都必須由方程產生。禁止直接寫「生成颱風」「生成溫帶氣旋」「生成鋒面」「生成三胞環流」等現象規則；程式可以指定初始場、邊界條件、地表條件與物理源項，天氣結構必須由方程自行演化。

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

## 3. 正式動力核心

reference physics 採 rotating、fully compressible、non-hydrostatic Euler atmosphere。Earth preset 第一版使用 traditional shallow-atmosphere approximation；這只忽略高度相對地球半徑很小時的部分 deep-atmosphere metric terms，完全不刪除垂直維度，也不把垂直動量改成 hydrostatic constraint。

### 3.1 乾空氣質量

`∂rho_d/∂t + div(rho_d u) = 0`

乾空氣是基本 conserved mass。加入水物質後仍以 dry-air-coupled variables 為主，避免凝結與蒸發在人為改變乾空氣總量。

### 3.2 三維動量

連續方程概念：

`∂(rho_m u)/∂t + div(rho_m u⊗u) + grad(p) = -rho_m grad(Phi) - 2 rho_m Omega×u + F`

`u` 是真正三維速度，`rho_m` 包含乾空氣及所有水物質質量。離散實作使用 staggered velocity / mass flux，因此不要求程式逐字照 Cartesian conservative form；但是壓力梯度、重力、科氏力、摩擦／湍混合都必須有明確來源與診斷。

### 3.3 熱力預報量

鎖定 dry-density-coupled moist potential-temperature family，而不在 WebGPU `f32` 直接預報巨大背景 total energy。

`theta = T (p_ref/p)^kappa`

`theta_m = theta * [1 + (R_v/R_d) q_v]`

其中 `q_v` 是相對乾空氣質量的水汽 mixing ratio。cell prognostic 儲存 `rho_d theta_m`；terrain coordinate 實作時再乘對應 Jacobian/metric volume factor。

在 physical-volume form，壓力由 equation of state 診斷：

`p = p_ref * [R_d rho_d theta_m / p_ref]^gamma`

這個選擇與 MPAS-A 類 fully compressible non-hydrostatic core 的做法一致，而且適合 hydrostatic reference-state / perturbation single-precision strategy。

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
- model-top sponge，只處理人工上邊界反射
- controlled numerical diffusion / divergence damping

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