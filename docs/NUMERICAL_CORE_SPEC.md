# Numerical Core Specification v1

## 1. 水平網格：gnomonic equiangular cubed sphere

全球水平域分成六個 panel。每個 panel 使用 `alpha,beta ∈ [-pi/4,+pi/4]`，令 `X=tan(alpha), Y=tan(beta)`，透過每個 face 固定正交 basis `(C,A,B)` 映射：

`r_hat = normalize(C + X A + Y B)`

正式 uniform grid 為每面 `N × N` cells。Stage 3/4 只做 uniform cubed sphere；variable resolution 等核心驗證完成後再加入。

所有靜態幾何由 CPU Float64 一次建立，再上傳 GPU：cell center、cell area、edge endpoints/midpoint、great-circle edge length、tangent-plane edge normal、neighbor index、canonical orientation/sign、panel-seam transform。正式 timestep 禁止用 xyz 每步重新搜尋 face；P2 顯示 repeated Float32 remap 會累積可觀位置誤差。

### Staggering
- scalar mass/thermodynamic fields：cell centers
- horizontal velocity / mass flux：horizontal edges，儲存 edge-normal component
- vertical velocity `w`：vertical layer interfaces
- vorticity：由 edge circulation 診斷到適合位置

採 C-grid family，不照搬 FV3 的 D-grid/C-grid 混合。

## 2. 垂直網格：SLEVE-like hybrid geometric height

reference level 為 `zeta_k`，真實高度：

`z(x,y,zeta) = zeta + h_L(x,y) B_L(zeta) + h_S(x,y) B_S(zeta)`

`h_L` 是低通大尺度地形，`h_S = h - h_L`。小尺度地形用較低的消失高度 `H_S`，大尺度地形用較高 `H_L`，避免細山峰把座標扭曲帶到中高對流層。

第一版 decay function：

`B(zeta;H) = 1 - [6s^5 - 15s^4 + 10s^3]`, `s=clamp(zeta/H,0,1)`

它在 surface 為 1、在 `H` 為 0，且端點一階／二階導數平滑。預設候選 `H_S≈6 km`, `H_L≈18 km`；真實地形階段再依 mountain-rest benchmark 調整。

垂直 spacing 使用 stretched levels。第一個全球 dry-core 候選 `H_top=40 km`, `Nz=48–72`；高解析深對流／熱帶氣旋 preset 可用 `Nz=72–120`。model top 上方約 20–25% depth 為 sponge layer。

## 3. Prognostic state 與 reference atmosphere

Production state：
- dry density / dry mass
- `rho_d theta_m`
- horizontal edge-normal velocity or coupled momentum
- vertical-interface `w`
- 後續 `rho_d q_j`

CPU Float64 建立 hydrostatic base state `rho0(z), theta0(z), p0(z)`。GPU 使用 reference-state + perturbation-friendly representation；pressure-gradient operator 必須讓 resting hydrostatic base state 在離散上盡可能精確抵消，避免巨大背景壓力差造成假風。

## 4. Conservative finite-volume transport

所有 cell-centered conserved scalar：

`M_i^(n+1) = M_i^n - dt Σ_e F_e + dt S_i`

每一條 shared edge 的 `F_e` 只計算一份，對左右 cell 等量異號更新；global conservation 不靠事後 normalization。

Reconstruction roadmap：
- prototype/debug：1st-order upwind，驗證 conservation/positivity
- production dry core：至少 2nd/3rd-order monotone reconstruction
- moisture：positive-definite high-order flux-limited scheme，PPM/FCT/WENO-family 到 Stage 5 benchmark 再定

P2 的一階迎風實測雖然守恆且不產生負 tracer，但數值擴散過大，所以不能成為 production advection。

## 5. Pressure gradient 與地形

pressure force 使用 finite-volume face-force / consistent-metric formulation。shared face 的力必須具 equal-and-opposite pairwise contribution，並特別驗證 steep-terrain resting atmosphere。

必測：flat isothermal hydrostatic rest、stratified hydrostatic rest、Schär-like mountain rest、mountain wave。若靜止大氣在地形上自己持續生成顯著風，禁止進後續天氣階段。

## 6. 時間積分：HEVI split-explicit RK family

鎖定結構：
1. outer meteorological step：RK3 family
2. horizontal slow modes：explicit
3. acoustic fast modes：split/substep
4. vertical acoustic coupling：per-column implicit tridiagonal solve（HEVI）

P3 顯示全顯式 vertical acoustics 在 acoustic CFL > 1 後迅速不穩定；Crank–Nicolson column solve 在 CFL 3–10 仍穩定。Production 可調整 semi-implicit discretization 與 off-centering，但不得退回由最細 `dz/c_s` 限制所有氣象尺度 timestep 的架構。

runtime 每步計算 CFL diagnostics；超限時縮短 dt 或 pause/report，禁止 silently explode。

## 7. 旋轉與球面幾何

Coriolis 以向量形式 `-2 Omega × u` 為基礎再投影到當地 basis，資料結構保留完整 planet rotation vector。Earth traditional approximation 可在 reference mode 中省略非傳統分量，但不能把程式死寫成只認 `f=2 Omega sin(phi)`。

## 8. Numerical stabilization

只允許可診斷、可關閉、尺度選擇性的 stabilization：weak divergence damping、scale-selective hyperdiffusion/filter、model-top sponge、monotonic limiter。每項都要能量／質量 effect diagnostic，禁止拿大阻尼掩蓋錯誤物理。

## 9. Precision

GPU prognostic state 用 `f32`；CPU reference/tests 用 `f64`。

- 幾何以 unit sphere + local metric scale 表示，不用 6,371,000 m global Cartesian 大數做鄰格小差。
- hydrostatic base state 與 perturbation 分離。
- global reductions 使用 hierarchical pairwise reduction，必要時 compensated accumulation。
- conservation 的 CPU gold/reference 使用 Float64；GPU tolerance 依 cell count 與 reduction method 制定。

## 10. Variable resolution roadmap

uniform cubed sphere 驗證完成後比較：
1. stretched cubed sphere：拓撲簡單，適合固定關注區
2. block-structured nested patch：適合任意局部高解析與熱帶氣旋觀察

dynamic AMR / moving nest 暫不採用。

## 11. 參考
- FV3: https://www.gfdl.noaa.gov/fv3/
- FV3 key components: https://www.gfdl.noaa.gov/fv3/fv3-key-components/
- FV3 idealized tests: https://www.gfdl.noaa.gov/fv3/fv3-idealized-tests/
- ICON SLEVE: https://gmd.copernicus.org/articles/14/985/2021/
- Schär et al. 2002 DOI: 10.1175/1520-0493(2002)130<2459:ANTFVC>2.0.CO;2