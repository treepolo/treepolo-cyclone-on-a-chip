# Stage 4 — 旋轉全球乾大氣 / Rotating Global Dry Atmosphere

狀態：**CPU V2 reference path 已完成並通過 5/5 測試。WebGPU 旋轉核心、CPU/GPU 一致性驗證與 30 日 Held–Suarez 開發驗證已實作，但在真實裝置結果回報前 Stage 4 不封關。**

## 1. 本階段範圍

Stage 4 在 Stage 3 的三維 fully-compressible / nonhydrostatic dry core 上加入：

- Earth rotation / Coriolis；
- cubed-sphere 上的局地 east/north tangent basis；
- C-grid edge-normal velocity ↔ cell horizontal vector wind reconstruction；
- 使用與質量輸送相同 flux 的三維水平動量輸送；
- Held–Suarez Newtonian thermal relaxation；
- near-surface Rayleigh drag；
- zonal-mean temperature / zonal wind / meridional wind / overturning diagnostics；
- WebGPU rotating-core path 與真機 validation harness。

仍然沒有加入水汽、雲、微物理、地形、海氣交互作用，也沒有任何 hard-coded jet / Hadley cell / cyclone generator。

## 2. 球面風場與 Coriolis

每個 cubed-sphere cell 由 unit radial vector 建立局地 east / north 正交 tangent basis。四條 C-grid edge-normal velocity 以最小平方法重建為 cell-centered `(u_east, v_north)`；更新後再投影回 shared-edge normal velocity。

Stage 4 reference path 使用 traditional shallow-atmosphere Coriolis：

`f = 2 Ω sin(phi)`

局地水平速度滿足：

`du/dt = f v`

`dv/dt = -f u`

Coriolis 子步使用解析旋轉矩陣，因此純 Coriolis 本身不會因顯式 Euler step 人為增減速度振幅。Stage 2 規格保留完整 planet rotation vector，未把資料模型永久限制成只能使用 `f`-plane 形式；未來若需要 non-traditional Coriolis 可擴充。

Stage 4 timestep ordering：

```text
Coriolis half-step
→ Stage 3 pressure / HEVI / buoyancy / mass + thermodynamic transport
→ 3D horizontal momentum transport
→ Held–Suarez thermal relaxation + surface drag
→ Coriolis half-step
```

## 3. 三維水平動量輸送

Stage 4 不另外創造一套與質量脫鉤的風場 advection。

`DryCoreCpu` / Stage 3 GPU core 產生的 horizontal / vertical mass flux 被直接重用：

- horizontal mass flux 選 upwind cell 的 global 3D tangent wind，形成 `F_mass * u_vec` momentum flux；
- vertical perturbation mass flux 同樣攜帶水平 momentum；
- cell momentum divergence 與 mass divergence 使用同一組 shared-face flux；
- 更新後的 global vector 重新投影到 tangent plane，再以 delta 形式加回 C-grid edge velocity。

這樣可以避免「質量往 A 方向走、動量卻由另一套插值往 B 方向走」造成的非物理脫鉤。

Stage 4 目前仍沿用 Stage 3 donor-cell / upwind transport 作 correctness-first 骨架。Production high-order monotone momentum/scalar transport 仍屬後續 dry-core refinement，不能把目前一階數值擴散當成最終天氣精度。

## 4. Held–Suarez forcing

採標準 dry dynamical-core benchmark 的核心參數：

- `T0 = 315 K`
- equator-to-pole contrast `ΔTy = 60 K`
- vertical stability term `Δθz = 10 K`
- stratospheric floor `Tmin = 200 K`
- boundary-layer top `sigma_b = 0.7`
- free-atmosphere thermal relaxation `1/40 day`
- near-surface thermal relaxation `1/4 day`
- surface Rayleigh drag `1/day`

目標溫度：

`Teq = max(200 K, [315 - 60 sin²(phi) - 10 ln(sigma) cos²(phi)] sigma^kappa)`

Newtonian relaxation 直接作用於 potential temperature，並使用 exponential update；near-surface drag 同樣使用 exponential decay，避免因 forcing timestep 本身引入不必要的 Euler instability。

另外建立接近 Held–Suarez 全球平均 target 的一維 hydrostatic reference atmosphere，而不是沿用 Stage 3 固定 288 K isothermal reference，以降低大尺度背景 pressure perturbation。

初始條件加入最大約 `0.05 K` 的平滑 zonal wave thermal perturbation 作 symmetry breaker。它只打破完全球對稱，沒有指定 jet、Hadley cell、Rossby wave 或 cyclone circulation。

## 5. Reference timestep

CPU prototype 顯示目前 fully-compressible/nonhydrostatic split core 在此粗解析 Held–Suarez case：

- `dt ≈ 20–25 s`：reference path 穩定；
- `dt ≈ 30 s`：開始接近不理想；
- 更大 timestep 可能激發過強 vertical fast mode。

因此 Stage 4 correctness / climate development gate 固定 `dt = 20 s`。目前不使用大 artificial damping 掩蓋 timestep 問題。後續再透過更完整的 outer RK / fast-mode treatment / GPU batching 改善效率。

## 6. CPU V2 validation — 5/5 PASS

`npm test` 會先跑 Stage 3 7/7，再跑 Stage 4 V2 5/5。

Stage 4 CPU tests：

1. **cubed-sphere solid-body wind reconstruction**
   - N=16；
   - global relative L2 約 `3.86e-6`；
   - seam/interior RMS error ratio 約 `6.65`，低於事前 `10×` gate。
2. **inertial oscillation**
   - 45° latitude；
   - 一個完整 inertial period 後振幅／相位誤差落在 Float64 roundoff 級。
3. **discrete spherical geostrophic balance**
   - N=32；
   - analytic spherical geopotential gradient 與 Coriolis 的 RMS imbalance 約 `0.2%`，低於 `0.5%` gate。
4. **rotating hydrostatic rest**
   - Coriolis + momentum transport 打開、external forcing 關閉；
   - mass conserved、無 NaN / negative density / pressure、靜力大氣保持近乎靜止。
5. **Held–Suarez one-day sanity**
   - `6 × 4 × 4 × 12`、top 30 km、`dt=20 s`；
   - 1 day 積分保持有效狀態、mass drift 約 `10^-15` 級並自行產生非零 global wind。

一日 sanity 只檢查 solver/forcing 沒有結構性錯誤，不拿它證明三胞環流氣候已收斂。

## 7. WebGPU Stage 4 architecture

`src/gpu/rotatingDryCoreGpu.ts` 在 Stage 3 GPU core 上加入：

- Coriolis compute pass；
- cell wind reconstruction；
- horizontal / vertical momentum-flux passes；
- pre-transport density recovery；
- momentum divergence / tangent projection；
- Held–Suarez thermal forcing；
- near-surface drag；
- multi-step batching。

所有 Stage 4 compute pipeline 維持 WebGPU baseline target：**每個 stage 最多 8 個 storage buffers**。

GPU batching 允許一次 command encoder 包含多個 atmospheric timestep，避免 30-day low-resolution test 產生數十萬次 JavaScript queue submission。

## 8. 真機 Gate A — GPU/CPU 短期一致性

入口：`stage4.html` → `執行一致性驗證 / Run agreement validation`。

案例：

- grid `6 × 4 × 4 × 12`；
- Held–Suarez reference + 0.05 K wave perturbation；
- 額外加入平滑 analytic zonal wind，使 Coriolis / momentum transport 立即有非零輸入；
- CPU Float64 與 GPU Float32 獨立積分；
- `dt = 5 s`；
- checkpoints `1, 10, 100, 250, 500`。

事前鎖定 threshold：

| Metric | Gate |
|---|---:|
| GPU dry-mass drift | `≤ 2e-6` |
| `rhoD` CPU/GPU relative L2 | `≤ 1e-4` |
| `rhoThetaM` CPU/GPU relative L2 | `≤ 1e-4` |
| max `|Δu|` | `≤ 0.05 m/s` |
| max `|Δw|` | `≤ 0.02 m/s` |
| NaN / negative density / pressure | forbidden |

真機結果產生後若 FAIL，先找 solver / WGSL 差異，不以事後放寬 gate 當修正。

## 9. 真機 Gate B — 30-day Held–Suarez development

入口：`stage4.html` → `執行 30 日驗證 / Run 30-day validation`。

低解析 development grid：`6 × 4 × 4 × 12`，`dt=20 s`，30 simulated days。

每天 readback 並診斷：

- dry-mass drift；
- max vertical velocity；
- upper-midlatitude maximum zonal westerly；
- tropical low-level mean zonal wind；
- meridional overturning streamfunction；
- NH/SH dominant overturning sign。

事前 development gates：

- `|mass drift| ≤ 5e-5`；
- upper-midlatitude max westerly `> 0.5 m/s`；
- tropical low-level mean zonal wind `< 0`（easterly）；
- max overturning streamfunction `> 1e9 kg/s`；
- NH / SH dominant overturning signs opposite；
- `max |w| < 50 m/s`；
- no invalid state。

**這是 30 日、極粗解析的開發 gate，只證明旋轉乾大氣能朝合理的大尺度環流方向自行發展。正式 Held–Suarez climatology 仍需要更高解析、更長 spin-up / averaging（數百日級）並與標準文獻 zonal-mean T/u/overturning 定量比較。**

## 10. Stage 4 封關條件

Stage 4 目前只有 CPU side 可以宣稱完成。正式封關需要：

1. CPU V2 5/5 — **PASS**。
2. real-device Stage 4 WGSL pipeline smoke — pending。
3. GPU/CPU short-term agreement — pending。
4. 30-day Held–Suarez development gate — pending。

以上真機 gate 通過前，不把 Stage 4 標成 COMPLETE。

即使 Stage 4 封關，Stage 3 曾觀察到的 long-run hydrostatic residual 仍保留為 continuous regression diagnostic；不能把 balance error 誤認為真實 circulation。
