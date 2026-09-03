# Stage 4 RK3 + split-explicit 正式重構規格

狀態：**設計鎖定；GPU production implementation 尚未完成。**

## 1. 為何要重構

Stage 4 真機 timestep sensitivity 與 split prototype 已經把問題定位到 large-step / fast-mode time integration：

- 舊 production `dt=10 s`：day 2 absorber `max|w| = 13.640 m/s @ 38.47 km`；
- 全模型 `dt=5 s`：day 2 absorber `0.705 m/s`；
- 全模型 `dt=2.5 s`：day 2 absorber `0.202 m/s`，但 mass drift 惡化到 `-1.250e-4`；
- outer `dt=10 s` + `4 × 2.5 s` fast acoustic/gravity substeps：day 2 absorber `0.219 m/s`，mass drift `-3.767e-6`。

因此正式方向是：保留 meteorological outer step，同時把 fast acoustic/gravity modes subcycle，而不是把所有 slow transport/physics 一起縮到 2.5 s。

## 2. 被否決的錯誤 RK3 scaffold

曾實作：

1. 以最新 RK predictor `P` 為初值直接跑一個有限時間的 fast+slow stage；
2. 得到 `advanced`；
3. 最後用 `B + (advanced - P)` 回錨到 large-step base `B`。

真機在第一個 `0.25 day` checkpoint 前即產生 NaN，因此此 formulation 已刪除並標記 REJECTED。

結構錯誤在於：對 Wicker–Skamarock / WRF split-explicit RK3，stage 2/3 不是「從 predictor 跑完整 finite-step map 再減回 predictor」。正確方法是：

- predictor 只用來評估該 RK stage 的 slow RHS，並在 acoustic loop 期間凍結；
- 每個 RK stage 的 small-step integration 都從 large-step base time `t` 重新積到該 stage 目標時間；
- acoustic small-step variables 定義成目前 full state 相對最新 RK predictor 的 perturbations；
- fast pressure/acoustic/gravity terms以 predictor-relative perturbation equations 積分。

對非線性、implicit HEVI、Rayleigh damping、exact Coriolis/forcing maps，`B + [M(P)-P]` 不等價於上述方法。

## 3. RK3 stage schedule

採 Wicker–Skamarock / ARW family：

- stage 1 target：`t + dt/3`
- stage 2 target：`t + dt/2`
- stage 3 target：`t + dt`

slow RHS evaluation points：

- stage 1：`Phi^t`
- stage 2：`Phi*`
- stage 3：`Phi**`

對目前 `dt=10 s`、acoustic ratio `ns=4`：

- stage 1：single acoustic step `10/3 s`
- stage 2：2 acoustic steps × `2.5 s`
- stage 3：4 acoustic steps × `2.5 s`

第一 stage 單步 `dt/3` 是 Wicker–Skamarock / ARW 允許的 modified first acoustic stage。

## 4. Project-specific state decomposition

目前 prognostic storage：

- cell `rhoD`
- cell `rhoThetaM`
- edge-normal `uEdge`
- interface `wInterface`

每一個 RK stage 必須保存：

- large-step base `B = Phi^t`
- latest predictor `P = Phi^t / Phi* / Phi**`
- acoustic current state `A`
- predictor-relative acoustic perturbation `delta = A - P`
- frozen slow tendencies `S(P)`

禁止再用事後 re-anchor finite-step map 取代 predictor-relative acoustic equations。

## 5. Fast / slow operator boundary

### Frozen slow RHS（每 RK stage 只評估一次）

- horizontal + vertical advective transport of `rhoD` / `rhoThetaM` excluding reference acoustic flux；
- horizontal momentum advection；
- vertical transport of horizontal momentum；
- `w` horizontal / vertical advection；
- Coriolis tendency；
- Held–Suarez thermal tendency；
- near-surface drag tendency；
- future scale-selective slow filters where appropriate。

這些必須改成 **tendency form**，不能在 acoustic loop 後用一個 finite-step map 代替。

### Fast acoustic/gravity correction（每 acoustic small step）

- horizontal pressure/acoustic correction；
- vertical pressure/acoustic HEVI column solve；
- gravity/buoyancy correction in a consistently derived small-step form；
- horizontal acoustic-divergence damping；
- model-top implicit Rayleigh `w` absorber。

fast equations 必須以 predictor-relative perturbations 為主，而不是每個 acoustic substep 把 full slow operator 重算一次。

## 6. Conservation / mass-flux requirement

上一輪已驗證：horizontal momentum vertical carrier 必須和 continuity 使用相同的 effective vertical mass flux。

正式 RK3 split 需要每個 stage：

1. acoustic loop 累積 / 時間平均 HEVI reference mass flux `Fref_stage`；
2. slow scalar tendency 中包含 perturbation vertical flux `Fpert_stage`；
3. momentum transport 使用同一個 stage-consistent `Fref_stage + Fpert_stage`；
4. final RK update 以 conservative scalar tendency 組合，不能事後 normalization。

## 7. 實作順序

### A. CPU reference first

先建立 CPU Float64 stage-tendency API：

- `computeSlowTendencies(P)`：純函數，不修改 state；
- predictor-relative acoustic stage integrator；
- RK3 stage restart semantics；
- stage mass-flux accumulator。

### B. CPU regression gates

至少新增：

1. scalar linear ODE RK3 stage algebra；
2. predictor/base restart regression：stage 2/3 必須從 base time 積分，不可 chained finite-step；
3. hydrostatic rest under RK3 split；
4. acoustic standing-wave stability / phase；
5. stage mass conservation；
6. Stage 4 rotating hydrostatic rest；
7. short Held–Suarez sanity。

### C. GPU port

只有 CPU reference 全部通過後，才把同一 state/tendency decomposition port 到 WGSL。GPU/CPU agreement 仍是 production 前 gate。

### D. Real-device gates

先 2-day structural gate：

- no invalid；
- `|mass drift| <= 5e-5`；
- absorber `max|w| < 2 m/s`；
- no localized horizontal-wind blow-up。

再回到正式 30-day Held–Suarez gate；原 gate 不放寬。

## 8. 已鎖定的否定項

- 不再提高 Rayleigh peak 來掩蓋 time-integration 問題；
- 不把 production dt 全面降到 2.5 s 當作最終解；
- 不再使用 `base + (advanced - predictor)` 的 finite-step re-anchor RK 假近似；
- 不 clamp state；
- 不放寬 `max|w|` gate。

## 9. References

- Wicker, L. J. & Skamarock, W. C. (2002), *Time-Splitting Methods for Elastic Models Using Forward Time Schemes*, Monthly Weather Review 130, 2088–2097.
- WRF-ARW Technical Note v4, Section 3.1: Runge–Kutta and acoustic time-split integration.
