# Stage 4 時間積分診斷 — 2026-09-04

## 結論

Stage 4 長期失穩的主要已知觸發因子已定位到 **large-step / fast-mode time integration**，不是單純的 model-top absorber 強度、垂直解析度、水平 CFL、水平散度或缺失的 3-D momentum transport。

目前 production `stage4.html` 仍不是封關版本；本文件記錄 real-device ablation，作為後續 RK3 + split-explicit 重構的依據。

## A. Timestep sensitivity

固定：

- cubed sphere `N=8`；
- `Nz=48`, `H_top=40 km`；
- Held–Suarez forcing；
- HEVI `epsilon=0.10`；
- 30–40 km implicit Rayleigh absorber, peak `0.2 s^-1`；
- continuity-consistent `Fref + Fpert` horizontal-momentum vertical carrier；
- complete 3-D momentum transport；
- time-normalized horizontal divergence damping。

只改完整 timestep，real-device day-2：

| full dt | global max |w| | absorber max |w| | peak location | max edge wind | mass drift |
| ---: | ---: | ---: | --- | ---: | ---: |
| 10 s | 13.640 m/s | 13.640 m/s | 38.47 km, -82.1 deg | 17.241 m/s | -2.686e-6 |
| 5 s | 0.851 m/s | 0.705 m/s | 0.39 km, -72.3 deg | 3.663 m/s | -1.994e-5 |
| 2.5 s | 0.363 m/s | 0.202 m/s | 0.39 km, -58.4 deg | 4.093 m/s | -1.250e-4 |

Absorber `w` ratio relative to 10 s：

- 5 s: `0.052`；
- 2.5 s: `0.015`。

因此 10-s large-step ordering 明顯激發 fast vertical/acoustic mode。但把所有 slow operators 一起縮成 2.5 s 會累積更大的 GPU/f32 mass drift，不能把「全模型 dt=2.5 s」當 production 解法。

## B. Fast-only split prototype

第二個 ablation 固定 meteorological outer step `dt_outer=10 s`，只 substep fast operators：

- pressure / horizontal acoustic pressure-gradient update；
- HEVI vertical acoustic update；
- explicit buoyancy；
- acoustic-divergence damping。

Slow scalar transport、3-D momentum transport、Held–Suarez 與 drag 仍每 10 s 更新一次。HEVI `Fref` 在 acoustic loop 中做時間平均，再交給 outer `Fref + Fpert` continuity/momentum bookkeeping。

Real-device day-2：

| fast split | global max |w| | absorber max |w| | max edge wind | mass drift |
| --- | ---: | ---: | ---: | ---: |
| 2 x 5 s | 0.827 m/s | 0.677 m/s | 3.717 m/s | -5.775e-6 |
| 4 x 2.5 s | 0.359 m/s | 0.219 m/s | 3.268 m/s | -3.767e-6 |

`4 x 2.5 s` 相對舊 10-s full-step absorber `w` ratio = `0.016`，且 mass drift 仍保持數 ppm 等級；因此 **fast acoustic/gravity substepping 同時解除了已知 top-w runaway 與 small-full-dt mass-drift tradeoff**。

這個結果是進入 RK3 重構的 gate。

## C. RK3 target

`docs/NUMERICAL_CORE_SPEC.md` 原本已鎖定：

1. outer meteorological step: RK3 family；
2. horizontal slow modes: explicit；
3. acoustic fast modes: split/substep；
4. vertical acoustic coupling: per-column implicit HEVI。

ARW/Wicker–Skamarock family 的 predictor form：

- stage 1: `Phi* = Phi^t + dt/3 R(Phi^t)`；
- stage 2: `Phi** = Phi^t + dt/2 R(Phi*)`；
- stage 3: `Phi^(t+dt) = Phi^t + dt R(Phi**)`。

對 `ns=4` acoustic ratio，診斷 prototype 採：

- stage 1: `1 x dt/3` acoustic step；
- stage 2: `2 x dt/4`；
- stage 3: `4 x dt/4`。

新平行路徑：`src/gpu/stage4Rk3SplitPrototypeGpu.ts` / `stage4-rk3.html`。

此 prototype 使用 predictor-reset algebra：stage advance 後只保留 `(advanced - predictor)` increment，再 re-anchor 到 large-step base state。每一 stage 都時間平均 HEVI reference mass flux後再做 scalar / momentum transport。

**仍未宣稱 production RK3 完成。** 當前 prototype 的 slow momentum / forcing 仍在 acoustic loop 後套用；若 real-device predictor-reset 診斷通過，下一步是把 stage slow RHS 固定（freeze）並在 acoustic small-step correction 中一致加入，之後才替換 production Stage-4 integrator。

## 禁止的捷徑

- 不把 production timestep 直接降到 2.5 s 當修復；
- 不提高 `max|w|` gate；
- 不 state clamp；
- 不再靠加強 absorber peak 掩蓋 time-integration 問題；
- 不把三個完整 10-s operator step 串起來冒充 RK3。
