# Stage 4 上界／sponge 定位診斷 / Upper-boundary and sponge diagnosis

日期 / Date: 2026-09-04

## 真機定位結果 / Real-device localization result

在 `N=8 × Nz=20`, `H_top=30 km`, `dt=10 s`, HEVI `epsilon=0.10`, time-normalized horizontal divergence damping 與既有 top sponge 下，30-day Held–Suarez gate 於 day 14.75 失敗：

- global `max |w| = 11.7864 m/s`;
- max-|w| location = `z=20.28 km`, latitude `82.1 deg`;
- max `|w|` below sponge = `11.7864 m/s`;
- max `|w|` in sponge = `8.3876 m/s`;
- max edge wind = `87.268 m/s`;
- horizontal divergence RMS = `5.043e-7 s^-1`;
- max horizontal CFL = `9.013e-4`;
- max vertical CFL = `6.404e-2`;
- dry-mass drift = `-3.068e-5`.

從約 day 1 到 day 14.5，全域最大垂直速度大多位於 `z≈27.31 km` 的 sponge 區；失敗 checkpoint 才跳到 `20.28 km`、高緯度，位於原 sponge 起點 `22.5 km` 下方。

因此目前證據不支持 advective CFL 崩潰，也不支持 horizontal divergence 突然爆炸。更符合資料的是模式頂／上層波動長時間累積與吸收層解析不足。

## 發現：30 km × 20 層的 sponge 數值上只有約兩個有效 interior interface

`buildStretchedVerticalGrid(20, 30000, 1.4)` 的上層 interface 約為：

- `20.28 km`
- `22.46 km`
- `24.80 km`
- `27.31 km`
- `30.00 km` model top

目前 sponge 從 `0.75 H_top = 22.5 km` 開始；`22.46 km` 尚未進入 sponge，`30 km` 為剛性 `w=0` 邊界，所以真正有非零 damping 且仍是 interior `w` DOF 的主要 interface 只有約 `24.80` 與 `27.31 km`。這不足以構成良好解析的 absorbing layer。

## 與既有數值規格的落差

`docs/NUMERICAL_CORE_SPEC.md` 已指定第一版全球 dry core 候選：

- `H_top = 40 km`;
- `Nz = 48–72`;
- model-top 上方約 `20–25%` depth 為 sponge layer。

先前 long-run gate 的 `30 km × 20` 僅為早期低成本 development grid，現在已證明不足以作為上界穩定性驗收網格。

## 修正 / Correction

長期 Stage 4 gate 改為：

- horizontal cubed-sphere `N=8` 不變；
- `Nz: 20 -> 48`;
- `H_top: 30 km -> 40 km`;
- stretch `1.4` 不變；
- sponge start fraction `0.75` 不變，因此 sponge 為約 `30–40 km`；
- `dt=10 s` 不變；
- HEVI `epsilon=0.10` 不變；
- time-normalized horizontal divergence damping 不變；
- all physical and numerical pass/fail gates unchanged。

在 `Nz=48, H_top=40 km, stretch=1.4` 下，上方 25% sponge 有約 7 個 active interior interfaces。驗證程式新增 invariant：active sponge interior interfaces 必須 `>= 6`，否則 long-run gate 拒絕啟動。

此變更的目的不是用更強阻尼掩蓋不穩定，而是讓既有 sponge treatment 得到最低限度的垂直解析度，並把 artificial model top 從先前問題集中的 20–30 km 區域向上移開。

## 下一個判讀 / Next interpretation

若 `40 km × 48` 能顯著降低／延後上層 `w` growth，表示先前主要瓶頸確實包含 under-resolved absorbing layer。

若在有充分 sponge layers 後仍於 30 日內出現 sponge 以下的 `max |w| > 10 m/s`，且 CFL 仍低，下一步才進入規格已預留的 scale-selective hyperdiffusion/filter 與 outer RK3/split-explicit time integration，而不是再調 sponge 強度或放寬 gate。
