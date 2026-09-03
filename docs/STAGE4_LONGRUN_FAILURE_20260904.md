# Stage 4 長時間 Held–Suarez 首次真機失敗與修正 / First long-run failure and correction

日期 / Date: 2026-09-04

## 真機結果 / Real-device result

Stage 4 Gate A（GPU Float32 / CPU Float64 short-term agreement）通過：

- 500 steps
- GPU mass drift `6.138e-8`
- `rhoD` relative L2 `1.398e-5`
- `rhoThetaM` relative L2 `3.932e-6`
- max `|Δu| = 2.578e-3 m/s`
- max `|Δw| = 2.302e-3 m/s`

原 30-day Held–Suarez gate 使用 `dt=20 s`，在第 4 日失敗。前 3 日：

| Day | mass drift | upper-midlatitude westerly | tropical low-level zonal wind | max overturning | max `|w|` |
|---:|---:|---:|---:|---:|---:|
| 1 | `2.885e-7` | `2.068 m/s` | `-0.491 m/s` | `5.119e10 kg/s` | `0.6785 m/s` |
| 2 | `2.352e-7` | `3.608 m/s` | `-0.255 m/s` | `2.576e10 kg/s` | `2.644 m/s` |
| 3 | `3.601e-7` | `4.364 m/s` | `-0.304 m/s` | `3.478e10 kg/s` | `7.208 m/s` |
| 4 | `NaN` | `NaN` | `NaN` | `NaN` | `NaN` |

這代表西風、低層熱帶東風與翻轉環流在失敗前確實已開始自行形成，但垂直快模態的振幅快速增長，最終使狀態失效。不能把這次結果判為氣候 gate 通過。

## 技術判讀 / Technical interpretation

兩個先前尚未被長期證據驗證的項目被暴露出來：

1. Stage 2 數值規格原本要求 model-top sponge，但 Stage 3/4 第一版尚未實作。30 km rigid `w=0` artificial top 會反射非靜力重力／聲波，長時間下可能使快模態能量累積。
2. `dt=20 s` 只通過一日 CPU sanity；一日穩定不足以證明它可作長期 reference timestep。第 4 日真機爆炸否定了先前較樂觀的長期假設。

本次不使用 density/pressure clamp、事後 normalization 或放寬 circulation gate 來掩蓋問題。

## 修正 / Correction

### Model-top absorbing layer

新增 `src/physics/modelTopSponge.ts` 與 WebGPU 對應 pipeline：

- sponge start = `0.75 H_top`；
- 往上使用平滑 `sin²` ramp；
- top maximum Rayleigh rate = `1/600 s^-1`；
- 只阻尼 vertical velocity `w`；
- 下方 75% 大氣完全不受 sponge 影響；
- 不修改 mass / thermodynamic variables。

它的用途是吸收人工模式頂反射的垂直波，不是壓掉對流層的 Hadley circulation。

### Long-run reference timestep

30-day development gate 改為 `dt=10 s`。這是由 long-run stability evidence 修正 reference timestep，不是放寬驗收。

### Better early diagnostics

30-day gate 改為每 `0.25 day` readback；若以下任一條件發生則提早 FAIL：

- invalid / NaN / non-positive density or pressure；
- `|dry-mass drift| > 5e-5`；
- `max |w| >= 10 m/s` numerical-stability guard。

最終 circulation gates（westerly / tropical easterly / overturning / opposite hemispheric signs）維持原本物理方向要求。

## 狀態 / Status

修正已提交，等待同一真實 WebGPU 裝置重新執行：

1. `npm test`
2. Stage 4 rotating + top-sponge smoke
3. GPU/CPU agreement
4. 30-day Held–Suarez development gate

在新的 30-day gate 通過以前，Stage 4 仍未封關。
