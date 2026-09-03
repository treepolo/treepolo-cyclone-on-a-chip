# UI Specification — 中英雙語同時顯示 / Simultaneous Chinese + English UI

## 硬性要求 / Hard requirement

Cyclone on a Chip 的所有使用者介面文字必須**同時顯示繁體中文與英文**。不使用「中文 / English」語言切換作為主要介面模式，也不能只在部分頁面翻譯。

All user-facing UI text in Cyclone on a Chip must show **Traditional Chinese and English simultaneously**. A Chinese/English language toggle is not the primary presentation model, and translation must not be limited to selected screens.

## 顯示順序 / Presentation order

預設使用：`繁體中文 / English`。

Default format: `Traditional Chinese / English`.

例如 / Examples:

- `重設 / Reset`
- `熱泡 / Thermal bubble`
- `診斷 / Diagnostics`
- `乾空氣質量漂移 / Dry mass drift`
- `計算 smoke 測試通過 / Compute smoke OK`

## 適用範圍 / Scope

必須雙語的內容包括：按鈕、標題、欄位名稱、狀態、警告、錯誤、紀錄訊息、說明文字、設定、工具提示、互動工具名稱與後續正式介面的所有功能名稱。

Bilingual presentation applies to buttons, headings, field labels, statuses, warnings, errors, log messages, explanatory text, settings, tooltips, interaction tools, and all future production UI feature names.

物理符號、SI 單位、數值與通用縮寫（例如 `ρ`, `p`, `m/s`, `Pa`, `WebGPU`, `CFL`）不需要重複翻譯，但其文字標籤仍需雙語。

Physical symbols, SI units, numbers, and universal abbreviations such as `ρ`, `p`, `m/s`, `Pa`, `WebGPU`, and `CFL` do not need duplication, while their textual labels still require both languages.
