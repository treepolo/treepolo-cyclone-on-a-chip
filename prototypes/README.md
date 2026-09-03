# Stage 2 prototypes

## P2 + P3 Float64 reference

需要 Python 3 與 NumPy：

```bash
python prototypes/stage2_reference.py
```

預設跑 P2 的 `N=12,24` 與 P3 的 acoustic CFL `0.25,1,3,10`。要加跑 P2 `N=48`：

```bash
python prototypes/stage2_reference.py --full
```

輸出為 JSON，可直接和 `docs/PROTOTYPE_RESULTS.md` 比較。

## P1 WebGPU benchmark

WebGPU 需要 secure context；本機請從 Repo 根目錄啟動 HTTP server，例如：

```bash
python -m http.server 8000
```

再開啟 `http://localhost:8000/prototypes/p1_webgpu.html`，按 **Run P1**。結果會列出裝置 WebGPU limits、測試網格大小、耗時與 `cellUpdatesPerSecond`。

P1 只用來量裝置的基本 compute/data-path 吞吐量，不代表完整大氣 solver 的最終效能。