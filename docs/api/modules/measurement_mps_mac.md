# Mac GPU（MPS）計測の考察

## 目的

Apple Silicon 上の **Metal Performance Shaders（MPS）** バックエンドで、Phase 3 のデコード・HBM 系ベンチを **CPU ではなくオンデバイス GPU** で実行し、レイテンシ等を測る。

## PyTorch における位置づけ

| バックエンド | 選択条件（本リポジトリ） |
|--------------|-------------------------|
| **CUDA** | `torch.cuda.is_available()`（NVIDIA・Linux/Windows 等） |
| **MPS** | CUDA が無く、`torch.backends.mps.is_available()`（macOS 13+・Apple GPU） |
| **CPU** | 上記いずれも不可、または `--cpu` 強制 |

実装は [`core/inference_device.py`](../../../core/inference_device.py) の `select_inference_device` に集約する。

## 計測上の注意（壁時計）

MPS は CUDA と同様、**非同期実行**になり得る。ステップ時間だけを取る場合、**各ステップの前後で `torch.mps.synchronize()`** しないと、計測が短く見積もられる。`decode_benchmark` のデコードループでは `sync_inference_device` を挟む。

## メモリ指標の限界

- **CUDA**: `torch.cuda.max_memory_allocated` でピークを JSON に載せられる（`hbm_proxy_cuda_peak_bytes`）。
- **MPS**: 現行 PyTorch では **CUDA と同等のピークバイト API がない**ことが多い。JSON では **ピーク列は null のまま**とし、**レイテンシと HBM フック集計（活性化バイト推定）**を主とする。
- **HBM テンプレ**（`hbm_budget_probe`）は forward フックでテンソルサイズを足すため、**MPS 上でも相対比較には使える**（絶対値は演算融合・実装依存）。

## 演算・精度

一部演算が MPS で未実装の場合、PyTorch が CPU にフォールバックすることがある。そのとき **「GPU 専用」とは限らない**。結果の解釈では `device` ログと再現環境（OS・PyTorch 版）を併記する。

## 運用上の推奨

1. 論文・表用の **再現コマンド**に `python -c "import torch; print(torch.backends.mps.is_available())"` の結果を脚注する。
2. **CUDA マシンとの数値は直接比較しない**（ハード・ドライバが異なる）。同一 Mac 上での baseline / two_tier **相対比**を優先する（[Phase 3 計画 §5](../../planning/Phase3_計画_二階建てと実証.md)）。

## 関連コード

- `select_inference_device` / `sync_inference_device` — `core/inference_device.py`
- デコード計測 — `experiments/decode_benchmark.py`
- スイープ／主張バンドル — `experiments/two_tier_sweep.py`、`experiments/phase3_claim_run.py`
- HBM — `experiments/hbm_budget_probe.py`

## 改訂履歴

| 日付 | 内容 |
|------|------|
| 2026-03-25 | 初版（MPS 選択・同期・メモリ指標の限界） |
