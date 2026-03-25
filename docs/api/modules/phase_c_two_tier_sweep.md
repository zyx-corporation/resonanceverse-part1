# Phase 3 M3: `two_tier_sweep` マージ JSON（`two_tier_sweep.v1`）

[`experiments/two_tier_sweep.py`](../../../experiments/two_tier_sweep.py) が出力する **`two_tier_sweep_ok`** 行の JSON。

## 構造

| フィールド | 型 | 意味 |
|------------|-----|------|
| `schema_version` | 文字列 | `two_tier_sweep.v1` |
| `baseline` | オブジェクト | [decode_benchmark.v1](phase_c_decode_metrics.md)（`variant: baseline`） |
| `two_tier_stub` | オブジェクト | 同上（`variant: two_tier_stub`） |
| `comparison.latency_ms_p50_ratio_baseline_over_two_tier` | 数値または null | baseline の p50 / two_tier の p50。**1 より大きい**と two_tier 側の p50 が短い |

スタブ段階では Router オーバーヘッドで **two_tier が遅くなる**こともあり得る。解釈は [計測戦略の考察](measurement_strategy.md) に準拠し、**同一マシン・同一 seed** での相対比較を主とする。

---
*v1 — 2026-03-25*
