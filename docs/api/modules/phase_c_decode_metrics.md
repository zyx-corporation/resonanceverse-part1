# Phase 3 M1: デコード計測 JSON スキーマ（`decode_benchmark.v1`）

[`experiments/decode_benchmark.py`](../../../experiments/decode_benchmark.py) が出力する **`decode_benchmark_ok`** 行の JSON オブジェクト仕様。

## 共通フィールド

| フィールド | 型 | 意味 |
|------------|-----|------|
| `schema_version` | 文字列 | 固定: `decode_benchmark.v1` |
| `variant` | 文字列 | `baseline` または `two_tier_stub` |
| `demo` | 真偽 | `true` のとき `_demo_linear` 経路（transformers 不要） |
| `model` | 文字列 | HF モデル名、または `_demo_linear` |
| `device` | 文字列 | `cpu` / `cuda:0` 等 |
| `max_new_tokens` | 整数 | 1 チェーンあたりの**デコード**ステップ数（プレフィル後） |
| `warmup_rounds` | 整数 | 計測に入る前の全体チェーン反復 |
| `repeats` | 整数 | 記録に使うチェーン反復 |
| `total_decode_steps` | 整数 | 集計したステップ数（通常 `max_new_tokens * repeats`） |
| `latency_ms_p50` / `latency_ms_p95` | 数値 | ステップごとの経過時間（秒）から算出したミリ秒パーセンタイル |
| `per_step_time_mean_s` | 数値 | ステップ時間の算術平均 |
| `hbm_proxy_cuda_peak_bytes` | 整数または null | CUDA 時 `max_memory_allocated`、CPU 時 null（**HBM の厳密分解は M4**） |

## `two_tier_stub` 追加

| フィールド | 型 | 意味 |
|------------|-----|------|
| `router_keep_fraction_mean` | 数値 | 各ステップの「展開」マスク True 割合の平均（スタブ挙動のログ） |
| `router_stub_mode` | 文字列 | `priority_threshold`（既定）または `step_stride`（`--router-step-stride` 指定時） |
| `router_step_stride` | 整数 | `router_stub_mode` が `step_stride` のときの **N**（ステップ番号 mod N==0 で keep） |

## 関連

- [Phase 3 計画](../../planning/Phase3_計画_二階建てと実証.md)  
- [`two_tier_sweep` スキーマ](phase_c_two_tier_sweep.md)（M3）

---
*v1.1 — 2026-03-25（Router stride モード・P3）*
