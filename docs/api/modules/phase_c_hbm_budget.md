# Phase C HBM バイト予算表（M4）

## 目的

二階建て計画 §7.2 の **CSV テンプレ行**に沿って、前向き 1 回あたりの **活性化バイト推定**をログに埋める。絶対精度より **baseline／two_tier の相対比較**を想定する。

## 実装

| 項目 | 内容 |
|------|------|
| スクリプト | [`experiments/hbm_budget_probe.py`](../../../experiments/hbm_budget_probe.py) |
| オフライン／CI | `--demo`（`transformers` 不要の `_DemoLM`） |
| 実モデル | `--model gpt2`（既定）— `AutoModelForCausalLM`、初回は重み取得あり |

## スキーマ（`hbm_budget.v1`）

| フィールド | 型 | 説明 |
|------------|-----|------|
| `schema_version` | str | 固定 `hbm_budget.v1` |
| `variant` | str | デモ時 `demo_stub`、実モデル時はモデル名 |
| `demo` | bool | デモモードか |
| `device` | str | 実行デバイス |
| `batch_size` | int | バッチサイズ |
| `seq_len` | int | 系列長 |
| `rows` | list | テンプレ順の行（`TEMPLATE_ORDER` と同順） |
| `rows[].layer` | str | 例: `Embedding`, `Attention_QKV`, … |
| `rows[].act_b` | int \| null | 当該バケットに集計した forward 出力バイト合算（未該当は null） |
| `rows[].fwd_io_b` 等 | null | 予約（現状未使用） |
| `total_act_bytes_estimated` | int | `act_b` の合算（null 行は除く） |
| `disclaimer` | str | 重複計上・融合未分離などの限界 |

**GPT-2 注意**: `Attention_Score` / `Attention_Softmax` はアテンション内で融合しているため、非デモでは **null** になりうる（`notes` に記載）。

## 手順（固定）

1. `--seed` で `set_experiment_seed` を適用。  
2. 対象モジュールに forward hook を登録し、出力テンソルごとに `numel * element_size` をバケットに加算。  
3. 標準出力先頭に `hbm_budget_probe_ok`、続けて JSON 1 行。

## 関連

- [モジュール ↔ 実証スクリプト対応表](module_benchmark_map.md)  
- [Phase3 計画（二階建てと実証）](../../planning/Phase3_計画_二階建てと実証.md)

---
*v0.1 — 2026-03-25*
