# Phase 3 補足: 因果 LM 層恒等スキップ実測（`causal_lm_layer_skip_benchmark.v1`）

[`experiments/gpt2_layer_skip_benchmark.py`](../../../experiments/gpt2_layer_skip_benchmark.py) は、[`AutoModelForCausalLM`](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForCausalLM) で読み込んだモデルのデコーダ層列（[`causal_lm_layers.get_decoder_module_list`](../../../core/two_tier/causal_lm_layers.py) で解決）の一部を [`IdentityTransformerBlock`](../../../core/two_tier/gpt2_identity_skip.py)（別名 `IdentityGPT2Block`）に**一時置換**し、**フル系列・`use_cache=False`** の 1 回 forward の時間差を測る。`--demo` のみ従来どおり小さな `GPT2LMHeadModel`。

## 解釈

- **数学的**には元の LM と一致しない（恒等は残差ブロックの省略）。
- **実装目的**: 層計算を実際に省いたときの **壁時計・CUDA ピーク**の傾向を、baseline（全層本来）と比較する。

## スキーマ（主要フィールド）

| フィールド | 意味 |
|------------|------|
| `decoder_stack_kind` | 層列の取り出し経路（例: `transformer.h`、`model.layers`） |
| `model_class` | `type(model).__name__` |
| `layer_execute_mask` | 各層を実行するか（True=本来のブロック） |
| `forward_ms_mean_full` / `forward_ms_mean_skip` | 全層 vs マスク適用時の平均 forward 時間 |
| `time_ratio_full_over_skip` | 全層時間 ÷ スキップ後時間（>1 ならスキップ側が速い） |

---
*v0.1 — 2026-03-25*
