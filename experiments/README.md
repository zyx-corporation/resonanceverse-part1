# `experiments/` — v7 実験設計との対応

**正本**: [v7.0 実証実験設計書](../docs/v7/Resonanceverse_v7.0_Experimental_Design.md) · [EXPERIMENT_ROADMAP_v7.md](../docs/planning/EXPERIMENT_ROADMAP_v7.md)

## フェーズ別の入口（現行スクリプト）

| v7 | 主なスクリプト |
|----|----------------|
| **I（概念実証）** | `evel_benchmarks.py`（軽量コア）、`efficiency_benchmark.py`（計算量）、`emergence_metrics.py`（創発） |
| **II（安定性・遅延）** | *将来*: 遅延 τ スイープ・V_K ログ用ハーネスを追加予定 |
| **III（あわい・朧）** | 理論指標は v7 定義。**人間評価**は外部コーパス・手順が必要 |
| **IV（Transformer）** | `slm_resonance_lm.py`、`slm_downstream.py`、`decode_benchmark.py`、`two_tier_sweep.py`、`phase3_claim_run.py` |

## レガシー Phase B/C の記録

- 下流・τ: `slm_downstream.py`、`squad_span.py`
- 資源・デコード: `hbm_budget_probe.py`、`phase3_claim_run.py`
- ベースライン JSON: `experiments/baselines/*.json`

ログの既定出力先は `experiments/logs/`（`.gitignore` 対象）が多い。
