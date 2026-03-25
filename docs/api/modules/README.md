# API・モジュール設計リファレンス

自動生成の Sphinx 等は未整備のため、本ディレクトリでは設計仕様と `core/` 実装の対応をまとめます。

- [アルゴリズム仕様（文章表現による設計）](algorithm_spec.md)
- [アーキテクチャ概要（モジュール構造・クラス設計）](../../implementation/architecture_overview.md)
- 実装コード: リポジトリ直下の [`core/`](../../../core/)

## Phase A / A′（実証の索引）

| 文書 | 内容 |
|------|------|
| [モジュール ↔ 実証スクリプト対応表](module_benchmark_map.md) | `core` と `experiments`／`tests` の対応 |
| [Phase A 再現手順](phase_a_reproduction.md) | コマンド、`--seed`、結果の読み方、`regression_check` |
| [計測戦略の考察](measurement_strategy.md) | 指標の限界、ベースライン運用、Phase C との関係 |
| [Phase A′ の締めと CI 蓄積の考察](phase_a_prime_closure_and_ci_accumulation.md) | 完了基準、CI が蓄積する／しないもの、Artifacts 等の任意強化 |
| [Phase B データ・評価プロトコル](phase_b_data_protocol.md) | Wikitext／ランダム、perplexity 定義、再現性 |
| [Phase B 下流タスク・評価プロトコル](phase_b_downstream_protocol.md) | SST-2 分類入口、`slm_downstream.py`、baseline／awai 比較 |
| [Phase C デコード計測スキーマ（M1）](phase_c_decode_metrics.md) | `decode_benchmark.v1`、p50/p95、CUDA ピーク代理 |
| [Phase C two_tier_sweep（M3）](phase_c_two_tier_sweep.md) | baseline／two_tier_stub マージ JSON |
| [Phase C HBM バイト予算（M4）](phase_c_hbm_budget.md) | `hbm_budget.v1`、`hbm_budget_probe.py` |
| [Phase C 品質 τ 事前登録（P1）](phase_c_quality_tau_prereg.md) | 登録表・[Phase3 計画 §5.1](../../planning/Phase3_計画_二階建てと実証.md) |
| [Phase 1B: 文化的調製と SLM 橋](phase_1b_cultural_slm.md) | `CulturalModulationAdapter`、ファサード／`AwaiIntegratedSLM` 接続 |
| [SLM 導入前後のパフォーマンス比較](slm_performance_comparison.md) | `slm_resonance_lm` / `slm_perf_compare` の JSON 指標・記録テンプレート |

## 性能実証・ベンチとの対応（ロードマップ）

設計ドキュメントと実行可能な実証の**突き合わせ**は、
[実証ロードマップ（軽量コアと SLM／二階建て）](../planning/
Resonanceverse実証ロードマップ_軽量コアとSLM二階建て.md) の **Phase A′** として
位置づけています。含まれる開発の例は次のとおりです。

| 項目 | 内容 |
|------|------|
| 対応表 | 本 README・`algorithm_spec.md`・`architecture_overview.md` のモジュールと、`experiments/*.py` および将来の二階建てコードとの対応 |
| 計測 | `core/` への軽量計測フック（メモリ・時間）、ログ形式のテンプレート |
| Phase C | `HBM_bytes/token`、レイテンシ、baseline との比較スクリプト（二階建て実装計画に準拠） |

詳細は上記ロードマップの「Phase A′」「Phase C」を参照してください。
