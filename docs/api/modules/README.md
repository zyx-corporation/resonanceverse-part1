# API・モジュール設計リファレンス

自動生成の Sphinx 等は未整備のため、本ディレクトリでは設計仕様と `core/` 実装の対応をまとめます。

- [アルゴリズム仕様（文章表現による設計）](algorithm_spec.md)
- [アーキテクチャ概要（モジュール構造・クラス設計）](../../implementation/architecture_overview.md)
- 実装コード: リポジトリ直下の [`core/`](../../../core/)

## 共鳴コンポーネントの対応（理論・実装の索引）

「共鳴」という語が **複数クラス**に付いている。混同しないための対応である。

| クラス（`core/`） | 役割の要約 |
|-------------------|------------|
| **`resonance.ResonanceEngine`** | ノード部分集合×文脈から共鳴スコアを出す。**学習可能な** `W`（N×N×d 系）。`LightweightResonanceFacade`・`evel_benchmarks` 等。 |
| **`resonant_core.ResonantCore`** | 隠れ状態を 6 次元に射影し softmax、**場バッファ** `W`（1×N×d）をインプレース更新。ドリフトは専用 `Generator`。**`AwaiIntegratedSLM`・`slm_downstream` の awai** 経路。 |
| **`resonant_core.AwaiIntegratedSLM`** | HF 因果 LM の `hidden_states` に `ResonantCore` を重ね、**H+6 連結**で語彙ロジット（`slm_resonance_lm`）。 |

数式上の「共鳴テンソル」との 1:1 対応は置かず、上表を**実装の正**とする。

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
| [Phase C 品質 τ 事前登録（P1）](phase_c_quality_tau_prereg.md) | v0.1 閾値・v0.2〜v0.4・**AwaiIntegratedSLM 語彙ヘッド H+6** 注記・[Phase3 計画 §5.1](../../planning/Phase3_計画_二階建てと実証.md) |
| [Phase 3 P0 主張バンドル（スナップショット手順）](phase3_p0_baseline_snapshot.md) | `phase3_claim_run`・メタ JSON・表の埋め方・評価・**P2／P3（任意）**・τ との役割分担 |
| [主張表（論文・対外向け）](../planning/Resonanceverse_主張表_論文用_ja.md) | 査読用の主張一覧・要旨・本文ドラフト（τ・資源・Phase 4・再現性） |
| [Mac GPU（MPS）計測の考察](measurement_mps_mac.md) | デバイス選択・同期・メモリ指標の限界・運用上の注意 |
| [Phase C GPT-2 層スキップ実測](phase_c_gpt2_layer_skip.md) | `causal_lm_layer_skip_benchmark.v1`・恒等ブロック置換 |
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
