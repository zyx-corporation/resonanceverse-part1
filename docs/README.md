# Resonanceverse 文書インデックス

リポジトリ内の Markdown を用途別に配置しています。エントリは [ルート README](../README.md) の「ドキュメント」からも辿れます。

## 理論・実験の正本（v7.0）

| 文書 | 内容 |
|------|------|
| [Resonanceverse 理論 v7.0](v7/Resonanceverse_Theory_v7.0.md) | 有向遅延共鳴場（DDRF）、定理群、「あわい」、朧、Transformer 方式 B |
| [実証実験設計 v7.0](v7/Resonanceverse_v7.0_Experimental_Design.md) | Phase I〜IV、判定基準、スケジュール |
| [v7 ディレクトリ索引](v7/README.md) | 上記への入口 |
| [理論索引（実装との対応）](theory/resonanceverse_theory.md) | v7 と `core/` の対応・旧版リンク |
| [EXPERIMENT_ROADMAP_v7](planning/EXPERIMENT_ROADMAP_v7.md) | v7 フェーズと `experiments/` の対応 |
| [Phase I-A / III-A 本番手順設計（v7）](planning/Phase_IA_IIIA_本番手順設計_v7.md) | データ・アノテ・解析・判定の設計チェックリスト |

- **実装フェーズ・チェックリスト（和文・レガシー）**: [ROADMAP_ja.md](../ROADMAP_ja.md)（ルート）

## 実装・文書の現状（スナップショット）

2026 年 3 月時点のリポジトリ状態の要約です。詳細は各リンク先の本文を正とする。

- **理論の正本**は **v7.0**（`docs/v7/`）。旧来の長文・統合稿は `docs/theory/` および `docs/archive/` に**参考**として残す。数式・主張の衝突時は v7 を優先する。
- **Phase 3（主張・実測・レガシー）**: [Phase3 実測と主張完成計画](planning/Phase3_実測と主張完成計画_ja.md)・[主張表（論文用）](planning/Resonanceverse_主張表_論文用_ja.md)、`experiments/phase3_claim_run.py` と `experiments/baselines/`。
- **`ResonantCore`**: 場ドリフトの `Generator`・`field_drift_seed`、`eval` で場固定、`attention_mask` による除外平均。
- **共鳴コンポーネント**: [API・モジュール索引の「共鳴コンポーネントの対応」](api/modules/README.md) を実装上の正とする。
- **品質ゲート**: `tests/` の `pytest`。分散同期スモークのテンソル経路は numpy ペイロード。
- **v7 実験ハーネス**: `experiments/v7_run_suite.py`（`--demo` で Phase I-A/I-B/II-A/III 合成を一括）、[EXPERIMENT_ROADMAP_v7](planning/EXPERIMENT_ROADMAP_v7.md) を参照。JSONL パイロットは `experiments/v7_phase1a_pilot_jsonl.py` と [`data/v7_phase1a_pilot.jsonl`](data/v7_phase1a_pilot.jsonl)。**実証ベースライン一括**は `experiments/v7_empirical_run.py` と [事前登録スタブ v1](planning/v7_phase1a_empirical_prereg_v1.json)。**本番コーパス（Phase I-A）は MRMP のみ** — [v7_corpus_MRMP.md](planning/v7_corpus_MRMP.md)。整形 [`v7_mrmp_prepare.py`](../experiments/v7_mrmp_prepare.py)、サンプル [`v7_mrmp_sample.jsonl`](../experiments/data/v7_mrmp_sample.jsonl)。**6 軸 LLM 審判** [`v7_phase1a_llm_judge_six_axes.py`](../experiments/v7_phase1a_llm_judge_six_axes.py)（`--demo` 可）。**Phase II-A 成果物の検証**は [`v7_phase2a_bundle_validate.py`](../experiments/v7_phase2a_bundle_validate.py)（`--strict` は再現ログ生成後向け；CI では非 strict）。**コード指紋**は [`v7_phase2a_repro_manifest.py`](../experiments/v7_phase2a_repro_manifest.py) の `--pin-code-only`。**理論との橋**（R(τ) と τ*）は [v7_phase2a_theory_bridge.md](planning/v7_phase2a_theory_bridge.md)。

## ディレクトリ構成

| ディレクトリ | 内容 |
|--------------|------|
| [`v7/`](v7/README.md) | **理論・実験設計の正本（v7.0）** |
| [`api/modules/`](api/modules/README.md) | アルゴリズム仕様・モジュール索引・Phase A/A′ 再現・計測戦略 |
| [`implementation/`](implementation/) | モジュール構造・二階建て計画・創発設計・LLM 選定・RCL 等 |
| [`planning/`](planning/) | 実証ロードマップ・Phase3・主張表・**EXPERIMENT_ROADMAP_v7** |
| [`theory/`](theory/) | 理論索引（v7 入口）・数学的基礎・長文参考稿 |
| [`archive/`](archive/) | 旧版の保管（例: 統合長文理論） |
| [`tutorials/`](tutorials/getting_started.md) | クイックスタート |
| [`experiments/`](../experiments/README.md) | 実験スクリプト索引（v7 対応表） |

## 計画・ロードマップ（`planning/`）

| 文書 | 概要 |
|------|------|
| [EXPERIMENT_ROADMAP_v7](planning/EXPERIMENT_ROADMAP_v7.md) | **v7 Phase I〜IV** と既存スクリプトの対応 |
| [v7 Phase I-A コーパス — MRMP のみ](planning/v7_corpus_MRMP.md) | MRMP のみ（HF・ライセンス・Phase I-A 対応） |
| [ROADMAP_ja.md（ルート）](../ROADMAP_ja.md) | 和文チェックリスト（Phase 1A〜3）· v7 正本へのリンク |
| [ROADMAP Phase 4（分散・エッジ・和文）](planning/ROADMAP_Phase4_分散とエッジ_ja.md) | Jetson 等・独立ロードマップ |
| [実証ロードマップ（軽量コアと SLM／二階建て）](planning/Resonanceverse実証ロードマップ_軽量コアとSLM二階建て.md) | Phase A〜C（レガシー実証のハブ） |
| [Phase3 計画（二階建てと実証）](planning/Phase3_計画_二階建てと実証.md) | Phase 3 マイルストーン |
| [Phase3 実測と主張完成計画（採用済み）](planning/Phase3_実測と主張完成計画_ja.md) | P0〜P3 |
| [Resonanceverse 主張表（論文・対外向け）](planning/Resonanceverse_主張表_論文用_ja.md) | 主張一覧 |
| [Phase II-A と理論の「橋」](planning/v7_phase2a_theory_bridge.md) | コーパス R(τ) と τ*／τ*_exp の区別・追加実験 |
| [Phase II-A 数値 τ 掃引（合成）](planning/v7_phase2a_numeric_tau_exp.md) | delay_sweep・alpha スイープと設計書 II-A の位置づけ |
| [Phase IV 最小再現（方式 B 周辺）](planning/v7_phase4_integration_repro.md) | decode / two-tier スタブと本番 Phase IV の差 |
| [日本語ローカル SLM 運用プラン（M3 Max / 128GB）](planning/v7_local_slm_m3_japanese_plan.md) | Swallow / rinna・`hf_local`・チャンク・**SLM 同士審判一致**（§10） |

## 理論（`theory/`）

| 文書 | 概要 |
|------|------|
| [理論索引（v7 正本・実装対応）](theory/resonanceverse_theory.md) | **入口** |
| [数学的基礎（証明の完全版・先行版）](theory/mathematical_foundation.md) | v7 への参照あり |
| [動的共鳴に基づく自己創生的AIの数理基盤と実装](theory/Resonanceverse：動的共鳴に基づく自己創生的AIの数理基盤と実装.md) | 長文・参考 |
| [Complete Mathematical Framework…（英語）](theory/Resonanceverse：Complete%20Mathematical%20Framework%20and%20Implementation%20for%20Autopoietic%20AI%20Through%20Dynamic%20Resonance.md) | 英語版・参考 |
| [理論的基盤と課題（部内限定版）](theory/Resonanceverse：動的共鳴に基づく自己創生的AIの理論的基盤と課題【機密%20-%20部内限定】ver.%206.0.md) | 機密ラベル付き |
| [認知の戦略的制約とおぼろ性：情報理論的考察](theory/認知の戦略的制約とおぼろ性：情報理論的考察.md) | おぼろ性・情報理論 |
| [archive: 旧・統合長文「完全理論と実装」](archive/theory_resonanceverse_legacy_integrated.md) | v7 以前の統合稿（参照用） |

## 実装設計（`implementation/`）

| 文書 | 概要 |
|------|------|
| [実装基本設計（モジュール構造とクラス設計）](implementation/architecture_overview.md) | 全体モジュール構成 |
| [二階建てアーキテクチャ実装計画 v0.1](implementation/Resonanceverse：2階建てアーキテクチャ実装計画%20v0.1.md) | Phase C 周辺 |
| [共鳴場による創発の観測・利用 設計 v0.1](implementation/🛠️%20共鳴場による創発の観測・利用%20設計%20v0.1.md) | 創発指標 |
| [共鳴制御（RCL）拡張設計 v0.1](implementation/🛠️%20Resonanceverse%20実装論：共鳴制御（RCL）拡張設計%20v0.1.md) | 多主体・監査 |
| [LLM 選定戦略](implementation/🤖%20Resonanceverse実装におけるLLM選定戦略.md) | 下層モデル比較 |

## API・実証索引（`api/modules/`）

[API・モジュール設計リファレンス](api/modules/README.md) を起点に、再現手順・対応表・計測考察へリンクしています。

---

*配置ルール: 新規文書は上表のいずれかのディレクトリへ置き、本インデックスに 1 行追加すること。理論・実験計画の**正本**は `docs/v7/` に置く。*
