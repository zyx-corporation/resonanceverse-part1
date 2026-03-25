# Resonanceverse ロードマップ（和文・実装チェックリスト）

本稿はリポジトリ上の**フェーズ対応とチェックリスト**をまとめたものです（**Phase 1A〜3**。**Phase 4**＝分散・エッジは [独立ロードマップ](docs/planning/ROADMAP_Phase4_分散とエッジ_ja.md)）。
理論・判定表の詳細は [実証ロードマップ（軽量コアと SLM／二階建て）](
docs/planning/Resonanceverse実証ロードマップ_軽量コアとSLM二階建て.md) を参照してください。

## README 簡易フェーズと実証ロードマップの対応

| 本稿（ROADMAP_ja） | 実証ロードマップ（詳細） |
|--------------------|--------------------------|
| Phase 1A（下記・完了分） | **Phase A** ＋ **Phase A′**（軽量コア・索引・CI 回帰） |
| Phase 1B（下記・橋渡し） | 文化的調製の最小実装・SLM 橋スモーク → ロードマップ **Phase B** へ接続 |
| Phase 2–3（SLM／二階建て） | **Phase 2**＝SLM 入口・下流。 **Phase 3**＝[二階建て・資源実証の計画](docs/planning/Phase3_計画_二階建てと実証.md)（実証ロードマップ **Phase C** 相当） |
| **Phase 4**（分散・エッジ） | **別ロードマップ** — [ROADMAP_Phase4_分散とエッジ_ja.md](docs/planning/ROADMAP_Phase4_分散とエッジ_ja.md)（実証ロードマップの [Phase 4 節](docs/planning/Resonanceverse実証ロードマップ_軽量コアとSLM二階建て.md) と対応。Phase A〜C とは別軸） |

長期の研究協働・透明性は、技術実証ロードマップとは別文書の
[透明性・協働ロードマップ](docs/planning/
🚀%20透明性確保による人間-AI協働研究発展ロードマップ.md)
を参照してください。

---

## Phase 1A: 軽量実装実証（≈ Phase A ＋ 並行 A′）— 完了扱い

- [x] 基本共鳴エンジン・ROI・ファサード
- [x] 創発指標・効率ベンチのスクリプト
- [x] `docs/api/modules` におけるモジュール↔ベンチ対応表・再現手順の整備（Phase A′）—
[対応表](docs/api/modules/module_benchmark_map.md)・[再現手順](docs/api/modules/phase_a_reproduction.md)

---

## Phase 1B: 文化的調製と SLM 橋渡し（ロードマップ Phase B への接続）

- [x] 文化的調製（埋め込み→スカラー／`cultural_scale`）— `core/cultural_modulation.py`
- [x] `LightweightResonanceFacade` への `cultural_scale` 連携（Phase 1B）
- [x] `AwaiIntegratedSLM` への `CulturalModulationAdapter` 接続（`cultural_modulation=True`）— `core/resonant_core.py`
- [x] SLM 橋スモーク（`--cultural-modulation` 任意）—
  [`experiments/slm_bridge_smoke.py`](experiments/slm_bridge_smoke.py)；
  最小学習は [`experiments/slm_resonance_lm.py`](experiments/slm_resonance_lm.py) の `--cultural-modulation`
- [x] 仕様メモ — [phase_1b_cultural_slm.md](docs/api/modules/phase_1b_cultural_slm.md)

---

## Phase 2–3（SLM／二階建て）

技術実証の段階分け・判定表は [実証ロードマップ（軽量コアと SLM／二階建て）](
docs/planning/Resonanceverse実証ロードマップ_軽量コアとSLM二階建て.md)
を参照してください。

### Phase 2（入口・進行中）

- [x] `AwaiIntegratedSLM` の最小学習ループ（次トークン CE）—
[`experiments/slm_resonance_lm.py`](experiments/slm_resonance_lm.py)（`--demo` でオフライン／CI、`--model gpt2` で実モデル）
- [x] 固定データ経路（`--data wikitext`）・前処理・**perplexity**（`--eval-ppl`）—
[`experiments/slm_data.py`](experiments/slm_data.py) ＋ [Phase B データプロトコル](docs/api/modules/phase_b_data_protocol.md)
- [x] 下流タスクの固定プロトコル（入口）— [phase_b_downstream_protocol.md](docs/api/modules/phase_b_downstream_protocol.md)、[`experiments/slm_downstream.py`](experiments/slm_downstream.py)（SST-2／BoolQ／`--demo`）  
- [x] 抽出型 QA（SQuAD v1 スパン）— [`experiments/squad_span.py`](experiments/squad_span.py)（`--demo` で CI／オフライン、[phase_b_downstream_protocol.md](docs/api/modules/phase_b_downstream_protocol.md) v0.3）

### Phase 3（二階建て・資源実証／ロードマップ Phase C 相当）

詳細計画は **[Phase3 計画（二階建てと実証）](docs/planning/Phase3_計画_二階建てと実証.md)** を参照（マイルストーン M1〜M5、スコープ・非スコープ）。**§2 の「表」を実測で埋める手順**は **[Phase3 実測と主張完成計画](docs/planning/Phase3_実測と主張完成計画_ja.md)**（採用済み）。

- [x] **M1** 計測ハーネス — [`experiments/decode_benchmark.py`](experiments/decode_benchmark.py)、[phase_c_decode_metrics.md](docs/api/modules/phase_c_decode_metrics.md)（`--demo` で CI 可）
- [x] **M2** 二階建てスケルトン — [`core/two_tier/`](core/two_tier/)（`SequenceControllerStub`／`BlockRouterStub`／`check_quality_tau`）
- [x] **M3** 同一条件スイープ — [`experiments/two_tier_sweep.py`](experiments/two_tier_sweep.py)、[phase_c_two_tier_sweep.md](docs/api/modules/phase_c_two_tier_sweep.md)
- [x] **M4** HBM バイト予算表テンプレートのログ充填 — [`experiments/hbm_budget_probe.py`](experiments/hbm_budget_probe.py)、[phase_c_hbm_budget.md](docs/api/modules/phase_c_hbm_budget.md)（`--demo` で CI 可）
- [x] **M5（任意）** 抽出型 QA の下流プロトコル拡張 — [`experiments/squad_span.py`](experiments/squad_span.py)、[phase_b_downstream_protocol.md](docs/api/modules/phase_b_downstream_protocol.md)

---

## 関連リンク

| 文書 | 内容 |
|------|------|
| [実証ロードマップ（詳細・フェーズ定義）](docs/planning/Resonanceverse実証ロードマップ_軽量コアとSLM二階建て.md) | Phase A〜C の主張・成果物・関連文書 |
| [Phase 4 分散・エッジ（和文・独立）](docs/planning/ROADMAP_Phase4_分散とエッジ_ja.md) | Jetson 等・チェックリスト（本稿の Phase 1A〜3 とは別軸） |
| [Phase3 計画（二階建てと実証）](docs/planning/Phase3_計画_二階建てと実証.md) | Phase 3 の目的・マイルストーン・スコープ（Phase C 相当） |
| [docs/README.md](docs/README.md) | 理論・実装・計画の索引 |
| [README.md](README.md) | プロジェクト概要・インストール・クイックスタート |

---

*README のロードマップ節は本ファイルへ分離。更新は本稿と README のリンク整合を保つこと。*
