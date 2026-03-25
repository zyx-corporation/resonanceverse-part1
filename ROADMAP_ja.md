# Resonanceverse ロードマップ（和文・実装チェックリスト）

本稿はリポジトリ上の**フェーズ対応とチェックリスト**をまとめたものです。理論・判定表の詳細は [実証ロードマップ（軽量コアと SLM／二階建て）](docs/planning/Resonanceverse実証ロードマップ_軽量コアとSLM二階建て.md) を参照してください。

## README 簡易フェーズと実証ロードマップの対応

| 本稿（ROADMAP_ja） | 実証ロードマップ（詳細） |
|--------------------|--------------------------|
| Phase 1A（下記・完了分） | **Phase A** ＋ **Phase A′**（軽量コア・索引・CI 回帰） |
| Phase 1B（下記・橋渡し） | 文化的調製の最小実装・SLM 橋スモーク → ロードマップ **Phase B** へ接続 |
| Phase 2–3（SLM／二階建て） | ロードマップ **Phase B**（SLM アダプタ本格）→ **Phase C**（二階建て・HBM 等） |

長期の研究協働・透明性は、技術実証ロードマップとは別文書の [透明性・協働ロードマップ](docs/planning/🚀%20透明性確保による人間-AI協働研究発展ロードマップ.md) を参照してください。

---

## Phase 1A: 軽量実装実証（≈ Phase A ＋ 並行 A′）— 完了扱い

- [x] 基本共鳴エンジン・ROI・ファサード
- [x] 創発指標・効率ベンチのスクリプト
- [x] `docs/api/modules` におけるモジュール↔ベンチ対応表・再現手順の整備（Phase A′）— [対応表](docs/api/modules/module_benchmark_map.md)・[再現手順](docs/api/modules/phase_a_reproduction.md)

---

## Phase 1B: 文化的調製と SLM 橋渡し（ロードマップ Phase B への接続）

- [x] 文化的調製（埋め込み→スカラー／`cultural_scale`）— `core/cultural_modulation.py`
- [x] `LightweightResonanceFacade` への `cultural_scale` 連携（Phase 1B）
- [x] SLM 橋スモーク（`AwaiIntegratedSLM`）— [`experiments/slm_bridge_smoke.py`](experiments/slm_bridge_smoke.py)（`transformers` 利用時）
- [ ] 大規模ノード・分散（Jetson 等）— 設計文書参照（Phase C 周辺または別プロジェクト）

---

## Phase 2–3（SLM／二階建て）

技術実証の段階分け・判定表は [実証ロードマップ（軽量コアと SLM／二階建て）](docs/planning/Resonanceverse実証ロードマップ_軽量コアとSLM二階建て.md) を参照してください。

### Phase 2（入口・進行中）

- [x] `AwaiIntegratedSLM` の最小学習ループ（次トークン CE）— [`experiments/slm_resonance_lm.py`](experiments/slm_resonance_lm.py)（`--demo` でオフライン／CI、`--model gpt2` で実モデル）
- [x] 固定データ経路（`--data wikitext`）・前処理・**perplexity**（`--eval-ppl`）— [`experiments/slm_data.py`](experiments/slm_data.py) ＋ [Phase B データプロトコル](docs/api/modules/phase_b_data_protocol.md)
- [ ] 下流タスク（分類・QA 等）の固定プロトコル
- [ ] 二階建て・HBM 等は Phase 3（ロードマップ Phase C）へ

---

## 関連リンク

| 文書 | 内容 |
|------|------|
| [実証ロードマップ（詳細・フェーズ定義）](docs/planning/Resonanceverse実証ロードマップ_軽量コアとSLM二階建て.md) | Phase A〜C の主張・成果物・関連ドキュメント |
| [docs/README.md](docs/README.md) | 理論・実装・計画の索引 |
| [README.md](README.md) | プロジェクト概要・インストール・クイックスタート |

---

*README のロードマップ節は本ファイルへ分離。更新は本稿と README のリンク整合を保つこと。*
