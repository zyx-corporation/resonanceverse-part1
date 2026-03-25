# Resonanceverse 文書インデックス

リポジトリ内の Markdown を用途別に配置しています。エントリは [ルート README](../README.md) の「ドキュメント」からも辿れます。

- **実装フェーズ・チェックリスト（和文）**: [ROADMAP_ja.md](../ROADMAP_ja.md)（ルート）

## ディレクトリ構成

| ディレクトリ | 内容 |
|--------------|------|
| [`api/modules/`](api/modules/README.md) | アルゴリズム仕様・モジュール索引・Phase A/A′ 再現・計測戦略・[Phase 1B](api/modules/phase_1b_cultural_slm.md)・[Phase B LM](api/modules/phase_b_data_protocol.md)／[下流](api/modules/phase_b_downstream_protocol.md) |
| [`implementation/`](implementation/) | モジュール構造・二階建て計画・創発設計・LLM 選定・RCL 等の実装設計 |
| [`planning/`](planning/) | 実証ロードマップ・透明性ロードマップ・2 週間プロトタイプ原稿（⚡） |
| [`theory/`](theory/) | 理論整理・数学的基礎・長文の数理／英語フレームワーク・認知とおぼろ性 |
| [`tutorials/`](tutorials/getting_started.md) | クイックスタート・2 週間計画（`getting_started.md`） |

## 計画・ロードマップ（`planning/`）

| 文書 | 概要 |
|------|------|
| [ROADMAP_ja.md（ルート）](../ROADMAP_ja.md) | README から分離した和文チェックリスト・フェーズ対応表（Phase 1A〜3） |
| [ROADMAP Phase 4（分散・エッジ・和文）](planning/ROADMAP_Phase4_分散とエッジ_ja.md) | Jetson 等・**独立ロードマップ**（Phase A〜C とは別軸） |
| [実証ロードマップ（軽量コアと SLM／二階建て）](planning/Resonanceverse実証ロードマップ_軽量コアとSLM二階建て.md) | Phase A〜C の切り分け・判定表・関連リンクのハブ（Phase 4 の位置づけは [ROADMAP Phase 4](planning/ROADMAP_Phase4_分散とエッジ_ja.md) へ） |
| [Phase3 計画（二階建てと実証）](planning/Phase3_計画_二階建てと実証.md) | **Phase 3**（二階建て・HBM・同一条件スイープ）のマイルストーンとスコープ |
| [Phase3 実測と主張完成計画（採用済み）](planning/Phase3_実測と主張完成計画_ja.md) | §2 の表を埋める実測・τ 事前登録・任意拡張（P0〜P3） |
| [透明性確保による人間-AI協働研究発展ロードマップ](planning/🚀%20透明性確保による人間-AI協働研究発展ロードマップ.md) | 研究協働・透明性の長期構想 |
| [⚡ Resonanceverse軽量実証実装：2週間プロトタイプ](planning/⚡%20Resonanceverse軽量実証実装：2週間プロトタイプ.md) | 詳細版プロトタイプ計画（[`tutorials/getting_started.md`](tutorials/getting_started.md) と併用可） |

## 理論（`theory/`）

| 文書 | 概要 |
|------|------|
| [Resonanceverse 理論（完全理論と実装）](theory/resonanceverse_theory.md) | コードと対応する理論の整理 |
| [数学的基礎（証明の完全版）](theory/mathematical_foundation.md) | 証明系の整理 |
| [動的共鳴に基づく自己創生的AIの数理基盤と実装](theory/Resonanceverse：動的共鳴に基づく自己創生的AIの数理基盤と実装.md) | 数理と実装の長文 |
| [Complete Mathematical Framework…（英語）](theory/Resonanceverse：Complete%20Mathematical%20Framework%20and%20Implementation%20for%20Autopoietic%20AI%20Through%20Dynamic%20Resonance.md) | 英語版フレームワーク |
| [理論的基盤と課題（部内限定版）](theory/Resonanceverse：動的共鳴に基づく自己創生的AIの理論的基盤と課題【機密%20-%20部内限定】ver.%206.0.md) | 機密ラベル付き整理 |
| [認知の戦略的制約とおぼろ性：情報理論的考察](theory/認知の戦略的制約とおぼろ性：情報理論的考察.md) | おぼろ性・情報理論 |

## 実装設計（`implementation/`）

| 文書 | 概要 |
|------|------|
| [実装基本設計（モジュール構造とクラス設計）](implementation/architecture_overview.md) | 全体モジュール構成 |
| [二階建てアーキテクチャ実装計画 v0.1](implementation/Resonanceverse：2階建てアーキテクチャ実装計画%20v0.1.md) | Phase C 周辺の本格実証設計 |
| [共鳴場による創発の観測・利用 設計 v0.1](implementation/🛠️%20共鳴場による創発の観測・利用%20設計%20v0.1.md) | 創発指標・実験との対応 |
| [共鳴制御（RCL）拡張設計 v0.1](implementation/🛠️%20Resonanceverse%20実装論：共鳴制御（RCL）拡張設計%20v0.1.md) | 多主体・監査・制御ループ |
| [LLM 選定戦略](implementation/🤖%20Resonanceverse実装におけるLLM選定戦略.md) | 下層モデル比較の方針 |

## API・実証索引（`api/modules/`）

[API・モジュール設計リファレンス](api/modules/README.md) を起点に、再現手順・対応表・計測考察へリンクしています。

---

*配置ルール: 新規文書は上表のいずれかのディレクトリへ置き、本インデックスに 1 行追加すること。*
