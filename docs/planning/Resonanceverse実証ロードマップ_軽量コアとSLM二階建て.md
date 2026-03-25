---
title: Resonanceverse 実証ロードマップ（軽量コアと SLM／二階建て）
created: 2026-03-25
update: 2026-03-25T12:00:00+09:00
author: "Tomoyuki Kano <tomyuk@zyxcorp.jp>"
tags: ["Resonanceverse", "実証", "ロードマップ", "軽量実装", "二階建て", "SLM", "LLM"]
topic: 研究開発計画
source_type: planning
language: ja
summary: >
  PyTorch コア単体で検証可能な主張と、SLM/LLM を下層に据えた二階建てが本質となる主張を切り分け、
  段階的な実証ロードマップと関連文書への導線を示す。docs/api/modules の索引・性能実証向けコード開発（A′）を含む。
status: published
keywords: Resonanceverse, 実証, ロードマップ, 軽量プロトタイプ, 二階建て, SLM, 動的共鳴, 創発, API, 性能実証
document_type: roadmap
---

# Resonanceverse 実証ロードマップ（軽量コアと SLM／二階建て）

## 1. 本ロードマップの位置づけ

Resonanceverse 関連の文書には、次のような**異なる目的**が併存する。

- **軽量プロトタイプ**（大規模 LLM に依存せず、共鳴・ROI・創発指標などの**核**を検証する）
- **二階建てアーキテクチャ**（既存 Transformer／SLM を下層に保ちつつ、上層で制御・スパース化・KV 階層化し、**等品質・省資源**を狙う）

実証実験では「**何を主張するか**」によって、**PyTorch 実装のコアだけで足りるフェーズ**と、**SLM／LLM 接続および二階建てがほぼ必須のフェーズ**が分かれる。本稿はその**切り分け**と**推奨する段階順**をロードマップとして固定する。

## 2. 実証の二系統（依存関係の整理）

### 2.1 軽量コア（PyTorch）だけで十分になりやすい主張

次は、**自然言語の意味品質を LLM で担保しなくても**、再現性のある形で検証しやすい。

| 領域 | 例 | リポジトリ上の寄与 |
|------|-----|-------------------|
| 動的共鳴・場の更新 | 温度、朧度、安定化、バッファ更新 | `core/resonant_core.py` 等 |
| ROI／階層近似 | 全対全との時間・メモリ比較 | `core/roi_selector.py`, `experiments/efficiency_benchmark.py` |
| 創発の作動的指標 | 混合基準線との乖離、新規性、アブレーション | `experiments/emergence_metrics.py` |
| 埋め込み＋ファサード経路 | トークン ID からの場構築（小語彙・短系列） | `core/lightweight_resonance.py` |

**含意**: 理論・アルゴリズムの「**形式化と指標の健全性**」は、まずここで固められる。

### 2.2 SLM／LLM（および二階建て）が本質的になりやすい主張

次は、**入力が自然言語の高次元意味**に依存し、評価が**ベンチマーク品質**や**実タスク**に直結するため、下層に **SLM または LLM**（しばしば既存 Transformer ブロック）を置く実験がほぼ必須になる。

| 領域 | 例 | 理由 |
|------|-----|------|
| 文脈理解・タスク精度 | MMLU 系、長文、コード等 | 比較対象が「言語モデルとしての品質」 |
| 文化的ニュアンス（「あわい」等） | 婉曲・間合いの理解 | 単純パターン超えの意味推論が絡む |
| 等品質・省資源の同時達成 | `HBM_bytes/token`、p95 レイテンシ等 | 実装が**実際のデコーダ KV／アテンション**に依存 |
| 二階建て設計そのもの | Controller／Router／KV 階層 | 下層が「詳細展開用 Transformer」である前提 |

**含意**: 「**PyTorch の利用以上に**、SLM／LLM との接続による二階建てが**必要では**」という問いに対しては、**検証したい主張が上表の下側に入るなら、はい、そのフェーズでは二階建て（少なくとも下層に SLM／LLM）を組む必要がある**、と答えられる。

## 3. フェーズ別ロードマップ（推奨順）

### Phase A — 軽量コア実証（現在のリポジトリの主戦場）

**目的**: 共鳴場・ROI・創発指標の**定義・再現性・反証可能性**を固める。

- 成果物: 固定シードでのログ、スクリプト再現、`experiments/` のベンチ結果の解釈ルール
- 関連文書: [2 週間軽量プロトタイプ計画](../tutorials/getting_started.md)、[創発の観測・利用 設計 v0.1](../implementation/🛠️%20共鳴場による創発の観測・利用%20設計%20v0.1.md)

**この段階では、二階建て全体は必須ではない。**

### Phase A′ — `docs/api/modules` 整備と性能実証向けコード開発

[API・モジュール設計リファレンス](../api/modules/README.md) は現状、設計文書と `core/` への**索引**である。性能実証を「設計仕様と突き合わせて再現・報告できる」状態にするため、次の**コード／文書開発**を本ロードマップに明示的に含める。

**目的**: アルゴリズム仕様・モジュール責務と、`experiments/` および将来の二階建て実装の**対応表**を保ちながら、ベンチとログの手順を固定する。

| 種別 | 内容（例） |
|------|------------|
| 文書 | `algorithm_spec.md`／`architecture_overview.md` との**モジュール対応表**（どのクラスがどのベンチで検証されるか）— [module_benchmark_map.md](../api/modules/module_benchmark_map.md) を初版として追加 |
| コード | 既存: `experiments/efficiency_benchmark.py`, `emergence_metrics.py`, `evel_benchmarks.py` を、上記対応表から辿れるよう README に整理 |
| コード | **計測フック**（任意）: `core/` 主要パスに、トークンあたり活性メモリ・経過時間を記録する軽量フック（PyTorch `cuda.memory_stats` / `time.perf_counter` 等）。CLI または設定で on/off |
| コード | **CI**: `pytest`、`instrument_smoke`、短系列 `efficiency_benchmark` + `regression_check.py`（`tests/baselines/efficiency_cpu_short.json`、`--max-slowdown`） |
| 文書 | Sphinx／MkDocs 等の**自動 API ドキュメント**は Phase A′ 末または Phase B で導入を検討（必須ではない） |
| 文書 | **Phase A′ 締め**・CI による継続的評価の蓄積の考察（[phase_a_prime_closure_and_ci_accumulation.md](../api/modules/phase_a_prime_closure_and_ci_accumulation.md)） |

**成果物**: 再現手順（[phase_a_reproduction.md](../api/modules/phase_a_reproduction.md)）、モジュール↔実験スクリプト対応（[module_benchmark_map.md](../api/modules/module_benchmark_map.md)）、乱数固定（`core/reproducibility.py`）、各スクリプトの `--seed`、CI ゲートとベースライン運用（上記「CI 蓄積」文書）。

**Phase A と同時並行で進め、Phase B 以降では SLM 統合モジュール列も対応表に追加する。**

### Phase B — SLM アダプタ接続（橋渡し）

**目的**: 自然言語タスクに**触れつつ**、まだ「フル二階建て」ではなく、**隠れ状態＋共鳴層**の統合を検証する。

- **Phase 1B（README）**: 文化的調製の最小実装（`cultural_modulation`）と `AwaiIntegratedSLM` のスモーク（`experiments/slm_bridge_smoke.py`）で、本フェーズへの入口を固定する。
- **Phase 2（README）**: `experiments/slm_resonance_lm.py` による次トークン CE の最小学習ループ（`--demo` または HF モデル）で、損失曲線・凍結学習の土台を置く。`slm_data.py` と [Phase B データプロトコル](../api/modules/phase_b_data_protocol.md) で Wikitext 経路と perplexity を固定。
- 例: `transformers` による小規模因果 LM（CPU／単 GPU 可能な範囲）と `ResonantCore`／ヘッドの接続
- 成果物: 損失曲線・サンプル入出力・失敗モードの記録
- 関連: `core/resonant_core.py` の `AwaiIntegratedSLM` 方針、[LLM 選定戦略](../implementation/🤖%20Resonanceverse実装におけるLLM選定戦略.md)

**ここで初めて「言語品質」との対話が始まる。**

### Phase C — 二階建て（上層制御＋下層 Transformer）本格実証

**目的**: [二階建てアーキテクチャ実装計画](../implementation/Resonanceverse：2階建てアーキテクチャ実装計画%20v0.1.md) に沿った **Quality ≥ τ 制約下での資源指標**の実証。

- 必須になりやすい要素: ブロック化、Sequence Controller／Router、KV 階層、（任意）Draftor、蒸留・ルート損失
- 成果物: 等品質比較・等資源比較の表、消融計画、アブレーション
- **性能実証向けコード（本フェーズの中心）**:
  - 計画書の **HBM バイト予算表**に対応する、レイヤ別／前向き IO の**ログ実装**（`HBM_bytes/token` 等）
  - **p50／p95 レイテンシ**のデコードループ計測（バッチサイズ・系列長を前登録）
  - Baseline（Transformer 単体）と two-tier の**同一条件スイープ**用スクリプト（`experiments/` または専用 `benchmarks/`）
  - Phase A′ で定義した **モジュール↔計測**の対応を、Router／KV 等の新モジュール行に拡張

**「実証実験において PyTorch 以上に二階建てが必要か」が最も強く YES になるのはこのフェーズである。**

## 4. 判定表（どのフェーズで何を約束できるか）

| 主張のタイプ | Phase A | A′ | Phase B | Phase C |
|----------------|---------|-----|---------|---------|
| 共鳴場の更新式・安定性のスモーク | ◎ | ◎ | ◎ | ◎ |
| ROI／階層近似の計算量傾向 | ◎ | ◎ | ◎ | ◎ |
| 創発指標（KL／新規性／アブレーション） | ◎ | ◎ | ◎ | ◎ |
| **API／モジュール索引とベンチの対応・再現手順** | △ | ◎ | ◎ | ◎ |
| 自然言語タスク上の優位／同等 | △（限定的） | △ | ◎ | ◎ |
| 長文・実運用に近い効率（HBM／レイテンシ） | × | △ | △ | ◎ |
| 二階建て設計そのものの妥当性 | × | × | △ | ◎ |

◎＝主張として成立しやすい、△＝条件付き、×＝そのフェーズ単体では弱い。  
**A′** は Phase A と並行可能な「索引・性能実証ハーネス」フェーズである。

## 5. 関連ドキュメント（読む順の目安）

1. 本稿（実証の切り分けとフェーズ）
2. [2 週間軽量プロトタイプ計画](../tutorials/getting_started.md)
3. [API・モジュール設計リファレンス（索引）](../api/modules/README.md) — Phase A′ の作業入口
4. [二階建てアーキテクチャ実装計画 v0.1](../implementation/Resonanceverse：2階建てアーキテクチャ実装計画%20v0.1.md)
5. [創発の観測・利用 設計 v0.1](../implementation/🛠️%20共鳴場による創発の観測・利用%20設計%20v0.1.md)
6. [実装基本設計（モジュール構造）](../implementation/architecture_overview.md)
7. 研究協働・透明性の長期構想: [透明性確保による人間-AI協働研究発展ロードマップ](./🚀%20透明性確保による人間-AI協働研究発展ロードマップ.md)（本稿の**技術実証**とは別軸のロードマップ）

## 6. 改訂履歴

| 日付 | 内容 |
|------|------|
| 2026-03-25 | 初版〜 Phase A′ 更新: 再現性・索引・`StageTimer`（Facade / ResonantCore）、`instrument_smoke`、`regression_check` + `tests/baselines/efficiency_cpu_short.json`、CI 連携 |
