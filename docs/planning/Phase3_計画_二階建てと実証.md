---
title: Phase 3 計画（二階建て実証・ロードマップ Phase C 相当）
created: 2026-03-25
author: "Tomoyuki Kano <tomyuk@zyxcorp.jp>"
tags: ["Resonanceverse", "Phase 3", "二階建て", "実証", "HBM", "ロードマップ"]
language: ja
document_type: planning
---

# Phase 3 計画（二階建て実証・ロードマップ Phase C 相当）

## 1. 位置づけ

| 用語 | 意味 |
|------|------|
| **本稿の Phase 3** | [ROADMAP_ja.md](../../ROADMAP_ja.md) における **「Phase 2–3」のうち、二階建て・資源実証に進む段階** |
| **実証ロードマップとの対応** | [Resonanceverse実証ロードマップ（軽量コアと SLM／二階建て）](Resonanceverse実証ロードマップ_軽量コアとSLM二階建て.md) の **Phase C（二階建て本格実証）** と**概ね一致**させる |
| **Phase 2 との境界** | Phase 2 まで：**単一スタック**上の SLM 橋・最小学習・下流入口（`slm_resonance_lm` / `slm_downstream`）。Phase 3：**上層制御＋下層 Transformer** の分離、**品質 τ 制約下の HBM／レイテンシ**の主役化 |
| **Phase 4 との境界** | [Phase 4 独立ロードマップ](ROADMAP_Phase4_分散とエッジ_ja.md)（Jetson 等の**ノード間分散**）は本稿のスコープ**外**。Phase 3 は**単一ノード／単一プロセス**を主戦場とし、必要なら将来 Phase 4 と接続する |

## 2. 目的（何を「計画完了」とみなすか）

1. **等品質比較・等資源比較**の表を出せること（定義は [二階建てアーキテクチャ実装計画 v0.1](<../implementation/Resonanceverse：2階建てアーキテクチャ実装計画 v0.1.md>) §1）。
2. **Baseline（Transformer 単体）**と **two-tier（上層＋下層）**を**事前登録した条件**（系列長・バッチ・モデル）で並べられること。
3. 指標として少なくとも **`HBM_bytes/token`（または同等のレイヤ別メモリログ）**と **デコード p50／p95** の**再現手順**が `docs/api/modules` に結びつくこと。

「フル製品の二階建て完成」ではなく、**主張可能な実証の最小セット**を完了条件とする。

## 3. スコープ

### 3.1 含む（優先順）

| 優先 | 内容 | 根拠文書・既存資産 |
|------|------|---------------------|
| P0 | **計測ハーネス**: デコードループの時間（p50/p95）、CUDA メモリピーク／層別に近い分解 | `instrumentation.StageTimer`、[計測戦略の考察](../api/modules/measurement_strategy.md) |
| P0 | **Baseline vs 二階建て**の同一条件スイープ用 CLI または設定駆動スクリプト（`experiments/` または `benchmarks/`） | 実証ロードマップ Phase C の「同一条件スイープ」 |
| P1 | **HBM バイト予算表**に対応するログ列（テンプレは [二階建て計画 §7.2](<../implementation/Resonanceverse：2階建てアーキテクチャ実装計画 v0.1.md>)） | 表の「2枚」比較が可能な粒度 |
| P1 | **Router／Controller のスタブ**（活性ブロック選択の疑似実装）と、スキップ率・品質への影響の記録 | 創発・消融と [創発の観測・利用 設計](<../implementation/🛠️ 共鳴場による創発の観測・利用 設計 v0.1.md>) との接続 |
| P2 | **下流**: 抽出型 QA（SQuAD 形式）など Phase 2 で切ったタスクの**プロトコル草案＋最小コード** | [phase_b_downstream_protocol.md](../api/modules/phase_b_downstream_protocol.md) の拡張 |

### 3.2 含まない（明示的に後回し）

- **Phase 4** の分散共鳴・クラスタ同期の実装本体（計画・インタフェースのメモのみ可）。
- **商用スケール**の全自動運用（監査・多テナント）は [透明性・協働ロードマップ](<./🚀 透明性確保による人間-AI協働研究発展ロードマップ.md>) 側。
- **特定クラウド専用**の最適化のみに依存する結果（再現手順が閉じないもの）。

## 4. マイルストーン（案）

| ID | 名称 | 主な成果物 | 完了の目安 |
|----|------|------------|------------|
| **M1** | 計測の固定化 | デコード＋メモリの JSON スキーマ、`module_benchmark_map` への行追加 | Baseline のみでも通る |
| **M2** | 二階建てスケルトン | `core/` に Controller/Router のインタフェースとスタブ、**品質 τ** の検査フック | 1 モデル・短系列でスモーク |
| **M3** | 同一条件スイープ | `baseline` / `two_tier` の2系統を同一 YAML／CLI で実行し、表用 CSV／JSON を出力 | 実証ロードマップ Phase C の「表」に相当 |
| **M4** | HBM 予算表の充填 | レイヤ別ログがテンプレの行に対応し、差分が説明可能 | 少なくとも 1 対 1 比較が掲載可能 |
| **M5（任意）** | 下流拡張 | 抽出型 QA のプロトコル v0.1 とスクリプト入口 | Phase 2 の「未登録」を縮小 |

マイルストーンの順序は **M1 → M2 → M3** を推奨（計測が先にないと比較が成立しない）。

## 5. リスクと前提

| 項目 | 内容 |
|------|------|
| **ハード依存** | GPU 世代・ドライバで数値が変わる。Phase A′ と同様、**相対悪化＋同一マシン**を主にし、絶対値は参考。 |
| **品質 τ** | タスク・モデルごとに閾値を事前登録（[創発設計](<../implementation/🛠️ 共鳴場による創発の観測・利用 設計 v0.1.md>) の思想に整合）。 |
| **実装負荷** | フル二階建ては長期。**M2 はスタブでよい**（Router を乱数／ヒューリスティックでも可）が、**ログ形式**は本番想定に合わせる。 |

### 5.1 品質 τ の事前登録（P1）

数値の登録表は [phase_c_quality_tau_prereg.md](../api/modules/phase_c_quality_tau_prereg.md) を参照（実験前に固定）。

## 6. 関連ドキュメント（読む順の目安）

1. [実証ロードマップ（軽量コアと SLM／二階建て）](Resonanceverse実証ロードマップ_軽量コアとSLM二階建て.md) — Phase C の定義  
2. [二階建てアーキテクチャ実装計画 v0.1](<../implementation/Resonanceverse：2階建てアーキテクチャ実装計画 v0.1.md>) — 目的関数・HBM 表  
3. [ROADMAP_ja.md](../../ROADMAP_ja.md) — Phase 1A〜3 のチェックリスト／分散は [ROADMAP_Phase4_分散とエッジ_ja.md](ROADMAP_Phase4_分散とエッジ_ja.md)  
4. [module_benchmark_map.md](../api/modules/module_benchmark_map.md) — 実装とベンチの対応（Phase 3 で拡張）  
5. **[Phase 3 実測と主張完成計画（採用済み）](Phase3_実測と主張完成計画_ja.md)** — §2 の「表」を埋める実行手順（P0〜P3）  
6. [phase_c_quality_tau_prereg.md](../api/modules/phase_c_quality_tau_prereg.md) — τ 事前登録（P1）

## 7. 改訂履歴

| 日付 | 内容 |
|------|------|
| 2026-03-25 | 初版（Phase 3 計画の独立化） |
| 2026-03-25 | M1〜M3 実装: `decode_benchmark`／`core/two_tier`／`two_tier_sweep`、計測スキーマ文書 |
| 2026-03-25 | M4〜M5: `hbm_budget_probe`（`hbm_budget.v1`）、`squad_span`（`squad_span.v1`）と API モジュール文書 |
| 2026-03-25 | Phase 4 の境界リンクを [ROADMAP_Phase4_分散とエッジ_ja.md](ROADMAP_Phase4_分散とエッジ_ja.md)（独立ロードマップ）へ更新 |
| 2026-03-25 | [Phase3 実測と主張完成計画](Phase3_実測と主張完成計画_ja.md) を追加（§2 の表を埋める採用済み手順） |
| 2026-03-25 | §5.1 τ 事前登録、`phase3_claim_run`／Router stride（P0〜P3 実装） |
