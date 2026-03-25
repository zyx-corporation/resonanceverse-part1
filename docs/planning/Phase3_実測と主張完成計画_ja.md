---
title: Phase 3 実測と主張完成計画（採用済み）
created: 2026-03-25
author: "Tomoyuki Kano <tomyuk@zyxcorp.jp>"
tags: ["Resonanceverse", "Phase 3", "実測", "τ", "ロードマップ"]
language: ja
document_type: planning
---

# Phase 3 実測と主張完成計画（採用済み）

## 1. 位置づけ

[Phase 3 計画（二階建てと実証）](Phase3_計画_二階建てと実証.md) の **§2 目的**は、実装（M1〜M5）に加え、**同一マシンで再現可能な数値の表**まで含む。**本稿はその「表を埋める」実行計画**を固定する。コードの新規マイルストーンではなく、**計測・記録・事前登録**のタスク分解である。

## 2. 目的（この計画で「完了」とするもの）

| # | 完了条件 | 検証方法 |
|---|----------|----------|
| A | **baseline／two_tier_stub** のデコード指標が、**事前登録した 1 条件組**で JSON に残る | `two_tier_sweep` の出力＋メタデータ（§4） |
| B | **同一モデル・代表系列長**で **HBM テンプレ**（`hbm_budget.v1`）が 1 本残る | `hbm_budget_probe` の出力 |
| C | **品質 τ** が実験前に文書化されている（事後変更しない） | 本リポジトリの短文（§5）または実験ノートへのリンク |
| D（任意） | 抽出型 QA の **実データ 1 ラン**（短ステップ可） | `squad_span` の JSON（`squad_span.v1`） |
| E（任意） | Router を **一段だけ**非乱数化し、挙動を 1 段落で説明できる | `core/two_tier/` の変更＋コミットメッセージまたは設計メモ |

## 3. 優先順（実施順）

1. **P0 — 実測スイープ＋HBM**（必須。§2 の表の核）
2. **P1 — τ 事前登録**（P0 と同じ週に文書化推奨）
3. **P2 — `squad_span` 実データ**（任意・ネットワーク・時間あり）
4. **P3 — Router 一段拡張**（任意・フル二階建ては対象外）

## 4. P0: 同一条件の記録（必須メタデータ）

各 JSON に加え、実験ノートまたは表の脚注に以下を**固定**する。

- 日付（UTC またはローカルとタイムゾーン）
- マシン識別子（例: 機種・GPU 型番・ドライバ版）
- `python`・`torch`・`transformers` の主要バージョン（`pip freeze` の抜粋で可）
- **seed**（`--seed`）
- モデル名（例: `gpt2`）、`max_new_tokens` / `repeats`、`seq_len`・`batch_size`（HBM 側）

**推奨コマンド例**（ルートで。GPU 利用時は `--cpu` を外す）:

```bash
# 一括（P0 + メタ JSON）— 推奨
python experiments/phase3_claim_run.py --model gpt2 --cpu --seed 0 \
  --max-new-tokens 16 --repeats 10 --seq-len 128 --batch-size 1 \
  --out-dir experiments/logs/phase3_claim

# 個別実行
python experiments/two_tier_sweep.py --model gpt2 --cpu --max-new-tokens 16 --repeats 10 \
  --seed 0 --out experiments/logs/phase3_claim_two_tier_sweep.json
python experiments/hbm_budget_probe.py --model gpt2 --cpu --seq-len 128 --batch-size 1 \
  --seed 0 --out experiments/logs/phase3_claim_hbm_budget.json
```

`experiments/logs/` は `.gitignore` 対象のため、**表用に要約した行**は `docs/api/modules/` の短文か、リポジトリ管理する `experiments/baselines/` 型のパスへ**コピー可否を別途決める**（本稿では必須としない）。

## 5. P1: 品質 τ の事前登録

- **対象**: 少なくとも「デコード経路で採用する／しない」の判定に使う指標 1 つ（例: 検証 perplexity の上限、下流 accuracy の下限）。
- **形式**: [Phase 3 計画 §5](Phase3_計画_二階建てと実証.md) と [創発の観測・利用 設計](<../implementation/🛠️ 共鳴場による創発の観測・利用 設計 v0.1.md>) に矛盾しない**一文＋数値**。
- **置き場所**: [phase_c_quality_tau_prereg.md](../api/modules/phase_c_quality_tau_prereg.md) および [Phase3 計画 §5.1](Phase3_計画_二階建てと実証.md)。

## 6. P2: 抽出型 QA（任意）

- `python experiments/squad_span.py --model distilbert-base-uncased --cpu --max-steps 50 --max-train-samples 256 --max-eval-samples 128`
- または **`phase3_claim_run.py --with-squad`**（`--demo` 時は `squad_span` 合成経路）。
- 初回はデータ・重み取得にネットワークが必要な場合がある。

## 7. P3: Router 一段拡張（任意）

- **実装**: `BlockRouterStub` の **`step_stride`**（`decode_benchmark` / `two_tier_sweep` の **`--router-step-stride N`**）。ステップ番号 mod N==0 のときだけ `keep=True`（**決定的**）。
- **非スコープ**: KV 階層の本実装・本番最適 Router。

## 8. スコープ外

- [Phase 4 独立ロードマップ](ROADMAP_Phase4_分散とエッジ_ja.md) に含まれる分散・Jetson の実装本体。
- 二階建て**フル**製品化。

## 9. 関連スキーマ

- [phase_c_decode_metrics.md](../api/modules/phase_c_decode_metrics.md)（`decode_benchmark.v1`）
- [phase_c_two_tier_sweep.md](../api/modules/phase_c_two_tier_sweep.md)（`two_tier_sweep.v1`）
- [phase_c_hbm_budget.md](../api/modules/phase_c_hbm_budget.md)（`hbm_budget.v1`）
- [phase_b_downstream_protocol.md](../api/modules/phase_b_downstream_protocol.md)（`squad_span.v1`）
- [phase_c_quality_tau_prereg.md](../api/modules/phase_c_quality_tau_prereg.md)（τ 事前登録）
- `phase3_claim_bundle.v1` / `phase3_claim_meta.v1` — [`experiments/phase3_claim_run.py`](../../experiments/phase3_claim_run.py)（P0）
- **実行手順の索引（表の転記・メタ）** — [phase3_p0_baseline_snapshot.md](../api/modules/phase3_p0_baseline_snapshot.md)
- **P2・P3（任意）の具体コマンド** — [phase3_p0_baseline_snapshot.md §Phase 3（続き）](../api/modules/phase3_p0_baseline_snapshot.md)
- **論文・対外向けの主張表と本文** — [Resonanceverse_主張表_論文用_ja.md](Resonanceverse_主張表_論文用_ja.md)

## 実装メモ（P0〜P3）

| 項目 | スクリプト／コード |
|------|-------------------|
| P0 | [`experiments/phase3_claim_run.py`](../../experiments/phase3_claim_run.py)（`two_tier_sweep` + `hbm_budget_probe` + メタ JSON） |
| P1 | [phase_c_quality_tau_prereg.md](../api/modules/phase_c_quality_tau_prereg.md) |
| P2 | 上記 `--with-squad`（`--squad-demo` で合成のみ） |
| P3 | `core/two_tier/stubs.py` の `step_stride`、`decode_benchmark` の `--router-step-stride` |

## 10. 改訂履歴

| 日付 | 内容 |
|------|------|
| 2026-03-25 | 初版（「次ステップ」提案の採用を文書化） |
| 2026-03-25 | P0〜P3 実装追記、`phase3_claim_run`・τ 登録・Router stride |
| 2026-03-25 | P0 手順のリポジトリ固定 [phase3_p0_baseline_snapshot.md](../api/modules/phase3_p0_baseline_snapshot.md)、P1 と P0 の関係を [phase_c_quality_tau_prereg.md](../api/modules/phase_c_quality_tau_prereg.md) に追記 |
| 2026-03-25 | P2・P3 のコマンド例を [phase3_p0_baseline_snapshot.md](../api/modules/phase3_p0_baseline_snapshot.md) に集約、ユニットテストで P2 同梱を固定 |
| 2026-03-25 | P2（SQuAD 同梱デモ）・P3（`router-step-stride 4` デモ）を実行し `experiments/baselines/` に要約 JSON を追加 |
| 2026-03-25 | [Resonanceverse_主張表_論文用_ja.md](Resonanceverse_主張表_論文用_ja.md) を追加（論文用要旨・主張一覧・四系統の本文） |
| 2026-03-25 | `slm_downstream` の **awai 読み出し**（`--awai-readout`）と **事前登録 v0.3（T3″）** を反映。[phase_b_downstream_protocol.md](../api/modules/phase_b_downstream_protocol.md) v0.4 にプロトコル追記 |
