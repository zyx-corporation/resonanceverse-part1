# Phase C 品質 τ 事前登録（Phase 3 P1）

## 目的

[Phase 3 計画 §5](../../planning/Phase3_計画_二階建てと実証.md) に従い、**実験前**に採用する品質閾値 τ を固定し、事後の都合付けを避ける。

## Phase 3 P0（主張バンドル）との関係

- [`phase3_claim_run`](../../../experiments/phase3_claim_run.py) の出力（[phase3_p0_baseline_snapshot.md](phase3_p0_baseline_snapshot.md)）は **baseline／two_tier_stub のレイテンシ**と **HBM テンプレ**を主とする**資源比較**である。
- **T1** は現状 `two_tier_sweep` の棄却には用いない（比率・ログの報告用）。
- **T2〜T4** は、それぞれ `slm_resonance_lm`（perplexity）、`slm_downstream`（accuracy）、`squad_span`（exact match）で**等品質**を主張する際のゲートである。P0 のみを実行した場合、τ による除外は発生しない。
- P0 に **`--with-squad`** を付けて SQuAD を同梱する場合は **T4** を参照する。

## 登録（v0.1）

| ID | 対象 | 指標 | τ（閾値） | 解釈 |
|----|------|------|-----------|------|
| T1 | デコード比較（`decode_benchmark` / `two_tier_sweep`） | スタブ経路を**棄却**する条件 | **現時点では未使用** | two_tier_stub は計測・比率用。τ による棄却は、今後 perplexity 等と接続するときに T2 を参照。 |
| T2 | 因果 LM 検証（任意・`slm_resonance_lm` + `--eval-ppl`） | 検証 **perplexity** | **τ_ppl = 50.0**（上限） | 検証分割で ppl が τ_ppl **超過**なら「等品質」とはみなさない（報告から除外または再調整）。 |
| T3 | 下流 二値（`slm_downstream`） | 検証 **accuracy** | **τ_acc = 0.55**（下限） | 検証 accuracy が τ_acc **未満**なら「等品質」とはみなさない。 |
| T4 | 抽出型 QA（`squad_span`） | 検証 **exact_match_eval** | **τ_em = 0.10**（下限） | 短い finetune では低めに設定。未達なら「主張用の等品質表」から除外。 |

## 運用

- 上表の数値は **研究方針の出発点**であり、論文・対外報告で変更する場合は **改訂日と理由**を本ファイルの改訂履歴に残す。
- [創発の観測・利用 設計](<../implementation/🛠️ 共鳴場による創発の観測・利用 設計 v0.1.md>) の事前登録思想と整合する。

## 登録（v0.2）— τ 再挑戦用（実験前に固定）

v0.1 の **閾値 τ の数値は変更しない**（事後調整の回避）。**手続き・ハイパラのみ**を、等品質に近づけるための再挑戦として別枠で登録する。

| ID | 対象 | v0.1 との差分（固定） | 照合する τ（v0.1 と同一） |
|----|------|----------------------|---------------------------|
| **T3′** | GLUE SST-2（`slm_downstream`） | **エンコーダ未凍結**（`--freeze-encoder` を付けない）。`max_steps=200`、`lr=2e-5`、`seed=0`、`max_train_samples=4000`、`max_eval_samples=872`、`batch_size=16`、`distilbert-base-uncased`。baseline／awai の**同一条件比較**を維持。 | τ_acc = **0.55** |
| **T4′** | SQuAD v1（`squad_span`） | `max_steps=600`、`max_train_samples=3000`、`max_eval_samples=500`、`max_length=256`、`seed=0`、`distilbert-base-uncased`。学習率はスクリプト既定（`3e-5`）。 | τ_em = **0.10** |

**解釈:** T3′ は v0.1 の「凍結・300 steps」に対し、**表現の更新を許した短い再挑戦**である。T4′ は v0.1 の「極短学習・各 128 例」に対し、**ステップ数とデータ量を増やした再挑戦**である。いずれも未達の場合、等品質主張は引き続き行わない。

## 登録（v0.3）— T3″（awai 読み出しの事前固定）

v0.1 の **閾値 τ_acc は変更しない**。T3′ で **awai・読み出し `narrow`（6→ラベル直結）** が τ 未達であったため、**アーキテクチャ（読み出し）のみ**を変えた別プロトコルとして登録する。学習条件は **T3′ と同一**（未凍結、`max_steps=200`、`lr=2e-5`、`seed=0`、サンプル数・バッチ同上、`distilbert-base-uncased`）。

| ID | 対象 | 固定する追加条件 | 照合する τ |
|----|------|------------------|------------|
| **T3″** | GLUE SST-2・`integration=awai` | `--awai-readout projected` **または** `dual`（いずれか一方を実験前に採用し、採用理由を記録） | τ_acc = **0.55** |

**解釈:** `projected`（6 次元→hidden 再投影→ラベル）および `dual`（encoder プールと共鳴プールの連結）は、**同一学習条件下で τ を満たしうる**（実測は [`phase2_tau_gate_summary_v1.json`](../../../experiments/baselines/phase2_tau_gate_summary_v1.json) 参照）。**baseline 数値との完全一致**は主張の対象としない。`narrow` は T3′ の記録どおり τ の下に留まる。

## 登録（v0.4）— T2′（因果 LM・長めステップの再挑戦）

v0.1 の **τ_ppl は変更しない**。**極短ステップ（例: 40）**の記録が τ を大きく超過したため、**同一スクリプト**で学習ステップのみ伸ばした別プロトコルを登録する（事後の閾値すり替えではない）。

| ID | 対象 | 固定条件 | 照合する τ |
|----|------|----------|------------|
| **T2′** | `slm_resonance_lm`・Wikitext・`--eval-ppl` | `gpt2`、`max_steps=200`、`lr=3e-5`、`seed=0`、`max_chars=80000`、`batch=2`、`seq_len=32`、`eval_frac=0.1`。**`--baseline-hf` なし**（AwaiIntegratedSLM）。**`--freeze-base` なし**（ベース LM も更新）。 | τ_ppl = **50.0** |

**解釈:** 未達の場合は従来どおり「等品質」主張に用いない。達成した場合も **因果 LM 全体**の品質であり、Phase 3 デコード資源主張の代替にはならない。

**実装注記（2026-03-25）:** `AwaiIntegratedSLM` の語彙ヘッドは、旧実装の **`Linear(6, vocab)` のみ**では情報が細りすぎ、T2′ 実測で検証 ppl が極大となった。**最終隠れ状態と共鳴 6 次元を連結した `Linear(hidden+6, vocab)`** に変更した（下流 `dual` と同型）。以降の T2 再計測はこの実装を前提とする。

## 改訂履歴

| 日付 | 内容 |
|------|------|
| 2026-03-25 | 初版（Phase 3 P1・τ 事前登録） |
| 2026-03-25 | P0 主張バンドルとの関係（資源比較と τ の役割分担）を追記 |
| 2026-03-25 | **v0.2**（T3′・T4′ 再挑戦プロトコル、閾値は v0.1 と同一） |
| 2026-03-25 | **v0.3**（T3″: `--awai-readout projected` / `dual`、T3′ と同一学習条件） |
| 2026-03-25 | **v0.4**（T2′: 因果 LM 長めステップ・Awai 統合・Wikitext） |
| 2026-03-25 | **AwaiIntegratedSLM** 語彙ヘッドを **H+6 連結**に変更（T2′ 実測の極大 ppl の実装原因の除去）。再計測 JSON は `…_v05_awai_head_dual.json` |
