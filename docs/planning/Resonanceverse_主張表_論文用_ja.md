---
title: Resonanceverse 主張表（論文・対外向け）
created: 2026-03-25
update: 2026-03-25
language: ja
document_type: planning
summary: 品質 τ・資源実測・Phase 4・再現性の四系統を、論文に転用可能な表と本文で固定する。
---

# Resonanceverse 主張表（論文・対外向け）

本稿は、実装リポジトリに紐づく**主張の境界**と**証拠の置き場所**を、査読・投稿用の文章として**一段**まとめたものである。数値は**記録済みの要約 JSON**に依拠し、未実施の項目は**未実施**と明示する。

---

## 要旨（Abstract 用ドラフト）

本稿が扱う Resonanceverse の実証は、**単一ノード上**で、**因果言語モデル**（例: GPT-2）を下層に置いた**二階建てスタブ**（Controller／Router）と、**ベースライン・デコーダー単体**の比較を、**事前登録した計測**（デコード 1 ステップあたりのレイテンシ、HBM テンプレに沿った活性化バイト推定）で記録する。

**資源・時間**だけを主張する限り、**言語品質の同等性**は別途、**事前登録した閾値 τ**（検証 perplexity 上限、下流 accuracy 下限、抽出型 QA の exact match 下限）を満たす実験が必要である。現行スタブでは、Router のオーバーヘッドがレイテンシに表れうることが、同一条件の記録から示唆される。

**品質 τ の照合:** T2（因果 LM ppl）は **v0.1 および v0.4（T2′）**のいずれでも **Awai 統合は τ 未達**（baseline-hf 比較は要約 JSON）。T3（GLUE SST-2）では**v0.2（T3′）**で **baseline** は τ_acc を満たし、**awai・読み出し `narrow`** は未達であった。**v0.3（T3″）**として**同一学習条件**で **`--awai-readout projected` または `dual`** を採用した awai も **τ_acc を満たす**記録を得た（下表・要約 JSON）。**ゲート上は** baseline／awai（T3″）のいずれも τ に届くが、**数値の完全一致や二階建て一般の優位**は主張しない。T4（SQuAD EM）は**v0.2（T4′）で τ_em を満たす**（v0.1 極短学習は未達のまま併記）。資源比較と品質ゲートは**分離**して報告する。

**複数ノード・エッジ配備**（分散共鳴場の同期・観測）は、本リポジトリの Phase A〜C の**必須経路ではなく**、別ロードマップとして位置づける（[ROADMAP Phase 4](ROADMAP_Phase4_分散とエッジ_ja.md) 内「固定プロトコル草案（v0.1）」）。単一ホスト上の **IPC 往復**（バイト: [`distributed_sync_smoke_sample_v1.json`](../../experiments/baselines/distributed_sync_smoke_sample_v1.json)、**テンソル**: [`distributed_sync_smoke_tensor_dim64_v1.json`](../../experiments/baselines/distributed_sync_smoke_tensor_dim64_v1.json)）は**オーダー感の記録**であり、分散レイテンシの代替ではない。

---

## 表 1：主張一覧（ID・状態・根拠）

| ID | 主張の内容 | 状態 | 根拠（リポジトリ） | 品質 τ（該当時） |
|----|------------|------|-------------------|------------------|
| **C-T2** | 「等品質」として報告する場合、**perplexity** が事前登録上限以下である | **記録済・τ 未達**（v0.1 極短／**v0.4 T2′** いずれも awai **未達**；T2′ の **baseline-hf** も τ 未達） | v0.1: [`slm_resonance_lm_tau_t2_wikitext_gpt2_v1.json`](../../experiments/baselines/slm_resonance_lm_tau_t2_wikitext_gpt2_v1.json)；T2′: [`slm_resonance_lm_tau_t2_prime_prereg_v04_awai.json`](../../experiments/baselines/slm_resonance_lm_tau_t2_prime_prereg_v04_awai.json)、[`…_baseline_hf.json`](../../experiments/baselines/slm_resonance_lm_tau_t2_prime_prereg_v04_baseline_hf.json)、[`phase2_tau_gate_summary_v1.json`](../../experiments/baselines/phase2_tau_gate_summary_v1.json) | τ_ppl = **50.0**（T2′: awai ≈ **5.08×10⁴**、baseline-hf ≈ **69**） |
| **C-T3** | 二値下流で「等品質」とする場合、**検証 accuracy** が事前登録下限以上 | **記録済**（v0.1 凍結: 両系**未達**／T3′: baseline **達成**・awai **`narrow` 未達**／**T3″: awai `projected`・`dual` 達成**／**BoolQ・awai `projected`**） | T3″ SST-2: [`…_projected_prereg_v03.json`](../../experiments/baselines/slm_downstream_tau_t3_sst2_awai_projected_prereg_v03.json)、[`…_dual_prereg_v03.json`](../../experiments/baselines/slm_downstream_tau_t3_sst2_awai_dual_prereg_v03.json)；BoolQ: [`slm_downstream_boolq_awai_projected_prereg_v04.json`](../../experiments/baselines/slm_downstream_boolq_awai_projected_prereg_v04.json)、[`phase2_tau_gate_summary_v1.json`](../../experiments/baselines/phase2_tau_gate_summary_v1.json) | τ_acc = **0.55**（T3″ SST-2: projected ≈ **0.839**；BoolQ projected ≈ **0.628**） |
| **C-T4** | 抽出型 QA で「等品質」とする場合、**exact_match_eval** が事前登録下限以上 | **記録済**（v0.1 短学習**未達**／**v0.2 T4′ 長め学習**で**達成**） | [`squad_span_tau_t4_hf_prereg_v02.json`](../../experiments/baselines/squad_span_tau_t4_hf_prereg_v02.json)、短学習: [`squad_span_tau_t4_hf_v1.json`](../../experiments/baselines/squad_span_tau_t4_hf_v1.json) | τ_em = **0.10**（T4′: EM ≈ **0.384**） |
| **C-res** | 同一条件で **baseline** と **two_tier_stub** のデコード指標が JSON に残る | **記録済**（CPU / MPS） | [`phase3_claim_gpt2_summary_v1.json`](../../experiments/baselines/phase3_claim_gpt2_summary_v1.json)、[`phase3_claim_gpt2_mps_summary_v1.json`](../../experiments/baselines/phase3_claim_gpt2_mps_summary_v1.json) | T1 は棄却に**未使用** |
| **C-res** | 同一モデル・代表系列長で **HBM テンプレ**（活性化バイト合算）が 1 本残る | **記録済** | 上記バンドル内 `hbm_budget`、要約 JSON | — |
| **C-res** | Router **`step_stride`**（決定的）のログ | **デモ + gpt2 実モデル 記録済** | デモ: [`two_tier_sweep_stride4_demo_summary_v1.json`](../../experiments/baselines/two_tier_sweep_stride4_demo_summary_v1.json)；**gpt2**: [`two_tier_stride4_gpt2_cpu_full.json`](../../experiments/baselines/two_tier_stride4_gpt2_cpu_full.json) | — |
| **C-dist** | 複数ノード・エッジ上での共鳴場分割・同期の**固定プロトコル** | **草案**（単一ホスト IPC・**テンソル往復**スモークを記録） | [ROADMAP_Phase4_分散とエッジ_ja.md](ROADMAP_Phase4_分散とエッジ_ja.md)、[`distributed_sync_smoke_sample_v1.json`](../../experiments/baselines/distributed_sync_smoke_sample_v1.json)、[`distributed_sync_smoke_tensor_dim64_v1.json`](../../experiments/baselines/distributed_sync_smoke_tensor_dim64_v1.json) | — |
| **C-rep** | 再現に必要な **seed・device・ソフトウェア版**をメタに記載する | **実装済** | `phase3_claim_meta.v1`、[phase3_p0_baseline_snapshot.md](../api/modules/phase3_p0_baseline_snapshot.md) | — |

---

## 実施記録（τ 照合・資源の追加・2026-03-25）

**総合要約:** [`phase2_tau_gate_summary_v1.json`](../../experiments/baselines/phase2_tau_gate_summary_v1.json)

**T2:** **v0.1**（`max_steps=40`）は **Awai 統合で ppl が極大**（[`slm_resonance_lm_tau_t2_wikitext_gpt2_v1.json`](../../experiments/baselines/slm_resonance_lm_tau_t2_wikitext_gpt2_v1.json)）。**事前登録 v0.4（T2′）**（200 steps・未凍結・[`slm_resonance_lm_tau_t2_prime_prereg_v04_awai.json`](../../experiments/baselines/slm_resonance_lm_tau_t2_prime_prereg_v04_awai.json)／比較用 [`…_baseline_hf.json`](../../experiments/baselines/slm_resonance_lm_tau_t2_prime_prereg_v04_baseline_hf.json)）でも **awai は τ 未達**、**baseline-hf も ppl≈69 で τ_ppl=50 未達**。因果 LM の等品質主張には用いない。

**T3:** **v0.1（凍結・300 steps）**では **baseline／awai とも τ_acc 未達**。**v0.2（T3′）**は **未凍結・200 steps**（[phase_c_quality_tau_prereg.md](../api/modules/phase_c_quality_tau_prereg.md)）。**baseline**（[`slm_downstream_tau_t3_sst2_baseline_uf_prereg_v02.json`](../../experiments/baselines/slm_downstream_tau_t3_sst2_baseline_uf_prereg_v02.json)）は **≈0.865** で **τ 達成**。**awai・`narrow`**（[`slm_downstream_tau_t3_sst2_awai_uf_prereg_v02.json`](../../experiments/baselines/slm_downstream_tau_t3_sst2_awai_uf_prereg_v02.json)）は **≈0.509** で **未達**。**v0.3（T3″）**は学習条件を T3′ と同一にし **読み出しのみ** `--awai-readout projected` または `dual`。**projected**（[`slm_downstream_tau_t3_sst2_awai_projected_prereg_v03.json`](../../experiments/baselines/slm_downstream_tau_t3_sst2_awai_projected_prereg_v03.json)）**≈0.839**、**dual**（[`slm_downstream_tau_t3_sst2_awai_dual_prereg_v03.json`](../../experiments/baselines/slm_downstream_tau_t3_sst2_awai_dual_prereg_v03.json)）**≈0.847** で **いずれも τ_acc 達成**。**BoolQ**（別タスク・[`slm_downstream_boolq_awai_projected_prereg_v04.json`](../../experiments/baselines/slm_downstream_boolq_awai_projected_prereg_v04.json)）で **awai・projected** が **acc≈0.63** と **τ を満たす**記録を追加。**事前登録ゲート**上は SST-2／BoolQ で揃えられる場合があるが、**二階建て一般の優位**は主張しない。

**T4:** **v0.1**（極短 finetune、[`squad_span_tau_t4_hf_v1.json`](../../experiments/baselines/squad_span_tau_t4_hf_v1.json)）は **EM = 0.0** で **τ 未達**。**事前登録 v0.2（T4′）**（600 steps・学習 3000・評価 500 例、[`squad_span_tau_t4_hf_prereg_v02.json`](../../experiments/baselines/squad_span_tau_t4_hf_prereg_v02.json)）では **EM ≈ 0.384** と **τ_em = 0.10 を満たす**。

**資源（stride・gpt2）:** `--router-step-stride 4`・`gpt2`・CPU・seed=0・上記 Phase 3 と整合するデコード条件で `two_tier_sweep` を実行済み。p50 比 ≈ **0.989**、`router_keep_fraction_mean` = **0.25**（[`two_tier_stride4_gpt2_cpu_full.json`](../../experiments/baselines/two_tier_stride4_gpt2_cpu_full.json)）。

---

## （1）品質 τ と実証パイプライン（論文用本文）

**研究上の区別:** 本リポジトリの Phase 3 主張バンドル（`phase3_claim_run`）は、**資源・時間**の比較を主目的とする。したがって、**言語モデルとしての品質**（例: 検証分割における perplexity）や**下流タスク精度**を「等しい」として主張する場合は、**別パイプライン**で指標を取得し、**実験前に登録した閾値 τ**（[phase_c_quality_tau_prereg.md](../api/modules/phase_c_quality_tau_prereg.md)）と照合する。これは、事後的な閾値調整を避けるための**事前登録**の枠組みである。

**実施手順（推奨）:** 因果 LM の検証 perplexity は `slm_resonance_lm` と `--eval-ppl` を用い、τ_ppl を超過する場合は「等品質」としての報告から除外する。二値分類・BoolQ 等は `slm_downstream` で検証 accuracy を取得し、τ_acc 未満を同様に扱う（awai では **`--awai-readout`** を事前登録に含める）。抽出型 QA は `squad_span` で検証 exact match を取得し、τ_em 未満を同様に扱う。いずれも、**Phase 3 のレイテンシログ単体では代替できない**。

---

## （2）Phase 3 資源実測と数値の位置づけ（論文用本文）

**設定:** モデルは Hugging Face の `gpt2`、デコードは `max_new_tokens=16`、`repeats=10`、HBM 側は `seq_len=128`、`batch_size=1`、**seed=0**、スタブは `two_tier_stub` 経路を含む（詳細は [phase3_p0_baseline_snapshot.md](../api/modules/phase3_p0_baseline_snapshot.md)）。

**記録済みの要約（同一マシン上の相対比を主とする）:**

| 条件 | device | baseline p50 (ms) | two_tier p50 (ms) | p50 比（baseline÷two_tier） | HBM 合算 (bytes) |
|------|--------|-------------------|-------------------|-------------------------------|------------------|
| 記録例 A | cpu | 11.51 | 11.99 | 0.960 | 52690944 |
| 記録例 B | mps | 8.29 | 9.58 | 0.865 | 52690944 |
| stride=4・gpt2・cpu | cpu | 12.07 | 12.20 | 0.989 | （別 JSON 内 decode 指標） |

**解釈:** 上記比は **1 未満**であり、**当該スタブ設定では** baseline の 1 ステップあたりが短い傾向を示す。これは Router／Controller の**オーバーヘッド**が支配的になりうることを意味し、**二階建て一般の優劣**を意味しない。**`step_stride=4`** の実モデル記録では、p50 比が **1 に近づく**（≈0.99）一方、`router_keep_fraction_mean` = 0.25 がログに残る（決定的 Router の挙動確認用）。

---

## （3）Phase 4（分散・エッジ）と将来工作（論文用本文）

**スコープ外の明示:** 複数ノード上での共鳴場の分割・同期・観測、およびエッジデバイス（例: Jetson）への配備は、**単一ノード実証（Phase A〜C）の完了条件には含めない**（[ROADMAP_Phase4_分散とエッジ_ja.md](ROADMAP_Phase4_分散とエッジ_ja.md)）。**固定プロトコル草案（v0.1）**を同文書に記した。本論文の主張が**スケールアウト**に及ぶ場合は、**別途**ハード選定とプロトコル固定を行う必要がある。

---

## （4）再現性とデータ可用性（論文用本文）

**メタデータ:** 主張バンドルは `phase3_claim_meta.v1` により、生成時刻（UTC）、プラットフォーム、`python`／`torch`／`transformers` の版、CUDA／MPS の可用性、実験パラメータ（seed、モデル名、系列長等）を記録する。

**完全ログ:** 生の `phase3_claim_bundle.json` は `experiments/logs/` に出力され、**バージョン管理の対象外**（`.gitignore`）とする。リポジトリには**要約 JSON**（`experiments/baselines/`）を置き、表の転記に用いる。

**対外報告:** 絶対レイテンシはハード依存であるため、**同一マシン・同一 seed**での baseline と two_tier の**相対比**を主とし、メタを脚注または補助資料に示すことを推奨する。

---

## 改訂履歴

| 日付 | 内容 |
|------|------|
| 2026-03-25 | 初版（主張表・論文用本文・四系統の採用） |
| 2026-03-25 | T2／T3 スモーク記録・τ 未達の明示、gpt2・`stride=4` 資源記録、[phase2_tau_gate_summary_v1.json](../../experiments/baselines/phase2_tau_gate_summary_v1.json) |
| 2026-03-25 | T4（SQuAD HF・短学習）記録・τ 未達の明示、[squad_span_tau_t4_hf_v1.json](../../experiments/baselines/squad_span_tau_t4_hf_v1.json) |
| 2026-03-25 | T3（GLUE SST-2 実データ）・`slm_downstream` の JSON 安全化、Phase 4 プロトコル草案 |
| 2026-03-25 | T3（SST-2・`integration=awai` 同一条件比較） |
| 2026-03-25 | 事前登録 **v0.2**（[phase_c_quality_tau_prereg.md](../api/modules/phase_c_quality_tau_prereg.md)）：T3′（未凍結）・T4′（長め SQuAD）、[`phase2_tau_gate_summary_v1.json`](../../experiments/baselines/phase2_tau_gate_summary_v1.json) 更新、Phase 4 IPC サンプル JSON |
| 2026-03-25 | 事前登録 **v0.3**（T3″: `slm_downstream` の awai 読み出し `projected`／`dual`）、実測 JSON・要約表・主張表 C-T3 更新 |
| 2026-03-25 | 事前登録 **v0.4**（T2′）、BoolQ・`load_dataset("boolq")`、Phase 4 tensor 往復スモーク、要約 JSON・主張表 §T2／T3 追記 |
| 2026-03-25 | 主張表 表1（C-T2／C-T3／C-dist）を T2′・BoolQ・tensor IPC に同期、BoolQ ラベル単体テスト |
