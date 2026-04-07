# RVT-EXP-2026-008 と本リポジトリのアーキテクチャ対応（齟齬の解消）

**目的**: 実験計画書 RVT-EXP-2026-008（地形の設計的自己整合性）に書かれた**最小構成**と、本リポジトリの **v7 Phase II-A / I-A** 実装が「別物なのか／どこまで同じか」を固定する。

**結論（先に）**

- **対立しない**。計画書は **観測・介入・生成**の 3 層を含む**拡張スタック**を仮定し、本リポは現状 **観測層の一部（前向き 1 回・事後スカラー化）**までが整備されている。
- 「アーキテクチャが違う」**ではなく**、「**実装されている層の深さが違う**」。未実装部分を明示すれば齟齬は解消される。

---

## 1. 三層スタックで整理する

| 層 | 計画書 RVT-EXP-2026-008 | 本リポジトリ（現状） |
|----|-------------------------|----------------------|
| **L1 観測** | 注意から \(W_\text{asym}\) を**抽出**し、必要なら軸別ノルムを記録 | **実装済み**: 層ごとの注意（ヘッド**平均**後）から話者ブロック間の **スカラー** \(S_{\text{asym},ab}, S_{\text{asym},ba}\) を計算 |
| **L2 介入** | \(W_\text{asym}\) を注意に**残差注入**（WASYM / SYM 対照） | **本線統合**: [`v7_phase2a_empirical_run.py`](../../experiments/v7_phase2a_empirical_run.py) の ``--rvt-l2-mode sym|wasym``。**``attn_implementation=\"eager\"``** の Causal LM（GPT2 / Llama / Mistral 等・[`rvt_exp_2026_008_attn_inject.py`](../../experiments/rvt_exp_2026_008_attn_inject.py)）。**SDPA では不可**（重み非可視） |
| **L3 生成ループ** | 朧モニタが **logit 列**を見て温度を更新し **generate** と結合 | **スタンドアロン実験 CLI**: [`rvt_exp_2026_008_oboro_generate.py`](../../experiments/rvt_exp_2026_008_oboro_generate.py)（``--profile full`` で v2 トレース、``--demo`` で HF なし合成 JSON）。**審判・統合チャットの generate とは意図的に未接続**（バンドル同梱は ``v7_phase4_minimal_repro.py`` の ``--with-oboro-standalone-demo``） |

本リポの主系は **L1 のみ**。計画書の実験 A の **BASE 条件**（抽出のみ）は L1 に対応。**WASYM / SYM** は L2 なので**別実装**が必要。

---

## 2. 用語・記号の対応表

| 計画書 | 本リポ | 注記 |
|--------|--------|------|
| \(W_\text{asym}\)（\(|A|\times|B|\times 6\)） | 直接的な対応オブジェクト**なし** | 計画書は**ヘッド別**非対称を \(M\in\mathbb{R}^{H\times 6}\) で 6 軸に射影 |
| （ヘッド別）\(S_{\text{asym},h}\) | **部分対応**: `output_attentions` の **層テンソル** `[H,L,L]` が取得可能だが、現行パイプラインは **平均後** `(L,L)` のみ使用 | `hf_forward_attention_layer_matrix` は **mean(dim=0)** 済み |
| 話者ブロック A/B | **同一**: MRMP 窓の `speaker_src` / `speaker_tgt`（表示名）に基づくトークン集合 | `speaker_token_indices_mrmp_window` |
| ブロック間非対称（スカラー） | **`S_asym_ab`, `S_asym_ba`** | `pair_block_asymmetry_frobenius`；事前登録 [v7_phase2a_empirical.py](../../experiments/v7_phase2a_empirical.py) と一致 |
| \(\phi\)（6 軸分解） | **相関検証用**は Phase I-A（`\|S_\text{asym}\|_F` とラベル）。**ヘッド→6軸の学習可能 \(M\)** は [`rvt_exp_2026_008_train_head_axis_m.py`](../../experiments/rvt_exp_2026_008_train_head_axis_m.py) | **計画書 6 軸名 ↔ v7 審判キー（ab）**の固定表は [`rvt_exp_2026_008_judge_axis_mapping.py`](../../experiments/rvt_exp_2026_008_judge_axis_mapping.py)。v7 は **12 キー**（ab/ba）。[v7_phase1a_llm_judge_six_axes.py](../../experiments/v7_phase1a_llm_judge_six_axes.py) |
| `awai_vector`（計画書） | **`core/v7_awai_metrics.py` 等の Ω 系列** | **定義式が別**。同一モジュールに「計画書の AwaiVector」を足すか、文書上で **RVT-Awai** と **v7-Ω** と区別 |
| `OboroMonitor` | **`ResonantCore` の θ／朧度** | **入力が違う**（logit 履歴 vs 隠れ状態）。計画書実験 C の固着モニタは**生成時ログット**前提 |
| `AwaiIntegratedSLM` | **SLM 学習・Streamlit 統合チャット** | 計画書の「最小三要素」とは**目的が異なる**（統合モデル vs 地形検証パイプライン） |

---

## 3. 数学的な含意（スカラー S と テンソル W）

事前登録の **\(S_{\text{asym},ab}\)** は、単一の行ソフトマックス注意 \(A\) とブロック \((I_a,I_b)\) に対する

\[
\| A[I_a,I_b] - A[I_b,I_a]^\top \|_F
\]

**1 スカラー**。

計画書の **\(W_\text{asym}[i,j,k]\)** は、各ヘッドのブロック間非対称を **軸 \(k\)** に線形結合した **6 チャネル**。

- **関係**: 「全ヘッドを均等重みで足してから 1 本のブロック差を取る」ことと、「ヘッドごとに差を取ってから \(M\) で結合」は**一般には等価ではない**。
- **実務**: Phase II-A を維持したまま計画書 A をやるなら、**別スクリプト**で `[H,|A|,|B|]` 相当を切り出し、\(M\)（固定または学習）で 6 軸に射影する**拡張行**を定義すればよい。既存 `R(τ)` はそのまま「スカラー系」の主解析として残せる。

---

## 4. データ・再現性（Day 0）

| 計画書 | 本リポ |
|--------|--------|
| `windows.jsonl`、3146 サンプル | `v7_mrmp_prepare.py` → `experiments/logs/mrmp_prepared/windows.jsonl`；行数は環境依存 |
| シード 42 | 各スクリプトの `--seed` / `set_experiment_seed` で**個別指定**（リポ全体固定 42 の単一名義ファイルは無い） |
| gpt2 124M | `v7_phase2a_empirical_run.py` の既定どおり **HF `gpt2`** |

**自動検証（推奨）**: 整形直後に必須キー・行数・（任意で）`manifest.json` の `n_utterance_rows` 一致を確認する。続けて軽く RVT バッチだけ試すなら [`experiments/run_rvt_explore.sh`](../../experiments/run_rvt_explore.sh)（窓ファイルが無い環境では何もしない）。

**Phase II-A 本線との任意連携**: [`experiments/run_phase2a_mrmp_tau.sh`](../../experiments/run_phase2a_mrmp_tau.sh) に **`RVT_EXPLORE=1`** を渡すと、ブートストラップ完了後に **同一 `WINDOWS`** で `run_rvt_explore.sh` を実行する（小バッチ既定・`RVT_EXPLORE_SKIP_DAY0` 既定 `1` で Day0 の二重実行を避ける）。ルートの [`run_phase2a_mrmp_n3146_rinna.sh`](../../run_phase2a_mrmp_n3146_rinna.sh) からも環境変数は子プロセスへ引き継がれる。

```bash
python experiments/rvt_exp_2026_008_day0_checks.py \
  --windows experiments/logs/mrmp_prepared/windows.jsonl \
  --min-rows 1
```

厳密に manifest まで一致させる場合: `--strict-manifest`（既定で同ディレクトリの `manifest.json` を読む。読まないときは `--no-manifest`）。

---

## 5. 実装ギャップ一覧（優先度の目安）

**2026-04 更新（計画書フルに**近づける**スタック：本線 L2・教師あり M・ε スイープ・Oboro v2・八条件・ executor。L2 は eager Causal LM。fused SDPA カーネルへの直接パッチは未対応。）**

| # | 項目 | リポ内の入口 |
|---|------|----------------|
| 1 | ヘッド別注意 `(H,L,L)` | [`experiments/rvt_exp_2026_008_attention.py`](../../experiments/rvt_exp_2026_008_attention.py) |
| 2 | ヘッド Frobenius → 6 軸への射影（**学習可能 M**・読込/合成訓練） | [`experiments/rvt_exp_2026_008_w_asym.py`](../../experiments/rvt_exp_2026_008_w_asym.py)·[`rvt_exp_2026_008_head_axis_matrix.py`](../../experiments/rvt_exp_2026_008_head_axis_matrix.py)·[`rvt_exp_2026_008_train_head_axis_m.py`](../../experiments/rvt_exp_2026_008_train_head_axis_m.py)·`mrmp_row.py` の ``--head-axis-matrix`` |
| 3 | L2（eager 事後ブレンド + **Phase II-A 本線**） | [`rvt_exp_2026_008_attn_inject.py`](../../experiments/rvt_exp_2026_008_attn_inject.py)·[`v7_phase2a_empirical_run.py`](../../experiments/v7_phase2a_empirical_run.py) ``--rvt-l2-mode`` |
| 4 | `RvtExp008AwaiVector` | [`experiments/rvt_exp_2026_008_awai_vector.py`](../../experiments/rvt_exp_2026_008_awai_vector.py) |
| 5 | L3（Oboro：lite / **full v2** トレース・**--demo**） | [`rvt_exp_2026_008_oboro_generate.py`](../../experiments/rvt_exp_2026_008_oboro_generate.py)·[`run_rvt_oboro_demo.sh`](../../experiments/run_rvt_oboro_demo.sh)。プランでは ``args_hint.oboro_demo=true``（[`rvt_exp_2026_008_plan_execute.py`](../../experiments/rvt_exp_2026_008_plan_execute.py)） |
| 6 | 実験 D：**八条件グリッド**・プラン JSON・**任意実行**・**無人連続実行** | [`rvt_exp_2026_008_eight_conditions.py`](../../experiments/rvt_exp_2026_008_eight_conditions.py)·[`rvt_exp_2026_008_ablation_runner.py`](../../experiments/rvt_exp_2026_008_ablation_runner.py)（`--preset eight_grid`・`--run-eight-grid`・`--manifest`・`--execute`・`--no-dry-run`）·[`experiments/run_rvt_eight_grid_unattended.sh`](../../experiments/run_rvt_eight_grid_unattended.sh)·[`rvt_exp_2026_008_plan_execute.py`](../../experiments/rvt_exp_2026_008_plan_execute.py) |
| 7 | MRMP ``windows.jsonl`` 1 行 / バッチ CLI・Awai 蓄積 | [`experiments/rvt_exp_2026_008_mrmp_row.py`](../../experiments/rvt_exp_2026_008_mrmp_row.py) · シェル [`experiments/run_rvt_mrmp_batch.sh`](../../experiments/run_rvt_mrmp_batch.sh) |
| 8 | L2 logits スモーク | [`experiments/rvt_exp_2026_008_l2_smoke.py`](../../experiments/rvt_exp_2026_008_l2_smoke.py) |

**未着手・注意**

- L2 は各 ``modeling_*`` の **eager** ``eager_attention_forward`` をラップ（既知プリパッチ＋未登録は介入適用時に遅延パッチ）。**SDPA / flash 経路**は未対応。厳密入れ子セッションは ``gpt2_rvt_inject_session``。
- L3（Oboro）は **スタンドアロン CLI**（``--demo``・``v7_phase4`` 同梱可）。統合チャット **generate 未接続**。コアは **貪欲 1 トークン forward**（サンプリング・長文ベンチの**別タスク定義**も未接続）。
- **教師あり M**: ``rvt_exp_2026_008_train_head_axis_m.py`` の ``--supervised-jsonl``（``per_head_block_frobenius`` + ``w_target`` または審判 ``*_ab``）。**ε スイープ**: [`rvt_exp_2026_008_wasym_eps_sweep.py`](../../experiments/rvt_exp_2026_008_wasym_eps_sweep.py)。executor **本実行**は HF・窓 JSONL 要・**CI は dry-run**推奨。

---

## 6. 「齟齬解消」の使い方

- **論文・補遺**: 「本リポの Phase II-A は RVT-EXP-2026-008 の **L1・BASE 観測**に対応し、**L2/L3 は当該計画書の追加実装**である」と書ける。
- **開発**: 新規コードは `experiments/rvt_exp_2026_008_*.py` のように**名前空間を分け**、v7 事前登録済み指標（`S_asym_ab`・`R(τ)`）を**上書きしない**。

---

**更新**: 2026-04-03 初版（リポ内整合用）。
