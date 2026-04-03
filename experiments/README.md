# `experiments/` — v7 実験設計との対応

**正本**: [v7.0 実証実験設計書](../docs/v7/Resonanceverse_v7.0_Experimental_Design.md) · [EXPERIMENT_ROADMAP_v7.md](../docs/planning/EXPERIMENT_ROADMAP_v7.md)

**日本語ローカル SLM（M3 Max / 128GB）**: 採用モデル・フェーズ・再現メタ — [`docs/planning/v7_local_slm_m3_japanese_plan.md`](../docs/planning/v7_local_slm_m3_japanese_plan.md)

## v7 一括実行

```bash
python experiments/v7_run_suite.py --demo --out experiments/logs/v7_suite/suite.json
```

要約 JSON（デモ）: [`baselines/v7_suite_bundle_demo_v1.json`](baselines/v7_suite_bundle_demo_v1.json)

最小の次（**gpt2・層別 S_asym 統計・ラベル無し**）: [`baselines/v7_phase1a_hf_gpt2_summary_v1.json`](baselines/v7_phase1a_hf_gpt2_summary_v1.json)

Phase II-A（MRMP τ 掃引・n=3146 ログ参照）: [`baselines/v7_phase2a_mrmp_tau_n3146_bundle_v1.json`](baselines/v7_phase2a_mrmp_tau_n3146_bundle_v1.json)

Phase II-A（**6 軸審判済み** `v7_judge_10k.jsonl`・先頭 3146 行・補助解析同梱）: [`baselines/v7_phase2a_mrmp_tau_n3146_judge10k_bundle_v1.json`](baselines/v7_phase2a_mrmp_tau_n3146_judge10k_bundle_v1.json)

Phase III（合成・あわい Ω）: [`baselines/v7_phase3a_synthetic_bundle_v1.json`](baselines/v7_phase3a_synthetic_bundle_v1.json)

Phase I-A（MRMP 審判 6 軸 × Frobenius）: [`baselines/v7_phase1a_mrmp_judge_v7_axes_bundle_v1.json`](baselines/v7_phase1a_mrmp_judge_v7_axes_bundle_v1.json) · [`run_phase1a_mrmp_v7_axes.sh`](run_phase1a_mrmp_v7_axes.sh)

Phase IV 周辺（最小再現バンドル・demo）: [`baselines/v7_phase4_minimal_repro_demo_v1.json`](baselines/v7_phase4_minimal_repro_demo_v1.json) · [`v7_phase4_minimal_repro.py`](v7_phase4_minimal_repro.py) · [v7_phase4_integration_repro.md](../docs/planning/v7_phase4_integration_repro.md)

## v7 フェーズ別スクリプト

| スクリプト | 内容 |
|------------|------|
| `v7_run_suite.py` | I-A・I-B・II-A・III-A（合成）を連続実行 |
| `v7_phase1a_phi_correlation.py` | Phase I-A: S_asym と相関（`--demo`＝合成） |
| `v7_phase1b_directed_tensor.py` | Phase I-B: 有向テンソル非対称性 |
| `v7_phase2a_delay_sweep.py` | Phase II-A: τ 掃引（合成テンソル）。`--alpha-list` で強凸性代理の感度表（`v7_phase2a_alpha_sweep.v1`） |
| `v7_phase2a_compare_runs.py` | 複数 `*_with_contrib.json` / tau_summary の比較表（JSON・MD） |
| `v7_phase2a_primary_aux_tau_association.py` | 主 R_mean(τ) vs 補助各軸 R_mean(τ) の Pearson / 簡易部分相関（探索的） |
| `v7_phase4_minimal_repro.py` | Phase IV 全景ではない。`--demo --cpu` で `two_tier_sweep`（＋任意 `squad_span` デモ）を 1 JSON に |
| `v7_phase3a_awai_metrics.py` | あわい Ω（合成軌跡） |
| `run_phase3a_synthetic.sh` | Phase III（合成）Ω デモの一括 JSON 出力（`SEED` / `T` / `D` / `OUT` で上書き可） |
| `v7_phase2a_bundle_validate.py` | Phase II-A ベースライン JSON の `artifacts` 存在・主要 JSON スキーマ検証（`--strict` で欠落も失敗）。`--out-prefix` / `OUT_PREFIX` でパス置換。`--verify-manifest` で [`v7_phase2a_repro_manifest.py`](v7_phase2a_repro_manifest.py) 出力の SHA256 照合 |
| `v7_phase2a_repro_manifest.py` | 再現・補遺用マニフェスト（コード・事前登録の SHA256、任意で成果物・図）。`--pin-code-only` は CI 向け（ログ不要）。`--verify` で照合 |
| `v7_phase2a_tau_plots.py` | Phase II-A `by_tau` から **PNG**（R_mean・R_var＋平滑・n(τ)；補助 6 軸は別ファイル）。**`--paper`** で論文向け（英語・単欄幅・Okabe–Ito 色・パネル A–C / a–f・既定 300 dpi **PNG+PDF**）。キャプション案: [`docs/planning/v7_phase2a_paper_figures.md`](../docs/planning/v7_phase2a_paper_figures.md)。通常時 **macOS** は Hiragino 優先 |
| `v7_phase1a_pilot_jsonl.py` | Phase I-A: **JSONL + ラベル**と最終層 `S_asym` の相関（`--demo` で合成特徴） |
| `run_mrmp_llm_judge_chunks.sh` | MRMP `windows.jsonl` に対し LLM 審判を **CHUNK×N_CHUNKS** で追記実行 |

データ例: [`data/v7_phase1a_pilot.jsonl`](data/v7_phase1a_pilot.jsonl)

人手なし代理（トークン数 × S_asym）: [`v7_phase1a_autoproxy.py`](v7_phase1a_autoproxy.py)

**実証ベースライン一括（Phase I-A v1）**: [`v7_empirical_run.py`](v7_empirical_run.py) — メタデータ＋パイロット JSONL＋autoproxy＋（HF 時）参照文の層別統計。事前登録: `docs/planning/v7_phase1a_empirical_prereg_v1.json`。

**Phase II-A（MRMP 実データ τ 掃引・R(τ)）事前登録**: [`docs/planning/v7_phase2a_prereg_v1.json`](../docs/planning/v7_phase2a_prereg_v1.json)（revision 1: `span_spec`・`s_asym_definition_revision`）。**複数 run 比較表**: [`v7_phase2a_compare_runs.py`](v7_phase2a_compare_runs.py)。**主解析 vs 補助 6 軸の τ 系列関連（探索的）**: [`v7_phase2a_primary_aux_tau_association.py`](v7_phase2a_primary_aux_tau_association.py)。**合成テンソルの α スイープ**（理論橋の数値側）: [`v7_phase2a_delay_sweep.py`](v7_phase2a_delay_sweep.py) `--alpha-list` — [v7_phase2a_numeric_tau_exp.md](../docs/planning/v7_phase2a_numeric_tau_exp.md)。集約・ブロック Frobenius: [`v7_phase2a_empirical.py`](v7_phase2a_empirical.py)。**実データ CLI**（`windows.jsonl`・gpt2・話者ブロック）: [`v7_phase2a_empirical_run.py`](v7_phase2a_empirical_run.py)（例: `--max-dialogues 2 --cpu` でスモーク）。入力行に 6 軸数値（`trust_ab` … `history_ba`）があれば、事前登録の補助解析どおり **`auxiliary_label_delay_coherence`**（軸ごとに主解析と同型の遅延積平均）を JSON に自動同梱する。**結果サマリ**（Var の局所最大・移動平均）: [`v7_phase2a_tau_summary.py`](v7_phase2a_tau_summary.py)（`--out-md` / **`--out-json`**。**n(τ)** 全表・変化境界・R_var 上位との併記。`auxiliary_label_delay_coherence` がある入力では `auxiliary_summary` も同梱。長大時は `--n-table-max-rows`）**対話別寄与の出力**（ブートストラップ用）: `v7_phase2a_empirical_run.py` に **`--export-contributions`**。**R_mean のブートストラップ CI**: [`v7_phase2a_tau_bootstrap.py`](v7_phase2a_tau_bootstrap.py)（入力 JSON に `contributions_by_tau` が必要）。**同一対話のペア差** `R(τ_a)−R(τ_b)` は **`--paired-diff 0,1`** のように指定。**本番ログ例（先頭 3146 窓・対話 30・B=4000）**: [`logs/v7_phase2a_mrmp_tau_n3146_with_contrib.json`](logs/v7_phase2a_mrmp_tau_n3146_with_contrib.json) · [`logs/v7_phase2a_mrmp_tau_n3146_bootstrap.json`](logs/v7_phase2a_mrmp_tau_n3146_bootstrap.json) · [`logs/v7_phase2a_mrmp_tau_n3146_bootstrap.md`](logs/v7_phase2a_mrmp_tau_n3146_bootstrap.md)。**一括再現**: [`run_phase2a_mrmp_tau.sh`](run_phase2a_mrmp_tau.sh)（`MAX_ROWS` / `OUT_PREFIX` で上書き可。`VALIDATE_STRICT=1` で [`v7_phase2a_bundle_validate.py`](v7_phase2a_bundle_validate.py) `--strict`。`GENERATE_PLOTS=1` で [`v7_phase2a_tau_plots.py`](v7_phase2a_tau_plots.py) により R_mean / R_var / n(τ) の PNG）。シミュレーション専用: [`v7_phase2a_delay_sweep.py`](v7_phase2a_delay_sweep.py)。

**MRMP 全文取得**: [`fetch_mrmp_corpus.py`](fetch_mrmp_corpus.py) — 公式 GitHub の浅い clone → `experiments/logs/mrmp_repo/`（`.gitignore`）。手順: [`docs/planning/v7_corpus_MRMP.md`](../docs/planning/v7_corpus_MRMP.md)。

**MRMP 実証用整形**: [`v7_mrmp_prepare.py`](v7_mrmp_prepare.py) — `windows.jsonl` / `dialogue_eval.jsonl` / `manifest.json` を `experiments/logs/mrmp_prepared/` に生成。サンプル: [`data/v7_mrmp_sample.jsonl`](data/v7_mrmp_sample.jsonl)。Frobenius 相関は [`v7_phase1a_pilot_jsonl.py`](v7_phase1a_pilot_jsonl.py) の `--mrmp-labels`（`--max-rows` で件数制限）。

**6 軸 LLM 審判**: [`v7_phase1a_llm_judge_six_axes.py`](v7_phase1a_llm_judge_six_axes.py) — `trust_ab`…`history_ba` を付与。`--demo`（決定論疑似）、`OPENAI_API_KEY` + `--provider openai`、または **`--provider hf_local`**（例: Swallow-7B-Instruct・MPS/CPU・[`v7_local_slm_m3_japanese_plan.md`](../docs/planning/v7_local_slm_m3_japanese_plan.md)）。続けて Frobenius 相関: `v7_phase1a_pilot_jsonl.py --jsonl <出力JSONL>`（`--demo` または HF）。

**審判 SLM 同士の一致（探索）**: [`v7_llm_judge_slm_pair_agreement.py`](v7_llm_judge_slm_pair_agreement.py)、[`run_local_slm_judge_pair_agreement.sh`](run_local_slm_judge_pair_agreement.sh) — 同一 `id` の審判済み JSONL 2 本から 12 軸の Pearson r・平均絶対差を JSON 化（手順は [`v7_local_slm_m3_japanese_plan.md`](../docs/planning/v7_local_slm_m3_japanese_plan.md) §10）。

**ローカル SLM スモーク**: [`v7_local_env_check.py`](v7_local_env_check.py)、[`run_local_slm_phase1_smoke.sh`](run_local_slm_phase1_smoke.sh)、[`run_local_slm_phase2_smoke.sh`](run_local_slm_phase2_smoke.sh)（Swallow `hf_local`・1 行）、[`run_local_slm_smoke_all.sh`](run_local_slm_smoke_all.sh)（Phase 1→bundle 検証→Phase 2）。Phase 1 ポインタのみの bundle: [`baselines/v7_local_slm_phase1_smoke_bundle_v1.json`](baselines/v7_local_slm_phase1_smoke_bundle_v1.json)（`v7_phase2a_bundle_validate.py` 非 strict）

**ローカル SLM 本番寄り**: [`run_local_slm_phase3_mrmp_chunk.sh`](run_local_slm_phase3_mrmp_chunk.sh)（`OFFSET` / `MAX_ROWS`・rinna 既定で `run_phase2a_mrmp_tau.sh` 相当）、[`run_local_slm_phase1_compare_en_ja.sh`](run_local_slm_phase1_compare_en_ja.sh)（gpt2 vs rinna 比較表）、[`run_mrmp_llm_judge_chunks_hf_local.sh`](run_mrmp_llm_judge_chunks_hf_local.sh)（審判 `hf_local` チャンク）、[`run_local_slm_phase4_judge_pair.sh`](run_local_slm_phase4_judge_pair.sh)（2 モデル審判→一致 JSON/MD・探索）、[`run_local_slm_phase4_swallow_7b_13b.sh`](run_local_slm_phase4_swallow_7b_13b.sh)（Swallow 7B vs 13B）。**チャンク凍結後の再現マニフェスト**は [`docs/planning/v7_local_slm_m3_japanese_plan.md`](../docs/planning/v7_local_slm_m3_japanese_plan.md) §11。`run_phase2a_mrmp_tau.sh` は **`OFFSET`・`MODEL`・`REVISION`・`CPU=1`** も受け付け。オペレータ草案 bundle: [`baselines/v7_local_slm_phase3_operator_bundle_v1.json`](baselines/v7_local_slm_phase3_operator_bundle_v1.json)。6 軸審判プロンプト正本: [`prompts/v7_llm_judge_prompt_v1.json`](prompts/v7_llm_judge_prompt_v1.json)

**大規模・チャンク**: `--offset` + `--max-rows` で入力スライス。`--resume --out-jsonl <同一ファイル>` で出力行数＝次の offset として**追記再開**。429/5xx は指数バックオフで `--max-retries`（既定 8）。レート緩和に `--sleep-after-request 秒`。

**一括（推奨）**: [`run_mrmp_llm_judge_chunks.sh`](run_mrmp_llm_judge_chunks.sh) — 既定で `CHUNK=1000`・`N_CHUNKS=10`（計 1 万行）。`bash experiments/run_mrmp_llm_judge_chunks.sh` または `CHUNK=500 N_CHUNKS=20 bash ...`。`--demo` で API なしスモーク。環境変数 `SRC` / `OUT` で入出力パスを変更可。

**JSONL 相関のスライス**: `v7_phase1a_pilot_jsonl.py` にも `--offset` / `--max-rows`（ストリーミング読み）。

**API キー（git 管理外）**: リポジトリ直下に [`.env.example`](../.env.example) をコピーして **`.env`** を作成し、`OPENAI_API_KEY=` を記入（`.env` / `.env.local` は `.gitignore`）。スクリプト実行時に [`local_env.py`](local_env.py) が自動で読み込む。

**n=400 実走の要約（相関のみ）**: [`baselines/v7_phase1a_llm_judge_n400_gpt2_correlations_v1.json`](baselines/v7_phase1a_llm_judge_n400_gpt2_correlations_v1.json)（詳細ログは `experiments/logs/`）。

## Phase II-A グラフ（PNG）の読み方

[`v7_phase2a_tau_plots.py`](v7_phase2a_tau_plots.py) が `*_with_contrib.json` から出す図の意味（事前登録の**コーパス代理**であり、理論の τ* そのものではない）。

| ファイル（例） | 内容 |
|------------------|------|
| `*_tau_primary.png` | **上**: 各 τ の **R_mean**（遅延 τ を変えたときの注意ベース積の対話平均）。**中**: **R_var**（対話間分散）と移動平均。**下**: **n(τ)**（その τ で計算に使えた対話数）。高 τ の R_var は **n が小さい**と膨らみうるので下段と併読。 |
| `*_tau_auxiliary_rvar.png` | 6 軸ラベルについて**主解析と同型**の遅延積系列の **R_var vs τ**（軸ごと）。注意ベースの主解析とは**別スケール・別対象**；探索的な対照用。 |

**論文・投稿用**は `python experiments/v7_phase2a_tau_plots.py <*_with_contrib.json> --paper` で `*_tau_paper_primary` / `*_tau_paper_auxiliary_rvar` の **PNG+PDF** を生成（詳細・英日キャプション案は上記 `v7_phase2a_paper_figures.md`）。

### Phase II-A：理論との「橋」・再現・シェル便利機能

- **理論境界と不足パート**（論文の主張の線引き）: [`docs/planning/v7_phase2a_theory_bridge.md`](../docs/planning/v7_phase2a_theory_bridge.md)（コーパス R(τ) と定理 3.3 の τ* / 設計書 τ*_exp の対応関係）。
- **再現・査読用チェックリスト**
  1. 環境: `requirements.txt`、GPU/CPU は `with_contrib.json` の `inference_device` に記録される。
  2. パイプライン: `bash experiments/run_phase2a_mrmp_tau.sh`（`WINDOWS` / `MAX_ROWS` / `OUT_PREFIX` を本文・補遺に明記）。
  3. 検証: `python experiments/v7_phase2a_bundle_validate.py --bundle <bundle.json> --strict --out-prefix "$OUT_PREFIX"`。
  4. マニフェスト（任意・補遺推奨）: `python experiments/v7_phase2a_repro_manifest.py --bundle <bundle.json> --out-prefix "$OUT_PREFIX" --out "${OUT_PREFIX}_repro_manifest.json"`。凍結時は同ファイルを論文補遺に添付し、査読後に `python experiments/v7_phase2a_repro_manifest.py --verify <manifest.json>` または `v7_phase2a_bundle_validate.py --strict ... --verify-manifest <manifest.json>` で照合。
  5. **CI**（ログなし）では `v7_phase2a_repro_manifest.py --pin-code-only` でパイプラインコードのハッシュを毎回記録できる。
- **`run_phase2a_mrmp_tau.sh`**: `GENERATE_PAPER_PLOTS=1` で論文用プロット。`WRITE_REPRO_MANIFEST=1` と **`BUNDLE_JSON=experiments/baselines/…_bundle_v1.json`** で `${OUT_PREFIX}_repro_manifest.json` を生成（図まで必須にする場合は `REPRO_MANIFEST_STRICT=1`）。

## レガシー Phase B/C の記録

- 下流・τ: `slm_downstream.py`、`squad_span.py`
- 資源・デコード: `hbm_budget_probe.py`、`phase3_claim_run.py`
- ベースライン JSON: `experiments/baselines/*.json`

ログの既定出力先は `experiments/logs/`（`.gitignore` 対象）が多い。
