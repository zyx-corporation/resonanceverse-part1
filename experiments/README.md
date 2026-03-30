# `experiments/` — v7 実験設計との対応

**正本**: [v7.0 実証実験設計書](../docs/v7/Resonanceverse_v7.0_Experimental_Design.md) · [EXPERIMENT_ROADMAP_v7.md](../docs/planning/EXPERIMENT_ROADMAP_v7.md)

## v7 一括実行

```bash
python experiments/v7_run_suite.py --demo --out experiments/logs/v7_suite/suite.json
```

要約 JSON（デモ）: [`baselines/v7_suite_bundle_demo_v1.json`](baselines/v7_suite_bundle_demo_v1.json)

最小の次（**gpt2・層別 S_asym 統計・ラベル無し**）: [`baselines/v7_phase1a_hf_gpt2_summary_v1.json`](baselines/v7_phase1a_hf_gpt2_summary_v1.json)

## v7 フェーズ別スクリプト

| スクリプト | 内容 |
|------------|------|
| `v7_run_suite.py` | I-A・I-B・II-A・III-A（合成）を連続実行 |
| `v7_phase1a_phi_correlation.py` | Phase I-A: S_asym と相関（`--demo`＝合成） |
| `v7_phase1b_directed_tensor.py` | Phase I-B: 有向テンソル非対称性 |
| `v7_phase2a_delay_sweep.py` | Phase II-A: τ 掃引 |
| `v7_phase3a_awai_metrics.py` | あわい Ω（合成軌跡） |
| `v7_phase1a_pilot_jsonl.py` | Phase I-A: **JSONL + ラベル**と最終層 `S_asym` の相関（`--demo` で合成特徴） |

データ例: [`data/v7_phase1a_pilot.jsonl`](data/v7_phase1a_pilot.jsonl)

人手なし代理（トークン数 × S_asym）: [`v7_phase1a_autoproxy.py`](v7_phase1a_autoproxy.py)

**実証ベースライン一括（Phase I-A v1）**: [`v7_empirical_run.py`](v7_empirical_run.py) — メタデータ＋パイロット JSONL＋autoproxy＋（HF 時）参照文の層別統計。事前登録: `docs/planning/v7_phase1a_empirical_prereg_v1.json`。

**MRMP 全文取得**: [`fetch_mrmp_corpus.py`](fetch_mrmp_corpus.py) — 公式 GitHub の浅い clone → `experiments/logs/mrmp_repo/`（`.gitignore`）。手順: [`docs/planning/v7_corpus_MRMP.md`](../docs/planning/v7_corpus_MRMP.md)。

**MRMP 実証用整形**: [`v7_mrmp_prepare.py`](v7_mrmp_prepare.py) — `windows.jsonl` / `dialogue_eval.jsonl` / `manifest.json` を `experiments/logs/mrmp_prepared/` に生成。サンプル: [`data/v7_mrmp_sample.jsonl`](data/v7_mrmp_sample.jsonl)。Frobenius 相関は [`v7_phase1a_pilot_jsonl.py`](v7_phase1a_pilot_jsonl.py) の `--mrmp-labels`（`--max-rows` で件数制限）。

**6 軸 LLM 審判**: [`v7_phase1a_llm_judge_six_axes.py`](v7_phase1a_llm_judge_six_axes.py) — `trust_ab`…`history_ba` を付与。`--demo`（決定論疑似）または `OPENAI_API_KEY` + `--provider openai`。続けて Frobenius 相関: `v7_phase1a_pilot_jsonl.py --jsonl <出力JSONL>`（`--demo` または HF）。

**API キー（git 管理外）**: リポジトリ直下に [`.env.example`](../.env.example) をコピーして **`.env`** を作成し、`OPENAI_API_KEY=` を記入（`.env` / `.env.local` は `.gitignore`）。スクリプト実行時に [`local_env.py`](local_env.py) が自動で読み込む。

**n=400 実走の要約（相関のみ）**: [`baselines/v7_phase1a_llm_judge_n400_gpt2_correlations_v1.json`](baselines/v7_phase1a_llm_judge_n400_gpt2_correlations_v1.json)（詳細ログは `experiments/logs/`）。

## レガシー Phase B/C の記録

- 下流・τ: `slm_downstream.py`、`squad_span.py`
- 資源・デコード: `hbm_budget_probe.py`、`phase3_claim_run.py`
- ベースライン JSON: `experiments/baselines/*.json`

ログの既定出力先は `experiments/logs/`（`.gitignore` 対象）が多い。
