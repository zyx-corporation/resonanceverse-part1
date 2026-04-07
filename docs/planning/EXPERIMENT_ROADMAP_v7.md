# 実験ロードマップ v7.0（理論正本との対応）

**正本**: [Resonanceverse v7.0 実証実験設計書](../v7/Resonanceverse_v7.0_Experimental_Design.md)

本文書は **Phase I〜IV** と本リポジトリの **スクリプト** の対応を示す。レガシー（Phase A〜C）は [ROADMAP_ja.md](../../ROADMAP_ja.md)。

## 実証実験（Phase I-A ベースライン v1）

設計書の本番規模（数千ターン・人手アノテ）に先立ち、**パイロット JSONL × HF** で再現可能な実走ログを 1 JSON にまとめる。事前登録スタブ: [`v7_phase1a_empirical_prereg_v1.json`](v7_phase1a_empirical_prereg_v1.json)。手順の位置づけ: [Phase_IA_IIIA_本番手順設計_v7](Phase_IA_IIIA_本番手順設計_v7.md)「リポジトリ内実証ベースライン」。

```bash
python experiments/v7_empirical_run.py --demo --out experiments/logs/v7_empirical/bundle_demo.json
python experiments/v7_empirical_run.py --cpu --model gpt2 --seed 0 --out experiments/logs/v7_empirical/bundle_hf.json
```

ベースライン例（demo）: [`experiments/baselines/v7_empirical_bundle_demo_v1.json`](../../experiments/baselines/v7_empirical_bundle_demo_v1.json)

**本番コーパス（v7）**: **MRMP のみ** — 取得・仕様・6 軸との境界は [v7_corpus_MRMP.md](v7_corpus_MRMP.md)。HF: `nu-dialogue/multi-relational-multi-party-chat-corpus`。

**整形**: `python experiments/v7_mrmp_prepare.py` → `experiments/logs/mrmp_prepared/windows.jsonl` 等。サンプル行: [`experiments/data/v7_mrmp_sample.jsonl`](../../experiments/data/v7_mrmp_sample.jsonl)。相関は `v7_phase1a_pilot_jsonl.py --mrmp-labels`。

**整形後の健全性（Day 0）**: `python experiments/rvt_exp_2026_008_day0_checks.py --strict-manifest`（必須キー・行数・`manifest.json` の `n_utterance_rows` 一致）。外部計画 RVT-EXP-2026-008 と本リポ実装の対応は [rvt_exp_2026_008_architecture_bridge.md](rvt_exp_2026_008_architecture_bridge.md)。

**RVT L1 拡張（MRMP 複数行・任意 Awai 蓄積）**: `python experiments/rvt_exp_2026_008_mrmp_row.py --jsonl … --line N --max-rows K` または [`run_rvt_mrmp_batch.sh`](../../experiments/run_rvt_mrmp_batch.sh) / ワンショット [`run_rvt_explore.sh`](../../experiments/run_rvt_explore.sh)（`JSONL` / `LINE` / `MAX_ROWS` / `MODEL` / `ACCUMULATE_AWAI`・[架け橋 §5](rvt_exp_2026_008_architecture_bridge.md)）。

**6 軸を LLM で付ける場合**: 事前登録フィールド [`v7_phase1a_empirical_prereg_v1.json`](v7_phase1a_empirical_prereg_v1.json) の `llm_judge_six_axes`、手順は [Phase_IA_IIIA_本番手順設計_v7](Phase_IA_IIIA_本番手順設計_v7.md)「A.3′」。実行: [`../../experiments/v7_phase1a_llm_judge_six_axes.py`](../../experiments/v7_phase1a_llm_judge_six_axes.py)（`--demo` または `OPENAI_API_KEY`）。

## 実行コマンド（一括）

```bash
# 軽量デモ（CI 向け・JSON 出力）
python experiments/v7_run_suite.py --demo --out experiments/logs/v7_suite/suite.json

# フル（時間がかかる）
python experiments/v7_run_suite.py --out experiments/logs/v7_suite/suite_full.json
```

デモ結果の要約 JSON 例: [`experiments/baselines/v7_suite_bundle_demo_v1.json`](../../experiments/baselines/v7_suite_bundle_demo_v1.json)

## 最小の「次」（HF ベースライン・Phase I-A）

合成デモのあと、**実モデルで層別 `S_asym` 統計**だけ取る場合（人手ラベル不要）:

```bash
python experiments/v7_phase1a_phi_correlation.py --cpu --model gpt2 --seed 0 \
  --text "Hello, we need to align on the deadline before Friday." \
  --out experiments/logs/v7_suite/phase1a_hf_gpt2_stats.json
```

`GPT2LMHead` 系は **`attn_implementation=\"eager\"`** で `output_attentions` を取得する（スクリプト内で指定済み）。要約: [`experiments/baselines/v7_phase1a_hf_gpt2_summary_v1.json`](../../experiments/baselines/v7_phase1a_hf_gpt2_summary_v1.json)

### JSONL パイロット（例示ラベル × Frobenius）

短い対話＋数値ラベル（`intent_ab`, `trust_ab` 等）を [`experiments/data/v7_phase1a_pilot.jsonl`](../../experiments/data/v7_phase1a_pilot.jsonl) に置き、相関を出す。

```bash
# 合成特徴（HF 不要）
python experiments/v7_phase1a_pilot_jsonl.py --demo --out experiments/logs/v7_suite/pilot_demo.json

# 本番に近い（gpt2・最終層・各 text 前向き）
python experiments/v7_phase1a_pilot_jsonl.py --cpu --model gpt2 --seed 0 \
  --jsonl experiments/data/v7_phase1a_pilot.jsonl \
  --out experiments/logs/v7_suite/pilot_hf.json
```

要約（demo）: [`experiments/baselines/v7_phase1a_pilot_demo_summary_v1.json`](../../experiments/baselines/v7_phase1a_pilot_demo_summary_v1.json)

### 人手なし（代理変数）

トークン数（または `--demo` 時は文字長）と最終層 `||S_asym||_F` の相関。**6 軸の代替ではない**（[Phase_IA_IIIA_本番手順設計_v7](Phase_IA_IIIA_本番手順設計_v7.md)「人手を省略する場合」参照）。

```bash
python experiments/v7_phase1a_autoproxy.py --demo --out experiments/baselines/v7_phase1a_autoproxy_demo.json
python experiments/v7_phase1a_autoproxy.py --cpu --model gpt2 --seed 0 --out experiments/logs/v7_suite/autoproxy_hf.json
```

## 対応表（実装済みハーネス）

| v7 | 検証の核 | スクリプト |
|----|----------|------------|
| **Phase I-A** | S_asym と 6 軸（または代理特徴）の相関 | [`experiments/v7_phase1a_phi_correlation.py`](../../experiments/v7_phase1a_phi_correlation.py)（`--demo`＝合成検証；実モデルは `--demo` なしで層別 S_asym 統計） |
| **Phase I-B** | 有向テンソルが対称化に落ちない | [`experiments/v7_phase1b_directed_tensor.py`](../../experiments/v7_phase1b_directed_tensor.py) |
| **Phase II-A** | τ 掃引・安定性プロキシ | [`experiments/v7_phase2a_delay_sweep.py`](../../experiments/v7_phase2a_delay_sweep.py) |
| **Phase III（合成）** | あわい Ω の数値出力 | [`experiments/v7_phase3a_awai_metrics.py`](../../experiments/v7_phase3a_awai_metrics.py) |
| **Phase III-A（本番）** | 人間「間合い」アノテとの相関 | **未着手**（コーパス・アノテが必要） |
| **Phase IV** | 方式 B 統合 | 既存: `AwaiIntegratedSLM`・`decode_benchmark`・`two_tier_sweep` 等。**最小バンドル**: [`v7_phase4_minimal_repro.py`](../../experiments/v7_phase4_minimal_repro.py)（[`v7_phase4_integration_repro.md`](v7_phase4_integration_repro.md)） |
| **Phase II-A 実データ（感度）** | 複数 run 比較・主–補助 τ 系列 | [`v7_phase2a_compare_runs.py`](../../experiments/v7_phase2a_compare_runs.py)、[`v7_phase2a_primary_aux_tau_association.py`](../../experiments/v7_phase2a_primary_aux_tau_association.py) |
| **Phase II-A 合成（μ 感度）** | alpha スイープ・理論橋・Lyapunov・論文表 | [`v7_phase2a_delay_sweep.py`](../../experiments/v7_phase2a_delay_sweep.py) `--alpha-list`（[数値 τ 説明](v7_phase2a_numeric_tau_exp.md)）。束ね: [`v7_phase2a_theory_bridge_synth.py`](../../experiments/v7_phase2a_theory_bridge_synth.py)。V 代理: [`v7_phase2a_tau_exp_lyapunov_stub.py`](../../experiments/v7_phase2a_tau_exp_lyapunov_stub.py)。比較表: [`v7_phase2a_paper_tau_comparison.py`](../../experiments/v7_phase2a_paper_tau_comparison.py) |
| **Phase I-A（MRMP 6 軸）** | 審判 JSONL × Frobenius | [`run_phase1a_mrmp_v7_axes.sh`](../../experiments/run_phase1a_mrmp_v7_axes.sh)、バンドル [`v7_phase1a_mrmp_judge_v7_axes_bundle_v1.json`](../../experiments/baselines/v7_phase1a_mrmp_judge_v7_axes_bundle_v1.json) |
| **ローカル SLM（日本語）** | M3 想定の注意・審判・チャンク・SLM 同士一致・再現マニフェスト §11 | [v7_local_slm_m3_japanese_plan.md](v7_local_slm_m3_japanese_plan.md)、[`v7_llm_judge_slm_pair_agreement.py`](../../experiments/v7_llm_judge_slm_pair_agreement.py)、[`run_local_slm_phase4_judge_pair.sh`](../../experiments/run_local_slm_phase4_judge_pair.sh)、[`run_local_slm_phase4_swallow_7b_13b.sh`](../../experiments/run_local_slm_phase4_swallow_7b_13b.sh) |
| **コーパス Day 0（MRMP 窓）** | `windows.jsonl` スキーマ・件数 | [`experiments/rvt_exp_2026_008_day0_checks.py`](../../experiments/rvt_exp_2026_008_day0_checks.py)；計画書対応表 [rvt_exp_2026_008_architecture_bridge.md](rvt_exp_2026_008_architecture_bridge.md) |

## レガシー実証との関係

| レガシー | v7 との関係 |
|----------|-------------|
| Phase A〜A′ | I-B の最小動作＋ CI 回帰 |
| Phase B / C / 3 | 品質・資源の記録（方式 B 全体とは別軸） |
| Phase 4 | 分散ロードマップ（v7 PoP 議論とは別） |

## テスト

`tests/test_v7_experiments.py` が `--demo` 相当の関数をオフライン検証する。MRMP Day 0 検証ロジックは `tests/test_rvt_exp_2026_008_day0_checks.py`。

---

*改訂時は v7 実験設計書の版と同期すること。*
