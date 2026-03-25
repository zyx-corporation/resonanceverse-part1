# Phase A 再現手順（軽量コア実証）

## 環境

- Python 3.10+（CI は 3.11）
- 依存: `pip install -r requirements.txt`（リポジトリルート）
- GPU は任意。CPU のみでも全スクリプトは実行可能。

## 乱数シード

再現性のため、各スクリプトは **`--seed`**（またはデフォルト）で `core.reproducibility.set_experiment_seed` を呼ぶ。

**注意**: CUDA 上では演算順の非決定性により数値が完全には一致しない場合がある。比較には同一デバイス・同一バージョンを推奨。

## コマンド一覧

リポジトリルートで実行する。

### 1. スモーク（統合）

```bash
python experiments/evel_benchmarks.py --seed 0
```

期待: 終了時に `smoke_ok` と損失値が表示される。

### 2. 効率ベンチマーク

```bash
python experiments/efficiency_benchmark.py --cpu --seed 42 \
  --seq-lens 64 128 256 --repeats 5 \
  --out experiments/logs/efficiency_benchmark.json
```

### 3. 創発指標ログ

```bash
python experiments/emergence_metrics.py --seed 0 \
  --out experiments/logs/emergence_run.json
```

生成物: `emergence_run.json` と `emergence_run_prereg.yaml`（事前登録パラメータ）。

### 4. 計測フック（StageTimer）

`LightweightResonanceFacade.forward(..., instrument=timer)` に `core.instrumentation.StageTimer` を渡すと、embedding／pool／roi／engine 等の区間ごとに経過時間（CUDA 時は割り当てバイト差分も）が `timer.records` に溜まる。

```bash
python experiments/instrument_smoke.py --cpu --seed 0
# ファイル保存 + ベースライン比較:
python experiments/instrument_smoke.py --cpu --seed 0 --out experiments/logs/instrument.json
python experiments/regression_check.py --mode instrument \
  --baseline tests/baselines/instrument_cpu.json \
  --current experiments/logs/instrument.json \
  --max-slowdown 100
```

計測の意図・限界は [計測戦略の考察](measurement_strategy.md) を参照。

### 5. 単体テスト（CI と同等）

```bash
pip install pytest
pytest tests/ -v
```

CI では `instrument_smoke`、短系列の `efficiency_benchmark`、および `regression_check.py`（ベースライン JSON との比較）も実行される。

### 6. 効率ベンチの回帰検査

```bash
python experiments/efficiency_benchmark.py --cpu --seed 42 --seq-lens 32 64 --repeats 2 \
  --out experiments/logs/current_eff.json
python experiments/regression_check.py --mode efficiency \
  --baseline tests/baselines/efficiency_cpu_short.json \
  --current experiments/logs/current_eff.json \
  --max-slowdown 80
```

`mean_time_s` がベースラインの `--max-slowdown` 倍を超えて**悪化**していると非ゼロ終了。同一マシンで基準を取り直す場合は `tests/baselines/efficiency_cpu_short.json` を更新（PR 説明に理由を記載）。

instrument 用は `--mode instrument` と `tests/baselines/instrument_cpu.json` を用いる。

### 7. instrument 計測の回帰検査

`efficiency` と同様、`--mode instrument` で `instrument_smoke` の `stages[].elapsed_s` をベースラインと比較する。

### 8. Phase 1B（文化的調製・SLM 橋スモーク）

単体テスト:

```bash
pytest tests/test_phase1b.py -v
```

`transformers` が入っている環境で、因果 LM ＋ `AwaiIntegratedSLM` の前向き確認:

```bash
python experiments/slm_bridge_smoke.py --cpu --seed 0
```

初回は `gpt2` 等の重み取得でネットワークが必要な場合がある。

### 9. Phase 2 入口（最小学習ループ）

オフライン・CI 向けデモ（ミニスタブ、ネットワーク不要）:

```bash
python experiments/slm_resonance_lm.py --demo --max-steps 20 --cpu --freeze-base
```

実モデル（`transformers`、初回は重みダウンロードあり）:

```bash
python experiments/slm_resonance_lm.py --model gpt2 --max-steps 50 --cpu --freeze-base
```

Wikitext（要ネット初回）と perplexity:

```bash
python experiments/slm_resonance_lm.py --model gpt2 --data wikitext --max-steps 30 --cpu \
  --freeze-base --eval-ppl --max-chars 100000
```

**導入前（HF のみ）と導入後（AwaiIntegratedSLM）のパフォーマンス比較**（同一 `--model`・`--data`・ステップ数など）:

- ベースライン: `--baseline-hf`（`ResonantCore` なし、HF 因果 LM のロジットのみ）。
- 統合: フラグなし（既定の `AwaiIntegratedSLM`）。
- JSON に `train_time_s`・`steps_per_sec`・CUDA 時は `cuda_peak_memory_bytes` が含まれる。

```bash
# 個別実行の例（GPU: --cpu を外す）
python experiments/slm_resonance_lm.py --model gpt2 --data random --max-steps 200 --batch 8 \
  --seq-len 64 --baseline-hf --out experiments/logs/slm_baseline.json
python experiments/slm_resonance_lm.py --model gpt2 --data random --max-steps 200 --batch 8 \
  --seq-len 64 --freeze-base --out experiments/logs/slm_awai.json
```

まとめて比較する場合（`--freeze-base` は両実行に効く。ベースラインのみ学習させたいときは片方を個別コマンドで実行）:

```bash
python experiments/slm_perf_compare.py --model gpt2 --data random --max-steps 200 \
  --batch 8 --seq-len 64 --out experiments/logs/slm_compare.json
```

`--baseline-hf` に `--freeze-base` を付けるとベース全体が固定され学習が実質止まるため、ベースライン側は通常 **freeze なし**、統合側は **`--freeze-base`** といった**意図的に異なる**学習設定で比較することが多い。

比較結果の指標の意味・JSON の読み方・記録用テンプレートは [SLM 導入前後のパフォーマンス比較](slm_performance_comparison.md) に整理してある。

手順の固定化は [Phase B データ・評価プロトコル](phase_b_data_protocol.md) を参照。

## 結果の解釈（Phase A）

| 観点 | 見るファイル／値 | メモ |
|------|-------------------|------|
| 共鳴・ROI が動く | `evel_benchmarks` が異常終了しない | アサート通過 |
| 区間計測 | `instrument_smoke` の `stages[].elapsed_s` | ボトルネック把握用。CUDA 時は `cuda_allocated_delta_bytes` |
| 効率 | `efficiency_benchmark.json` の `time_ratio_full_over_roi` | CPU 短系列では ROI オーバーヘッドで 1 未満になりうる。長系列・GPU で傾向確認 |
| 創発 | `kl_p_q`, `mean_novelty_*`, アブレーション | シャッフルで指標がどう変わるかを比較（設計書の反証プロトコル） |

## 関連

- [Phase B データ・評価プロトコル](phase_b_data_protocol.md)（Wikitext／perplexity）
- [モジュール↔実験対応表](module_benchmark_map.md)
- [Phase A′ の締めと CI 蓄積の考察](phase_a_prime_closure_and_ci_accumulation.md)（完了基準・CI が蓄積する情報の区別）
- [実証ロードマップ（全体）](../planning/Resonanceverse実証ロードマップ_軽量コアとSLM二階建て.md)
