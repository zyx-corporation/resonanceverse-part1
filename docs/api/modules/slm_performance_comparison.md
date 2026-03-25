# SLM 導入前後のパフォーマンス比較（整理メモ）

`experiments/slm_resonance_lm.py` と `experiments/slm_perf_compare.py` が出す **JSON** を、報告・再現用に整理する。

## 1. 何と何を比較しているか

| 側 | `integration`（JSON） | 実装 |
|----|------------------------|------|
| **導入前（ベースライン）** | `hf_baseline` | `--baseline-hf` — HF 因果 LM の語彙ロジットのみ（`HfCausalLMOnly`） |
| **導入後（Resonanceverse 統合）** | `awai_resonance` | 既定 — `AwaiIntegratedSLM`（HF ＋ `ResonantCore` ＋ `out_head`） |

**別系統のベンチ**: 軽量コアの ROI／全対全は `efficiency_benchmark.py`、区間時間は `instrument_smoke.py`。本稿は **因果 LM 最小学習ループ**に特化する。

## 2. 結果の出し方

| 方法 | 出力 |
|------|------|
| まとめ実行 | `python experiments/slm_perf_compare.py ... --out experiments/logs/slm_compare.json` → 先頭行 `slm_perf_compare_ok` ＋ マージ JSON |
| 個別実行 | `slm_resonance_lm.py` を `--baseline-hf` あり／なしで 2 回、`--out` で別ファイル |

`slm_perf_compare` は **同一 CLI 引数**を両者に渡す。ベースラインだけフル学習・Awai だけ `--freeze-base` など **非対称**にしたい場合は個別コマンドで実行する（[Phase A 再現手順](phase_a_reproduction.md) §9）。

## 3. 単発実行 JSON（`slm_resonance_lm_ok`）の主なフィールド

| フィールド | 型 | 意味 |
|------------|-----|------|
| `integration` | 文字列 | `hf_baseline` / `awai_resonance` / `demo_stub` |
| `model` | 文字列 | HF モデル名（例: `gpt2`） |
| `data` | 文字列 | `random` または `wikitext` |
| `max_steps` | 整数 | 学習ステップ数 |
| `freeze_base` | 真偽 | ベース LM を勾配オフにしたか |
| `final_loss` / `loss_start` | 数値 | 最終／最初のミニバッチ損失（次トークン CE） |
| `train_time_s` | 数値 | **学習ループ**の経過時間（秒） |
| `steps_per_sec` | 数値 | `max_steps / train_time_s` |
| `cuda_peak_memory_bytes` | 整数または null | CUDA 時、学習後のピーク割り当て（CPU では null） |
| `perplexity_train` / `perplexity_eval` | 数値または null | `--eval-ppl` かつ `wikitext` のとき |

## 4. マージ JSON（`slm_perf_compare_ok`）の `comparison` ブロック

| フィールド | 解釈 |
|------------|------|
| `train_time_ratio_baseline_over_awai` | **ベースラインの `train_time_s` ÷ Awai の `train_time_s`**。**1 より大きい**と Awai 側が同ステップ数を**より短時間**で終えた（スループット有利）。 |
| `throughput_ratio_awai_over_baseline` | **Awai の `steps_per_sec` ÷ ベースラインの `steps_per_sec`**。**1 より大きい**と Awai のステップレートが高い。上記の時間比と整合（同一実行では概ね逆数関係に近い）。 |
| `cuda_peak_memory_bytes` | `baseline` / `awai` のピークバイト。差分でオーバーヘッドの目安に。 |

数値は実行ごとに変動する。**同一マシン・同一 `seed`・同一 `--model`・`--batch`・`--seq-len`・`--max-steps`** で並べたときの相対比較が主用途。

## 5. 品質指標との読み分け

- **速度・メモリ**: `train_time_s`、`steps_per_sec`、`cuda_peak_memory_bytes`。
- **学習の当たり**: `final_loss`、`loss_start`（ランダムデータでは参考程度）。
- **データ固定の品質**: `wikitext` ＋ `--eval-ppl` の `perplexity_eval` / `perplexity_train`（定義は [Phase B データ・評価プロトコル](phase_b_data_protocol.md)）。

ベースラインが全パラメータ学習、Awai が `--freeze-base` のみ更新、といった **学習対象が異なる**設定では、損失・ppl は直接「優劣」にならない。**どちらを更新したか**を必ず記録する。

## 6. 記録テンプレート（コピー用）

実験ノートや PR にそのまま貼れる形。

~~~markdown
## 環境
- 日付:
- OS / GPU / ドライバ:
- PyTorch / CUDA:
- HF_HOME またはキャッシュパス:

## コマンド
    python experiments/slm_perf_compare.py --model ... --data ... --max-steps ... --seed ...

## サマリ（JSON から転記）

| 項目 | ベースライン (hf_baseline) | Awai (awai_resonance) |
|------|------------------------------|-------------------------|
| train_time_s | | |
| steps_per_sec | | |
| cuda_peak_memory_bytes (MiB 換算可) | | |
| final_loss | | |
| perplexity_eval (wikitext + --eval-ppl 時) | | |

## 比較（comparison）

| train_time_ratio_baseline_over_awai | throughput_ratio_awai_over_baseline |
|-------------------------------------|-------------------------------------|
| | |

## 注意
- freeze 方針（両方 / 片方のみ）:
- 解釈上の限界（ランダムデータのみ等）:
~~~

MiB 換算: `cuda_peak_memory_bytes / (1024*1024)`。

## 7. 関連

- [Phase A 再現手順](phase_a_reproduction.md)（コマンド例）
- [モジュール ↔ 実証スクリプト対応表](module_benchmark_map.md)
- [計測戦略の考察](measurement_strategy.md)（CPU 回帰ベースラインとの関係）
