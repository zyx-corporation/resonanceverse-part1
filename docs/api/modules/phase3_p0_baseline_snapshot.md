# Phase 3 P0：主張バンドル（実行手順と表の埋め方）

[Phase 3 実測と主張完成計画（採用済み）](../../planning/Phase3_実測と主張完成計画_ja.md) の **§2 A・B**（`two_tier_sweep` + `hbm_budget_probe` + メタ）を**同一コマンド**で残す手順である。

## 公式コマンド（採用済み）

オフライン・CI 向け（`transformers` の重み取得なし）:

```bash
python experiments/phase3_claim_run.py --demo --cpu --seed 0 \
  --max-new-tokens 16 --repeats 10 --seq-len 128 --batch-size 1 \
  --out-dir experiments/logs/phase3_claim
```

実モデル（初回は Hugging Face から取得。GPU 利用時は `--cpu` を外す）:

```bash
python experiments/phase3_claim_run.py --model gpt2 --seed 0 \
  --max-new-tokens 16 --repeats 10 --seq-len 128 --batch-size 1 \
  --out-dir experiments/logs/phase3_claim_gpt2
```

**GPU で再計測する**には **`--cpu` を付けない**こと。

- **NVIDIA + CUDA**: `two_tier_sweep` / `hbm_budget_probe` / `phase3_claim_run` は `cuda` を使用（`meta.cuda`）。
- **Mac（Apple Silicon）**: `torch.backends.mps.is_available()` が真なら **MPS** を使用。レイテンシは `torch.mps.synchronize()` で同期。ピーク VRAM は CUDA と同等の JSON 列は **null** になり得る。詳細は [measurement_mps_mac.md](measurement_mps_mac.md)。

デバイス選択は [`core/inference_device.py`](../../../core/inference_device.py) に集約されている。

## 実モデル例（`gpt2`・表用サマリー）

次は **同一条件**で実モデルを 1 回走らせた結果の要約である（**当該マシン・CPU 上の値**であり、再現時は相対比を主とする）。

| 項目 | 値 |
|------|-----|
| コマンド | 上記 `--out-dir experiments/logs/phase3_claim_gpt2` |
| `torch` / `transformers` | 2.9.1 / 4.55.2（要約 JSON に記載） |
| baseline `latency_ms_p50` / `p95` | 11.51 / 13.19 |
| two_tier_stub `latency_ms_p50` / `p95` | 11.99 / 13.67 |
| `latency_ms_p50_ratio_baseline_over_two_tier` | 0.960 |
| HBM `total_act_bytes_estimated` | 52690944 |

リポジトリに置いた要約（ホスト名なし）: [`experiments/baselines/phase3_claim_gpt2_summary_v1.json`](../../../experiments/baselines/phase3_claim_gpt2_summary_v1.json)。完全な `phase3_claim_bundle.json` は `experiments/logs/phase3_claim_gpt2/` に生成される（`.gitignore`）。

## 実モデル例（`gpt2`・**MPS**・表用サマリー）

Mac（Apple Silicon）で **`--cpu` なし**かつ `mps.is_available()` のとき、`device` は **`mps`**。次は同一条件（`seed`・系列長等）で **MPS 上**に 1 回計測した結果である。`hbm_proxy_cuda_peak_bytes` は **null**（[measurement_mps_mac.md](measurement_mps_mac.md)）。

| 項目 | 値 |
|------|-----|
| コマンド | `--out-dir experiments/logs/phase3_claim_mps`（他は上記実モデルと同じ） |
| `meta.mps.available` | true |
| baseline `latency_ms_p50` / `p95` | 8.29 / 11.40 |
| two_tier_stub `latency_ms_p50` / `p95` | 9.58 / 15.12 |
| `latency_ms_p50_ratio_baseline_over_two_tier` | 0.865 |
| HBM `total_act_bytes_estimated` | 52690944 |

要約 JSON: [`experiments/baselines/phase3_claim_gpt2_mps_summary_v1.json`](../../../experiments/baselines/phase3_claim_gpt2_mps_summary_v1.json)。

## 実験結果の評価（読み方）

### 主張の範囲

P0 の出力は **baseline と two_tier_stub の資源・時間ログ**である（[phase_c_quality_tau_prereg.md](phase_c_quality_tau_prereg.md) の P0 との関係）。この結果**だけ**では **言語品質・perplexity・下流精度の優劣は判定しない**。品質閾値 τ（T2〜T4）は `slm_resonance_lm` / `slm_downstream` / `squad_span` 等の**別実験**で適用する。

### レイテンシ比 `latency_ms_p50_ratio_baseline_over_two_tier`

定義は **baseline の p50 ÷ two_tier_stub の p50**（[`phase_c_two_tier_sweep.md`](phase_c_two_tier_sweep.md)、実装は `experiments/two_tier_sweep.py`）。上記の記録例では **比 &lt; 1** であるため、**当該スタブ設定では baseline の 1 ステップあたりが速い**（ms が小さい）。two_tier_stub は各ステップで Controller・Router の**追加計算**が入るため、**ブロック省略の効果よりオーバーヘッドが支配的**になり得る。

**注意**: これは「二階建てが一般に不利」という意味ではない。**Router スタブの挙動**（乱数・`keep` の分布など）に依存する。安全な主張は「**現行スタブ・この条件下では**オーバーヘッドがレイテンシに表れている」までである。

### CPU と MPS の絶対値

同一 `gpt2`・同一 `seed` でも **CPU と MPS で ms は直接並べない**（実行デバイスが異なる）。**同一マシン内の相対**（baseline vs two_tier）と、メタに **device・torch 版・日時**を残すことが優先である。

### HBM `total_act_bytes_estimated`

記録例では CPU / MPS で **合算値が一致**し得る（同一モデル・同一系列長・フック集計のため）。解釈は引き続き **絶対真値ではなく、テンプレ行への相対比較・説明用**（各 JSON の `disclaimer`）である。

### 品質 τとの関係

- **T1** はデコード比較の**棄却閾値としては未使用**（比率は報告用）。
- P0 のみ実行した場合、**T2〜T4 は未適用**（perplexity / accuracy / EM を測っていない）。

### 次の論点（要約）

資源表としてのログは揃う一方、**等品質の主張**は別パイプラインが必要である。二階建ての優位性を論じるには、Router 方針・実効スキップ・**τ を満たす条件**での再測定が後続となる。

## 出力（`experiments/logs/` は .gitignore）

| ファイル | スキーマ | 内容 |
|----------|----------|------|
| `phase3_claim_meta.json` | `phase3_claim_meta.v1` | 日時（UTC）、プラットフォーム、`python` / `torch` / `transformers`、`cuda`、`mps`、実験パラメータ |
| `phase3_claim_bundle.json` | `phase3_claim_bundle.v1` | 上記メタ + `two_tier_sweep` + `hbm_budget`（+ 任意で `squad_span`） |

標準出力に `phase3_claim_run_ok` と `out_dir` が出れば成功である。

## 表に転記する際の最小フィールド

- **メタ**: `generated_at_utc`、`versions`、`experiment`（`seed`・`model`・`demo`・系列長・バッチ）
- **two_tier_sweep**: `baseline` / `two_tier_stub` の `latency_ms_p50`・`latency_ms_p95`、`comparison.latency_ms_p50_ratio_baseline_over_two_tier`
- **hbm_budget**: `total_act_bytes_estimated`、代表 `rows[].layer` と `act_b`（デモはスタブ行）

数値は**マシン依存**である。対外報告では相対比と同一マシンである旨を明記する（[Phase 3 計画 §5](../../planning/Phase3_計画_二階建てと実証.md)）。

## P1（品質 τ）との関係

資源比較（P0）と「等品質」判定（τ）は別軸である。登録表と適用関係は [phase_c_quality_tau_prereg.md](phase_c_quality_tau_prereg.md) を参照する。

査読・投稿用に主張を一段まとめた文書は [Resonanceverse_主張表_論文用_ja.md](../../planning/Resonanceverse_主張表_論文用_ja.md) を参照する。

## Phase 3（続き）: P2・P3（任意）

[Phase3 実測と主張完成計画 §3](../../planning/Phase3_実測と主張完成計画_ja.md) の次優先である。P0・P1 が済んでいれば、ここから先を進める。

### P2 — `squad_span` をバンドルに同梱

オフライン（合成 `run_demo`・CI と同型）:

```bash
python experiments/phase3_claim_run.py --demo --cpu --seed 0 \
  --max-new-tokens 16 --repeats 10 --seq-len 128 --batch-size 1 \
  --with-squad --squad-demo \
  --out-dir experiments/logs/phase3_claim_p2_demo
```

`--demo` 時は `--squad-demo` が暗黙で真になり得る。実データ＋ HF は `--with-squad` のみ（`distilbert`、初回はネットワークあり）。**τ T4**（抽出型 QA の EM 下限）を同梱報告するときに参照する。

### P3 — Router の決定的モード（`step_stride`）

`two_tier_stub` 側で **ステップ番号 mod N == 0 のときだけ `keep=True`**（乱数ではなく決定的）:

```bash
python experiments/two_tier_sweep.py --model gpt2 --cpu --seed 0 \
  --max-new-tokens 16 --repeats 10 --router-step-stride 4 \
  --out experiments/logs/two_tier_stride4.json
```

`phase3_claim_run` にも **`--router-step-stride N`** を付与できる。挙動の説明は [Phase3 実測計画 §7](../../planning/Phase3_実測と主張完成計画_ja.md) と `core/two_tier/stubs.py` を参照。

### 実施済み（デモ・リポジトリ要約）

| 項目 | 内容 |
|------|------|
| **P2** | [`experiments/baselines/phase3_claim_p2_demo_with_squad_summary_v1.json`](../../../experiments/baselines/phase3_claim_p2_demo_with_squad_summary_v1.json)（`squad_span` 同梱・合成 `run_demo`） |
| **P3** | [`experiments/baselines/two_tier_sweep_stride4_demo_summary_v1.json`](../../../experiments/baselines/two_tier_sweep_stride4_demo_summary_v1.json)（`--router-step-stride 4`・`router_stub_mode`: `step_stride`・`router_keep_fraction_mean`: 0.25） |

実モデル（`gpt2`）での P3 再計測は、上記 P3 コマンドで `--demo` を外し `--model gpt2 --cpu` 等とすればよい。

## 改訂履歴

| 日付 | 内容 |
|------|------|
| 2026-03-25 | 初版（P0 手順のリポジトリ固定、§2 完了のための索引） |
| 2026-03-25 | 実モデル `gpt2` 1 回実行の表用サマリー（`experiments/baselines/phase3_claim_gpt2_summary_v1.json`） |
| 2026-03-25 | CUDA GPU 再計測の条件（`--cpu` なし・MPS 未対応）を追記 |
| 2026-03-25 | Apple MPS 対応と [measurement_mps_mac.md](measurement_mps_mac.md) への参照 |
| 2026-03-25 | `gpt2`・MPS 計測例（`phase3_claim_gpt2_mps_summary_v1.json`） |
| 2026-03-25 | 実験結果の評価（読み方）節を追記 |
| 2026-03-25 | P2・P3（任意）の手順節を追記 |
| 2026-03-25 | P2・P3 デモ実施の要約 JSON（`experiments/baselines/*_p2_*`・`two_tier_sweep_stride4_*`） |
| 2026-03-25 | [Resonanceverse_主張表_論文用_ja.md](../../planning/Resonanceverse_主張表_論文用_ja.md) へのリンク（P1 節） |
