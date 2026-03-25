# モジュール ↔ 実証スクリプト対応表（Phase A / A′）

実証ロードマップ **Phase A / A′**（軽量コア）および **Phase 1B**（文化的調製・SLM 橋）で検証する `core/` コンポーネントと、`experiments/` および `tests/` の対応である。

## 対応一覧

| `core` モジュール | 役割（要約） | 主な検証スクリプト | 備考 |
|-------------------|-------------|-------------------|------|
| `resonance.ResonanceEngine` | ノード部分集合×文脈の共鳴スコア（Softmax） | `evel_benchmarks`（ファサード経由）、`tests/test_smoke.py` | `LightweightResonanceFacade` 内で使用 |
| `roi_selector.DynamicROISelector` | 階層 ROI＋朧度によるカーネル | `efficiency_benchmark`（ROI 経路）、`evel_benchmarks` | ベースラインは全対全 matmul |
| `lightweight_resonance.LightweightResonanceFacade` | 埋め込み→(N,d) 場→ROI＋Engine | `emergence_metrics`, `evel_benchmarks`, `instrument_smoke`, `tests/test_smoke.py` | `forward(..., instrument=StageTimer)` で区間計測可。`cultural_scale`（Phase 1B） |
| `cultural_modulation` | 埋め込み→調製スカラー（あわい前処理のスタブ） | `tests/test_phase1b.py` | `CulturalModulationAdapter`, `awai_pressure_from_embeddings` |
| `instrumentation.StageTimer` | 前向き各区間の時間・CUDA 割り当て差 | `instrument_smoke`, `tests/test_instrumentation.py` | Phase A′ 計測フック |
| `resonant_core.ResonantCore` | 6 次元射影・場 W 更新・朧度 | `evel_benchmarks`, `tests/test_smoke.py`, `tests/test_instrumentation.py` | `forward(..., instrument=StageTimer)` 可 |
| `autopoiesis.AutopoieticInference` | 場の離散更新ループ | `evel_benchmarks` | `ToySeqModel` スタブ |
| `config_utils` | `configs/default_config.yaml` から kwargs | 各スクリプト（ResonantCore／Autopoiesis 初期化） | |

## 指標・ログの対応

| 実証の種類 | スクリプト | 出力（例） | 解釈の参照 |
|------------|-----------|------------|------------|
| スモーク・勾配通過 | `experiments/evel_benchmarks.py` | 標準出力 `smoke_ok` | 下記 `phase_a_reproduction.md` |
| 効率（時間・CUDA メモリ） | `experiments/efficiency_benchmark.py` | JSON | ROI 対全対全比はハード依存 |
| 創発（KL・新規性・アブレーション） | `experiments/emergence_metrics.py` | JSON + `*_prereg.yaml` | [創発の観測・利用 設計 v0.1](../implementation/🛠️%20共鳴場による創発の観測・利用%20設計%20v0.1.md) |
| 回帰（import・forward） | `pytest tests/` | コンソール | CI と同じ |
| Phase 1B（SLM 橋） | `experiments/slm_bridge_smoke.py` | 標準出力 `slm_bridge_ok` | `transformers` ＋ `AwaiIntegratedSLM` |
| Phase 2（最小学習） | `slm_resonance_lm.py` + `slm_data.py` | JSON または `slm_resonance_lm_ok` | `--data random` または `wikitext`、`--eval-ppl` で ppl |
| Phase 2（導入前後の速度・メモリ比較） | `slm_perf_compare.py`（内部で `slm_resonance_lm` を 2 回） | `slm_perf_compare_ok` + JSON | 同一 CLI で baseline／Awai を連続実行。指標の整理は [slm_performance_comparison.md](slm_performance_comparison.md) |
| Phase B プロトコル | [phase_b_data_protocol.md](phase_b_data_protocol.md) | 手順・指標定義 | Wikitext-2-raw、eval 分割、perplexity |
| 計測フック（区間時間） | `instrument_smoke.py` + `regression_check --mode instrument` | JSON（`--out`）、ベースライン [`instrument_cpu.json`](../../../tests/baselines/instrument_cpu.json) | 区間 `elapsed_s` の悪化検出 |
| 効率（短系列・回帰） | `efficiency_benchmark.py` + `regression_check.py` | JSON | ベースライン [`tests/baselines/efficiency_cpu_short.json`](../../../tests/baselines/efficiency_cpu_short.json)、`--max-slowdown` で悪化検出 |

## 未カバー（ロードマップ Phase B 本格以降）

| モジュール | 理由 |
|------------|------|
| `AwaiIntegratedSLM` の本番評価 | 最小学習は `slm_resonance_lm`。固定ベンチ・下流タスクは未 |
| 二階建て（Controller／Router／KV） | 実装・`benchmarks/` は Phase C |

## 改訂

| 日付 | 内容 |
|------|------|
| 2026-03-25 | 初版（Phase A′） |
| 2026-03-25 | Phase 1B（文化的調製・`slm_bridge_smoke`） |
| 2026-03-25 | Phase 2 入口（`slm_resonance_lm`） |
