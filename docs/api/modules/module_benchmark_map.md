# モジュール ↔ 実証スクリプト対応表（Phase A / A′）

実証ロードマップ **Phase A / A′**（軽量コア）および **Phase 1B**
（文化的調製・SLM 橋）で検証する `core/` コンポーネントと、`experiments/`
および `tests/` の対応である。

## 対応一覧

| `core` モジュール | 役割（要約） | 主な検証スクリプト | 備考 |
|-------------------|-------------|-------------------|------|
| `resonance.ResonanceEngine` | ノード部分集合×文脈の共鳴スコア | `evel_benchmarks`、`tests/test_smoke.py` | `LightweightResonanceFacade` 内で使用 |
| `roi_selector.DynamicROISelector` | 階層 ROI＋朧度カーネル | `efficiency_benchmark`、`evel_benchmarks` | ベースラインは全対全 matmul |
| `lightweight_resonance.LightweightResonanceFacade` | 埋め込み→場→ROI＋Engine | `emergence_metrics`, `evel_benchmarks` | `instrument=StageTimer` 可 |
| `cultural_modulation` | 埋め込み→調製スカラー（スタブ） | `tests/test_phase1b.py` | `CulturalModulationAdapter` |
| `instrumentation.StageTimer` | 前向き各区間の時間・CUDA 差分 | `instrument_smoke`, `tests/test_instrumentation.py` | Phase A′ 計測フック |
| `resonant_core.ResonantCore` | 6 次元射影・場 W 更新・朧度 | `evel_benchmarks`, `tests/test_smoke.py` | `instrument=StageTimer` 可 |
| `autopoiesis.AutopoieticInference` | 場の離散更新ループ | `evel_benchmarks` | `ToySeqModel` スタブ |
| `config_utils` | `configs/default_config.yaml` から kwargs | 各スクリプト（ResonantCore／Autopoiesis 初期化） | |

## 指標・ログの対応

| 実証の種類 | スクリプト | 出力（例） | 解釈の参照 |
|------------|-----------|------------|------------|
| スモーク・勾配通過 | `experiments/evel_benchmarks.py` | 標準出力 `smoke_ok` | 下記 `phase_a_reproduction.md` |
| 効率（時間・CUDA メモリ） | `experiments/efficiency_benchmark.py` | JSON | ROI 対全対全比はハード依存 |
| 創発（KL・新規性・アブレーション） | `experiments/emergence_metrics.py` | JSON + `*_prereg.yaml` | [創発の観測・利用 設計](../implementation/🛠️%20共鳴場による創発の観測・利用%20設計%20v0.1.md) |
| 回帰（import・forward） | `pytest tests/` | コンソール | CI と同じ |
| Phase 1B（SLM 橋） | `experiments/slm_bridge_smoke.py` | 標準出力 `slm_bridge_ok` | `transformers` ＋ `AwaiIntegratedSLM` |
| Phase 2（最小学習） | `slm_resonance_lm.py` + `slm_data.py` | JSON または `slm_resonance_lm_ok` | `--data random` または `wikitext` |
| Phase 2（速度・メモリ比較） | `slm_perf_compare.py` | `slm_perf_compare_ok` + JSON | baseline／Awai を連続実行 |
| Phase 2（下流・分類／QA 二値入口） | `slm_downstream.py` | `slm_downstream_ok` + JSON | `--task sst2`／`boolq` または `--demo`；[phase_b_downstream_protocol.md](phase_b_downstream_protocol.md) |
| Phase B プロトコル | [phase_b_data_protocol.md](phase_b_data_protocol.md) | 手順・指標定義 | Wikitext-2-raw、eval 分割、perplexity |
| 計測フック（区間時間） | `instrument_smoke.py` | JSON（`--out`）、ベースライン | 区間 `elapsed_s` の悪化検出 |
| 効率（短系列・回帰） | `efficiency_benchmark.py` + `regression_check.py` | JSON | ベースライン、`--max-slowdown` で悪化検出 |
| Phase 3 M1（デコード計測） | `decode_benchmark.py` | `decode_benchmark_ok` + JSON | [phase_c_decode_metrics.md](phase_c_decode_metrics.md)、`--demo` |
| Phase 3 M3（baseline／two_tier スイープ） | `two_tier_sweep.py` | `two_tier_sweep_ok` + JSON | [phase_c_two_tier_sweep.md](phase_c_two_tier_sweep.md) |
| Phase 3 M4（HBM バイト予算テンプレ） | `hbm_budget_probe.py` | `hbm_budget_probe_ok` + JSON | [phase_c_hbm_budget.md](phase_c_hbm_budget.md)、`--demo` |
| Phase 3 M5（抽出型 QA・SQuAD） | `squad_span.py` | `squad_span_ok` + JSON | [phase_b_downstream_protocol.md](phase_b_downstream_protocol.md)、`--demo` |
| Phase 3（主張バンドル P0） | `phase3_claim_run.py` | `phase3_claim_run_ok` + `phase3_claim_bundle.json` | [Phase3 実測計画](../../planning/Phase3_実測と主張完成計画_ja.md)、`--demo` |
| Phase 3（τ 事前登録 P1） | [phase_c_quality_tau_prereg.md](phase_c_quality_tau_prereg.md) | 登録表 | [Phase3 計画 §5.1](../../planning/Phase3_計画_二階建てと実証.md) |
| Phase 3（二階建てスタブ） | `core/two_tier/` | `pytest` | Router／Controller／品質 τ、`--router-step-stride`（P3） |

## 未カバー（ロードマップ Phase B 本格以降）

| モジュール | 理由 |
|------------|------|
| `AwaiIntegratedSLM` の大規模ベンチ本番評価 | 下流入口は `slm_downstream`（SST-2）。MMLU 等の固定運用は未 |
| 二階建て（本番 Router／KV 階層） | スタブは `core/two_tier`・計測は `decode_benchmark`。フル KV 階層は継続開発 |

## 改訂

| 日付 | 内容 |
|------|------|
| 2026-03-25 | 初版（Phase A′） |
| 2026-03-25 | Phase 1B（文化的調製・`slm_bridge_smoke`） |
| 2026-03-25 | Phase 2 入口（`slm_resonance_lm`） |
| 2026-03-25 | Phase 2 下流入口（`slm_downstream.py`、SST-2 プロトコル） |
| 2026-03-25 | Phase 2 拡張（`--task boolq`、SQuAD 抽出型は未） |
| 2026-03-25 | Phase 3 M1〜M3（`decode_benchmark`、`two_tier_sweep`、`core/two_tier`） |
| 2026-03-25 | Phase 3 M4〜M5（`hbm_budget_probe`、`squad_span`） |
| 2026-03-25 | Phase 3 P0〜P3（`phase3_claim_run`、`phase_c_quality_tau_prereg`、Router stride） |
