# 実験ロードマップ v7.0（理論正本との対応）

**正本**: [Resonanceverse v7.0 実証実験設計書](../v7/Resonanceverse_v7.0_Experimental_Design.md)

本文書は **Phase I〜IV** と本リポジトリの **スクリプト** の対応を示す。レガシー（Phase A〜C）は [ROADMAP_ja.md](../../ROADMAP_ja.md)。

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

## 対応表（実装済みハーネス）

| v7 | 検証の核 | スクリプト |
|----|----------|------------|
| **Phase I-A** | S_asym と 6 軸（または代理特徴）の相関 | [`experiments/v7_phase1a_phi_correlation.py`](../../experiments/v7_phase1a_phi_correlation.py)（`--demo`＝合成検証；実モデルは `--demo` なしで層別 S_asym 統計） |
| **Phase I-B** | 有向テンソルが対称化に落ちない | [`experiments/v7_phase1b_directed_tensor.py`](../../experiments/v7_phase1b_directed_tensor.py) |
| **Phase II-A** | τ 掃引・安定性プロキシ | [`experiments/v7_phase2a_delay_sweep.py`](../../experiments/v7_phase2a_delay_sweep.py) |
| **Phase III（合成）** | あわい Ω の数値出力 | [`experiments/v7_phase3a_awai_metrics.py`](../../experiments/v7_phase3a_awai_metrics.py) |
| **Phase III-A（本番）** | 人間「間合い」アノテとの相関 | **未着手**（コーパス・アノテが必要） |
| **Phase IV** | 方式 B 統合 | 既存: `AwaiIntegratedSLM`・`decode_benchmark`・`two_tier_sweep` 等 |

## レガシー実証との関係

| レガシー | v7 との関係 |
|----------|-------------|
| Phase A〜A′ | I-B の最小動作＋ CI 回帰 |
| Phase B / C / 3 | 品質・資源の記録（方式 B 全体とは別軸） |
| Phase 4 | 分散ロードマップ（v7 PoP 議論とは別） |

## テスト

`tests/test_v7_experiments.py` が `--demo` 相当の関数をオフライン検証する。

---

*改訂時は v7 実験設計書の版と同期すること。*
