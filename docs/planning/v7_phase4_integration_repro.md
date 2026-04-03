# Phase IV（方式 B 統合）— リポジトリ内の「最小再現」と本番の差

## 正本との関係

[実証実験設計 v7.0 §8 付近](../v7/Resonanceverse_v7.0_Experimental_Design.md) の Phase IV は、6 軸テンソル・DDE・朧などを**下流タスク**に載せた統合検証を想定する。

本リポジトリでは、その**全体パイプライン**は未実装だが、次が**方式 B 周辺の計測レール**として既に存在する。

| 要素 | スクリプト / モジュール |
|------|-------------------------|
| デコードレイテンシ・メモリ | [`experiments/decode_benchmark.py`](../../experiments/decode_benchmark.py) |
| baseline vs two-tier スタブ比較 | [`experiments/two_tier_sweep.py`](../../experiments/two_tier_sweep.py) |
| 統合 SLM（実装索引） | `AwaiIntegratedSLM` 等 — [`docs/api/modules/README.md`](../api/modules/README.md) |

## 最小バンドル

[`experiments/v7_phase4_minimal_repro.py`](../../experiments/v7_phase4_minimal_repro.py) は **`--demo --cpu`** で上記 M3 相当（`two_tier_sweep`）を 1 JSON にまとめる。任意で **`--with-squad-span`** で Phase 3 M5 のデモ辞書を同梱する（SQuAD 本番 HF ではない）。

```bash
python experiments/v7_phase4_minimal_repro.py --demo --cpu \
  --out experiments/logs/v7_phase4_minimal_demo.json
```

ベースライン参照用: [`experiments/baselines/v7_phase4_minimal_repro_demo_v1.json`](../../experiments/baselines/v7_phase4_minimal_repro_demo_v1.json)（生成コマンド実行後に中身を更新する運用でもよい）。

## 本番 Phase IV に向けた残差

- 同一モデル・同一データで **方式 B オン/オフ**の下流指標（EM/F1 等）を事前登録した閾値と照合する。
- 資源・レイテンシのみの主張に留める場合は、[主張表（論文用）](Resonanceverse_主張表_論文用_ja.md) の注意（品質 τ）に従う。
