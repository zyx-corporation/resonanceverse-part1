# Phase 1B: 文化的調製と SLM 橋（実装スコープ v0.1）

ロードマップ上の **Phase 1B** は、大規模言語意味に依存しない**最小の調製経路**と、
因果 LM への**接続点**をコードで固定する。

## コンポーネント

| モジュール | 役割 |
|------------|------|
| `core/cultural_modulation.py` | `awai_pressure_from_embeddings`（スタブ）、`CulturalModulationAdapter`（埋め込み → 調製） |
| `core/lightweight_resonance.LightweightResonanceFacade` | `cultural_scale` を共鳴次元特徴に乗算（トークン ID 経路） |
| `core/resonant_core.AwaiIntegratedSLM` | 任意で `cultural_modulation=True` — `ResonantCore` 出力に `CulturalModulationAdapter` を乗算 |

## 解釈

- **SLM 経路**では、調製は「朧度」とは独立した**学習可能なゲート**として働く
  （`out_head` 前の共鳴特徴に作用）。
- **ベースライン比較**（`--baseline-hf`）には調製は含めない。
  `slm_resonance_lm.py` では `--cultural-modulation` と `--baseline-hf` を排他。

## 検証コマンド

```bash
# ファサードのみ（単体テストと同等）
pytest tests/test_phase1b.py -v

# HF + Awai（オプションで調製あり）
python experiments/slm_bridge_smoke.py --cpu --cultural-modulation

# 最小学習（デモ／実モデル）
python experiments/slm_resonance_lm.py --demo --cultural-modulation \
    --max-steps 10 --cpu --freeze-base
```

## 関連

- [Phase A 再現手順](phase_a_reproduction.md)
- [モジュール ↔ 実証スクリプト対応表](module_benchmark_map.md)
- [ROADMAP_ja.md](../../../ROADMAP_ja.md)
