# Resonanceverse: 次世代AI共鳴アーキテクチャ

[![CI](https://github.com/zyx-corporation/resonanceverse-part1/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/zyx-corporation/resonanceverse-part1/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Resonanceverse** は、オートポイエーシスと動的共鳴による社会的文脈理解を目指す AI アーキテクチャの研究実装です。従来の大規模言語モデル（LLM）に依存しない軽量コア（共鳴場・ROI・創発指標）から検証できます。

## 理論的コア（Theoretical Core）

Tomoyuki Kano 氏の理論整理に基づき、以下の観点をコード上で扱います。

1. **情報理論的非分解性**: 階層分解による情報損失の回避。
2. **動的共鳴の最適性**: 変分原理下での適応的重み付け。
3. **大域的漸近安定性**: Lyapunov 理論による安定性の議論枠。
4. **計算複雑度 O(N log N) 寄り**: 動的関心領域（ROI）制御による削減の実証。
5. **最大エントロピー収束**: Softmax 分布への正則化イメージ。
6. **戦略的曖昧性**: 「おぼろ（Obscurity）」による計算資源の調整。

## 主な特徴

### 効率性

- **O(N²) 全対全との比較**: ROI 階層とベンチで比較可能（`experiments/efficiency_benchmark.py`）。
- **軽量コア**: 埋め込み＋共鳴場＋ ROI のスタンドアロン経路。

### 文化的理解（設計・文書）

- **「あわい」等の文化的ニュアンス**: 設計文書・プロトタイプ計画に記載（実装は段階的拡張）。

### 創発性（作動的定義）

- **混合基準線・KL・新規性・アブレーション**: `experiments/emergence_metrics.py` と設計書に準拠。

### 理論的革新性

- **動的共鳴場・オートポイエーシス**: 詳細は [理論ドキュメント](docs/theory/resonanceverse_theory.md) を参照。

## 性能実証（文書・目標値の例）

以下は設計・プロトタイプ文書に記載された目標・例示です。実測は `experiments/` のスクリプトで再現してください。

| 指標 | 従来手法（例） | Resonanceverse（文書値） | 改善率（例） |
|------|----------------|--------------------------|--------------|
| 社会的文脈理解 | 85.3% (GPT-4) | **89.3%** | +4.7% |
| 「あわい」理解 | 41.7% (Claude-4) | **77.5%** | +85.8% |
| 計算効率 | O(N²) | O(N log N) 寄り | **65% 削減（目標）** |
| 推論速度 | 45 tok/s | **142 tok/s** | +215% |
| 創発性スコア | 0.23 | **0.67** | +191% |

## プロジェクト構成

- `core/`: 共鳴エンジン（`ResonanceEngine`）、`ResonantCore`、`LightweightResonanceFacade`、ROI、オートポイエーシス。
- `configs/`: グローバル設定（YAML）。
- `experiments/`: スモーク、創発指標、効率ベンチ。
- `docs/`: 理論・計画・実装設計・チュートリアル・API 索引（[文書インデックス](docs/README.md)）。

## インストール

### 必要要件

- Python 3.11+（3.10 でも動作確認しやすい構成）
- PyTorch 2.x（CPU / CUDA / MPS いずれも可）
- 推奨: 8GB+ RAM

### 基本インストール

```bash
git clone https://github.com/zyx-corporation/resonanceverse-part1.git
cd resonanceverse-part1
pip install -r requirements.txt
```

`pip install -e .` や M3 Max 専用オプションは、パッケージメタデータ整備後に追記予定です。

## クイックスタート（現在のリポジトリに即した例）

```python
import torch
from core import LightweightResonanceFacade

facade = LightweightResonanceFacade(
    vocab_size=8000,
    embed_dim=128,
    resonance_dim=6,
    num_nodes=128,
    tau=1.0,
)
tok = torch.randint(0, 8000, (2, 64))
out = facade(tok)
print(out["resonance_scores"].shape)  # 上位ノードに対する共鳴スコア
```

```bash
# スモーク（SLM なし）
python experiments/evel_benchmarks.py

# 創発指標ログ（JSON + 事前登録 YAML）
python experiments/emergence_metrics.py --out experiments/logs/emergence_run.json

# 効率比較（CPU 例）
python experiments/efficiency_benchmark.py --cpu --seq-lens 64 128 256 512
```

## ドキュメント

### 文書インデックス

- **[`docs/README.md`（全一覧・ディレクトリ案内）](docs/README.md)**

### 理論・設計（抜粋）

- [Resonanceverse 理論（完全理論と実装）](docs/theory/resonanceverse_theory.md)
- [数学的基礎（証明の完全版）](docs/theory/mathematical_foundation.md)
- [アーキテクチャ設計（モジュール構造とクラス設計）](docs/implementation/architecture_overview.md)

### チュートリアル・実証計画

- [始め方・2 週間軽量プロトタイプ計画](docs/tutorials/getting_started.md)
- [ロードマップ（和文チェックリスト）](ROADMAP_ja.md)
- [実証ロードマップ（軽量コアと SLM／二階建て）](docs/planning/Resonanceverse実証ロードマップ_軽量コアとSLM二階建て.md)

### API・モジュール索引

- [API / モジュール設計リファレンス](docs/api/modules/README.md)（[アルゴリズム仕様](docs/api/modules/algorithm_spec.md) を含む）
- [モジュール ↔ 実証スクリプト対応表（Phase A）](docs/api/modules/module_benchmark_map.md)
- [Phase A 再現手順](docs/api/modules/phase_a_reproduction.md)

（`examples/` のデモスクリプトは未同梱のため、上記 `experiments/` を参照してください。追加の理論・実装・計画文書は [文書インデックス](docs/README.md) を参照。）

## 実験・検証

上記クイックスタートのコマンドに加え、ログは `experiments/logs/` に出力されます（`.gitignore` 対象）。

SLM（HF ベースラインと Awai 統合）の**パフォーマンス比較結果の整理**（JSON フィールド、`comparison` の比の読み方、記録テンプレート）は [docs/api/modules/slm_performance_comparison.md](docs/api/modules/slm_performance_comparison.md) を参照。

## 開発・貢献

```bash
pip install -r requirements.txt
pip install pytest
pytest tests/ -v
```

手順・DCO・軽量 CLA・**ブランチ保護の採用ポリシー**は **[CONTRIBUTING.md](CONTRIBUTING.md)** を参照してください。倫理・利用方針は [ETHICAL.md](ETHICAL.md)（[和文 ETHICAL_ja.md](ETHICAL_ja.md)）、ライセンス付与は [CLA.md](CLA.md)（[和文 CLA_ja.md](CLA_ja.md)）です。

## ロードマップ

フェーズ対応・実装チェックリストの**和文まとめ**は **[ROADMAP_ja.md](ROADMAP_ja.md)** を参照してください（**Phase 3**＝二階建て・資源実証は [Phase3 計画](docs/planning/Phase3_計画_二階建てと実証.md)）。**Phase 4**（大規模ノード・分散／Jetson 等）は **[独立ロードマップ](docs/planning/ROADMAP_Phase4_分散とエッジ_ja.md)** で追跡し、Phase A〜C の技術実証とは別軸です。理論・判定表の詳細は [実証ロードマップ（軽量コアと SLM／二階建て）](docs/planning/Resonanceverse実証ロードマップ_軽量コアとSLM二階建て.md)、長期の協働・透明性は [透明性・協働ロードマップ](docs/planning/🚀%20透明性確保による人間-AI協働研究発展ロードマップ.md) です。

## ライセンス

本プロジェクトは [GNU Affero General Public License v3.0](LICENSE) の下で公開されています。

## 謝辞

- **理論的基盤**: マトゥラーナ・ヴァレラのオートポイエーシス理論
- **数学的基礎**: McMullin (2004) の計算オートポイエーシス研究
- **技術的基盤**: PyTorch、Apple Metal Performance Shaders 等

## お問い合わせ

- **著者**: Tomoyuki Kano
- **所属**: ZYX Corp 人工叡智研究室
- **Email**: tomyuk@zyxcorp.jp
- **ORCID**: [0009-0004-8213-4631](https://orcid.org/0009-0004-8213-4631)

## スター・フォロー

[![Star History Chart](https://api.star-history.com/svg?repos=zyx-corporation/resonanceverse-part1&type=Timeline)](https://star-history.com/#zyx-corporation/resonanceverse-part1&Timeline)

---

**「共鳴による理解、理解による共創」** — Resonanceverse は、人間と AI の新しい協働関係を目指す研究プロジェクトです。
