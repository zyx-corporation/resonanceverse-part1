# Resonanceverse: Autopoietic AI Framework

動的共鳴メカニズムとオートポイエーシス（自己創生）理論を統合した、次世代AIフレームワークの実装プロジェクトです。

## 🚀 理論的コア (Theoretical Core)

本プロジェクトは、Tomoyuki Kano氏による論文に基づき、以下の6つの定理を実装しています:

1. **情報理論的非分解性**: 階層分解による情報損失の回避。
2. **動的共鳴の最適性**: 変分原理下での適応的重み付け。
3. **大域的漸近安定性**: Lyapunov理論による安定性保証。
4. **計算複雑度 O(N log N)**: 動的関心領域(ROI)制御による高速化。
5. **最大エントロピー収束**: Softmax分布への理論的収束。
6. **戦略的曖昧性**: 「おぼろ(Obscurity)」による計算資源の最適化。

## 📂 プロジェクト構成

- `core/`: 共鳴エンジン、自己創生更新ロジック。
- `models/`: SLM (Llama-3等) との統合アダプター。
- `modules/`: 「間合い（あわい）」等の文化的ニュアンス計算モジュール。

## 🛠 セットアップ

```bash
git clone [https://github.com/your-username/resonanceverse-ai.git](https://github.com/your-username/resonanceverse-ai.git)
cd resonanceverse-ai
pip install -r requirements.txt
python experiments/train.py
