# Resonanceverse 理論（リポジトリ索引・v7 正本）

**リポジトリ上の理論の正本**は **[Resonanceverse 理論 v7.0（有向遅延共鳴場）](../v7/Resonanceverse_Theory_v7.0.md)** である。本文書はその入口として、実装との対応と旧文書の扱いを示す。

## v7.0 の要点（要約）

- **有向遅延共鳴場（DDRF, ℛ_D）**: 関係テンソル **W_D** は一般に **w_ij ≠ w_ji**（非対称）とし、遅延 **τ_ij** を伴う遅延微分方程式としてダイナミクスを記述する。
- **6 軸評価**: trust・authority・proximity・intent・affect・history（各軸の非対称性の取り扱いは v7 本文の通り）。
- **「あわい」**: 遅延窓における関係性の未完了状態として **A_ij(t)** を定義（v7 §4）。
- **朧（Obscurity）**: 減算的忘却ではなく、**一次過程の計算論的解除**として **Θ_id** を再定義（v7 §6）。
- **Transformer 接続**: **中間層介入型（方式 B）**——アテンションの非対称成分 **S_asym** を 6 軸への写像 **φ** の主要シグナルとする（v7 §5）。

## 実装コードとの対応（現行）

| v7 の概念 | 本リポジトリの主な実装 |
|-----------|-------------------------|
| 共鳴スコア・学習可能な関係テンソル（N×N×d 系） | `core/resonance.py` の `ResonanceEngine` |
| 場バッファ・朧度 θ・オートポイエーシス様更新 | `core/resonant_core.py` の `ResonantCore`・`AwaiIntegratedSLM` |
| ROI / 期待 O(N log N) 寄りの更新 | `core/roi_selector.py`・`LightweightResonanceFacade` |
| 二階建てスタブ（デコード経路・Router） | `core/two_tier/`・`experiments/decode_benchmark.py` 等 |
| 実験計画（フェーズ I〜IV） | **[実証実験設計 v7.0](../v7/Resonanceverse_v7.0_Experimental_Design.md)**・[EXPERIMENT_ROADMAP_v7.md](../planning/EXPERIMENT_ROADMAP_v7.md) |

**注意**: 本コードベースは歴史的経緯により **対称テンソル／無遅延近似**の経路を含む。DDRF の全要素（厳密な τ 監視・φ の学習・DDE ソルバ）が実装済みとは限らない。詳細は各モジュールの docstring と `docs/api/modules/` を参照する。

## 実験・検証の正本

- **[Resonanceverse v7.0 実証実験設計書](../v7/Resonanceverse_v7.0_Experimental_Design.md)**（Phase I〜IV、判定基準）
- リポジトリ内の実行スクリプト対応: [EXPERIMENT_ROADMAP_v7.md](../planning/EXPERIMENT_ROADMAP_v7.md)
- レガシー実証（Phase A〜C・主張表・τ）: [docs/api/modules/README.md](../api/modules/README.md)・[Phase3 実測と主張完成計画](../planning/Phase3_実測と主張完成計画_ja.md)

## 旧版・参考文書

- **統合長文（v7 以前の「完全理論と実装」）**: [archive/theory_resonanceverse_legacy_integrated.md](../archive/theory_resonanceverse_legacy_integrated.md)（参照用・正本ではない）
- その他、`docs/theory/` 内の長文・英語稿・部内版は **参考**として残す。数式・主張の衝突時は **v7.0** を優先する。

---

*文書 ID 索引: RVT-2026-007-V7.0（理論）、RVT-EXP-2026-007（実験設計）*
