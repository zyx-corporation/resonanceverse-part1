---
title: Phase II-A — 理論 τ* と合成 delay_sweep の対応（付録・導入手順）
created: 2026-04-07
language: ja
document_type: planning
---

# Phase II-A — 理論 τ* と合成 delay_sweep の対応（付録・導入手順）

本付録は、[実証実験設計 v7.0 §3.1](../v7/Resonanceverse_v7.0_Experimental_Design.md)・[理論 v7.0 定理3.3 系](../v7/Resonanceverse_Theory_v7.0.md)の **τ***（ホプフ分岐閾値）と、リポジトリの **`v7_phase2a_delay_sweep`** が実装する**離散反対称テンソル＋整数ラグ**を、**無断に同一視しない**ための**ゲートとチェックリスト**である。

**正本（短表）**: [実装マスタープラン §2](v7_phase2a_implementation_master_plan.md)  
**チェックイン数値**: [`v7_phase2a_theoretical_tau_reference_v1.json`](v7_phase2a_theoretical_tau_reference_v1.json)（未確定時は `theoretical_tau_star`: null）

---

## 1. 理論側で固定されていること

- **τ*** は付録 A の記号どおり「ホプフ分岐閾値」として定義される（[理論 v7.0 系3.3.1](../v7/Resonanceverse_Theory_v7.0.md)）。
- 定理3.3の条件4（τ_max < τ*）は**連続時間**の遅延系における安定性の境界として働く。
- **閉形式の τ*(μ, η, N, …)** を本リポジトリの理論 Markdown に**同梱していない**。設計書 §3.1 は、μ・N を変えながら **τ*_exp** を数値的に記録し、理論 τ* との**乖離基準（20% / 50%）**で読む手続きを述べるにとどまる。

---

## 2. リポジトリ側（合成）で固定されていること

- `experiments/v7_phase2a_delay_sweep.py` は **Euler 風の離散更新**・**反対称テンソル**・**単一整数ラグ τ**（履歴リング）・係数 **`alpha`（減衰）**, **`beta`（遅延結合強度）**, **`noise`**, **`dt`** を用いる。
- **学習率 η**・**損失の強凸性 μ** は、理論の記号と**字句一致しない**。`alpha` を μ の代理とみなすときは、[対応表（索引）](v7_phase2a_theory_numeric_architecture_data_ja.md) §1 に**スケール対応を 1 行追加**したうえで比較表に載せる。
- Lyapunov スタブの **τ*_exp 系ラベル**（`tau_exp_numeric_stub_*`）は設計書手続きの**完成実装ではない**（[数値 τ 説明](v7_phase2a_numeric_tau_exp.md)）。

---

## 3. 同一視のゲート（必須）

次を**すべて**満たすとき初めて、`theoretical_tau_star` にスカラーをチェックインし、**乖離％を本番主張**としてよい。

1. **簡約モデル文書**: DDRF のどの部分（次元・更新則・平衡点 W_D\*）を、`delay_sweep` の離散写像に**縮約**したかを、論文・付録・本リポの節として 1 本に書く（式番号または疑似コード）。
2. **連続 τ ↔ 離散ラグ**: 連続遅延と整数 τ の対応（例: τ_cont ≈ τ_lag·dt のような**明示的スケール**）を一文で宣言する。
3. **数値導出**: その簡約系上で τ* を
   - 閉形式（あれば）、または
   - 線形化・固有値横切り、分岐追跡、ブランチスイッチング探索  
   のいずれかで**再現可能に**求め、手順を同じ付録に書く。
4. **プロvenance**: `theoretical_provenance_ja` に式番号・付録・コミット可能なスクリプトパスを書く。
5. **ハイパラ整合**: `v7_phase2a_paper_tau_comparison.py` を回す **`N, d, dt, alpha, beta, noise, seed`** が、上記導出に使ったパラメタと**一致**する（デモ用の別セットならキャプションで「デモ専用」と切り分ける）。

---

## 4. チェックイン手順（`theoretical_tau_star` を null から埋める）

1. 本付録 §3 のゲートを満たす PR をマージする（簡約＋導出＋スケール宣言）。
2. [`v7_phase2a_theoretical_tau_reference_v1.json`](v7_phase2a_theoretical_tau_reference_v1.json) を更新する:
   - `theoretical_tau_star`: 数値（浮動でも可。離散ラグと連続 τ のどちらかを `tau_star_unit` 等の任意フィールドで明示してよい）。
   - `revision`: 上げる（例: v1.2）。
   - `theoretical_provenance_ja`: 導出の出所を具体的に。
3. [実装マスタープラン §2](v7_phase2a_implementation_master_plan.md) の対応表に、**新しいパラメタ対応行**があれば 1 行追加する。
4. `paper_tau_comparison` を **`--theoretical-reference-json`** で再実行し、成果 JSON を再現ログに残す。

---

## 5. 設計書 §3.1 との位置づけ（参照のみ）

設計書は **N ∈ {10, 50, 100, 500}**、**μ ∈ {0.1, 0.5, 1.0, 2.0}** などのスイープ例を挙げる。リポの既定 `paper_tau_comparison` の `N, alpha` はこれと**一致しない**。乖離％を「設計書 II-A 型」と名乗る場合は、**上記スイープに合わせた再実行**と、理論 τ* が**同じ簡約・同じ (μ, N) 点**で与えられていることを確認する。

---

## 6. 現状（本リポジトリ）

- `theoretical_tau_star` は **null**（導出未チェックイン）。
- CI は**示教用**に `paper_tau_comparison --theoretical-tau-star 2.5` を**CLI で上書き**している。これは**本ファイルの null と矛盾しない**（デモ数値であり本番の理論参照ではない）。

---

## 7. スカラー線形サロゲート（教材用・理論 τ* ではない）

**目的**: 「数値を 1 つ持って感度を見る」ための**最小離散モデル**。`theoretical_tau_star` の**代替にはならない**。

モデル（離散）:  
`x_{t+1} = a·x_t + b·x_{t-τ}`、`a = 1 - dt·α`、`b = dt·β / N`。

各整数 τ ≥ 0 について同伴行列の固有値の**最大絶対値**（スペクトル半径）を計算し、**最初に 1 を超える τ** を `tau_first_unstable_lag` として報告する（τ=0 は `|a+b|`）。

**スクリプト**: [`experiments/v7_phase2a_scalar_delay_tau_suggest.py`](../../experiments/v7_phase2a_scalar_delay_tau_suggest.py)

```bash
python experiments/v7_phase2a_scalar_delay_tau_suggest.py --N 10 --dt 0.05 --alpha 0.15 --beta 0.85 --tau-max 200
```

`paper_tau_comparison` 既定に近い (N, dt, α, β) では、しばしば **探索範囲内で不安定化しない**（`tau_first_unstable_lag`: null）。β を大きくすると τ=0 で既に |a+b|>1 となる例がある。**tensor delay_sweep の分岐と一致する保証はない**。

再生成したブロックは [`v7_phase2a_theoretical_tau_reference_v1.json`](v7_phase2a_theoretical_tau_reference_v1.json) の `scalar_linear_surrogate` に**手動マージ**するか、`--out` で断片 JSON を出して差分レビューする。

---

*本付録の改訂は `v7_phase2a_theoretical_tau_reference_v1.json` の `theoretical_tau_derivation_appendix_md` と整合させる。*
