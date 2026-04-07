# Phase II-A 実データ解析と理論 v7.0 の「橋」

**実装・レール分離の正本**: [v7_phase2a_implementation_master_plan.md](v7_phase2a_implementation_master_plan.md)（目的階層・理論↔数値対応表・実装順・ラベル規約）。**対応表の拡張・接続点・インフラ前提**は [v7_phase2a_theory_numeric_architecture_data_ja.md](v7_phase2a_theory_numeric_architecture_data_ja.md)。**理論 τ* を数値チェックインするまでのゲート**は [v7_phase2a_theoretical_tau_bridge_appendix_ja.md](v7_phase2a_theoretical_tau_bridge_appendix_ja.md)。

本稿は、**現行リポジトリの Phase II-A（MRMP 窓・凍結 LM・注意ベース R(τ)）**が何を示しうるかと、[Resonanceverse 理論 v7.0](../v7/Resonanceverse_Theory_v7.0.md)・[実証実験設計 v7.0](../v7/Resonanceverse_v7.0_Experimental_Design.md)が定める **τ*（ホプフ分岐閾値）** などが**どこまで同一視できないか**を整理する。論文・対外向けの主張の境界を明確にするのが目的である。

## 1. いまのパイプラインが測っているもの

- **入力**: 対話窓の JSONL（任意で 6 軸 LLM 審判スコア同梱）。**凍結言語モデル**の層出力から、事前登録 [`v7_phase2a_prereg_v1.json`](v7_phase2a_prereg_v1.json) に沿った **注意ベースの遅延積統計**を τ ごとに集約する（`v7_phase2a_empirical_run.py`）。
- **出力**: `by_tau` の **R_mean / R_var**、対話別寄与（`--export-contributions`）、任意で **補助** `auxiliary_label_delay_coherence`（6 軸について主解析と同型の遅延積系列）。
- **解釈上の位置づけ**: **コーパス上の記述統計**であり、学習中の共鳴テンソル **W_D(t)** の時間発展を直接積分した量ではない。

## 2. 理論・実験設計書が予測・定義しているもの

- **定理 3.3 系 3.3.1**（理論正本）: 遅延微分方程式としての共鳴ダイナミクスに対し、**Lyapunov–Krasovskii 型**の経路で大域漸近安定性を論じ、**ホプフ分岐閾値 τ*** を「設計上の上限」として位置づける。
- **実験 II-A**（実験設計正本 §3.1）: **τ*_exp** を、**N ノードのテンソル時間発展**をシミュレーションし、汎関数 **V_K** と **dV_K/dt** の符号転換などから**数値的に**記録する手続きとして定義する（μ、N などのスイープ含む）。

つまり理論側の τ* / 設計書側の τ*_exp は、**DDRF の閉じた（または簡約された）ダイナミクス**に紐づく。一方、現行 Phase II-A 実データ解析の **R(τ)** は **観測された注意行列からの統計**であり、定義域・生成過程が異なる。

## 3. いま足りない「橋」（同一視してはいけない箇所）

| 観点 | 理論 / 設計書 | 現行実データ Phase II-A |
|------|----------------|-------------------------|
| 対象ダイナミクス | W_D(t) の更新・安定性 | 凍結 LM の層アテンションに対する窓ごとの統計 |
| τ の意味 | 遅延パラメータ（モデル式上） | 実装で指定した **離散ラグ**（トークン／窓インデックス） |
| 「閾値」の語の用法 | τ_max と τ* の比較（安定性） | 図上の **R_var の峰・機械的な τ 候補**（探索的） |
| 補助 6 軸 | （主解析とは別経路の）ラベルベース対照 | 注意ベース主解析と**スケール非互換**で並べるのみ |

事前登録の `theory_note_ja` どおり、**コーパス代理の τ 候補**と **定理 3.3 経路の τ*** は**別物**としてラベルし、本文でも混同しないこと。

## 4. 橋を架けるために追加で有効な実験・解析（優先度の例）

1. **実験設計書 II-A 型の数値実験**（接続の骨格）: 小規模 N でテンソル（または承認済み簡約モデル）を時間発展させ、**τ*_exp** を記録し、理論式から得る **τ*** との比較プロトコル（設計書の乖離基準）に載せる。リポジトリ内の**合成離散テンソル**は、単発掃引 [`v7_phase2a_delay_sweep.py`](../../experiments/v7_phase2a_delay_sweep.py)（`--alpha-list` で感度）、説明は [v7_phase2a_numeric_tau_exp.md](v7_phase2a_numeric_tau_exp.md)。**V_K 代理・離散 ΔV に基づく数値スタブ**（`tau_exp_numeric_stub_*`、設計書 §3.1 を参照した手続きのレール・非同一視）: [`v7_phase2a_tau_exp_lyapunov_stub.py`](../../experiments/v7_phase2a_tau_exp_lyapunov_stub.py)。**1 JSON に束ねるワンショット**（τ 掃引＋α 感度＋ラベル規約・再現コマンド同梱）: [`v7_phase2a_theory_bridge_synth.py`](../../experiments/v7_phase2a_theory_bridge_synth.py)（`--demo` で軽量）。
2. **同一モデル族でのスイープ**: μ（強凸性代理）、N、層 l を変えたとき **τ*_exp** または R(τ) の形状がどう動くかを表にし、理論が予測する単調性・スケール傾向と**照合可能な形**で報告する。実データの複数 JSON 横並び: [`v7_phase2a_compare_runs.py`](../../experiments/v7_phase2a_compare_runs.py)。
3. **写像 φ（Phase I）との接続**: S_asym と 6 軸の相関が取れたコーパスでは、**注意ベース R(τ)** と **ラベルベース補助系列**の**独立性／冗長性**（例: 部分相関・層別）を探索的に記述する。τ 系列の Pearson / 簡易部分相関: [`v7_phase2a_primary_aux_tau_association.py`](../../experiments/v7_phase2a_primary_aux_tau_association.py)。層別 Frobenius×6 軸（審判 JSONL）: [`run_phase1a_mrmp_v7_axes.sh`](../../experiments/run_phase1a_mrmp_v7_axes.sh)。
4. **報告の厳格化**: 主対比は事前登録どおり（例: 事前指定のペア差・CI）。τ 軸全体の峰は**多重比較未調整**である旨を本文で固定する。
5. **方式 B 周辺の最小再現**（Phase IV 全景の代替ではない）: [`v7_phase4_minimal_repro.py`](../../experiments/v7_phase4_minimal_repro.py)・[v7_phase4_integration_repro.md](v7_phase4_integration_repro.md)。

6. **論文向け比較表（合成指標・理論注入・乖離％）**: [`v7_phase2a_paper_tau_comparison.py`](../../experiments/v7_phase2a_paper_tau_comparison.py)・[v7_phase2a_paper_figures.md](v7_phase2a_paper_figures.md) の表節。

## 5. 記述・ラベル規約（推奨）

- **tau_star_corpus_proxy** 等: コーパス上で機械的に取った τ の候補（探索的）。
- **tau_star_exp**: 設計書 II-A に従いシミュレーションで記録した閾値のみに用いる。
- **theoretical_tau_star**: 理論式・導出からの τ* のみに用いる。

図表キャプションでは **Figure X はコーパス統計**、**別図または別節で τ*_exp** と書き分けると、査読で誤読されにくい。

## 関連リンク

- 事前登録: [`v7_phase2a_prereg_v1.json`](v7_phase2a_prereg_v1.json)
- 論文用図・キャプション注意: [`v7_phase2a_paper_figures.md`](v7_phase2a_paper_figures.md)
- 再現・ハッシュ手順: [`../../experiments/README.md`](../../experiments/README.md)（Phase II-A 再現性・マニフェスト）
