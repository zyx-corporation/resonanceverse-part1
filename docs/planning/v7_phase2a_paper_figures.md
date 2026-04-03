# Phase II-A 図表（論文・投稿用）

実データ JSON（`v7_phase2a_empirical_run` の `*_with_contrib.json`）から、次で印刷向け図を出力する。

```bash
python experiments/v7_phase2a_tau_plots.py path/to/*_with_contrib.json --paper
```

既定で **PNG（300 dpi）** と **PDF**（ベクタ、LaTeX 貼り付け向け）を保存する。ファイル名は入力 stem に基づき、例として次のようになる（通常モードの `*_tau_primary.png` とは別名で上書きしない）。

- `{stem}_tau_paper_primary.png` / `.pdf`
- `{stem}_tau_paper_auxiliary_rvar.png` / `.pdf`（6 軸補助がある場合）

画面プレビュー用は `--paper` を外す。タイトル行を図内に付けたい場合は `--show-suptitle`。

## 図 1（主解析）— `*_tau_primary`

**内容（パネル A–C）**

- **(A)** 遅延 τ に対する **R_mean**：各対話で定義した遅延積の平均を、対話間で等価重みで集約した量（事前登録 `v7_phase2a_prereg_v1.json` の主解析）。
- **(B)** **R_var**（対話間分散）と、τ 方向の移動平均（平滑窓は実行時パラメータ、既定 5）。
- **(C)** 各 τ で主解析に寄与した **有効対話数 n(τ)**。高 τ では n が減るため、(B) の分散ピークは標本サイズと併記して解釈する。

**英語キャプション（案）**

> **Figure X.** Phase II-A primary analysis on the MRMP window corpus. (A) Mean cross-dialogue statistic R as a function of lag τ (see preregistered definition). (B) Between-dialogue variance of R across τ (solid) and a moving average (dashed). (C) Number of dialogues contributing at each τ. Model: [MODEL], layer: [LAYER]. D = [N_DIALOGUES] dialogues in this run.

**日本語キャプション（案）**

> **図 X.** MRMP 窓コーパスに対する Phase II-A 主解析。(A) 遅延 τ に対する対話間平均 R。(B) τ ごとの対話間分散（実線）と移動平均（破線）。(C) 各 τ の有効対話数。モデル [MODEL]、層 [LAYER]。本実行の対話数 D = [N_DIALOGUES]。

---

## 図 2（補助）— `*_tau_auxiliary_rvar`（6 軸ラベルがある場合のみ）

**内容**

- 各サブパネル：LLM 審判で付与した 6 軸スコアについて、主解析と**同型**の遅延積を τ 上で集約した系列の **対話間分散 R_var**（事前登録の `auxiliary_label_delay_coherence`）。注意ベースの主解析とは**別スケール**。

**英語キャプション（案）**

> **Figure Y.** Auxiliary analysis: between-dialogue variance of the label-based delay-coherence statistic (preregistered six axes), plotted versus τ. Not directly comparable in magnitude to attention-based R_var in Figure X.

**日本語キャプション（案）**

> **図 Y.** 補助解析：事前登録どおり 6 軸ラベルについて主解析と同型の遅延整合量の対話間分散を τ に対してプロットしたもの。注意ベースの主解析図とはスケールが異なる。

---

## 注意（投稿時）

- τ* の機械的候補や高 τ の峰は **探索的**；**多重比較未調整**（事前登録のとおり）。本文では主対比（例 Δ(0,1)）と n(τ) を明記する。
- 理論の Lyapunov–Krasovskii 経路の τ* とは **別定義**（事前登録の注記どおり）。

理論との境界・追加実験の整理は **[Phase II-A と理論の「橋」](v7_phase2a_theory_bridge.md)** を参照。
