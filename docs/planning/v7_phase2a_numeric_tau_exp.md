# 数値 τ 掃引（合成テンソル）と設計書 II-A の位置づけ

## 実装

[`experiments/v7_phase2a_delay_sweep.py`](../../experiments/v7_phase2a_delay_sweep.py) は、小規模の**反対称テンソル**に遅延結合を入れた離散時間シミュレーションで、各 τ におけるエネルギー（||W||_F）尾部の **oscillation_score**（変動係数）を記録する。

[`experiments/v7_phase2a_tau_exp_lyapunov_stub.py`](../../experiments/v7_phase2a_tau_exp_lyapunov_stub.py) は、**同一ダイナミクス**上で V(t)=½||W||²_F（W*=0 の合成仮定）と尾区間の **離散 ΔV** から、設計書 §3.1 を**参照した**数値スタブ `tau_exp_numeric_stub_*` を返す（完全な τ*_exp 手続き・理論 V_K ではない—スクリプト先頭の注意書きどおり）。

- 単発掃引の JSON に **`tau_exp_proxy_oscillation_jump`**（従来 `tau_largest_jump` と同値）と **`design_bridge_ja`** を含む。
- **`--alpha-list a,b,c,...`**: `alpha` を**強凸性代理**（減衰項の係数）としてスイープし、各 alpha で得た **`tau_exp_proxy_oscillation_jump`** を `by_alpha` に並べる（`schema_version`: `v7_phase2a_alpha_sweep.v1`）。

## 設計書 II-A との関係

[v7.0 実証実験設計書 §3.1](../v7/Resonanceverse_v7.0_Experimental_Design.md) の **τ*_exp** は、共鳴テンソル時間発展と **Lyapunov–Krasovskii 汎関数**の符号などから定義される**別手続き**である。

本スクリプトの出力は:

- **τ*_exp そのものではない**（`theory_bridge` のとおり）。
- ただし「**μ（ここでは alpha）を変えたときに、振動代理の急増 τ がどう動くか**」という**感度表**として、理論が予測する傾向と**定性的に**照合する素材になりうる。

## 推奨コマンド例

```bash
# 単発（既定ハイパラ）
python experiments/v7_phase2a_delay_sweep.py --tau-max 8 --steps 2000 --out experiments/logs/phase2a_synth_tau.json

# alpha スイープ（軽量）
python experiments/v7_phase2a_delay_sweep.py --tau-max 8 --steps 1500 \
  --alpha-list 0.1,0.15,0.2,0.25,0.3 --out experiments/logs/phase2a_synth_alpha_sweep.json
```

理論橋ドキュメント向けに **上記 2 本を 1 JSON にまとめる**（ラベル規約・参照パス・再現コマンド同梱）:

```bash
python experiments/v7_phase2a_theory_bridge_synth.py --demo \
  --out experiments/logs/v7_phase2a_theory_bridge_synth_demo.json
```

Lyapunov V 代理スタブ（`tau_exp_numeric_stub_*`）の軽量例:

```bash
python experiments/v7_phase2a_tau_exp_lyapunov_stub.py --demo --seed 0 \
  --out experiments/logs/phase2a_tau_exp_lyapunov_stub_demo.json
```

論文用の**比較表**（合成指標＋理論 τ* 注入＋乖離％・`figure_role`）: [`v7_phase2a_paper_tau_comparison.py`](../../experiments/v7_phase2a_paper_tau_comparison.py)。手順の文脈は [`v7_phase2a_paper_figures.md`](v7_phase2a_paper_figures.md) の「表（合成 τ 指標と理論参照・乖離）」。

MRMP 実データの R(τ) との併記は [`v7_phase2a_theory_bridge.md`](v7_phase2a_theory_bridge.md) を参照する。

---

## τ*_exp の操作定義（リポジトリ固定・スタブ）

設計書 §3.1 の **τ*_exp** は Lyapunov–Krasovskii 手続きに沿った**別物**である。本リポジトリがログに出す **`tau_exp_numeric_stub_*`** は、次の**操作定義**に従う**数値スタブ**であり、設計書の完成実装の置き換えではない。

| 項目 | 固定内容 |
|------|-----------|
| ダイナミクス | `v7_phase2a_delay_sweep` と同一の反対称テンソル＋整数ラグ τ |
| 汎関数 V | **段階 0**: `V(t)=½‖W(t)‖²_F`（W\*=0 仮定）。**段階 2**: `--krasovskii-gamma γ>0` で `V += γ·Σ_{k=1}^{τ} ½‖W(t−k)‖²_F`（離散遅延和・連続 Krasovskii の厳密形ではない） |
| ΔV | **連続時間の dV/dt ではない**。`burn_frac` 以降の `V_{t+1}-V_t` の列から尾統計を取る |
| `tau_exp_numeric_stub_mean_dV` | 掃引 τ=0,1,… のうち、尾平均 ΔV が `mean_dv_threshold` を**上回った最初の τ** |
| `tau_exp_numeric_stub_frac_positive` | 同上で、尾において ΔV>0 の**割合**が `frac_positive_threshold` を上回った最初の τ |
| ラベル | 図表では `tau_exp_numeric_stub_*` と明記し、理論 τ*・`tau_star_corpus_proxy` と混線しない |

理論 τ* の**数値参照**はリポジトリが自動計算しない。[`v7_phase2a_theoretical_tau_reference_v1.json`](v7_phase2a_theoretical_tau_reference_v1.json) にチェックインし、`v7_phase2a_paper_tau_comparison.py --theoretical-reference-json …` で読み込む（CLI の `--theoretical-tau-star` が優先）。**注入前の同一視条件**は [理論 τ* 付録（ゲート）](v7_phase2a_theoretical_tau_bridge_appendix_ja.md)。**教材用スカラー線形サロゲート**（理論 τ* ではない）の再生成: [`v7_phase2a_scalar_delay_tau_suggest.py`](../../experiments/v7_phase2a_scalar_delay_tau_suggest.py)・付録 §7。
