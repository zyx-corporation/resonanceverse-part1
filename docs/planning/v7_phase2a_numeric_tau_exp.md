# 数値 τ 掃引（合成テンソル）と設計書 II-A の位置づけ

## 実装

[`experiments/v7_phase2a_delay_sweep.py`](../../experiments/v7_phase2a_delay_sweep.py) は、小規模の**反対称テンソル**に遅延結合を入れた離散時間シミュレーションで、各 τ におけるエネルギー（||W||_F）尾部の **oscillation_score**（変動係数）を記録する。

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

MRMP 実データの R(τ) との併記は [`v7_phase2a_theory_bridge.md`](v7_phase2a_theory_bridge.md) を参照する。
