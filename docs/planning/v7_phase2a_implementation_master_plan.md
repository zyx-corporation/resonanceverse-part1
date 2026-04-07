# Phase II-A 周辺：全体実装マスタープラン（採用版）

本稿は、v7 の **Phase II-A（τ・安定性・MRMP）** と**理論 3.3 / 設計書 §3.1**をつなぐ実装を、**レールを混線させず**進めるための**正本**である。  
コーディングの優先度・理論と数値の対応・ラベル規約をここに固定し、個別スクリプトの docstring は本稿を参照する。

**関連**: [Phase II-A と理論の「橋」](v7_phase2a_theory_bridge.md) · [数値 τ 掃引](v7_phase2a_numeric_tau_exp.md) · [図表（論文）](v7_phase2a_paper_figures.md) · [**理論↔数値・接続点・データ前提（索引）**](v7_phase2a_theory_numeric_architecture_data_ja.md) · 事前登録 [`v7_phase2a_prereg_v1.json`](v7_phase2a_prereg_v1.json)

---

## 1. 目的階層（レールの分離）

同一の図表に **別レールの主張**を載せない。キャプションでレール名を明示する。

| レール | 主目的 | 主なアーティファクト | 成功判定の例（設計書・事前登録に従う） |
|--------|--------|----------------------|----------------------------------------|
| **A. II-A 数値（定理支持）** | 定理 3.3 / 設計書 §3.1 に**寄せた**離散シミュレーションで τ 挙動・V 代理を記録 | `delay_sweep`・`tau_exp_lyapunov_stub`・（将来）V_K 本実装 | 理論 τ* 参照と **20%/50% 乖離**の帯（比較表） |
| **B. MRMP 実データ（記述・φ）** | 凍結 LM の注意から R(τ)・Var・探索的 τ 候補 | `v7_phase2a_empirical_run`・`tau_plots` | 事前登録の主対比・ブートストラップ・n(τ) の併記 |
| **C. 合成玩具（感度）** | μ（α）・N などの**定性的感度** | `delay_sweep --alpha-list`・理論橋バンドル | 傾向表・**理論 τ* 同一視禁止** |
| **D. Phase IV（方式 B）** | デコード・二階建て・（任意）下流 | `v7_phase4_minimal_repro`・`decode_benchmark` | 資源・レイテンシ・別途定めた下流指標 |
| **E. RVT-008（MRMP 行・L2）** | ヘッド行列・八条件・Oboro デモ（統合 generate 非接続） | `mrmp_row`・`ablation_runner`・`oboro_generate --demo` | 計画書・Day0・マニフェスト |

**禁止**: レール B の峰を「定理の τ* 実証」として書く。レール A のスタブだけで「τ*_exp 完成」と書く。

**JSON メタ**: 主要スクリプトの成果物に `rail_id`（単一）または `rail_ids`（複合バンドル）と `implementation_master_plan_md` を付与する。定数は `experiments/v7_phase2a_rail_metadata.py`。

---

## 2. 理論 ↔ 数値 対応表（実装の設計図）

新規実装・パラメータ変更時は、**左列と右列の対応を1行追加**してからコードを書く。

| 理論・設計書側 | リポジトリ側（現状 / 予定） | 備考 |
|----------------|----------------------------|------|
| 連続時間の遅延 τ | `delay_sweep` / スタブの **整数ラグ**（履歴リング長） | `dt` で離散化；**連続 τ との対応**を本文で宣言 |
| 強凸性 μ | 減衰 **`alpha`**（合成）・設計書の μ スイープは **対応表で明示** | 「α = μ の代理」と書くなら**スケール根拠**を付ける |
| V_K（一般） | **段階 0**: ½‖W‖²_F・W\*=0（`simulate_tau_v_k_series`）。**段階 2**: `krasovskii_gamma>0` で離散遅延和項（`delay_sweep` 同ファイル） | W_D\* 一般・連続 Krasovskii 厳密形は未実装 |
| dV_K/dt の符号 | **現状**: 離散 **ΔV** の尾平均・正比率（`tau_exp_lyapunov_stub`） | 連続時間の符号ではないと**常に明記** |
| τ*_exp（設計書手続き） | **予定**: V・符号・（任意）スペクトルに沿った**操作定義**と JSON スキーマ | スタブの `tau_exp_numeric_stub_*` は**別ラベル**のまま |
| 理論 τ* | **リポジトリは自動算出しない**；`paper_tau_comparison --theoretical-tau-star` で注入 | 出所は provenance 文字列で固定 |
| MRMP の τ 候補 | `tau_star_corpus_proxy`・事前登録の Var 規則 | レール B のみ |

**拡張**: 上表をスクリプト・JSON フィールドまで伸ばした版、**アーキテクチャ接続点**（MRMP→実証→後処理、合成、RVT、Phase IV）、**MRMP・審判・ローカル SLM**の前提整理は [v7_phase2a_theory_numeric_architecture_data_ja.md](v7_phase2a_theory_numeric_architecture_data_ja.md) を参照（本 §2 の PR と同期して更新する）。

---

## 3. τ・ラベル規約（一本化）

| ラベル | 意味 |
|--------|------|
| `theoretical_tau_star` | 理論・別導出から与えた参照値（比較表で注入） |
| `tau_exp_numeric_stub_*` | Lyapunov **スタブ**（ΔV 規則）；**§3.1 の完成実装ではない** |
| `tau_exp_proxy_oscillation_jump` | 振動スコア急増の **探索的代理** |
| `tau_star_corpus_proxy` | MRMP 等の**コーパス上**機械候補 |

図表では **Figure / Table のレール**（合成・理論・実データ）をキャプションで分ける。[論文図表案](v7_phase2a_paper_figures.md)

---

## 4. 推奨実装順序（チェックリスト）

1. **対応表の更新**（上表に新行・パラメータ対応を追記）— 本ファイルの PR。
2. **理論 τ* の数値参照**の固定（論文・付録・別スクリプト）→ 比較表に注入。
3. **V_K 段階導入**（項の追加順・W\* の定義）: 段階 2 として `simulate_tau_v_k_series` の離散遅延和項（`krasovskii_gamma`）を導入済み。W_D\* 一般・連続 Krasovskii 厳密形は未 → テストと JSON スキーマ `v7_phase2a_*.v1` を分離しつつ拡張。
4. **τ*_exp 操作定義**: リポジトリ側の**スタブ**は [`v7_phase2a_numeric_tau_exp.md`](v7_phase2a_numeric_tau_exp.md) と Lyapunov JSON の `tau_exp_operational_spec_md` で固定。設計書 §3.1 の完成手続きへの追従は別 PR。
5. **MRMP** はレール B のまま拡張（審判・マニフェスト・再現）；レール A と**混線しない**。
6. **Phase IV** は 1 タスクにスコープを絞り、方式 B on/off の指標を事前登録。

**既存スクリプト（すで利用可）**

- 合成掃引: `experiments/v7_phase2a_delay_sweep.py`
- Lyapunov スタブ: `experiments/v7_phase2a_tau_exp_lyapunov_stub.py`
- 理論橋 1 JSON: `experiments/v7_phase2a_theory_bridge_synth.py`
- 論文比較表: `experiments/v7_phase2a_paper_tau_comparison.py`
- 実データ: `experiments/v7_phase2a_empirical_run.py`
- スイート: `experiments/v7_run_suite.py --with-theory-bridge`

---

## 5. 品質・回帰

- **CI**: `--demo` 経路でスモーク（現行 `.github/workflows/ci.yml`）。
- **本番再現**: マニフェスト・`windows.jsonl` ハッシュ・モデル ID をログに残す。
- **スキーマ**: 新レールは `schema_version` を分け、後方互換でない変更は revision を上げる。

---

## 6. リスク（再掲）

- 指標を増やすほど **τ* 実証したように読まれる** → ラベルとキャプションで必ず区別。
- **理論 τ* 未確定**のまま乖離％だけ出す → 主張が空。**参照値と導出の出所**を先に固定。
- **コーパスと合成を同一図で混線** → 査読コストが増大。

---

*改訂時は `v7_phase2a_prereg_v1.json` の `implementation_master_plan_md` と整合し、必要なら revision を上げる。*
