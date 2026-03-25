# Phase B 下流タスク・評価プロトコル（固定化 v0.5）

## 目的

因果 LM の [Phase B データ・評価プロトコル](phase_b_data_protocol.md) に加え、**分類・読解 QA（二値）**で Resonanceverse 統合（`ResonantCore` 等）とベースラインを**同一手順**で比較するための入口を固定する。

## 実装

| 項目 | 内容 |
|------|------|
| スクリプト | [`experiments/slm_downstream.py`](../../../experiments/slm_downstream.py) |
| `--task sst2` | **GLUE SST-2**（二値感情分類）— `datasets.load_dataset("glue", "sst2")` |
| `--task boolq` | **BoolQ**（Yes/No 読解 QA）— `datasets.load_dataset("boolq")`（`datasets` の GLUE ビルダに `boolq` が無いためスタンドアロン）。入力は `"{question} {passage}"` を単一シーケンスとしてトークナイズ。長文は `--max-seq-len` で切り詰め（推奨 256〜512）。 |
| ベースライン | `transformers.AutoModel` の **last_hidden_state** をマスク平均プール → `Linear(H, 2)` |
| Awai 経路 | 同一エンコーダ隠れ状態 → **`ResonantCore`** → マスク平均プール → **読み出し**（下表） |
| 読み出し `--awai-readout` | `narrow`: **`Linear(6, num_labels)`**（従来）。`projected`: **`Linear(6, H)`** → **`Linear(H, num_labels)`**。`dual`: **encoder プール `(H,)`** と **共鳴プール `(6,)`** を連結 → **`Linear(H+6, num_labels)`**。baseline では未使用。 |
| オプション | `--integration awai-cultural` で `CulturalModulationAdapter` による共鳴特徴の調製 |

品質 τ（検証 accuracy）の**事前登録**および **T3″**（同一学習条件での読み出し固定）は [Phase C 品質 τ 事前登録](phase_c_quality_tau_prereg.md) を参照。

JSON には **`task`**（例: `glue_sst2`, `glue_boolq`, `synthetic`）に加え、CLI の `--task` を **`glue_task`**（`sst2` / `boolq`）として記録する。Awai 系では **`awai_readout`**（`narrow` / `projected` / `dual`）を記録する。

## 手順（固定）

1. `--seed` で `set_experiment_seed` を適用。  
2. 訓練は `max_train_samples`（0 なら訓練分割の全件）を `shuffle(seed)` 後に先頭から切り出し。  
3. 学習は **ランダムミニバッチ**（`max_steps` 回、`batch_size` ごとに `randperm` 先頭からサンプル）。  
4. 評価は **検証分割**（`max_eval_samples` が 0 のときは検証分割の全件）で **ミニバッチ集約 accuracy**。  
5. オフライン／CI: `--demo`（合成データ・`_DemoEncoder`）で `--integration` のみ検証（`glue_task` は CLI どおり記録可）。

## 指標

- **accuracy_eval**: 検証分割の分類精度（主報告）。  
- **final_loss**（訓練 CE）: 最終ステップのミニバッチ損失（比較の補助）。

## 抽出型 QA（SQuAD v1・別エントリ）

| 項目 | 内容 |
|------|------|
| スクリプト | [`experiments/squad_span.py`](../../../experiments/squad_span.py) |
| タスク | **SQuAD v1**（開始・終了トークン）— `datasets.load_dataset("squad")` |
| モデル | `AutoModelForQuestionAnswering`（既定 `distilbert-base-uncased`） |
| オフライン／CI | `--demo`（合成データ・`MiniSpanQA`、`transformers` 不要） |
| JSON | **`schema_version`**: `squad_span.v1`；**`exact_match_eval`**（`--max-eval-samples` 指定時）、**`final_loss`** |

因果 LM ＋単一線形ヘッドの [`slm_downstream.py`](../../../experiments/slm_downstream.py) とは**別プロトコル**（損失・ヘッドが異なる）。

## 短いランダムステップの限界

学習は**短いランダムステップ**であり、フル finetune の再現ではない。`slm_downstream` では同一スクリプト内で **baseline / awai / awai-cultural** を比較する用途を主とする。

## 関連

- [Phase B データ・評価プロトコル（LM）](phase_b_data_protocol.md)  
- [Phase C 品質 τ 事前登録](phase_c_quality_tau_prereg.md)（T3′／T3″）  
- [モジュール ↔ 実証スクリプト対応表](module_benchmark_map.md)

---
*v0.5 — 2026-03-25（BoolQ: `load_dataset("boolq")` スタンドアロン）*  
*v0.4 — 2026-03-25（`--awai-readout`・読み出し三種・JSON `awai_readout`）*  
*v0.3 — 2026-03-25（抽出型 QA: `squad_span.py`・`squad_span.v1`）*
