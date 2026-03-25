# Phase B データ・評価プロトコル（固定化 v0.1）

## 目的

`AwaiIntegratedSLM`（HF 因果 LM ＋ `ResonantCore` ＋語彙ヘッド）について、**再現可能な**データ経路と **perplexity** 定義を固定する。下流タスク（分類・QA）は本稿のスコープ外（別途登録）。

## データソース

| モード | スクリプト引数 | 内容 |
|--------|----------------|------|
| ランダム | `--data random`（`--demo` 時は固定） | 一様乱数トークン。オフライン・CI・アルゴリズム健全性のみ。 |
| Wikitext | `--data wikitext` | `datasets` の `wikitext/wikitext-2-raw-v1` **train** 先頭を、`--max-chars` 文字まで読み、**行（段落）ごとに** `encode` してトークン ID を連結（長文一括 `encode` はしない）。トークン上限は `max_tokens`（未指定時は `max_chars` から推定、最大 10 万）。 |

初回は Hugging Face Hub からの取得が必要。キャッシュは `datasets` の既定キャッシュに依存する。

## 前処理

1. **ランダム**は一様乱数で 1D トークン列。**Wikitext** は行ごとに `tokenizer.encode(..., add_special_tokens=False)` し、得た ID を連結（長文一括 `encode` はしない）。トークン上限は `max_tokens`（未指定時は `max_chars` から推定、最大 10 万）で切り詰め。  
2. 末尾 `eval_frac`（既定 **0.1**）を**評価用**に分離（時系列は連続文字列なので、末尾ブロックを eval とみなす簡易分割）。  
3. train / eval それぞれを `seq_len` トークンの非重複チャンクに分割（端数は捨てる）。  
4. 学習は `batch` ごとのミニバッチを、チャンク列を**循環**して供給（`max_steps` 回）。

## 損失と perplexity

- **学習損失**: 因果 LM の**次トークン**クロスエントロピー。`logits[:, :-1]` と `input_ids[:, 1:]` を対応付ける（`experiments/slm_resonance_lm.py` の `causal_lm_loss`）。  
- **Perplexity**: 評価チャンクについて、バッチごとの CE（トークン平均）を出し、バッチ間で**単純平均**して \( \mathrm{NLL} \) とし、**\( \exp(\mathrm{NLL}) \)** を報告する（`--eval-ppl`）。  
  - 厳密な「全トークンでの加重平均」ではないが、**同一スクリプト内で比較**する用途に足る。

## 再現性

- `set_experiment_seed`（`--seed`）を使用。  
- ランダムデータは再現性検証用にのみ使い、**数値比較は Wikitext 等の固定データ**で行うことを推奨。

## 関連

- 実装: [`experiments/slm_resonance_lm.py`](../../../experiments/slm_resonance_lm.py), [`experiments/slm_data.py`](../../../experiments/slm_data.py)  
- [モジュール↔実験対応表](module_benchmark_map.md)

---

*2026-03-25: v0.1*
