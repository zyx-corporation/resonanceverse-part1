---
title: Phase I-A 用コーパス — MRMP のみ（v7）
created: 2026-03-29
language: ja
document_type: planning
---

# Phase I-A 用コーパス — Multi-Relational Multi-Party Chat Corpus（MRMP）のみ

本稿は **MRMP 以外のコーパス候補を扱わず**、取得・仕様・Resonanceverse Phase I-A との対応だけを固定する。

## 正本リンク

| 種別 | URL |
|------|-----|
| Hugging Face Datasets | https://huggingface.co/datasets/nu-dialogue/multi-relational-multi-party-chat-corpus |
| GitHub | https://github.com/nu-dialogue/multi-relational-multi-party-chat-corpus |
| ライセンス | **CC BY-SA 4.0**（派生物の配布条件に注意） https://creativecommons.org/licenses/by-sa/4.0/deed.ja |

## コーパス概要（公式要約）

- 日本語の**マルチパーティ雑談**に着目。約 **1,000 対話**（初対面と家族入りなど関係性の違いを含む）。
- 対話データに **発話列**（話者 ID・テキスト・`mention_to`）、**話者ごとの評価スコア**（1〜5）が付く。別に**話者メタ**（ペルソナ・性格尺度・属性・ペア関係など）がある。
- データセットカードに記載の**利用上の注意**（個人特定・なりすまし禁止・属性推定時の配慮）を遵守すること。

## 取得方法

### 全文取得（推奨・リポジトリ同梱スクリプト）

公式 GitHub を**浅い clone** し、`dialogues/**/*.json` と `interlocutors.json` をローカルに揃える。保存先は既定で `experiments/logs/mrmp_repo/`（**git 管理外**）。

```bash
python experiments/fetch_mrmp_corpus.py
# 取り直す場合
python experiments/fetch_mrmp_corpus.py --force
```

完了後、`experiments/logs/mrmp_repo/fetch_meta.json` に対話 JSON ファイル数・`git rev` などが入る（**960** 対話ファイル＋話者 JSON が揃えば全文相当）。

### Hugging Face（任意・環境依存）

`datasets` のバージョンによっては、データセット**ローディングスクリプト非対応**で失敗する。成功する環境では次のように 2 サブセットを読める（Dataset Card 参照）。

```python
from datasets import load_dataset

dialogue_dataset = load_dataset(
    "nu-dialogue/multi-relational-multi-party-chat-corpus",
    name="dialogue",
    trust_remote_code=True,
)
interlocutor_dataset = load_dataset(
    "nu-dialogue/multi-relational-multi-party-chat-corpus",
    name="interlocutor",
    trust_remote_code=True,
)
```

### GitHub（手動）

リポジトリ内 `multi_relational_multi_party_chat_corpus/dialogues/` に対話 JSON、`interlocutors.json` に話者情報（README のディレクトリ構成に従う）。

## 規模・分割（HF カード記載）

| 項目 | 値（目安） |
|------|------------|
| train 対話数 | 960 |
| train 発話数 | 約 106,680 |
| validation / test | カード上は未記入（要: 利用時に最新の Dataset Card を確認） |

## フィールドと Phase I-A の対応

### そのまま使える対応

| v7 設計（A2-2 付近） | MRMP の例 |
|----------------------|-----------|
| `dialogue_id` | `dialogue_id` |
| `turn_idx` | `utterances.utterance_id`（対話内 0 始まり） |
| 話者 ID | `interlocutor_id`（文字列 ID） |
| `context_type`（文脈カテゴリ） | `dialogue_type`（例: 初対面 / 家族）へのマッピングを事前登録で固定 |

### マルチパーティ → 有序ペア (A→B)

MRMP は **3 話者**などが含まれる。**設計書の (A→B)** に落とす規則を 1 通りに固定する必要がある。例（事前登録の候補）:

- **規則 P1**: 各発話について、**発話者を src**、**直前発話の話者を tgt** とする（連鎖ペア）。
- **規則 P2**: `mention_to` が非空なら、**src→各メンション先**を 1 行ずつ生成する。

本リポジトリの JSONL スキーマ（`speaker_src`, `speaker_tgt`）に合わせるときは、上記のどちらを採用したかを**解析コードと同一の JSON に記録**する。

### v7 の「6 軸」と MRMP の評価スコア

MRMP の `evaluations`（情報量・理解度・親しみやすさ・興味・積極性・満足度）は、**対話品質・話者印象に近い多次元評価**であり、v7 の **trust / authority / proximity / intent / affect / history の 6 軸とは同一ではない**。  
次のどちらかを事前に決める:

- **A**: 6 軸は**別途人手アノテ**のみを主解析とし、MRMP 評価は補助または無関係として扱う。  
- **B**: MRMP 評価を**代理指標**として明示的に限定した補助解析のみ行う（主張の境界を文書化）。

## 本リポジトリでのデータの置き方

- **生コーパスを git に含めない**（ライセンス・容量・再配布条件のため）。`experiments/data/` には**自前で生成したサブセットや変換 JSONL のサンプル行**のみを置く運用を推奨。
- 派生物を公開する場合は **CC BY-SA 4.0** の要件（表示・同一許諾条件など）に従う。

## 引用（論文記載用）

Dataset Card / GitHub README に BibTeX が掲載されている。投稿時は **SemDial 2025** および **言語処理学会年次大会 2025** の両方が案内されているので、該当するものを使用する。

---

*改訂時は MRMP の Dataset Card・GitHub の版情報と同期すること。*
