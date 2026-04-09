# Resonanceverse理論 SLM実装方針
## モデル選定・日本語対応・実装コスト

**著者**: 加納 智之（Tomoyuki Kano）
**所属**: ZYX Corp 株式会社 人工叡智研究室（Artificial Sapience Lab）
**連絡先**: tomyuk@zyxcorp.jp
**ORCID**: 0009-0004-8213-4631
**文書ID**: RVT-IMPL-2026-008-SLM
**対応理論**: Resonanceverse Theory（RVT-2026-008-V8.0）
**対応実装計画書**: RVT-IMPL-2026-008
**作成日**: 2026-04-08T06:54:17+09:00
**ステータス**: Confirmed

---

## 0. 方針の要旨

Resonanceverse理論の三要素（W_asym・awai_vector・R_oboro）をSLMに実装するにあたり、モデルの選定・日本語対応・実装コストの三軸から方針を整理する。

推奨は**段階的アプローチ**を軸とし、**Qwen2.5-3Bで先行着手**しGemma 4への移植を後続とする。Qwen2.5とGemma 4のW_asym実装は層識別ロジックの追加のみで移植できるため、実装コストと実験の反復サイクルを両立できる。

---

## 1. モデル選定

### 1.1 推奨：Qwen2.5-3B（第一候補）

| 項目 | 値 |
|------|-----|
| アーキテクチャ | GQA標準、全層均質 |
| コンテキスト長 | 128K トークン |
| VRAM目安 | float16: 約6GB |
| W_asym実装コスト | **低い**（層識別不要・均質） |
| 日本語品質 | **高い**（多言語学習・18兆トークン） |
| ライセンス | Apache 2.0 |
| 処理速度 | 基準（× 1.0） |

**選定理由**：全層が均質なGQAアーキテクチャであり、グローバル層とスライディング層の識別が不要だ。W_asym抽出のコードが最もシンプルになる。128Kコンテキストにより実験C（20ターン以上の長期対話）に対応できる。MRMPコーパスが日本語対話である点で、Qwen2.5の多言語学習データは直接的な優位性を持つ。

### 1.2 代替A：Qwen2.5-7B（品質優先）

| 項目 | 値 |
|------|-----|
| アーキテクチャ | GQA標準、全層均質（3Bと同一構造） |
| コンテキスト長 | 128K トークン |
| VRAM目安 | float16: 約14GB / 4bit量子化: 約5GB |
| W_asym実装コスト | **低い**（3Bと同一コード） |
| 日本語品質 | **高い**（3Bより豊富な表現） |
| 処理速度 | 基準 × 2〜2.5倍 |

**3Bから7Bに移行する条件**：実験AでφのGoodness（3軸以上で |r| > 0.2）が得られない場合、または軸別awai_vectorの分離が不十分と判断された場合。コードは3Bと同一であり、`model_name` の変更のみで移行できる。

### 1.3 代替B：Gemma 4 E4B（移植先候補）

| 項目 | 値 |
|------|-----|
| アーキテクチャ | ハイブリッド（SWA+Global交互） + PLE |
| コンテキスト長 | 128K トークン |
| VRAM目安 | 約8GB |
| W_asym実装コスト | **高い**（グローバル層識別 + PLE考慮が必要） |
| 日本語品質 | 良好（140言語以上対応） |
| Day 0確認事項 | 多い（4項目） |

**Gemma 4 固有の実装課題**については第4節で詳述する。

### 1.4 代替C：Gemma 4 26B A4B（理論整合性最大）

| 項目 | 値 |
|------|-----|
| アーキテクチャ | MoE + ハイブリッドアテンション |
| コンテキスト長 | 256K トークン |
| VRAM目安 | 推論時4B相当（約8GB） |
| グローバル層 | sliding_window_pattern=6 → インデックス [5,11,17,23,29] |
| head_dim | Global: 512 / Sliding: 256（差異への対応が必要） |

**理論との整合性**：グローバルアテンション層が明示的に存在し、「話者全体にわたる関係性の非対称性」を捉えるW_asym抽出が理論的に最も整合する。ただし実装コストが高く、Day 0確認事項が多いため実験の先行着手には適さない。

### 1.5 非推奨：TinyLlama 1.1B

コンテキスト長が2048トークンに制限されており、実験C（20ターン以上の長期対話）で致命的な制約になる。日本語学習データが不十分であり、MRMPコーパスへの使用は適さない。

---

## 2. モデル比較表

| 評価軸 | Qwen2.5-3B | Qwen2.5-7B | Gemma 4 E4B | Gemma 4 26B A4B | TinyLlama 1.1B |
|--------|-----------|-----------|------------|----------------|----------------|
| W_asym実装コスト | ★★★ 低 | ★★★ 低 | ★ 高 | ★ 高 | ★★★ 低 |
| 長期対話対応 | ◎ 128K | ◎ 128K | ◎ 128K | ◎ 256K | ✗ 2K |
| 日本語品質 | ◎ 多言語強 | ◎ より豊富 | ○ | ○ | △ |
| 処理速度 | ◎ 基準 | ○ 2〜2.5倍遅 | ○ | ◎ 4B相当 | ◎ 最速 |
| φ妥当性の期待値 | ○ | ◎ ヘッド多 | ◎ Global層明確 | ◎ Global層明確 | △ ヘッド少 |
| Day 0確認事項 | 少ない | 少ない | 多い（4項目） | 多い（4項目） | 最少 |
| 理論的整合性 | ○ 均質GQA | ○ 均質GQA | ◎ Global層が対応 | ◎ 同上+大文脈 | △ |

---

## 3. 段階的実装アプローチ

```
Qwen2.5-3B で実験A〜D
        ↓
φ妥当性確認（実験A通過）
        ↓
7Bへの変更判断（φが弱い場合）
        ↓
Gemma 4への移植検討（実験A〜D完了後）
```

**移植コスト**：Qwen2.5とGemma 4の差分は `get_global_indices()` という関数一つだ。Qwen2.5の実装が動いた段階でGemma 4への移植は軽微な作業になる。

---

## 4. W_asym実装コスト詳細

### 4.1 Qwen2.5（低コスト）

全層が均質なGQAであり、層識別が不要だ。

```python
def extract_W_asym_qwen(model, tokens, mask_A, mask_B):
    outputs = model(tokens, output_attentions=True)
    W_list = []
    for attn in outputs.attentions:  # 全層を均等に使用
        # GQAの場合: num_key_value_heads < num_attention_heads
        # repeat_interleaveでKVヘッドを展開してから処理
        n_heads = model.config.num_attention_heads
        n_kv    = model.config.num_key_value_heads
        if n_kv < n_heads:
            n_rep = n_heads // n_kv
            attn = attn.repeat_interleave(n_rep, dim=1)

        A_AB = attn[:, :, mask_A, :][:, :, :, mask_B]
        A_BA = attn[:, :, mask_B, :][:, :, :, mask_A]
        S = A_AB - A_BA.transpose(-1, -2)
        W_list.append(S)
    return torch.stack(W_list).mean(dim=0)
```

### 4.2 Gemma 4（高コスト・移植時に対応）

グローバル層のみから抽出する。Day 0でconfig.jsonの実値を確認してから実装する。

```python
# Day 0確認後に記述する
def get_global_indices(config):
    # sliding_window_pattern の実値を要確認
    pattern = config.sliding_window_pattern
    return [i for i in range(config.num_hidden_layers)
            if (i + 1) % pattern == 0]

def extract_W_asym_gemma4(model, tokens, mask_A, mask_B):
    cfg   = model.config.text_config  # E4Bはネスト構造
    g_idx = get_global_indices(cfg)
    outputs = model(tokens, output_attentions=True)

    W_list = []
    for i in g_idx:  # グローバル層のみ
        attn = outputs.attentions[i]
        if attn is None:
            continue  # 共有KVキャッシュ層はNoneの可能性
        A_AB = attn[:, :, mask_A, :][:, :, :, mask_B]
        A_BA = attn[:, :, mask_B, :][:, :, :, mask_A]
        S = A_AB - A_BA.transpose(-1, -2)
        W_list.append(S)
    return torch.stack(W_list).mean(dim=0)
```

### 4.3 Gemma 4 固有の確認事項（Day 0）

Gemma 4を使用する場合、以下をすべて確認してから実装に着手する。

- `config.json` の `sliding_window_pattern` 実値（グローバル層インデックスの計算に使用）
- `global_head_dim` の実値（E4BとE4B A4Bで異なる可能性）
- `output_attentions=True` でグローバル層のアテンション行列が `None` でないことの確認
- E4BのPLE変調がアテンション抽出に影響しないことの確認

**不確実性の表明**：W_asymがアテンション計算後のPLE変調前の状態を表すことを実験記録に明示する。これは実装を妨げないが、W_asymが何を表しているかの解釈に影響する。

---

## 5. 日本語対応の観点

### 5.1 MRMPコーパスとの適合性

MRMPコーパス（windows.jsonl、3146サンプル）が日本語対話であることを前提として各モデルの日本語適合性を評価する。

**Qwen2.5**：18兆トークンの学習データに日本語・中国語・英語を含む多言語データが豊富に含まれる。MRMPコーパスの処理において、軸別キーワードスコアの精度とφの写像精度が最も高いと予測される。実験Bの軸別アノテーション（日本語6軸評価）への応答品質も期待できる。

**Gemma 4**：140言語以上の学習データを持ち日本語対応は良好だが、Qwen2.5ほどの日本語特化ではない。実験Bでの日本語6軸評価応答の品質は確認が必要だ。

**TinyLlama**：英語中心の学習データ。日本語での軸別キーワード検出・関係性推論タスクへの対応が不十分。日本語MRMPコーパスへの使用は非推奨とする。

### 5.2 Qwen2.5派生の日本語対応版

Qwen2.5をベースとした日本語特化の派生モデルが複数公開されている。Resonanceverse実装への適合性という観点で評価を記す。

**Flux-Japanese-Qwen2.5-32B-Instruct-V1.0（Deep-Analysis-Research）**

Qwen2.5-32B-Instructをベースに日本語知識・推論・言語性能を強化したモデル。Open LLM Japanese LLM Leaderboardにおいて、元のQwen2.5-32B-Instruct（0.6553）を大きく上回るスコア0.7417を記録している。特に基礎分析（FA）・要約（SUM）・コード生成（CG）で顕著な改善が確認されており、英語性能の低下は元モデルとの差1%未満とほぼ無視できる水準だ。ライセンスはApache 2.0。

Resonanceverse実装への評価としては、日本語性能の向上幅が大きく、将来的に**実験BのLLMジャッジ（自動アノテーション）として使用する候補**として位置づけられる。実験Bでは日本語6軸評価のスコアを自動生成するためにLLMジャッジを使用するが、日本語対話の微細なニュアンスを捉えるためにこのモデルが有効になりうる。

ただし32Bという規模はW_asymを抽出するベースモデルとしては過剰であり、**実験用のベースモデルとしては使用しない**。W_asym実装の先行着手はQwen2.5-3Bで行い、日本語LLMジャッジとしての採用はフェーズB-1の結果を見てから判断する。

**TinySwallow-1.5B-Instruct（SakanaAI）**

Qwen2.5-32BからQwen2.5-1.5BへのTAID（知識蒸留）で作成された日本語特化モデル。アーキテクチャはQwen2.5-1.5Bと同一であり実装コストは変わらない。ただし研究・開発目的のみの提供であり商用利用不可のため、ZYX Corp.での実用展開を念頭に置く場合は使用できない。

### 5.3 実験Bへの影響

実験B（6軸awai_vectorの地形検証）では、日本語発話テキストから6軸の転換場面を抽出し、awai_vectorとアノテーション評価の相関を計算する。この実験において日本語品質は直接的に結果に影響する。

Qwen2.5-3Bで先行着手する根拠の一つは、日本語MRMPコーパスへの適合性がTinyLlamaやgpt2より明確に高い点にある。フェーズB-1の自動評価においてLLMジャッジとしてFlux-Japanese-Qwen2.5-32Bを採用するかどうかは、3Bでの実験Aの結果を確認してから判断する。

---

## 6. 処理速度と実験設計への影響

### 6.1 3B vs 7Bの処理時間

7Bは3Bの約2〜2.5倍の処理時間がかかる。実験A〜Dで3146サンプルを処理する場合の概算を示す。

```
1ターンあたりの推論時間（GPU環境、概算）:
  Qwen2.5-3B: 約0.5〜1秒
  Qwen2.5-7B: 約1〜2秒

実験A: 3146サンプル × 平均10ターン
  3B: 約5〜8時間
  7B: 約10〜16時間
```

実験の修正→再実行のサイクルを考えると、3Bの方が扱いやすい。φの妥当性が確認された後に7Bへ移行するかを判断する。

### 6.2 R_oboroの固着度計算への影響

R_oboroの固着度計算はlogitsのコサイン類似度に基づく。モデルの語彙サイズが大きいほど計算量が増えるが、これはW_asym抽出と比較して支配的なコストではない。

---

## 7. Day 0 確認チェックリスト

### 7.1 Qwen2.5-3B（推奨・先行着手）

```
□ pip install transformers torch — Qwen2.5のロード確認
□ output_attentions=True で全層のアテンション行列が返ることを確認
□ config.num_key_value_heads vs num_attention_heads を確認
  （GQA展開の必要性）
□ MRMPコーパス（windows.jsonl）の話者構造確認
  — 有効サンプル数・平均ターン数・20ターン以上のサンプル数
□ 合成対話データ（10サンプル）でW_asym抽出パイプラインの動作確認
□ 日本語対話テキストでの軸別キーワードスコアの挙動確認
```

### 7.2 Gemma 4（移植時）

```
□ config.json の sliding_window_pattern 実値確認
□ global_head_dim の実値確認（E4B / 26B A4B で異なる可能性）
□ output_attentions=True でグローバル層が None でないか確認
□ PLEの介在がアテンション抽出に影響しないことの確認
□ 共有KVキャッシュ層のアテンション行列形状確認
□ transformers 通常インストールで Gemma 4 が動作するか確認
  （ソースインストール不要であれば不要）
```

---

## 8. 実装環境設定（更新版）

RVT-IMPL-2026-008 の環境設定を以下のように更新する。

```
ベースモデル（SLM版・第一候補）: Qwen2.5-3B-Instruct
ベースモデル（SLM版・代替A）  : Qwen2.5-7B-Instruct
ベースモデル（SLM版・移植先） : Gemma 4 E4B または 26B A4B
ベースモデル（API版）         : Claude API（claude-sonnet-4-20250514）
実行環境                      : SLMローカル（ネットワーク遮断）
フレームワーク                 : PyTorch + HuggingFace Transformers
コンテキスト長                 : 128K（Qwen2.5-3B/7B）
乱数シード                     : 42（全実験共通）
```

---

## 付録A：モデルアーキテクチャ詳細

### Qwen2.5-3B

```
vocab_size           : 151,936
hidden_size          : 2,048
num_hidden_layers    : 36
num_attention_heads  : 16
num_key_value_heads  : 8    ← GQA（展開処理が必要）
head_dim             : 128
max_position_embeddings: 131,072 (128K)
sliding_window       : なし（全層均質）
ライセンス           : Apache 2.0
```

### Flux-Japanese-Qwen2.5-32B-Instruct-V1.0（LLMジャッジ候補）

```
ベースモデル         : Qwen2.5-32B-Instruct
用途               : 実験BのLLMジャッジ（フェーズB-1自動アノテーション）
日本語Leaderboard   : 0.7417（元モデル0.6553比 +0.0864）
英語性能への影響     : 元モデルとの差 < 1%
ライセンス          : Apache 2.0
位置づけ           : W_asym実装のベースモデルとしては使用しない
                    日本語対話の微細なニュアンス評価に活用
```

### Gemma 4 26B A4B（参考）

```
num_hidden_layers    : 30
num_attention_heads  : 8
num_key_value_heads  : 4
head_dim（Sliding）  : 256
global_head_dim      : 512
sliding_window_pattern: 6  → グローバル層: [5,11,17,23,29]
max_position_embeddings: 131,072 or 262,144
ライセンス           : Apache 2.0
```

---

## 付録B：φ妥当性確認が失敗した場合の対応

実験Aでφの妥当性（3軸以上で |r| > 0.2）が確認されない場合、以下の順で対応する。

**ステップ1**：Qwen2.5-7Bへの移行。ヘッド数の増加によりS_asymのパターン多様性が増し、6軸への射影精度が向上する可能性がある。

**ステップ2**：射影行列Mの初期化戦略の見直し。ランダム初期化ではなく、関係性アノテーション付きサブセットによる教師ありファインチューニングでMを学習する。

**ステップ3**：Gemma 4への移植。グローバルアテンション層はスライディング層より長距離の関係性を捉えるため、φの写像精度が向上する可能性がある。

**ステップ4**：スカラーS_asymへの退行。6軸分解を放棄し、スカラーのS_asymとawai_signalによる実装に切り替える。この場合、実験Dにおけるスカラー版との比較が成立しなくなるため、実験設計の修正が必要になる。

---

**文書情報**

文書ID: RVT-IMPL-2026-008-SLM
作成日: 2026-04-08T06:54:17+09:00
最終更新: 2026-04-08T07:10:00+09:00
著者: 加納 智之（Tomoyuki Kano）
所属: ZYX Corp 株式会社 人工叡智研究室

Copyright © 2026, Tomoyuki Kano / ZYX Corp. All rights reserved.
