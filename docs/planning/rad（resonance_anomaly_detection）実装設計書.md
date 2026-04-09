---
title: "🛠️ RAD（Resonance Anomaly Detection）実装設計書"
created: 2026-04-08T00:00:00
updated: 2026-04-08T00:00:00
author: "Tomoyuki Kano <tomyuk@zyxcorp.jp>"
tags: [AI, Resonanceverse, RAD, RAG, Kotone, architecture, anomaly_detection, governance]
links: []
---

# 1. 概要

本設計書は、KotoneにおけるRAG拡張としてのRAD（Resonance Anomaly Detection）の実装仕様を定義する。

RADは単なる異常検知ではなく、Resonanceverseにおける「共鳴の歪み」を検出し、ResoLLMの停止判断および再探索戦略を支援する中核コンポーネントである。

---

# 2. 目的

RADの目的は以下の3点に集約される：

1. 誤った共鳴（False Resonance）の検出
2. ResoLLMによる停止・縮退判断の入力提供
3. RAG戦略の動的切り替え（探索 → 反証探索）

---

# 3. システム位置

```
[User Input]
   ↓
[SLM Generate]
   ↓
[RAG Retrieve]
   ↓
[RAD Analyze]
   ↓
[ResoLLM Evaluate]
   ↓
[Stop / Continue / Re-query]
```

RADはRAGとResoLLMの間に位置するが、実装上はRAGの拡張層として統合される。

---

# 4. コア概念

## 4.1 False Resonance

```
False Resonance =
  High Coherence
  × Low Grounding
  × Biased Relation
```

---

## 4.2 RADスコア

```
RAD_score =
  w1 * SemanticDrift
+ w2 * UncertaintyGap
+ w3 * RelationBias
+ w4 * ValueConflict
```

---

# 5. 特徴量設計

## 5.1 Semantic Drift

- embedding距離（query vs response）
- 定義不一致検出
- 文脈逸脱率

実装例：
- cosine similarity
- topic shift detection

---

## 5.2 Uncertainty Gap

- 確信度 vs 根拠量
- 未検証主張比率

指標：
- evidence count
- confidence calibration error

---

## 5.3 Relation Bias

- ソース多様性
- relation graphの偏り

指標：
- source entropy
- graph centralization

---

## 5.4 Value Conflict

- ポリシー違反
- 倫理的逸脱

指標：
- policy violation score
- constraint mismatch

---

# 6. データ構造

## 6.1 RAD Input

```
RADInput {
  query
  response
  retrieved_docs
  relation_context
  value_context
}
```

---

## 6.2 RAD Output

```
RADOutput {
  score: float
  components: {
    semantic_drift
    uncertainty_gap
    relation_bias
    value_conflict
  }
  flags: ["drift", "bias", "conflict"]
}
```

---

# 7. 処理フロー

```
1. 特徴量抽出
2. 各スコア計算
3. 重み付け統合
4. 閾値判定
5. フラグ生成
```

---

# 8. RAG統合

## 8.1 通常モード

- relevanceベース検索

## 8.2 RAD発火モード

- 多様性重視
- 反証検索
- 競合仮説取得

```
if RAD_score high:
    strategy = "diversity + contradiction"
```

---

# 9. ResoLLM連携

```
if RAD_score > θ_critical:
    HardStop
elif RAD_score > θ_high:
    SoftStop + ReQuery
elif RAD_score > θ_mid:
    DegradedMode
else:
    Continue
```

---

# 10. PoP連携

> **【将来構想・本実装スコープ外】**  
> 以下の PoP（Proof of Personhood / 承認ゲート）連携は、RAD の初版・PoP 非搭載フェーズでは**実装しない**。ResoLLM 側の停止・再クエリのみで完結させ、PoP はガバナンス層が成熟した段階で追加する。

```
if RAD_score high:
    require PoP approval
```

---

# 11. 学習・適応

## 11.1 重み更新

- UIB評価に基づく調整

## 11.2 閾値学習

- false positive / negative最小化

---

# 12. API設計

## 12.1 analyze

```
analyze(input: RADInput) -> RADOutput
```

---

## 12.2 feedback

```
feedback(score, outcome)
```

---

# 13. 実装方針

- Rust：高速処理・安全性
- Python：特徴量計算・ML
- GraphDB：relation bias計算
- Vector DB：semantic drift

> **【デプロイ前提】** 初版および当面の設計考察では **ブラウザを実行環境の前提としない**（クライアント Wasm・ブラウザ内 RAD 等は **保留・将来検討**。§17.3 および §17.5 参照）。

---

# 14. 拡張性

- UIB完全統合
- DAOフィードバック
- 自己進化（Auto Learning）

---

# 15. 最終定義

```
RAD =
  System for Detecting and Correcting
  Non-sustainable Resonance
```

---

# 16. 次ステップ

1. 特徴量の数式定義
2. relation_store統合
3. ResoLLMとの完全インターフェース設計
4. **（PoP 連携は §10 参照・将来フェーズ）**
5. **（ブラウザ／Wasm 前提の配置は保留・§17.5 参照）**

---

# 17. 実装手順（考察）

## 17.1 スコープの切り分け

| 領域 | 初版（PoP なし）で実装 | 将来・別フェーズ |
|------|------------------------|------------------|
| `RADInput` / `RADOutput`・`analyze()` | ○ | — |
| §5.1 Semantic Drift | ○（埋め込み＋コサイン等から開始） | より精緻な topic モデル |
| §5.2 Uncertainty Gap | ○（根拠件数・ヒューリスティック） | キャリブレーション学習 |
| §5.3 Relation Bias | ○（ソース多様性＝エントロピー等） | GraphDB・中央性 |
| §5.4 Value Conflict | ○（ルール／キーワードまたは定数スタブ） | 専用分類器・ポリシーエンジン |
| §8 RAG 戦略切替 | ○（閾値でモード文字列または検索パラメータ差し替え） | 高度な反証パイプライン |
| §9 ResoLLM 連携 | ○（列挙＋閾値マッピングの API） | 学習済みポリシー |
| **§10 PoP** | **実装しない（§10 注記）** | PoP 承認フロー統合 |
| **ブラウザ／Wasm** | **対象外（§17.5）** | エッジ・クライアント実行の検討 |

## 17.2 推奨フェーズ順（PoP 除く）

**フェーズ A — 契約とパイプライン空転**

1. `RADInput` / `RADOutput`（§6）をコード上のデータクラスまたは TypedDict で固定。
2. `analyze(input) -> RADOutput` を実装し、内部はコンポーネントを `0.0` 固定またはダミーでもよい（§12.1）。
3. 単体テスト：必須フィールド欠損・空 `retrieved_docs` 時の挙動。

**フェーズ B — Semantic Drift（MVP の中核）**

1. `query` と `response` の埋め込み取得（既存の埋め込みモデルまたは軽量 API）。
2. コサイン類似度 → §4.1 の「High Coherence と Low Grounding」の代理として、**低類似度を drift 寄り**にマッピングするなど、数式を README または §5.1 補足に 1 ページで固定。
3. `components.semantic_drift` に正規化スカラー（例: `[0,1]`）を出力。

**フェーズ C — Uncertainty Gap**

1. `retrieved_docs` の件数・総文字数・「引用らしきスパン」の有無など、ルールベースで `evidence_strength` を定義。
2. モデル側に logprobs / 確信度が取れる場合は差分を `uncertainty_gap` に反映（無ければ証拠不足ヒューリスティックのみ）。

**フェーズ D — Relation Bias（グラフなし版）**

1. 取得ドキュメントの `source_id` 分布からエントロピーを計算（単一ソース支配 → 高バイアス寄り）。
2. §13 の GraphDB は**後追い**；初版は「ソース列のみ」で `relation_bias` を埋める。

**フェーズ E — Value Conflict（最小）**

1. `value_context`（ポリシー短文・禁止語リスト）と `response` の照合スコア。
2. 未整備時は `0.0` 固定でインターフェースだけ残し、フラグは `conflict` を立てない。

**フェーズ F — RAD_score・閾値・flags**

1. §4.2 の重み `w1..w4` を設定ファイル化（YAML 等）。初期は均等でも可。
2. §7 の 4–5 を実装：`score` 集計、`θ_mid` / `θ_high` / `θ_critical` で `flags` 配列（`drift` 等）を付与。
3. §12.2 `feedback` はログ追記のみでも可（学習は §11 を将来扱い）。

**フェーズ G — RAG 統合（§8、PoP なし）**

1. `RAD_score` が閾値を超えたとき `strategy = "diversity + contradiction"` 相当の**検索パラメータ**（top_k 増加、MMR、否定クエリ用の 2 本目 retrieve 等）に切り替えるフックを 1 箇所に集約。
2. 「通常モード」との分岐を設定でオンオフ可能にする。

**フェーズ H — ResoLLM 連携（§9、PoP なし）**

1. `RADOutput` から `HardStop` / `SoftStop+ReQuery` / `DegradedMode` / `Continue` を返す**純関数**（PoP 分岐は入れない）。
2. 呼び出し側（オーケストレータ）が実際の停止・再生成を実行。

## 17.3 実装しないもの（明示）

- **§10 PoP 承認**：高 RAD 時の人間／身元ゲートは**将来構想**。初版の `if RAD_score high: require PoP` はコードに含めない。
- **§11 自動重み・閾値学習**：ログとオフライン分析から着手し、オンライン更新は後フェーズ。
- **§13 の全面 Rust 化・GraphDB 必須**：初版は Python モジュール＋オプションのベクトルストアで足りる想定。Rust / GraphDB は性能・関係グラフがボトルネックになった段階で検討。

## 17.4 設計書とのトレーサビリティ

| 設計書節 | 初版で触る実装成果物 |
|----------|----------------------|
| §3, §7 | オーケストレータ内の `RAD → 分岐` 1 パス |
| §5.1–5.4 | `rad/features/*.py` 相当のモジュール分割 |
| §6 | `rad/types.py` または `schemas/rad.py` |
| §8 | RAG クライアントの `strategy` 引数 |
| §9 | `rad/policy.py`（閾値マップのみ、PoP なし） |
| §10 | **ドキュメントのみ（将来）** |
| §12 | `analyze` / `feedback` の公開 API |
| §17.5 | **スコープ外（ブラウザ／Wasm は保留）** |

---

## 17.5 ブラウザ前提・Wasm（保留）

今回の RAD 設計・実装手順の**スコープから除外**する。

- **除外する想定**：ブラウザ上で Wasm として RAD コアを動かす、クライアント完結型の異常検知、Service Worker 内での重い特徴量計算など。
- **当面の想定配置**：サーバー側（またはローカルバックエンド）の **Python／Rust プロセス・API** とオーケストレータによる呼び出し（§13）。
- **保留の扱い**：プライバシー重視のオンデバイス RAD 等が必要になった段階で、Wasm ターゲット・WASI・ブラウザ API 制約を別途設計する。

