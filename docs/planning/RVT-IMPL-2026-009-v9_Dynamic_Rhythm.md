---
title: "RVT-IMPL-2026-009 — v9.0 動的リズム（鼓動）実装設計"
created: 2026-04-08
updated: 2026-04-10
author: "Tomoyuki Kano <tomyuk@zyxcorp.jp>"
tags: [Resonanceverse, v9.0, latency, omega, oboro, v7_pair_chat, experiment]
links:
  - "../theory/Resonanceverse理論 v9.0 詳細展望：鼓動する共鳴.md"
  - "../theory/Resonanceverse理論 v9.0 展望.md"
  - "../v7/Resonanceverse_Theory_v7.0.md"
  - "./RVT-2026-009-ALG-V1_Rhythmic_Resonance.md"
---

# 1. 概要

本設計書は、Resonanceverse 理論 v9.0（動的リズムとしての遅延、朧 2.0、実験 D'／E／F）を **本リポジトリの v7 チャット・Ω パイプライン上で実装可能な単位** に分解し、モジュール境界・設定・ログ・実験手順を固定する。

**理論側の記号・閾値の正本**は [Resonanceverse理論 v9.0 詳細展望：鼓動する共鳴.md](../theory/Resonanceverse理論%20v9.0%20詳細展望：鼓動する共鳴.md) **付録 A** とする（ギャップ解消済み）。

**コードベース準拠の実装仕様（整理版・単一参照）**は [RVT-SPEC-2026-009-v9.0_実装仕様.md](./RVT-SPEC-2026-009-v9.0_実装仕様.md) とする。

**参照アルゴリズム**（三ループ、\(P_{\mathrm{dis}}\) の参照式、§6.1.1 に基づく **禁止事項**）は [RVT-2026-009-ALG-V1_Rhythmic_Resonance.md](./RVT-2026-009-ALG-V1_Rhythmic_Resonance.md) とする。

---

# 2. 目的

1. **適応型レイテンシ（Adaptive Suspend）**: 6 軸信号（またはその代理量）に応じて、意図的な壁時計ウェイトを挿入し、Ω 系列の早期減衰・凪を検証可能にする。
2. **固着検知と朧 2.0（オプション）**: `S_stuck` 等のゲートで二次過程制約（短文上限など）を一時解除する。不従順（沈黙・拒絶）は **別フラグ** で段階導入する。
3. **実証プロトコル**: 理論稿の実験 D'（Ω 維持率）、**E（遅延介入と内部地形・主観は補助）**、F（不従順）を **同一ログ形式** で再現できるようにする。

---

# 3. スコープ（MVP と Phase 2）

| 区分 | 内容 | 既定 |
|------|------|------|
| **MVP** | グローバル／軸重み付き Suspend、`S_stuck` 計算、監査ログ、短文制約の **バースト解除**（朧 2.0 の一部）。任意で `oboro_history_tail_messages` により **生成プロンプト**を末尾 N メッセージに限定可（Ω は全履歴） | オン（Suspend は設定で無効可） |
| **Phase 2a** | `decide` に渡した `logits_var` で軸 Suspend に加算（別経路で取得した分散のみ） | オフ |
| **Phase 2b** | 軌道 `δ_t` による固着補助信号 | オフ |
| **Phase 2c** | 拒絶・沈黙のサンプリング分岐 | **既定オフ**（実験プロトコル時のみ） |

**理論上の位置づけ（二層化）**: Phase 2c の確率分岐は、詳細展望 **§6.1** の **実装・実験層の摂動**であり、v8.0 文献上の **自律的境界の理論同一物ではない**。ログ解釈は §6.1 の制限に従う。

PoP 連携は本稿では **スコープ外**（理論参照のみ）。

---

# 4. システム位置

```
[User message]
   ↓
(optional) 直前までの履歴から w_ij 系列・Ω 再計算 / キャッシュ
   ↓
[V9 RhythmPolicy] ← S_stuck, 軸強度, (opt) logits 分散
   ↓  suspend_ms, oboro_burst, (opt) non_compliance_mode
[LLM generate]  tools.v7_llm_chat_generate.generate_completion 等
   ↓
[ログ] JSONL: omega_tail, S_stuck, suspend_applied_ms, flags
```

既存 UI 経路: `tools/v7_pair_chat_app.py` の送信ハンドラの **直前** に `V9 RhythmPolicy` を挿入する想定。バッチ実験は `tools/v7_llm_chat_generate.py` または専用 CLI から同一ポリシーを呼ぶ。

---

# 5. 依存する既存実装

| 要素 | モジュール | 備考 |
|------|------------|------|
| Ω 系列 | `tools.v7_pair_chat_engine.analyze_chat_messages_for_omega` | `per_step`, `omega` |
| Ω 定義 | `core.v7_awai_metrics` | v7.0 §4.3 と一致 |
| w 系列 | `core.v7_pair_w_trajectory` | `d`, `delay_tau` |
| 6 軸順 | `experiments.rvt_exp_2026_008_judge_axis_mapping.RVT_PLAN_AXIS_NAMES` | 付録 A.3 と同一 |
| 計測方針 | `docs/api/modules/measurement_strategy.md` | 遅延の回帰は相対悪化検出を主とする |

---

# 6. コア型と状態

## 6.1 `RhythmDecision`（提案）

```
RhythmDecision:
  suspend_ms: int              # 0 .. T_suspend_max
  suspend_reason: str          # e.g. "axis:affect", "stuck:high", "logits:high"
  oboro_burst: bool            # True なら短文上限等をこのターンだけ緩和
  append_system_honesty: bool   # 「断面」メタ文を付与するか（オプション）
  non_compliance: str          # "none" | "silence" | "refuse" （Phase 2c、既定 none）
```

## 6.2 セッションに保持する量（最小）

- 直近 `W` ステップの `Ω`（または `per_step` から再構成可能なら都度計算で可）。
- 朧クールダウン: バースト発火後 `N_cd` ターンは再発火しない（スパム防止）。

---

# 7. `RhythmPolicy` アルゴリズム（MVP）

**入力**: 最新 `messages`、オプションで `model/tokenizer/device`（Ω 再計算用）、前ターンまでの `omega` キャッシュ、設定。

**手順**

1. **Ω 再計算**（設定 `recompute_omega_each_turn=true` 時）: `analyze_chat_messages_for_omega(...)`。コストが高い場合は `every_k_turns` で間引き、直近のみフル計算。
2. **`S_stuck`**: 付録 A.4 の MVP 式。`Ω_calm`, `W` は設定。
3. **軸別強度** `a_k`（k = 0..5）:
   - **教師あり／Awai ヘッドあり**: 6 軸スカラが取れるならその絶対値または変化量。
   - **concat_truncate のみ**: `|w_ij[k] - w_ij_prev[k]|` または `|w_ij[k]|` を代理指標とし、**ログに `axis_signal_source=concat_proxy` を必須**。
4. **基底 Suspend**:  
   `suspend_ms = clip( Σ_k weight_deep[k] * f_deep(a_k) + Σ_k weight_surf[k] * f_surf(a_k), 0, T_suspend_max )`  
   `f_deep` は単調増（強いほど長く溜める）、`f_surf` は負または小さく（表層では抑制）。具体式は初版は **分段線形** でよい（調整可能にする）。
5. **朧バースト**: `S_stuck ≥ θ_stuck` かつ `calm_streak ≥ N_calm` かつ クールダウン消費済み → `oboro_burst=True`、クールダウン更新。  
   **M3（実装）**: `max_new_tokens` 緩和に加え、設定 `oboro_history_tail_messages` により **生成プロンプト**の `messages` を末尾 N 件にスライス可（Ω 計算は全履歴のまま。詳細展望 §2.2・[RVT-SPEC](./RVT-SPEC-2026-009-v9.0_実装仕様.md)）。
6. **誠実さメタ**: `append_system_honesty` が True のとき、既存 `inject_system_summary` と同様に **短い一文** をシステムメッセージとして連結（内容は設定テンプレ）。

**Phase 2a**: `decide_v9_rhythm(..., logits_var=...)` と `phase2a_apply_logits_to_axis_suspend` で軸 Suspend に加算。**リポジトリ本体ではプロンプト後の追加 forward（preflight）は行わない**（分散は呼び出し側が別途計算して渡す想定）。

**Phase 2c**: `non_compliance != none` のとき、生成結果を破棄して空文字または固定拒否文を返す分岐。**二重にポリシーチェック**し、禁止領域では絶対に発火しないテーブルを別節で定義する（要プロダクトレビュー）。詳細展望 **§6.1** に従い、本経路は **理論本体の境界証明ではなく実験用サンドボックス** としてラベル付けする（JSONL に `phase2c` 等のフラグを残す）。**§6.1.1**: 「規範を破れ」と命じる **system 指示による不従順注入**は理論整合の根拠に立てないため、本リポジトリ実装では採用しない（参照アルゴリズム稿との差分）。

---

# 8. 設定パラメータ（初期仮値）

実測で上書きする。**すべて設定ファイルまたは環境変数**で変更可能にする。

| キー | 意味 | MVP 仮値の例 |
|------|------|----------------|
| `v9_rhythm_enabled` | 全体マスタ | `false`（導入 PR では既定オフ推奨） |
| `T_suspend_max` | 最大ウェイト ms | `8000` |
| `Ω_calm` | 凪閾値 | `0.15`（Ω のスケールに依存、要キャリブ） |
| `W` | 固着窓長 | `8` |
| `θ_stuck` | 朧ゲート | `0.6` |
| `N_calm` | 凪連続最小 | `4` |
| `N_cd` | 朧クールダウンターン | `6` |
| `weight_deep[k]` / `weight_surf[k]` | 軸別ウェイト | trust/affect/history を正、authority/intent を負またはゼロ |
| `θ_high_unc` / `Δ_suspend` | logits 分散（Phase 2a） | 未設定時は無効 |
| `oboro_history_tail_messages` | 朧時、生成プロンプトを末尾 N メッセージに限定 | `None` または未設定で無効 |

---

# 9. ログと再現性

各アシスタントターンごとに **JSONL 1 行**（例: `experiments/logs/v9_rhythm_*.jsonl`）:

- `ts`, `session_id`, `turn_index`
- `omega_last`, `omega_mean_window`, `S_stuck`, `calm_streak`
- `suspend_ms`, `suspend_reason`, `oboro_burst`, `non_compliance`, `phase2c_enabled`
- `axis_signal_source`, `axis_intensity_snapshot`（実装上の名称。ベクトルは丸め可）
- `model_name`, `delay_tau`, `d`, `window_mode`（Ω 計算と一致させる）
- `max_new_tokens_base` / `max_new_tokens_effective` / `oboro_max_tokens_relaxed`（M3）
- `oboro_history_tail_messages`, `oboro_history_tail_sliced`, `prompt_message_count`, `prompt_message_count_full`（末尾スライス適用時）
- `phase2c_enabled` と `non_compliance` により **Phase 2c 摂動層**がログから判別できる。

**解釈**: `suspend_ms` は壁時計介入のログ代理（詳細展望 **付録 A.5**・**§6.3**）。Ω・`S_stuck` 等は主指標候補（実験 E の主証拠側）。

乱数・GPU 非決定性は [measurement_strategy.md](../api/modules/measurement_strategy.md) に従い、**効果検証は A/B の相対比較**を主とする。

---

# 10. 安全・ガバナンス

1. **オプトアウト**: `v9_rhythm_enabled=false` でポリシーは no-op。CLI／Streamlit に明示トグル。
2. **上限**: `suspend_ms` は常に `T_suspend_max` でクリップ。負値禁止。
3. **透明性**: 開発・実験モードでは UI に「意図的遅延を挿入中」と表示できるようフックを残す（本番は任意）。
4. **朧・不従順**: Phase 2c は既定オフ。有効化時は **対象ユーザー同意** と **監査ログ** を実験設計書に紐づける。理論上の位置づけは詳細展望 **§6.1**（二層化）。製品境界の正本は同展望 **付録 A.7**。
5. **ポリシー優先**: 既存の安全フィルタ・システムプロンプトが常に優先。`non_compliance` は安全ルーティンと矛盾する場合は無効化。

---

# 11. 実験プロトコル（理論対応）

## 11.1 実験 D'（Ω 維持率・遅延の効果）

- **対照**: `suspend_ms = 0`、同一プロンプト列・同一モデル。
- **処理群**: MVP ポリシー有効、短文制約あり（朧なし／ありをサブ条件）。
- **主指標**: 会話後半（例: 後半 50% ターン）の Ω の平均・分位数。統計は事前登録した検定またはベイズ近似（別紙）。

## 11.2 実験 E（遅延介入と内部地形）

**Ω と awai_vector／6 軸代理は別指標**であり、どちらを主終点とするかは事前登録で確定する。本節の記述はその確定に先立つ暫定構造である。

- **主証拠（候補 A・Ω 主終点）**: D' と同一 JSONL に `suspend_ms` 条件で **Ω 系列・`S_stuck`** を比較。窓・検定・事前登録はプロトコル稿で固定。awai_vector（`axis_intensity_snapshot`）は補助として記録する。  
- **主証拠（候補 B・awai_vector 主終点）**: 別系 Awai ヘッドが利用可能な場合、ターン整合と軸射影の方法をプロトコルで固定した上で awai 要約を主終点とする。Ω を補助に回す。  
- **主終点の確定**: 候補 A・B のいずれか一方、または両者を主終点とする場合の多重比較補正を、実験 E 事前登録時に明示する（詳細展望 §4.1・付録 A.2.1 参照）。  
- **補助**: アンケート ID を紐づけ、LIKERT 等は **主証拠ではなく補助的拘束**（詳細展望 §6.2）。

既定ログ出力: `omega_last`・`omega_mean_window`・`S_stuck`・`axis_intensity_snapshot` は MVP 時点から記録する。別系 Awai ヘッドを接続する場合は、実装で `axis_signal_source` に識別子（例: `awai_head`）を出す想定とし、ターン整合確認後に主終点候補として使用可（現行フックは `concat_proxy` 等）。

## 11.3 実験 F（不従順）

- Phase 2c がオンの実験ブランチのみ。長期 Ω とユーザー離脱率・安全インシデントを併記。

---

# 12. テスト戦略

| 種別 | 内容 |
|------|------|
| 単体 | `S_stuck`、クリップ、`N_calm` ゲート、クールダウン（決定論的系列で期待値一致） |
| 結合 | モック LLM 前後で `suspend_ms` が記録されること、`v9_rhythm_enabled=false` で 0 であること |
| 回帰 | 意図的遅延導入後も `regression_check` の緩い閾値内であること（ベースライン更新は理由付きで） |

---

# 13. マイルストーン（推奨）

1. **M0**: `core/` に `v9_rhythm_policy.py`（名前任意）で `RhythmDecision` と純関数ポリシー、単体テスト。
2. **M1**: `v7_pair_chat_app.py` からオプションで呼び出し、JSONL ログ出力。
3. **M2**: `v7_llm_chat_generate.py` または実験用 CLI でバッチ A/B。
4. **M3**: 朧バースト（短文制約緩和・`max_new_tokens` 調整・任意で `oboro_history_tail_messages` による生成プロンプト末尾スライス）を既存制約ロジックに接続。
5. **M4**: Phase 2a/2b/2c をフラグ分離で追加。

---

# 14. バックログ分界（Issue／マイルストーン）

v9 動的リズム（本稿・009）と、**別目的の実装線**を同一バックログに混在させないための運用を次に置く。

1. **ラベル例**（Issue／マイルストーンで用途を分離する）  
   - **`rvt-009-rhythm`**: 本稿・`core/v9_rhythm_policy.py`・`tools/v7_*` の v9 パイプライン・JSONL・実験 D'/E/F。  
   - **`phase3-two-tier`**: `core/two_tier/` の Router／Controller・ブロックスキップ等（スタブは計測接続用、[stubs モジュール](../../core/two_tier/stubs.py) 冒頭のとおり本番スキップは未実装）。  
   - **`slm-roadmap`** または Phase4／008 系 planning に紐づく別線（例: [rvt_exp_2026_008_architecture_bridge.md](./rvt_exp_2026_008_architecture_bridge.md)）。  
   「未実装」の文言が並んでも、**成功指標と着手条件が異なる**ため、009 のマイルストーン表（§13）と混在させない。

2. **本稿の改訂履歴（§15）**は **009 に関する差分のみ**とする。二階建ての本番化や SLM／分散の進捗は **別トラック**の文書・履歴に書き、本稿では必要時のみ **横参照**する。

3. **二階建てを本番相当に近づける**際に、**v9 JSONL** へ同一セッションで router 統計等を載せる接続を行う場合は、**009 の単一 PR に閉じず、別設計レビュー**とする（ログスキーマ・再現性・因果解釈の責務が増えるため）。

---

# 15. 改訂履歴

| 日付 | 内容 |
|------|------|
| 2026-04-08 | 初版（付録 A 整合、MVP／Phase 2 分割、ログ・安全・実験対応） |
| 2026-04-08 | **M0**: `core/v9_rhythm_policy.py` と `tests/test_v9_rhythm_policy.py` を追加（`RhythmDecision` / `decide_v9_rhythm` 等） |
| 2026-04-08 | **M1**: `tools/v7_v9_rhythm_hooks.py`、`v7_pair_chat_app` 統合（オプション JSONL・Suspend・誠実さ system 付与） |
| 2026-04-08 | **M2**: `tools/v7_llm_chat_with_v9.py`（共通 1 ターン生成）、`tools/v9_batch_chat_turn.py` CLI、Streamlit を共通関数へ寄せた |
| 2026-04-08 | **M3**: 朧バースト時の `max_new_tokens` 緩和（`effective_max_new_tokens_for_oboro`・JSONL に base/effective を記録） |
| 2026-04-08 | **M4**: Phase 2a（`decide` + `logits_var` による軸 Suspend 加算）、2b（`compute_s_stick_trajectory`）、2c（朧時抽選）を `V9RhythmConfig` で分離。preflight（追加 forward）は採用せず |
| 2026-04-08 | 詳細展望 §4.1／§6 と整合: 実験 E を内部地形主証拠に更新、§6.1 二層化をスコープ・ログ・安全節に明示 |
| 2026-04-08 | [RVT-SPEC-2026-009-v9.0_実装仕様.md](./RVT-SPEC-2026-009-v9.0_実装仕様.md) を追加（コード準拠の整理版）、§1 から参照 |
| 2026-04-08 | [RVT-2026-009-ALG-V1](./RVT-2026-009-ALG-V1_Rhythmic_Resonance.md) を追加（§6.1.1 整合・参照アルゴリズム、system 注入禁止の明示） |
| 2026-04-08 | SPEC／ALG: **`temperature` は固着 \(F\)／`S_stuck` 非連動**（現行実装）を §1・§3.4・処理順に明示 |
| 2026-04-09 | 整合性確認に基づく三点更新: (1) 朧の実装スコープ明示（history リセット・HistoryAxisV2 は v9 リズムモジュール外）、(2) **詳細展望 §3.2** 末尾に γ 可変・HistoryAxisV2 非採用を明示、(3) §11.2 実験 E 主終点を Ω と awai_vector の別指標として整理・事前登録での確定を明記。詳細展望 §2.2・§3.2・§4.1・付録 A.2.1、SPEC §1 と整合。 |
| 2026-04-09 | §7・§8・§9: **M3** `oboro_history_tail_messages` と JSONL フィールド名をコードに合わせて追記・`axis_snapshot` を `axis_intensity_snapshot` に修正 |
| 2026-04-09 | §3 MVP 行: 朧時の **生成プロンプト末尾スライス**（任意）をスコープ表に明記 |
| 2026-04-10 | **§14** 採用: Issue ラベル分界（`rvt-009-rhythm`／`phase3-two-tier`／`slm-roadmap` 等）、改訂履歴は 009 のみ、二階建て×v9 JSONL は別設計レビュー |
