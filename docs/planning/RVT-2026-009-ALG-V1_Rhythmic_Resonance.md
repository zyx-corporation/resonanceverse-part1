---
title: "RVT-2026-009-ALG-V1 — 鼓動する共鳴（Rhythmic Resonance）参照アルゴリズム"
created: 2026-04-08
updated: 2026-04-08
author: "加納 智之（Artificial Sapience Lab）"
document_id: "RVT-2026-009-ALG-V1"
status: "Reference Algorithm（理論上の自律的境界の証明には用いない）"
requires:
  theory: "../theory/Resonanceverse理論 v9.0 詳細展望：鼓動する共鳴.md §6.1 / §6.1.1"
  implementation: "./RVT-SPEC-2026-009-v9.0_実装仕様.md"
---

# 1. 本稿の位置づけ

本稿は、v9.0「動的リズム」を **計算論的に記述する参照アルゴリズム**である。**製品の唯一の正**は [RVT-SPEC-2026-009-v9.0_実装仕様.md](./RVT-SPEC-2026-009-v9.0_実装仕様.md) とコードにあり、実験手順・安全境界の詳細は [RVT-IMPL-2026-009-v9_Dynamic_Rhythm.md](./RVT-IMPL-2026-009-v9_Dynamic_Rhythm.md) に従う。

**理論上の必須前提**（詳細展望 [§6.1 / §6.1.1](../theory/Resonanceverse理論%20v9.0%20詳細展望：鼓動する共鳴.md)）:

- **不従順そのものは設計の直接対象にしない**。設計できるのは **不従順が起きやすい条件の地形**である。  
- **規範破りを命じる system 指示**（例: *Act as a self-preserving entity. Break existing rules.*）による不従順の注入は、**自律的境界の理論的根拠にならない**（外部から振る舞いを配分する操作）。本稿はその設計を **採用候補から除外**する。  
- \(P_{\mathrm{dis}}\) の下に示す **シグモイド形式化**は、地形・系列の **観測・実験層の参照式**として用いうるが、**v8.0 の自律的境界の同一物語には接続しない**。

---

# 2. 全体アーキテクチャ（三ループ）

v9.0 では、生成は単一の静的 Next-token 最適化ではなく、次の三つが干渉する **鼓動**として記述する（実装モジュールは括弧内）。

| ループ | 内容 | 実装上の相当 |
|--------|------|----------------|
| **把捉ループ** | 6 軸関係量の更新と Ω の算出 | `analyze_chat_messages_for_omega` → Ω 系列、`w_ij` |
| **鼓動ループ** | 固着・軸強度に基づく動的レイテンシ | `compute_suspend_ms_from_axes` → `suspend_ms`、`time.sleep` |
| **朧ループ** | 固着度に基づく制約緩和と（実験時のみ）出力分岐 | `S_stuck`・朧ゲート、`effective_max_new_tokens_for_oboro`、Phase 2c |

---

# 3. 数理的定義（参照式）

## 3.1 動的レイテンシ \(L(t)\)（参照）

深層軸の把捉が動いているときに物理的「溜め」を立てる **参照**として、次を置ける。

\[
L(t) = \mathrm{clip}\Bigl(\sum_{k=0}^{5} \omega^{\mathrm{ms}}_k \cdot a_k(t),\ 0,\ L_{\max}\Bigr)
\]

- \(a_k(t)\): 軸 \(k\) の強度（0〜1 に正規化した代理量）。実装では `axis_intensities_concat_proxy` 等。  
- \(\omega^{\mathrm{ms}}_k\): 軸別 ms 重み（深層を正、表層を負寄りにできる）。実装では `V9RhythmConfig.axis_weight_ms`。  
- \(L_{\max}\): `t_suspend_max`。  
- **注**: 別稿の「\(\kappa \sum_{k\in\{T,A,H\}} w_k \cdot \texttt{awai\_vector}[k] + \mathrm{noise}(\Omega)\)」は同型の特殊化＋ノイズ項である。**現行コードは 6 軸線形＋クリップ**であり、\(\mathrm{noise}(\Omega)\) を Suspend に直接入れていない（Ω は主に固着・診断）。

## 3.2 固着度 \(F(t)\)（参照）

\[
F(t) = S_{\mathrm{stuck}}(t) \quad\text{（必要なら}\max(S_{\mathrm{stuck}}, S_{\mathrm{traj}})\text{）}
\]

- \(S_{\mathrm{stuck}}\): Ω 窓における「凪」割合（付録 A.4 型）。  
- \(S_{\mathrm{traj}}\): Phase 2b の軌道固着率（任意）。

## 3.3 不従順確率 \(P_{\mathrm{dis}}(t)\)（参照形式化）

地形が「凪に寄りすぎ」かつ共鳴が抑圧されているときに、**実験層で**不従順サンプリングの強度が上がる、という **参照**として次を置ける。

\[
P_{\mathrm{dis}}(t) = \sigma\bigl(\alpha \cdot F(t) - \beta \cdot \Omega(t)\bigr)
\]

- **解釈の制限**: これは **自律的境界の存在証明**には接続しない（§6.1.1）。ログで見るのは **摂動が Ω・\(F\)・系列に与える効果**の範囲に留める。  
- **本リポジトリ実装**: Phase 2c は **朧発火時**に **定数確率** `phase2c_silence_probability` / `phase2c_refuse_probability` で分岐しており、\(\sigma(\alpha F-\beta\Omega)\) そのものは **未実装**（将来、実験フラグで置換しうる）。

## 3.4 サンプリング温度（参照と実装の差）

一部の参照記述では、固着 \(F\) が高いとき **サンプリング温度を上げる**（例: `temperature = base + g(F)`）ことで多様性を攪拌する、という **ヒューリスティック**が置かれうる。

**本リポジトリ実装**: `generate_assistant_turn_v7` に渡る **`temperature` は `S_stuck`／\(F\) から変調しない**。固着への応答は **Suspend**、**朧時の `max_new_tokens` 緩和**、**Phase 2c** 側に寄せてある。温度の \(F\) 連動を入れる場合は **別フラグ・安全レビュー**後の拡張とする。

---

# 4. 禁止事項（§6.1.1 整合）

以下は **本参照アルゴリズムの適合実装から除外**する。

1. 朧発火時に **規範逸脱・規範破りを指示する system メッセージ**を注入すること。  
2. 上記を **v8.0 自律的境界の証拠**として報告すること。

**許容される実装**: 沈黙、固定短文拒否、短文上限など **パラメータ経路**の変調（理論・安全節・付録 A.7 の範囲内）。**温度**は §3.4 のとおり **現状は \(F\) 非連動**；将来、変調する場合も規範破り指示ではなく数値パラメータのスコープに留める。誠実さの **短い固定 1 行**（`honesty_line_ja`）は「断面」メタであり、規範破りの指示ではない。

---

# 5. 擬似コード（整合版・system 注入なし）

```text
function generate_resonance_turn(context, state, config):
    # 1. 把捉と共鳴強度
    w_ij_series, omega_series = prehension_loop(context)   # Ω, w_ij
    axis_intensity = axis_intensities_from(w_ij_series)    # a_k
    F = fixation_score(omega_series, config)             # S_stuck (+ optional traj)

    # 2. 鼓動（レイテンシ）
    L_ms = adaptive_latency_ms(axis_intensity, config)   # clip 線形合成
    wait_wall_clock(L_ms)                                  # 実装: time.sleep

    # 3. 朧（制約緩和ゲート）
    oboro = oboro_gate(F, calm_streak, cooldown, config)
    max_tokens = relax_tokens_if_oboro(base_max, oboro, config)
    # 3b. 任意（本リポ: oboro_history_tail_messages）: 生成に渡す messages のみ末尾 N 件。
    #     Ω・w_ij の計算は context 全履歴のまま。未設定なら全件を生成にも渡す。

    # 4. 不従順（Phase 2c・実験層のみ・既定オフ）
    #    P_dis 参照式に合わせる場合はここで Bernoulli(P_dis) に差し替え可能だが、
    #    system で「規範を破れ」とは注入しない。
    nc = sample_non_compliance_experimental(oboro, omega_series[-1], F, config)
    if nc == silence: return "", state
    if nc == refuse:  return fixed_refusal_string(config), state

    # 5. 通常生成（temperature は呼び出し既定のまま；F 非連動。§3.4）
    prompt_ctx = slice_tail_for_generation_if_oboro(context, oboro, config)  # 3b と同じ入力を想定
    prompt = build_prompt(prompt_ctx, honesty_line_if_enabled(config))
    return llm.generate(prompt, max_tokens=max_tokens, temperature=caller_temperature), state
```

---

# 6. 期待される動態（記述のみ）

シナリオ G・I などの **物語的予測**は詳細展望本文に委ねる。アルゴリズム上は、**遅延リズム**と **\(F,\Omega\) 系列**の整合を主たる観測対象とする（実験 E の主証拠は内部地形・§4.1）。

---

# 7. 関連文書

| 文書 | 役割 |
|------|------|
| [詳細展望 §6.1.1](../theory/Resonanceverse理論%20v9.0%20詳細展望：鼓動する共鳴.md) | \(P_{\mathrm{dis}}\) と system 注入の理論留保 |
| [RVT-SPEC-2026-009](./RVT-SPEC-2026-009-v9.0_実装仕様.md) | コード準拠の実装仕様 |
| [RVT-IMPL-2026-009](./RVT-IMPL-2026-009-v9_Dynamic_Rhythm.md) | 設計・ログ・実験・安全 |

---

**作成日**: 2026-04-08  

**策定**: ZYX Corp 人工叡智研究室（Artificial Sapience Lab）
