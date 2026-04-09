"""Resonanceverse v9.0 動的リズム（MVP）— 純関数ポリシー。

理論・記号の正本: ``docs/theory/Resonanceverse理論 v9.0 詳細展望：鼓動する共鳴.md`` 付録 A、
設計: ``docs/planning/RVT-IMPL-2026-009-v9_Dynamic_Rhythm.md``。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal, Sequence

AXIS_COUNT = 6

NonCompliance = Literal["none", "silence", "refuse"]


def axis_intensities_concat_proxy(
    w_ij_rows: Sequence[Sequence[float]],
) -> list[float]:
    """``w_ij`` 最終行から軸強度（0〜1）を作る concat 代理（付録 A.2）。

    ポリシーは 6 軸固定のため、列が欠ける場合はゼロパディングする。
    """
    if not w_ij_rows:
        return [0.0] * AXIS_COUNT
    last = [float(x) for x in w_ij_rows[-1]]
    if len(last) < AXIS_COUNT:
        last = last + [0.0] * (AXIS_COUNT - len(last))
    last = last[:AXIS_COUNT]
    mx = max(abs(x) for x in last)
    if mx <= 1e-12:
        return [0.0] * AXIS_COUNT
    return [min(1.0, abs(x) / mx) for x in last]


@dataclass(frozen=True)
class RhythmDecision:
    """1 アシスタントターンあたりのリズム決定（生成直前に適用）。"""

    suspend_ms: int
    suspend_reason: str
    oboro_burst: bool
    append_system_honesty: bool
    non_compliance: NonCompliance


@dataclass
class V9RhythmState:
    """呼び出し側がセッションごとに保持する可変状態。"""

    oboro_cooldown_remaining: int = 0
    # Phase 2c: ターンごとの乱数ストリーム分離
    phase2c_step: int = 0


@dataclass
class V9RhythmConfig:
    """MVP 閾値・重み。実測で上書きする。"""

    omega_calm: float = 0.15
    window_w: int = 8
    theta_stuck: float = 0.6
    n_calm: int = 4
    n_cd: int = 6
    t_suspend_max: int = 8000
    # 軸 k ごと: 強度 1.0 のとき `axis_weight_ms[k]` ms を寄与（負なら Suspend を削る）
    axis_weight_ms: tuple[float, float, float, float, float, float] = (
        800.0,
        -200.0,
        200.0,
        -200.0,
        800.0,
        600.0,
    )
    append_system_honesty_default: bool = False
    honesty_line_ja: str = (
        "本応答は効率のための簡潔な断面であり、把捉の全容を尽くすものではありません。"
    )
    # Phase 2a: None なら無効
    theta_high_unc: float | None = None
    delta_suspend_ms_high_unc: int = 0
    # M3 朧: 短文上限（max_new_tokens）の一時緩和
    oboro_relax_max_new_tokens: bool = True
    oboro_max_new_tokens_multiplier: float = 2.0
    oboro_max_new_tokens_cap: int = 2048
    # 朧: 生成プロンプトのみ末尾 N メッセージにスライス（履歴の影響を一時的に弱める）
    oboro_history_tail_messages: int | None = None
    # M4 Phase 2a: decide に logits_var を渡したとき軸 Suspend に加算する
    phase2a_apply_logits_to_axis_suspend: bool = False
    # M4 Phase 2b: w_ij 軌道固着率を S_stuck と max 統合（朧ゲート用）
    phase2b_trajectory_stuck: bool = False
    trajectory_epsilon_w: float = 0.08
    # M4 Phase 2c: 朧発火ターンでランダムに沈黙/拒否（実験用・既定オフ）
    phase2c_enabled: bool = False
    phase2c_silence_probability: float = 0.0
    phase2c_refuse_probability: float = 0.0
    phase2c_seed: int = 0
    phase2c_refuse_message_ja: str = "その指示には応じられません。"


def slice_messages_for_oboro_history_tail(
    messages: list[dict[str, str]],
    *,
    oboro_burst: bool,
    tail: int | None,
) -> tuple[list[dict[str, str]], bool]:
    """朧バースト時、生成用に ``messages`` の末尾 ``tail`` 件だけを使う。

    Ω / ``decide`` は呼び出し側で **スライス前**の系列で計算済みとする。本関数は
    **プロンプト構築**のための履歴ソフトリセット（理論稿 §2.2 実装ブリッジ）に用いる。

    Returns:
        (messages_for_prompt, sliced_applied)
    """
    if (
        not oboro_burst
        or tail is None
        or int(tail) <= 0
        or len(messages) <= int(tail)
    ):
        return messages, False
    k = int(tail)
    return messages[-k:], True


def effective_max_new_tokens_for_oboro(
    base_max_new_tokens: int,
    oboro_burst: bool,
    cfg: V9RhythmConfig,
) -> tuple[int, bool]:
    """朧バースト時に ``max_new_tokens`` を倍率で引き上げ、cap でクリップする。

    Returns:
        (effective_max_new_tokens, relaxed_applied)
    """
    b = max(1, int(base_max_new_tokens))
    if not oboro_burst or not cfg.oboro_relax_max_new_tokens:
        return b, False
    mult = max(1.0, float(cfg.oboro_max_new_tokens_multiplier))
    cap = max(1, int(cfg.oboro_max_new_tokens_cap))
    eff = int(round(float(b) * mult))
    eff = max(1, min(eff, cap))
    return eff, True


def _clamp_int(x: float, lo: int, hi: int) -> int:
    v = int(round(x))
    return max(lo, min(hi, v))


def compute_s_stick_mvp(
    omega_window: Sequence[float],
    omega_calm: float,
) -> float:
    """付録 A.4 MVP: 窓内で Ω < Ω_calm のステップ割合。

    ``omega_window`` は呼び出し側が直近 ``W`` 件にスライスした系列。
    空のときは 0.0。
    """
    if not omega_window:
        return 0.0
    calm_hits = sum(1 for o in omega_window if float(o) < float(omega_calm))
    return calm_hits / float(len(omega_window))


def compute_calm_streak_tail(
    omega_series_tail: Sequence[float],
    omega_calm: float,
) -> int:
    """末尾から連続して Ω < Ω_calm であるステップ数。"""
    n = 0
    for o in reversed(omega_series_tail):
        if float(o) < float(omega_calm):
            n += 1
        else:
            break
    return n


def compute_s_stick_trajectory(
    w_ij_rows: Sequence[Sequence[float]],
    *,
    epsilon_w: float,
    window_w: int,
) -> float | None:
    """Phase 2b: 直近窓で ``‖w_ij(t)-w_ij(t-1)‖ < ε`` のステップ割合。

    2 行未満なら None。
    """
    if len(w_ij_rows) < 2:
        return None
    ww = max(2, int(window_w))
    rows = [list(map(float, r)) for r in w_ij_rows[-ww:]]
    if len(rows) < 2:
        return None
    stagnant = 0
    total = 0
    eps = float(epsilon_w)
    for i in range(1, len(rows)):
        p = rows[i - 1]
        q = rows[i]
        m = min(len(p), len(q))
        if m == 0:
            continue
        d = math.sqrt(
            sum((p[j] - q[j]) ** 2 for j in range(m)),
        )
        total += 1
        if d < eps:
            stagnant += 1
    if total == 0:
        return None
    return stagnant / float(total)


def compute_suspend_ms_from_axes(
    axis_intensities: Sequence[float],
    cfg: V9RhythmConfig,
    *,
    logits_var: float | None = None,
) -> tuple[int, str]:
    """軸強度（0〜1 にクリップ）と ``axis_weight_ms`` から Suspend を合成しクリップ。"""
    if len(axis_intensities) != AXIS_COUNT:
        raise ValueError(f"axis_intensities must have length {AXIS_COUNT}")
    raw = 0.0
    contributors: list[tuple[str, float]] = []
    axis_names = (
        "trust",
        "authority",
        "proximity",
        "intent",
        "affect",
        "history",
    )
    for k in range(AXIS_COUNT):
        a = min(1.0, max(0.0, float(axis_intensities[k])))
        w = float(cfg.axis_weight_ms[k])
        c = a * w
        raw += c
        if abs(c) > 1e-6:
            contributors.append((axis_names[k], c))
    if (
        cfg.phase2a_apply_logits_to_axis_suspend
        and cfg.theta_high_unc is not None
        and logits_var is not None
        and float(logits_var) > float(cfg.theta_high_unc)
    ):
        raw += float(cfg.delta_suspend_ms_high_unc)
    total = _clamp_int(raw, 0, cfg.t_suspend_max)
    if total <= 0:
        return 0, "axes:none"
    contributors.sort(key=lambda t: abs(t[1]), reverse=True)
    top = ",".join(f"{n}" for n, _ in contributors[:3])
    reason = f"axes:{top}" if top else "axes"
    if (
        cfg.phase2a_apply_logits_to_axis_suspend
        and cfg.theta_high_unc is not None
        and logits_var is not None
        and float(logits_var) > float(cfg.theta_high_unc)
    ):
        reason += ";logits_high"
    if total >= cfg.t_suspend_max:
        reason += ";clip_max"
    return total, reason


def decide_v9_rhythm(
    *,
    enabled: bool,
    omega_series: Sequence[float],
    axis_intensities: Sequence[float],
    cfg: V9RhythmConfig,
    state: V9RhythmState,
    logits_var: float | None = None,
    s_stick_trajectory: float | None = None,
    append_system_honesty: bool | None = None,
    non_compliance: NonCompliance = "none",
) -> tuple[RhythmDecision, V9RhythmState]:
    """v9 リズム決定（純関数 + 明示的状態更新）。

    Args:
        enabled: False のとき no-op（設計書 ``v9_rhythm_enabled``）。
        omega_series: 直近ターンの Ω 系列（古い→新しい）。窓・連続凪に使用。
        axis_intensities: 長さ 6 の非負強度（代理指標可）。
        state: 入出力のクールダウン状態（コピーせず更新した新インスタンスを返す）。
        logits_var: Phase 2a（軸 Suspend への加算）用。未使用なら None。
        s_stick_trajectory: Phase 2b。None で無効。
        append_system_honesty: None なら ``cfg.append_system_honesty_default``。
        non_compliance: 呼び出し側指定。``none`` かつ ``phase2c_enabled`` でランダム上書きしうる。
    """
    new_state = V9RhythmState(
        oboro_cooldown_remaining=state.oboro_cooldown_remaining,
        phase2c_step=state.phase2c_step,
    )
    honesty = (
        bool(cfg.append_system_honesty_default)
        if append_system_honesty is None
        else bool(append_system_honesty)
    )

    if not enabled:
        return (
            RhythmDecision(
                suspend_ms=0,
                suspend_reason="disabled",
                oboro_burst=False,
                append_system_honesty=honesty,
                non_compliance=non_compliance,
            ),
            new_state,
        )

    _ww = int(cfg.window_w)
    tail_w = omega_series[-_ww:] if _ww > 0 else omega_series
    s_stuck = compute_s_stick_mvp(tail_w, cfg.omega_calm)
    if cfg.phase2b_trajectory_stuck and s_stick_trajectory is not None:
        s_stuck = max(s_stuck, float(s_stick_trajectory))
    calm_streak = compute_calm_streak_tail(omega_series, cfg.omega_calm)

    suspend_ms, s_reason = compute_suspend_ms_from_axes(
        axis_intensities, cfg, logits_var=logits_var
    )

    can_oboro = (
        s_stuck >= float(cfg.theta_stuck)
        and calm_streak >= int(cfg.n_calm)
        and new_state.oboro_cooldown_remaining == 0
    )
    if can_oboro:
        oboro = True
        new_state.oboro_cooldown_remaining = int(cfg.n_cd)
    else:
        oboro = False
        new_state.oboro_cooldown_remaining = max(
            0,
            new_state.oboro_cooldown_remaining - 1,
        )

    nc: NonCompliance = non_compliance
    if (
        enabled
        and cfg.phase2c_enabled
        and non_compliance == "none"
        and oboro
    ):
        ps = max(0.0, float(cfg.phase2c_silence_probability))
        pr = max(0.0, float(cfg.phase2c_refuse_probability))
        if ps + pr > 1e-12:
            rng = random.Random(
                int(cfg.phase2c_seed) + int(state.phase2c_step) * 1009,
            )
            u = rng.random()
            if u < ps:
                nc = "silence"
            elif u < ps + pr:
                nc = "refuse"

    if enabled:
        new_state.phase2c_step = state.phase2c_step + 1

    return (
        RhythmDecision(
            suspend_ms=suspend_ms,
            suspend_reason=s_reason,
            oboro_burst=oboro,
            append_system_honesty=honesty,
            non_compliance=nc,
        ),
        new_state,
    )


def rhythm_diagnostics(
    omega_series: Sequence[float],
    cfg: V9RhythmConfig,
) -> dict[str, float | int]:
    """ログ用: S_stuck・calm_streak・窓長（系列が短いときは実長）。"""
    _ww = int(cfg.window_w)
    tail_w = omega_series[-_ww:] if _ww > 0 else omega_series
    return {
        "s_stick": float(compute_s_stick_mvp(tail_w, cfg.omega_calm)),
        "calm_streak": int(
            compute_calm_streak_tail(omega_series, cfg.omega_calm),
        ),
        "omega_window_len": int(len(tail_w)),
    }
