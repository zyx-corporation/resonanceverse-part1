"""v7 Streamlit チャット向け v9.0 リズムフック（Ω 計算 + ポリシー + JSONL）。

Phase 2c（沈黙／拒否抽選）は理論稿 §6.1 の二層化に従い、自律的境界の同一物ではない実験層としてログ解釈する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.v9_rhythm_policy import (
    V9RhythmConfig,
    V9RhythmState,
    RhythmDecision,
    axis_intensities_concat_proxy,
    compute_s_stick_trajectory,
    decide_v9_rhythm,
    rhythm_diagnostics,
)
from tools.v7_pair_chat_engine import analyze_chat_messages_for_omega


def append_v9_rhythm_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """JSONL 1 行追記（親ディレクトリを必要なら作成）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def v9_rhythm_before_chat_completion(
    *,
    enabled: bool,
    fast_tokenizer_ok: bool,
    chat_messages: list[dict[str, str]],
    model: Any,
    tokenizer: Any,
    device: Any,
    layer_index: int,
    d: int,
    delay_tau: int,
    pair_projector: Any,
    omega_window_mode: str,
    omega_last_n: int,
    model_name: str,
    cfg: V9RhythmConfig,
    state: V9RhythmState,
) -> tuple[RhythmDecision, V9RhythmState, dict[str, Any], str]:
    """アシスタント生成直前: Ω / w_ij を取り、リズム決定とログ用辞書（ts なし）を返す。

    Returns:
        decision, new_state, log_payload（session_id / turn_index / ts は呼び出し側）,
        axis_signal_source
    """
    axis_signal_source = "concat_proxy"
    omega_series: list[float] = []
    w_ij_rows: list[list[float]] = []
    pair_map = (
        "PairRelationLinear"
        if pair_projector is not None
        else "concat_truncate"
    )

    if enabled and fast_tokenizer_ok and len(chat_messages) >= 2:
        cor = analyze_chat_messages_for_omega(
            chat_messages,
            model=model,
            tokenizer=tokenizer,
            device=device,
            layer_index=layer_index,
            d=int(d),
            delay_tau=int(delay_tau),
            pair_projector=pair_projector,
            window_mode=str(omega_window_mode),
            last_n=int(omega_last_n),
        )
        if cor.get("error"):
            axis_signal_source = "omega_error"
        else:
            omega_series = [float(x) for x in (cor.get("omega") or [])]
            w_ij_rows = cor.get("w_ij") or []
    elif enabled and not fast_tokenizer_ok:
        axis_signal_source = "omega_unavailable"
    elif enabled and len(chat_messages) < 2:
        axis_signal_source = "too_few_messages"

    axis_intensities = axis_intensities_concat_proxy(w_ij_rows)
    s_traj: float | None = None
    if cfg.phase2b_trajectory_stuck and w_ij_rows:
        s_traj = compute_s_stick_trajectory(
            w_ij_rows,
            epsilon_w=float(cfg.trajectory_epsilon_w),
            window_w=int(cfg.window_w),
        )
    decision, new_state = decide_v9_rhythm(
        enabled=enabled,
        omega_series=omega_series,
        axis_intensities=axis_intensities,
        cfg=cfg,
        state=state,
        s_stick_trajectory=s_traj,
    )

    diag = rhythm_diagnostics(omega_series, cfg)
    _ww = int(cfg.window_w)
    tail_w = omega_series[-_ww:] if _ww > 0 else omega_series
    omega_mean_window = (
        sum(tail_w) / float(len(tail_w)) if tail_w else None
    )

    log_payload: dict[str, Any] = {
        "schema_version": "v9_rhythm_log.v1",
        "model_name": model_name,
        "axis_signal_source": axis_signal_source,
        "pair_map": pair_map,
        "s_stick_trajectory": s_traj,
        "phase2b_trajectory_stuck": bool(cfg.phase2b_trajectory_stuck),
        "omega_last": omega_series[-1] if omega_series else None,
        "omega_mean_window": omega_mean_window,
        "axis_intensity_snapshot": [round(x, 6) for x in axis_intensities],
        "suspend_ms": decision.suspend_ms,
        "suspend_reason": decision.suspend_reason,
        "oboro_burst": decision.oboro_burst,
        "append_system_honesty": decision.append_system_honesty,
        "non_compliance": decision.non_compliance,
        "phase2c_enabled": bool(cfg.phase2c_enabled),
        "d": int(d),
        "delay_tau": int(delay_tau),
        "omega_window_mode": str(omega_window_mode),
        "omega_last_n": int(omega_last_n),
        "layer_index": int(layer_index),
    }
    log_payload.update(
        {k: diag[k] for k in ("s_stick", "calm_streak", "omega_window_len")}
    )

    return decision, new_state, log_payload, axis_signal_source
