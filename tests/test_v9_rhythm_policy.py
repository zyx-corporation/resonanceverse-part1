"""core.v9_rhythm_policy — v9.0 動的リズム MVP（純関数）。"""

from __future__ import annotations

import pytest

from core.v9_rhythm_policy import (
    slice_messages_for_oboro_history_tail,
    V9RhythmConfig,
    V9RhythmState,
    axis_intensities_concat_proxy,
    effective_max_new_tokens_for_oboro,
    compute_calm_streak_tail,
    compute_s_stick_mvp,
    compute_suspend_ms_from_axes,
    decide_v9_rhythm,
    rhythm_diagnostics,
)
from tools.v7_v9_rhythm_hooks import append_v9_rhythm_jsonl


def test_compute_s_stick_mvp_empty():
    assert compute_s_stick_mvp([], omega_calm=0.15) == 0.0


def test_compute_s_stick_mvp_half():
    w = [0.1, 0.2, 0.5, 0.01]
    assert compute_s_stick_mvp(w, omega_calm=0.15) == pytest.approx(0.5)


def test_calm_streak_tail():
    om = [0.5, 0.01, 0.01, 0.02]
    assert compute_calm_streak_tail(om, omega_calm=0.15) == 3
    assert compute_calm_streak_tail([0.2, 0.5], omega_calm=0.15) == 0


def test_suspend_ms_clip_max():
    cfg = V9RhythmConfig(
        t_suspend_max=100,
        axis_weight_ms=(500.0,) * 6,
    )
    axes = [1.0] * 6
    ms, reason = compute_suspend_ms_from_axes(axes, cfg)
    assert ms == 100
    assert "clip_max" in reason


def test_suspend_ms_negative_net_to_zero():
    cfg = V9RhythmConfig(
        axis_weight_ms=(-500.0,) * 6,
    )
    ms, reason = compute_suspend_ms_from_axes([0.5] * 6, cfg)
    assert ms == 0
    assert reason == "axes:none"


def test_decide_disabled():
    cfg = V9RhythmConfig()
    st = V9RhythmState()
    dec, st2 = decide_v9_rhythm(
        enabled=False,
        omega_series=[0.5, 0.5],
        axis_intensities=[1.0] * 6,
        cfg=cfg,
        state=st,
    )
    assert dec.suspend_ms == 0
    assert dec.suspend_reason == "disabled"
    assert dec.oboro_burst is False
    assert st2.oboro_cooldown_remaining == 0


def test_oboro_gate_and_cooldown():
    cfg = V9RhythmConfig(
        window_w=8,
        theta_stuck=0.6,
        n_calm=4,
        n_cd=3,
        omega_calm=0.2,
        axis_weight_ms=(0.0,) * 6,
    )
    # 6 件中 5 件が凪 → S_stuck ≈ 0.83、末尾 4 連続凪
    omega = [0.5, 0.5, 0.1, 0.1, 0.1, 0.1]
    st = V9RhythmState()
    dec1, st1 = decide_v9_rhythm(
        enabled=True,
        omega_series=omega,
        axis_intensities=[0.0] * 6,
        cfg=cfg,
        state=st,
    )
    assert dec1.oboro_burst is True
    assert st1.oboro_cooldown_remaining == 3

    dec2, st2 = decide_v9_rhythm(
        enabled=True,
        omega_series=omega,
        axis_intensities=[0.0] * 6,
        cfg=cfg,
        state=st1,
    )
    assert dec2.oboro_burst is False
    assert st2.oboro_cooldown_remaining == 2

    dec3, st3 = decide_v9_rhythm(
        enabled=True,
        omega_series=omega,
        axis_intensities=[0.0] * 6,
        cfg=cfg,
        state=st2,
    )
    assert dec3.oboro_burst is False
    assert st3.oboro_cooldown_remaining == 1

    dec4, st4 = decide_v9_rhythm(
        enabled=True,
        omega_series=omega,
        axis_intensities=[0.0] * 6,
        cfg=cfg,
        state=st3,
    )
    assert dec4.oboro_burst is False
    assert st4.oboro_cooldown_remaining == 0

    dec5, st5 = decide_v9_rhythm(
        enabled=True,
        omega_series=omega,
        axis_intensities=[0.0] * 6,
        cfg=cfg,
        state=st4,
    )
    assert dec5.oboro_burst is True
    assert st5.oboro_cooldown_remaining == 3


def test_logits_var_phase2a():
    cfg = V9RhythmConfig(
        axis_weight_ms=(0.0,) * 6,
        theta_high_unc=1.0,
        delta_suspend_ms_high_unc=500,
        t_suspend_max=8000,
        phase2a_apply_logits_to_axis_suspend=True,
    )
    ms, reason = compute_suspend_ms_from_axes(
        [0.0] * 6, cfg, logits_var=2.0
    )
    assert ms == 500
    assert "logits_high" in reason


def test_rhythm_diagnostics():
    cfg = V9RhythmConfig(window_w=3, omega_calm=0.2)
    d = rhythm_diagnostics([0.1, 0.1, 0.5, 0.1], cfg)
    assert d["omega_window_len"] == 3
    assert d["calm_streak"] == 1
    assert d["s_stick"] == pytest.approx(2.0 / 3.0)


def test_axis_intensities_length():
    cfg = V9RhythmConfig()
    with pytest.raises(ValueError, match="length 6"):
        compute_suspend_ms_from_axes([0.1, 0.2], cfg)


def test_axis_intensities_concat_proxy():
    assert axis_intensities_concat_proxy([]) == [0.0] * 6
    assert axis_intensities_concat_proxy([[0.0, 0.0, 3.0, -3.0, 0.0, 0.0]]) == [
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
    ]
    assert axis_intensities_concat_proxy([[1.0, 2.0]]) == [0.5, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_compute_s_stick_trajectory():
    from core.v9_rhythm_policy import compute_s_stick_trajectory

    assert compute_s_stick_trajectory([], epsilon_w=0.1, window_w=8) is None
    assert compute_s_stick_trajectory([[0.0, 1.0]], epsilon_w=0.1, window_w=8) is None
    w = [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]
    st = compute_s_stick_trajectory(w, epsilon_w=0.5, window_w=8)
    assert st is not None
    assert 0.0 <= st <= 1.0


def test_phase2b_merges_s_stick_for_oboro():
    cfg = V9RhythmConfig(
        window_w=4,
        omega_calm=0.5,
        theta_stuck=0.4,
        n_calm=1,
        n_cd=2,
        axis_weight_ms=(0.0,) * 6,
        phase2b_trajectory_stuck=True,
    )
    omega = [1.0, 1.0, 0.01]
    dec, _ = decide_v9_rhythm(
        enabled=True,
        omega_series=omega,
        axis_intensities=[0.0] * 6,
        cfg=cfg,
        state=V9RhythmState(),
        s_stick_trajectory=0.95,
    )
    assert dec.oboro_burst is True


def test_phase2c_silence_when_oboro():
    cfg = V9RhythmConfig(
        window_w=8,
        omega_calm=10.0,
        theta_stuck=0.0,
        n_calm=0,
        n_cd=1,
        axis_weight_ms=(0.0,) * 6,
        phase2c_enabled=True,
        phase2c_silence_probability=1.0,
        phase2c_refuse_probability=0.0,
        phase2c_seed=42,
    )
    omega = [0.01] * 10
    dec, _ = decide_v9_rhythm(
        enabled=True,
        omega_series=omega,
        axis_intensities=[0.0] * 6,
        cfg=cfg,
        state=V9RhythmState(),
    )
    assert dec.oboro_burst is True
    assert dec.non_compliance == "silence"


def test_effective_max_new_tokens_for_oboro():
    cfg = V9RhythmConfig(
        oboro_relax_max_new_tokens=True,
        oboro_max_new_tokens_multiplier=2.0,
        oboro_max_new_tokens_cap=256,
    )
    e, rel = effective_max_new_tokens_for_oboro(100, True, cfg)
    assert rel is True
    assert e == 200
    e2, rel2 = effective_max_new_tokens_for_oboro(200, True, cfg)
    assert rel2 is True
    assert e2 == 256
    e3, rel3 = effective_max_new_tokens_for_oboro(100, False, cfg)
    assert rel3 is False
    assert e3 == 100
    cfg_off = V9RhythmConfig(oboro_relax_max_new_tokens=False)
    e4, rel4 = effective_max_new_tokens_for_oboro(100, True, cfg_off)
    assert rel4 is False
    assert e4 == 100


def test_slice_messages_for_oboro_history_tail():
    msgs = [{"role": "user", "content": str(i)} for i in range(5)]
    out, ok = slice_messages_for_oboro_history_tail(
        msgs,
        oboro_burst=False,
        tail=2,
    )
    assert out is msgs and ok is False
    out2, ok2 = slice_messages_for_oboro_history_tail(
        msgs,
        oboro_burst=True,
        tail=None,
    )
    assert out2 is msgs and ok2 is False
    out3, ok3 = slice_messages_for_oboro_history_tail(
        msgs,
        oboro_burst=True,
        tail=2,
    )
    assert ok3 is True
    assert len(out3) == 2
    assert out3[-1]["content"] == "4"


def test_append_v9_rhythm_jsonl(tmp_path):
    p = tmp_path / "r.jsonl"
    append_v9_rhythm_jsonl(p, {"a": 1})
    append_v9_rhythm_jsonl(p, {"b": 2})
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert '"a": 1' in lines[0]
