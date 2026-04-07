"""計画書 6 軸 ↔ v7 審判キー対応。"""

from __future__ import annotations

from experiments.rvt_exp_2026_008_head_axis_matrix import row_targets_six_ab
from experiments.rvt_exp_2026_008_judge_axis_mapping import (
    ORDERED_V7_AB_KEYS,
    RVT_PLAN_AXIS_NAMES,
    validate_mapping,
)


def test_validate_mapping():
    validate_mapping()


def test_ordered_keys_match_plan_axes():
    assert len(RVT_PLAN_AXIS_NAMES) == 6
    assert len(ORDERED_V7_AB_KEYS) == 6
    assert ORDERED_V7_AB_KEYS[0] == "trust_ab"


def test_row_targets_from_row():
    row = {k: 0.1 * i for i, k in enumerate(ORDERED_V7_AB_KEYS)}
    w = row_targets_six_ab(row)
    assert w is not None and w.shape == (6,)
