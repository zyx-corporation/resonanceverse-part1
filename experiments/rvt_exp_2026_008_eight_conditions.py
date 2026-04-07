"""
RVT-EXP-2026-008 実験 D: 八条件グリッド（計画書の ablation 直列化用）。

各要素は ``RvtExp008ConditionFlags``。**解釈は計画書・事前登録に合わせて固定**すること。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.rvt_exp_2026_008_ablation_flags import (  # noqa: E402
    RvtExp008ConditionFlags,
)


def iter_eight_experiment_conditions() -> Iterator[
    tuple[str, RvtExp008ConditionFlags]
]:
    """
    8 スロットの代表グリッド（番号はログ用 ID のみ）。

    D0: BASE 観測のみ（L1 抽出オフ）
    D1: L1 抽出（w_asym）
    D2: L1 + L2 SYM 注入
    D3: L1 + L2 WASYM 注入
    D4: L1 + Awai 蓄積
    D5: L3 Oboro モニタ（+ L1 抽出）
    D6: 長文生成タスクフラグ（+ L1）
    D7: D2〜D6 を束ねた「フル最小」（``full_minimal`` 相当）
    """
    yield (
        "D0_base_observe",
        RvtExp008ConditionFlags(
            base_observe_only=True,
            w_asym_extract=False,
        ),
    )
    yield (
        "D1_l1_w_asym",
        RvtExp008ConditionFlags(
            base_observe_only=True,
            w_asym_extract=True,
        ),
    )
    yield (
        "D2_l2_sym",
        RvtExp008ConditionFlags(
            base_observe_only=False,
            w_asym_extract=True,
            attn_inject_sym=True,
        ),
    )
    yield (
        "D3_l2_wasym",
        RvtExp008ConditionFlags(
            base_observe_only=False,
            w_asym_extract=True,
            attn_inject_wasym=True,
        ),
    )
    yield (
        "D4_l1_awai",
        RvtExp008ConditionFlags(
            base_observe_only=True,
            w_asym_extract=True,
            awai_vector_on=True,
        ),
    )
    yield (
        "D5_l3_oboro",
        RvtExp008ConditionFlags(
            base_observe_only=True,
            w_asym_extract=True,
            oboro_monitor_on=True,
        ),
    )
    yield (
        "D6_longform_flag",
        RvtExp008ConditionFlags(
            base_observe_only=True,
            w_asym_extract=True,
            longform_generation_task=True,
        ),
    )
    yield (
        "D7_full_minimal",
        RvtExp008ConditionFlags(
            base_observe_only=True,
            w_asym_extract=True,
            attn_inject_sym=True,
            awai_vector_on=True,
            oboro_monitor_on=True,
            longform_generation_task=True,
            full_eight_grid_slot=True,
        ),
    )


def eight_condition_count() -> int:
    return sum(1 for _ in iter_eight_experiment_conditions())
