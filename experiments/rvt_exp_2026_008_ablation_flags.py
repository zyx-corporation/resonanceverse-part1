"""
RVT-EXP-2026-008 実験 D: 八条件 ablation のフラグ表現（スキーマ）。

``ablation_runner`` / ``eight_conditions`` / ``plan_execute`` が同一キーを参照する。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RvtExp008ConditionFlags:
    """計画書フラグテーブルに合わせた最小集合（名前はログ用）。"""

    base_observe_only: bool = True
    w_asym_extract: bool = False
    attn_inject_wasym: bool = False
    attn_inject_sym: bool = False
    awai_vector_on: bool = False
    oboro_monitor_on: bool = False
    longform_generation_task: bool = False
    full_eight_grid_slot: bool = False

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = "rvt_exp_008_ablation_flags.v1"
        return d
