"""
Phase II-A 周辺の JSON 成果物に付与するレール ID（実装マスタープラン §1 と対応）。

同一キー（rail_id / rail_ids / implementation_master_plan_md）で混線を防ぐ。
"""

from __future__ import annotations

from typing import Any

IMPLEMENTATION_MASTER_PLAN_MD = "docs/planning/v7_phase2a_implementation_master_plan.md"

A_IIA_NUMERIC_SYNTHETIC = "A_IIA_numeric_synthetic"
C_SYNTHETIC_SENSITIVITY_MU_PROXY = "C_synthetic_sensitivity_mu_proxy"
B_EMPIRICAL_MRMP = "B_empirical_MRMP"
B_EMPIRICAL_MRMP_COMPARE_RUNS = "B_empirical_MRMP_compare_runs"
D_PHASE_IV_MINIMAL_REPRO = "D_phase_IV_minimal_repro"
E_RVT_008_ABLATION_PLAN = "E_RVT_008_ablation_plan"
THEORY_REFERENCE_INJECTED = "theory_reference_injected"
B_MRMP_TABLE_PLACEHOLDER = "B_empirical_MRMP_table_placeholder"


def with_rail(rail_id: str) -> dict[str, Any]:
    return {
        "rail_id": rail_id,
        "implementation_master_plan_md": IMPLEMENTATION_MASTER_PLAN_MD,
    }


def with_rails(rail_ids: list[str]) -> dict[str, Any]:
    return {
        "rail_ids": list(rail_ids),
        "implementation_master_plan_md": IMPLEMENTATION_MASTER_PLAN_MD,
    }


def paper_tau_comparison_rail_metadata() -> dict[str, Any]:
    """論文比較表: 合成 A・理論参照行・MRMP プレースホルダ行を 1 表に載せる。"""
    return with_rails(
        [
            A_IIA_NUMERIC_SYNTHETIC,
            THEORY_REFERENCE_INJECTED,
            B_MRMP_TABLE_PLACEHOLDER,
        ]
    )
