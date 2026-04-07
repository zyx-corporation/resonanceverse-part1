"""プラン executor dry-run。"""

from __future__ import annotations

from pathlib import Path

from experiments.rvt_exp_2026_008_ablation_flags import RvtExp008ConditionFlags
from experiments.rvt_exp_2026_008_ablation_runner import build_plan
from experiments.rvt_exp_2026_008_eight_conditions import eight_condition_count
from experiments.rvt_exp_2026_008_plan_execute import run_ablation_plan


def test_eight_condition_count_is_8() -> None:
    assert eight_condition_count() == 8


def test_run_ablation_plan_dry_run_smoke(capsys) -> None:
    f = RvtExp008ConditionFlags(
        base_observe_only=True,
        w_asym_extract=True,
        attn_inject_sym=True,
    )
    plan = build_plan(f)
    rc = run_ablation_plan(plan, dry_run=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "rvt_exp_2026_008_mrmp_row.py" in out
    assert "--rvt-l2-mode" in out
    assert "sym" in out


def test_argv_mrmp_rvt_l2_wasym_in_dry_run(capsys) -> None:
    from experiments.rvt_exp_2026_008_plan_execute import _argv_for_step

    repo = Path(__file__).resolve().parent.parent
    step = {
        "id": "rvt_mrmp_row_batch",
        "args_hint": {
            "jsonl": "x.jsonl",
            "line": 1,
            "max_rows": 3,
            "model": "gpt2",
            "rvt_l2_mode": "wasym",
            "rvt_l2_eps": 0.07,
            "rvt_l2_all_layers": True,
        },
    }
    argv = _argv_for_step(step, repo)
    assert argv is not None
    assert "--rvt-l2-mode" in argv
    i = argv.index("--rvt-l2-mode")
    assert argv[i + 1] == "wasym"
    assert "--rvt-l2-eps" in argv
    j = argv.index("--rvt-l2-eps")
    assert argv[j + 1] == "0.07"
    assert "--rvt-l2-all-layers" in argv


def test_argv_oboro_demo_mode():
    from experiments.rvt_exp_2026_008_plan_execute import _argv_for_step

    repo = Path(__file__).resolve().parent.parent
    step = {
        "id": "rvt_l3_oboro_lite",
        "args_hint": {
            "oboro_profile": "lite",
            "oboro_demo": True,
        },
    }
    argv = _argv_for_step(step, repo)
    assert argv is not None
    assert "--demo" in argv
    assert "--cpu" not in argv
    i = argv.index("--profile")
    assert argv[i + 1] == "lite"


def test_run_ablation_plan_oboro_demo_dry_run_includes_demo_flag(
    capsys,
) -> None:
    plan = {
        "steps": [
            {
                "id": "rvt_l3_oboro_lite",
                "args_hint": {
                    "oboro_demo": True,
                    "oboro_profile": "full",
                },
            },
        ],
    }
    rc = run_ablation_plan(plan, dry_run=True)
    assert rc == 0
    assert "--demo" in capsys.readouterr().out
