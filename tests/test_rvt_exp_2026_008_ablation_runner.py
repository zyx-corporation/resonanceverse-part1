"""RVT ablation plan JSON。"""

import sys

import pytest

from experiments.rvt_exp_2026_008_ablation_flags import RvtExp008ConditionFlags
from experiments.rvt_exp_2026_008_ablation_runner import (
    build_plan,
    emit_shell_snippets,
    main,
)


def test_build_plan_explore_prepends_shell():
    f = RvtExp008ConditionFlags(base_observe_only=True, w_asym_extract=True)
    p = build_plan(f, prepend_rvt_explore=True)
    assert p["steps"][0]["id"] == "rvt_explore"
    assert "run_rvt_explore.sh" in p["steps"][0]["shell_example"]


def test_build_plan_observe_has_mrmp_step():
    f = RvtExp008ConditionFlags(base_observe_only=True, w_asym_extract=True)
    p = build_plan(f)
    assert p["schema_version"] == "rvt_exp_008_ablation_plan.v1"
    row_step = next(s for s in p["steps"] if s["id"] == "rvt_mrmp_row_batch")
    assert "run_rvt_mrmp_batch.sh" in row_step["shell_example"]
    assert "rvt_l2_mode" not in row_step["args_hint"]


def test_ablation_main_emit_shell_stdout(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["rvt_ablation", "--preset", "observe", "--emit-shell"],
    )
    main()
    out = capsys.readouterr().out
    assert "run_rvt_mrmp_batch" in out
    assert "rvt_mrmp_row_batch" in out


def test_ablation_main_preset_explore_first(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["rvt_ablation", "--preset", "explore", "--emit-shell"],
    )
    main()
    out = capsys.readouterr().out
    assert out.index("run_rvt_explore") < out.index("run_rvt_mrmp_batch")


def test_emit_shell_contains_batch():
    f = RvtExp008ConditionFlags(w_asym_extract=True, awai_vector_on=True)
    p = build_plan(f)
    txt = emit_shell_snippets(p)
    assert "ACCUMULATE_AWAI=1" in txt


def test_build_plan_inject_adds_smoke():
    f = RvtExp008ConditionFlags(
        w_asym_extract=True,
        attn_inject_sym=True,
    )
    p = build_plan(f)
    ids = {s["id"] for s in p["steps"]}
    assert "rvt_l2_smoke" in ids
    assert "rvt_l2_shell" in ids
    mrmp = next(s for s in p["steps"] if s["id"] == "rvt_mrmp_row_batch")
    assert mrmp["args_hint"].get("rvt_l2_mode") == "sym"
    assert mrmp["args_hint"].get("rvt_l2_eps") == 0.05
    assert "RVT_L2_MODE=sym" in mrmp["shell_example"]


def test_build_plan_inject_wasym_shell_and_hint():
    f = RvtExp008ConditionFlags(
        w_asym_extract=True,
        attn_inject_wasym=True,
    )
    p = build_plan(f)
    mrmp = next(s for s in p["steps"] if s["id"] == "rvt_mrmp_row_batch")
    assert mrmp["args_hint"].get("rvt_l2_mode") == "wasym"
    assert "RVT_L2_MODE=wasym" in mrmp["shell_example"]


def test_build_plan_oboro_step():
    f = RvtExp008ConditionFlags(oboro_monitor_on=True)
    p = build_plan(f)
    st = next(s for s in p["steps"] if s["id"] == "rvt_l3_oboro_lite")
    assert st["args_hint"].get("oboro_profile") == "full"
    assert st["args_hint"].get("oboro_demo") is False
    assert "run_rvt_oboro_demo" in st["shell_example"]


def test_eight_grid_plans_count():
    from experiments.rvt_exp_2026_008_eight_conditions import (
        iter_eight_experiment_conditions,
    )

    plans = [build_plan(f) for _, f in iter_eight_experiment_conditions()]
    assert len(plans) == 8
    assert all("steps" in p for p in plans)


def test_eight_grid_d3_mrmp_wasym_hint():
    from experiments.rvt_exp_2026_008_eight_conditions import (
        iter_eight_experiment_conditions,
    )

    by_id = {
        cid: build_plan(fl)
        for cid, fl in iter_eight_experiment_conditions()
    }
    d3 = by_id["D3_l2_wasym"]
    mrmp = next(s for s in d3["steps"] if s["id"] == "rvt_mrmp_row_batch")
    assert mrmp["args_hint"].get("rvt_l2_mode") == "wasym"


def test_eight_grid_d1_prepend_explore_leads_with_rvt_explore():
    from experiments.rvt_exp_2026_008_eight_conditions import (
        iter_eight_experiment_conditions,
    )

    fl = next(
        f
        for cid, f in iter_eight_experiment_conditions()
        if cid == "D1_l1_w_asym"
    )
    p = build_plan(fl, prepend_rvt_explore=True)
    assert p["steps"][0]["id"] == "rvt_explore"


def test_eight_grid_run_manifest_records_prepend_explore(
    tmp_path,
    monkeypatch,
):
    import json

    monkeypatch.setattr(
        "experiments.rvt_exp_2026_008_plan_execute.run_ablation_plan",
        lambda _plan, *, dry_run, continue_on_error: 0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--preset",
            "eight_grid",
            "--run-eight-grid",
            "--eight-grid-prepend-explore",
            "--no-dry-run",
            "--manifest",
            str(tmp_path / "m.json"),
        ],
    )
    from experiments.rvt_exp_2026_008_ablation_runner import main

    with pytest.raises(SystemExit) as ei:
        main()
    assert int(ei.value.code) == 0
    data = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert data["prepend_explore"] is True


def test_eight_grid_prepend_without_run_eight_errors(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--preset", "observe", "--eight-grid-prepend-explore"],
    )
    from experiments.rvt_exp_2026_008_ablation_runner import main

    with pytest.raises(SystemExit) as ei:
        main()
    assert int(ei.value.code) != 0
    err = capsys.readouterr().err.lower()
    assert "prepend-explore" in err
    assert "run-eight-grid" in err


def test_eight_grid_run_writes_manifest(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(
        "experiments.rvt_exp_2026_008_plan_execute.run_ablation_plan",
        lambda _plan, *, dry_run, continue_on_error: 0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--preset",
            "eight_grid",
            "--run-eight-grid",
            "--no-dry-run",
            "--manifest",
            str(tmp_path / "m.json"),
        ],
    )
    from experiments.rvt_exp_2026_008_ablation_runner import main

    with pytest.raises(SystemExit) as ei:
        main()
    assert int(ei.value.code) == 0
    data = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == "rvt_eight_grid_unattended.v1"
    assert data["failed_any"] is False
    assert len(data["conditions"]) == 8
    assert data["conditions"][0]["condition_id"] == "D0_base_observe"
    assert data.get("prepend_explore") is False
