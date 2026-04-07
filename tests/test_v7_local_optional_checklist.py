"""v7 ローカル任意チェックリスト。"""

from __future__ import annotations

from experiments.v7_local_optional_checklist import run_checklist


def test_run_checklist_schema(tmp_path):
    p = run_checklist(cwd=tmp_path)
    assert p["schema_version"] == "v7_local_optional_checklist.v1"
    assert "disk_free_gb_approx" in p
    assert "has_hf_token" in p
    assert "mlx_import_available" in p
    assert "torch" in p
