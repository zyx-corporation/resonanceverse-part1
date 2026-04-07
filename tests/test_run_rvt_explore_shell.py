"""run_rvt_explore.sh: JSONL 不在時は exit 0 でスキップメッセージのみ。"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_run_rvt_explore_skips_when_jsonl_missing(tmp_path) -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "experiments" / "run_rvt_explore.sh"
    fake_jsonl = tmp_path / "no_such_windows.jsonl"
    env = {**os.environ, "JSONL": str(fake_jsonl)}
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "RVT explore" in proc.stderr
    assert "スキップ" in proc.stderr
