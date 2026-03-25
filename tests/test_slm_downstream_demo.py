"""slm_downstream.py のデモモード（ネットワーク不要）。"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_slm_downstream_demo_runs():
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "slm_downstream.py"),
        "--demo",
        "--max-steps",
        "2",
        "--cpu",
        "--integration",
        "awai",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr + out.stdout
    assert "slm_downstream_ok" in out.stdout
    data = json.loads(out.stdout.split("slm_downstream_ok", 1)[1].strip())
    assert data["mode"] == "demo"
    assert data["task"] == "synthetic"
    assert data.get("glue_task") == "sst2"
    assert data["integration"] == "awai"
    assert "accuracy_eval" in data
