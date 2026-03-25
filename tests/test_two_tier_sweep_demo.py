"""two_tier_sweep.py --demo（ネットワーク不要）。"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_two_tier_sweep_demo():
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "two_tier_sweep.py"),
        "--demo",
        "--cpu",
        "--max-new-tokens",
        "3",
        "--repeats",
        "2",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr + out.stdout
    assert "two_tier_sweep_ok" in out.stdout
    data = json.loads(out.stdout.split("two_tier_sweep_ok", 1)[1].strip())
    assert data["schema_version"] == "two_tier_sweep.v1"
    assert "baseline" in data and "two_tier_stub" in data
