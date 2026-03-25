"""slm_resonance_lm.py のデモモード（ネットワーク不要）。"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_slm_resonance_lm_demo_runs():
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "slm_resonance_lm.py"),
        "--demo",
        "--max-steps",
        "2",
        "--cpu",
        "--freeze-base",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr + out.stdout
    assert "slm_resonance_lm_ok" in out.stdout
    data = json.loads(out.stdout.split("slm_resonance_lm_ok", 1)[1].strip())
    assert data["mode"] == "demo"
    assert data.get("integration") == "demo_stub"
    assert "train_time_s" in data and data["train_time_s"] >= 0
    assert "final_loss" in data


def test_slm_resonance_lm_baseline_hf_rejects_demo():
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "slm_resonance_lm.py"),
        "--demo",
        "--baseline-hf",
        "--max-steps",
        "1",
        "--cpu",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode != 0
