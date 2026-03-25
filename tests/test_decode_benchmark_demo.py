"""decode_benchmark.py --demo（ネットワーク不要）。"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_decode_benchmark_demo_baseline():
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "decode_benchmark.py"),
        "--demo",
        "--cpu",
        "--max-new-tokens",
        "4",
        "--warmup",
        "1",
        "--repeats",
        "2",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr + out.stdout
    assert "decode_benchmark_ok" in out.stdout
    data = json.loads(out.stdout.split("decode_benchmark_ok", 1)[1].strip())
    assert data["schema_version"] == "decode_benchmark.v1"
    assert data["variant"] == "baseline"
    assert data["latency_ms_p50"] >= 0


def test_decode_benchmark_demo_two_tier_stub():
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "decode_benchmark.py"),
        "--demo",
        "--cpu",
        "--two-tier-stub",
        "--max-new-tokens",
        "3",
        "--repeats",
        "2",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr + out.stdout
    data = json.loads(out.stdout.split("decode_benchmark_ok", 1)[1].strip())
    assert data["variant"] == "two_tier_stub"
    assert "router_keep_fraction_mean" in data
