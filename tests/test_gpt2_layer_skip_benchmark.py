"""gpt2_layer_skip_benchmark --demo（transformers のみ・ネット不要）。"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gpt2_layer_skip_benchmark_demo():
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "gpt2_layer_skip_benchmark.py"),
            "--demo",
            "--cpu",
            "--seq-len",
            "32",
            "--repeats",
            "3",
            "--warmup",
            "1",
            "--skip-every",
            "2",
            "--seed",
            "0",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "gpt2_layer_skip_benchmark_ok" in proc.stdout
    data = json.loads(proc.stdout.split("gpt2_layer_skip_benchmark_ok", 1)[1].strip())
    assert data["schema_version"] == "causal_lm_layer_skip_benchmark.v1"
    assert "decoder_stack_kind" in data
    assert data["layers_skipped"] > 0
    assert data["forward_ms_mean_skip"] <= data["forward_ms_mean_full"] * 1.05
