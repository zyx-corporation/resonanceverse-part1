"""v7_slm_judge_triple_run のデモモード（HF 不要）。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_triple_run_demo_smoke():
    out = _ROOT / "experiments" / "logs" / "slm_judge_triple_pytest"
    script = _ROOT / "experiments" / "v7_slm_judge_triple_run.py"
    sample = _ROOT / "experiments" / "data" / "v7_mrmp_sample.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo",
            "--jsonl",
            str(sample),
            "--n",
            "3",
            "--out-dir",
            str(out),
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "v7_slm_judge_triple_run_ok" in proc.stdout
    ok_line = [ln for ln in proc.stdout.splitlines() if "v7_slm_judge_triple_run_ok" in ln][0]
    ok_payload = json.loads(ok_line.split("v7_slm_judge_triple_run_ok", 1)[1].strip())
    assert ok_payload.get("total_wall_s") is not None
    assert float(ok_payload["total_wall_s"]) > 0
    summary_path = out / "triple_run_summary.json"
    assert summary_path.is_file()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data.get("schema_version") == "v7_slm_judge_triple_run.v1"
    assert data.get("demo") is True
    pair = data.get("pair_agreement") or {}
    assert "7b_a_vs_7b_b" in pair
    inner = pair["7b_a_vs_7b_b"]["summary"]
    assert inner.get("schema_version") == "v7_llm_judge_slm_pair_agreement.v1"
    wc = data.get("wall_clock_s") or {}
    assert wc.get("total_wall_s", 0) > 0
    assert "subprocess_judge_7b_a_s" in wc
    assert "subprocess_pair_7b_a_vs_7b_b_s" in wc
