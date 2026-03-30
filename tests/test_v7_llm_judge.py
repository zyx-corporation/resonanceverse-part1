"""v7 6軸 LLM審判（demo 経路）。"""

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_v7_llm_judge_run_judge_demo():
    from experiments.v7_phase1a_llm_judge_six_axes import run_judge

    rows = [
        {
            "id": "t1",
            "text": "A: こんにちは B: どうも",
            "speaker_src": "A",
            "speaker_tgt": "B",
        }
    ]
    p = run_judge(
        rows=rows,
        demo=True,
        provider="openai",
        openai_model="gpt-4o-mini",
        temperature=0.2,
        seed=0,
    )
    assert p["schema_version"] == "v7_llm_judge_six_axes.v1"
    assert len(p["rows"]) == 1
    r0 = p["rows"][0]
    assert "trust_ab" in r0 and "history_ba" in r0
    assert r0["llm_judge_meta"]["mode"] == "demo_deterministic_hash"


def test_v7_llm_judge_cli_demo(tmp_path):
    out_j = tmp_path / "j.jsonl"
    r = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "experiments" / "v7_phase1a_llm_judge_six_axes.py"),
            "--demo",
            "--jsonl",
            str(_ROOT / "experiments" / "data" / "v7_mrmp_sample.jsonl"),
            "--out-jsonl",
            str(out_j),
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    lines = out_j.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    row = json.loads(lines[0])
    assert row["llm_judge_meta"]["prompt_template_id"] == "v7_llm_judge_prompt_v1"
