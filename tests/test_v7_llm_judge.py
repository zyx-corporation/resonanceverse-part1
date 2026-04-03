"""v7 6軸 LLM審判（demo 経路）。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_v7_llm_judge_slm_pair_agreement(tmp_path):
    from experiments.v7_llm_judge_slm_pair_agreement import run_pair_agreement

    def row(rid: str, t_ab: float, scale: float = 1.0):
        base = {
            "id": rid,
            "text": "x",
            "trust_ab": t_ab * scale,
            "trust_ba": 0.5 * scale,
            "authority_ab": 0.3,
            "authority_ba": 0.3,
            "proximity_ab": 0.4,
            "proximity_ba": 0.4,
            "intent_ab": 0.5,
            "intent_ba": 0.5,
            "affect_ab": 0.6,
            "affect_ba": 0.6,
            "history_ab": 0.7,
            "history_ba": 0.7,
        }
        return base

    pa = tmp_path / "a.jsonl"
    pb = tmp_path / "b.jsonl"
    lines_a = [
        json.dumps(row("u1", 0.2)),
        json.dumps(row("u2", 0.5)),
        json.dumps(row("u3", 0.8)),
    ]
    pa.write_text("\n".join(lines_a) + "\n", encoding="utf-8")
    pb.write_text("\n".join(lines_a) + "\n", encoding="utf-8")
    out = run_pair_agreement(pa, pb)
    assert out["schema_version"] == "v7_llm_judge_slm_pair_agreement.v1"
    assert out["n_rows_used"] == 3
    tr = out["by_axis"]["trust_ab"]["pearson_r"]
    assert tr == tr and tr > 0.99

    pb2 = tmp_path / "b2.jsonl"
    lines_b2 = [
        json.dumps(row("u1", 0.2, 2.0)),
        json.dumps(row("u2", 0.5, 2.0)),
        json.dumps(row("u3", 0.8, 2.0)),
    ]
    pb2.write_text("\n".join(lines_b2) + "\n", encoding="utf-8")
    out2 = run_pair_agreement(pa, pb2)
    assert abs(out2["by_axis"]["trust_ab"]["pearson_r"] - 1.0) < 1e-9
    assert "runtime_meta" in out2
    assert "python" in out2["runtime_meta"]


def test_pair_agreement_markdown():
    from experiments.v7_llm_judge_slm_pair_agreement import (
        pair_agreement_markdown,
        run_pair_agreement,
    )

    # 最小ペイロード（run_pair_agreement 経由で fingerprint 等を含める）
    pa = Path(__file__).resolve().parents[1] / "experiments" / "data" / "v7_mrmp_sample.jsonl"
    if not pa.is_file():
        pytest.skip("sample jsonl missing")
    # 同一ファイルを A=B にして r=1
    out = run_pair_agreement(pa, pa)
    md = pair_agreement_markdown(out)
    assert "Pearson r" in md
    assert "trust_ab" in md


def test_v7_llm_judge_slm_pair_agreement_cli_out_md_only(tmp_path):
    def full_row(rid: str):
        return {
            "id": rid,
            "text": "t",
            **{k: 0.5 for k in (
                "trust_ab", "trust_ba", "authority_ab", "authority_ba",
                "proximity_ab", "proximity_ba", "intent_ab", "intent_ba",
                "affect_ab", "affect_ba", "history_ab", "history_ba",
            )},
        }

    pa = tmp_path / "x.jsonl"
    line = json.dumps(full_row("z1"))
    pa.write_text(line + "\n", encoding="utf-8")
    out_md = tmp_path / "o.md"
    r = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "experiments" / "v7_llm_judge_slm_pair_agreement.py"),
            "--jsonl-a",
            str(pa),
            "--jsonl-b",
            str(pa),
            "--out-md",
            str(out_md),
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert out_md.is_file()
    assert "trust_ab" in out_md.read_text(encoding="utf-8")


def test_v7_llm_judge_prompt_fingerprint_stable():
    from experiments.v7_phase1a_llm_judge_six_axes import judge_prompt_fingerprint_sha256

    assert (
        judge_prompt_fingerprint_sha256()
        == "1e142cad3c2b6e0f812c3ba919e20ecf7a44710d2ba2f3391dfb535696d54821"
    )


def test_parse_llm_judge_json_response():
    from experiments.v7_phase1a_llm_judge_six_axes import parse_llm_judge_json_response

    raw = '説明\n{"trust_ab": 0.1, "trust_ba": 0.2, "authority_ab": 0.3, "authority_ba": 0.4, "proximity_ab": 0.5, "proximity_ba": 0.6, "intent_ab": 0.7, "intent_ba": 0.8, "affect_ab": 0.9, "affect_ba": 0.1, "history_ab": 0.2, "history_ba": 0.3}\n末尾'
    d = parse_llm_judge_json_response(raw)
    assert d["trust_ab"] == 0.1
    assert d["history_ba"] == 0.3


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
