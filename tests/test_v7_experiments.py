"""v7 実験ハーネス（オフライン・軽量）。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def test_v7_phase1a_synthetic_correlation_strong():
    from experiments.v7_phase1a_phi_correlation import run_synthetic_demo

    p = run_synthetic_demo(seed=42, n_samples=300)
    assert p["schema_version"] == "v7_phase1a.v1"
    # 合成ラベルは ||S_asym||_F と強く相関
    r_trust = p["correlations_vs_scalar_feature"]["trust"]["pearson_r"]
    assert r_trust == r_trust  # not NaN
    assert r_trust > 0.85


def test_v7_phase2a_sweep_runs():
    from experiments.v7_phase2a_delay_sweep import run_sweep

    out = run_sweep(tau_max=4, seed=0, N=6, d=4, steps=200, dt=0.05, alpha=0.2, beta=0.5, noise=0.01)
    assert out["schema_version"] == "v7_phase2a.v1"
    assert len(out["by_tau"]) == 5


def test_v7_phase1b_asymmetry_positive():
    from experiments.v7_phase1b_directed_tensor import run_demo

    p = run_demo(seed=0, steps=30, N=8, d=6, lr=0.05)
    assert p["final_asymmetry_fro"] > 0.0


def test_v7_phase3a_awai():
    from experiments.v7_phase3a_awai_metrics import run_demo

    p = run_demo(seed=1, T=50, d=6)
    assert p["schema_version"] == "v7_phase3a.v1"
    assert p["awai_mean"] >= 0.0


def test_v7_run_suite_demo():
    from experiments.v7_run_suite import _load_mod

    p1a = _load_mod("v7_phase1a_phi_correlation.py", "t1")
    p2a = _load_mod("v7_phase2a_delay_sweep.py", "t2")
    b = {
        "phase1a": p1a.run_synthetic_demo(seed=0, n_samples=50),
        "phase2a": p2a.run_sweep(
            tau_max=3, seed=0, N=6, d=4, steps=100, dt=0.05, alpha=0.2, beta=0.5, noise=0.01
        ),
    }
    assert "phase1a" in b and "phase2a" in b


def test_v7_phase1a_pilot_jsonl_demo():
    from pathlib import Path

    from experiments.v7_phase1a_pilot_jsonl import load_jsonl, run_pilot

    p = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "data"
        / "v7_phase1a_pilot.jsonl"
    )
    rows = load_jsonl(p)
    out = run_pilot(
        rows=rows,
        demo=True,
        model_name="gpt2",
        cpu=True,
        seed=0,
        layer_index=-1,
    )
    assert out["schema_version"] == "v7_phase1a_pilot.v1"
    assert out["n_rows"] == 8
    assert out["correlations_label_vs_fro"]["intent_ab"]["n"] == 8


def test_v7_cli_smoke(tmp_path):
    out = tmp_path / "suite.json"
    r = subprocess.run(
        [sys.executable, str(_ROOT / "experiments" / "v7_run_suite.py"), "--demo", "--out", str(out)],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert out.is_file()
    assert "v7_suite_bundle" in out.read_text()
