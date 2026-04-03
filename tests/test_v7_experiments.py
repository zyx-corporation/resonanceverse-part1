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


def test_v7_phase2a_auxiliary_label_delay_coherence():
    from experiments.v7_phase2a_empirical import (
        auxiliary_label_delay_coherence_by_axis,
        series_has_auxiliary_label_scores,
    )

    series = [
        {
            "dialogue_id": "D1",
            "values": [
                {
                    "t": 0,
                    "s_asym_ab": 1.0,
                    "s_asym_ba": 0.5,
                    "trust_ab": 1.0,
                    "trust_ba": 0.5,
                },
                {
                    "t": 1,
                    "s_asym_ab": 1.0,
                    "s_asym_ba": 0.5,
                    "trust_ab": 2.0,
                    "trust_ba": 0.25,
                },
            ],
        },
    ]
    assert series_has_auxiliary_label_scores(series) is True
    aux = auxiliary_label_delay_coherence_by_axis(series, tau_max=1)
    assert set(aux.keys()) == {
        "trust",
        "authority",
        "proximity",
        "intent",
        "affect",
        "history",
    }
    assert len(aux["trust"]) == 2
    assert aux["trust"][0]["tau"] == 0
    assert aux["trust"][0]["n"] == 1


def test_v7_phase2a_tau_summary_auxiliary_table():
    from experiments.v7_phase2a_tau_summary import summarize_auxiliary_label_delay

    aux = {
        "trust": [
            {
                "tau": 0,
                "R_mean": 1.0,
                "R_var": 0.1,
                "n": 2,
                "data_source": "x",
                "mode": "empirical",
                "tau_star_candidate": False,
            },
            {
                "tau": 1,
                "R_mean": 1.0,
                "R_var": 0.9,
                "n": 2,
                "data_source": "x",
                "mode": "empirical",
                "tau_star_candidate": False,
            },
        ],
    }
    per = summarize_auxiliary_label_delay(aux, smooth_window=3)
    assert per["trust"]["R_var_global_max_tau"] == 1
    assert per["trust"]["R_var_global_max"] == pytest.approx(0.9)


def test_v7_phase2a_summarize_n_per_tau():
    from experiments.v7_phase2a_tau_summary import summarize_n_per_tau

    rows = [
        {"tau": 0, "n": 10, "R_mean": 1.0, "R_var": 0.1},
        {"tau": 1, "n": 10, "R_mean": 1.0, "R_var": 0.2},
        {"tau": 2, "n": 8, "R_mean": 1.0, "R_var": 0.3},
    ]
    s = summarize_n_per_tau(rows)
    assert s["n_min"] == 8
    assert s["n_max"] == 10
    assert s["tau_at_n_min"] == 2
    assert len(s["n_drop_transitions"]) == 1
    assert s["n_drop_transitions"][0]["tau"] == 2


def test_v7_phase2a_bundle_validate_ok_minimal(tmp_path):
    import json

    from experiments.v7_phase2a_bundle_validate import validate_artifacts

    sub = tmp_path / "logs"
    sub.mkdir(parents=True)
    (sub / "w.json").write_text(
        json.dumps(
            {
                "schema_version": "v7_phase2a_empirical.v1",
                "by_tau": [
                    {
                        "tau": 0,
                        "R_mean": 1.0,
                        "R_var": 0.0,
                        "n": 1,
                        "data_source": "x",
                        "mode": "empirical",
                        "tau_star_candidate": False,
                    },
                ],
                "inference_device": "cpu",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (sub / "sum.json").write_text(
        json.dumps(
            {
                "schema_version": "v7_phase2a_tau_summary.v1",
                "summary": {"n_tau_points": 1},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (sub / "b.json").write_text(
        json.dumps(
            {
                "schema_version": "v7_phase2a_bootstrap.v1",
                "by_tau": [{"tau": 0, "mean": 1.0}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (sub / "s.md").write_text("# x\n", encoding="utf-8")
    (sub / "b.md").write_text("# y\n", encoding="utf-8")
    (sub / "p.log").write_text("ok\n", encoding="utf-8")

    arts = {
        "with_contributions_jsonl": "logs/w.json",
        "tau_summary_md": "logs/s.md",
        "tau_summary_json": "logs/sum.json",
        "bootstrap_json": "logs/b.json",
        "bootstrap_md": "logs/b.md",
        "pipeline_log": "logs/p.log",
    }
    errs, warns = validate_artifacts(tmp_path, arts, strict=True)
    assert errs == []
    assert warns == []


def test_v7_phase2a_bundle_validate_strict_missing(tmp_path):
    from experiments.v7_phase2a_bundle_validate import validate_artifacts

    arts = {"with_contributions_jsonl": "nope/missing.json"}
    errs, warns = validate_artifacts(tmp_path, arts, strict=True)
    assert errs
    assert not warns

    errs2, warns2 = validate_artifacts(tmp_path, arts, strict=False)
    assert not errs2
    assert warns2


def test_v7_phase2a_bundle_remap_paths():
    from experiments.v7_phase2a_bundle_validate import (
        default_prefix_from_artifacts,
        remap_artifact_paths,
    )

    arts = {
        "with_contributions_jsonl": "experiments/logs/v7_phase2a_mrmp_tau_n3146_with_contrib.json",
        "tau_summary_json": "experiments/logs/v7_phase2a_mrmp_tau_n3146_summary.json",
        "pipeline_log": "experiments/logs/run_phase2a_mrmp_tau_gpu.log",
    }
    bd = default_prefix_from_artifacts(arts)
    assert bd == "experiments/logs/v7_phase2a_mrmp_tau_n3146"
    m = remap_artifact_paths(
        arts,
        bundle_default_prefix=bd,
        out_prefix="experiments/logs/custom_run",
    )
    assert m["with_contributions_jsonl"] == "experiments/logs/custom_run_with_contrib.json"
    assert m["tau_summary_json"] == "experiments/logs/custom_run_summary.json"
    assert m["pipeline_log"] == "experiments/logs/run_phase2a_mrmp_tau_gpu.log"


def test_v7_phase2a_tau_summary_cli_out_json(tmp_path):
    import json

    inp = tmp_path / "emp.json"
    inp.write_text(
        json.dumps(
            {
                "schema_version": "v7_phase2a_empirical.v1",
                "n_dialogues": 1,
                "by_tau": [
                    {
                        "tau": 0,
                        "R_mean": 1.0,
                        "R_var": 0.2,
                        "n": 1,
                        "data_source": "x",
                        "mode": "empirical",
                        "tau_star_candidate": False,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out_j = tmp_path / "summary.json"
    r = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "experiments" / "v7_phase2a_tau_summary.py"),
            str(inp),
            "--out-json",
            str(out_j),
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(out_j.read_text(encoding="utf-8"))
    assert data["schema_version"] == "v7_phase2a_tau_summary.v1"
    assert data["smooth_window"] == 5
    assert data["summary"]["n_tau_points"] == 1
    assert data["n_per_tau"]["n_min"] == 1
    assert data["n_per_tau"]["per_tau"][0] == {"tau": 0, "n": 1}


def test_v7_phase2a_tau_plots_cli(tmp_path):
    import json

    pytest.importorskip("matplotlib")

    inp = tmp_path / "mini_with_contrib.json"
    inp.write_text(
        json.dumps(
            {
                "schema_version": "v7_phase2a_empirical.v1",
                "n_dialogues": 5,
                "by_tau": [
                    {
                        "tau": i,
                        "R_mean": float(i * 0.1 + 1),
                        "R_var": float((i % 4) + 0.5),
                        "n": max(1, 8 - i // 2),
                        "data_source": "x",
                        "mode": "empirical",
                        "tau_star_candidate": False,
                    }
                    for i in range(20)
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    r = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "experiments" / "v7_phase2a_tau_plots.py"),
            str(inp),
            "--out-dir",
            str(tmp_path),
            "--dpi",
            "80",
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "mini_with_contrib_tau_primary.png").is_file()


def test_v7_phase2a_tau_plots_paper_cli(tmp_path):
    import json

    pytest.importorskip("matplotlib")

    inp = tmp_path / "mini2.json"
    inp.write_text(
        json.dumps(
            {
                "schema_version": "v7_phase2a_empirical.v1",
                "model": "gpt2",
                "layer_index": -1,
                "n_dialogues": 3,
                "by_tau": [
                    {
                        "tau": i,
                        "R_mean": 1.0,
                        "R_var": 0.5 + 0.01 * i,
                        "n": 3,
                        "data_source": "x",
                        "mode": "empirical",
                        "tau_star_candidate": False,
                    }
                    for i in range(12)
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    r = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "experiments" / "v7_phase2a_tau_plots.py"),
            str(inp),
            "--out-dir",
            str(tmp_path),
            "--paper",
            "--formats",
            "png",
            "--dpi",
            "96",
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "mini2_tau_paper_primary.png").is_file()


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


def test_v7_phase1a_autoproxy_demo():
    from experiments.v7_phase1a_autoproxy import _BUILTIN_TEXTS, run_autoproxy

    out = run_autoproxy(
        texts=list(_BUILTIN_TEXTS),
        demo=True,
        model_name="gpt2",
        cpu=True,
        seed=0,
    )
    assert out["schema_version"] == "v7_phase1a_autoproxy.v1"
    assert out["n_rows"] == len(_BUILTIN_TEXTS)
    assert "disclaimer" in out


def test_v7_empirical_run_demo():
    from pathlib import Path

    from experiments.v7_empirical_run import run_empirical_bundle

    jsonl = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "data"
        / "v7_phase1a_pilot.jsonl"
    )
    b = run_empirical_bundle(
        demo=True,
        model_name="gpt2",
        cpu=True,
        seed=0,
        jsonl_path=jsonl,
        reference_text="test",
    )
    assert b["schema_version"] == "v7_empirical_bundle.v1"
    assert b["phase1a_pilot_jsonl"]["n_rows"] == 8
    assert b["phase1a_autoproxy"]["schema_version"] == "v7_phase1a_autoproxy.v1"
    assert b["phase1a_reference_layers"]["mode"] == "omitted_in_demo"
    assert b["phase1a_synthetic_sanity"]["mode"] == "synthetic_demo"


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


def test_v7_empirical_cli_smoke(tmp_path):
    out = tmp_path / "empirical.json"
    r = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "experiments" / "v7_empirical_run.py"),
            "--demo",
            "--out",
            str(out),
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert out.is_file()
    assert "v7_empirical_bundle" in out.read_text()


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


def test_v7_phase2a_repro_manifest_pin_code_cli(tmp_path):
    out = tmp_path / "m.json"
    r = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "experiments" / "v7_phase2a_repro_manifest.py"),
            "--bundle",
            str(_ROOT / "experiments" / "baselines" / "v7_phase2a_mrmp_tau_n3146_bundle_v1.json"),
            "--pin-code-only",
            "--out",
            str(out),
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == "v7_phase2a_repro_manifest.v1"
    assert data["pin_code_only"] is True
    assert "experiments/v7_phase2a_empirical_run.py" in data["files"]

    r2 = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "experiments" / "v7_phase2a_repro_manifest.py"),
            "--verify",
            str(out),
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    assert r2.returncode == 0, r2.stderr
    assert "experiments/run_local_slm_phase2_smoke.sh" in data["files"]
    assert "experiments/run_local_slm_smoke_all.sh" in data["files"]
    assert "experiments/prompts/v7_llm_judge_prompt_v1.json" in data["files"]
    assert "experiments/v7_llm_judge_slm_pair_agreement.py" in data["files"]
    assert "experiments/run_local_slm_judge_pair_agreement.sh" in data["files"]


def test_v7_local_slm_phase1_smoke_bundle_baseline():
    p = _ROOT / "experiments" / "baselines" / "v7_local_slm_phase1_smoke_bundle_v1.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["schema_version"] == "v7_phase2a_bundle_pointer.v1"
    arts = data.get("artifacts") or {}
    rel = arts.get("with_contributions_jsonl")
    assert isinstance(rel, str) and rel.endswith("_with_contrib.json")


def test_v7_local_slm_phase3_operator_bundle_baseline():
    p = _ROOT / "experiments" / "baselines" / "v7_local_slm_phase3_operator_bundle_v1.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["schema_version"] == "v7_phase2a_bundle_pointer.v1"
    assert "run_local_slm_phase3_mrmp_chunk.sh" in str(data.get("reproduce_shell", ""))


def test_v7_phase2a_delay_alpha_sweep():
    from experiments.v7_phase2a_delay_sweep import run_alpha_sweep

    out = run_alpha_sweep(
        alphas=[0.12, 0.18],
        tau_max=4,
        seed=0,
        N=6,
        d=4,
        steps=120,
        dt=0.05,
        beta=0.5,
        noise=0.01,
    )
    assert out["schema_version"] == "v7_phase2a_alpha_sweep.v1"
    assert len(out["by_alpha"]) == 2
    assert "tau_exp_proxy_oscillation_jump" in out["by_alpha"][0]


def test_v7_phase2a_compare_runs_cli(tmp_path):
    a = {
        "schema_version": "v7_phase2a_empirical.v1",
        "model": "gpt2",
        "layer_index": -1,
        "n_dialogues": 2,
        "by_tau": [
            {"tau": 0, "R_mean": 0.5, "R_var": 0.02, "n": 2},
            {"tau": 1, "R_mean": 0.4, "R_var": 0.05, "n": 2},
        ],
    }
    p1 = tmp_path / "r1.json"
    p2 = tmp_path / "r2.json"
    p1.write_text(json.dumps(a), encoding="utf-8")
    b = dict(a)
    b["by_tau"] = [
        {"tau": 0, "R_mean": 0.6, "R_var": 0.01, "n": 2},
        {"tau": 1, "R_mean": 0.3, "R_var": 0.08, "n": 2},
    ]
    p2.write_text(json.dumps(b), encoding="utf-8")
    out_j = tmp_path / "cmp.json"
    out_m = tmp_path / "cmp.md"
    r = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "experiments" / "v7_phase2a_compare_runs.py"),
            str(p1),
            str(p2),
            "--labels",
            "a,b",
            "--out-json",
            str(out_j),
            "--out-md",
            str(out_m),
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(out_j.read_text(encoding="utf-8"))
    assert data["schema_version"] == "v7_phase2a_compare_runs.v1"
    assert len(data["runs"]) == 2
    assert "gpt2" in out_m.read_text(encoding="utf-8")


def test_v7_phase2a_primary_aux_tau_association():
    from experiments.v7_phase2a_primary_aux_tau_association import analyze

    def row(tau: int, rm: float, rv: float, n: int) -> dict:
        return {"tau": tau, "R_mean": rm, "R_var": rv, "n": n}

    aux = {
        "trust": [row(0, 0.1, 0.01, 3), row(1, 0.2, 0.02, 3), row(2, 0.15, 0.02, 3)],
        "intent": [row(0, 0.5, 0.01, 3), row(1, 0.4, 0.02, 3), row(2, 0.45, 0.02, 3)],
    }
    data = {
        "by_tau": [row(0, 1.0, 0.1, 3), row(1, 1.1, 0.1, 3), row(2, 1.05, 0.1, 3)],
        "auxiliary_label_delay_coherence": aux,
    }
    out = analyze(data, min_n=1)
    assert out["schema_version"] == "v7_phase2a_primary_aux_tau_assoc.v1"
    assert out["n_tau_used"] == 3
    assert "trust" in out["by_axis"]


def test_v7_phase4_minimal_bundle_build():
    from experiments.v7_phase4_minimal_repro import build_bundle

    b = build_bundle(
        demo=True,
        cpu=True,
        seed=0,
        max_new_tokens=2,
        warmup=1,
        repeats=1,
        with_squad_span=False,
    )
    assert b["schema_version"] == "v7_phase4_minimal_repro.v1"
    assert b["two_tier_sweep"]["schema_version"] == "two_tier_sweep.v1"


def test_v7_phase2a_repro_manifest_verify_detects_tamper(tmp_path):
    import hashlib

    from experiments.v7_phase2a_repro_manifest import verify_manifest

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "x.bin").write_bytes(b"alpha")
    h = hashlib.sha256(b"alpha").hexdigest()
    man = {
        "schema_version": "v7_phase2a_repro_manifest.v1",
        "files": {"x.bin": {"sha256": h, "bytes": 5}},
    }
    mp = repo / "manifest.json"
    mp.write_text(json.dumps(man), encoding="utf-8")
    assert verify_manifest(mp, repo=repo) == []

    (repo / "x.bin").write_bytes(b"beta")
    errs = verify_manifest(mp, repo=repo)
    assert any("不一致" in e for e in errs)
