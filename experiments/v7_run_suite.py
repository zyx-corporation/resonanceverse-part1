"""
v7 実験スイート一括実行: Phase I-A（demo）, I-B, II-A, III-A（合成）を連続し JSON をまとめる。

例::
    python experiments/v7_run_suite.py --demo --out experiments/logs/v7_suite/suite.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_mod(fname: str, as_name: str):
    path = _ROOT / "experiments" / fname
    spec = importlib.util.spec_from_file_location(as_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser(description="v7 suite runner")
    p.add_argument("--demo", action="store_true", help="軽量パラメータ（CI）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    p1a = _load_mod("v7_phase1a_phi_correlation.py", "v7_p1a")
    p1a_ap = _load_mod("v7_phase1a_autoproxy.py", "v7_p1a_ap")
    p1b = _load_mod("v7_phase1b_directed_tensor.py", "v7_p1b")
    p2a = _load_mod("v7_phase2a_delay_sweep.py", "v7_p2a")
    p3a = _load_mod("v7_phase3a_awai_metrics.py", "v7_p3a")

    if args.demo:
        n_s, tau_max, steps_2a, N2, d2 = 120, 6, 800, 8, 6
        steps_1b = 40
    else:
        n_s, tau_max, steps_2a, N2, d2 = 400, 12, 4000, 12, 6
        steps_1b = 80

    bundle = {
        "schema_version": "v7_suite_bundle.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "phase1a": p1a.run_synthetic_demo(seed=args.seed, n_samples=n_s),
        "phase1a_autoproxy": p1a_ap.run_autoproxy(
            texts=list(p1a_ap._BUILTIN_TEXTS),
            demo=args.demo,
            model_name="gpt2",
            cpu=args.demo,
            seed=args.seed,
        ),
        "phase1b": p1b.run_demo(seed=args.seed, steps=steps_1b, N=16, d=6, lr=0.05),
        "phase2a": p2a.run_sweep(
            tau_max=tau_max,
            seed=args.seed,
            N=N2,
            d=d2,
            steps=steps_2a,
            dt=0.05,
            alpha=0.15,
            beta=0.85,
            noise=0.02,
        ),
        "phase3a": p3a.run_demo(seed=args.seed, T=200, d=6),
    }

    js = json.dumps(bundle, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js, encoding="utf-8")
    print("v7_run_suite_ok", json.dumps({"out": str(args.out) if args.out else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
