"""
v7 Phase I-A 実証ベースライン（リポジトリ内）の一括実行。

- --demo: HF なし。パイロット／autoproxy は合成経路。層別 HF 統計は省略（CI 向け）。
- 既定: gpt2 でパイロット JSONL・autoproxy・参照 1 文の層別 S_asym 統計をまとめた JSON を出力。

本番規模の主張境界は docs/planning/v7_phase1a_empirical_prereg_v1.json を参照。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_REFERENCE_TEXT = (
    "Hello, we need to align on the deadline before Friday."
)
_PREREG_REL = "docs/planning/v7_phase1a_empirical_prereg_v1.json"


def run_empirical_bundle(
    *,
    demo: bool,
    model_name: str,
    cpu: bool,
    seed: int,
    jsonl_path: Path,
    reference_text: str,
) -> dict[str, Any]:
    from experiments.v7_experiment_meta import collect_runtime_meta
    from experiments.v7_phase1a_autoproxy import _BUILTIN_TEXTS, run_autoproxy
    from experiments.v7_phase1a_phi_correlation import run_hf_no_labels, run_synthetic_demo
    from experiments.v7_phase1a_pilot_jsonl import load_jsonl, run_pilot

    prereg_path = _ROOT / _PREREG_REL
    bundle: dict[str, Any] = {
        "schema_version": "v7_empirical_bundle.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prereg_relative_path": _PREREG_REL,
        "prereg_present": prereg_path.is_file(),
        "runtime": collect_runtime_meta(),
        "seed": seed,
        "model": model_name,
        "demo": demo,
        "jsonl": str(jsonl_path.relative_to(_ROOT)) if jsonl_path.is_relative_to(_ROOT) else str(jsonl_path),
    }

    rows = load_jsonl(jsonl_path)
    bundle["phase1a_pilot_jsonl"] = run_pilot(
        rows=rows,
        demo=demo,
        model_name=model_name,
        cpu=cpu,
        seed=seed,
        layer_index=-1,
    )
    bundle["phase1a_autoproxy"] = run_autoproxy(
        texts=list(_BUILTIN_TEXTS),
        demo=demo,
        model_name=model_name,
        cpu=cpu,
        seed=seed,
    )

    if demo:
        bundle["phase1a_reference_layers"] = {
            "schema_version": "v7_phase1a.v1",
            "mode": "omitted_in_demo",
            "note": "層別 HF 統計は transformers 前向きが必要。--demo なしで再実行すること。",
        }
        bundle["phase1a_synthetic_sanity"] = run_synthetic_demo(seed=seed, n_samples=120)
    else:
        bundle["phase1a_reference_layers"] = run_hf_no_labels(
            model_name=model_name,
            text=reference_text,
            cpu=cpu,
            seed=seed,
        )
        bundle["phase1a_synthetic_sanity"] = None

    return bundle


def main() -> None:
    p = argparse.ArgumentParser(description="v7 Phase I-A 実証ベースライン一括")
    p.add_argument("--demo", action="store_true", help="HF 省略（合成パイロット／autoproxy）")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--jsonl",
        type=Path,
        default=_ROOT / "experiments" / "data" / "v7_phase1a_pilot.jsonl",
    )
    p.add_argument(
        "--reference-text",
        default=DEFAULT_REFERENCE_TEXT,
        help="HF 時の層別統計用 1 文",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    payload = run_empirical_bundle(
        demo=args.demo,
        model_name=args.model,
        cpu=args.cpu,
        seed=args.seed,
        jsonl_path=args.jsonl,
        reference_text=args.reference_text,
    )

    if payload["phase1a_pilot_jsonl"].get("error"):
        sys.exit(1)
    if payload["phase1a_autoproxy"].get("error"):
        sys.exit(1)
    if not args.demo and payload["phase1a_reference_layers"].get("error"):
        sys.exit(1)

    js = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js, encoding="utf-8")
    print(
        "v7_empirical_run_ok",
        json.dumps(
            {"demo": args.demo, "out": str(args.out) if args.out else None},
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
