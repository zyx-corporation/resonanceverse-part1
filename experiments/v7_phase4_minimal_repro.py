"""
設計書 Phase IV の**全景ではない**。方式 B 周辺の既存計測（M1/M3）を 1 JSON に束ねる最小再現。

- two_tier_sweep: baseline vs two_tier_stub のレイテンシ比較（decode_benchmark 再利用）
- 任意: --with-squad-span で squad_span --demo を同梱（レガシー実証との接続）

例::

    python experiments/v7_phase4_minimal_repro.py \\
        --demo --cpu --out experiments/logs/v7_phase4_minimal_demo.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_squad_span():
    path = _ROOT / "experiments" / "squad_span.py"
    spec = importlib.util.spec_from_file_location("squad_span_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def build_bundle(
    *,
    demo: bool,
    cpu: bool,
    seed: int,
    max_new_tokens: int,
    warmup: int,
    repeats: int,
    with_squad_span: bool,
    with_phase3a_synthetic: bool,
    with_rvt_pointers: bool = False,
    with_oboro_standalone_demo: bool = False,
) -> dict[str, Any]:
    from experiments.two_tier_sweep import run_two_tier_sweep

    out: dict[str, Any] = {
        "schema_version": "v7_phase4_minimal_repro.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note_ja": (
            "Phase IV（方式 B 統合・下流タスク）の代替ではない。"
            " decode / two-tier スタブの最小スモーク。"
            " 本番は設計書 Phase IV と core の統合経路を参照。"
        ),
        "two_tier_sweep": run_two_tier_sweep(
            seed=seed,
            cpu=cpu,
            demo=demo,
            max_new_tokens=max_new_tokens,
            warmup=warmup,
            repeats=repeats,
        ),
    }
    if with_squad_span:
        from core.inference_device import select_inference_device

        mod = _load_squad_span()
        device = select_inference_device(force_cpu=cpu)
        out["squad_span_demo"] = mod.run_demo(device, max_steps=10, seed=seed)
    if with_phase3a_synthetic:
        from core.v7_awai_metrics import run_synthetic_demo

        out["phase3a_synthetic_awai"] = run_synthetic_demo(
            seed=seed,
            T=200,
            d=6,
        )
    if with_oboro_standalone_demo:
        from experiments.rvt_exp_2026_008_oboro_generate import (
            build_oboro_demo_payload,
        )

        out["oboro_standalone_demo"] = {
            "schema_packaging": "v7_phase4_oboro_standalone.v1",
            "note_ja": (
                "L3 Oboro はスタンドアロン CLI。審判・チャット generate 未接続。"
            ),
            "lite": build_oboro_demo_payload(profile="lite"),
            "full": build_oboro_demo_payload(profile="full"),
        }
    if with_rvt_pointers:
        bridge = "docs/planning/rvt_exp_2026_008_architecture_bridge.md"
        out["rvt_pointers"] = {
            "schema_version": "v7_phase4_rvt_pointers.v1",
            "architecture_bridge_md": bridge,
            "judge_axis_mapping_py": (
                "experiments/rvt_exp_2026_008_judge_axis_mapping.py"
            ),
            "scripts": {
                "attn_inject": "experiments/rvt_exp_2026_008_attn_inject.py",
                "oboro_generate": (
                    "experiments/rvt_exp_2026_008_oboro_generate.py"
                ),
                "mrmp_row": "experiments/rvt_exp_2026_008_mrmp_row.py",
                "phase2a_empirical_run": (
                    "experiments/v7_phase2a_empirical_run.py"
                ),
                "ablation_runner": (
                    "experiments/rvt_exp_2026_008_ablation_runner.py"
                ),
                "plan_execute": "experiments/rvt_exp_2026_008_plan_execute.py",
                "eight_grid_unattended_sh": (
                    "experiments/run_rvt_eight_grid_unattended.sh"
                ),
                "mrmp_batch_sh": "experiments/run_rvt_mrmp_batch.sh",
                "l2_smoke": "experiments/rvt_exp_2026_008_l2_smoke.py",
            },
            "cli_examples_ja": {
                "phase2a_l2": (
                    "python experiments/v7_phase2a_empirical_run.py "
                    "--rvt-l2-mode sym|wasym（attn_implementation=eager の "
                    "Causal LM）"
                ),
                "mrmp_head_matrix": (
                    "python experiments/rvt_exp_2026_008_mrmp_row.py "
                    "--jsonl experiments/logs/mrmp_prepared/windows.jsonl "
                    "--head-axis-matrix path.npy"
                ),
                "mrmp_l2": (
                    "python experiments/rvt_exp_2026_008_mrmp_row.py "
                    "--rvt-l2-mode sym --rvt-l2-eps 0.05 ..."
                ),
                "eight_grid_dry": (
                    "bash experiments/run_rvt_eight_grid_unattended.sh"
                ),
                "eight_grid_run": (
                    "RVT_EIGHT_GRID_NO_DRY_RUN=1 bash "
                    "experiments/run_rvt_eight_grid_unattended.sh"
                ),
                "eight_grid_prepend_explore": (
                    "RVT_EIGHT_GRID_PREPEND_EXPLORE=1 bash "
                    "experiments/run_rvt_eight_grid_unattended.sh"
                ),
                "oboro_demo": (
                    "bash experiments/run_rvt_oboro_demo.sh"
                ),
            },
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="v7 Phase IV 周辺: 最小再現バンドル")
    p.add_argument(
        "--demo",
        action="store_true",
        help="HF/transformers を使わない経路",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument(
        "--with-squad-span",
        action="store_true",
        help="squad_span のデモ結果を同梱",
    )
    p.add_argument(
        "--with-phase3a-synthetic",
        action="store_true",
        help="core.v7_awai_metrics の合成 Ω サマリを同梱（Phase III-A 実証なし）",
    )
    p.add_argument(
        "--with-rvt-pointers",
        action="store_true",
        help="RVT-EXP-2026-008 索引（パス文字列のみ・HF なし）を同梱",
    )
    p.add_argument(
        "--with-oboro-standalone-demo",
        action="store_true",
        help=(
            "Oboro L3 の合成デモ JSON（lite/full・HF なし）を同梱"
        ),
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if not args.demo:
        err = json.dumps(
            {"error": "non_demo_requires_hf", "hint": "use --demo for CI"},
            ensure_ascii=False,
        )
        print(err, file=sys.stderr)
        raise SystemExit(2)

    bundle = build_bundle(
        demo=True,
        cpu=bool(args.cpu),
        seed=args.seed,
        max_new_tokens=max(1, args.max_new_tokens),
        warmup=max(0, args.warmup),
        repeats=max(1, args.repeats),
        with_squad_span=bool(args.with_squad_span),
        with_phase3a_synthetic=bool(args.with_phase3a_synthetic),
        with_rvt_pointers=bool(args.with_rvt_pointers),
        with_oboro_standalone_demo=bool(args.with_oboro_standalone_demo),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(bundle, indent=2, ensure_ascii=False) + "\n"
    args.out.write_text(txt, encoding="utf-8")
    ok = json.dumps({"out": str(args.out)}, ensure_ascii=False)
    print("v7_phase4_minimal_repro_ok", ok)


if __name__ == "__main__":
    main()
