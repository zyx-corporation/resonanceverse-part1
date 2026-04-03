"""
設計書 Phase IV の**全景ではない**。方式 B 周辺の既存計測（M1/M3）を 1 JSON に束ねる最小再現。

- two_tier_sweep: baseline vs two_tier_stub のレイテンシ比較（decode_benchmark 再利用）
- 任意: --with-squad-span で squad_span --demo を同梱（レガシー実証との接続）

例::

    python experiments/v7_phase4_minimal_repro.py --demo --cpu --out experiments/logs/v7_phase4_minimal_demo.json
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
) -> dict[str, Any]:
    from experiments.two_tier_sweep import run_two_tier_sweep

    out: dict[str, Any] = {
        "schema_version": "v7_phase4_minimal_repro.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note_ja": (
            "Phase IV（方式 B 統合・下流タスク）の代替ではない。decode / two-tier スタブの最小スモーク。"
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
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="v7 Phase IV 周辺: 最小再現バンドル")
    p.add_argument("--demo", action="store_true", help="HF/transformers を使わない経路")
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
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if not args.demo:
        print(json.dumps({"error": "non_demo_requires_hf", "hint": "use --demo for CI"}), file=sys.stderr)
        raise SystemExit(2)

    bundle = build_bundle(
        demo=True,
        cpu=bool(args.cpu),
        seed=args.seed,
        max_new_tokens=max(1, args.max_new_tokens),
        warmup=max(0, args.warmup),
        repeats=max(1, args.repeats),
        with_squad_span=bool(args.with_squad_span),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("v7_phase4_minimal_repro_ok", json.dumps({"out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
