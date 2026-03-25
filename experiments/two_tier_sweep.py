"""
Phase 3 M3: baseline と two-tier-stub を同一ハイパラで連続実行し、比較 JSON を 1 行で出す。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_decode_benchmark():
    path = _ROOT / "experiments" / "decode_benchmark.py"
    spec = importlib.util.spec_from_file_location("decode_benchmark_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def run_two_tier_sweep(
    *,
    seed: int = 0,
    cpu: bool = False,
    demo: bool = False,
    model: str = "gpt2",
    max_new_tokens: int = 8,
    warmup: int = 1,
    repeats: int = 3,
    router_step_stride: int | None = None,
) -> dict:
    """baseline / two_tier_stub を同一条件で実行し `two_tier_sweep.v1` オブジェクトを返す。"""
    from core.inference_device import reset_peak_memory_stats_if_cuda, select_inference_device

    device = select_inference_device(force_cpu=cpu)
    reset_peak_memory_stats_if_cuda(device)

    mod = _load_decode_benchmark()
    common = dict(
        demo=demo,
        model_name=model,
        device=device,
        max_new_tokens=max_new_tokens,
        warmup=warmup,
        repeats=repeats,
        seed=seed,
        router_step_stride=router_step_stride,
    )
    b = mod.run_decode_benchmark(two_tier_stub=False, **common)
    reset_peak_memory_stats_if_cuda(device)
    t = mod.run_decode_benchmark(two_tier_stub=True, **common)

    p50b, p50t = b.get("latency_ms_p50"), t.get("latency_ms_p50")
    ratio = None
    if (
        isinstance(p50b, (int, float))
        and isinstance(p50t, (int, float))
        and p50t
        and not (p50t != p50t)
        and not (p50b != p50b)
    ):
        ratio = float(p50b) / float(p50t)

    return {
        "schema_version": "two_tier_sweep.v1",
        "baseline": b,
        "two_tier_stub": t,
        "comparison": {
            "latency_ms_p50_ratio_baseline_over_two_tier": ratio,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase 3 M3: baseline vs two_tier_stub sweep",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--router-step-stride",
        type=int,
        default=None,
        metavar="N",
        help="decode_benchmark と同じ。two_tier_stub 側にのみ適用",
    )
    args = p.parse_args()

    merged = run_two_tier_sweep(
        seed=args.seed,
        cpu=args.cpu,
        demo=args.demo,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
        repeats=args.repeats,
        router_step_stride=args.router_step_stride,
    )
    line = json.dumps(merged, ensure_ascii=False)
    print("two_tier_sweep_ok", line)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        txt = json.dumps(merged, indent=2, ensure_ascii=False)
        args.out.write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
