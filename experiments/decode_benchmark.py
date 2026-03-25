"""
Phase 3 M1: 因果デコード 1 トークンずつのレイテンシ計測（p50/p95）と CUDA ピークメモリ。

--demo: transformers 不要。同一 JSON スキーマでオフライン検証可能。
--two-tier-stub: Router/Controller スタブを各ステップに挟み、keep 割合をログ（M2 接続）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn

from core.reproducibility import set_experiment_seed
from core.two_tier import (
    BlockRouterStub,
    SequenceControllerStub,
    router_keep_fraction,
)


def _percentiles_ms(step_times_s: list[float]) -> tuple[float, float]:
    if not step_times_s:
        return float("nan"), float("nan")
    ms = np.array(step_times_s, dtype=np.float64) * 1000.0
    return float(np.percentile(ms, 50)), float(np.percentile(ms, 95))


def run_decode_benchmark(
    *,
    demo: bool,
    model_name: str,
    device: torch.device,
    max_new_tokens: int,
    warmup: int,
    repeats: int,
    two_tier_stub: bool,
    seed: int,
    router_step_stride: int | None = None,
) -> dict[str, Any]:
    set_experiment_seed(seed)
    step_times_s: list[float] = []
    keep_fracs: list[float] = []

    if demo:
        h = 64
        layer = nn.Linear(h, h).to(device)
        x = torch.randn(1, 1, h, device=device)
        ctrl = SequenceControllerStub(h).to(device)
        router = BlockRouterStub(tau=0.5, step_stride=router_step_stride)

        def one_step() -> None:
            nonlocal x
            t0 = time.perf_counter()
            x = layer(x)
            if two_tier_stub:
                pr = ctrl(x)
                keep = router(pr)
                keep_fracs.append(router_keep_fraction(keep))
            step_times_s.append(time.perf_counter() - t0)

        for _ in range(warmup):
            if two_tier_stub:
                router.reset()
            for _ in range(max_new_tokens):
                one_step()
        for _ in range(repeats):
            if two_tier_stub:
                router.reset()
            for _ in range(max_new_tokens):
                one_step()
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
        ctrl = (
            SequenceControllerStub(model.config.hidden_size).to(device)
            if two_tier_stub
            else None
        )
        router = (
            BlockRouterStub(tau=0.5, step_stride=router_step_stride)
            if two_tier_stub
            else None
        )

        prompt = "The quick brown fox jumps."

        def run_chain(record: bool) -> None:
            if two_tier_stub and router is not None:
                router.reset()
            input_ids = tok.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                pre = model(
                    input_ids,
                    use_cache=True,
                    output_hidden_states=two_tier_stub,
                )
            past_key_values = pre.past_key_values
            next_id = pre.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for _ in range(max_new_tokens):
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model(
                        next_id,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=two_tier_stub,
                    )
                if two_tier_stub and ctrl is not None and router is not None:
                    h_last = out.hidden_states[-1][:, -1:, :]
                    pr = ctrl(h_last)
                    keep = router(pr)
                    if record:
                        keep_fracs.append(router_keep_fraction(keep))
                next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                past_key_values = out.past_key_values
                elapsed = time.perf_counter() - t0
                if record:
                    step_times_s.append(elapsed)

        for _ in range(warmup):
            run_chain(record=False)
        for _ in range(repeats):
            run_chain(record=True)

    p50, p95 = _percentiles_ms(step_times_s)
    total_steps = len(step_times_s)
    mean_s = float(np.mean(step_times_s)) if step_times_s else float("nan")

    cuda_peak = None
    if device.type == "cuda":
        cuda_peak = int(torch.cuda.max_memory_allocated(device))

    payload: dict[str, Any] = {
        "schema_version": "decode_benchmark.v1",
        "variant": "two_tier_stub" if two_tier_stub else "baseline",
        "demo": demo,
        "model": "_demo_linear" if demo else model_name,
        "device": str(device),
        "max_new_tokens": max_new_tokens,
        "warmup_rounds": warmup,
        "repeats": repeats,
        "total_decode_steps": total_steps,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
        "per_step_time_mean_s": mean_s,
        "hbm_proxy_cuda_peak_bytes": cuda_peak,
    }
    if two_tier_stub and keep_fracs:
        payload["router_keep_fraction_mean"] = float(np.mean(keep_fracs))
    if two_tier_stub:
        payload["router_stub_mode"] = (
            "step_stride" if router_step_stride is not None else "priority_threshold"
        )
        if router_step_stride is not None:
            payload["router_step_stride"] = int(router_step_stride)
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 3 M1: decode latency + memory JSON")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--two-tier-stub",
        action="store_true",
        help="各ステップで SequenceControllerStub+BlockRouterStub を実行（M2）",
    )
    p.add_argument(
        "--router-step-stride",
        type=int,
        default=None,
        metavar="N",
        help="指定時は優先度ではなくステップ番号 mod N==0 で keep（決定的・P3）。--two-tier-stub と併用",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        payload = run_decode_benchmark(
            demo=args.demo,
            model_name=args.model,
            device=device,
            max_new_tokens=args.max_new_tokens,
            warmup=args.warmup,
            repeats=args.repeats,
            two_tier_stub=args.two_tier_stub,
            seed=args.seed,
            router_step_stride=args.router_step_stride,
        )
    except Exception as e:
        print("decode_benchmark_skip:", e)
        sys.exit(0)

    print("decode_benchmark_ok", json.dumps(payload, ensure_ascii=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
