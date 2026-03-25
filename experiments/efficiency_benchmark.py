"""
系列長スイープ: フル二乗スコア行列 vs ROI 階層パスの時間・（CUDA 時）メモリ。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from datetime import datetime, timezone
from typing import Any

import torch

from core.reproducibility import set_experiment_seed
from core.roi_selector import DynamicROISelector


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_full_quadratic(seq_len: int, dim: int, device: torch.device, repeats: int) -> dict[str, float]:
    x = torch.randn(seq_len, dim, device=device)
    t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    import time

    times = []
    for _ in range(repeats):
        _sync(device)
        if t0 is not None:
            t0.record()
        else:
            t_start = time.perf_counter()
        attn = x @ x.t()
        y = attn @ x
        if t0 is not None:
            t1.record()
            torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1) / 1000.0)
        else:
            _sync(device)
            times.append(time.perf_counter() - t_start)

    mem_peak = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(repeats):
            attn = x @ x.t()
            y = attn @ x
        mem_peak = torch.cuda.max_memory_allocated(device)

    return {
        "mean_time_s": float(sum(times) / len(times)),
        "cuda_peak_bytes": int(mem_peak),
    }


def measure_roi_path(num_nodes: int, dim: int, device: torch.device, repeats: int) -> dict[str, float]:
    roi = DynamicROISelector(num_nodes)
    resonance_tensor = torch.randn(num_nodes, dim, device=device)
    current_state = torch.randn(1, dim, device=device)

    import time

    times = []
    for _ in range(repeats):
        _sync(device)
        if device.type == "cuda":
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            _ = roi.select_and_compute(current_state, resonance_tensor)
            t1.record()
            torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1) / 1000.0)
        else:
            t_start = time.perf_counter()
            _ = roi.select_and_compute(current_state, resonance_tensor)
            _sync(device)
            times.append(time.perf_counter() - t_start)

    mem_peak = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(repeats):
            _ = roi.select_and_compute(current_state, resonance_tensor)
        mem_peak = torch.cuda.max_memory_allocated(device)

    return {
        "mean_time_s": float(sum(times) / len(times)),
        "cuda_peak_bytes": int(mem_peak),
    }


def run_sweep(
    seq_lens: list[int],
    dim: int,
    num_nodes: int,
    device: torch.device,
    repeats: int,
) -> dict[str, Any]:
    rows = []
    for seq_len in seq_lens:
        full = measure_full_quadratic(seq_len, dim, device, repeats)
        roi = measure_roi_path(num_nodes, dim, device, repeats)
        ratio = full["mean_time_s"] / roi["mean_time_s"] if roi["mean_time_s"] > 0 else float("inf")
        rows.append(
            {
                "seq_len": seq_len,
                "full_quadratic": full,
                "roi_tiers": roi,
                "time_ratio_full_over_roi": ratio,
            }
        )
    return {"dim": dim, "num_nodes": num_nodes, "device": str(device), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-nodes", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/logs/efficiency_benchmark.json"))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_experiment_seed(args.seed)

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")

    result = run_sweep(args.seq_lens, args.dim, args.num_nodes, device, args.repeats)
    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": args_dict,
        "result": result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["result"], indent=2))


if __name__ == "__main__":
    main()
