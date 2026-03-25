"""
因果言語モデルのデコーダ層の一部を恒等に置換し、フル系列 forward の時間差を測る（資源削減の実測）。

- ``--demo``: ネットワーク不要。小さな ``GPT2LMHeadModel(config)`` のランダム初期重み。
- 実モデル: ``--model`` に Hugging Face の ``*ForCausalLM`` 名（例: ``gpt2``、``TinyLlama/TinyLlama-1.1B-Chat-v1.0``）。
  未対応アーキテクチャは ``core/two_tier/causal_lm_layers.py`` に分岐を追加。

``use_cache=False`` の 1 回 forward。恒等スキップは数学的には元の LM と一致しないが、
層計算省略による **時間差** を主張するための最小経路。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from core.inference_device import max_memory_allocated_bytes, reset_peak_memory_stats_if_cuda, select_inference_device, sync_inference_device
from core.reproducibility import set_experiment_seed
from core.two_tier.causal_lm_layers import get_decoder_module_list
from core.two_tier.gpt2_identity_skip import decoder_layer_skip_context


def _time_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    repeats: int,
    device: torch.device,
) -> list[float]:
    times: list[float] = []
    for _ in range(repeats):
        sync_inference_device(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids, use_cache=False)
        sync_inference_device(device)
        times.append(time.perf_counter() - t0)
    return times


def run_benchmark(
    *,
    demo: bool,
    model_name: str,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    repeats: int,
    warmup: int,
    seed: int,
    skip_every: int,
    execute_mask: list[bool] | None,
) -> dict:
    set_experiment_seed(seed)
    from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

    if demo:
        config = GPT2Config(
            n_layer=6,
            n_head=4,
            n_embd=128,
            n_positions=max(512, seq_len + 8),
            vocab_size=256,
        )
        model = GPT2LMHeadModel(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model = model.to(device)
    model.eval()
    blocks, stack_kind = get_decoder_module_list(model)
    n_layer = len(blocks)

    if execute_mask is not None:
        mask = list(execute_mask)
        if len(mask) != n_layer:
            raise ValueError("execute_mask length must match n_layer")
    else:
        if skip_every < 1:
            raise ValueError("skip_every must be >= 1")
        mask = [i % skip_every == 0 for i in range(n_layer)]

    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )

    # ウォームアップ
    for _ in range(warmup):
        with torch.no_grad():
            model(input_ids, use_cache=False)
    reset_peak_memory_stats_if_cuda(device)

    t_full = _time_forward(model, input_ids, repeats=repeats, device=device)
    peak_full = max_memory_allocated_bytes(device)

    reset_peak_memory_stats_if_cuda(device)

    with decoder_layer_skip_context(blocks, mask, device=device):
        t_skip = _time_forward(model, input_ids, repeats=repeats, device=device)
    peak_skip = max_memory_allocated_bytes(device)

    layers_executed = int(sum(mask))
    ratio_time = float(np.mean(t_full) / max(np.mean(t_skip), 1e-12))

    return {
        "schema_version": "causal_lm_layer_skip_benchmark.v1",
        "demo": demo,
        "model": "_demo_random_gpt2" if demo else model_name,
        "model_class": type(model).__name__,
        "decoder_stack_kind": stack_kind,
        "device": str(device),
        "seq_len": seq_len,
        "batch_size": batch_size,
        "n_layer": n_layer,
        "layer_execute_mask": mask,
        "layers_executed": layers_executed,
        "layers_skipped": n_layer - layers_executed,
        "forward_ms_mean_full": float(np.mean(t_full) * 1000.0),
        "forward_ms_mean_skip": float(np.mean(t_skip) * 1000.0),
        "time_ratio_full_over_skip": ratio_time,
        "cuda_peak_bytes_full": peak_full,
        "cuda_peak_bytes_skip": peak_skip,
        "disclaimer": (
            "恒等スキップは数学的に元の LM と一致しない。層計算省略による時間・メモリ傾向の実測用。"
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Causal LM decoder layer identity skip timing (full sequence forward)"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument(
        "--model",
        default="gpt2",
        help="Hugging Face の因果 LM 名（--demo 時は無視）。未対応時は causal_lm_layers を拡張",
    )
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument(
        "--skip-every",
        type=int,
        default=2,
        metavar="N",
        help="1 ならマスク全 True（スキップなし）。2 なら i%%N==0 の層だけ実行（既定で比較用）",
    )
    p.add_argument(
        "--execute-mask",
        type=str,
        default=None,
        help="例: 1,1,0,1,0,1（カンマ区切り 0/1。--demo の n_layer=6 と一致させる）",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    device = select_inference_device(force_cpu=args.cpu)
    em: list[bool] | None = None
    if args.execute_mask:
        em = [bool(int(x.strip())) for x in args.execute_mask.split(",")]

    try:
        payload = run_benchmark(
            demo=args.demo,
            model_name=args.model,
            device=device,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            repeats=args.repeats,
            warmup=args.warmup,
            seed=args.seed,
            skip_every=args.skip_every,
            execute_mask=em,
        )
    except Exception as e:
        print("gpt2_layer_skip_benchmark_skip:", e)
        sys.exit(0)

    print("gpt2_layer_skip_benchmark_ok", json.dumps(payload, ensure_ascii=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
