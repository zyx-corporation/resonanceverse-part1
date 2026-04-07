"""
RVT L2 統合スモーク: **eager** Causal LM で ``sym`` 注入有無の logits 差分を 1 回計測する。

審判・チャット経路とは独立。**短い文**・**評価モード**想定。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def run_smoke(
    *,
    model_name: str,
    text: str,
    cpu: bool,
    layer_index: int,
    inject_mode: str,
    eps: float,
    seed: int,
) -> dict[str, Any]:
    import torch

    from core.inference_device import select_inference_device
    from core.reproducibility import set_experiment_seed
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from experiments.rvt_exp_2026_008_attn_inject import (
        apply_rvt_inject_to_causal_lm,
        clear_rvt_inject_from_causal_lm,
        gpt2_rvt_inject_session,
        iter_causal_lm_decoder_layers,
    )

    set_experiment_seed(seed)
    device = select_inference_device(force_cpu=cpu)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    ids = ids.to(device)

    with torch.no_grad():
        base_logits = model(ids).logits.float()

    n_layers = len(iter_causal_lm_decoder_layers(model))
    li = layer_index if layer_index >= 0 else n_layers - 1
    li = max(0, min(int(li), n_layers - 1))

    with torch.no_grad(), gpt2_rvt_inject_session():
        apply_rvt_inject_to_causal_lm(
            model,
            layer_indices=[li],
            mode=inject_mode,
            eps=eps,
        )
        inj_logits = model(ids).logits.float()
        clear_rvt_inject_from_causal_lm(model)

    delta = (base_logits - inj_logits).abs().max().item()
    rel = float(delta / (base_logits.abs().max().item() + 1e-8))
    return {
        "schema_version": "rvt_exp_008_l2_smoke.v1",
        "ok": True,
        "model": model_name,
        "inject_layer": li,
        "inject_mode": inject_mode,
        "inject_eps": eps,
        "logits_abs_max_delta": float(delta),
        "logits_rel_delta_vs_abs_max": rel,
        "n_tokens": int(ids.shape[1]),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="RVT L2 attn inject smoke (eager Causal LM)",
    )
    p.add_argument("--model", default="gpt2")
    p.add_argument(
        "--text",
        default="Hello, this is a short test for attention inject.",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--mode", default="sym", choices=("sym", "wasym", "base"))
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if args.mode == "base":
        payload = {
            "schema_version": "rvt_exp_008_l2_smoke.v1",
            "ok": False,
            "error": "use mode sym or wasym for inject smoke",
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        sys.exit(2)
    payload = run_smoke(
        model_name=args.model,
        text=args.text,
        cpu=args.cpu,
        layer_index=args.layer,
        inject_mode=args.mode,
        eps=args.eps,
        seed=args.seed,
    )
    js = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js + "\n", encoding="utf-8")
    print(js)


if __name__ == "__main__":
    main()
