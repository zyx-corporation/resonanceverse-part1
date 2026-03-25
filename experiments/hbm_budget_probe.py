"""
Phase 3 M4: HBM バイト予算表テンプレに沿った活性化バイト推定（前向き 1 回）。

- 各対象モジュールの forward 出力テンソルについて numel * element_size を加算。
- GPT2 系は Attention 内スコア／Softmax が融合しているため、行によっては null。
- --demo: transformers 不要のダミーモジュール木。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn

from core.inference_device import select_inference_device
from core.reproducibility import set_experiment_seed

# 二階建て計画 §7.2 CSV と対応（列名は JSON 用に snake_case）
TEMPLATE_ORDER = [
    "Embedding",
    "Attention_QKV",
    "Attention_Score",
    "Attention_Softmax",
    "Attention_WeightedSum",
    "FFN_1",
    "FFN_2",
    "Norm_Residual",
    "Optimizer_State",
    "Collective_Comm",
]


def _bytes_for_tensor(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def _gpt2_bucket(name: str) -> str | None:
    """モジュール名をテンプレ行に分類。None は集計対象外。"""
    if "wte" in name or "wpe" in name:
        return "Embedding"
    if name.endswith("c_attn") or ".c_attn" in name:
        return "Attention_QKV"
    if "attn.c_proj" in name:
        return "Attention_WeightedSum"
    if "mlp.c_fc" in name:
        return "FFN_1"
    if "mlp.c_proj" in name:
        return "FFN_2"
    if name.endswith("ln_1") or name.endswith("ln_2") or ".ln_1" in name or ".ln_2" in name:
        return "Norm_Residual"
    return None


def _demo_bucket(name: str) -> str | None:
    if name.startswith("embedding"):
        return "Embedding"
    if "qkv" in name:
        return "Attention_QKV"
    if "score" in name:
        return "Attention_Score"
    if "softmax" in name:
        return "Attention_Softmax"
    if "weighted" in name:
        return "Attention_WeightedSum"
    if "ffn1" in name:
        return "FFN_1"
    if "ffn2" in name:
        return "FFN_2"
    if "norm" in name:
        return "Norm_Residual"
    return None


class _DemoBlock(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.norm = nn.LayerNorm(h)
        self.qkv = nn.Linear(h, h)
        self.score = nn.Linear(h, h)
        self.softmax = nn.Softmax(dim=-1)
        self.weighted = nn.Linear(h, h)
        self.ffn1 = nn.Linear(h, 4 * h)
        self.ffn2 = nn.Linear(4 * h, h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.qkv(x)
        x = self.score(x)
        x = self.softmax(x)
        x = self.weighted(x)
        x = self.ffn1(x)
        x = self.ffn2(x)
        return x


class _DemoLM(nn.Module):
    def __init__(self, vocab: int, hidden: int, n_layers: int):
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden})()
        self.embedding = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(_DemoBlock(hidden) for _ in range(n_layers))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x


def run_probe(
    *,
    demo: bool,
    model_name: str,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    set_experiment_seed(seed)
    acc: dict[str, int] = defaultdict(int)
    hooks: list[Any] = []

    def make_hook(bucket: str):
        def _fn(_mod, _inp, out):
            if isinstance(out, torch.Tensor):
                acc[bucket] += _bytes_for_tensor(out)

        return _fn

    if demo:
        model = _DemoLM(vocab=256, hidden=64, n_layers=2).to(device)
        bucket_fn = _demo_bucket
        input_ids = torch.randint(
            0, 256, (batch_size, seq_len), device=device, dtype=torch.long
        )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        bucket_fn = _gpt2_bucket
        input_ids = torch.randint(
            0,
            int(model.config.vocab_size),
            (batch_size, seq_len),
            device=device,
            dtype=torch.long,
        )

    for name, mod in model.named_modules():
        b = bucket_fn(name)
        if b is None:
            continue
        hooks.append(mod.register_forward_hook(make_hook(b)))

    model.eval()
    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    rows: list[dict[str, Any]] = []
    for layer in TEMPLATE_ORDER:
        act_b = acc.get(layer)
        row: dict[str, Any] = {
            "layer": layer,
            "fwd_io_b": None,
            "act_b": act_b,
            "bwd_io_b": None,
            "reuse_distance": None,
            "optimization": None,
            "expected_reduction_pct": None,
        }
        if layer in ("Optimizer_State", "Collective_Comm"):
            row["notes"] = "forward のみ計測; 学習時は別途"
        if layer in ("Attention_Score", "Attention_Softmax") and not demo:
            row["notes"] = "GPT2Attention 内融合のため未分離（null の可能性）"
            if acc.get(layer) is None or acc.get(layer) == 0:
                row["act_b"] = None
        rows.append(row)

    total = sum(v for v in acc.values() if v is not None)
    return {
        "schema_version": "hbm_budget.v1",
        "variant": "demo_stub" if demo else model_name,
        "demo": demo,
        "device": str(device),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "rows": rows,
        "total_act_bytes_estimated": int(total),
        "disclaimer": (
            "act_b は forward 出力テンソルの単純合算。重複計上や融合演算の未分離があり、"
            "絶対値ではなく baseline/two_tier の相対比較用。"
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 3 M4: HBM budget template probe")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    device = select_inference_device(force_cpu=args.cpu)
    try:
        payload = run_probe(
            demo=args.demo,
            model_name=args.model,
            device=device,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    except Exception as e:
        print("hbm_budget_probe_skip:", e)
        sys.exit(0)

    print("hbm_budget_probe_ok", json.dumps(payload, ensure_ascii=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
