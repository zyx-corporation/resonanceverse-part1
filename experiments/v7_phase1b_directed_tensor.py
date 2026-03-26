"""
v7 Phase I-B: 有向テンソル W_D の最小ダイナミクス — w_ij と w_ji の独立した変化の確認（合成）。

ResonanceEngine を数ステップ勾配で更新し、||W - W^T||_F がゼロに落ちないことを確認。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.resonance import ResonanceEngine  # noqa: E402


def run_demo(
    *, seed: int, steps: int, N: int, d: int, lr: float
) -> dict[str, Any]:
    torch.manual_seed(seed)
    eng = ResonanceEngine(num_nodes=N, dim=d, tau=1.0)
    opt = torch.optim.SGD(eng.parameters(), lr=lr)
    trace: list[dict[str, float]] = []
    ctx = F.normalize(torch.randn(d), dim=-1)
    idx = torch.arange(N)

    for t in range(steps):
        opt.zero_grad()
        rs = eng(idx, ctx)
        loss = rs.sum() + 0.01 * eng.W.pow(2).mean()
        loss.backward()
        opt.step()
        w = eng.W.detach()
        asym = float(torch.norm(w - w.transpose(0, 1)))
        trace.append(
            {
                "step": t,
                "loss": float(loss.detach()),
                "asymmetry_fro": asym,
            }
        )

    w = eng.W.detach()
    return {
        "schema_version": "v7_phase1b.v1",
        "N": N,
        "d": d,
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "final_asymmetry_fro": float(torch.norm(w - w.transpose(0, 1))),
        "trace_tail": trace[-5:] if len(trace) >= 5 else trace,
        "note": "ResonanceEngine はパラメータ W が一般に非対称。対称化制約は無い。",
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="v7 Phase I-B: 有向テンソル最小ダイナミクス",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--d", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    payload = run_demo(
        seed=args.seed,
        steps=args.steps,
        N=args.N,
        d=args.d,
        lr=args.lr,
    )
    js = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js, encoding="utf-8")
    print(
        "v7_phase1b_ok",
        json.dumps(
            {"final_asymmetry_fro": payload["final_asymmetry_fro"]},
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
