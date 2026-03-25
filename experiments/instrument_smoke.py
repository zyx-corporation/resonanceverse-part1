"""
LightweightResonanceFacade + StageTimer のスモーク（CI 用）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from core.instrumentation import StageTimer
from core.lightweight_resonance import LightweightResonanceFacade
from core.reproducibility import set_experiment_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="StageTimer smoke on LightweightResonanceFacade")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu", action="store_true", help="device を cpu に固定（efficiency_benchmark と同様）")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="指定時は JSON をファイルへ保存（regression_check --mode instrument 用）",
    )
    args = parser.parse_args()
    if args.cpu:
        args.device = "cpu"

    set_experiment_seed(args.seed)
    device = torch.device(args.device)

    facade = LightweightResonanceFacade(
        vocab_size=512,
        embed_dim=64,
        resonance_dim=6,
        num_nodes=64,
        tau=1.0,
    ).to(device)

    tok = torch.randint(0, 512, (2, 32), device=device)
    timer = StageTimer(device)
    out = facade(tok, instrument=timer)

    assert "resonance_scores" in out
    assert len(timer.records) >= 1

    payload = {
        "seed": args.seed,
        "device": str(device),
        "stages": timer.to_jsonable(),
        "output_shapes": {
            "resonance_scores": list(out["resonance_scores"].shape),
        },
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
