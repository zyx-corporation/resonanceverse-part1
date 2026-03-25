"""
Phase 1B: Hugging Face 因果 LM + ResonantCore（AwaiIntegratedSLM）の前向きスモーク。

ロードマップ Phase B（SLM アダプタ本格）の橋渡し。`transformers` が入っている環境で実行。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from core.reproducibility import set_experiment_seed
from core.resonant_core import AwaiIntegratedSLM


def main() -> None:
    parser = argparse.ArgumentParser(description="AwaiIntegratedSLM smoke (HF + ResonantCore)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="gpt2", help="transformers モデル名（既定: 小さめ gpt2）")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_experiment_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print("slm_bridge_skip: transformers not installed:", e)
        sys.exit(0)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    base.eval()
    model = AwaiIntegratedSLM(base).to(device)
    model.eval()

    text = "Hello"
    ids = tok.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(ids)
    print("slm_bridge_ok", "logits", tuple(logits.shape), "device", device)


if __name__ == "__main__":
    main()
