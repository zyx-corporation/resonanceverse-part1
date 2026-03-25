"""
Resonanceverse 統合のスモークテスト（SLM なし）。本番学習は Hugging Face モデルと別途配線。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from core.autopoiesis import AutopoieticInference
from core.config_utils import autopoietic_kwargs, resonant_core_kwargs
from core.lightweight_resonance import LightweightResonanceFacade
from core.reproducibility import set_experiment_seed
from core.resonant_core import ResonantCore


class ToySeqModel(torch.nn.Module):
    """last_hidden_state を返す最小スタブ（AutopoieticInference 用）。"""

    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.lin = torch.nn.Linear(hidden, hidden)

    def forward(self, input_ids: torch.Tensor):
        x = self.emb(input_ids)
        h = self.lin(x)
        out = type("Obj", (), {})()
        out.last_hidden_state = h
        return out


def compute_resonant_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """語彙予測のクロスエントロピー（スモーク用）。"""
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Resonanceverse smoke benchmark (no HF SLM)")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)

    set_experiment_seed(args.seed)
    rc_kw = resonant_core_kwargs()
    core = ResonantCore(32, num_nodes=32, **rc_kw).to(device)
    x = torch.randn(2, 8, 32, device=device)
    y = core(x)
    assert y.shape == (2, 8, 6)

    toy = ToySeqModel(vocab=256, hidden=32).to(device)
    ap_kw = autopoietic_kwargs()
    loop = AutopoieticInference(toy, **ap_kw)
    ids = torch.randint(0, 256, (2, 16), device=device)
    field = torch.zeros(2, 16, 32, device=device)
    field = loop.step(ids, field)
    assert field.shape == (2, 16, 32)

    facade = LightweightResonanceFacade(
        vocab_size=8000,
        embed_dim=128,
        resonance_dim=6,
        num_nodes=128,
        tau=1.0,
    ).to(device)
    tok = torch.randint(0, 8000, (2, 64), device=device)
    out = facade(tok)
    assert "resonance_scores" in out

    targets = torch.randint(0, 500, (2, 64), device=device)
    logits = torch.randn(2, 64, 500, device=device, requires_grad=True)
    loss = compute_resonant_loss(logits, targets)
    loss.backward()

    for _ in range(args.steps):
        core(x)
    print("smoke_ok", float(loss.item()))


if __name__ == "__main__":
    main()
