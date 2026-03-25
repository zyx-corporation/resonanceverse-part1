"""リポジトリルートを pythonpath に含めた上でのスモークテスト（CI 用）。"""

import torch

from core import LightweightResonanceFacade, ResonantCore


def test_lightweight_resonance_facade_forward():
    facade = LightweightResonanceFacade(
        vocab_size=256,
        embed_dim=32,
        resonance_dim=6,
        num_nodes=32,
        tau=1.0,
    )
    tok = torch.randint(0, 256, (1, 16))
    out = facade(tok)
    assert "resonance_scores" in out
    assert out["resonance_scores"].dim() >= 1


def test_resonant_core_forward():
    core = ResonantCore(16, num_nodes=16)
    x = torch.randn(1, 8, 16)
    y = core(x)
    assert y.shape == (1, 8, 6)
