"""Phase 1B: 文化的調製と SLM 橋渡し（スタブ／軽量）。"""

import torch
import torch.nn as nn

from core import (
    AwaiIntegratedSLM,
    CulturalModulationAdapter,
    LightweightResonanceFacade,
    awai_pressure_from_embeddings,
)


def test_awai_pressure_from_embeddings():
    x = torch.randn(2, 8, 16)
    p = awai_pressure_from_embeddings(x)
    assert p.shape == (2,)
    assert (p >= 0).all() and (p <= 1).all()


def test_cultural_modulation_adapter():
    ad = CulturalModulationAdapter(32)
    x = torch.randn(1, 4, 32)
    s = ad(x)
    assert s.shape == (1, 1, 1)


def test_facade_cultural_scale():
    facade = LightweightResonanceFacade(
        vocab_size=64,
        embed_dim=32,
        resonance_dim=6,
        num_nodes=16,
        tau=1.0,
    )
    tok = torch.randint(0, 64, (1, 8))
    emb = facade.embedding(tok)
    scale = CulturalModulationAdapter(32)(emb)
    out = facade(tok, cultural_scale=scale)
    assert "resonance_scores" in out


class _FakeHF(nn.Module):
    """AwaiIntegratedSLM が期待する最小インタフェース。"""

    def __init__(self, hidden: int = 32, vocab: int = 100):
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": hidden, "vocab_size": vocab})()
        self.emb = nn.Embedding(vocab, hidden)

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False):
        h = self.emb(input_ids)
        out = type("Out", (), {})()
        out.hidden_states = [h, h]
        return out


def test_awai_integrated_slm_fake_base():
    base = _FakeHF()
    m = AwaiIntegratedSLM(base)
    ids = torch.randint(0, 100, (1, 5))
    logits = m(ids)
    assert logits.shape == (1, 5, 100)


def test_awai_integrated_slm_cultural_modulation():
    base = _FakeHF()
    m = AwaiIntegratedSLM(base, cultural_modulation=True)
    ids = torch.randint(0, 100, (1, 5))
    logits = m(ids)
    assert logits.shape == (1, 5, 100)
    assert m.cultural_adapter is not None
