import torch

from core.instrumentation import StageTimer
from core.lightweight_resonance import LightweightResonanceFacade
from core.resonant_core import ResonantCore


def test_stage_timer_records_stages():
    device = torch.device("cpu")
    timer = StageTimer(device)
    facade = LightweightResonanceFacade(
        vocab_size=128,
        embed_dim=32,
        resonance_dim=6,
        num_nodes=32,
        tau=1.0,
    )
    tok = torch.randint(0, 128, (1, 16))
    out = facade(tok, instrument=timer)
    assert "resonance_scores" in out
    names = {r["name"] for r in timer.records}
    assert "embedding" in names
    assert "roi_select" in names
    assert len(timer.records) >= 5


def test_resonant_core_instrumented():
    device = torch.device("cpu")
    timer = StageTimer(device)
    core = ResonantCore(32, num_nodes=32)
    x = torch.randn(1, 8, 32)
    y = core(x, instrument=timer)
    assert y.shape == (1, 8, 6)
    names = {r["name"] for r in timer.records}
    assert "project" in names
    assert "field_update" in names
    assert "obscurity_output" in names
