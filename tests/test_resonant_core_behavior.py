"""ResonantCore: eval 時の場固定・マスク付き平均・ドリフト RNG。"""

import torch

from core.resonant_core import ResonantCore
from core.reproducibility import set_experiment_seed


def test_resonant_core_eval_does_not_update_w() -> None:
    core = ResonantCore(16, num_nodes=8)
    x = torch.randn(1, 4, 16)
    core.eval()
    w0 = core.W.clone()
    core(x)
    assert torch.allclose(core.W, w0)


def test_resonant_core_train_updates_w() -> None:
    core = ResonantCore(16, num_nodes=8)
    x = torch.randn(1, 4, 16)
    core.train()
    w0 = core.W.clone()
    core(x)
    assert not torch.allclose(core.W, w0)


def test_resonant_core_attention_mask_forward_shape() -> None:
    core = ResonantCore(16, num_nodes=8)
    x = torch.randn(1, 4, 16)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    y = core(x, attention_mask=mask)
    assert y.shape == (1, 4, 6)


def test_resonant_core_field_drift_reproducible_with_same_seed() -> None:
    set_experiment_seed(12345)
    a = ResonantCore(8, num_nodes=4, field_drift_seed=777)
    b = ResonantCore(8, num_nodes=4, field_drift_seed=777)
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 2, 8)
    a.train()
    b.train()
    a(x)
    b(x)
    assert torch.allclose(a.W, b.W)
