"""AwaiClassifier の読み出しモード（narrow / projected / dual）の形状検査。"""

import torch

from experiments.slm_downstream import AwaiClassifier, _DemoEncoder


def test_awai_readout_shapes():
    device = torch.device("cpu")
    enc = _DemoEncoder(100, 64).to(device)
    b, s = 2, 12
    x = torch.randint(0, 100, (b, s), device=device)
    mask = torch.ones(b, s, dtype=torch.long, device=device)
    y = 2

    for readout in ("narrow", "projected", "dual"):
        m = AwaiClassifier(enc, y, num_nodes=32, readout=readout).to(device)
        logits = m(x, mask)
        assert logits.shape == (b, y), readout

