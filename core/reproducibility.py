"""実験の再現性用シード設定（Phase A / A′）。"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_experiment_seed(seed: int) -> None:
    """Python / NumPy / PyTorch の乱数を固定（完全決定論は GPU 演算の非決定性で保証されない）。"""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
