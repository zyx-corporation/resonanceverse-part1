"""experiments/slm_data.py のユニット（ネットワーク不要）。"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import slm_data


def test_token_ids_to_chunks():
    t = torch.arange(20)
    c = slm_data.token_ids_to_chunks(t, seq_len=6)
    assert c.shape == (3, 6)


def test_batched_chunks():
    c = torch.arange(24).reshape(4, 6)
    batches = list(slm_data.batched_chunks(c, batch_size=2, device=torch.device("cpu")))
    assert len(batches) == 2
    assert batches[0].shape == (2, 6)


def test_train_eval_split():
    t = torch.arange(100)
    tr, ev = slm_data.train_eval_split(t, eval_frac=0.1)
    assert tr.numel() + ev.numel() == 100
    assert ev.numel() == 10
