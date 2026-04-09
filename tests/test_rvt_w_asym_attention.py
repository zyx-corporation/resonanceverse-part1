"""tools.rvt_w_asym_attention のユニットテスト（torch のみ）。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.rvt_w_asym_attention import (  # noqa: E402
    maybe_expand_gqa_attention,
    mean_frobenius_w_asym_per_layer,
    slice_chat_messages_for_window,
    w_asym_tensor_heads,
)


def test_maybe_expand_gqa_repeats_kv_heads():
    cfg = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2)
    x = torch.randn(1, 2, 5, 5)
    y = maybe_expand_gqa_attention(x, cfg)
    assert y.shape == (1, 4, 5, 5)


def test_maybe_expand_gqa_noop_when_heads_match():
    cfg = SimpleNamespace(num_attention_heads=4, num_key_value_heads=4)
    x = torch.randn(1, 4, 5, 5)
    y = maybe_expand_gqa_attention(x, cfg)
    assert y.shape == x.shape


def test_w_asym_tensor_heads_skew_symmetric_block():
    # 単一ヘッド L=4, A={0,1}, B={2,3}
    a = torch.zeros(1, 1, 4, 4)
    a[0, 0, 0, 2] = 1.0
    a[0, 0, 2, 0] = 0.0
    s = w_asym_tensor_heads(a, [0], [2])
    assert s.shape == (1, 1, 1, 1)
    assert abs(float(s[0, 0, 0, 0]) - 1.0) < 1e-5


def test_mean_frobenius_w_asym_per_layer():
    cfg = SimpleNamespace(num_attention_heads=2, num_key_value_heads=2)
    l = 6
    ia, ib = [0, 1], [4, 5]
    t1 = torch.softmax(torch.randn(1, 2, l, l), dim=-1)
    t2 = torch.softmax(torch.randn(1, 2, l, l), dim=-1)
    fro = mean_frobenius_w_asym_per_layer((t1, t2), ia, ib, cfg)
    assert len(fro) == 2
    assert all(isinstance(x, float) for x in fro)
    assert all(not np.isnan(x) for x in fro)


def test_slice_chat_messages_last_n():
    m = [{"role": "user", "content": str(i)} for i in range(5)]
    out = slice_chat_messages_for_window(
        m,
        window_mode="last_n",
        last_n=2,
    )
    assert len(out) == 2
    assert out[0]["content"] == "3"
