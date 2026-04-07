"""tools.v7_awai_chat_generate の初期化・生成スモーク。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.resonant_core import AwaiIntegratedSLM

from tools.v7_awai_chat_generate import (
    generate_completion_awai,
    init_awai_out_head_from_lm_head,
)
from tools.v7_pair_chat_engine import unpack_model_bundle


class _StubWithLmHead(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 32):
        super().__init__()
        self.config = type(
            "Cfg",
            (),
            {
                "hidden_size": hidden_size,
                "vocab_size": vocab_size,
                "pad_token_id": 0,
                "n_positions": 128,
            },
        )()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False):
        h = self.emb(input_ids)
        out = type("Out", (), {})()
        out.hidden_states = [h, h]
        return out


def test_init_awai_out_head_from_lm_head():
    base = _StubWithLmHead()
    awai = AwaiIntegratedSLM(base)
    assert init_awai_out_head_from_lm_head(awai)
    h = int(base.config.hidden_size)
    assert torch.allclose(awai.out_head.weight[:, :h], base.lm_head.weight)


def test_generate_completion_awai_greedy_smoke():
    device = torch.device("cpu")
    awai = AwaiIntegratedSLM(_StubWithLmHead())
    init_awai_out_head_from_lm_head(awai)

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, prompt, return_tensors=None, add_special_tokens=False):
            return {"input_ids": torch.tensor([[2, 3, 4, 5]], dtype=torch.long)}

        def decode(self, ids, skip_special_tokens=True):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return "".join(str(int(x)) for x in ids)

    out = generate_completion_awai(
        awai,
        _Tok(),
        device,
        "hello",
        max_new_tokens=3,
        do_sample=False,
        repetition_penalty=1.0,
    )
    assert isinstance(out, str)


def test_unpack_model_bundle_legacy_and_four():
    dev = torch.device("cpu")
    assert unpack_model_bundle((1, 2, dev)) == (1, 2, dev, None)
    x = object()
    assert unpack_model_bundle((1, 2, dev, x)) == (1, 2, dev, x)
