"""`get_decoder_module_list` が GPT-2 系で層リストを返すことの単体テスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.two_tier.causal_lm_layers import get_decoder_module_list, num_decoder_layers
from transformers import GPT2Config, GPT2LMHeadModel


def test_gpt2_decoder_list():
    cfg = GPT2Config(n_layer=3, n_head=2, n_embd=32, vocab_size=100)
    m = GPT2LMHeadModel(cfg)
    blocks, kind = get_decoder_module_list(m)
    assert kind == "transformer.h"
    assert len(blocks) == 3
    assert num_decoder_layers(m) == 3
