"""
デコーダブロック列の一部を恒等に差し替え、層計算を省略する（フル系列 forward・use_cache=False）。

対象は `causal_lm_layers.get_decoder_module_list` で解決した ``ModuleList``（GPT-2 系・Llama 系など）。

恒等は hidden_states をそのまま返すため、残差付きブロックと数学的には一致しないが、
**FLOPs／壁時計の削減**を測るための最小「本実装」経路として用いる。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn as nn


class IdentityTransformerBlock(nn.Module):
    """デコーダブロックと同様に **kwargs を受け取り、隠れ状態だけそのまま返す。"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value=None,
        cache_position=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        out: tuple = (hidden_states,)
        if output_attentions:
            out += (None,)
        return out


# 後方互換名
IdentityGPT2Block = IdentityTransformerBlock


@contextmanager
def decoder_layer_skip_context(
    block_modules: nn.ModuleList,
    execute_mask: list[bool],
    *,
    device: torch.device,
) -> Generator[None, None, None]:
    """
    ``execute_mask[i]`` が False のブロックを ``IdentityTransformerBlock`` に一時置換する。

    Args:
        block_modules: ``get_decoder_module_list(model)[0]``
        execute_mask: 長さ ``len(block_modules)``。True=本来のブロック、False=恒等スキップ。
    """
    if len(execute_mask) != len(block_modules):
        raise ValueError("execute_mask length must match number of decoder blocks")
    backup: list[tuple[int, nn.Module]] = []
    try:
        for i, run in enumerate(execute_mask):
            if not run:
                backup.append((i, block_modules[i]))
                block_modules[i] = IdentityTransformerBlock().to(device=device)
        yield
    finally:
        for idx, mod in backup:
            block_modules[idx] = mod


@contextmanager
def gpt2_layer_skip_context(
    h_modules: nn.ModuleList,
    execute_mask: list[bool],
    *,
    device: torch.device,
) -> Generator[None, None, None]:
    """後方互換: ``decoder_layer_skip_context`` と同じ。"""
    with decoder_layer_skip_context(h_modules, execute_mask, device=device):
        yield
