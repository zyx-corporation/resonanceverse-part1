"""
Hugging Face `*ForCausalLM` からデコーダブロック列（``nn.ModuleList``）を取得する。

比較実験でモデルを差し替えるときは、ここにアーキテクチャ分岐を追加する。
"""

from __future__ import annotations

import torch.nn as nn


def get_decoder_module_list(model: nn.Module) -> tuple[nn.ModuleList, str]:
    """
    層恒等スキップの対象となる ``ModuleList`` と、識別子を返す。

    Returns:
        (blocks, stack_kind): ``stack_kind`` はログ用の短いラベル。

    Raises:
        ValueError: 未対応アーキテクチャ（``causal_lm_layers.py`` にパスを追加すること）。
    """
    # GPT-2 / GPT-Neo / GPT-J / DistilGPT2 等
    tr = getattr(model, "transformer", None)
    if tr is not None and hasattr(tr, "h"):
        return tr.h, "transformer.h"

    # Llama / Mistral / Qwen / Phi / Gemma 等（LlamaForCausalLM 系）
    inner = getattr(model, "model", None)
    if inner is not None:
        layers = getattr(inner, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return layers, "model.layers"
        dec = getattr(inner, "decoder", None)
        if dec is not None:
            dl = getattr(dec, "layers", None)
            if isinstance(dl, nn.ModuleList):
                return dl, "model.decoder.layers"

    raise ValueError(
        f"未対応の CausalLM 構造です: {type(model).__name__}. "
        f"`{__name__}.get_decoder_module_list` に分岐を追加してください。"
    )


def num_decoder_layers(model: nn.Module) -> int:
    blocks, _ = get_decoder_module_list(model)
    return len(blocks)
