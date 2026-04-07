"""
RVT-EXP-2026-008 L1 拡張: 層ごとの注意をヘッド別 (H, L, L) で取得する。

``v7_phase1a_phi_correlation.hf_forward_attention_layer_matrix`` はヘッド平均後の
単一行列のみを返す。本モジュールはその前段相当。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def hf_forward_attention_layer_heads_numpy(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    text: str,
    layer_index: int,
) -> tuple[np.ndarray | None, dict[str, Any] | None, int]:
    """
    1 回 ``output_attentions=True`` 前向きし、指定層の注意を ``(H, L, L)`` で返す。

    ``layer_attn[batch=0]`` が ``(H, L, L)`` の行ソフトマックス前提（GPT2 系 eager）。
    """
    import torch

    batch = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = batch["input_ids"].to(device)
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    attns = out.attentions
    ntok = int(ids.shape[1])
    if not attns:
        return None, {"schema_version": "rvt_exp_2026_008.v1", "error": "no_attentions"}, ntok
    n_layers = len(attns)
    li = layer_index if layer_index >= 0 else n_layers - 1
    li = max(0, min(int(li), n_layers - 1))
    layer_attn = attns[li]
    if layer_attn is None:
        return None, {"schema_version": "rvt_exp_2026_008.v1", "error": "layer_attn_none"}, ntok
    a = layer_attn[0].cpu().numpy().astype(np.float64)
    if a.ndim != 3:
        return None, {
            "schema_version": "rvt_exp_2026_008.v1",
            "error": "unexpected_attn_shape",
            "shape": list(a.shape),
        }, ntok
    return a, None, ntok
