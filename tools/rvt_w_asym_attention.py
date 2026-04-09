"""
RVT-IMPL-2026-008: Qwen 等 GQA モデル向け W_asym（ブロック間非対称）と注意行列の可視化用ユーティリティ。

MRMP 同型の ``User:`` / ``Assistant:`` 窓テキスト上で、
``speaker_token_indices_mrmp_window`` によるブロック index に対し
各層・各ヘッドの ``S = A_AB - A_BA^T`` を計算する（計画書 §4.1）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.v7_phase2a_empirical import speaker_token_indices_mrmp_window  # noqa: E402
from tools.v7_pair_chat_engine import (  # noqa: E402
    chat_messages_to_mrmp_turns,
    mrmp_window_text,
)


def maybe_expand_gqa_attention(
    attn_bhll: torch.Tensor,
    config: Any,
) -> torch.Tensor:
    """
    ``attn_bhll``: [B, H, L, L]。HF が KV ヘッド数で返す場合に query ヘッド数へ展開。
    """
    n_heads = int(getattr(config, "num_attention_heads", attn_bhll.shape[1]))
    n_kv = int(getattr(config, "num_key_value_heads", n_heads))
    h = int(attn_bhll.shape[1])
    if n_kv < n_heads and h == n_kv:
        rep = n_heads // n_kv
        return attn_bhll.repeat_interleave(rep, dim=1)
    return attn_bhll


def w_asym_tensor_heads(
    attn_bhll: torch.Tensor,
    idx_a: list[int],
    idx_b: list[int],
) -> torch.Tensor:
    """
    1 層分。``attn_bhll`` [B,H,L,L]、戻り値 [B, H, |A|, |B|]。
    """
    if not idx_a or not idx_b:
        raise ValueError("idx_a and idx_b must be non-empty")
    device = attn_bhll.device
    ia = torch.tensor(idx_a, device=device, dtype=torch.long)
    ib = torch.tensor(idx_b, device=device, dtype=torch.long)
    a_ab = attn_bhll[:, :, ia, :][:, :, :, ib]
    a_ba = attn_bhll[:, :, ib, :][:, :, :, ia]
    return a_ab - a_ba.transpose(-1, -2)


def mean_frobenius_w_asym_per_layer(
    attentions: tuple[torch.Tensor | None, ...],
    idx_a: list[int],
    idx_b: list[int],
    config: Any,
) -> list[float]:
    """各層について、全ヘッドの ||S_h||_F の平均。"""
    out: list[float] = []
    for raw in attentions:
        if raw is None:
            out.append(float("nan"))
            continue
        attn = maybe_expand_gqa_attention(raw, config)
        s = w_asym_tensor_heads(attn, idx_a, idx_b)
        # [B,H,na,nb] → ヘッドごとの Frobenius
        per_h = torch.linalg.matrix_norm(s[0], ord="fro", dim=(-2, -1))
        out.append(float(per_h.mean().item()))
    return out


def mean_head_attention_matrix(
    attn_layer: torch.Tensor,
    config: Any,
) -> np.ndarray:
    """1 層・バッチ 0。GQA 展開後にヘッド平均した [L,L]。"""
    attn = maybe_expand_gqa_attention(attn_layer, config)
    a = attn[0].float().mean(dim=0).cpu().numpy()
    return np.asarray(a, dtype=np.float64)


def slice_chat_messages_for_window(
    messages: list[dict[str, str]],
    *,
    window_mode: str,
    last_n: int,
) -> list[dict[str, str]]:
    """Ω タブと同じ窓規則でチャットメッセージを切り出す。"""
    if window_mode == "cumulative":
        return list(messages)
    if window_mode == "last_n":
        ln = max(2, int(last_n))
        if len(messages) <= ln:
            return list(messages)
        return list(messages[-ln:])
    raise ValueError(f"window_mode must be 'cumulative' or 'last_n', got {window_mode!r}")


def chat_window_text_from_messages(
    messages: list[dict[str, str]],
    *,
    window_mode: str,
    last_n: int,
) -> str:
    sub = slice_chat_messages_for_window(
        messages,
        window_mode=window_mode,
        last_n=last_n,
    )
    return mrmp_window_text(chat_messages_to_mrmp_turns(sub))


def downsample_square(m: np.ndarray, max_side: int = 160) -> np.ndarray:
    """長い系列でヒートマップが重くならないよう間引き。"""
    if m.ndim != 2:
        raise ValueError("expected 2D matrix")
    l = m.shape[0]
    if l <= max_side:
        return m
    step = max(1, int(np.ceil(l / max_side)))
    return m[::step, ::step]


def run_w_asym_attention_analysis(
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    window_text: str,
    utterer: str = "User",
    responder: str = "Assistant",
    layer_for_heatmap: int = -1,
    heatmap_max_side: int = 160,
) -> dict[str, Any]:
    """
    MRMP 風窓テキストで 1 回前向きし、W_asym（層別 Frobenius 平均）と注意ヒートマップ用行列を返す。
    """
    schema = "rvt_w_asym_attention.v1"
    idx_u, idx_a = speaker_token_indices_mrmp_window(
        window_text,
        utterer,
        responder,
        tokenizer,
    )
    if not idx_u or not idx_a:
        return {
            "schema_version": schema,
            "error": (
                f"話者ブロックが空です（User={len(idx_u)} Assistant={len(idx_a)}）。"
                " FastTokenizer・「User:」「Assistant:」行形式の窓テキストを確認してください。"
            ),
        }

    enc = tokenizer(
        window_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    n_tok = int(enc["input_ids"].shape[1])

    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    attns = out.attentions
    if not attns:
        return {
            "schema_version": schema,
            "error": "attentions が取得できません。モデルを attn_implementation='eager' でロードしてください。",
        }

    cfg = model.config
    fro_layers = mean_frobenius_w_asym_per_layer(attns, idx_u, idx_a, cfg)

    n_layers = len(attns)
    li = layer_for_heatmap if layer_for_heatmap >= 0 else n_layers - 1
    li = max(0, min(li, n_layers - 1))
    raw_layer = attns[li]
    full_layer: np.ndarray | None = None
    if raw_layer is None:
        attn_matrix = None
    else:
        full_layer = mean_head_attention_matrix(raw_layer, cfg)
        attn_matrix = downsample_square(full_layer, max_side=heatmap_max_side)

    # ブロック間平均注意（可読性用スカラー・ヒートマップと同じ層）
    iu = np.asarray(idx_u, dtype=np.int64)
    ia = np.asarray(idx_a, dtype=np.int64)
    if full_layer is not None and full_layer.shape[0] == n_tok:
        u_to_a = float(full_layer[np.ix_(iu, ia)].mean())
        a_to_u = float(full_layer[np.ix_(ia, iu)].mean())
    else:
        u_to_a = float("nan")
        a_to_u = float("nan")

    return {
        "schema_version": schema,
        "n_tokens": n_tok,
        "n_layers": n_layers,
        "layer_heatmap": li,
        "idx_user": idx_u,
        "idx_assistant": idx_a,
        "w_asym_mean_frobenius_per_layer": fro_layers,
        "mean_attn_user_to_assistant": u_to_a,
        "mean_attn_assistant_to_user": a_to_u,
        "attention_heatmap_matrix": attn_matrix,
        "window_text_preview": window_text[:2000]
        + ("…" if len(window_text) > 2000 else ""),
    }
