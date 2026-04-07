"""話者ペアごとの関係ベクトル系列 ``(T, d)`` を正とする v7 実装用ブリッジ。

手順（設計メモに準拠）:
1. 各ターンの隠れ状態 ``(S, H)`` と、話者ごとのトークン index をマスク平均でプール。
2. 有向ベクトル: 既定は ``concat_truncate``、または ``PairRelationLinear`` で
   ``Linear(2H → d)``（同一重みで入れ替えが ``w_ji``）。
3. ``core.v7_awai_metrics.omega_awai_series_from_w`` で Ω 時系列。

MRMP 窓テキストから index を得るには ``experiments.v7_phase2a_empirical.speaker_token_indices_mrmp_window``
を呼び出し側で用いる（core は experiments に依存しない）。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .v7_awai_metrics import omega_awai_series_from_w


class PairRelationLinear(nn.Module):
    """``concat(h_from, h_to)`` を ``Linear(2H, d)`` で射影（有向ペア用・学習可）。"""

    def __init__(self, hidden_size: int, d: int, *, bias: bool = True) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.d = int(d)
        self.proj = nn.Linear(2 * self.hidden_size, self.d, bias=bias)

    def forward(self, h_from: torch.Tensor, h_to: torch.Tensor) -> torch.Tensor:
        """``h_from``, ``h_to``: (H,) または同形状のバッチ ``(B, H)``。"""
        if h_from.shape != h_to.shape:
            raise ValueError("h_from and h_to must have the same shape")
        x = torch.cat([h_from, h_to], dim=-1)
        return self.proj(x)

    def directed_ij_ji(
        self,
        h_utterer: torch.Tensor,
        h_responder: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """発話者→応答者を ``w_ij``、逆を ``w_ji``（同一 ``proj``）。"""
        w_ij = self(h_utterer, h_responder)
        w_ji = self(h_responder, h_utterer)
        return w_ij, w_ji


def pool_hidden_mean(
    hidden_sh: torch.Tensor,
    token_indices: Sequence[int],
) -> torch.Tensor:
    """``hidden_sh`` (S, H) をトークン index で平均プール。空 index は零ベクトル。"""
    if hidden_sh.dim() != 2:
        raise ValueError("hidden_sh must be (S, H)")
    h_dim = hidden_sh.shape[-1]
    if not token_indices:
        return hidden_sh.new_zeros(h_dim)
    idx = torch.as_tensor(list(token_indices), dtype=torch.long, device=hidden_sh.device)
    s_len = hidden_sh.shape[0]
    idx = idx[(idx >= 0) & (idx < s_len)]
    if idx.numel() == 0:
        return hidden_sh.new_zeros(h_dim)
    return hidden_sh.index_select(0, idx).mean(dim=0)


def concat_truncate_pair_vector(h_i: torch.Tensor, h_j: torch.Tensor, d: int) -> torch.Tensor:
    """``concat(h_i, h_j)`` を長さ ``d`` に切り詰めまたは末尾ゼロパディング（学習なしの既定射影）。"""
    if h_i.shape != h_j.shape or h_i.dim() != 1:
        raise ValueError("h_i and h_j must be 1D same shape (H,)")
    x = torch.cat([h_i, h_j], dim=-1)
    h2 = int(x.shape[-1])
    if h2 >= d:
        return x[:d].contiguous()
    out = x.new_zeros(d)
    out[:h2] = x
    return out


def directed_pair_w_ij_w_ji(
    h_utterer: torch.Tensor,
    h_responder: torch.Tensor,
    d: int,
    *,
    pair_projector: PairRelationLinear | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """発話者→応答者の向きで ``w_ij``、逆を ``w_ji``（各 (d,)）。

    ``pair_projector`` を渡すと ``Linear(2H, d)`` 経由。未指定時は ``concat_truncate``。
    """
    if pair_projector is not None:
        if int(d) != pair_projector.d:
            raise ValueError(f"d={d} must match pair_projector.d={pair_projector.d}")
        return pair_projector.directed_ij_ji(h_utterer, h_responder)
    w_ij = concat_truncate_pair_vector(h_utterer, h_responder, d)
    w_ji = concat_truncate_pair_vector(h_responder, h_utterer, d)
    return w_ij, w_ji


def series_from_turn_hiddens(
    turn_hiddens: Sequence[torch.Tensor],
    utterer_indices: Sequence[Sequence[int]],
    responder_indices: Sequence[Sequence[int]],
    *,
    d: int,
    delay_tau: int = 0,
    pair_projector: PairRelationLinear | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """各ターンの (S,H) と話者トークン列から ``w_ij``, ``w_ji`` (T,d) と Ω (T,) を構築。

    ``turn_hiddens[t]`` は ``utterer_indices[t]`` / ``responder_indices[t]`` と同長の系列を想定。
    ``pair_projector`` 指定時は ``d`` を ``pair_projector.d`` と一致させること。
    """
    if len(turn_hiddens) != len(utterer_indices) or len(turn_hiddens) != len(responder_indices):
        raise ValueError("turn_hiddens, utterer_indices, responder_indices must match in length")
    rows_ij: list[torch.Tensor] = []
    rows_ji: list[torch.Tensor] = []
    for h, iu, ir in zip(turn_hiddens, utterer_indices, responder_indices, strict=True):
        if h.dim() != 2:
            raise ValueError("each turn hidden must be (S, H)")
        hu = pool_hidden_mean(h, iu)
        hr = pool_hidden_mean(h, ir)
        w_ij, w_ji = directed_pair_w_ij_w_ji(hu, hr, d, pair_projector=pair_projector)
        rows_ij.append(w_ij)
        rows_ji.append(w_ji)
    if not rows_ij:
        z = torch.zeros(0, d, device=torch.device("cpu"), dtype=torch.float32)
        return z, z, torch.zeros(0, device=torch.device("cpu"), dtype=torch.float32)
    w_ij_t = torch.stack(rows_ij, dim=0)
    w_ji_t = torch.stack(rows_ji, dim=0)
    omega = omega_awai_series_from_w(w_ij_t, w_ji_t, delay_tau=delay_tau)
    return w_ij_t, w_ji_t, omega


def batch_series_from_dialogues(
    dialogues: Sequence[
        tuple[
            Sequence[torch.Tensor],
            Sequence[Sequence[int]],
            Sequence[Sequence[int]],
        ]
    ],
    *,
    d: int,
    delay_tau: int = 0,
    pair_projector: PairRelationLinear | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """複数対話を同一手続きで処理（各要素は ``series_from_turn_hiddens`` の引数と同型）。"""
    return [
        series_from_turn_hiddens(
            h, iu, ir, d=d, delay_tau=delay_tau, pair_projector=pair_projector
        )
        for h, iu, ir in dialogues
    ]
