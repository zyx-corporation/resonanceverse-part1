"""
Phase 1B: 文化的文脈の作動的スカラー（「あわい」前処理の最小実装）。

大規模言語意味は置かず、埋め込み列から調製スカラーを得て共鳴場への入力を和らげる／強める。
ロードマップ Phase B（SLM 本接続）の前段として利用する。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def awai_pressure_from_embeddings(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    系列内分散に基づく簡易スカラー (B,) — 文脈の「ばらつき」指標（スタブ）。

    Args:
        x: (B, S, H) 埋め込みまたは隠れ状態
    Returns:
        (B,) ∈ (0,1) 近傍（シグモイド）
    """
    v = x.var(dim=1, unbiased=False).mean(dim=-1).clamp_min(eps)
    return torch.sigmoid(v)


class CulturalModulationAdapter(nn.Module):
    """
    埋め込みからバッチごとの調製係数 (B,1,1) を生成。
    `LightweightResonanceFacade.forward(..., cultural_scale=...)` に渡すか、
    手動で共鳴特徴 r に乗算する。
    """

    def __init__(self, hidden_dim: int, hidden_reduce: int | None = None):
        super().__init__()
        h = hidden_reduce or max(8, hidden_dim // 4)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, h),
            nn.Tanh(),
            nn.Linear(h, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,S,H) -> (B,1,1)"""
        pooled = x.mean(dim=1)
        return torch.sigmoid(self.mlp(pooled)).unsqueeze(-1)
