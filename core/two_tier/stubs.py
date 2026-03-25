"""
Phase 3 M2: 二階建てのスタブ（Router / Controller）。
本番のブロックスキップは未実装。計測・ログ形式の接続点として使用する。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SequenceControllerStub(nn.Module):
    """
    ブロック列の粗い優先度 (B, T, 1) を出すスタブ。
    入力: 隠れ状態 (B, S, H) をブロック境界で平均した (B, T, H) を想定するが、
    ここでは (B, S, H) をそのまま線形で圧縮する簡易版。
    """

    def __init__(self, hidden_size: int, num_blocks_hint: int = 8):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        self.num_blocks_hint = num_blocks_hint

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """(B, S, H) -> (B, S, 1) 優先度スカラー"""
        return torch.sigmoid(self.proj(hidden_states))


class BlockRouterStub(nn.Module):
    """
    優先度に基づき「詳細展開するか」のマスク (B, S) bool。
    tau 未満の位置はスキップ候補（スタブでは論理のみ。実モデルでは下層呼び出しを抑制）。

    ``step_stride`` を指定すると優先度は使わず、デコードステップ番号が ``stride`` の倍数のときだけ
    keep=True（**決定的**。Phase 3 P3 の「一段拡張」用）。
    """

    def __init__(self, tau: float = 0.5, step_stride: int | None = None):
        super().__init__()
        self.tau = tau
        self.step_stride = step_stride
        self._step = 0

    def reset(self) -> None:
        """デコードチェーンの先頭で呼ぶ（warmup／本計測の各チェーンごと）。"""
        self._step = 0

    def forward(self, priority: torch.Tensor) -> torch.Tensor:
        """
        Args:
            priority: (B, S, 1) または (B, S)
        Returns:
            keep: (B, S) bool — True の位置を「展開する」
        """
        if priority.dim() == 3:
            p = priority.squeeze(-1)
        else:
            p = priority
        if self.step_stride is not None:
            keep_val = (self._step % self.step_stride) == 0
            self._step += 1
            return torch.full_like(p, keep_val, dtype=torch.bool)
        return p >= self.tau


def router_keep_fraction(keep: torch.Tensor) -> float:
    """keep (B, S) bool の True 割合。"""
    return float(keep.float().mean().item())
