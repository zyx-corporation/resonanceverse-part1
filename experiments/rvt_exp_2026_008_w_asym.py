"""
RVT-EXP-2026-008: ヘッド別ブロック非対称スカラー → 任意行列 M で 6 軸への射影。

Phase II-A の ``pair_block_asymmetry_frobenius`` をヘッド次元で繰り返し、
計画書の ``W_asym`` の最小代理（ヘッド Frobenius ベクトルを M で線形結合）を構築する。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.v7_phase2a_empirical import pair_block_asymmetry_frobenius


def per_head_block_asym_frobenius(
    attention_heads: np.ndarray,
    idx_a: Sequence[int],
    idx_b: Sequence[int],
) -> np.ndarray:
    """
    ``attention_heads``: (H, L, L) 行ソフトマックス済み。
    戻り値: (H,) 各ヘッドの || A_h[ia,ib] - A_h[ib,ia].T ||_F。
    """
    if attention_heads.ndim != 3:
        raise ValueError("attention_heads must be (H, L, L)")
    out: list[float] = []
    for h in range(attention_heads.shape[0]):
        out.append(pair_block_asymmetry_frobenius(attention_heads[h], idx_a, idx_b))
    return np.asarray(out, dtype=np.float64)


def project_head_frobenius_to_six_axes(
    f_h: np.ndarray,
    head_to_axis: np.ndarray,
) -> np.ndarray:
    """
    ``f_h``: (H,)、``head_to_axis``: (H, 6) を M[h,k] とみなし、
    ``w[k] = sum_h f_h[h] * M[h,k]``（ヘッド強度と軸重みの双線形代理）。
    """
    if f_h.ndim != 1:
        raise ValueError("f_h must be 1D")
    m = np.asarray(head_to_axis, dtype=np.float64)
    if m.shape != (f_h.shape[0], 6):
        raise ValueError(f"head_to_axis must be (H, 6), got {m.shape}")
    return (f_h[:, None] * m).sum(axis=0)


def default_uniform_head_axis_matrix(n_heads: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """学習前の仮 M: 行方向に非負乱数を置き、列ごとに L1 正規化（列和=1）。"""
    if n_heads < 1:
        raise ValueError("n_heads must be positive")
    r = rng or np.random.default_rng(0)
    x = r.random((n_heads, 6), dtype=np.float64) + 1e-6
    return x / x.sum(axis=0, keepdims=True)
