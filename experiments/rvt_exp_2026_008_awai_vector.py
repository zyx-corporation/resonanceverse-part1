"""
RVT-EXP-2026-008: 計画書の AwaiVector 最小スタブ（v7 の Ω とは別物）。

ターンごとの 6 軸ベクトル ``w_axes ∈ R^6`` を列方向に積む simple バッファ。
"""

from __future__ import annotations

from typing import Any

import numpy as np


class RvtExp008AwaiVector:
    """``append`` で (6,) を貯め、``stack`` で (T, 6) を返す。"""

    __slots__ = ("_rows",)

    def __init__(self) -> None:
        self._rows: list[np.ndarray] = []

    def append(self, w_axes: np.ndarray) -> None:
        v = np.asarray(w_axes, dtype=np.float64).reshape(-1)
        if v.shape[0] != 6:
            raise ValueError(f"w_axes must have length 6, got {v.shape}")
        self._rows.append(v.copy())

    def stack(self) -> np.ndarray:
        if not self._rows:
            return np.zeros((0, 6), dtype=np.float64)
        return np.stack(self._rows, axis=0)

    def to_dict_log(self) -> dict[str, Any]:
        w = self.stack()
        return {
            "schema_version": "rvt_exp_008_awai_vector.v1",
            "T": int(w.shape[0]),
            "w_axes": w.tolist(),
        }
