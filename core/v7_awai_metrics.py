"""v7 §4 周辺の「あわい」関連メトリクス（合成・実装用）。

理論上の A_ij は Ω 三因子の積として与えられる（v7 定義4.1・§4.3）。ここでは
§4.3 の**簡略版** Ω を、時系列ベクトル w_ij, w_ji（遅延）に対して計算する。

**主張の境界**: 本モジュールは **合成軌跡や実装パイプライン**向けである。
人手の「間合い」評価との対応は **Phase III-A**（別途アノテ）で扱う。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def omega_awai_numpy(
    w_ij: np.ndarray,
    w_ji_delayed: np.ndarray,
    dw_ij: np.ndarray,
    dw_ji: np.ndarray,
) -> np.ndarray:
    """NumPy: 各時刻で Ω を計算（v7 §4.3 の簡略版）。

    Args:
        w_ij: (T, d)
        w_ji_delayed: (T, d)
        dw_ij: (T, d) — ẇ_ij（先頭行は 0 パディングでも可）
        dw_ji: (T, d) — 簡略 Ω では未使用（実験 CLI との引数互換のため保持）

    Returns:
        (T,) の Ω 値。
    """
    _ = dw_ji
    v1 = np.linalg.norm(dw_ij, axis=-1)
    num = np.sum(w_ij * w_ji_delayed, axis=-1)
    den = np.linalg.norm(w_ij, axis=-1) * np.linalg.norm(w_ji_delayed, axis=-1) + 1e-9
    cos_t = np.clip(num / den, -1.0, 1.0)
    v2 = 1.0 - cos_t
    v3 = _sigmoid_np(np.linalg.norm(w_ij - w_ji_delayed, axis=-1))
    return v1 * v2 * v3


def delay_series_torch(w: torch.Tensor, tau: int, *, pad_value: float = 0.0) -> torch.Tensor:
    """時系列 ``w`` の遅延コピー: ``out[..., t, :] = w[..., t - tau, :]``（``t < tau`` はパディング）。

    形状: ``(..., T, d)`` または ``(T, d)``。
    """
    if tau == 0:
        return w.clone()
    if w.dim() < 2:
        raise ValueError("w must be (T, d) or (..., T, d)")
    *lead, t_len, _d = w.shape
    out = torch.full(
        (*lead, t_len, _d),
        pad_value,
        dtype=w.dtype,
        device=w.device,
    )
    if t_len > tau:
        out[..., tau:, :] = w[..., :-tau, :].clone()
    return out


def backward_diff_torch(w: torch.Tensor) -> torch.Tensor:
    """先頭差分 0: ``out[..., 0, :] = 0``, ``out[..., t, :] = w[..., t, :] - w[..., t-1, :]``。"""
    if w.dim() < 2:
        raise ValueError("w must be (T, d) or (..., T, d)")
    if w.shape[-2] == 0:
        return w
    first = w[..., :1, :]
    w_prev = torch.cat([first, w[..., :-1, :]], dim=-2)
    return w - w_prev


def omega_awai_series_from_w(
    w_ij: torch.Tensor,
    w_ji: torch.Tensor,
    *,
    delay_tau: int = 0,
    eps: float = 1e-9,
) -> torch.Tensor:
    """``(T, d)`` または ``(B, T, d)`` の有向系列から Ω 時系列を返す。

    ``w_ji`` を ``delay_tau`` だけ遅らせたものを ``w_ij``・``ẇ_ij`` と組み合わせる
    （合成デモと同型の遅延扱い）。
    """
    w_ji_d = delay_series_torch(w_ji, delay_tau)
    dw_ij = backward_diff_torch(w_ij)
    dw_ji = backward_diff_torch(w_ji_d)
    return omega_awai_torch(w_ij, w_ji_d, dw_ij, dw_ji, eps=eps)


def omega_awai_torch(
    w_ij: torch.Tensor,
    w_ji_delayed: torch.Tensor,
    dw_ij: torch.Tensor,
    dw_ji: torch.Tensor,
    *,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Torch: ``omega_awai_numpy`` と同型。下流で勾配を通す可能性に備え sigmoid は torch 版。"""
    _ = dw_ji
    v1 = torch.linalg.norm(dw_ij, dim=-1)
    num = (w_ij * w_ji_delayed).sum(dim=-1)
    n_ij = torch.linalg.norm(w_ij, dim=-1)
    n_ji = torch.linalg.norm(w_ji_delayed, dim=-1)
    den = n_ij * n_ji + eps
    cos_t = (num / den).clamp(-1.0, 1.0)
    v2 = 1.0 - cos_t
    diff = torch.linalg.norm(w_ij - w_ji_delayed, dim=-1)
    v3 = torch.sigmoid(diff.clamp(-30.0, 30.0))
    return v1 * v2 * v3


def run_synthetic_demo(*, seed: int, T: int, d: int) -> dict[str, Any]:
    """滑らかな疑似軌跡で Ω の分布サマリを返す（人手ラベルなし）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float64)
    w_ij = np.stack([np.sin(0.1 * t + rng.uniform(0, 2)) for _ in range(d)], axis=-1)
    w_ji_del = np.stack([np.cos(0.12 * t + rng.uniform(0, 2)) for _ in range(d)], axis=-1)
    dw_ij = np.diff(np.vstack([w_ij[:1], w_ij]), axis=0)
    dw_ji = np.diff(np.vstack([w_ji_del[:1], w_ji_del]), axis=0)
    o = omega_awai_numpy(w_ij, w_ji_del, dw_ij, dw_ji)
    return {
        "schema_version": "v7_phase3a.v1",
        "mode": "synthetic_trajectory",
        "T": T,
        "d": d,
        "seed": seed,
        "awai_mean": float(o.mean()),
        "awai_std": float(o.std()),
        "awai_p95": float(np.percentile(o, 95)),
        "note": (
            "人手の間合い評価との相関は別途 Phase III-A。合成 Ω のみ — 実証未実施でも実装・CI から利用可。"
        ),
    }
