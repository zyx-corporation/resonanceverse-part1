"""
RVT-EXP-2026-008: ヘッド Frobenius ベクトル → 6 軸への線形結合行列 M（計画書準拠）。

- **推論**: NumPy ``(H, 6)`` を ``rvt_payload_from_mrmp_window_row`` 等に渡す。
- **学習**: ``rvt_exp_2026_008_train_head_axis_m.py``
  （合成デモ／審判ラベル付き JSONL の教師あり）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.rvt_exp_2026_008_judge_axis_mapping import (  # noqa: E402
    SUPERVISED_TARGET_KEYS_AB,
)
from experiments.rvt_exp_2026_008_w_asym import (  # noqa: E402
    default_uniform_head_axis_matrix,
    project_head_frobenius_to_six_axes,
)


def load_head_axis_matrix(path: Path, *, n_heads: int) -> np.ndarray:
    """
    ``.npy`` または JSON（``[[...], ...]``）から ``(H, l)`` を読み込み、
    ``n_heads`` と行数が一致することを検証する。
    """
    path = path.resolve()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        m = np.load(path).astype(np.float64)
    elif suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        m = np.asarray(raw, dtype=np.float64)
    else:
        raise ValueError(
            f"unsupported matrix format (use .npy or .json): {path}"
        )
    if m.ndim != 2 or m.shape[1] != 6:
        raise ValueError(f"M must be (H, 6), got {m.shape}")
    if m.shape[0] != int(n_heads):
        raise ValueError(
            f"M rows {m.shape[0]} != model n_heads {n_heads} ({path})"
        )
    return m


def save_head_axis_matrix_npy(path: Path, m: np.ndarray) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(m, dtype=np.float64))


def train_head_axis_m_synthetic(
    *,
    n_heads: int,
    n_samples: int,
    steps: int,
    seed: int,
    lr: float = 0.08,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    真の M_true を乱数で生成し、``w = f_h @ M_true``（行方向 ``(f_h * M).sum(0)`` と同等）
    から **Adam** で M を回復する合成実験。

    審判ラベルなしの**スモーク**用。本番は ``--pairs-jsonl`` 等で教師ありに拡張する。
    """
    import torch
    import torch.nn as nn

    rng = np.random.default_rng(seed)
    m_true = default_uniform_head_axis_matrix(n_heads, rng=rng)
    f_list: list[np.ndarray] = []
    w_list: list[np.ndarray] = []
    for _ in range(n_samples):
        f_h = rng.random(n_heads).astype(np.float64) + 0.05
        w_list.append(project_head_frobenius_to_six_axes(f_h, m_true))
        f_list.append(f_h)
    f_t = torch.tensor(np.stack(f_list), dtype=torch.float32)
    w_t = torch.tensor(np.stack(w_list), dtype=torch.float32)
    m_par = nn.Parameter(torch.randn(n_heads, 6, dtype=torch.float32) * 0.05)
    opt = torch.optim.Adam([m_par], lr=lr)
    last_loss = 0.0
    for _ in range(steps):
        pred = torch.matmul(f_t, m_par)
        loss = nn.functional.mse_loss(pred, w_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
    learned = m_par.detach().numpy().astype(np.float64)
    rel = float(
        np.linalg.norm(learned - m_true)
        / (np.linalg.norm(m_true) + 1e-12)
    )
    meta = {
        "schema_version": "rvt_exp_008_train_m_synthetic.v1",
        "n_heads": n_heads,
        "n_samples": n_samples,
        "steps": steps,
        "seed": seed,
        "final_mse": last_loss,
        "rel_fro_error_vs_synthetic_truth": rel,
    }
    return learned, meta


def row_targets_six_ab(row: dict[str, Any]) -> np.ndarray | None:
    """審判 6 軸（*_ab のみ）を 6 ベクトルに。欠落時は None。"""
    out: list[float] = []
    for k in SUPERVISED_TARGET_KEYS_AB:
        v = row.get(k)
        if not isinstance(v, (int, float)):
            return None
        out.append(float(v))
    return np.asarray(out, dtype=np.float64)


def train_head_axis_m_supervised_jsonl(
    path: Path,
    *,
    derive_target_ab: bool,
    steps: int,
    lr: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    JSONL: 各行に ``per_head_block_frobenius``（RVT mrmp 出力由来）と
    ``w_target``（長さ 6）**または**審判 ``*_ab`` 6 キー（``derive_target_ab``）。
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(int(seed))

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            fh = row.get("per_head_block_frobenius")
            if fh is None:
                continue
            f_h = np.asarray(fh, dtype=np.float64)
            if derive_target_ab:
                w_t = row_targets_six_ab(row)
            else:
                wta = row.get("w_target")
                if wta is None:
                    continue
                w_t = np.asarray(wta, dtype=np.float64)
            if w_t is None or w_t.shape != (6,):
                continue
            pairs.append((f_h, w_t))

    if len(pairs) < 2:
        raise ValueError("supervised: need >= 2 valid jsonl rows")
    n_heads = int(pairs[0][0].shape[0])
    for f_h, _w in pairs:
        if f_h.shape[0] != n_heads:
            raise ValueError(
                "supervised: inconsistent per_head_block_frobenius length",
            )

    f_stack = np.stack([p[0] for p in pairs])
    w_stack = np.stack([p[1] for p in pairs])
    f_t = torch.tensor(f_stack, dtype=torch.float32)
    w_tn = torch.tensor(w_stack, dtype=torch.float32)
    m_par = nn.Parameter(torch.randn(n_heads, 6, dtype=torch.float32) * 0.05)
    opt = torch.optim.Adam([m_par], lr=lr)
    last_loss = 0.0
    for _ in range(steps):
        pred = torch.matmul(f_t, m_par)
        loss = nn.functional.mse_loss(pred, w_tn)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
    learned = m_par.detach().numpy().astype(np.float64)
    meta = {
        "schema_version": "rvt_exp_008_train_m_supervised.v1",
        "n_heads": n_heads,
        "n_samples": len(pairs),
        "steps": steps,
        "seed": seed,
        "derive_target_ab": derive_target_ab,
        "final_mse": last_loss,
    }
    return learned, meta
