"""学習可能 M（合成回復）・ロード。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.rvt_exp_2026_008_head_axis_matrix import (
    load_head_axis_matrix,
    row_targets_six_ab,
    save_head_axis_matrix_npy,
    train_head_axis_m_supervised_jsonl,
    train_head_axis_m_synthetic,
)


def test_load_head_axis_matrix_json(tmp_path: Path) -> None:
    m = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]
    p = tmp_path / "m.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    got = load_head_axis_matrix(p, n_heads=2)
    assert got.shape == (2, 6)


def test_load_head_axis_matrix_npy(tmp_path: Path) -> None:
    m = np.random.default_rng(0).random((4, 6))
    p = tmp_path / "x.npy"
    save_head_axis_matrix_npy(p, m)
    got = load_head_axis_matrix(p, n_heads=4)
    assert got.shape == (4, 6)
    np.testing.assert_allclose(got, m)


def test_load_wrong_rows_raises(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    p.write_text(json.dumps([[0.1] * 6] * 3), encoding="utf-8")
    with pytest.raises(ValueError, match="M rows"):
        load_head_axis_matrix(p, n_heads=2)


def test_row_targets_six_ab_partial_returns_none():
    assert row_targets_six_ab({"trust_ab": 0.5}) is None


def test_train_supervised_jsonl(tmp_path):
    pytest.importorskip("torch")
    fh = [0.1, 0.2, 0.3]
    p = tmp_path / "s.jsonl"
    lines = []
    for i in range(3):
        lines.append(
            json.dumps(
                {
                    "per_head_block_frobenius": fh,
                    "w_target": [0.1 * i + 0.01 * j for j in range(6)],
                }
            )
        )
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    m, meta = train_head_axis_m_supervised_jsonl(
        p,
        derive_target_ab=False,
        steps=80,
        lr=0.1,
        seed=1,
    )
    assert m.shape == (3, 6)
    assert meta["n_samples"] == 3


def test_train_synthetic_m_recovery() -> None:
    pytest.importorskip("torch")
    learned, meta = train_head_axis_m_synthetic(
        n_heads=8,
        n_samples=400,
        steps=600,
        seed=0,
        lr=0.1,
    )
    assert learned.shape == (8, 6)
    assert meta["rel_fro_error_vs_synthetic_truth"] < 0.35
