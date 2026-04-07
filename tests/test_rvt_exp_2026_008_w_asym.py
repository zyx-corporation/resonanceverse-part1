"""RVT W_asym ヘルパ（合成注意）。"""

from __future__ import annotations

import numpy as np

from experiments.rvt_exp_2026_008_w_asym import (
    default_uniform_head_axis_matrix,
    per_head_block_asym_frobenius,
    project_head_frobenius_to_six_axes,
)


def test_per_head_block_asym_matches_scalar_mean_head():
    rng = np.random.default_rng(0)
    h0, l = 4, 8
    # 行ソフトマックス
    logits = rng.standard_normal((h0, l, l))
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(np.clip(x, -50, 50))
    a = e / e.sum(axis=-1, keepdims=True)
    mean_a = a.mean(axis=0)
    ia, ib = [0, 1], [2, 3]
    f = per_head_block_asym_frobenius(a, ia, ib)
    assert f.shape == (h0,)
    from experiments.v7_phase2a_empirical import pair_block_asymmetry_frobenius

    assert np.allclose(
        pair_block_asymmetry_frobenius(mean_a, ia, ib),
        per_head_block_asym_frobenius(
            np.broadcast_to(mean_a, (1, l, l)), ia, ib
        )[0],
    )


def test_project_head_frobenius_to_six_axes_shape():
    h = 3
    f = np.array([1.0, 2.0, 3.0])
    m = default_uniform_head_axis_matrix(h, rng=np.random.default_rng(1))
    w = project_head_frobenius_to_six_axes(f, m)
    assert w.shape == (6,)
