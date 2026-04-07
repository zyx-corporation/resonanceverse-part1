"""core.v7_awai_metrics — 合成 Ω（実証 Phase III-A なしでも利用可能な経路）。"""

from __future__ import annotations

import numpy as np
import torch

from core.v7_awai_metrics import omega_awai_numpy, omega_awai_torch, run_synthetic_demo


def test_omega_awai_numpy_torch_match():
    rng = np.random.default_rng(0)
    T, d = 5, 4
    w_ij = rng.standard_normal((T, d)).astype(np.float64)
    w_ji = rng.standard_normal((T, d)).astype(np.float64)
    dw_ij = np.diff(np.vstack([w_ij[:1], w_ij]), axis=0)
    dw_ji = np.diff(np.vstack([w_ji[:1], w_ji]), axis=0)
    on = omega_awai_numpy(w_ij, w_ji, dw_ij, dw_ji)
    tt = omega_awai_torch(
        torch.tensor(w_ij),
        torch.tensor(w_ji),
        torch.tensor(dw_ij),
        torch.tensor(dw_ji),
    )
    np.testing.assert_allclose(on, tt.detach().numpy(), rtol=1e-5, atol=1e-6)


def test_run_synthetic_demo_deterministic():
    a = run_synthetic_demo(seed=7, T=30, d=6)
    b = run_synthetic_demo(seed=7, T=30, d=6)
    assert a["awai_mean"] == b["awai_mean"]
    assert a["schema_version"] == "v7_phase3a.v1"
    assert "note" in a
