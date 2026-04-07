"""話者ペア (T,d) 系列 → Ω（実証不要経路）。"""

from __future__ import annotations

import pytest
import torch

from core.v7_awai_metrics import backward_diff_torch, delay_series_torch, omega_awai_series_from_w
from core.v7_pair_w_trajectory import (
    PairRelationLinear,
    directed_pair_w_ij_w_ji,
    pool_hidden_mean,
    series_from_turn_hiddens,
)


def test_pool_hidden_mean_empty_and_nonempty():
    h = torch.randn(8, 4)
    z = pool_hidden_mean(h, [])
    assert z.shape == (4,) and torch.allclose(z, torch.zeros(4))
    m = pool_hidden_mean(h, [1, 3, 3])
    exp = h[[1, 3, 3]].mean(dim=0)
    assert torch.allclose(m, exp)


def test_directed_pair_w_ij_w_ji():
    a = torch.ones(3)
    b = torch.zeros(3)
    w_ij, w_ji = directed_pair_w_ij_w_ji(a, b, d=4)
    assert w_ij.shape == (4,) and w_ji.shape == (4,)
    assert torch.allclose(w_ij[:3], a) and torch.allclose(w_ij[3], b[0])
    assert torch.allclose(w_ji[:3], b) and torch.allclose(w_ji[3], a[0])


def test_series_from_turn_hiddens_matches_manual_omega():
    torch.manual_seed(0)
    turns = [torch.randn(5, 2), torch.randn(5, 2), torch.randn(5, 2)]
    iu = [[0, 1], [0, 1], [0, 1]]
    ir = [[3, 4], [3, 4], [3, 4]]
    d = 4
    w_ij, w_ji, omega = series_from_turn_hiddens(turns, iu, ir, d=d, delay_tau=1)
    assert w_ij.shape == (3, d) and w_ji.shape == (3, d) and omega.shape == (3,)
    exp = omega_awai_series_from_w(w_ij, w_ji, delay_tau=1)
    assert torch.allclose(omega, exp)


def test_pair_relation_linear_matches_concat_when_identity():
    """``Linear`` を単位行列にすれば ``2H==d`` のとき concat_truncate と一致。"""
    h = 2
    d = 4
    lin = PairRelationLinear(h, d, bias=False)
    with torch.no_grad():
        lin.proj.weight.copy_(torch.eye(d))
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    w_ij_l, w_ji_l = lin.directed_ij_ji(a, b)
    w_ij_c, w_ji_c = directed_pair_w_ij_w_ji(a, b, d)
    assert torch.allclose(w_ij_l, w_ij_c)
    assert torch.allclose(w_ji_l, w_ji_c)


def test_series_from_turn_hiddens_with_pair_projector_grad():
    proj = PairRelationLinear(2, 4)
    turns = [torch.randn(5, 2, requires_grad=True)]
    w_ij, w_ji, omega = series_from_turn_hiddens(
        turns, [[0, 1]], [[3, 4]], d=4, pair_projector=proj
    )
    assert w_ij.shape == (1, 4)
    loss = omega.sum()
    loss.backward()
    assert proj.proj.weight.grad is not None


def test_directed_pair_d_mismatch_raises():
    proj = PairRelationLinear(3, 5)
    with pytest.raises(ValueError, match="pair_projector.d"):
        directed_pair_w_ij_w_ji(torch.zeros(3), torch.zeros(3), d=4, pair_projector=proj)


def test_delay_and_backward_diff_shapes():
    w = torch.randn(2, 4, 6)
    d0 = delay_series_torch(w, 0)
    assert torch.allclose(d0, w)
    d1 = delay_series_torch(w, 1)
    assert torch.allclose(d1[:, 0, :], torch.zeros(2, 6))
    assert torch.allclose(d1[:, 1, :], w[:, 0, :])
    bd = backward_diff_torch(w)
    assert bd.shape == w.shape
    assert torch.allclose(bd[:, 0, :], torch.zeros_like(bd[:, 0, :]))
