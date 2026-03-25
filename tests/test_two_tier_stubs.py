import torch

from core.two_tier import (
    BlockRouterStub,
    IdentityGPT2Block,
    SequenceControllerStub,
    check_quality_tau,
    router_keep_fraction,
)


def test_controller_router_forward():
    b, s, h = 2, 4, 16
    x = torch.randn(b, s, h)
    ctrl = SequenceControllerStub(h)
    router = BlockRouterStub(tau=0.5)
    pr = ctrl(x)
    keep = router(pr)
    assert keep.shape == (b, s)
    assert 0.0 <= router_keep_fraction(keep) <= 1.0


def test_identity_gpt2_block():
    b = IdentityGPT2Block()
    x = torch.randn(1, 4, 8)
    y = b(x)[0]
    assert y.shape == x.shape and torch.allclose(y, x)


def test_block_router_stride_deterministic():
    r = BlockRouterStub(tau=0.5, step_stride=2)
    p = torch.randn(1, 1, 1)
    r.reset()
    assert r(p).all()
    assert not r(p).any()
    assert r(p).all()
    r.reset()
    assert r(p).all()


def test_quality_tau():
    assert check_quality_tau(0.9, 0.8, higher_is_better=True).ok
    assert not check_quality_tau(0.7, 0.8, higher_is_better=True).ok
    assert check_quality_tau(0.3, 0.5, higher_is_better=False).ok
