"""RVT L2 注意ブレンド（tensor のみ）。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # noqa: F401

from experiments.rvt_exp_2026_008_attn_inject import (  # noqa: E402
    _blend_post_softmax,
    gpt2_rvt_inject_session,
    gpt2_rvt_inject_session_depth,
    hf_forward_attention_layer_heads_with_rvt_l2,
    hf_forward_attention_with_rvt_l2,
    model_supports_gpt2_rvt_inject,
    model_supports_rvt_l2_inject,
)


def test_blend_sym_row_stochastic_and_finite():
    # 対称な logits → softmax も対称 → sym ブレンド後も（再正規化前は）対称
    g = torch.randn(5, 5)
    logits = g + g.T
    p = torch.softmax(logits, dim=-1).unsqueeze(0).unsqueeze(0)
    q = _blend_post_softmax(p.clone(), mode="sym", eps=0.0)
    sumq = q.sum(dim=-1)
    assert torch.allclose(sumq, torch.ones_like(sumq), atol=1e-5)
    assert torch.isfinite(q).all()


def test_blend_wasym_still_row_stochastic():
    p = torch.softmax(torch.randn(1, 2, 4, 4), dim=-1)
    q = _blend_post_softmax(p.clone(), mode="wasym", eps=0.1)
    sumw = q.sum(dim=-1)
    assert torch.allclose(sumw, torch.ones_like(sumw), atol=1e-4)


def test_blend_base_identity():
    p = torch.softmax(torch.randn(2, 2, 3, 3), dim=-1)
    q = _blend_post_softmax(p, mode="base", eps=0.0)
    assert torch.allclose(p, q)


def test_model_supports_gpt2_lm_head():
    pytest.importorskip("transformers")
    from transformers import GPT2Config
    from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

    cfg = GPT2Config(
        n_layer=1,
        n_head=2,
        n_embd=32,
        vocab_size=100,
    )
    cfg.attn_implementation = "eager"
    tiny = GPT2LMHeadModel(cfg)
    assert model_supports_gpt2_rvt_inject(tiny) is True
    assert model_supports_gpt2_rvt_inject(MagicMock()) is False
    assert model_supports_rvt_l2_inject(tiny) is True


def test_model_supports_rvt_l2_rejects_non_eager():
    pytest.importorskip("transformers")
    from transformers import GPT2Config
    from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

    cfg = GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=50)
    cfg.attn_implementation = "sdpa"
    m = GPT2LMHeadModel(cfg)
    assert model_supports_rvt_l2_inject(m) is False


@patch(
    "experiments.rvt_exp_2026_008_attention."
    "hf_forward_attention_layer_heads_numpy",
    autospec=True,
)
def test_hf_forward_heads_with_rvt_base_delegates(mock_heads):
    mock_heads.return_value = (np.zeros((2, 3, 3), dtype=np.float64), None, 3)
    a, err, n = hf_forward_attention_layer_heads_with_rvt_l2(
        model=object(),
        tokenizer=object(),
        device=None,
        text="hello",
        layer_index=0,
        mode="base",
    )
    mock_heads.assert_called_once()
    assert err is None
    assert n == 3
    assert a is not None and a.shape == (2, 3, 3)


@patch(
    "experiments.v7_phase1a_phi_correlation.hf_forward_attention_layer_matrix",
    autospec=True,
)
def test_hf_forward_with_rvt_base_delegates(mock_phi):
    mock_phi.return_value = (np.zeros((2, 2), dtype=np.float64), None, 4)
    a, err, n = hf_forward_attention_with_rvt_l2(
        model=object(),
        tokenizer=object(),
        device=None,
        text="hello",
        layer_index=-1,
        mode="base",
    )
    mock_phi.assert_called_once()
    assert err is None
    assert n == 4
    assert a.shape == (2, 2)


def test_gpt2_rvt_inject_session_nested_releases_once():
    pytest.importorskip("transformers")
    import transformers.models.gpt2.modeling_gpt2 as gpt2_mod

    orig = gpt2_mod.eager_attention_forward
    assert gpt2_rvt_inject_session_depth() == 0
    with gpt2_rvt_inject_session():
        assert gpt2_mod.eager_attention_forward is not orig
        assert gpt2_rvt_inject_session_depth() == 1
        with gpt2_rvt_inject_session():
            assert gpt2_mod.eager_attention_forward is not orig
            assert gpt2_rvt_inject_session_depth() == 2
        assert gpt2_mod.eager_attention_forward is not orig
        assert gpt2_rvt_inject_session_depth() == 1
    assert gpt2_mod.eager_attention_forward is orig
    assert gpt2_rvt_inject_session_depth() == 0


def test_gpt2_rvt_inject_session_nested_inner_exception_restores():
    pytest.importorskip("transformers")
    import transformers.models.gpt2.modeling_gpt2 as gpt2_mod

    orig = gpt2_mod.eager_attention_forward
    with pytest.raises(RuntimeError, match="inner"):
        with gpt2_rvt_inject_session():
            with gpt2_rvt_inject_session():
                raise RuntimeError("inner")
    assert gpt2_mod.eager_attention_forward is orig
    assert gpt2_rvt_inject_session_depth() == 0


def test_gpt2_rvt_inject_session_outer_exception_restores():
    pytest.importorskip("transformers")
    import transformers.models.gpt2.modeling_gpt2 as gpt2_mod

    orig = gpt2_mod.eager_attention_forward
    with pytest.raises(RuntimeError, match="outer"):
        with gpt2_rvt_inject_session():
            raise RuntimeError("outer")
    assert gpt2_mod.eager_attention_forward is orig
    assert gpt2_rvt_inject_session_depth() == 0
