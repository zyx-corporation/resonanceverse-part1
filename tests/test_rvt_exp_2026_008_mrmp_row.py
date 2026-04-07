"""MRMP 1 行 → RVT payload（HF 無し・モック）。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pathlib import Path

from experiments.rvt_exp_2026_008_mrmp_row import (
    _read_jsonl_line,
    iter_jsonl_physical_lines,
    num_attention_heads_for_mrmp_model,
    run_mrmp_rvt_batch,
    rvt_payload_from_mrmp_window_row,
)


def test_iter_jsonl_physical_lines(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text("a\nb\nc\n", encoding="utf-8")
    rows = list(iter_jsonl_physical_lines(p, first_line=1, n_lines=2))
    assert rows == [(1, "b\n"), (2, "c\n")]


def test_read_jsonl_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\n{"b":2}\n', encoding="utf-8")
    assert _read_jsonl_line(p, 0) == {"a": 1}
    assert _read_jsonl_line(p, 1) == {"b": 2}
    with pytest.raises(IndexError):
        _read_jsonl_line(p, 5)


@patch(
    "experiments.rvt_exp_2026_008_mrmp_row."
    "hf_forward_attention_layer_heads_numpy"
)
@patch(
    "experiments.rvt_exp_2026_008_mrmp_row."
    "speaker_token_indices_mrmp_window"
)
def test_rvt_payload_ok(mock_sp, mock_heads):
    mock_sp.return_value = ([0, 1], [2, 3])
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((8, 8))
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(np.clip(x, -50, 50))
    ah = e / e.sum(axis=-1, keepdims=True)
    heads = np.stack(
        [ah + 1e-4 * rng.standard_normal((8, 8)) for _ in range(3)],
        axis=0,
    )
    heads = np.clip(heads, 1e-8, None)
    heads = heads / heads.sum(axis=-1, keepdims=True)
    mock_heads.return_value = (heads, None, 8)
    row = {
        "id": "T_u00001",
        "dialogue_id": "T",
        "text": "a: x\nb: y",
        "speaker_tgt": "a",
        "speaker_src": "b",
    }
    out = rvt_payload_from_mrmp_window_row(
        row,
        model=MagicMock(),
        tokenizer=MagicMock(),
        device=None,
        layer_index=-1,
        seed=42,
    )
    assert out["ok"] is True
    assert out["n_heads"] == 3
    assert len(out["w_axes_proxy"]) == 6
    assert out.get("head_axis_mode") == "random_uniform_l1col"


@patch(
    "experiments.rvt_exp_2026_008_mrmp_row."
    "hf_forward_attention_layer_heads_numpy"
)
@patch(
    "experiments.rvt_exp_2026_008_mrmp_row."
    "speaker_token_indices_mrmp_window"
)
def test_rvt_payload_head_axis_matrix_shape_error(mock_sp, mock_heads):
    mock_sp.return_value = ([0, 1], [2, 3])
    heads = np.zeros((3, 4, 4), dtype=np.float64)
    for h in range(3):
        np.fill_diagonal(heads[h], 1.0 / 4.0)
    mock_heads.return_value = (heads, None, 4)
    row = {
        "id": "T_u00001",
        "dialogue_id": "T",
        "text": "a: x\nb: y",
        "speaker_tgt": "a",
        "speaker_src": "b",
    }
    bad_m = np.ones((2, 6), dtype=np.float64)
    out = rvt_payload_from_mrmp_window_row(
        row,
        model=MagicMock(),
        tokenizer=MagicMock(),
        device=None,
        layer_index=-1,
        seed=42,
        head_axis_matrix=bad_m,
    )
    assert out["ok"] is False
    assert "head_axis_matrix" in out.get("error", "")


@patch(
    "experiments.rvt_exp_2026_008_mrmp_row._require_rvt_l2_model_or_exit",
)
@patch(
    "experiments.rvt_exp_2026_008_mrmp_row.rvt_payload_from_mrmp_window_row",
)
@patch("experiments.rvt_exp_2026_008_mrmp_row._load_hf")
def test_run_mrmp_rvt_batch_accumulates_awai(
    mock_hf,
    mock_rvt,
    _req,
    tmp_path,
):
    mock_hf.return_value = (None, None, None)
    p = tmp_path / "w.jsonl"
    p.write_text(
        '{"id":"a","text":"x","speaker_tgt":"u","speaker_src":"v"}\n'
        '{"id":"b","text":"y","speaker_tgt":"u","speaker_src":"v"}\n',
        encoding="utf-8",
    )
    mock_rvt.side_effect = [
        {"ok": True, "w_axes_proxy": [0.1 * i for i in range(6)]},
        {"ok": True, "w_axes_proxy": [0.2 * i for i in range(6)]},
    ]
    payloads, awai, any_ok, any_fail = run_mrmp_rvt_batch(
        p,
        first_line=0,
        max_rows=2,
        model_name="gpt2",
        cpu=True,
        layer_index=-1,
        seed=0,
        accumulate_awai=True,
    )
    assert len(payloads) == 2
    assert any_ok and not any_fail
    assert awai is not None
    assert awai["T"] == 2


@patch(
    "experiments.rvt_exp_2026_008_mrmp_row._require_rvt_l2_model_or_exit",
)
@patch("experiments.rvt_exp_2026_008_mrmp_row._load_hf")
def test_run_mrmp_rvt_batch_empty_and_bad_json(mock_hf, _req, tmp_path):
    mock_hf.return_value = (None, None, None)
    p = tmp_path / "w.jsonl"
    p.write_text("\nnot-json\n", encoding="utf-8")
    payloads, awai, any_ok, any_fail = run_mrmp_rvt_batch(
        p,
        first_line=0,
        max_rows=2,
        model_name="gpt2",
        cpu=True,
        layer_index=-1,
        seed=0,
        accumulate_awai=False,
    )
    assert len(payloads) == 2
    assert not any_ok
    assert any_fail
    assert awai is None


@patch(
    "experiments.rvt_exp_2026_008_mrmp_row."
    "hf_forward_attention_layer_heads_with_rvt_l2",
)
@patch(
    "experiments.rvt_exp_2026_008_mrmp_row."
    "hf_forward_attention_layer_heads_numpy",
)
@patch(
    "experiments.rvt_exp_2026_008_mrmp_row."
    "speaker_token_indices_mrmp_window",
)
def test_rvt_payload_l2_flags(
    mock_sp,
    mock_heads_base,
    mock_heads_l2,
):
    mock_sp.return_value = ([0, 1], [2, 3])
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((4, 4))
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(np.clip(x, -50, 50))
    ah0 = e / e.sum(axis=-1, keepdims=True)
    heads = np.stack([ah0, ah0 * 0.98 + 0.01 / 4], axis=0)
    heads = np.clip(heads, 1e-8, None)
    heads = heads / heads.sum(axis=-1, keepdims=True)
    mock_heads_base.return_value = (heads, None, 4)
    mock_heads_l2.return_value = (heads, None, 4)
    row = {
        "id": "T_u00001",
        "dialogue_id": "T",
        "text": "a: x\nb: y",
        "speaker_tgt": "a",
        "speaker_src": "b",
    }
    out = rvt_payload_from_mrmp_window_row(
        row,
        model=MagicMock(),
        tokenizer=MagicMock(),
        device=None,
        layer_index=-1,
        seed=42,
        rvt_l2_mode="sym",
        rvt_l2_eps=0.03,
        rvt_l2_all_layers=True,
    )
    mock_heads_l2.assert_called_once()
    mock_heads_base.assert_not_called()
    assert out["ok"] is True
    assert out["rvt_l2_intervention"]["mode"] == "sym"
    assert out["rvt_l2_intervention"]["eps"] == pytest.approx(0.03)
    assert out["rvt_l2_intervention"]["all_layers"] is True


def test_num_attention_heads_gpt2_config():
    pytest.importorskip("transformers")
    from transformers import GPT2Config
    from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

    cfg = GPT2Config(n_layer=1, n_head=8, n_embd=32, vocab_size=50)
    m = GPT2LMHeadModel(cfg)
    assert num_attention_heads_for_mrmp_model(m) == 8


def test_num_attention_heads_llama_config():
    pytest.importorskip("transformers")
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=11,
        num_hidden_layers=2,
    )
    m = LlamaForCausalLM(cfg)
    assert num_attention_heads_for_mrmp_model(m) == 11


def test_num_attention_heads_unknown_config_raises():
    class EmptyCfg:
        pass

    m = MagicMock()
    m.config = EmptyCfg()
    with pytest.raises(ValueError, match="cannot resolve"):
        num_attention_heads_for_mrmp_model(m)


@patch(
    "experiments.rvt_exp_2026_008_mrmp_row."
    "speaker_token_indices_mrmp_window",
)
def test_rvt_payload_empty_blocks(mock_sp):
    mock_sp.return_value = ([], [1])
    out = rvt_payload_from_mrmp_window_row(
        {"id": "x", "text": "t", "speaker_tgt": "a", "speaker_src": "b"},
        model=MagicMock(),
        tokenizer=MagicMock(),
        device=None,
        layer_index=-1,
        seed=0,
    )
    assert out["ok"] is False
