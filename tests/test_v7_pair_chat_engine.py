"""tools.v7_pair_chat_engine の軽量テスト（HF なし）。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.v7_pair_chat_engine import (
    DEFAULT_RVT_SLM_INSTRUCT,
    inference_dtype_for_device,
    mrmp_window_text,
)


def test_mrmp_window_text():
    w = mrmp_window_text([("A", "hi"), ("B", "there")])
    assert w == "A: hi\nB: there"


def test_default_rvt_slm_instruct_is_qwen_3b():
    assert DEFAULT_RVT_SLM_INSTRUCT == "Qwen/Qwen2.5-3B-Instruct"


def test_inference_dtype_for_device_cpu_is_float32():
    assert inference_dtype_for_device(torch.device("cpu")) == torch.float32


def test_inference_dtype_for_device_mps_is_float32():
    assert inference_dtype_for_device(torch.device("mps")) == torch.float32
