"""core.inference_device: CPU 強制とデバイス文字列の一貫性。"""

from __future__ import annotations

import torch

from core.inference_device import select_inference_device, sync_inference_device


def test_select_force_cpu():
    d = select_inference_device(force_cpu=True)
    assert d.type == "cpu"


def test_sync_cpu_noop():
    sync_inference_device(torch.device("cpu"))


def test_select_returns_valid_device_string():
    d = select_inference_device(force_cpu=False)
    assert d.type in ("cpu", "cuda", "mps")
