"""
推論用デバイス選択と計測同期（CUDA / Apple MPS / CPU）。

Mac（Apple Silicon）では ``torch.backends.mps.is_available()`` が True のとき
``mps`` を選ぶ。レイテンシ計測では非同期を避けるため、ステップ前後で
``sync_inference_device`` を呼ぶ。
"""

from __future__ import annotations

import torch


def select_inference_device(*, force_cpu: bool = False) -> torch.device:
    """
    推論デバイスを返す。優先順: CUDA → MPS → CPU。

    Args:
        force_cpu: True のとき常に CPU。
    """
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_inference_device(device: torch.device) -> None:
    """カーネル完了を待つ（計測の壁時計を揃える）。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory_stats_if_cuda(device: torch.device) -> None:
    """CUDA のピークメモリ統計をリセット。MPS には同等 API がないため no-op。"""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def max_memory_allocated_bytes(device: torch.device) -> int | None:
    """CUDA の max_memory_allocated。MPS では未対応のため None。"""
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None
