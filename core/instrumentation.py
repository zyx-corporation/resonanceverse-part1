"""前向きパスの区間計測（経過時間・CUDA 割り当て差分・区間ピーク）。Phase A′ 計測フック。"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

import torch

from core.inference_device import (
    max_memory_allocated_bytes,
    reset_peak_memory_stats_if_cuda,
)


class StageTimer:
    """
    `LightweightResonanceFacade.forward(..., instrument=timer)` に渡す。
    CPU では経過時間のみ。CUDA では各区間の allocated 差分に加え、区間開始時に
    ``reset_peak_memory_stats`` したうえでの ``max_memory_allocated``（区間内ピーク）
    を記録する。連続区間のみ想定（ネストした ``stage`` ではピークの解釈が崩れる）。
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.records: list[dict[str, Any]] = []

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            mem_before = torch.cuda.memory_allocated(self.device)
            reset_peak_memory_stats_if_cuda(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            rec: dict[str, Any] = {"name": name, "elapsed_s": elapsed}
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                mem_after = torch.cuda.memory_allocated(self.device)
                rec["cuda_allocated_delta_bytes"] = int(mem_after - mem_before)
                rec["cuda_peak_allocated_bytes"] = max_memory_allocated_bytes(
                    self.device,
                )
            else:
                rec["cuda_allocated_delta_bytes"] = None
                rec["cuda_peak_allocated_bytes"] = None
            self.records.append(rec)

    def to_jsonable(self) -> list[dict[str, Any]]:
        return list(self.records)
