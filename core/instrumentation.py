"""前向きパスの区間計測（経過時間・CUDA 割り当て差分）。Phase A′ 計測フック。"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

import torch


class StageTimer:
    """
    `LightweightResonanceFacade.forward(..., instrument=timer)` に渡す。
    CPU では経過時間のみ。CUDA では各区間の allocated バイト差分（同期後）を記録。
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.records: list[dict[str, Any]] = []

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            mem_before = torch.cuda.memory_allocated(self.device)
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
            else:
                rec["cuda_allocated_delta_bytes"] = None
            self.records.append(rec)

    def to_jsonable(self) -> list[dict[str, Any]]:
        return list(self.records)
