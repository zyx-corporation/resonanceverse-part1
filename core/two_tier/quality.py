"""Phase 3: 品質 τ の検査フック（スタブ／閾値比較）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QualityTauResult:
    ok: bool
    metric_name: str
    value: float
    tau: float
    mode: str  # "lte" | "gte"


def check_quality_tau(
    value: float,
    tau: float,
    *,
    higher_is_better: bool = True,
    metric_name: str = "metric",
) -> QualityTauResult:
    """
    品質が τ を満たすか（単一スカラ）。

    - higher_is_better=True: value >= tau で合格
    - higher_is_better=False: value <= tau で合格（例: loss）
    """
    if higher_is_better:
        ok = value >= tau
        mode = "gte"
    else:
        ok = value <= tau
        mode = "lte"
    return QualityTauResult(ok=ok, metric_name=metric_name, value=value, tau=tau, mode=mode)


def quality_report_jsonable(results: list[QualityTauResult]) -> list[dict[str, Any]]:
    return [
        {
            "metric": r.metric_name,
            "value": r.value,
            "tau": r.tau,
            "ok": r.ok,
            "mode": r.mode,
        }
        for r in results
    ]
