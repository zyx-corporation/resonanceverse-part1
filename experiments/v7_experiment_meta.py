"""v7 実証実行用: ランタイムと再現メタデータ（設計書 § ソフトウェア版）。"""

from __future__ import annotations

import platform
import sys
from typing import Any


def collect_runtime_meta() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch

        meta["torch"] = torch.__version__
    except ImportError:
        meta["torch"] = None
    try:
        import transformers

        meta["transformers"] = transformers.__version__
    except ImportError:
        meta["transformers"] = None
    return meta
