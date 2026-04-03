"""
日本語ローカル SLM プラン Phase 0: PyTorch / MPS / transformers の可用性を JSON で報告する。

例::

    python experiments/v7_local_env_check.py
"""

from __future__ import annotations

import json
import sys
from typing import Any


def main() -> None:
    out: dict[str, Any] = {
        "schema_version": "v7_local_env_check.v1",
        "python": sys.version.split()[0],
    }
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_available"] = bool(torch.cuda.is_available())
        mps = bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        out["mps_available"] = mps
        try:
            out["mps_built"] = bool(torch.backends.mps.is_built())
        except (AttributeError, RuntimeError):
            out["mps_built"] = None
    except ImportError as e:
        out["torch"] = None
        out["torch_error"] = str(e)

    try:
        import transformers

        out["transformers"] = transformers.__version__
    except ImportError as e:
        out["transformers"] = None
        out["transformers_error"] = str(e)

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
