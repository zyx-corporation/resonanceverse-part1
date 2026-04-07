"""
ローカル SLM 運用プランの「任意・実機」項目を **秘密を出さず** に JSON で確認する。

- MPS / CUDA の利用可否（torch）
- カレントボリュームの空き容量（手動判断用の目安 GB）
- HF トークン・HF_HOME の有無（値は出さない）
- MLX パッケージの有無（§13 の前置き）

CI や共有ログにそのまま貼れる想定（トークンは ``has_hf_token`` のみ）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _has_hf_token() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import HfFolder

        return bool(HfFolder.get_token())
    except Exception:
        return False


def run_checklist(*, cwd: Path | None = None) -> dict[str, object]:
    c = cwd or Path.cwd()
    free_b = shutil.disk_usage(c).free
    free_gb = round(free_b / (1024**3), 2)

    torch_cuda = False
    torch_mps = False
    torch_version: str | None = None
    try:
        import torch

        torch_version = getattr(torch, "__version__", None)
        torch_cuda = bool(torch.cuda.is_available())
        mps_mod = getattr(torch.backends, "mps", None)
        torch_mps = bool(mps_mod and torch.backends.mps.is_available())
    except Exception:
        pass

    mlx_ok = False
    try:
        import importlib.util

        mlx_ok = importlib.util.find_spec("mlx") is not None
    except Exception:
        mlx_ok = False

    return {
        "schema_version": "v7_local_optional_checklist.v1",
        "cwd": str(c.resolve()),
        "disk_free_gb_approx": free_gb,
        "hf_home_set": bool(os.environ.get("HF_HOME")),
        "has_hf_token": _has_hf_token(),
        "torch": {
            "version": torch_version,
            "cuda_available": torch_cuda,
            "mps_available": torch_mps,
        },
        "mlx_import_available": mlx_ok,
        "note_ja": (
            "空き GB は目安。"
            " gated モデル・大キャッシュは v7_local_slm_m3_japanese_plan §0 補足。"
            " MLX はリポ未同梱（§13 は別マニフェスト）。"
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="ローカル任意項目チェックリスト（JSON）")
    p.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="空き容量を見るディレクトリ（既定: カレント）",
    )
    args = p.parse_args()
    payload = run_checklist(cwd=args.cwd)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
