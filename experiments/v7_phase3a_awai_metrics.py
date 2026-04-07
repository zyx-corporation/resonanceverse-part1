"""
v7 Phase III 向け: 「あわい」測度 Ω の合成軌跡（人手アノテ無し）。

実体は ``core.v7_awai_metrics``。CLI のみ本ファイルに残す。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.v7_awai_metrics import run_synthetic_demo

# v7_run_suite / tests 向け後方互換名
run_demo = run_synthetic_demo


def main() -> None:
    p = argparse.ArgumentParser(description="v7: あわい Ω 合成メトリクス")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--d", type=int, default=6)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    payload = run_synthetic_demo(seed=args.seed, T=args.T, d=args.d)
    js = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js, encoding="utf-8")
    print("v7_phase3a_ok", json.dumps({"awai_mean": payload["awai_mean"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
