"""
ベースライン JSON と現在の計測結果を比較し、許容を超えた悪化があれば非ゼロ終了する。

- efficiency: efficiency_benchmark の result（系列長ごとの mean_time_s）
- instrument: instrument_smoke の stages（名前ごとの elapsed_s）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_efficiency_result(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    return data.get("result", data)


def check_efficiency_regression(
    baseline: dict[str, Any],
    current: dict[str, Any],
    max_slowdown: float,
) -> list[str]:
    """
    各 seq_len について full_quadratic / roi_tiers の mean_time_s が
    baseline * max_slowdown を超えたらエラー文字列を返す。
    """
    errors: list[str] = []
    b_rows = {r["seq_len"]: r for r in baseline.get("rows", [])}
    for row in current.get("rows", []):
        sl = row["seq_len"]
        if sl not in b_rows:
            errors.append(f"seq_len={sl}: baseline に同一系列長がありません")
            continue
        br = b_rows[sl]
        for key in ("full_quadratic", "roi_tiers"):
            bt = float(br[key]["mean_time_s"])
            ct = float(row[key]["mean_time_s"])
            limit = max(bt * max_slowdown, 1e-9)
            if ct > limit:
                errors.append(
                    f"seq_len={sl} {key}: mean_time_s={ct:.6g} > baseline*{max_slowdown} "
                    f"(baseline={bt:.6g}, limit≈{limit:.6g})"
                )
    return errors


def check_instrument_regression(
    baseline: dict[str, Any],
    current: dict[str, Any],
    max_slowdown: float,
) -> list[str]:
    """
    stages 各要素の name をキーに elapsed_s を比較。
    baseline にある名前は current にも存在すること。現在 > baseline * max_slowdown で失敗。
    """
    errors: list[str] = []
    b_map = {s["name"]: float(s["elapsed_s"]) for s in baseline.get("stages", [])}
    c_map = {s["name"]: float(s["elapsed_s"]) for s in current.get("stages", [])}

    for name, bt in b_map.items():
        if name not in c_map:
            errors.append(f"stage {name!r}: current に存在しません")
            continue
        ct = c_map[name]
        limit = max(bt * max_slowdown, 1e-12)
        if ct > limit:
            errors.append(
                f"stage {name!r}: elapsed_s={ct:.6g} > baseline*{max_slowdown} "
                f"(baseline={bt:.6g}, limit≈{limit:.6g})"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare benchmark JSON to baseline (efficiency or instrument)",
    )
    parser.add_argument(
        "--mode",
        choices=("efficiency", "instrument"),
        default="efficiency",
        help="efficiency: efficiency_benchmark の result。instrument: instrument_smoke の stages。",
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument(
        "--max-slowdown",
        type=float,
        default=25.0,
        help="現在の時間がベースラインの何倍まで許容するか（悪化検出）。",
    )
    args = parser.parse_args()

    if args.mode == "efficiency":
        baseline = _load_efficiency_result(args.baseline)
        current = _load_efficiency_result(args.current)
        errs = check_efficiency_regression(baseline, current, args.max_slowdown)
    else:
        baseline = _load_json(args.baseline)
        current = _load_json(args.current)
        errs = check_instrument_regression(baseline, current, args.max_slowdown)

    if errs:
        print("REGRESSION_CHECK_FAILED", file=sys.stderr)
        for e in errs:
            print(e, file=sys.stderr)
        raise SystemExit(1)
    print("regression_ok")


if __name__ == "__main__":
    main()
