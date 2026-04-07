"""
RVT-EXP-2026-008 Day 0 相当: MRMP 整形後の windows.jsonl
（および任意で manifest.json）の健全性検査。

v7_mrmp_prepare が書く行スキーマと整合。
CI や本番直前に行数・必須キー・manifest 件数の一致を確認する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REQUIRED_WINDOW_KEYS: tuple[str, ...] = (
    "schema_version",
    "id",
    "dialogue_id",
    "utterance_id",
    "pair_rule",
    "speaker",
    "speaker_src",
    "speaker_tgt",
    "window_turns",
    "text",
)


def validate_windows_jsonl(
    windows_path: Path,
    *,
    min_rows: int | None = None,
    expect_pair_rule: str | None = None,
    manifest_path: Path | None = None,
    strict_manifest: bool = False,
) -> dict[str, Any]:
    """
    ``windows.jsonl`` を 1 行ずつ検証し、結果 dict を返す。

    - ``ok``: エラーが無い
    - ``n_lines``: 有効 JSON 行数（空行は ``errors`` に載る）
    """
    errors: list[str] = []
    warnings: list[str] = []
    n_ok = 0
    first_bad: str | None = None

    if not windows_path.is_file():
        return {
            "schema_version": "rvt_exp_2026_008_day0.v1",
            "ok": False,
            "windows_path": str(windows_path.resolve()),
            "n_lines": 0,
            "errors": [f"windows file not found: {windows_path}"],
            "warnings": [],
        }

    with windows_path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                errors.append(f"line {lineno}: empty line")
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {lineno}: invalid json ({e})")
                if first_bad is None:
                    first_bad = f"line {lineno}"
                continue
            if not isinstance(row, dict):
                errors.append(f"line {lineno}: root must be object")
                continue
            for k in REQUIRED_WINDOW_KEYS:
                if k not in row:
                    errors.append(f"line {lineno}: missing key {k!r}")
            if expect_pair_rule is not None:
                if row.get("pair_rule") != expect_pair_rule:
                    errors.append(
                        f"line {lineno}: pair_rule expected {expect_pair_rule!r} "
                        f"got {row.get('pair_rule')!r}"
                    )
            n_ok += 1

    if min_rows is not None and n_ok < min_rows:
        errors.append(f"n_lines={n_ok} < min_rows={min_rows}")

    manifest_n: int | None = None
    if manifest_path is not None:
        if not manifest_path.is_file():
            msg = f"manifest not found: {manifest_path}"
            if strict_manifest:
                errors.append(msg)
            else:
                warnings.append(msg)
        else:
            try:
                mj = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_n = int(mj.get("n_utterance_rows", -1))
                if manifest_n >= 0 and manifest_n != n_ok:
                    errors.append(
                        f"manifest n_utterance_rows={manifest_n} "
                        f"!= windows lines={n_ok}"
                    )
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
                errors.append(f"manifest read/parse failed: {e}")

    ok = len(errors) == 0
    return {
        "schema_version": "rvt_exp_2026_008_day0.v1",
        "ok": ok,
        "windows_path": str(windows_path.resolve()),
        "manifest_path": (
            str(manifest_path.resolve()) if manifest_path else None
        ),
        "n_lines": n_ok,
        "manifest_n_utterance_rows": manifest_n,
        "min_rows": min_rows,
        "expect_pair_rule": expect_pair_rule,
        "errors": errors,
        "warnings": warnings,
        "first_json_error_line_hint": first_bad,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "RVT Day 0: MRMP windows.jsonl (+ optional manifest) sanity checks"
        ),
    )
    p.add_argument(
        "--windows",
        type=Path,
        default=(
            _ROOT
            / "experiments"
            / "logs"
            / "mrmp_prepared"
            / "windows.jsonl"
        ),
        help="Path to windows.jsonl",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest.json (default: windows parent / manifest.json)",
    )
    p.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not read manifest (ignore default manifest path)",
    )
    p.add_argument(
        "--min-rows",
        type=int,
        default=None,
        help="Fail if line count is below this",
    )
    p.add_argument(
        "--expect-pair-rule",
        type=str,
        default="P1_prev_speaker",
        help=(
            "If set (default: P1_prev_speaker), "
            "require pair_rule match per row"
        ),
    )
    p.add_argument(
        "--strict-manifest",
        action="store_true",
        help="Fail if manifest path missing when --manifest implied",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON report to this path",
    )
    args = p.parse_args()
    win = args.windows.resolve()
    man: Path | None = None
    if not args.no_manifest:
        if args.manifest:
            man = args.manifest.resolve()
        else:
            man = win.parent / "manifest.json"

    exp_rule = args.expect_pair_rule
    if exp_rule == "":
        exp_rule = None

    payload = validate_windows_jsonl(
        win,
        min_rows=args.min_rows,
        expect_pair_rule=exp_rule,
        manifest_path=man,
        strict_manifest=args.strict_manifest,
    )
    js = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js + "\n", encoding="utf-8")
    print(js)
    sys.exit(0 if payload["ok"] else 1)


if __name__ == "__main__":
    main()
