"""``triple_run_summary.json`` から計時・ペア一致を Markdown 表にする。

例::

    python experiments/v7_slm_judge_triple_tables.py \\
      --summary experiments/logs/slm_judge_triple_mrmp_n100/triple_run_summary.json
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

from experiments.v7_phase1a_pilot_jsonl import PILOT_KEYS  # noqa: E402


def _fmt_float(x: Any, nd: int = 4) -> str:
    if x is None:
        return ""
    try:
        v = float(x)
        if v != v:
            return "nan"
        return f"{v:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def tables_from_summary(data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("## 実行条件")
    lines.append("")
    lines.append("| 項目 | 値 |")
    lines.append("| --- | --- |")
    lines.append(f"| n（審判行数） | {data.get('n')} |")
    lines.append(f"| offset（CLI） | {data.get('offset_cli')} |")
    lines.append(f"| demo | {data.get('demo')} |")
    lines.append(f"| 入力（ソース） | `{data.get('input_jsonl_source', '')}` |")
    lines.append(f"| 審判に使った JSONL | `{data.get('judge_input_jsonl', '')}` |")
    if not data.get("demo"):
        lines.append(f"| 7B-a | `{data.get('hf_7b_a', '')}` |")
        lines.append(f"| 7B-b | `{data.get('hf_7b_b', '')}` |")
        lines.append(f"| 3B | `{data.get('hf_3b', '')}` |")
    lines.append("")

    wc = data.get("wall_clock_s") or {}
    if wc:
        lines.append("## 壁時計（秒）")
        lines.append("")
        lines.append("| ステップ | 秒 |")
        lines.append("| --- | --- |")
        for k in sorted(wc.keys()):
            if k == "total_wall_s":
                continue
            lines.append(f"| {k} | {_fmt_float(wc[k], 4)} |")
        lines.append(f"| **total_wall_s** | **{_fmt_float(wc.get('total_wall_s'), 4)}** |")
        lines.append("")

    pair_block = data.get("pair_agreement") or {}
    pair_labels = {
        "7b_a_vs_7b_b": "7B-a vs 7B-b",
        "7b_a_vs_3b": "7B-a vs 3B",
        "7b_b_vs_3b": "7B-b vs 3B",
    }

    lines.append("## ペア一致（軸ごと）")
    lines.append("")
    lines.append("Pearson r と MAD（平均絶対差）。比較: 7B-a~7B-b / 7B-a~3B / 7B-b~3B。")
    lines.append("")
    lines.append(
        "| 軸 | n | r(a~b) | MAD | r(a~3) | MAD | r(b~3) | MAD |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")

    for axis in PILOT_KEYS:
        row: list[str] = [axis, ""]
        for pk in ("7b_a_vs_7b_b", "7b_a_vs_3b", "7b_b_vs_3b"):
            cell = pair_block.get(pk) or {}
            summ = cell.get("summary") or {}
            by_ax = (summ.get("by_axis") or {}).get(axis) or {}
            n = by_ax.get("n")
            if row[1] == "" and n is not None:
                row[1] = str(int(n))
            row.append(_fmt_float(by_ax.get("pearson_r")))
            row.append(_fmt_float(by_ax.get("mean_abs_diff")))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("### ペア別サマリ（件数）")
    lines.append("")
    lines.append("| 比較 | n_used | n_intersection |")
    lines.append("| --- | --- | --- |")
    for pk, title in pair_labels.items():
        cell = pair_block.get(pk) or {}
        summ = cell.get("summary") or {}
        lines.append(
            f"| {title} | {summ.get('n_rows_used', '')} | "
            f"{summ.get('n_ids_intersection', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="triple_run_summary → Markdown 表")
    p.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="triple_run_summary.json のパス",
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="省略時は標準出力のみ",
    )
    args = p.parse_args()
    path = args.summary.resolve()
    if not path.is_file():
        print(json.dumps({"error": "summary_not_found", "path": str(path)}), file=sys.stderr)
        sys.exit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "v7_slm_judge_triple_run.v1":
        print(
            json.dumps({"error": "unexpected_schema", "path": str(path)}),
            file=sys.stderr,
        )
        sys.exit(2)
    md = tables_from_summary(data)
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
    sys.stdout.write(md)


if __name__ == "__main__":
    main()
