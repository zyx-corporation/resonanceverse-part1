"""
v7_phase1a_pilot_jsonl の JSON 出力から、6 軸×Frobenius の相関を Markdown 表にする。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="phi 相関 JSON → Markdown 表")
    p.add_argument("json_path", type=Path, help="v7_phase1a_pilot の出力 JSON")
    p.add_argument("--out", type=Path, default=None, help="省略時は標準出力のみ")
    args = p.parse_args()
    data = json.loads(args.json_path.read_text(encoding="utf-8"))
    cor = data.get("correlations_label_vs_fro", {})
    meta = {
        "schema_version": data.get("schema_version"),
        "n_rows": data.get("n_rows"),
        "model": data.get("model"),
        "layer_index": data.get("layer_index"),
        "feature": data.get("feature"),
    }
    lines = [
        "## v7 Phase I-A: LLM 6 軸 vs ||S_asym||_F（最終層）",
        "",
        f"- `n_rows`: {meta.get('n_rows')}",
        f"- `model`: {meta.get('model')}",
        f"- `layer_index`: {meta.get('layer_index')}",
        "",
        "| axis | n | Pearson r |",
        "|------|---|-----------|",
    ]
    for k in sorted(cor.keys()):
        v = cor[k]
        r = v.get("pearson_r")
        rs = f"{r:.6f}" if isinstance(r, (int, float)) and r == r else "—"
        lines.append(f"| {k} | {v.get('n')} | {rs} |")
    md = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
    print(md, end="")


if __name__ == "__main__":
    main()
