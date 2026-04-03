"""
複数の Phase II-A 出力 JSON（*_with_contrib.json または tau_summary JSON）を並べて比較表を出す。

感度分析・スイープ結果の一覧用（主解析の定義は変更しない）。

例::

    python experiments/v7_phase2a_compare_runs.py \\
      run_a.json run_b.json --labels baseline,layer8 --out-md compare.md --out-json compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _r_at(by_tau: list[dict[str, Any]], tau: int) -> dict[str, Any] | None:
    for r in by_tau:
        if int(r["tau"]) == tau:
            return r
    return None


def extract_from_empirical(data: dict[str, Any], *, path: Path, label: str) -> dict[str, Any]:
    by_tau = data.get("by_tau") or []
    if not by_tau:
        return {"label": label, "path": str(path), "error": "empty_by_tau"}
    taus = [int(r["tau"]) for r in by_tau]
    r0, r1 = _r_at(by_tau, 0), _r_at(by_tau, 1)
    r_vars = [float(r["R_var"]) for r in by_tau]
    imax = max(range(len(r_vars)), key=lambda i: r_vars[i])
    return {
        "label": label,
        "path": str(path),
        "source": "v7_phase2a_empirical",
        "model": data.get("model"),
        "layer_index": data.get("layer_index"),
        "n_dialogues": data.get("n_dialogues"),
        "inference_device": data.get("inference_device"),
        "tau_min": min(taus),
        "tau_max": max(taus),
        "n_tau": len(by_tau),
        "R_mean_tau0": float(r0["R_mean"]) if r0 else None,
        "R_mean_tau1": float(r1["R_mean"]) if r1 else None,
        "R_var_argmax_tau": int(taus[imax]),
        "R_var_argmax": float(r_vars[imax]),
        "has_auxiliary": bool(isinstance(data.get("auxiliary_label_delay_coherence"), dict)),
    }


def extract_from_summary(data: dict[str, Any], *, path: Path, label: str) -> dict[str, Any]:
    if data.get("schema_version") != "v7_phase2a_tau_summary.v1":
        return {"label": label, "path": str(path), "error": "not_tau_summary_v1"}
    s = data.get("summary") or {}
    tr = s.get("tau_range") or [None, None]
    return {
        "label": label,
        "path": str(path),
        "source": "v7_phase2a_tau_summary",
        "model": data.get("model"),
        "layer_index": data.get("layer_index"),
        "n_dialogues": data.get("n_dialogues_in_run"),
        "tau_max": tr[1] if len(tr) > 1 else None,
        "tau_range": s.get("tau_range"),
        "n_tau": s.get("n_tau_points"),
        "n_tau_points": s.get("n_tau_points"),
        "R_mean_argmax_tau": s.get("R_mean_argmax_tau"),
        "R_mean_tau0": None,
        "R_mean_tau1": None,
        "R_var_argmax_tau": None,
        "R_var_global_max_tau": s.get("R_var_global_max_tau"),
        "R_var_smoothed_argmax_tau": s.get("R_var_smoothed_argmax_tau"),
        "mechanical_tau_star_candidate_count": s.get("mechanical_tau_star_candidate_count"),
        "has_auxiliary": data.get("auxiliary_summary") is not None,
    }


def load_row(path: Path, label: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("by_tau"), list) and data.get("schema_version") == "v7_phase2a_empirical.v1":
        return extract_from_empirical(data, path=path, label=label)
    if data.get("schema_version") == "v7_phase2a_tau_summary.v1":
        return extract_from_summary(data, path=path, label=label)
    if isinstance(data.get("by_tau"), list):
        return extract_from_empirical(data, path=path, label=label)
    return {"label": label, "path": str(path), "error": "unknown_schema"}


def markdown_table(rows: list[dict[str, Any]]) -> str:
    keys = [
        "label",
        "source",
        "model",
        "layer_index",
        "n_dialogues",
        "tau_max",
        "n_tau",
        "R_mean_tau0",
        "R_mean_tau1",
        "R_var_argmax_tau",
        "R_var_global_max_tau",
        "has_auxiliary",
    ]
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    lines = [
        "### Phase II-A 実行比較（機械抽出）",
        "",
        "主対比・CI は各 run の bootstrap / 事前登録に従う。ここは探索的な横並び。",
        "",
        header,
        sep,
    ]
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k, "")
            if v is None:
                v = ""
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Phase II-A 複数 JSON の比較表")
    p.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="*_with_contrib.json または *_summary.json",
    )
    p.add_argument(
        "--labels",
        type=str,
        default=None,
        help="カンマ区切り（省略時はファイル stem）",
    )
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    args = p.parse_args()

    labels = (
        [x.strip() for x in args.labels.split(",") if x.strip()]
        if args.labels
        else [p.stem for p in args.inputs]
    )
    if len(labels) != len(args.inputs):
        print(
            json.dumps({"error": "labels_count_mismatch", "n_in": len(args.inputs), "n_lab": len(labels)}),
            file=sys.stderr,
        )
        raise SystemExit(2)

    rows = [load_row(path.resolve(), lab) for path, lab in zip(args.inputs, labels)]
    payload = {
        "schema_version": "v7_phase2a_compare_runs.v1",
        "note_ja": "横並びは感度・スイープ用。理論 τ* やコーパス代理 τ の主張は各 run のキャプション規約に従う。",
        "runs": rows,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown_table(rows), encoding="utf-8")
    print("v7_phase2a_compare_runs_ok", json.dumps({"n": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
