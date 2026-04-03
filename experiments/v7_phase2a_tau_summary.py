"""
Phase II-A 出力 JSON（v7_phase2a_empirical_run）の後処理:
Var(τ) のピーク・移動平均・機械的候補の整理（解釈補助）。

主解析の定義は変えない。別紙サマリ用。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float64)
    k = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x.astype(np.float64), k, mode="same")


def local_maxima_indices(v: np.ndarray) -> list[int]:
    """厳密な山（両隣より大きい）。端は除外。"""
    out: list[int] = []
    for i in range(1, len(v) - 1):
        if v[i] > v[i - 1] and v[i] > v[i + 1]:
            out.append(i)
    return out


def _nan_argmax_tau(taus: np.ndarray, vals: np.ndarray) -> int:
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return 0
    if np.all(np.isnan(v)):
        return int(taus[0])
    return int(taus[int(np.nanargmax(v))])


def _nan_argmin_tau(taus: np.ndarray, vals: np.ndarray) -> int:
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return 0
    if np.all(np.isnan(v)):
        return int(taus[0])
    return int(taus[int(np.nanargmin(v))])


def summarize_by_tau(rows: list[dict[str, Any]], *, smooth_window: int = 5) -> dict[str, Any]:
    taus = np.array([int(r["tau"]) for r in rows], dtype=np.int64)
    r_mean = np.array([float(r["R_mean"]) for r in rows], dtype=np.float64)
    r_var = np.array([float(r["R_var"]) for r in rows], dtype=np.float64)
    n = np.array([int(r["n"]) for r in rows], dtype=np.int64)

    r_var_s = rolling_mean(r_var, smooth_window)
    peaks = local_maxima_indices(r_var)
    peaks_s = local_maxima_indices(r_var_s)
    peak_taus = [int(taus[i]) for i in peaks]
    peak_vals = [float(r_var[i]) for i in peaks]
    order = np.argsort(-r_var)
    top_k = min(8, len(rows))
    top_tau = [int(taus[i]) for i in order[:top_k]]
    top_var = [float(r_var[i]) for i in order[:top_k]]

    mech_true = [int(r["tau"]) for r in rows if r.get("tau_star_candidate")]

    r_var_glob_i = int(np.nanargmax(r_var)) if not np.all(np.isnan(r_var)) else 0

    return {
        "tau_range": [int(taus[0]), int(taus[-1])],
        "n_tau_points": len(rows),
        "R_mean_argmax_tau": _nan_argmax_tau(taus, r_mean),
        "R_mean_argmin_tau": _nan_argmin_tau(taus, r_mean),
        "R_var_global_max_tau": int(taus[r_var_glob_i]),
        "R_var_global_max": float(np.nanmax(r_var)) if r_var.size else float("nan"),
        "local_maxima_R_var": [
            {"tau": peak_taus[j], "R_var": peak_vals[j]} for j in range(len(peak_taus))
        ],
        "local_maxima_R_var_smoothed": [
            {"tau": int(taus[i]), "R_var_smooth": float(r_var_s[i])}
            for i in peaks_s
        ],
        "R_var_smoothed_argmax_tau": _nan_argmax_tau(taus, r_var_s),
        "top_R_var_tau": [{"tau": top_tau[k], "R_var": top_var[k]} for k in range(len(top_tau))],
        "smooth_window": smooth_window,
        "mechanical_tau_star_candidate_count": len(mech_true),
        "note_ja": "local_maxima は R_var(τ) の両隣より大きい点。mechanical 候補は隣接差分の符号で多数出るため、ピークと併記して解釈する。",
    }


def summarize_n_per_tau(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    各 τ の有効対話数 n（by_tau 行の ``n``）。高 τ で n が減ると Var(τ) の解釈に注意。
    """
    pairs: list[dict[str, int]] = []
    for r in rows:
        pairs.append({"tau": int(r["tau"]), "n": int(r["n"])})
    if not pairs:
        return {
            "per_tau": [],
            "n_min": 0,
            "n_max": 0,
            "tau_at_n_min": None,
            "tau_at_n_max": None,
            "n_drop_transitions": [],
            "note_ja": "by_tau が空。",
        }
    ns = np.array([p["n"] for p in pairs], dtype=np.int64)
    taus = np.array([p["tau"] for p in pairs], dtype=np.int64)
    i_min = int(np.argmin(ns))
    i_max = int(np.argmax(ns))
    transitions: list[dict[str, int]] = []
    prev_n: int | None = None
    for p in pairs:
        if prev_n is not None and p["n"] != prev_n:
            transitions.append(
                {"tau": p["tau"], "n_from": prev_n, "n_to": p["n"]}
            )
        prev_n = p["n"]
    return {
        "per_tau": pairs,
        "n_min": int(ns.min()),
        "n_max": int(ns.max()),
        "tau_at_n_min": int(taus[i_min]),
        "tau_at_n_max": int(taus[i_max]),
        "n_drop_transitions": transitions,
        "note_ja": "高 τ で n が減るとき Var(τ) のピークは標本サイズと併記して解釈する。",
    }


def format_n_tau_markdown(
    n_summary: dict[str, Any],
    *,
    max_rows_full: int = 400,
) -> list[str]:
    """n(τ) 表用の Markdown 行リスト（長いときは先頭・末尾に分割）。"""
    per = n_summary.get("per_tau") or []
    lines = [
        "### n(τ)（主解析の有効対話数）",
        "",
        "各 τ について R(τ) に寄与した対話数。n が小さい τ では R_var の山がサンプル不足で膨らみうる。",
        "",
        f"- n の範囲: **{n_summary['n_min']}** … **{n_summary['n_max']}**（最小 τ={n_summary['tau_at_n_min']}、最大 τ={n_summary['tau_at_n_max']}）",
        f"- n が変化する境界: **{len(n_summary.get('n_drop_transitions') or [])}** 回",
        "",
        n_summary["note_ja"],
        "",
    ]
    if len(per) <= max_rows_full:
        lines.extend(["| τ | n |", "|---|---|"])
        for p in per:
            lines.append(f"| {p['tau']} | {p['n']} |")
        lines.append("")
        return lines
    half = max_rows_full // 2
    lines.append(f"（行数が多いため表は先頭 {half} 行と末尾 {half} 行のみ）")
    lines.append("")
    lines.extend(["| τ | n |", "|---|---|"])
    for p in per[:half]:
        lines.append(f"| {p['tau']} | {p['n']} |")
    lines.append("| … | … |")
    for p in per[-half:]:
        lines.append(f"| {p['tau']} | {p['n']} |")
    lines.append("")
    return lines


def summarize_auxiliary_label_delay(
    aux: dict[str, list[dict[str, Any]]],
    *,
    smooth_window: int,
) -> dict[str, dict[str, Any]]:
    """auxiliary_label_delay_coherence（軸 → by_tau 行）を軸ごとに summarize_by_tau。"""
    out: dict[str, dict[str, Any]] = {}
    for axis, axis_rows in sorted(aux.items()):
        if not axis_rows:
            continue
        out[axis] = summarize_by_tau(axis_rows, smooth_window=smooth_window)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Phase II-A τ 掃引 JSON のサマリ（Var ピーク等）")
    p.add_argument("json_path", type=Path, help="v7_phase2a_mrmp_tau_*.json")
    p.add_argument("--smooth", type=int, default=5, help="R_var 移動平均窓（奇数推奨）")
    p.add_argument("--out-md", type=Path, default=None, help="Markdown 出力")
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="構造化サマリ JSON（summary・auxiliary_summary・smooth_window）",
    )
    p.add_argument(
        "--n-table-max-rows",
        type=int,
        default=400,
        help="n(τ) 表を分割表示する閾値（これを超える τ 点数は先頭・末尾のみ）",
    )
    args = p.parse_args()

    data = json.loads(args.json_path.read_text(encoding="utf-8"))
    rows = data.get("by_tau") or []
    if not rows:
        print(json.dumps({"error": "no by_tau"}), file=sys.stderr)
        sys.exit(1)

    smooth = max(1, args.smooth)
    summ = summarize_by_tau(rows, smooth_window=smooth)
    n_tau = summarize_n_per_tau(rows)
    payload: dict[str, Any] = {
        "schema_version": "v7_phase2a_tau_summary.v1",
        "source": str(args.json_path),
        "input_schema": data.get("schema_version"),
        "n_dialogues_in_run": data.get("n_dialogues"),
        "smooth_window": smooth,
        "summary": summ,
        "n_per_tau": n_tau,
    }
    aux_raw = data.get("auxiliary_label_delay_coherence")
    if isinstance(aux_raw, dict) and aux_raw:
        payload["auxiliary_summary"] = summarize_auxiliary_label_delay(
            aux_raw, smooth_window=smooth
        )

    lines = [
        "## Phase II-A τ 掃引サマリ（後処理）",
        "",
        f"- 入力: `{args.json_path}`",
        f"- 対話数（実行時）: {data.get('n_dialogues')}",
        f"- τ 点: {summ['n_tau_points']}（{summ['tau_range'][0]} … {summ['tau_range'][1]}）",
        "",
        "### R_mean",
        "",
        f"- 最大の τ: **{summ['R_mean_argmax_tau']}**",
        f"- 最小の τ: **{summ['R_mean_argmin_tau']}**",
        "",
        "### R_var（対話間分散）",
        "",
        f"- 全体最大: τ = **{summ['R_var_global_max_tau']}**（値 ≈ {summ['R_var_global_max']:.4f}）",
        f"- 移動平均（窓 {summ['smooth_window']}）の最大 τ: **{summ.get('R_var_smoothed_argmax_tau', '—')}**",
        "",
    ]
    lines.extend(format_n_tau_markdown(n_tau, max_rows_full=max(50, args.n_table_max_rows)))
    lines.extend(
        [
            "### Var(τ) の局所最大（両隣より高い山）",
            "",
            "| τ | R_var |",
            "|---|-------|",
        ]
    )
    for item in summ["local_maxima_R_var"][:20]:
        lines.append(f"| {item['tau']} | {item['R_var']:.6f} |")
    if len(summ["local_maxima_R_var"]) > 20:
        lines.append(f"| … | （他 {len(summ['local_maxima_R_var']) - 20} 峰） |")
    lines.extend(
        [
            "",
            "### Var の移動平均に対する局所最大（ノイズ抑制の参考）",
            "",
            "| τ | R_var (smooth) |",
            "|---|------------------|",
        ]
    )
    for item in summ.get("local_maxima_R_var_smoothed", [])[:15]:
        lines.append(f"| {item['tau']} | {item['R_var_smooth']:.6f} |")
    n_map = {p["tau"]: p["n"] for p in n_tau["per_tau"]}
    lines.extend(
        [
            "",
            "### R_var 上位（n(τ) 併記）",
            "",
            "| τ | R_var | n |",
            "|---|-------|---|",
        ]
    )
    for item in summ["top_R_var_tau"]:
        t = item["tau"]
        lines.append(f"| {t} | {item['R_var']:.6f} | {n_map.get(t, '—')} |")
    lines.append("")
    lines.extend(
        [
            "### 機械的 `tau_star_candidate`",
            "",
            f"- 件数: **{summ['mechanical_tau_star_candidate_count']}** / {summ['n_tau_points']}",
            "",
            summ["note_ja"],
            "",
        ]
    )
    aux_summ = payload.get("auxiliary_summary")
    if isinstance(aux_summ, dict) and aux_summ:
        lines.extend(
            [
                "### 補助: 6 軸 label delay coherence（Var(τ) 要約）",
                "",
                "事前登録 `auxiliary_label_delay_coherence` と同型の R(τ)。各軸について対話間分散 R_var のピーク τ（主解析と同じ後処理）。",
                "",
                "| 軸 | Var 最大 τ | max R_var | mechanical 候補数 |",
                "|----|------------|-----------|-------------------|",
            ]
        )
        for axis in sorted(aux_summ.keys()):
            s = aux_summ[axis]
            lines.append(
                f"| {axis} | **{s['R_var_global_max_tau']}** | {s['R_var_global_max']:.6f} | "
                f"{s['mechanical_tau_star_candidate_count']} |"
            )
        lines.append("")
    md = "\n".join(lines) + "\n"
    print(md)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
