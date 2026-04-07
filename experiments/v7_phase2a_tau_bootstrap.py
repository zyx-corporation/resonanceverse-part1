"""
Phase II-A: contributions_by_tau 付き JSON から R_mean のブートストラップ信頼区間（対話単位再標本）。

前提: v7_phase2a_empirical_run.py --export-contributions で生成した
``contributions_by_tau`` を含む出力 JSON。
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

from experiments.v7_phase2a_rail_metadata import (  # noqa: E402
    B_EMPIRICAL_MRMP,
    with_rail,
)


def bootstrap_mean_ci(
    values: list[float],
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    """対話別 R_d のリストを i.i.d. 再標本として平均のパーセンタイル CI。"""
    arr = np.array(values, dtype=np.float64)
    n = int(arr.size)
    if n < 2:
        m = float(arr.mean()) if n else float("nan")
        return {
            "R_mean_observed": m,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_dialogues": float(n),
        }
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(arr[idx].mean())
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "R_mean_observed": float(arr.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_dialogues": float(n),
    }


def _tau_to_dialogue_R(contribs: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for block in contribs:
        tau = int(block["tau"])
        out[tau] = {
            str(x["dialogue_id"]): float(x["R_d"])
            for x in (block.get("per_dialogue") or [])
            if isinstance(x.get("R_d"), (int, float))
        }
    return out


def paired_mean_diff_ci(
    contribs: list[dict[str, Any]],
    tau_a: int,
    tau_b: int,
    *,
    n_boot: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    """
    同一対話 ID で揃えた差分 Δ_d = R_d(τ_a) - R_d(τ_b) の平均について、
    対話を復元抽出してブートストラップ CI（独立 τ の CI 比較より適切な場合がある）。
    """
    by_t = _tau_to_dialogue_R(contribs)
    if tau_a not in by_t or tau_b not in by_t:
        return {
            "error": "missing_tau",
            "tau_a": tau_a,
            "tau_b": tau_b,
        }
    da, db = by_t[tau_a], by_t[tau_b]
    common = sorted(set(da.keys()) & set(db.keys()))
    if len(common) < 2:
        deltas = [da[k] - db[k] for k in common]
        m = float(np.mean(deltas)) if deltas else float("nan")
        return {
            "tau_a": tau_a,
            "tau_b": tau_b,
            "delta_mean_observed": m,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_paired_dialogues": len(common),
            "note": "n<2",
        }
    deltas = [float(da[k] - db[k]) for k in common]
    st = bootstrap_mean_ci(
        deltas,
        n_boot=max(100, n_boot),
        seed=seed + tau_a * 31 + tau_b * 7,
        alpha=alpha,
    )
    return {
        "tau_a": tau_a,
        "tau_b": tau_b,
        "delta_mean_observed": st["R_mean_observed"],
        "ci_low": st["ci_low"],
        "ci_high": st["ci_high"],
        "n_paired_dialogues": int(st["n_dialogues"]),
        "interpretation_ja": "E[Δ] = E[R(τ_a)-R(τ_b)] の対話平均。0 が CI 外なら τ 間の差に注意。",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Phase II-A R_mean のブートストラップ CI")
    p.add_argument(
        "json_path",
        type=Path,
        help="contributions_by_tau を含む JSON",
    )
    p.add_argument("--boot", type=int, default=4000, help="再標本回数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument(
        "--paired-diff",
        action="append",
        default=[],
        metavar="TAU_A,TAU_B",
        help="同一対話で揃えた平均差 R(τ_a)-R(τ_b) の CI（繰り返し可）。例: --paired-diff 0,1",
    )
    args = p.parse_args()

    data = json.loads(args.json_path.read_text(encoding="utf-8"))
    contribs = data.get("contributions_by_tau")
    if not contribs:
        print(
            json.dumps(
                {
                    "error": "no contributions_by_tau",
                    "hint": (
                        "re-run v7_phase2a_empirical_run.py "
                        "with --export-contributions"
                    ),
                }
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    rows_out: list[dict[str, Any]] = []
    for block in contribs:
        tau = int(block["tau"])
        per_d = block.get("per_dialogue") or []
        vals = [
            float(x["R_d"])
            for x in per_d
            if isinstance(x.get("R_d"), (int, float))
        ]
        st = bootstrap_mean_ci(
            vals,
            n_boot=max(100, args.boot),
            seed=args.seed + tau * 17,
            alpha=args.alpha,
        )
        st["tau"] = tau
        rows_out.append(st)

    payload = {
        "schema_version": "v7_phase2a_bootstrap.v1",
        **with_rail(str(data.get("rail_id") or B_EMPIRICAL_MRMP)),
        "source": str(args.json_path),
        "n_boot": max(100, args.boot),
        "alpha": args.alpha,
        "method_ja": (
            "対話 ID 単位の R_d を、対話数 n 個からの復元抽出で"
            "ブートストラップ（平均の分布の 2.5–97.5%）"
        ),
        "by_tau": rows_out,
    }

    paired: list[dict[str, Any]] = []
    for spec in args.paired_diff:
        parts = str(spec).replace(" ", "").split(",")
        if len(parts) != 2:
            print(json.dumps({"error": "bad_paired_diff", "spec": spec}), file=sys.stderr)
            sys.exit(1)
        ta, tb = int(parts[0]), int(parts[1])
        paired.append(
            paired_mean_diff_ci(
                contribs,
                ta,
                tb,
                n_boot=max(100, args.boot),
                seed=args.seed,
                alpha=args.alpha,
            )
        )
    if paired:
        payload["paired_mean_diffs"] = paired
        payload["method_paired_ja"] = (
            "各対話について R_d(τ_a)-R_d(τ_b) を定義し、その n 個の平均を"
            "対話単位で復元抽出してブートストラップ"
        )

    js = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js, encoding="utf-8")
    print(
        "v7_phase2a_bootstrap_ok",
        json.dumps({"taus": len(rows_out)}, ensure_ascii=False),
    )

    if args.out_md:
        lines = [
            "## Phase II-A R_mean のブートストラップ CI（対話再標本）",
            "",
            f"- 入力: `{args.json_path}`",
            f"- B = {payload['n_boot']}, α = {args.alpha}",
            "",
            "| τ | n | R_mean | CI low | CI high |",
            "|---|---|--------|--------|---------|",
        ]
        for r in rows_out[:40]:
            row = (
                f"| {r['tau']} | {int(r['n_dialogues'])} | "
                f"{r['R_mean_observed']:.4f} | "
                f"{r['ci_low']:.4f} | {r['ci_high']:.4f} |"
            )
            lines.append(row)
        if len(rows_out) > 40:
            lines.append(f"| … | | | | （他 {len(rows_out) - 40} 行） |")
        if paired:
            lines.extend(
                [
                    "",
                    "### ペア差 Δ = R(τ_a) − R(τ_b)（同一対話 ID）",
                    "",
                    "| τ_a | τ_b | n | Δ_mean | CI low | CI high |",
                    "|-----|-----|---|--------|--------|---------|",
                ]
            )
            for q in paired:
                if q.get("error"):
                    continue
                lines.append(
                    f"| {q['tau_a']} | {q['tau_b']} | {q.get('n_paired_dialogues', '')} | "
                    f"{q.get('delta_mean_observed', float('nan')):.4f} | "
                    f"{q.get('ci_low', float('nan')):.4f} | "
                    f"{q.get('ci_high', float('nan')):.4f} |"
                )
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
