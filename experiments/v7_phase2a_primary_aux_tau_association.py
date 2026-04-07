"""
主解析 by_tau の R_mean(τ) と、auxiliary_label_delay_coherence 各軸の R_mean(τ) の **τ 系列**での関連。

- Pearson: 共通 τ 上のベクトル間の相関（サンプル数 = τ 点数；探索的）。
- partial_pearson: 軸 k について、他軸の R_mean を説明変数としたときの「残差同士」の相関（簡易部分関連）。

スケールは主・補助で異なりうるため、解釈は補助図と同様に慎重に。
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


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or b.size < 3:
        return None
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    return None if r != r else r


def _ols_residual(y: np.ndarray, x_design: np.ndarray) -> np.ndarray:
    """y - X @ beta, X に定数列を含む。"""
    n = y.size
    X = np.column_stack([np.ones(n), x_design])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def align_primary_auxiliary(
    by_tau: list[dict[str, Any]],
    aux: dict[str, list[dict[str, Any]]],
    *,
    min_n: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """tau -> primary R_mean, aux R_mean per axis（n 十分な τ のみ）。"""
    primary_map = {int(r["tau"]): r for r in by_tau}
    taus_sorted = sorted(primary_map.keys())
    axis_names = sorted(aux.keys())
    series_p: list[float] = []
    series_axes: dict[str, list[float]] = {a: [] for a in axis_names}
    taus_out: list[int] = []

    aux_by_tau = {a: {int(r["tau"]): r for r in aux[a]} for a in axis_names}

    for t in taus_sorted:
        pr = primary_map[t]
        if int(pr["n"]) < min_n:
            continue
        ok = True
        vals: dict[str, float] = {}
        for a in axis_names:
            row = aux_by_tau[a].get(t)
            if row is None or int(row.get("n", 0)) < min_n:
                ok = False
                break
            vals[a] = float(row["R_mean"])
        if not ok:
            continue
        taus_out.append(t)
        series_p.append(float(pr["R_mean"]))
        for a in axis_names:
            series_axes[a].append(vals[a])

    y = np.array(series_p, dtype=np.float64)
    xa = {a: np.array(series_axes[a], dtype=np.float64) for a in axis_names}
    return np.array(taus_out, dtype=np.int64), y, xa


def analyze(
    data: dict[str, Any],
    *,
    min_n: int,
) -> dict[str, Any]:
    by_tau = data.get("by_tau") or []
    aux = data.get("auxiliary_label_delay_coherence")
    if not by_tau or not isinstance(aux, dict) or not aux:
        return {
            "schema_version": "v7_phase2a_primary_aux_tau_assoc.v1",
            "error": "missing_by_tau_or_auxiliary",
            "n_tau_used": 0,
        }

    taus, y, xa = align_primary_auxiliary(by_tau, aux, min_n=min_n)
    axes = sorted(xa.keys())
    if y.size < 3:
        return {
            "schema_version": "v7_phase2a_primary_aux_tau_assoc.v1",
            "n_tau_used": int(y.size),
            "note_ja": "有効 τ が少なすぎる",
            "by_axis": {},
        }

    by_axis: dict[str, Any] = {}
    # 全軸行列（部分相関用）
    mat = np.column_stack([xa[a] for a in axes])

    for i, a in enumerate(axes):
        xi = xa[a]
        rp = _pearson(y, xi)
        others = [j for j in range(len(axes)) if j != i]
        r_partial = None
        if len(others) >= 1 and y.size > len(others) + 2:
            z = mat[:, others]
            ry = _ols_residual(y, z)
            rxi = _ols_residual(xi, z)
            r_partial = _pearson(ry, rxi)
        by_axis[a] = {
            "pearson_Rmean_primary_vs_axis": rp,
            "partial_pearson_controlling_other_axes": r_partial,
        }

    return {
        "schema_version": "v7_phase2a_primary_aux_tau_assoc.v1",
        "note_ja": "τ が点の単位。多重比較未調整。主・補助はスケール非互換のまま R_mean 系列のみ比較。",
        "min_n": min_n,
        "n_tau_used": int(y.size),
        "taus_used": [int(t) for t in taus.tolist()],
        "by_axis": by_axis,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="主解析 vs 補助軸 R_mean(τ) 系列の関連（探索的）")
    p.add_argument("with_contrib_json", type=Path, help="*_with_contrib.json")
    p.add_argument("--min-n", type=int, default=1, help="各 τ で主・補助とも n>=この値のみ使用")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    path = args.with_contrib_json.resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    out = analyze(data, min_n=max(1, args.min_n))
    out = {**out, **with_rail(str(data.get("rail_id") or B_EMPIRICAL_MRMP))}
    js = json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js, encoding="utf-8")
    else:
        sys.stdout.write(js)
    print(
        "v7_phase2a_primary_aux_tau_assoc_ok",
        json.dumps({"n_tau_used": out.get("n_tau_used")}, ensure_ascii=False),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
