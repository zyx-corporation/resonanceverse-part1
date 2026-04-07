"""
1 次元線形遅差分の安定性から「最初に不安定化する整数ラグ τ」を求める。

モデル（教材用サロゲート）::

    x_{t+1} = a x_t + b x_{t-τ},  a = 1 - dt·α,  b = dt·β / N

delay_sweep の (N, dt, alpha, beta) をスカラーに縮約した対応であり、
**DDRF の定理3.3 の理論 τ* ではない**。`theoretical_tau_star` への自動代入はしない。
参照 JSON の `scalar_linear_surrogate` ブロック用の数値を生成する。

用法::

    python experiments/v7_phase2a_scalar_delay_tau_suggest.py \\
      --N 10 --dt 0.05 --alpha 0.15 --beta 0.85 --tau-max 80
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


def companion_spectral_radius(tau: int, *, a: float, b: float) -> float:
    if tau < 0:
        raise ValueError("tau must be >= 0")
    if tau == 0:
        return abs(a + b)
    dim = tau + 1
    m = np.zeros((dim, dim), dtype=np.float64)
    m[0, 0] = a
    m[0, tau] = b
    for i in range(1, dim):
        m[i, i - 1] = 1.0
    vals = np.linalg.eigvals(m)
    return float(np.max(np.abs(vals)))


def smallest_tau_unstable(
    *,
    tau_max: int,
    a: float,
    b: float,
    rho_threshold: float = 1.0 + 1e-9,
) -> int | None:
    for tau in range(0, tau_max + 1):
        if companion_spectral_radius(tau, a=a, b=b) > rho_threshold:
            return tau
    return None


def build_surrogate_block(
    *,
    N: int,
    dt: float,
    alpha: float,
    beta: float,
    tau_max: int,
) -> dict[str, Any]:
    a = float(1.0 - dt * alpha)
    b = float(dt * beta / max(N, 1))
    tau_first = smallest_tau_unstable(tau_max=tau_max, a=a, b=b)
    rhos = [
        companion_spectral_radius(t, a=a, b=b)
        for t in range(0, min(6, tau_max + 1))
    ]
    return {
        "model_id": "scalar_linear_delay_recurrence_v1",
        "warning_ja": (
            "DDRF の理論 τ* ではない。論文比較表の theoretical_tau_star 行や "
            "乖離％の本番主張に使わないこと。教材・感度・付録参照用。"
        ),
        "recurrence_ja": "x_{t+1} = a·x_t + b·x_{t-τ}、a=1-dt·α、b=dt·β/N",
        "hyperparams": {
            "N": N,
            "dt": dt,
            "alpha": alpha,
            "beta": beta,
        },
        "derived_ab": {"a": a, "b": b},
        "tau_max_searched": tau_max,
        "tau_first_unstable_lag": tau_first,
        "spectral_radius_tau_0_to_5": rhos,
        "pointer_appendix_md": (
            "docs/planning/v7_phase2a_theoretical_tau_bridge_appendix_ja.md"
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="線形スカラー遅差分の不安定化 τ（教材用サロゲート）",
    )
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--beta", type=float, default=0.85)
    p.add_argument("--tau-max", type=int, default=120)
    p.add_argument("--out", type=Path, default=None, help="JSON 断片をファイルへ")
    args = p.parse_args()

    block = build_surrogate_block(
        N=args.N,
        dt=args.dt,
        alpha=args.alpha,
        beta=args.beta,
        tau_max=args.tau_max,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"scalar_linear_surrogate": block}, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    tag = "v7_phase2a_scalar_delay_tau_suggest_ok"
    print(tag, json.dumps(block, ensure_ascii=False))


if __name__ == "__main__":
    main()
