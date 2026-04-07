"""
設計書 II-A §3.1 型の数値スタブ: V_K 代理と離散 ΔV の尾平均で τ を掃引する。

[V_K の理論形は W_D* 一般ではない] 本スクリプトは合成遅延テンソル系で
**W* = 0** とみなし V(t) = ½||W(t)||²_F とする。Krasovskii 汎関数の追加項は入れない。

[連続時間 dV_K/dt ではない] 尾区間での **V_{t+1}-V_t** の平均を「符号代理」とする。

出力の **tau_exp_numeric_stub_*** は設計書の τ*_exp の完全実装ではない。
同一ダイナミクス上で正の平均 ΔV に最初に達する τ を記録するハーネス。
振動代理・MRMP R(τ) とは別ラベルで報告すること。
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

from experiments.v7_phase2a_delay_sweep import (  # noqa: E402
    simulate_tau_v_k_series,
)
from experiments.v7_phase2a_rail_metadata import (  # noqa: E402
    A_IIA_NUMERIC_SYNTHETIC,
    with_rail,
)


def metrics_from_v_series(
    v: np.ndarray,
    *,
    burn_frac: float,
) -> dict[str, float]:
    n = len(v)
    burn = int(n * burn_frac)
    burn = max(0, min(burn, n - 2))
    v_tail = v[burn:]
    d_v = np.diff(v_tail.astype(np.float64))
    if d_v.size == 0:
        return {
            "V_tail_mean": float(v_tail.mean()) if v_tail.size else 0.0,
            "mean_dV_tail": 0.0,
            "frac_dV_positive_tail": 0.0,
        }
    return {
        "V_tail_mean": float(v_tail.mean()),
        "mean_dV_tail": float(d_v.mean()),
        "frac_dV_positive_tail": float((d_v > 0).mean()),
    }


def run_lyapunov_tau_exp_stub_sweep(
    *,
    tau_max: int,
    steps: int,
    seed: int,
    N: int,
    d: int,
    dt: float,
    alpha: float,
    beta: float,
    noise: float,
    burn_frac: float,
    mean_dv_threshold: float,
    frac_positive_threshold: float,
    krasovskii_gamma: float = 0.0,
) -> dict[str, Any]:
    g = float(krasovskii_gamma)
    v_k_profile = (
        "w_squared_only" if g == 0.0 else "krs_discrete_delay_integral"
    )
    rows: list[dict[str, Any]] = []
    for tau in range(0, tau_max + 1):
        v = simulate_tau_v_k_series(
            N=N,
            d=d,
            tau=tau,
            steps=steps,
            dt=dt,
            alpha=alpha,
            beta=beta,
            noise=noise,
            seed=seed + tau * 17,
            krasovskii_gamma=g,
        )
        m = metrics_from_v_series(v, burn_frac=burn_frac)
        rows.append({"tau": tau, **m})

    tau_mean: int | None = None
    for r in rows:
        if r["mean_dV_tail"] > mean_dv_threshold:
            tau_mean = int(r["tau"])
            break

    tau_frac: int | None = None
    for r in rows:
        if r["frac_dV_positive_tail"] > frac_positive_threshold:
            tau_frac = int(r["tau"])
            break

    return {
        "schema_version": "v7_phase2a_tau_exp_lyapunov_stub.v1",
        **with_rail(A_IIA_NUMERIC_SYNTHETIC),
        "mode": "lyapunov_V_mean_dV_numeric_stub",
        "design_doc_pointer": (
            "docs/v7/Resonanceverse_v7.0_Experimental_Design.md §3.1"
        ),
        "tau_exp_operational_spec_md": (
            "docs/planning/v7_phase2a_numeric_tau_exp.md"
        ),
        "v_k_profile": v_k_profile,
        "krasovskii_gamma": g,
        "theory_pointer": (
            "docs/v7/Resonanceverse_Theory_v7.0.md（定理 3.3 系・V_K）"
        ),
        "W_star_assumption_ja": (
            "合成反対称テンソル系で平衡を W*=0 と仮定。"
            "V(t)=½||W||_F^2 は理論の V_K(W_D*) 一般形の代理。"
        ),
        "dV_discrete_ja": (
            "burn 以降の時間系列で V_{t+1}-V_t をとり、尾区間の平均・正比率を報告。"
            "連続時間の dV_K/dt ではない。"
        ),
        "tau_exp_labeling_ja": (
            "mean_dV 版: 尾平均 ΔV が閾値を超えた最小 τ。"
            "frac_positive 版: 尾で ΔV>0 の割合が閾値を超えた最小 τ。"
            "設計書 τ*_exp の完全再現ではない。"
        ),
        "mean_dv_threshold": float(mean_dv_threshold),
        "frac_positive_threshold": float(frac_positive_threshold),
        "burn_frac": float(burn_frac),
        "N": N,
        "d": d,
        "steps": steps,
        "dt": dt,
        "alpha": alpha,
        "beta": beta,
        "noise": noise,
        "seed_base": seed,
        "tau_exp_numeric_stub_mean_dV": tau_mean,
        "tau_exp_numeric_stub_frac_positive": tau_frac,
        "by_tau": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="v7 Phase II-A: Lyapunov V 代理に基づく τ 数値スタブ（設計書 §3.1 参照）",
    )
    p.add_argument("--demo", action="store_true", help="CI 向けに τ・ステップを抑える")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tau-max", type=int, default=10)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--d", type=int, default=6)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--beta", type=float, default=0.85)
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--burn-frac", type=float, default=0.5)
    p.add_argument(
        "--mean-dv-threshold",
        type=float,
        default=1e-5,
        help=(
            "尾平均 ΔV がこの値を超えた最小 τ を "
            "tau_exp_numeric_stub_mean_dV とする"
        ),
    )
    p.add_argument(
        "--frac-positive-threshold",
        type=float,
        default=0.52,
        help="尾で ΔV>0 の割合がこの値を超えた最小 τ（>0.5 でやや頑健）",
    )
    p.add_argument(
        "--krasovskii-gamma",
        type=float,
        default=0.0,
        help=(
            "V に離散遅延和項 γ·Σ½‖W(t−k)‖² を足す（0 で従来の ½‖W‖² のみ）。"
            "docs/planning/v7_phase2a_numeric_tau_exp.md 参照。"
        ),
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    tau_max = args.tau_max
    steps = args.steps
    N, d = args.N, args.d
    if args.demo:
        tau_max = min(tau_max, 5)
        steps = min(steps, 600)
        N = min(N, 8)
        d = min(d, 4)

    payload = run_lyapunov_tau_exp_stub_sweep(
        tau_max=tau_max,
        steps=steps,
        seed=args.seed,
        N=N,
        d=d,
        dt=args.dt,
        alpha=args.alpha,
        beta=args.beta,
        noise=args.noise,
        burn_frac=args.burn_frac,
        mean_dv_threshold=args.mean_dv_threshold,
        frac_positive_threshold=args.frac_positive_threshold,
        krasovskii_gamma=args.krasovskii_gamma,
    )
    js = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js, encoding="utf-8")
    meta = {
        "taus": len(payload["by_tau"]),
        "tau_mean": payload["tau_exp_numeric_stub_mean_dV"],
        "tau_frac": payload["tau_exp_numeric_stub_frac_positive"],
    }
    tag = "v7_phase2a_tau_exp_lyapunov_stub_ok"
    print(tag, json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
