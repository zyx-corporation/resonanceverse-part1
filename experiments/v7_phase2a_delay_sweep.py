"""
v7 Phase II-A: 遅延 τ の掃引と安定性プロキシ（離散時間・小規模テンソル）。

定理3.3 の V_K の厳密実装ではなく、遅延結合があるときのエネルギー
||W||_F の後半区間における変動係数（oscillation_score）を記録し、
τ を増やしたときの挙動差を見るハーネス。
`simulate_tau_v_k_series` は ½||W||²_F の時系列（Lyapunov スタブ用）。
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


def _ring_shift(hist: list[np.ndarray], w_new: np.ndarray) -> None:
    for i in range(len(hist) - 1, 0, -1):
        hist[i] = hist[i - 1]
    hist[0] = w_new


def _simulate_tau_frobenius_norm_trace(
    *,
    N: int,
    d: int,
    tau: int,
    steps: int,
    dt: float,
    alpha: float,
    beta: float,
    noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(各ステップの ||W||_F トレース, 最終 W)。"""
    rng = np.random.default_rng(seed)
    w0 = rng.standard_normal((N, N, d)) * 0.08
    w0 = 0.5 * (w0 - np.transpose(w0, (1, 0, 2)))
    hist: list[np.ndarray] = [w0.copy() for _ in range(tau + 1)]

    trace = np.empty(steps, dtype=np.float64)
    for ti in range(steps):
        w = hist[0]
        w_tau = hist[tau] if tau > 0 else w
        d_w = np.zeros_like(w)
        for i in range(N):
            for j in range(N):
                acc = np.zeros(d, dtype=np.float64)
                for k in range(N):
                    acc += w[i, k] * w_tau[k, j]
                d_w[i, j] = -alpha * w[i, j] + beta * acc / max(N, 1)
        w_new = w + dt * d_w + noise * rng.standard_normal((N, N, d))
        w_new = 0.5 * (w_new - np.transpose(w_new, (1, 0, 2)))
        _ring_shift(hist, w_new)
        trace[ti] = float(np.linalg.norm(hist[0]))
    return trace, hist[0].copy()


def _simulate_tau_v_k_discrete_series(
    *,
    N: int,
    d: int,
    tau: int,
    steps: int,
    dt: float,
    alpha: float,
    beta: float,
    noise: float,
    seed: int,
    krasovskii_gamma: float,
) -> np.ndarray:
    """
    各ステップの離散 V: ½‖W(t)‖²_F + γ·Σ_{k=1}^{τ} ½‖W(t−k)‖²_F。

    理論の Krasovskii 第 2 項の**離散ラグ和の代理**（連続積分の厳密対応ではない）。
    """
    rng = np.random.default_rng(seed)
    w0 = rng.standard_normal((N, N, d)) * 0.08
    w0 = 0.5 * (w0 - np.transpose(w0, (1, 0, 2)))
    hist: list[np.ndarray] = [w0.copy() for _ in range(tau + 1)]
    v_out = np.empty(steps, dtype=np.float64)
    g = float(krasovskii_gamma)
    for ti in range(steps):
        w = hist[0]
        w_tau = hist[tau] if tau > 0 else w
        d_w = np.zeros_like(w)
        for i in range(N):
            for j in range(N):
                acc = np.zeros(d, dtype=np.float64)
                for k in range(N):
                    acc += w[i, k] * w_tau[k, j]
                d_w[i, j] = -alpha * w[i, j] + beta * acc / max(N, 1)
        w_new = w + dt * d_w + noise * rng.standard_normal((N, N, d))
        w_new = 0.5 * (w_new - np.transpose(w_new, (1, 0, 2)))
        _ring_shift(hist, w_new)
        fro0 = float(np.linalg.norm(hist[0]))
        vk = 0.5 * (fro0 * fro0)
        if g != 0.0 and tau > 0:
            for j in range(1, tau + 1):
                froj = float(np.linalg.norm(hist[j]))
                vk += g * 0.5 * (froj * froj)
        v_out[ti] = vk
    return v_out


def simulate_tau_v_k_series(
    *,
    N: int,
    d: int,
    tau: int,
    steps: int,
    dt: float,
    alpha: float,
    beta: float,
    noise: float,
    seed: int,
    krasovskii_gamma: float = 0.0,
) -> np.ndarray:
    """
    各ステップの V_K 代理（平衡 W*=0 の合成仮定）。

    krasovskii_gamma=0（既定）: V=½‖W(t)‖²_F のみ。
    krasovskii_gamma>0: 上式に離散遅延和項 γ·Σ ½‖W(t−k)‖²_F を加える（段階 2）。
    """
    if float(krasovskii_gamma) == 0.0:
        tr, _w = _simulate_tau_frobenius_norm_trace(
            N=N,
            d=d,
            tau=tau,
            steps=steps,
            dt=dt,
            alpha=alpha,
            beta=beta,
            noise=noise,
            seed=seed,
        )
        return 0.5 * (tr * tr)
    return _simulate_tau_v_k_discrete_series(
        N=N,
        d=d,
        tau=tau,
        steps=steps,
        dt=dt,
        alpha=alpha,
        beta=beta,
        noise=noise,
        seed=seed,
        krasovskii_gamma=krasovskii_gamma,
    )


def simulate_tau(
    *,
    N: int,
    d: int,
    tau: int,
    steps: int,
    dt: float,
    alpha: float,
    beta: float,
    noise: float,
    seed: int,
) -> dict[str, Any]:
    trace, w_final = _simulate_tau_frobenius_norm_trace(
        N=N,
        d=d,
        tau=tau,
        steps=steps,
        dt=dt,
        alpha=alpha,
        beta=beta,
        noise=noise,
        seed=seed,
    )
    energy = trace.tolist()

    burn = steps // 2
    e2 = np.array(energy[burn:], dtype=np.float64)
    mean_e = float(e2.mean()) if len(e2) else 0.0
    std_e = float(e2.std()) if len(e2) else 0.0
    osc = float(std_e / (mean_e + 1e-9))

    return {
        "tau": tau,
        "energy_mean_tail": mean_e,
        "energy_std_tail": std_e,
        "oscillation_score": osc,
        "asymmetry_fro_tail": float(
            np.linalg.norm(w_final - np.transpose(w_final, (1, 0, 2)))
        ),
    }


def run_sweep(
    *,
    tau_max: int,
    seed: int,
    N: int,
    d: int,
    steps: int,
    dt: float,
    alpha: float,
    beta: float,
    noise: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for tau in range(0, tau_max + 1):
        rows.append(
            simulate_tau(
                N=N,
                d=d,
                tau=tau,
                steps=steps,
                dt=dt,
                alpha=alpha,
                beta=beta,
                noise=noise,
                seed=seed + tau * 17,
            )
        )
    oscs = [r["oscillation_score"] for r in rows]
    # 簡易「分岐」プロキシ: スコアが隣接 τ で最大急増した点
    jumps = [0.0]
    for i in range(1, len(oscs)):
        jumps.append(float(oscs[i] - oscs[i - 1]))
    k = int(np.argmax(jumps))
    return {
        "schema_version": "v7_phase2a.v1",
        "N": N,
        "d": d,
        "steps": steps,
        "dt": dt,
        "alpha": alpha,
        "beta": beta,
        "noise": noise,
        "seed_base": seed,
        "note": "離散近似。oscillation_score は遅延導入後のエネルギー変動の目安。",
        "by_tau": rows,
        "tau_largest_jump": k,
        "tau_exp_proxy_oscillation_jump": k,
        "oscillation_scores": oscs,
        "design_bridge_ja": (
            "tau_largest_jump / tau_exp_proxy_oscillation_jump は"
            "設計書 II-A の τ*_exp ではなく、本離散テンソル系における"
            "振動スコア急増の τ の探索的代理。理論 τ* との接続は "
            "docs/planning/v7_phase2a_numeric_tau_exp.md を参照。"
        ),
    }


def run_alpha_sweep(
    *,
    alphas: list[float],
    tau_max: int,
    seed: int,
    N: int,
    d: int,
    steps: int,
    dt: float,
    beta: float,
    noise: float,
) -> dict[str, Any]:
    """強凸性代理として alpha をスイープ（各値で独立 run_sweep）。"""
    rows_out: list[dict[str, Any]] = []
    for ai, alpha in enumerate(alphas):
        sw = run_sweep(
            tau_max=tau_max,
            seed=seed + ai * 97,
            N=N,
            d=d,
            steps=steps,
            dt=dt,
            alpha=float(alpha),
            beta=beta,
            noise=noise,
        )
        rows_out.append(
            {
                "alpha": float(alpha),
                "tau_exp_proxy_oscillation_jump": sw["tau_largest_jump"],
                "oscillation_scores_tail": sw["oscillation_scores"][-3:],
            }
        )
    return {
        "schema_version": "v7_phase2a_alpha_sweep.v1",
        "note_ja": "各要素は v7_phase2a.v1 の単発掃引と同型。alpha を μ の離散代理として見た感度表。",
        "N": N,
        "d": d,
        "tau_max": tau_max,
        "steps": steps,
        "dt": dt,
        "beta": beta,
        "noise": noise,
        "seed_base": seed,
        "by_alpha": rows_out,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="v7 Phase II-A: τ 掃引・安定性プロキシ")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--d", type=int, default=6)
    p.add_argument("--tau-max", type=int, default=12)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--beta", type=float, default=0.85)
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument(
        "--alpha-list",
        type=str,
        default=None,
        help=(
            "カンマ区切りの alpha 一覧（指定時は各値で掃引し by_alpha を出力；"
            "単発の by_tau は出さない）"
        ),
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.alpha_list:
        raw = [x.strip() for x in args.alpha_list.split(",") if x.strip()]
        alphas = [float(x) for x in raw]
        if len(alphas) < 2:
            err = {"error": "alpha_list_need_at_least_2"}
            print(json.dumps(err), file=sys.stderr)
            raise SystemExit(2)
        payload = run_alpha_sweep(
            alphas=alphas,
            tau_max=args.tau_max,
            seed=args.seed,
            N=args.N,
            d=args.d,
            steps=args.steps,
            dt=args.dt,
            beta=args.beta,
            noise=args.noise,
        )
    else:
        payload = run_sweep(
            tau_max=args.tau_max,
            seed=args.seed,
            N=args.N,
            d=args.d,
            steps=args.steps,
            dt=args.dt,
            alpha=args.alpha,
            beta=args.beta,
            noise=args.noise,
        )
    js = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js, encoding="utf-8")
    if "by_tau" in payload:
        meta = {"taus": len(payload["by_tau"])}
    else:
        meta = {"alphas": len(payload.get("by_alpha") or [])}
    print("v7_phase2a_ok", json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
