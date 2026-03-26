"""
v7 Phase II-A: 遅延 τ の掃引と安定性プロキシ（離散時間・小規模テンソル）。

定理3.3 の V_K の厳密実装ではなく、遅延結合があるときのエネルギー
||W||_F の後半区間における変動係数（oscillation_score）を記録し、
τ を増やしたときの挙動差を見るハーネス。
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
    rng = np.random.default_rng(seed)
    w0 = rng.standard_normal((N, N, d)) * 0.08
    w0 = 0.5 * (w0 - np.transpose(w0, (1, 0, 2)))
    hist: list[np.ndarray] = [w0.copy() for _ in range(tau + 1)]

    energy: list[float] = []
    for _ in range(steps):
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
        energy.append(float(np.linalg.norm(hist[0])))

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
            np.linalg.norm(hist[0] - np.transpose(hist[0], (1, 0, 2)))
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
        "oscillation_scores": oscs,
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
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

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
    print("v7_phase2a_ok", json.dumps({"taus": len(payload["by_tau"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
