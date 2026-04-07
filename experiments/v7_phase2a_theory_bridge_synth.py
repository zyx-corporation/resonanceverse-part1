"""
Phase II-A と理論 v7.0 の「橋」— 合成離散テンソル 1 JSON バンドル。

橋ドキュメント docs/planning/v7_phase2a_theory_bridge.md §4 の
「数値実験を 1 本に固定」するため、`v7_phase2a_delay_sweep` の単発掃引と
alpha スイープをまとめ、ラベル規約・参照パス・再現コマンドを同梱する。

MRMP 実データの R(τ) とは別物（設計書 τ*_exp でもない）。
`single_tau_sweep.design_bridge_ja` を参照。
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

from experiments.v7_phase2a_delay_sweep import (  # noqa: E402
    run_alpha_sweep,
    run_sweep,
)


def build_theory_bridge_bundle(
    *,
    demo: bool = False,
    tau_max: int = 8,
    steps: int = 2000,
    seed: int = 0,
    N: int = 12,
    d: int = 6,
    dt: float = 0.05,
    alpha: float = 0.15,
    beta: float = 0.85,
    noise: float = 0.02,
    alphas: tuple[float, ...] = (0.1, 0.15, 0.2, 0.25, 0.3),
    single_tau_sweep: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Parameters
    ----------
    demo:
        True のとき τ 範囲・ステップ・N を抑えて CI 向けに高速化する
        （`single_tau_sweep` を渡した場合はスキップし、既存結果をそのまま使う）。
    single_tau_sweep:
        `run_sweep` の戻り（`v7_phase2a.v1`）を渡すと単発掃引を省略し、
        `v7_run_suite` 等で二重計算を避ける。
    """
    if single_tau_sweep is None:
        if demo:
            tau_max = min(tau_max, 6)
            steps = min(steps, 800)
            N = min(N, 8)
            d = min(d, 6)
            alphas = tuple(alphas[:4]) if len(alphas) >= 4 else alphas

        single = run_sweep(
            tau_max=tau_max,
            seed=seed,
            N=N,
            d=d,
            steps=steps,
            dt=dt,
            alpha=alpha,
            beta=beta,
            noise=noise,
        )
    else:
        single = single_tau_sweep
        if single.get("schema_version") != "v7_phase2a.v1":
            raise ValueError(
                "single_tau_sweep must be v7_phase2a.v1 run_sweep output",
            )
        by_tau = single.get("by_tau")
        if not isinstance(by_tau, list) or not by_tau:
            raise ValueError("single_tau_sweep.by_tau must be non-empty list")
        tau_max = len(by_tau) - 1
        N = int(single["N"])
        d = int(single["d"])
        steps = int(single["steps"])
        dt = float(single["dt"])
        alpha = float(single["alpha"])
        beta = float(single["beta"])
        noise = float(single["noise"])
        if demo:
            alphas = tuple(alphas[:4]) if len(alphas) >= 4 else alphas
    sens = run_alpha_sweep(
        alphas=list(alphas),
        tau_max=tau_max,
        seed=seed + 1,
        N=N,
        d=d,
        steps=steps,
        dt=dt,
        beta=beta,
        noise=noise,
    )

    cmd_base = (
        f"python experiments/v7_phase2a_delay_sweep.py --tau-max {tau_max} "
        f"--steps {steps} --seed {seed} --N {N} --d {d} "
        f"--dt {dt} --alpha {alpha} --beta {beta} --noise {noise}"
    )
    alpha_csv = ",".join(str(a) for a in alphas)
    cmd_alpha = (
        f"python experiments/v7_phase2a_delay_sweep.py --tau-max {tau_max} "
        f"--steps {steps} --seed {seed} --N {N} --d {d} "
        f"--dt {dt} --beta {beta} --noise {noise} "
        f'--alpha-list {alpha_csv}'
    )
    out_tau = " --out experiments/logs/phase2a_synth_tau.json"
    out_alpha = " --out experiments/logs/phase2a_synth_alpha_sweep.json"

    return {
        "schema_version": "v7_phase2a_theory_bridge_synth.v1",
        "mode": "theory_bridge_synthetic_bundle",
        "demo": bool(demo),
        "purpose_ja": (
            "設計書 II-A 型の完全な τ*_exp 手続きの代替ではなく、"
            "離散反対称テンソル＋遅延結合における振動代理の τ 掃引と "
            "alpha（強凸性代理）感度を 1 JSON にまとめたもの。"
            " MRMP の R(τ) や理論 τ* との同一視禁止。"
            " 橋ドキュメント §3–5。"
        ),
        "references": {
            "theory_bridge_md": "docs/planning/v7_phase2a_theory_bridge.md",
            "numeric_tau_md": "docs/planning/v7_phase2a_numeric_tau_exp.md",
            "delay_sweep_py": "experiments/v7_phase2a_delay_sweep.py",
            "experimental_design_md": (
                "docs/v7/Resonanceverse_v7.0_Experimental_Design.md"
            ),
        },
        "comparison_protocol_note_ja": (
            "設計書 II-A の τ*_exp と理論 τ* の照合・乖離基準は上記 experimental_design "
            "の手続きに従う。本バンドルの数値は振動代理であり、その代替にはならない。"
        ),
        "labeling_conventions_ja": {
            "tau_star_corpus_proxy": (
                "MRMP 実データで機械的に取った τ 候補（探索的）。"
                "本 JSON には含めない。"
            ),
            "tau_exp_proxy_oscillation_jump": (
                "本スクリプト内の single_tau_sweep と delay_sweep が返す "
                "振動スコア急増の τ。設計書の τ*_exp ではない。"
            ),
            "theoretical_tau_star": (
                "理論正本・導出からの τ*。本 JSON は数値比較の素材のみ。"
            ),
        },
        "reproduce_commands": [
            (
                "python experiments/v7_phase2a_theory_bridge_synth.py "
                + ("--demo " if demo else "")
                + "--out experiments/logs/v7_phase2a_theory_bridge_synth.json"
            ),
            cmd_base + out_tau,
            cmd_alpha + out_alpha,
        ],
        "hyperparams": {
            "tau_max": tau_max,
            "steps": steps,
            "seed": seed,
            "N": N,
            "d": d,
            "dt": dt,
            "alpha_single_sweep": alpha,
            "beta": beta,
            "noise": noise,
            "alphas_sensitivity": list(alphas),
        },
        "single_tau_sweep": single,
        "alpha_sensitivity": sens,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="v7 Phase II-A: 理論橋用 合成バンドル（τ 掃引＋α 感度）1 JSON",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="CI 向けに τ_max・steps・N を抑える",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tau-max", type=int, default=8)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    payload = build_theory_bridge_bundle(
        demo=args.demo,
        tau_max=args.tau_max,
        steps=args.steps,
        seed=args.seed,
    )
    js = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js, encoding="utf-8")
    meta = {
        "demo": payload["demo"],
        "taus": len(payload["single_tau_sweep"]["by_tau"]),
    }
    print("v7_phase2a_theory_bridge_ok", json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
