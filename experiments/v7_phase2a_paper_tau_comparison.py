"""
論文向け Phase II-A: τ 指標の比較表（JSON / 任意 MD）。

同一ハイパラで delay_sweep（振動代理）と Lyapunov スタブを走らせ、
行ごとに τ 値・設計書 §3.1 の乖離基準（20% / 50%）用の相対誤差を並べる。

理論 τ* の数値はリポジトリが閉形式で与えない。`--theoretical-tau-star` で
論文用の参照値を注入する。未指定時は理論行は null・乖離は出さない。

MRMP の tau_star_corpus_proxy は本スクリプトでは実行せず、表にプレースホルダ行のみ。
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

from experiments.v7_phase2a_delay_sweep import run_sweep  # noqa: E402
from experiments.v7_phase2a_tau_exp_lyapunov_stub import (  # noqa: E402
    run_lyapunov_tau_exp_stub_sweep,
)


def relative_error_percent_versus_theory(
    empirical: int | float | None,
    theory: float | None,
) -> float | None:
    """100 * |τ_emp - τ_theory| / max(|τ_theory|, ε)。"""
    if empirical is None or theory is None:
        return None
    te = float(theory)
    if abs(te) < 1e-12:
        return None
    return float(abs(float(empirical) - te) / abs(te) * 100.0)


def design_criterion_verdict_ja(rel_pct: float | None) -> str | None:
    """設計書 §3.1 の乖離基準に照らした短文（理論値が無いときは None）。"""
    if rel_pct is None:
        return None
    if rel_pct <= 20.0:
        return "乖離20%以内（設計書: 定理3.3の数値的支持ありの帯）"
    if rel_pct <= 50.0:
        return "20%超〜50%以内（中間帯; 解釈は慎重に）"
    return "乖離50%超（設計書: V_K 構成の見直しを検討する帯）"


def build_paper_tau_comparison_bundle(
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
    theoretical_tau_star: float | None,
    theoretical_provenance_ja: str,
    lyapunov_burn_frac: float,
    lyapunov_mean_dv_threshold: float,
    lyapunov_frac_positive_threshold: float,
) -> dict[str, Any]:
    sweep = run_sweep(
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
    ly = run_lyapunov_tau_exp_stub_sweep(
        tau_max=tau_max,
        steps=steps,
        seed=seed,
        N=N,
        d=d,
        dt=dt,
        alpha=alpha,
        beta=beta,
        noise=noise,
        burn_frac=lyapunov_burn_frac,
        mean_dv_threshold=lyapunov_mean_dv_threshold,
        frac_positive_threshold=lyapunov_frac_positive_threshold,
    )

    tau_osc = int(sweep["tau_exp_proxy_oscillation_jump"])
    tau_lv_mean = ly["tau_exp_numeric_stub_mean_dV"]
    tau_lv_frac = ly["tau_exp_numeric_stub_frac_positive"]

    theory = theoretical_tau_star

    def row(
        row_id: str,
        label_ja: str,
        label_en: str,
        tau_val: int | float | None,
        figure_role: str,
        notes_ja: str,
    ) -> dict[str, Any]:
        r: dict[str, Any] = {
            "row_id": row_id,
            "label_ja": label_ja,
            "label_en": label_en,
            "tau": tau_val,
            "figure_role": figure_role,
            "notes_ja": notes_ja,
        }
        rep = relative_error_percent_versus_theory(tau_val, theory)
        r["discrepancy_vs_theory_percent"] = rep
        r["design_criterion_verdict_ja"] = design_criterion_verdict_ja(rep)
        return r

    paper_table_rows: list[dict[str, Any]] = [
        row(
            "theoretical_tau_star",
            "理論 τ*（参照値・注入）",
            "Theoretical τ* (reference, user-supplied)",
            theory,
            "theory_reference",
            theoretical_provenance_ja,
        ),
        row(
            "tau_exp_numeric_stub_mean_dV",
            "数値スタブ（尾平均 ΔV>閾値の最小 τ）",
            "Numeric stub (min τ with positive mean ΔV tail)",
            tau_lv_mean,
            "synthetic_lyapunov_proxy",
            ly["tau_exp_labeling_ja"],
        ),
        row(
            "tau_exp_numeric_stub_frac_positive",
            "数値スタブ（尾 ΔV>0 割合>閾値の最小 τ）",
            "Numeric stub (min τ with fraction ΔV>0)",
            tau_lv_frac,
            "synthetic_lyapunov_proxy_alt",
            ly["tau_exp_labeling_ja"],
        ),
        row(
            "tau_exp_proxy_oscillation_jump",
            "振動代理（oscillation_score 隣接差最大の τ）",
            "Oscillation proxy (largest jump in oscillation_score)",
            tau_osc,
            "synthetic_oscillation_proxy",
            sweep["design_bridge_ja"],
        ),
        {
            "row_id": "tau_star_corpus_proxy",
            "label_ja": "コーパス代理（MRMP・探索的）",
            "label_en": "Corpus proxy (MRMP, exploratory)",
            "tau": None,
            "figure_role": "empirical_corpus_placeholder",
            "notes_ja": (
                "本表では未実行。`v7_phase2a_empirical_run.py` の JSON から "
                "tau_star_candidate / 図上の峰を別表で記載すること。"
            ),
            "discrepancy_vs_theory_percent": None,
            "design_criterion_verdict_ja": None,
        },
    ]

    return {
        "schema_version": "v7_phase2a_paper_tau_comparison.v1",
        "mode": "paper_comparison_table",
        "design_doc_pointer": (
            "docs/v7/Resonanceverse_v7.0_Experimental_Design.md §3.1"
        ),
        "design_discrepancy_criteria_ja": (
            "理論 τ* と実験 τ の相対乖離が 20% 以内なら「定理3.3の数値的支持あり」、"
            "50% を超える場合はリアプノフ=クラソフスキー汎関数の構成見直しを検討、"
            "と設計書に記載。"
        ),
        "theoretical_tau_star_injected": theory,
        "theoretical_provenance_ja": theoretical_provenance_ja,
        "hyperparams": {
            "tau_max": tau_max,
            "steps": steps,
            "seed": seed,
            "N": N,
            "d": d,
            "dt": dt,
            "alpha": alpha,
            "beta": beta,
            "noise": noise,
        },
        "sources": {
            "delay_sweep": sweep,
            "lyapunov_stub": ly,
        },
        "paper_table_rows": paper_table_rows,
    }


def render_markdown_table(bundle: dict[str, Any]) -> str:
    """キャプション用コピペ向けの簡易 Markdown 表。"""
    lines = [
        "| row_id | τ | discrepancy_vs_theory_% | figure_role |",
        "|--------|---|-------------------------|-------------|",
    ]
    for r in bundle.get("paper_table_rows") or []:
        rid = r.get("row_id", "")
        tv = r.get("tau")
        tau_s = "" if tv is None else str(tv)
        dp = r.get("discrepancy_vs_theory_percent")
        dps = "" if dp is None else f"{dp:.2f}"
        fr = r.get("figure_role", "")
        lines.append(f"| `{rid}` | {tau_s} | {dps} | `{fr}` |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(
        description="v7 Phase II-A: 論文向け τ 比較表 JSON（＋任意 MD）",
    )
    p.add_argument("--demo", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tau-max", type=int, default=10)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--d", type=int, default=6)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--beta", type=float, default=0.85)
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument(
        "--theoretical-tau-star",
        type=float,
        default=None,
        help="論文用の理論 τ* 参照値（未指定なら理論行は null）",
    )
    p.add_argument(
        "--theoretical-provenance-ja",
        type=str,
        default="",
        help="理論値の出所（論文・別導出ノート等）。未指定時は定型文",
    )
    p.add_argument("--mean-dv-threshold", type=float, default=1e-5)
    p.add_argument("--frac-positive-threshold", type=float, default=0.52)
    p.add_argument("--burn-frac", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="比較表の Markdown を別ファイルへ",
    )
    args = p.parse_args()

    tau_max = args.tau_max
    steps = args.steps
    N, d = args.N, args.d
    if args.demo:
        tau_max = min(tau_max, 5)
        steps = min(steps, 600)
        N = min(N, 8)
        d = min(d, 4)

    prov = (args.theoretical_provenance_ja or "").strip()
    if not prov:
        prov = (
            "未記載。論文・別稿の閉形式・数値導出から "
            "`--theoretical-tau-star` で注入した値。"
        )

    bundle = build_paper_tau_comparison_bundle(
        tau_max=tau_max,
        steps=steps,
        seed=args.seed,
        N=N,
        d=d,
        dt=args.dt,
        alpha=args.alpha,
        beta=args.beta,
        noise=args.noise,
        theoretical_tau_star=args.theoretical_tau_star,
        theoretical_provenance_ja=prov,
        lyapunov_burn_frac=args.burn_frac,
        lyapunov_mean_dv_threshold=args.mean_dv_threshold,
        lyapunov_frac_positive_threshold=args.frac_positive_threshold,
    )

    js = json.dumps(bundle, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js, encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(
            render_markdown_table(bundle),
            encoding="utf-8",
        )
    meta = {
        "rows": len(bundle["paper_table_rows"]),
        "theory": bundle["theoretical_tau_star_injected"],
    }
    tag = "v7_phase2a_paper_tau_comparison_ok"
    print(tag, json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
