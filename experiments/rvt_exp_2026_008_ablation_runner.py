"""
RVT 実験 D: フラグから推奨ステップ列（JSON）を生成する。

**既定は JSON のみ**。``--execute`` で `rvt_exp_2026_008_plan_execute` 相当（
``--no-dry-run`` で本実行・HF 要）。
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

from experiments.rvt_exp_2026_008_ablation_flags import (  # noqa: E402
    RvtExp008ConditionFlags,
)
from experiments.v7_phase2a_rail_metadata import (  # noqa: E402
    E_RVT_008_ABLATION_PLAN,
    with_rail,
)


def build_plan(
    flags: RvtExp008ConditionFlags,
    *,
    prepend_rvt_explore: bool = False,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []

    if prepend_rvt_explore:
        steps.append(
            {
                "id": "rvt_explore",
                "script": "experiments/run_rvt_explore.sh",
                "purpose_ja": (
                    "Day0（任意）→ 小バッチ MRMP（既定 MAX_ROWS=5）。"
                    " JSONL 無しは exit 0"
                ),
                "args_hint": {
                    "JSONL": "experiments/logs/mrmp_prepared/windows.jsonl",
                    "SKIP_DAY0": "0",
                    "MAX_ROWS": 5,
                    "LINE": 0,
                },
                "shell_example": "bash experiments/run_rvt_explore.sh",
            }
        )

    if flags.w_asym_extract or flags.base_observe_only:
        mrmp_hint: dict[str, Any] = {
            "jsonl": "experiments/logs/mrmp_prepared/windows.jsonl",
            "line": 0,
            "max_rows": 10,
            "accumulate_awai": flags.awai_vector_on,
            "model": "gpt2",
        }
        if flags.attn_inject_wasym:
            mrmp_hint["rvt_l2_mode"] = "wasym"
            mrmp_hint["rvt_l2_eps"] = 0.05
        elif flags.attn_inject_sym:
            mrmp_hint["rvt_l2_mode"] = "sym"
            mrmp_hint["rvt_l2_eps"] = 0.05
        shell_ex = (
            "LINE=0 MAX_ROWS=10 CPU=1 ACCUMULATE_AWAI="
            + ("1" if flags.awai_vector_on else "0")
        )
        if flags.attn_inject_wasym:
            shell_ex += " RVT_L2_MODE=wasym RVT_L2_EPS=0.05"
        elif flags.attn_inject_sym:
            shell_ex += " RVT_L2_MODE=sym RVT_L2_EPS=0.05"
        shell_ex += " bash experiments/run_rvt_mrmp_batch.sh"
        steps.append(
            {
                "id": "rvt_mrmp_row_batch",
                "script": "experiments/rvt_exp_2026_008_mrmp_row.py",
                "purpose_ja": "MRMP 窓からヘッド Frobenius・6 軸代理（L1 拡張）",
                "args_hint": mrmp_hint,
                "shell_example": shell_ex,
            }
        )

    if flags.attn_inject_sym or flags.attn_inject_wasym:
        mode = "wasym" if flags.attn_inject_wasym else "sym"
        steps.append(
            {
                "id": "rvt_l2_smoke",
                "script": "experiments/rvt_exp_2026_008_l2_smoke.py",
                "purpose_ja": "eager Causal LM 注意ブレンドの logits 差分スモーク",
                "args_hint": {"mode": mode, "cpu": True},
                "shell_example": (
                    f"CPU=1 bash experiments/run_rvt_l2_smoke.sh --mode {mode}"
                ),
            }
        )
        steps.append(
            {
                "id": "rvt_l2_shell",
                "script": "experiments/run_rvt_l2_smoke.sh",
                "purpose_ja": "同上（環境変数 MODEL で Phase II-A 系 ID に差し替え可）",
                "args_hint": {"MODEL": "gpt2"},
                "shell_example": (
                    "CPU=1 MODEL=gpt2 bash experiments/run_rvt_l2_smoke.sh "
                    f"--mode {mode}"
                ),
            }
        )

    if flags.oboro_monitor_on or flags.longform_generation_task:
        steps.append(
            {
                "id": "rvt_l3_oboro_lite",
                "script": "experiments/rvt_exp_2026_008_oboro_generate.py",
                "purpose_ja": (
                    "貪欲デコード＋固着→温度。full プロファイルで計画書準拠トレース（v2）。"
                    " args_hint.oboro_demo=true で HF なし --demo（CI）。"
                ),
                "args_hint": {
                    "oboro_profile": "full",
                    "max_new_tokens": 8,
                    "prompt": "Hello",
                    "oboro_demo": False,
                },
                "shell_example": (
                    "python experiments/rvt_exp_2026_008_oboro_generate.py "
                    "--cpu --profile full --max-new-tokens 8 "
                    "--prompt 'Hello'\n"
                    "# HF なし: bash experiments/run_rvt_oboro_demo.sh"
                ),
            }
        )

    return {
        "schema_version": "rvt_exp_008_ablation_plan.v1",
        **with_rail(E_RVT_008_ABLATION_PLAN),
        "flags": flags.to_json(),
        "steps": steps,
        "note_ja": (
            "``rvt_exp_2026_008_plan_execute.py`` または "
            "``--execute``（``--no-dry-run`` で本実行）でホワイトリスト順に実行可。"
        ),
    }


def emit_shell_snippets(plan: dict[str, Any]) -> str:
    """プランの各ステップの ``shell_example`` をコメント付きで連結。"""
    lines: list[str] = []
    for s in plan.get("steps", []):
        sid = s.get("id", "")
        purpose = s.get("purpose_ja", "")
        ex = s.get("shell_example")
        lines.append(f"# {sid}: {purpose}")
        if ex:
            lines.append(ex)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="RVT ablation: flags → plan JSON")
    p.add_argument(
        "--preset",
        choices=(
            "explore",
            "observe",
            "observe_awai",
            "inject_sym",
            "full_minimal",
            "eight_grid",
        ),
        default="observe",
        help=(
            "代表的フラグ束。"
            "explore / eight_grid（8 条件 JSON 配列を出す）"
        ),
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--emit-shell",
        action="store_true",
        help="JSON のかわりに shell_example のみ標準出力",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="生成したプランを plan_execute で順に実行",
    )
    p.add_argument(
        "--no-dry-run",
        action="store_true",
        help="--execute 時に subprocess 本実行（要 HF 等）",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="--execute / --run-eight-grid 時に失敗しても続行",
    )
    p.add_argument(
        "--run-eight-grid",
        action="store_true",
        help="--preset eight_grid の各条件プランを順に plan_execute 相当で実行",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="--run-eight-grid 実行結果の集約 JSON（終了コード・条件 ID）",
    )
    p.add_argument(
        "--eight-grid-prepend-explore",
        action="store_true",
        help=(
            "--run-eight-grid 時、各条件のプラン先頭に "
            "rvt_explore（build_plan prepend）を付与"
        ),
    )
    args = p.parse_args()

    if args.run_eight_grid and args.preset != "eight_grid":
        p.error("--run-eight-grid は --preset eight_grid と併用してください")
    prep = bool(
        getattr(args, "eight_grid_prepend_explore", False),
    )
    if prep and not args.run_eight_grid:
        p.error(
            "--eight-grid-prepend-explore は "
            "--run-eight-grid と併用してください",
        )

    if args.preset == "eight_grid":
        from experiments.rvt_exp_2026_008_eight_conditions import (
            iter_eight_experiment_conditions,
        )

        if args.run_eight_grid:
            from experiments.rvt_exp_2026_008_plan_execute import (
                run_ablation_plan,
            )

            prepend_ex = bool(args.eight_grid_prepend_explore)
            manifest: dict[str, Any] = {
                "schema_version": "rvt_eight_grid_unattended.v1",
                "dry_run": not args.no_dry_run,
                "continue_on_error": bool(args.continue_on_error),
                "prepend_explore": prepend_ex,
                "conditions": [],
            }
            failed_any = False
            for cid, fl in iter_eight_experiment_conditions():
                plan = build_plan(fl, prepend_rvt_explore=prepend_ex)
                rc = run_ablation_plan(
                    plan,
                    dry_run=not args.no_dry_run,
                    continue_on_error=args.continue_on_error,
                )
                if rc != 0:
                    failed_any = True
                manifest["conditions"].append(
                    {
                        "condition_id": cid,
                        "exit_code": rc,
                        "n_steps": len(plan.get("steps", [])),
                    },
                )
                if rc != 0 and not args.continue_on_error:
                    break
            manifest["failed_any"] = failed_any
            if args.manifest:
                args.manifest.parent.mkdir(parents=True, exist_ok=True)
                args.manifest.write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            raise SystemExit(1 if failed_any else 0)

        grid: list[dict[str, Any]] = []
        for cid, fl in iter_eight_experiment_conditions():
            pl = build_plan(fl)
            grid.append({"condition_id": cid, **pl})
        txt = json.dumps(grid, indent=2, ensure_ascii=False)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(txt + "\n", encoding="utf-8")
        print(txt)
        return

    prepend_explore = args.preset == "explore"

    if args.preset in ("explore", "observe"):
        f = RvtExp008ConditionFlags(
            base_observe_only=True,
            w_asym_extract=True,
        )
    elif args.preset == "observe_awai":
        f = RvtExp008ConditionFlags(
            base_observe_only=True,
            w_asym_extract=True,
            awai_vector_on=True,
        )
    elif args.preset == "inject_sym":
        f = RvtExp008ConditionFlags(
            base_observe_only=False,
            w_asym_extract=True,
            attn_inject_sym=True,
        )
    else:
        f = RvtExp008ConditionFlags(
            base_observe_only=True,
            w_asym_extract=True,
            attn_inject_sym=True,
            awai_vector_on=True,
            oboro_monitor_on=True,
            longform_generation_task=True,
            full_eight_grid_slot=False,
        )

    plan = build_plan(f, prepend_rvt_explore=prepend_explore)
    if args.execute:
        from experiments.rvt_exp_2026_008_plan_execute import run_ablation_plan

        raise SystemExit(
            run_ablation_plan(
                plan,
                dry_run=not args.no_dry_run,
                continue_on_error=args.continue_on_error,
            )
        )
    if args.emit_shell:
        txt = emit_shell_snippets(plan)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(txt, encoding="utf-8")
        print(txt, end="")
    else:
        js = json.dumps(plan, indent=2, ensure_ascii=False)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(js + "\n", encoding="utf-8")
        print(js)


if __name__ == "__main__":
    main()
