"""
RVT 実験プラン（``build_plan`` 出力）を**ホワイトリスト**に基づき順に実行。

- **既定は表示のみ**（``dry_run=True``）。``dry_run=False`` で実際に subprocess。
- HF 取得・GPU が必要なステップあり。**CI では dry-run のみ推奨**。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
SKIP_EXECUTE_IDS = frozenset({"rvt_l2_shell"})


def _repo_root() -> Path:
    return _ROOT.resolve()


def _argv_for_step(step: dict[str, Any], repo: Path) -> list[str] | None:
    sid = str(step.get("id", ""))
    hint = step.get("args_hint") or {}
    if sid == "rvt_explore":
        return ["bash", str(repo / "experiments/run_rvt_explore.sh")]
    if sid == "rvt_mrmp_row_batch":
        _dj = "experiments/logs/mrmp_prepared/windows.jsonl"
        jsonl = str(hint.get("jsonl", _dj))
        line = str(hint.get("line", 0))
        max_rows = str(hint.get("max_rows", 10))
        model = str(hint.get("model", "gpt2"))
        cmd = [
            sys.executable,
            str(repo / "experiments/rvt_exp_2026_008_mrmp_row.py"),
            "--jsonl",
            jsonl,
            "--line",
            line,
            "--max-rows",
            max_rows,
            "--model",
            model,
            "--cpu",
        ]
        if hint.get("accumulate_awai"):
            cmd.append("--accumulate-awai")
        l2_mode = hint.get("rvt_l2_mode")
        if l2_mode is not None and str(l2_mode) != "base":
            cmd.extend(["--rvt-l2-mode", str(l2_mode)])
            cmd.extend(
                ["--rvt-l2-eps", str(hint.get("rvt_l2_eps", 0.05))],
            )
            if hint.get("rvt_l2_all_layers"):
                cmd.append("--rvt-l2-all-layers")
        return cmd
    if sid == "rvt_l2_smoke":
        mode = str(hint.get("mode", "sym"))
        return [
            sys.executable,
            str(repo / "experiments/rvt_exp_2026_008_l2_smoke.py"),
            "--cpu",
            "--mode",
            mode,
        ]
    if sid == "rvt_l3_oboro_lite":
        prof = str(hint.get("oboro_profile", "full"))
        if hint.get("oboro_demo"):
            return [
                sys.executable,
                str(repo / "experiments/rvt_exp_2026_008_oboro_generate.py"),
                "--demo",
                "--profile",
                prof,
            ]
        cmd = [
            sys.executable,
            str(repo / "experiments/rvt_exp_2026_008_oboro_generate.py"),
            "--cpu",
            "--max-new-tokens",
            str(hint.get("max_new_tokens", 8)),
            "--prompt",
            str(hint.get("prompt", "Hello")),
            "--profile",
            prof,
        ]
        return cmd
    return None


def run_ablation_plan(
    plan: dict[str, Any],
    *,
    dry_run: bool = True,
    continue_on_error: bool = False,
) -> int:
    """
    ``steps`` を順に実行。未知の ``id`` はスキップ（終了コード 0、stderr に警告）。

    戻り値: すべて成功で 0、失敗があれば 1（``continue_on_error`` 時は最後の非ゼロ）。
    """
    repo = _repo_root()
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    last_rc = 0
    failures = 0
    for step in plan.get("steps", []):
        sid = str(step.get("id", ""))
        if sid in SKIP_EXECUTE_IDS:
            continue
        argv = _argv_for_step(step, repo)
        if argv is None:
            print(
                f"[rvt_plan_execute] skip unknown id={sid!r}",
                file=sys.stderr,
            )
            continue
        if dry_run:
            print("[dry-run]", shlex_join_compat(argv))
            continue
        print("[exec]", shlex_join_compat(argv), flush=True)
        rc = subprocess.run(argv, cwd=str(repo)).returncode
        if rc != 0:
            failures += 1
            last_rc = rc
            if not continue_on_error:
                return rc
    if failures:
        return last_rc if last_rc != 0 else 1
    return 0


def shlex_join_compat(argv: Sequence[str]) -> str:
    try:
        import shlex

        return shlex.join(list(argv))
    except AttributeError:
        return " ".join(argv)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="RVT ablation plan executor")
    p.add_argument("--plan-json", type=Path, required=True)
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="コマンドを表示のみ（既定）",
    )
    p.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="実際に subprocess 実行",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="1 ステップ失敗でも続行",
    )
    args = p.parse_args()
    raw = args.plan_json.read_text(encoding="utf-8")
    plan = json.loads(raw)
    raise SystemExit(
        run_ablation_plan(
            plan,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )
    )


if __name__ == "__main__":
    main()
