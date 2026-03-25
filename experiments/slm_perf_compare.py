#!/usr/bin/env python3
"""
同一ハイパーパラメータで「HF ベースライン（--baseline-hf）」と「AwaiIntegratedSLM」を順に実行し、
パフォーマンス指標を 1 つの JSON にまとめる（GPU / 別モデルでの導入前後比較用）。

例::

    HF_HOME=$PWD/.hf_cache python experiments/slm_perf_compare.py \\
      --model gpt2 --data random --max-steps 100 --batch 4 --seq-len 64 \\
      --out experiments/logs/slm_compare_gpt2.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _run_slm(extra: list[str]) -> dict:
    script = str(_ROOT / "experiments" / "slm_resonance_lm.py")
    cmd = [sys.executable, script, *extra]
    proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"slm_resonance_lm failed (exit {proc.returncode})\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith("slm_resonance_lm_ok"):
            return json.loads(line.split("slm_resonance_lm_ok", 1)[1].strip())
    raise RuntimeError("slm_resonance_lm_ok が出力にありません:\n" + proc.stdout)


def _common_flags(ns: argparse.Namespace) -> list[str]:
    out: list[str] = [
        "--model",
        ns.model,
        "--data",
        ns.data,
        "--max-steps",
        str(ns.max_steps),
        "--seed",
        str(ns.seed),
        "--batch",
        str(ns.batch),
        "--seq-len",
        str(ns.seq_len),
        "--lr",
        str(ns.lr),
    ]
    if ns.cpu:
        out.append("--cpu")
    if ns.device:
        out.extend(["--device", ns.device])
    if ns.freeze_base:
        out.append("--freeze-base")
    if ns.data == "wikitext":
        out.extend(
            ["--max-chars", str(ns.max_chars), "--eval-frac", str(ns.eval_frac)]
        )
    if ns.eval_ppl:
        out.append("--eval-ppl")
    return out


def _ratio_awai_over_baseline(sa: object, sb: object) -> float | None:
    if not isinstance(sa, (int, float)) or not isinstance(sb, (int, float)):
        return None
    if not sb or sb <= 0:
        return None
    return float(sa) / float(sb)


def main() -> None:
    p = argparse.ArgumentParser(
        description="HF baseline vs AwaiIntegratedSLM（slm_resonance_lm）",
    )
    p.add_argument("--model", default="gpt2")
    p.add_argument("--data", choices=("random", "wikitext"), default="random")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--freeze-base", action="store_true")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-chars", type=int, default=200_000)
    p.add_argument("--eval-frac", type=float, default=0.1)
    p.add_argument("--eval-ppl", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    base = _common_flags(args)
    baseline = _run_slm(base + ["--baseline-hf"])
    awai = _run_slm(base)

    tb = float(baseline.get("train_time_s") or 0.0)
    ta = float(awai.get("train_time_s") or 0.0)
    sb = baseline.get("steps_per_sec")
    sa = awai.get("steps_per_sec")

    merged = {
        "baseline_hf": baseline,
        "awai_resonance": awai,
        "comparison": {
            "train_time_ratio_baseline_over_awai": (tb / ta) if ta > 0 else None,
            "throughput_ratio_awai_over_baseline": _ratio_awai_over_baseline(sa, sb),
            "cuda_peak_memory_bytes": {
                "baseline": baseline.get("cuda_peak_memory_bytes"),
                "awai": awai.get("cuda_peak_memory_bytes"),
            },
        },
    }

    line = json.dumps(merged, ensure_ascii=False)
    print("slm_perf_compare_ok", line)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
