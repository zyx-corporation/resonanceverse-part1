"""
7B 審判 ×2 と 3B 審判 ×1 を同一入力スライスで連続実行し、
ペア一致（7b_a–7b_b / 7b_a–3b / 7b_b–3b）の JSON をまとめる。

**既定**: 入力は MRMP 整形 ``windows.jsonl``、**先頭 100 行**（``--n`` / ``--offset`` で変更）。

1000 行に増やす例::

    python experiments/v7_slm_judge_triple_run.py --n 1000

CI / 配線確認（HF 不要）::

    python experiments/v7_slm_judge_triple_run.py --demo \\
      --jsonl experiments/data/v7_mrmp_sample.jsonl --n 3

サンプル 3 行だけでも ``--n 100`` したい場合（同一内容を id だけ変えて繰り返し）::

    python experiments/v7_slm_judge_triple_run.py --demo \\
      --jsonl experiments/data/v7_mrmp_sample.jsonl --n 100 \\
      --repeat-input-to-n

``triple_run_summary.json`` の ``wall_clock_s`` に、各サブプロセス秒数と ``total_wall_s`` を記録する。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.jsonl_slice import count_nonempty_lines  # noqa: E402

DEFAULT_7B_A = "tokyotech-llm/Swallow-7b-instruct-hf"
DEFAULT_7B_B = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_3B = "Qwen/Qwen2.5-3B-Instruct"
JUDGE_SCRIPT = _ROOT / "experiments" / "v7_phase1a_llm_judge_six_axes.py"
PAIR_SCRIPT = _ROOT / "experiments" / "v7_llm_judge_slm_pair_agreement.py"


def _run_cmd(argv: list[str], *, cwd: Path) -> float:
    """サブプロセスを実行し、経過秒（``perf_counter``）を返す。"""
    print("+", " ".join(argv), file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    subprocess.run(argv, cwd=str(cwd), check=True)
    return time.perf_counter() - t0


def _load_pair_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"error": "missing", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_all_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _write_cycled_jsonl(
    pool: list[dict[str, Any]],
    n: int,
    dest: Path,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as out:
        for i in range(n):
            row = json.loads(json.dumps(pool[i % len(pool)]))
            oid = str(row.get("id", "")).strip()
            row["id"] = f"{oid}__cycle_{i:05d}" if oid else f"cycle_{i:05d}"
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="7B×2 + 3B×1 SLM 審判の連続実行と pair_agreement 3 本"
    )
    _def_jsonl = (
        _ROOT / "experiments" / "logs" / "mrmp_prepared" / "windows.jsonl"
    )
    p.add_argument(
        "--jsonl",
        type=Path,
        default=_def_jsonl,
        help="審判入力 JSONL（text 必須・MRMP 窓推奨）",
    )
    p.add_argument(
        "--n",
        type=int,
        default=100,
        help="処理行数（MRMP windows.jsonl 先頭から max-rows 行）",
    )
    p.add_argument("--offset", type=int, default=0, help="入力先頭スキップ行数")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "experiments" / "logs" / "slm_judge_triple_mrmp_n100",
        help="審判 JSONL・一致 JSON・サマリの出力ディレクトリ",
    )
    p.add_argument("--hf-7b-a", default=DEFAULT_7B_A, help="審判モデル 7B（1 本目）")
    p.add_argument("--hf-7b-b", default=DEFAULT_7B_B, help="審判モデル 7B（2 本目）")
    p.add_argument("--hf-3b", default=DEFAULT_3B, help="審判モデル 3B")
    p.add_argument(
        "--demo",
        action="store_true",
        help="v7_phase1a_llm_judge_six_axes.py --demo（HF 不使用）",
    )
    p.add_argument("--cpu", action="store_true", help="hf_local 時 CPU 強制")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--hf-max-new-tokens", type=int, default=256)
    p.add_argument(
        "--skip-judge",
        action="store_true",
        help="審判を飛ばし（既存 JSONL がある前提で）一致集計のみ",
    )
    p.add_argument(
        "--only-compare",
        action="store_true",
        help="--skip-judge と同義（後方互換）",
    )
    p.add_argument(
        "--repeat-input-to-n",
        action="store_true",
        help=(
            "行数が offset+n 未満のとき、読み込んだ行を順に循環させ "
            "n 行の一時 JSONL を out-dir に書き、それを審判入力にする"
            "（公式サンプル 3 行で --n 100 など）"
        ),
    )
    args = p.parse_args()
    t_wall0 = time.perf_counter()
    wall_clock_s: dict[str, float] = {}

    jsonl_path = args.jsonl.resolve()
    if not jsonl_path.is_file():
        print(
            json.dumps(
                {"error": "jsonl_not_found", "path": str(jsonl_path)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    n_nonempty_raw = count_nonempty_lines(jsonl_path)
    need = args.offset + args.n
    cycled_effective_path: Path | None = None
    judge_jsonl = jsonl_path
    judge_offset = args.offset
    judge_n = args.n

    if n_nonempty_raw < need:
        if not args.repeat_input_to_n:
            print(
                json.dumps(
                    {
                        "error": "insufficient_jsonl_lines",
                        "nonempty_lines": n_nonempty_raw,
                        "offset": args.offset,
                        "n": args.n,
                        "need": need,
                        "hint_ja": "v7_mrmp_prepare で windows.jsonl を生成するか、"
                        "--n / --offset を減らすか、"
                        "同一内容の負荷試験なら --repeat-input-to-n を付ける。",
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            sys.exit(2)
        all_rows = _load_all_jsonl_rows(jsonl_path)
        if args.offset >= len(all_rows):
            print(
                json.dumps(
                    {
                        "error": "repeat_input_offset_out_of_range",
                        "offset": args.offset,
                        "rows_in_file": len(all_rows),
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            sys.exit(2)
        pool = all_rows[args.offset :]
        out_dir_early = args.out_dir.resolve()
        out_dir_early.mkdir(parents=True, exist_ok=True)
        cycled_effective_path = out_dir_early / "effective_input_cycled.jsonl"
        _write_cycled_jsonl(pool, args.n, cycled_effective_path)
        judge_jsonl = cycled_effective_path.resolve()
        judge_offset = 0
        judge_n = args.n

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "judge_7b_a": out_dir / "judge_7b_a.jsonl",
        "judge_7b_b": out_dir / "judge_7b_b.jsonl",
        "judge_3b": out_dir / "judge_3b.jsonl",
    }
    summaries = {
        "judge_7b_a": out_dir / "judge_7b_a_summary.json",
        "judge_7b_b": out_dir / "judge_7b_b_summary.json",
        "judge_3b": out_dir / "judge_3b_summary.json",
    }
    pair_out = {
        "7b_a_vs_7b_b": out_dir / "pair_agreement_7b_a_7b_b.json",
        "7b_a_vs_3b": out_dir / "pair_agreement_7b_a_3b.json",
        "7b_b_vs_3b": out_dir / "pair_agreement_7b_b_3b.json",
    }

    skip_judge = args.skip_judge or args.only_compare
    py = sys.executable

    if not skip_judge:
        jobs: list[tuple[str, str, Path, Path]] = [
            (
                "judge_7b_a",
                args.hf_7b_a,
                paths["judge_7b_a"],
                summaries["judge_7b_a"],
            ),
            (
                "judge_7b_b",
                args.hf_7b_b,
                paths["judge_7b_b"],
                summaries["judge_7b_b"],
            ),
            (
                "judge_3b",
                args.hf_3b,
                paths["judge_3b"],
                summaries["judge_3b"],
            ),
        ]
        for label, hf_model, out_jsonl, out_summary in jobs:
            cmd = [
                py,
                str(JUDGE_SCRIPT),
                "--jsonl",
                str(judge_jsonl),
                "--offset",
                str(judge_offset),
                "--max-rows",
                str(judge_n),
                "--out-jsonl",
                str(out_jsonl),
                "--out-summary",
                str(out_summary),
                "--seed",
                str(args.seed),
                "--temperature",
                str(args.temperature),
            ]
            if args.demo:
                cmd.append("--demo")
            else:
                cmd.extend(
                    [
                        "--provider",
                        "hf_local",
                        "--hf-model",
                        hf_model,
                        "--hf-max-new-tokens",
                        str(args.hf_max_new_tokens),
                    ]
                )
                if args.cpu:
                    cmd.append("--cpu")
            wall_clock_s[f"subprocess_{label}_s"] = _run_cmd(cmd, cwd=_ROOT)

    for key, po in pair_out.items():
        if key == "7b_a_vs_7b_b":
            pa, pb = paths["judge_7b_a"], paths["judge_7b_b"]
        elif key == "7b_a_vs_3b":
            pa, pb = paths["judge_7b_a"], paths["judge_3b"]
        else:
            pa, pb = paths["judge_7b_b"], paths["judge_3b"]
        cmd = [
            py,
            str(PAIR_SCRIPT),
            "--jsonl-a",
            str(pa),
            "--jsonl-b",
            str(pb),
            "--out-json",
            str(po),
        ]
        wall_clock_s[f"subprocess_pair_{key}_s"] = _run_cmd(cmd, cwd=_ROOT)

    wall_clock_s["total_wall_s"] = time.perf_counter() - t_wall0
    for k, v in list(wall_clock_s.items()):
        wall_clock_s[k] = round(float(v), 4)

    master: dict[str, Any] = {
        "schema_version": "v7_slm_judge_triple_run.v1",
        "demo": bool(args.demo),
        "n": int(args.n),
        "offset_cli": int(args.offset),
        "judge_max_rows": int(judge_n),
        "judge_offset": int(judge_offset),
        "input_jsonl_source": str(jsonl_path),
        "input_nonempty_lines_raw": int(n_nonempty_raw),
        "judge_input_jsonl": str(judge_jsonl),
        "repeat_input_cycled": bool(cycled_effective_path is not None),
        "effective_input_cycled_path": (
            str(cycled_effective_path) if cycled_effective_path else None
        ),
        "hf_7b_a": None if args.demo else args.hf_7b_a,
        "hf_7b_b": None if args.demo else args.hf_7b_b,
        "hf_3b": None if args.demo else args.hf_3b,
        "outputs": {k: str(v) for k, v in paths.items()},
        "pair_agreement": {
            k: {
                "path": str(v),
                "summary": _load_pair_summary(v),
            }
            for k, v in pair_out.items()
        },
        "wall_clock_s": wall_clock_s,
    }
    master_path = out_dir / "triple_run_summary.json"
    master_path.write_text(
        json.dumps(master, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        "v7_slm_judge_triple_run_ok",
        json.dumps(
            {
                "out_dir": str(out_dir),
                "summary": str(master_path),
                "n": args.n,
                "demo": args.demo,
                "total_wall_s": wall_clock_s.get("total_wall_s"),
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
