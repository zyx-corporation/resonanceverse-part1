"""
計画書の学習可能 M: 合成回復デモ → ``.npy`` 保存。

MRMP 行は ``--head-axis-matrix`` で供給する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.rvt_exp_2026_008_head_axis_matrix import (  # noqa: E402
    save_head_axis_matrix_npy,
    train_head_axis_m_supervised_jsonl,
    train_head_axis_m_synthetic,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="RVT: ヘッド→6軸 M（合成デモ or 審判教師あり JSONL）",
    )
    p.add_argument(
        "--supervised-jsonl",
        type=Path,
        default=None,
        help="mrmp 行 + per_head_block_frobenius + w_target または *_ab 6 軸",
    )
    p.add_argument(
        "--derive-target-ab",
        action="store_true",
        help="supervised-jsonl で trust_ab … history_ab から教師ベクトルを構成",
    )
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--out", type=Path, required=True, help="出力 .npy")
    p.add_argument("--meta-out", type=Path, default=None)
    args = p.parse_args()

    if args.supervised_jsonl is not None:
        try:
            m, meta = train_head_axis_m_supervised_jsonl(
                args.supervised_jsonl.resolve(),
                derive_target_ab=bool(args.derive_target_ab),
                steps=int(args.steps),
                lr=float(args.lr),
                seed=int(args.seed),
            )
        except ValueError as e:
            print(
                json.dumps({"error": str(e)}, ensure_ascii=False),
                file=sys.stderr,
            )
            sys.exit(2)
    else:
        if args.n_heads < 1 or args.n_samples < 8 or args.steps < 1:
            err = json.dumps(
                {"error": "invalid hyperparameters"},
                ensure_ascii=False,
            )
            print(err, file=sys.stderr)
            sys.exit(2)
        m, meta = train_head_axis_m_synthetic(
            n_heads=args.n_heads,
            n_samples=args.n_samples,
            steps=args.steps,
            seed=args.seed,
            lr=float(args.lr),
        )
    save_head_axis_matrix_npy(args.out, m)
    if args.meta_out:
        args.meta_out.parent.mkdir(parents=True, exist_ok=True)
        args.meta_out.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    out_js = json.dumps(
        {**meta, "out_npy": str(args.out)},
        indent=2,
        ensure_ascii=False,
    )
    print(out_js)


if __name__ == "__main__":
    main()
