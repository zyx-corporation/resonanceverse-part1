"""
2 本の審判済み JSONL（同一窓・異なる hf_local モデル想定）を ``id`` で突き合わせ、
12 軸ごとの Pearson r・平均絶対差・有効ペア数を JSON にまとめる。

入力はいずれも ``v7_phase1a_llm_judge_six_axes.py`` 出力相当（PILOT_KEYS が数値）。
人手金標準や OpenAI との一致ではなく、**モデル間ブレの探索用**。

例::

    python experiments/v7_llm_judge_slm_pair_agreement.py \\
      --jsonl-a experiments/logs/judge_swallow.jsonl \\
      --jsonl-b experiments/logs/judge_qwen.jsonl \\
      --out-json experiments/logs/judge_slm_pair_agreement.json
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

from experiments.v7_phase1a_llm_judge_six_axes import (  # noqa: E402
    judge_prompt_fingerprint_sha256,
)
from experiments.v7_phase1a_pilot_jsonl import PILOT_KEYS  # noqa: E402


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _load_jsonl_by_id(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """行 id -> 行 dict。重複 id は後勝ちし、warnings にメッセージを積む。"""
    by_id: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        rid = str(row.get("id", "")).strip()
        if not rid:
            rid = f"__line_{i}"
            warnings.append(f"{path}: line {i + 1} missing id, using {rid}")
        if rid in by_id:
            warnings.append(
                f"{path}: duplicate id {rid!r}, keeping last occurrence"
            )
        by_id[rid] = row
    return by_id, warnings


def _meta_sample(rows: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows.values():
        m = row.get("llm_judge_meta")
        if isinstance(m, dict):
            return {
                "model": m.get("model"),
                "provider": m.get("provider"),
                "prompt_template_id": m.get("prompt_template_id"),
                "hf_revision": m.get("hf_revision"),
            }
    return None


def run_pair_agreement(
    path_a: Path,
    path_b: Path,
) -> dict[str, Any]:
    rows_a, wa = _load_jsonl_by_id(path_a)
    rows_b, wb = _load_jsonl_by_id(path_b)
    ids_common = sorted(set(rows_a.keys()) & set(rows_b.keys()))

    xs: dict[str, list[float]] = {k: [] for k in PILOT_KEYS}
    ys: dict[str, list[float]] = {k: [] for k in PILOT_KEYS}
    skipped = 0

    for rid in ids_common:
        ra, rb = rows_a[rid], rows_b[rid]
        bad = False
        for k in PILOT_KEYS:
            va, vb = ra.get(k), rb.get(k)
            ok_a = isinstance(va, (int, float))
            ok_b = isinstance(vb, (int, float))
            if not ok_a or not ok_b:
                bad = True
                break
        if bad:
            skipped += 1
            continue
        for k in PILOT_KEYS:
            xs[k].append(float(ra[k]))
            ys[k].append(float(rb[k]))

    by_axis: dict[str, dict[str, Any]] = {}
    for k in PILOT_KEYS:
        xa = np.array(xs[k], dtype=np.float64)
        ya = np.array(ys[k], dtype=np.float64)
        n = int(xa.size)
        if n == 0:
            by_axis[k] = {"n": 0, "pearson_r": None, "mean_abs_diff": None}
            continue
        by_axis[k] = {
            "n": n,
            "pearson_r": _pearson_r(xa, ya),
            "mean_abs_diff": float(np.mean(np.abs(xa - ya))),
        }

    return {
        "schema_version": "v7_llm_judge_slm_pair_agreement.v1",
        "note_ja": (
            "SLM 同士の数値一致は探索用。"
            "人手・OpenAI との正当性は保証しない。"
        ),
        "prompt_fingerprint_sha256_expected": judge_prompt_fingerprint_sha256(),
        "path_a": str(path_a),
        "path_b": str(path_b),
        "n_rows_a": len(rows_a),
        "n_rows_b": len(rows_b),
        "n_ids_intersection": len(ids_common),
        "n_rows_used": len(xs[PILOT_KEYS[0]]),
        "n_rows_skipped_missing_or_nonnumeric": skipped,
        "load_warnings": wa + wb,
        "llm_judge_meta_sample_a": _meta_sample(rows_a),
        "llm_judge_meta_sample_b": _meta_sample(rows_b),
        "by_axis": by_axis,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="2 本の審判 JSONL の軸別一致（SLM 同士・探索）")
    p.add_argument("--jsonl-a", type=Path, required=True)
    p.add_argument("--jsonl-b", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()

    pa, pb = args.jsonl_a.resolve(), args.jsonl_b.resolve()
    if not pa.is_file() or not pb.is_file():
        print(json.dumps({"error": "input_not_found"}), file=sys.stderr)
        raise SystemExit(2)

    out = run_pair_agreement(pa, pb)
    for w in out.get("load_warnings") or []:
        print(f"v7_llm_judge_slm_pair_agreement_warn {w}", file=sys.stderr)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "v7_llm_judge_slm_pair_agreement_ok",
        json.dumps(
            {
                "n_used": out["n_rows_used"],
                "n_intersection": out["n_ids_intersection"],
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
