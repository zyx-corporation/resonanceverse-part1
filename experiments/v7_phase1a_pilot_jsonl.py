"""
v7 Phase I-A パイロット: JSONL（text + 数値ラベル）と最終層 ||S_asym||_F の相関。

- --demo: HF なし。text から決定論的合成 Frobenius（パイプライン検証）。
- 既定: HF で各 text を前向きし、最終層の frobenius_S_asym を特徴とする。

JSONL の各行: 必須 ``text``。数値ラベルは ``intent_ab``, ``trust_ab`` 等（下記 PILOT_KEYS）。
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

PILOT_KEYS = (
    "trust_ab",
    "trust_ba",
    "authority_ab",
    "authority_ba",
    "proximity_ab",
    "proximity_ba",
    "intent_ab",
    "intent_ba",
    "affect_ab",
    "affect_ba",
    "history_ab",
    "history_ba",
)

# MRMP 整形行（v7_mrmp_prepare.py）の正規化スコア（v7 の 6 軸ではない）
MRMP_LABEL_KEYS = (
    "mrmp_informativeness_01",
    "mrmp_comprehension_01",
    "mrmp_familiarity_01",
    "mrmp_interest_01",
    "mrmp_proactiveness_01",
    "mrmp_satisfaction_01",
)


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    return None if r != r else r


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_jsonl_slice(
    path: Path,
    offset: int,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    from experiments.jsonl_slice import iter_jsonl_slice

    return list(iter_jsonl_slice(path, offset, max_rows))


def run_pilot(
    *,
    rows: list[dict[str, Any]],
    demo: bool,
    model_name: str,
    cpu: bool,
    seed: int,
    layer_index: int,
    label_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    from experiments.v7_phase1a_phi_correlation import (
        extract_hf_attention_layer_stats,
        frobenius_s_asym_demo_from_text,
    )

    fros: list[float] = []
    kept_ids: list[str] = []
    kept_rows: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        if demo:
            fros.append(frobenius_s_asym_demo_from_text(text))
        else:
            stats, err, _ = extract_hf_attention_layer_stats(
                text=text,
                model_name=model_name,
                cpu=cpu,
                seed=seed,
            )
            if err is not None:
                return {**err, "pilot_partial": True}
            assert stats is not None
            li = layer_index if layer_index >= 0 else len(stats) - 1
            li = max(0, min(li, len(stats) - 1))
            fros.append(float(stats[li]["frobenius_S_asym"]))
        kept_ids.append(str(row.get("id", "")))
        kept_rows.append(row)

    keys = label_keys if label_keys is not None else PILOT_KEYS
    correlations: dict[str, dict[str, Any]] = {}
    for key in keys:
        ys: list[float] = []
        fs: list[float] = []
        for row, f in zip(kept_rows, fros):
            v = row.get(key)
            if isinstance(v, (int, float)):
                ys.append(float(v))
                fs.append(f)
        if len(ys) < 3:
            correlations[key] = {"n": len(ys), "pearson_r": None}
            continue
        r = _pearson(np.array(fs), np.array(ys))
        correlations[key] = {"n": len(ys), "pearson_r": r}

    return {
        "schema_version": "v7_phase1a_pilot.v1",
        "mode": "demo_synthetic_fro" if demo else "hf_last_layer_fro",
        "model": model_name if not demo else None,
        "layer_index": layer_index if not demo else None,
        "n_rows": len(fros),
        "row_ids": kept_ids,
        "feature": "frobenius_S_asym",
        "note": "パイロット用。ラベルは例示。本番は人手アノテと事前登録。"
        + (
            " MRMP キーは対話印象スコアであり v7 の 6 軸ではない。"
            if label_keys is not None and tuple(label_keys) == MRMP_LABEL_KEYS
            else ""
        ),
        "correlations_label_vs_fro": correlations,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="v7 Phase I-A: JSONL パイロット相関")
    p.add_argument(
        "--jsonl",
        type=Path,
        default=_ROOT / "experiments" / "data" / "v7_phase1a_pilot.jsonl",
    )
    p.add_argument("--demo", action="store_true", help="HF なし（CI）")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="使用する層インデックス（-1 で最終層）",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--offset",
        type=int,
        default=0,
        help="先頭からスキップする行数（大規模 JSONL のチャンク用）",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="最大 N 行（offset 適用後）",
    )
    p.add_argument(
        "--mrmp-labels",
        action="store_true",
        help="MRMP 正規化ラベル（mrmp_*_01）と Frobenius の相関",
    )
    args = p.parse_args()

    if args.offset > 0 or args.max_rows is not None:
        rows = load_jsonl_slice(
            args.jsonl,
            offset=max(0, args.offset),
            max_rows=args.max_rows,
        )
    else:
        rows = load_jsonl(args.jsonl)
    lk = MRMP_LABEL_KEYS if args.mrmp_labels else None
    payload = run_pilot(
        rows=rows,
        demo=args.demo,
        model_name=args.model,
        cpu=args.cpu,
        seed=args.seed,
        layer_index=args.layer,
        label_keys=lk,
    )
    js = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js, encoding="utf-8")
    print("v7_phase1a_pilot_ok", json.dumps({"n_rows": payload.get("n_rows")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
