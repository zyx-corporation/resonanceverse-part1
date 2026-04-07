"""
v7 Phase II-A: MRMP windows.jsonl から実データ R(τ) を計算する CLI。

事前登録: docs/planning/v7_phase2a_prereg_v1.json
コア: v7_phase2a_empirical.py（話者ブロック Frobenius・対話間等価集約）
注意行列: v7_phase1a_phi_correlation（**既定**）。
``--rvt-l2-mode sym|wasym`` は ``rvt_exp_2026_008_attn_inject``
（**eager** ロードの Causal LM。SDPA 不可）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.rvt_exp_2026_008_attn_inject import (  # noqa: E402
    hf_forward_attention_with_rvt_l2,
    model_supports_rvt_l2_inject,
    resolved_attn_implementation,
)
from experiments.v7_phase1a_pilot_jsonl import PILOT_KEYS  # noqa: E402
from experiments.v7_phase2a_empirical import (  # noqa: E402
    PREREG_SPAN_SPEC,
    R_tau_equal_weight_per_dialogue,
    auxiliary_label_delay_coherence_by_axis,
    dialogue_level_R_product_means,
    pair_block_asymmetry_frobenius,
    series_has_auxiliary_label_scores,
    speaker_token_indices_mrmp_window,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_rows(
    path: Path,
    offset: int,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    from experiments.jsonl_slice import iter_jsonl_slice

    return list(
        iter_jsonl_slice(path, offset=offset, max_rows=max_rows),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="v7 Phase II-A: MRMP 実データ R(τ)")
    p.add_argument(
        "--jsonl",
        type=Path,
        default=_ROOT / "experiments/logs/mrmp_prepared/windows.jsonl",
    )
    p.add_argument("--model", default="gpt2")
    p.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Hugging Face の revision（コミット SHA 等）を固定",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="注意行列の層（-1 で最終層）",
    )
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument(
        "--max-dialogues",
        type=int,
        default=None,
        help="デバッグ用: 先頭 N 対話のみ",
    )
    p.add_argument(
        "--tau-max",
        type=int,
        default=None,
        help="省略時は系列長から自動（各対話の窓数に依存）",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--include-series",
        action="store_true",
        help="出力 JSON に対話別 s_asym 系列を含める（大きくなる）",
    )
    p.add_argument(
        "--export-contributions",
        action="store_true",
        help="各 τ の対話別 R_d を contributions_by_tau に書き出す（ブートストラップ用）",
    )
    p.add_argument(
        "--rvt-l2-mode",
        choices=("base", "sym", "wasym"),
        default="base",
        help=(
            "RVT L2 注意介入（**attn_implementation=eager** の Causal LM。"
            "base=無介入・従来どおり）"
        ),
    )
    p.add_argument(
        "--rvt-l2-eps",
        type=float,
        default=0.05,
        help="wasym の ε（対称残差の強さ）",
    )
    p.add_argument(
        "--rvt-l2-all-layers",
        action="store_true",
        help="介入を全トランスフォーマ層に付与（既定は --layer のみ）",
    )
    args = p.parse_args()

    path = args.jsonl.resolve()
    if not path.is_file():
        err = json.dumps(
            {"error": "jsonl_not_found", "path": str(path)},
            ensure_ascii=False,
        )
        print(err, file=sys.stderr)
        sys.exit(1)

    rows = _load_rows(path, offset=max(0, args.offset), max_rows=args.max_rows)
    by_d: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        did = str(row.get("dialogue_id", "")).strip()
        if not did:
            continue
        by_d[did].append(row)

    dialogue_ids = sorted(by_d.keys())
    if args.max_dialogues is not None:
        dialogue_ids = dialogue_ids[: max(0, args.max_dialogues)]

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(json.dumps({"error": f"transformers: {e}"}), file=sys.stderr)
        sys.exit(1)

    from core.inference_device import select_inference_device
    from core.reproducibility import set_experiment_seed

    set_experiment_seed(args.seed)
    device = select_inference_device(force_cpu=args.cpu)
    hf_kw: dict[str, Any] = {}
    if args.revision:
        hf_kw["revision"] = args.revision
    tok = AutoTokenizer.from_pretrained(args.model, **hf_kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation="eager",
        **hf_kw,
    ).to(device)
    model.eval()
    inference_device_str = str(device)

    rvt_l2_block: dict[str, Any] | None = None
    if args.rvt_l2_mode != "base":
        if not model_supports_rvt_l2_inject(model):
            print(
                json.dumps(
                    {
                        "error": "rvt_l2_requires_eager_supported_causal_lm",
                        "model": args.model,
                        "attn_implementation": resolved_attn_implementation(
                            model,
                        ),
                        "hint": (
                            'from_pretrained(..., '
                            'attn_implementation="eager") '
                            "でロードされた GPT2 / Llama 系など"
                        ),
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            sys.exit(2)
        rvt_l2_block = {
            "mode": args.rvt_l2_mode,
            "eps": float(args.rvt_l2_eps),
            "all_layers": bool(args.rvt_l2_all_layers),
            "note_ja": (
                "事後 softmax ブレンド。"
                " sdpa ではなく eager でロード済みであること"
            ),
        }

    series_by_dialogue: list[dict[str, Any]] = []
    skipped_short = 0

    for did in dialogue_ids:
        drows = by_d[did]
        drows.sort(
            key=lambda r: (
                int(r.get("utterance_id", 0)),
                str(r.get("id", "")),
            ),
        )
        values: list[dict[str, Any]] = []
        for t_idx, row in enumerate(drows):
            text = str(row.get("text", "")).strip()
            if not text:
                skipped_short += 1
                continue
            utterer = str(row.get("speaker_tgt") or "").strip()
            responder = str(row.get("speaker_src") or "").strip()
            if not utterer or not responder:
                skipped_short += 1
                continue

            A, err, _ntok = hf_forward_attention_with_rvt_l2(
                model=model,
                tokenizer=tok,
                device=device,
                text=text,
                layer_index=args.layer,
                mode=args.rvt_l2_mode,
                eps=float(args.rvt_l2_eps),
                all_layers=bool(args.rvt_l2_all_layers),
            )
            if err is not None or A is None:
                skipped_short += 1
                continue

            iu, ir = speaker_token_indices_mrmp_window(
                text,
                utterer,
                responder,
                tok,
            )
            if not iu or not ir:
                s_ab = float("nan")
                s_ba = float("nan")
            else:
                s_ab = pair_block_asymmetry_frobenius(A, iu, ir)
                s_ba = pair_block_asymmetry_frobenius(A, ir, iu)

            entry: dict[str, Any] = {
                "t": t_idx,
                "s_asym_ab": s_ab,
                "s_asym_ba": s_ba,
            }
            for k in PILOT_KEYS:
                v = row.get(k)
                if isinstance(v, (int, float)):
                    entry[k] = float(v)
            values.append(entry)

        if len(values) < 2:
            continue
        series_by_dialogue.append({"dialogue_id": did, "values": values})

    max_tau = 0
    for d in series_by_dialogue:
        max_tau = max(max_tau, len(d["values"]) - 1)
    if args.tau_max is not None:
        max_tau = min(max_tau, max(0, args.tau_max))

    by_tau = R_tau_equal_weight_per_dialogue(
        series_by_dialogue,
        tau_max=max_tau,
    )

    payload: dict[str, Any] = {
        "schema_version": "v7_phase2a_empirical.v1",
        "mode": "mrmp_tau_sweep",
        "prereg_span_spec": PREREG_SPAN_SPEC,
        "model": args.model,
        "hf_revision": args.revision,
        "layer_index": args.layer,
        "inference_device": inference_device_str,
        "force_cpu": bool(args.cpu),
        "jsonl": (
            str(path.relative_to(_ROOT))
            if path.is_relative_to(_ROOT)
            else str(path)
        ),
        "windows_jsonl_sha256": _sha256_file(path),
        "n_dialogues": len(series_by_dialogue),
        "n_rows_input": len(rows),
        "skipped_or_short_rows": skipped_short,
        "by_tau": by_tau,
    }
    if rvt_l2_block is not None:
        payload["rvt_l2_intervention"] = rvt_l2_block
    if series_has_auxiliary_label_scores(series_by_dialogue):
        payload["auxiliary_label_delay_coherence"] = (
            auxiliary_label_delay_coherence_by_axis(
                series_by_dialogue,
                tau_max=max_tau,
            )
        )
    if args.include_series:
        payload["series_by_dialogue"] = series_by_dialogue
    if args.export_contributions:
        payload["contributions_by_tau"] = [
            {
                "tau": tau,
                "per_dialogue": dialogue_level_R_product_means(
                    series_by_dialogue, tau
                ),
            }
            for tau in range(0, max_tau + 1)
        ]

    js = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js, encoding="utf-8")
    info = json.dumps(
        {"taus": len(by_tau), "dialogues": len(series_by_dialogue)},
        ensure_ascii=False,
    )
    print("v7_phase2a_empirical_run_ok", info)


if __name__ == "__main__":
    main()
