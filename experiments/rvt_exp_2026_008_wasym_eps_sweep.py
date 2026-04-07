"""
WASYM 強度 ε の理論較正用スイープ: 同一窓で ``s_asym_ab`` を ε 毎に比較。

Phase II-A 本線と同じ ``hf_forward_attention_with_rvt_l2``（eager Causal LM）を使用。
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

from experiments.rvt_exp_2026_008_mrmp_row import (  # noqa: E402
    _load_hf,
    _read_jsonl_line,
)
from experiments.rvt_exp_2026_008_attn_inject import (  # noqa: E402
    hf_forward_attention_with_rvt_l2,
    model_supports_rvt_l2_inject,
    resolved_attn_implementation,
)
from experiments.v7_phase2a_empirical import (  # noqa: E402
    pair_block_asymmetry_frobenius,
    speaker_token_indices_mrmp_window,
)


def main() -> None:
    p = argparse.ArgumentParser(description="RVT WASYM ε スイープ（1 窓・スカラー S）")
    p.add_argument("--jsonl", type=Path, required=True)
    p.add_argument("--line", type=int, default=0)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument(
        "--eps-list",
        default="0.0,0.02,0.05,0.1,0.2",
        help="カンマ区切り。0.0 は介入なし（base）",
    )
    args = p.parse_args()

    row = _read_jsonl_line(args.jsonl.resolve(), args.line)
    text = str(row.get("text", "")).strip()
    utterer = str(row.get("speaker_tgt") or "").strip()
    responder = str(row.get("speaker_src") or "").strip()
    if not text or not utterer or not responder:
        emsg = json.dumps(
            {"error": "need text, speaker_tgt, speaker_src"},
            ensure_ascii=False,
        )
        print(emsg, file=sys.stderr)
        sys.exit(2)

    model, tok, device = _load_hf(args.model, cpu=args.cpu)
    if not model_supports_rvt_l2_inject(model):
        print(
            json.dumps(
                {
                    "error": "rvt_l2_requires_eager_supported_causal_lm",
                    "model": args.model,
                    "attn_implementation": resolved_attn_implementation(model),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        sys.exit(2)

    raw_eps = str(args.eps_list).split(",")
    eps_vals = [float(x.strip()) for x in raw_eps if x.strip()]
    base_s: float | None = None
    rows_out: list[dict[str, Any]] = []

    iu, ir = speaker_token_indices_mrmp_window(text, utterer, responder, tok)
    if not iu or not ir:
        eb = json.dumps({"error": "empty_speaker_blocks"}, ensure_ascii=False)
        print(eb, file=sys.stderr)
        sys.exit(2)

    for eps in eps_vals:
        mode = "base" if abs(eps) < 1e-15 else "wasym"
        A, err, ntok = hf_forward_attention_with_rvt_l2(
            model=model,
            tokenizer=tok,
            device=device,
            text=text,
            layer_index=args.layer,
            mode=mode,
            eps=eps if mode == "wasym" else 0.0,
            all_layers=False,
        )
        if err is not None or A is None:
            rows_out.append(
                {
                    "eps": eps,
                    "mode": mode,
                    "error": err,
                    "s_asym_ab": None,
                }
            )
            continue
        sab = pair_block_asymmetry_frobenius(A, iu, ir)
        if base_s is None and mode == "base":
            base_s = float(sab)
        rows_out.append(
            {
                "eps": eps,
                "mode": mode,
                "n_tokens": ntok,
                "s_asym_ab": float(sab),
                "delta_vs_baseline": (
                    None if base_s is None else float(sab) - float(base_s)
                ),
            }
        )

    payload = {
        "schema_version": "rvt_exp_008_wasym_eps_sweep.v1",
        "model": args.model,
        "layer_index": args.layer,
        "jsonl_line": args.line,
        "baseline_s_asym_ab": base_s,
        "rows": rows_out,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
