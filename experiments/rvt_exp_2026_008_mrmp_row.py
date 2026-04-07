"""
MRMP ``windows.jsonl`` の 1 行から RVT L1 拡張
（ヘッド別注意 → ブロック Frobenius → 6 軸代理）を JSON 化する。

Phase II-A と同一の話者ブロック規則
（``speaker_token_indices_mrmp_window``）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.rvt_exp_2026_008_attention import (  # noqa: E402
    hf_forward_attention_layer_heads_numpy,
)
from experiments.rvt_exp_2026_008_attn_inject import (  # noqa: E402
    hf_forward_attention_layer_heads_with_rvt_l2,
    model_supports_rvt_l2_inject,
    resolved_attn_implementation,
)
from experiments.rvt_exp_2026_008_head_axis_matrix import (  # noqa: E402
    load_head_axis_matrix,
)
from experiments.rvt_exp_2026_008_w_asym import (  # noqa: E402
    default_uniform_head_axis_matrix,
    per_head_block_asym_frobenius,
    project_head_frobenius_to_six_axes,
)
from experiments.v7_phase2a_empirical import (  # noqa: E402
    speaker_token_indices_mrmp_window,
)


def num_attention_heads_for_mrmp_model(model: Any) -> int:
    """
    ``--head-axis-matrix`` の行数検証用。**GPT2** の ``n_head``、
    **Llama 等**の ``num_attention_heads`` など設定の版差を吸収する。
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise TypeError(
            "num_attention_heads_for_mrmp_model: model.config required",
        )
    for key in ("n_head", "num_attention_heads", "n_heads"):
        v = getattr(cfg, key, None)
        if v is not None:
            return int(v)
    raise ValueError(
        "cannot resolve attention head count from config "
        f"(type={type(cfg).__name__})",
    )


def _require_rvt_l2_model_or_exit(
    model: Any,
    model_name: str,
    mode: str,
) -> None:
    if mode == "base":
        return
    if model_supports_rvt_l2_inject(model):
        return
    msg = json.dumps(
        {
            "error": "rvt_l2_requires_eager_supported_causal_lm",
            "model": model_name,
            "attn_implementation": resolved_attn_implementation(model),
        },
        ensure_ascii=False,
    )
    print(msg, file=sys.stderr)
    sys.exit(2)


def _read_jsonl_line(path: Path, line_index: int) -> dict[str, Any]:
    if line_index < 0:
        raise ValueError("line_index must be >= 0")
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == line_index:
                line = line.strip()
                if not line:
                    break
                return json.loads(line)
    raise IndexError(f"no line at index {line_index} in {path}")


def iter_jsonl_physical_lines(
    path: Path,
    *,
    first_line: int,
    n_lines: int,
) -> Iterator[tuple[int, str]]:
    """物理行インデックス ``first_line`` から最大 ``n_lines`` 行を `(i, raw)` で列挙。"""
    if first_line < 0 or n_lines < 1:
        raise ValueError("first_line >= 0 and n_lines >= 1 required")
    emitted = 0
    with path.open(encoding="utf-8") as f:
        for i, raw in enumerate(f):
            if i < first_line:
                continue
            yield i, raw
            emitted += 1
            if emitted >= n_lines:
                break


def rvt_payload_from_mrmp_window_row(
    row: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    layer_index: int,
    seed: int,
    head_axis_matrix: np.ndarray | None = None,
    rvt_l2_mode: str = "base",
    rvt_l2_eps: float = 0.05,
    rvt_l2_all_layers: bool = False,
) -> dict[str, Any]:
    from core.reproducibility import set_experiment_seed

    set_experiment_seed(seed)
    text = str(row.get("text", ""))
    utterer = str(row.get("speaker_tgt", "")).strip()
    responder = str(row.get("speaker_src", "")).strip()
    ia, ib = speaker_token_indices_mrmp_window(
        text, utterer, responder, tokenizer
    )
    rid = row.get("id", "")
    base_meta = {
        "schema_version": "rvt_exp_008_mrmp_row.v1",
        "row_id": rid,
        "dialogue_id": row.get("dialogue_id"),
        "speaker_tgt": utterer,
        "speaker_src": responder,
        "layer_index": layer_index,
    }
    if not ia or not ib:
        return {
            **base_meta,
            "ok": False,
            "error": "empty_speaker_blocks",
            "n_index_a": len(ia),
            "n_index_b": len(ib),
        }

    if rvt_l2_mode == "base":
        heads, err, ntok = hf_forward_attention_layer_heads_numpy(
            model=model,
            tokenizer=tokenizer,
            device=device,
            text=text,
            layer_index=layer_index,
        )
    else:
        heads, err, ntok = hf_forward_attention_layer_heads_with_rvt_l2(
            model=model,
            tokenizer=tokenizer,
            device=device,
            text=text,
            layer_index=layer_index,
            mode=rvt_l2_mode,
            eps=float(rvt_l2_eps),
            all_layers=bool(rvt_l2_all_layers),
        )
    if err is not None:
        return {**base_meta, "ok": False, "error": err, "n_tokens": ntok}
    assert heads is not None

    f_h = per_head_block_asym_frobenius(heads, ia, ib)
    h_ct = int(heads.shape[0])
    if head_axis_matrix is not None:
        m = np.asarray(head_axis_matrix, dtype=np.float64)
        if m.shape != (h_ct, 6):
            return {
                **base_meta,
                "ok": False,
                "error": (
                    f"head_axis_matrix shape {m.shape} != ({h_ct}, 6)"
                ),
            }
        axis_mode = "learned_file"
        note_ja = (
            "w_axes_proxy は外部ファイル等の学習可能 M（計画書の H×6 射影）による。"
        )
    else:
        rng = np.random.default_rng(int(seed))
        m = default_uniform_head_axis_matrix(h_ct, rng=rng)
        axis_mode = "random_uniform_l1col"
        note_ja = (
            "w_axes_proxy は仮 M（seed 固定の乱数列・列 L1 正規化）による代理。"
            " 学習済み M は experiments/rvt_exp_2026_008_train_head_axis_m.py 参照。"
        )
    w_axes = project_head_frobenius_to_six_axes(f_h, m)

    out: dict[str, Any] = {
        **base_meta,
        "ok": True,
        "n_tokens": ntok,
        "n_heads": h_ct,
        "head_axis_mode": axis_mode,
        "per_head_block_frobenius": [float(x) for x in f_h.tolist()],
        "w_axes_proxy": [float(x) for x in w_axes.tolist()],
        "note_ja": note_ja,
    }
    if rvt_l2_mode != "base":
        out["rvt_l2_intervention"] = {
            "mode": rvt_l2_mode,
            "eps": float(rvt_l2_eps),
            "all_layers": bool(rvt_l2_all_layers),
        }
    return out


def _load_hf(model_name: str, cpu: bool) -> tuple[Any, Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from core.inference_device import select_inference_device
    from core.reproducibility import set_experiment_seed

    set_experiment_seed(0)
    device = select_inference_device(force_cpu=cpu)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
    ).to(device)
    m.eval()
    return m, tok, device


def run_mrmp_rvt_batch(
    jsonl: Path,
    *,
    first_line: int,
    max_rows: int,
    model_name: str,
    cpu: bool,
    layer_index: int,
    seed: int,
    accumulate_awai: bool,
    head_axis_matrix: np.ndarray | None = None,
    head_axis_matrix_path: Path | None = None,
    rvt_l2_mode: str = "base",
    rvt_l2_eps: float = 0.05,
    rvt_l2_all_layers: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, bool, bool]:
    """
    モデル 1 回ロードで複数物理行を処理。

    戻り値: ``(payloads, awai_log|None, any_ok, any_fail)``
    """
    from experiments.rvt_exp_2026_008_awai_vector import RvtExp008AwaiVector

    model, tok, device = _load_hf(model_name, cpu=cpu)
    _require_rvt_l2_model_or_exit(model, model_name, rvt_l2_mode)
    hm = head_axis_matrix
    if head_axis_matrix_path is not None:
        hm = load_head_axis_matrix(
            head_axis_matrix_path.resolve(),
            n_heads=num_attention_heads_for_mrmp_model(model),
        )
    awai: RvtExp008AwaiVector | None
    awai = RvtExp008AwaiVector() if accumulate_awai else None
    out: list[dict[str, Any]] = []
    any_ok = False
    any_fail = False
    for phys_i, raw in iter_jsonl_physical_lines(
        jsonl.resolve(),
        first_line=first_line,
        n_lines=max_rows,
    ):
        line = raw.strip()
        if not line:
            payload = {
                "schema_version": "rvt_exp_008_mrmp_row.v1",
                "physical_line": phys_i,
                "ok": False,
                "error": "empty_line",
            }
            out.append(payload)
            any_fail = True
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            out.append(
                {
                    "schema_version": "rvt_exp_008_mrmp_row.v1",
                    "physical_line": phys_i,
                    "ok": False,
                    "error": f"json_decode:{e}",
                }
            )
            any_fail = True
            continue
        payload = rvt_payload_from_mrmp_window_row(
            row,
            model=model,
            tokenizer=tok,
            device=device,
            layer_index=layer_index,
            seed=seed,
            head_axis_matrix=hm,
            rvt_l2_mode=rvt_l2_mode,
            rvt_l2_eps=rvt_l2_eps,
            rvt_l2_all_layers=rvt_l2_all_layers,
        )
        payload["physical_line"] = phys_i
        out.append(payload)
        if payload.get("ok"):
            any_ok = True
            wap = payload.get("w_axes_proxy")
            if awai is not None and wap:
                awai.append(np.array(wap, dtype=np.float64))
        else:
            any_fail = True
    awai_log = awai.to_dict_log() if awai is not None else None
    if awai_log is not None:
        awai_log["schema_version"] = "rvt_exp_008_mrmp_batch_awai.v1"
    return out, awai_log, any_ok, any_fail


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "MRMP windows.jsonl → RVT ヘッド Frobenius + 6 軸代理"
            "（1 行 or バッチ）"
        ),
    )
    p.add_argument("--jsonl", type=Path, required=True)
    p.add_argument("--line", type=int, default=0, help="0 始まり物理行番号")
    p.add_argument(
        "--max-rows",
        type=int,
        default=1,
        help="この行から連続する物理行数（>=1）。>1 でバッチ・標準出력は JSONL",
    )
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--accumulate-awai",
        action="store_true",
        help=(
            "成功行の w_axes_proxy を RvtExp008AwaiVector に積む"
            "（バッチ時のみ意味あり）"
        ),
    )
    p.add_argument(
        "--awai-out",
        type=Path,
        default=None,
        help="--accumulate-awai 時の Awai 系列 JSON 出力",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="1 件でも ok=False なら終了コード 1",
    )
    p.add_argument(
        "--pretty-array",
        action="store_true",
        help=(
            "バッチ時も 1 本の JSON 配列で stdout"
            "（既定は JSONL 1 行 1 オブジェクト）"
        ),
    )
    p.add_argument(
        "--head-axis-matrix",
        type=Path,
        default=None,
        help=(
            "(H,6) の .npy / .json。H はモデルの注意ヘッド数"
            "（n_head / num_attention_heads 等）と一致必須"
        ),
    )
    p.add_argument(
        "--rvt-l2-mode",
        choices=("base", "sym", "wasym"),
        default="base",
        help="ヘッド注意取得時の L2 ブレンド（eager Causal LM のみ）",
    )
    p.add_argument("--rvt-l2-eps", type=float, default=0.05)
    p.add_argument(
        "--rvt-l2-all-layers",
        action="store_true",
        help="介入を全層に付与（既定は --layer のみ）",
    )
    args = p.parse_args()

    jsonl_path = args.jsonl.resolve()
    if args.max_rows < 1:
        msg = json.dumps(
            {"error": "max-rows must be >= 1"},
            ensure_ascii=False,
        )
        print(msg)
        sys.exit(2)

    if args.max_rows == 1:
        row = _read_jsonl_line(jsonl_path, args.line)
        model, tok, device = _load_hf(args.model, cpu=args.cpu)
        _require_rvt_l2_model_or_exit(model, args.model, args.rvt_l2_mode)
        hm: np.ndarray | None = None
        if args.head_axis_matrix is not None:
            hm = load_head_axis_matrix(
                args.head_axis_matrix.resolve(),
                n_heads=num_attention_heads_for_mrmp_model(model),
            )
        payload = rvt_payload_from_mrmp_window_row(
            row,
            model=model,
            tokenizer=tok,
            device=device,
            layer_index=args.layer,
            seed=args.seed,
            head_axis_matrix=hm,
            rvt_l2_mode=args.rvt_l2_mode,
            rvt_l2_eps=args.rvt_l2_eps,
            rvt_l2_all_layers=args.rvt_l2_all_layers,
        )
        acc_w = payload.get("w_axes_proxy")
        ok_acc = args.accumulate_awai and payload.get("ok") and acc_w
        if ok_acc:
            from experiments.rvt_exp_2026_008_awai_vector import (
                RvtExp008AwaiVector,
            )  # noqa: E402

            acc = RvtExp008AwaiVector()
            acc.append(np.array(acc_w, dtype=np.float64))
            awai_js = json.dumps(
                acc.to_dict_log(),
                indent=2,
                ensure_ascii=False,
            )
            if args.awai_out:
                args.awai_out.parent.mkdir(parents=True, exist_ok=True)
                args.awai_out.write_text(awai_js + "\n", encoding="utf-8")
            else:
                print("=== awai_vector ===", file=sys.stderr)
                print(awai_js, file=sys.stderr)
        js = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(js + "\n", encoding="utf-8")
        print(js)
        sys.exit(0 if payload.get("ok") else 1)

    payloads, awai_log, any_ok, any_fail = run_mrmp_rvt_batch(
        jsonl_path,
        first_line=args.line,
        max_rows=args.max_rows,
        model_name=args.model,
        cpu=args.cpu,
        layer_index=args.layer,
        seed=args.seed,
        accumulate_awai=args.accumulate_awai,
        head_axis_matrix_path=args.head_axis_matrix,
        rvt_l2_mode=args.rvt_l2_mode,
        rvt_l2_eps=args.rvt_l2_eps,
        rvt_l2_all_layers=args.rvt_l2_all_layers,
    )

    out_io: list[str] = []
    if args.pretty_array:
        out_io.append(json.dumps(payloads, ensure_ascii=False, indent=2))
    else:
        for pl in payloads:
            out_io.append(json.dumps(pl, ensure_ascii=False))

    combined = "\n".join(out_io) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(combined, encoding="utf-8")
    print(combined, end="")

    if args.accumulate_awai and awai_log is not None:
        ajs = json.dumps(awai_log, indent=2, ensure_ascii=False) + "\n"
        if args.awai_out:
            args.awai_out.parent.mkdir(parents=True, exist_ok=True)
            args.awai_out.write_text(ajs, encoding="utf-8")
        else:
            print("=== awai_accumulated ===", file=sys.stderr)
            print(ajs, end="", file=sys.stderr)

    if args.strict and any_fail:
        sys.exit(1)
    if not any_ok:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
