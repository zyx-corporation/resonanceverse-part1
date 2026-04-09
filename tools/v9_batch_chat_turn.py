#!/usr/bin/env python3
"""v7 チャット 1 ターンをバッチ生成（v9 リズム A/B 用）。

入力 JSON（例）::

    {
      "messages": [
        {"role": "user", "content": "こんにちは"},
        {"role": "assistant", "content": "どうも"},
        {"role": "user", "content": "調子は？"}
      ],
      "system_summary": "任意の system 文（省略可）"
    }

使用例（リポジトリルート）::

    python tools/v9_batch_chat_turn.py --input turn.json --model Qwen/Qwen2.5-3B-Instruct
    python tools/v9_batch_chat_turn.py --input turn.json --v9 --jsonl-log experiments/logs/v9_cli.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.v9_rhythm_policy import V9RhythmConfig, V9RhythmState
from tools.v7_llm_chat_with_v9 import append_v9_turn_jsonl, generate_assistant_turn_v7
from tools.v7_pair_chat_engine import (
    DEFAULT_RVT_SLM_INSTRUCT,
    load_causal_lm,
    resolve_device,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="v7 チャット 1 ターン + 任意 v9 リズム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "理論メモ: --p2c（Phase 2c）は詳細展望 §6.1 の二層化に従い、"
            "v8.0 自律的境界の理論同一物ではない実験用摂動として扱う。"
        ),
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="messages / system_summary を含む JSON",
    )
    p.add_argument("--model", type=str, default=DEFAULT_RVT_SLM_INSTRUCT)
    p.add_argument("--cpu", action="store_true", help="CPU 強制")
    p.add_argument("--v9", action="store_true", help="v9 リズム（Ω・Suspend）を有効化")
    p.add_argument(
        "--jsonl-log",
        type=Path,
        default=None,
        help="v9 有効時の JSONL 追記先（--v9 と併用）",
    )
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--layer-index", type=int, default=-1)
    p.add_argument("--d", type=int, default=6)
    p.add_argument("--delay-tau", type=int, default=0)
    p.add_argument("--omega-window", choices=["cumulative", "last_n"], default="cumulative")
    p.add_argument("--omega-last-n", type=int, default=8)
    p.add_argument("--session-id", type=str, default="", help="空なら UUID を採番")
    p.add_argument(
        "--no-oboro-relax-tokens",
        action="store_true",
        help="朧バースト時も max_new_tokens を増やさない（M3 オフ）",
    )
    p.add_argument(
        "--oboro-token-mult",
        type=float,
        default=2.0,
        help="朧時の max_new_tokens 倍率（既定 2）",
    )
    p.add_argument(
        "--oboro-token-cap",
        type=int,
        default=2048,
        help="朧緩和後の上限トークン数",
    )
    p.add_argument(
        "--oboro-history-tail",
        type=int,
        default=0,
        help="朧時のみ生成プロンプトを末尾 N メッセージに限定（0=無効）",
    )
    p.add_argument(
        "--p2a-axis-logits",
        action="store_true",
        help="M4 2a: decide 軸 Suspend に logits_var を加算（hook から渡す想定・CLI 単体では通常オフ）",
    )
    p.add_argument(
        "--theta-high-unc",
        type=float,
        default=0.0,
        help="logits 分散閾値（0 で軸 logits 加算オフ）",
    )
    p.add_argument(
        "--delta-high-unc",
        type=int,
        default=400,
        help="閾値超過時の追加 ms",
    )
    p.add_argument(
        "--p2b-traj",
        action="store_true",
        help="M4 2b: 軌道固着を S_stuck に統合",
    )
    p.add_argument("--traj-eps", type=float, default=0.08, help="軌道 L2 ε")
    p.add_argument(
        "--p2c",
        action="store_true",
        help="M4 2c: 朧時の沈黙/拒否抽選（§6.1 実験層、epilog 参照）",
    )
    p.add_argument("--p2c-silence-p", type=float, default=0.0)
    p.add_argument("--p2c-refuse-p", type=float, default=0.0)
    p.add_argument("--p2c-seed", type=int, default=0)
    args = p.parse_args()

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        print("error: messages が空または不正", file=sys.stderr)
        return 2
    system_summary = raw.get("system_summary")
    if system_summary is not None and not isinstance(system_summary, str):
        print("error: system_summary は文字列で指定してください", file=sys.stderr)
        return 2

    device = resolve_device(force_cpu=bool(args.cpu))
    model, tok = load_causal_lm(
        str(args.model),
        device=device,
        cpu=bool(args.cpu),
        attn_implementation="eager",
    )

    _thu = float(args.theta_high_unc)
    v9_cfg = V9RhythmConfig(
        oboro_relax_max_new_tokens=not bool(args.no_oboro_relax_tokens),
        oboro_max_new_tokens_multiplier=float(args.oboro_token_mult),
        oboro_max_new_tokens_cap=int(args.oboro_token_cap),
        oboro_history_tail_messages=(
            int(args.oboro_history_tail)
            if int(args.oboro_history_tail) > 0
            else None
        ),
        theta_high_unc=_thu if _thu > 1e-9 else None,
        delta_suspend_ms_high_unc=int(args.delta_high_unc),
        phase2a_apply_logits_to_axis_suspend=bool(args.p2a_axis_logits),
        phase2b_trajectory_stuck=bool(args.p2b_traj),
        trajectory_epsilon_w=float(args.traj_eps),
        phase2c_enabled=bool(args.p2c),
        phase2c_silence_probability=float(args.p2c_silence_p),
        phase2c_refuse_probability=float(args.p2c_refuse_p),
        phase2c_seed=int(args.p2c_seed),
    )
    v9_state = V9RhythmState()
    sid = (args.session_id or "").strip() or str(uuid.uuid4())
    turn_index = sum(1 for m in messages if (m.get("role") or "") == "assistant")

    reply, _, v9_payload = generate_assistant_turn_v7(
        chat_messages=messages,
        system_summary=system_summary,
        base_model=model,
        tokenizer=tok,
        device=device,
        awai_model=None,
        v9_enabled=bool(args.v9),
        v9_state=v9_state,
        v9_cfg=v9_cfg,
        pair_projector=None,
        layer_index=int(args.layer_index),
        d=int(args.d),
        delay_tau=int(args.delay_tau),
        omega_window_mode=str(args.omega_window),
        omega_last_n=int(args.omega_last_n),
        model_name=str(args.model),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        do_sample=True,
        no_repeat_ngram_size=4,
        repetition_penalty=1.25,
    )
    jlog = str(args.jsonl_log) if args.jsonl_log else ""
    append_v9_turn_jsonl(
        jlog,
        session_id=sid,
        turn_index=turn_index,
        reply=reply,
        v9_enabled=bool(args.v9),
        v9_payload=v9_payload,
    )

    out = {"reply": reply, "v9_log": v9_payload, "session_id": sid}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
