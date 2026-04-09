"""v7 チャット 1 ターン生成 + v9 リズム（バッチ CLI / Streamlit 共通）。"""

from __future__ import annotations

import time
from typing import Any

import torch

from core.v9_rhythm_policy import (
    V9RhythmConfig,
    V9RhythmState,
    effective_max_new_tokens_for_oboro,
    slice_messages_for_oboro_history_tail,
)
from tools.v7_awai_chat_generate import generate_completion_awai
from tools.v7_llm_chat_generate import (
    build_generation_prompt,
    generate_completion,
    inject_system_summary,
)
from tools.v7_pair_chat_engine import tokenizer_supports_offset_pool
from tools.v7_v9_rhythm_hooks import (
    append_v9_rhythm_jsonl,
    v9_rhythm_before_chat_completion,
)


def append_v9_turn_jsonl(
    path: str | None,
    *,
    session_id: str,
    turn_index: int,
    reply: str,
    v9_enabled: bool,
    v9_payload: dict[str, Any] | None,
) -> None:
    """v9 ログ 1 行（ペイロードが無い、または v9 オフ、またはパス空なら何もしない）。

    `suspend_ms` 等は壁時計介入の代理（理論稿 付録 A.5 / §6.3）。Phase 2c は §6.1 の実験層。
    """
    p = (path or "").strip()
    if not p or not v9_enabled or v9_payload is None:
        return
    append_v9_rhythm_jsonl(
        p,
        {
            "ts": time.time(),
            "session_id": session_id,
            "turn_index": int(turn_index),
            "reply_char_count": len(reply or ""),
            **v9_payload,
        },
    )


def generate_assistant_turn_v7(
    *,
    chat_messages: list[dict[str, str]],
    system_summary: str | None,
    base_model: Any,
    tokenizer: Any,
    device: torch.device,
    awai_model: Any | None,
    v9_enabled: bool,
    v9_state: V9RhythmState,
    v9_cfg: V9RhythmConfig,
    pair_projector: Any,
    layer_index: int,
    d: int,
    delay_tau: int,
    omega_window_mode: str,
    omega_last_n: int,
    model_name: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    do_sample: bool = True,
    no_repeat_ngram_size: int = 4,
    repetition_penalty: float = 1.25,
) -> tuple[str, V9RhythmState, dict[str, Any] | None]:
    """ユーザーメッセージまで積んだ ``chat_messages`` に対しアシスタント 1 ターンを生成する。

    v9 を有効にすると、生成前に Ω 再計算・Suspend・（設定により）誠実さ system 付与を行う。
    戻り値: ``(reply_text, new_v9_state, v9_log_payload | None)``。
    例外時は ``（エラー）…`` 形式の文字列を返し、フック前失敗時は状態・ペイロードは入らない。
    """
    new_state = v9_state
    v9_payload: dict[str, Any] | None = None
    try:
        fast_ok, _ = tokenizer_supports_offset_pool(tokenizer)
        v9_decision, new_state, v9_payload, _ = (
            v9_rhythm_before_chat_completion(
                enabled=v9_enabled,
                fast_tokenizer_ok=fast_ok,
                chat_messages=chat_messages,
                model=base_model,
                tokenizer=tokenizer,
                device=device,
                layer_index=int(layer_index),
                d=int(d),
                delay_tau=int(delay_tau),
                pair_projector=pair_projector,
                omega_window_mode=str(omega_window_mode),
                omega_last_n=int(omega_last_n),
                model_name=str(model_name),
                cfg=v9_cfg,
                state=v9_state,
            )
        )
        summary_memo = system_summary
        if v9_decision.append_system_honesty:
            _hl = v9_cfg.honesty_line_ja
            _sm = (summary_memo or "").strip()
            summary_memo = (_sm + "\n\n" + _hl).strip() if _sm else _hl
        if v9_decision.suspend_ms > 0:
            time.sleep(
                float(
                    min(v9_decision.suspend_ms, v9_cfg.t_suspend_max),
                )
                / 1000.0
            )
        msgs_prompt, hist_sliced = slice_messages_for_oboro_history_tail(
            chat_messages,
            oboro_burst=bool(v9_decision.oboro_burst),
            tail=v9_cfg.oboro_history_tail_messages,
        )
        if v9_payload is not None:
            v9_payload["oboro_history_tail_messages"] = (
                v9_cfg.oboro_history_tail_messages
            )
            v9_payload["oboro_history_tail_sliced"] = bool(hist_sliced)
            v9_payload["prompt_message_count"] = len(msgs_prompt)
            v9_payload["prompt_message_count_full"] = len(chat_messages)
        prompt = build_generation_prompt(
            inject_system_summary(msgs_prompt, summary_memo),
            tokenizer,
        )
        eff_tokens, oboro_relaxed = effective_max_new_tokens_for_oboro(
            int(max_new_tokens),
            bool(v9_decision.oboro_burst),
            v9_cfg,
        )
        if v9_payload is not None:
            v9_payload["max_new_tokens_base"] = int(max_new_tokens)
            v9_payload["max_new_tokens_effective"] = int(eff_tokens)
            v9_payload["oboro_max_tokens_relaxed"] = bool(oboro_relaxed)

        if v9_decision.non_compliance == "silence":
            reply = ""
        elif v9_decision.non_compliance == "refuse":
            reply = str(v9_cfg.phase2c_refuse_message_ja)
        elif awai_model is not None:
            reply = generate_completion_awai(
                awai_model,
                tokenizer,
                device,
                prompt,
                max_new_tokens=int(eff_tokens),
                temperature=float(temperature),
                do_sample=bool(do_sample),
                repetition_penalty=float(repetition_penalty),
            )
        else:
            reply = generate_completion(
                base_model,
                tokenizer,
                device,
                prompt,
                max_new_tokens=int(eff_tokens),
                temperature=float(temperature),
                do_sample=bool(do_sample),
                no_repeat_ngram_size=int(no_repeat_ngram_size),
                repetition_penalty=float(repetition_penalty),
            )
    except Exception as e:
        return f"（エラー）{e}", new_state, v9_payload
    return reply, new_state, v9_payload
