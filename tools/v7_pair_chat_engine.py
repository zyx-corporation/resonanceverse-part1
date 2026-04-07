"""v7 話者ペア (T,d) 系列・Ω を MRMP 風トランスクリプトから計算するエンジン（チャット UI 用）。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.inference_device import select_inference_device
from core.resonant_core import AwaiIntegratedSLM
from core.v7_pair_w_trajectory import PairRelationLinear, series_from_turn_hiddens
from experiments.v7_phase1a_phi_correlation import hf_forward_hidden_last_layer
from experiments.v7_phase2a_empirical import speaker_token_indices_mrmp_window


def mrmp_window_text(turns: list[tuple[str, str]]) -> str:
    """``(話者, 発話本文)`` の列から ``名前: 本文`` 改行区切りの窓テキストを生成。"""
    return "\n".join(f"{sp.strip()}: {tx.strip()}" for sp, tx in turns)


def slice_turns_for_pair_step(
    turns: list[tuple[str, str]],
    k: int,
    *,
    window_mode: str,
    last_n: int,
) -> tuple[list[tuple[str, str]], int]:
    """ペアステップ ``k``（``k>=1``）用の窓に含めるターン部分列と、全体における先頭 index。

    - ``cumulative``: ``turns[0 : k+1]``（累積全文）
    - ``last_n``: 直近 ``last_n`` ターン（``last_n`` は 2 未満なら 2 に繰り上げ）
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if window_mode == "cumulative":
        return turns[: k + 1], 0
    if window_mode == "last_n":
        ln = max(2, int(last_n))
        start_k = max(0, k + 1 - ln)
        return turns[start_k : k + 1], start_k
    raise ValueError(f"window_mode must be 'cumulative' or 'last_n', got {window_mode!r}")


def chat_messages_to_mrmp_turns(
    messages: list[dict[str, str]],
) -> list[tuple[str, str]]:
    """``role`` / ``content`` のチャット履歴を MRMP 風 ``(User|Assistant, 本文)`` に変換。"""
    out: list[tuple[str, str]] = []
    for m in messages:
        role = (m.get("role") or "user").strip().lower()
        content = (m.get("content") or "").strip()
        label = "Assistant" if role == "assistant" else "User"
        out.append((label, content))
    return out


def tokenizer_supports_offset_pool(tokenizer: Any) -> tuple[bool, str]:
    """``speaker_token_indices_mrmp_window`` に必要な ``offset_mapping`` が使えるか。"""
    from transformers import PreTrainedTokenizerFast

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        return False, "Fast トークナイザが必要です（例: gpt2 → GPT2TokenizerFast）。"
    return True, ""


def analyze_transcript_turns(
    turns: list[tuple[str, str]],
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layer_index: int = -1,
    d: int = 6,
    delay_tau: int = 0,
    pair_projector: PairRelationLinear | None = None,
    window_mode: str = "cumulative",
    last_n: int = 8,
    source: str = "mrmp_transcript",
) -> dict[str, Any]:
    """ターン 1..T-1 を MRMP 同型の (tgt=現在話者, src=直前話者) で解析。

    ``turns`` は時系列の (speaker, utterance)（発話本文に話者名プレフィックスを含めない）。

    ``window_mode``: ``cumulative``（先頭から累積）または ``last_n``（直近 N ターンのみ）。
    """
    ok, msg = tokenizer_supports_offset_pool(tokenizer)
    if not ok:
        return {"error": msg, "schema_version": "v7_pair_chat_analysis.v1"}

    if len(turns) < 2:
        return {
            "error": "話者ペアが必要です。ターンを 2 つ以上追加してください。",
            "schema_version": "v7_pair_chat_analysis.v1",
        }

    turn_hiddens: list[torch.Tensor] = []
    iu_list: list[list[int]] = []
    ir_list: list[list[int]] = []
    meta_rows: list[dict[str, Any]] = []

    for t in range(1, len(turns)):
        slice_turns, w_start = slice_turns_for_pair_step(
            turns,
            t,
            window_mode=window_mode,
            last_n=last_n,
        )
        window = mrmp_window_text(slice_turns)
        utterer, _u_txt = turns[t]
        responder, _r_txt = turns[t - 1]
        utterer = utterer.strip()
        responder = responder.strip()
        iu, ir = speaker_token_indices_mrmp_window(
            window, utterer, responder, tokenizer
        )
        h, err, ntok = hf_forward_hidden_last_layer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            text=window,
            layer_index=layer_index,
        )
        if err is not None or h is None:
            return {
                "error": f"t={t} で隠れ状態取得失敗: {err}",
                "schema_version": "v7_pair_chat_analysis.v1",
            }
        turn_hiddens.append(h.detach())
        iu_list.append(list(iu))
        ir_list.append(list(ir))
        meta_rows.append(
            {
                "t": t,
                "utterer": utterer,
                "responder": responder,
                "n_tokens": ntok,
                "n_idx_utterer": len(iu),
                "n_idx_responder": len(ir),
                "window_start_turn": w_start,
                "window_end_turn": t,
                "n_turns_in_window": len(slice_turns),
            }
        )

    w_ij, w_ji, omega = series_from_turn_hiddens(
        turn_hiddens,
        iu_list,
        ir_list,
        d=d,
        delay_tau=delay_tau,
        pair_projector=pair_projector,
    )

    return {
        "schema_version": "v7_pair_chat_analysis.v1",
        "source": source,
        "window_mode": window_mode,
        "last_n": int(max(2, last_n)) if window_mode == "last_n" else None,
        "implementation_note_ja": (
            "Ω は合成定義（v7 §4.3 簡略）。Phase III-A の人間妥当性とは別。"
        ),
        "n_turns_input": len(turns),
        "n_pair_steps": len(turn_hiddens),
        "per_step": meta_rows,
        "d": d,
        "delay_tau": delay_tau,
        "pair_map": "PairRelationLinear" if pair_projector is not None else "concat_truncate",
        "w_ij": w_ij.cpu().tolist(),
        "w_ji": w_ji.cpu().tolist(),
        "omega": omega.cpu().tolist(),
    }


def analyze_chat_messages_for_omega(
    messages: list[dict[str, str]],
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layer_index: int = -1,
    d: int = 6,
    delay_tau: int = 0,
    pair_projector: PairRelationLinear | None = None,
    window_mode: str = "cumulative",
    last_n: int = 8,
) -> dict[str, Any]:
    """通常 LLM チャット履歴（user/assistant）から Ω 系列を計算。"""
    turns = chat_messages_to_mrmp_turns(messages)
    return analyze_transcript_turns(
        turns,
        model=model,
        tokenizer=tokenizer,
        device=device,
        layer_index=layer_index,
        d=d,
        delay_tau=delay_tau,
        pair_projector=pair_projector,
        window_mode=window_mode,
        last_n=last_n,
        source="llm_chat",
    )


def load_causal_lm(model_name: str, *, device: torch.device, cpu: bool) -> tuple[Any, Any]:
    """HF 因果 LM とトークナイザをロード（評価モード）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model.eval()
    if cpu:
        model = model.to(torch.device("cpu"))
    else:
        model = model.to(device)
    return model, tok


def unpack_model_bundle(
    bundle: tuple[Any, ...] | None,
) -> tuple[Any, Any, torch.device, AwaiIntegratedSLM | None]:
    """``(base, tok, device)`` または ``(base, tok, device, awai)`` を展開する。"""
    if bundle is None:
        raise ValueError("model_bundle is None")
    if len(bundle) == 3:
        base, tok, dev = bundle
        return base, tok, dev, None
    if len(bundle) == 4:
        base, tok, dev, awai = bundle
        return base, tok, dev, awai  # type: ignore[misc]
    raise ValueError(f"unexpected model_bundle length {len(bundle)}")


def build_awai_integrated(
    base_model: Any,
    device: torch.device,
    *,
    cultural_modulation: bool = False,
) -> AwaiIntegratedSLM:
    awai = AwaiIntegratedSLM(
        base_model, cultural_modulation=cultural_modulation
    ).to(device)
    awai.eval()
    return awai


def load_awai_checkpoint(awai: AwaiIntegratedSLM, path: str, map_location: Any) -> None:
    """学習済み重みを読み込む。``state_dict`` / ``model`` キー付きチェックポイントにも対応。"""
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(blob, dict):
        for key in ("state_dict", "model", "awai_state_dict"):
            inner = blob.get(key)
            if isinstance(inner, dict):
                blob = inner
                break
    if not isinstance(blob, dict):
        raise TypeError("checkpoint must be a dict-like state_dict")
    awai.load_state_dict(blob, strict=False)


def resolve_device(*, force_cpu: bool) -> torch.device:
    return select_inference_device(force_cpu=force_cpu)
