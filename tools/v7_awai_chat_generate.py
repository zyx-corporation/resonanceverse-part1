"""AwaiIntegratedSLM 用のトークン単位生成（``model.generate`` 非対応のため自前ループ）。"""

from __future__ import annotations

from typing import Any

import torch

from core.resonant_core import AwaiIntegratedSLM

from tools.v7_llm_chat_generate import (
    _decode_generated_only,
    _truncate_simulated_turns,
)


def init_awai_out_head_from_lm_head(awai: AwaiIntegratedSLM) -> bool:
    """ベース因果 LM の ``lm_head`` 重みを ``out_head`` の先頭 ``hidden_size`` 列にコピーする。

    未学習の ``out_head`` よりチャットが破綻しにくい。共鳴 6 列はゼロ初期のまま。
    GPT-2 / Llama 系の ``nn.Linear(hidden, vocab)`` を想定。
    """
    base = awai.base_model
    lm = getattr(base, "lm_head", None)
    if lm is None or not isinstance(lm, torch.nn.Linear):
        return False
    h = int(base.config.hidden_size)
    v = int(base.config.vocab_size)
    w = lm.weight.data
    if w.shape != (v, h):
        return False
    with torch.no_grad():
        awai.out_head.weight.zero_()
        awai.out_head.weight[:, :h] = w.clone()
        if lm.bias is not None and awai.out_head.bias is not None:
            awai.out_head.bias.copy_(lm.bias)
        elif awai.out_head.bias is not None:
            awai.out_head.bias.zero_()
    return True


def _apply_repetition_penalty(
    logits: torch.Tensor,
    sequence_ids: torch.Tensor,
    penalty: float,
) -> None:
    """同一シーケンス内に現れたトークンに対し HF 風の繰り返しペナルティ（インプレース）。"""
    if penalty <= 1.0:
        return
    for tid in set(sequence_ids.view(-1).tolist()):
        tid = int(tid)
        if logits[tid] > 0:
            logits[tid] /= penalty
        else:
            logits[tid] *= penalty


def _max_context_length(config: Any) -> int:
    for name in ("n_positions", "max_position_embeddings", "seq_length"):
        v = getattr(config, name, None)
        if v is not None:
            return int(v)
    return 1024


@torch.no_grad()
def generate_completion_awai(
    awai: AwaiIntegratedSLM,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
    repetition_penalty: float = 1.25,
) -> str:
    """AwaiIntegratedSLM で 1 応答分を生成。``no_repeat_ngram`` は未対応（HF 専用）。"""
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError("batch size 1 のプロンプトのみ対応")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    max_ctx = _max_context_length(awai.base_model.config)

    awai.eval()
    prompt_1d = input_ids[0].detach().cpu()
    new_ids: list[int] = []
    cur = input_ids
    for _ in range(max(1, int(max_new_tokens))):
        if cur.size(1) >= max_ctx:
            break
        logits = awai(cur)
        next_logits = logits[0, -1, :].float().clone()
        _apply_repetition_penalty(
            next_logits, cur[0], float(repetition_penalty)
        )
        if do_sample:
            t = max(0.01, float(temperature))
            next_logits = next_logits / t
            probs = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        else:
            next_id = int(next_logits.argmax(dim=-1).item())
        new_ids.append(next_id)
        if eos_id is not None and next_id == int(eos_id):
            break
        cur = torch.cat(
            [cur, torch.tensor([[next_id]], device=device, dtype=cur.dtype)],
            dim=1,
        )

    tail = torch.tensor(new_ids, dtype=prompt_1d.dtype)
    out_1d = torch.cat([prompt_1d, tail], dim=0)
    raw = _decode_generated_only(tokenizer, prompt_1d, out_1d).strip()
    return _truncate_simulated_turns(raw)
