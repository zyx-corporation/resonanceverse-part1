"""因果 LM による通常チャット生成（Streamlit 等から利用）。"""

from __future__ import annotations

from typing import Any

import torch


def build_generation_prompt(
    messages: list[dict[str, str]],
    tokenizer: Any,
) -> str:
    """``messages`` は ``{"role": "user"|"assistant", "content": str}``。

    ``chat_template`` があれば ``apply_chat_template``、なければ User/Assistant 平文連結。
    """
    tmpl = getattr(tokenizer, "chat_template", None)
    if tmpl and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")
    # 末尾は改行＋スペースでトークン境界を切りやすくする（BPE 先頭欠け対策の補助）
    lines.append("Assistant:")
    return "\n".join(lines) + "\n"


def _decode_generated_only(
    tokenizer: Any,
    input_ids_1d: torch.Tensor,
    output_ids_1d: torch.Tensor,
) -> str:
    """GPT-2 等 BPE で「新規トークン列だけ decode」すると先頭文字が欠けることがあるため、
    全文 decode からプロンプト decode を取り除く。
    """
    full = tokenizer.decode(output_ids_1d, skip_special_tokens=True)
    pref = tokenizer.decode(input_ids_1d, skip_special_tokens=True)
    if full.startswith(pref):
        return full[len(pref) :].lstrip()
    n_in = int(input_ids_1d.shape[0])
    return tokenizer.decode(output_ids_1d[n_in:], skip_special_tokens=True).lstrip()


def _truncate_simulated_turns(text: str) -> str:
    """平文チャット形式で、生成が次の ``User:`` / ``Assistant:`` に突入したらそこで打ち切る。"""
    t = text.strip()
    for sep in ("\nUser:", "\nAssistant:"):
        if sep in t:
            t = t.split(sep, 1)[0].strip()
    return t


def generate_completion(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
    no_repeat_ngram_size: int = 4,
    repetition_penalty: float = 1.25,
) -> str:
    """単発デコード。平文プロンプト時は擬似ターン境界で打ち切り。"""
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    gen_kw: dict[str, Any] = {
        "max_new_tokens": max(1, int(max_new_tokens)),
        "pad_token_id": pad_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": max(1.0, float(repetition_penalty)),
    }
    ngs = int(no_repeat_ngram_size)
    if ngs >= 2:
        gen_kw["no_repeat_ngram_size"] = ngs
    if do_sample:
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = max(0.01, float(temperature))
    else:
        gen_kw["do_sample"] = False
    with torch.no_grad():
        out = model.generate(**enc, **gen_kw)
    in_1d = enc["input_ids"][0].detach().cpu()
    out_1d = out[0].detach().cpu()
    raw = _decode_generated_only(tokenizer, in_1d, out_1d).strip()
    return _truncate_simulated_turns(raw)
