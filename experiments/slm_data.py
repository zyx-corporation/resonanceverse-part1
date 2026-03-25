"""
Phase B: 因果 LM 用のトークン列バッチ生成（ランダム / Wikitext-2-raw）。

`slm_resonance_lm.py` から利用。再現性のため `seed` を渡す。
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import torch


def random_token_batches(
    vocab_size: int,
    batch: int,
    seq_len: int,
    device: torch.device,
) -> Iterator[torch.Tensor]:
    """無限に (B, S) のランダムトークン（学習スモーク用）。乱数は `set_experiment_seed` に依存。"""
    while True:
        yield torch.randint(0, vocab_size, (batch, seq_len), device=device)


def token_ids_to_chunks(
    token_ids_1d: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """
    1D トークン ID を (num_chunks, seq_len) に切る。端数は捨てる。
    """
    L = int(token_ids_1d.numel())
    usable = (L // seq_len) * seq_len
    if usable == 0:
        return token_ids_1d.new_empty((0, seq_len))
    return token_ids_1d[:usable].reshape(-1, seq_len)


def batched_chunks(
    chunks_2d: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Iterator[torch.Tensor]:
    """(N, S) をバッチ次元で分割。最後の不足バッチは捨てる。"""
    n = chunks_2d.size(0)
    for i in range(0, n - (n % batch_size), batch_size):
        yield chunks_2d[i : i + batch_size].to(device)


def cycle_batches(chunks_2d: torch.Tensor, batch_size: int, device: torch.device) -> Iterator[torch.Tensor]:
    """バッチを循環（max_steps 学習用）。"""
    if chunks_2d.numel() == 0:
        raise ValueError("empty chunks")
    it = batched_chunks(chunks_2d, batch_size, device)
    first = list(it)
    if not first:
        raise ValueError("no full batches; increase text or lower batch/seq_len")
    yield from itertools.cycle(first)


def load_wikitext_token_ids(
    tokenizer: Any,
    *,
    max_chars: int = 200_000,
    max_tokens: int | None = None,
    split: str = "train",
) -> torch.Tensor:
    """
    Wikitext-2-raw の先頭から `max_chars` 文字相当を、**段落ごとに**トークナイズして連結する。
    一度に長文を `encode` しない（モデル・トークナイザの想定長を超える警告を避ける）。
    ネットワークが必要（初回キャッシュ）。
    """
    from datasets import load_dataset

    mt = max_tokens if max_tokens is not None else min(100_000, max(8192, max_chars // 2))
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    ids_flat: list[int] = []
    n_chars = 0
    for row in ds:
        t = row["text"].strip()
        if not t:
            continue
        if n_chars + len(t) > max_chars:
            remain = max(0, max_chars - n_chars)
            if remain <= 0:
                break
            t = t[:remain]
        ids_flat.extend(tokenizer.encode(t, add_special_tokens=False))
        n_chars += len(t) + 2
        if n_chars >= max_chars or len(ids_flat) >= mt:
            break
    ids_flat = ids_flat[:mt]
    return torch.tensor(ids_flat, dtype=torch.long)


def train_eval_split(
    token_ids_1d: torch.Tensor,
    *,
    eval_frac: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """時系列を保持したまま末尾 eval_frac を評価用に分ける。"""
    n = int(token_ids_1d.numel())
    cut = int(n * (1.0 - eval_frac))
    return token_ids_1d[:cut].clone(), token_ids_1d[cut:].clone()
