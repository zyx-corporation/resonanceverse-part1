"""v7_pair_chat_engine の窓スライス・チャット変換。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.v7_pair_chat_engine import (
    chat_messages_to_mrmp_turns,
    slice_turns_for_pair_step,
)


def test_slice_cumulative():
    turns = [("A", "a"), ("B", "b"), ("C", "c")]
    s, start = slice_turns_for_pair_step(
        turns, 2, window_mode="cumulative", last_n=8
    )
    assert start == 0
    assert s == turns


def test_slice_last_n():
    turns = [("A", str(i)) for i in range(10)]
    s, start = slice_turns_for_pair_step(
        turns, 9, window_mode="last_n", last_n=4
    )
    assert start == 6
    assert len(s) == 4
    assert s[-1] == ("A", "9")


def test_slice_last_n_minimum_n():
    turns = [("A", "0"), ("B", "1"), ("C", "2")]
    s, _ = slice_turns_for_pair_step(
        turns, 2, window_mode="last_n", last_n=1
    )
    assert len(s) >= 2


def test_slice_bad_mode():
    with pytest.raises(ValueError):
        slice_turns_for_pair_step(
            [("A", "a")], 1, window_mode="x", last_n=3
        )


def test_chat_messages_to_mrmp_turns():
    m = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    t = chat_messages_to_mrmp_turns(m)
    assert t == [("User", "hi"), ("Assistant", "yo")]
