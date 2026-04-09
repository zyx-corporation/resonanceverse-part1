"""tools.v7_llm_chat_generate のプロンプト組み立てテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.v7_llm_chat_generate import (
    _truncate_simulated_turns,
    build_generation_prompt,
    inject_system_summary,
)


class _TokPlain:
    chat_template = None


def test_inject_system_summary_prepends():
    m = [{"role": "user", "content": "a"}]
    out = inject_system_summary(m, "  メモ  ")
    assert len(out) == 2
    assert out[0] == {"role": "system", "content": "メモ"}
    assert out[1] == m[0]


def test_inject_system_summary_empty_unchanged():
    m = [{"role": "user", "content": "x"}]
    assert inject_system_summary(m, "") == m
    assert inject_system_summary(m, "  \n  ") == m


def test_build_generation_prompt_plain_with_system():
    tok = _TokPlain()
    messages = [
        {"role": "system", "content": "常に日本語で答える。"},
        {"role": "user", "content": "hello"},
    ]
    p = build_generation_prompt(messages, tok)
    assert "System: 常に日本語で答える。" in p
    assert "User: hello" in p
    assert p.rstrip().endswith("Assistant:")


def test_build_generation_prompt_plain():
    tok = _TokPlain()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "bye"},
    ]
    p = build_generation_prompt(messages, tok)
    assert "User: hello" in p
    assert "Assistant: hi" in p
    assert "User: bye" in p
    assert p.rstrip().endswith("Assistant:")


def test_truncate_simulated_turns():
    s = "こんにちは。\nUser: 次の質問"
    assert _truncate_simulated_turns(s) == "こんにちは。"
    s2 = "a\nAssistant: b"
    assert _truncate_simulated_turns(s2) == "a"
