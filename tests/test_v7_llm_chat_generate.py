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
)


class _TokPlain:
    chat_template = None


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
