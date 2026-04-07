"""tools.v7_pair_chat_engine の軽量テスト（HF なし）。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.v7_pair_chat_engine import mrmp_window_text


def test_mrmp_window_text():
    w = mrmp_window_text([("A", "hi"), ("B", "there")])
    assert w == "A: hi\nB: there"
