"""jsonl_slice の行スキップ。"""

import json
from pathlib import Path

from experiments.jsonl_slice import count_nonempty_lines, iter_jsonl_slice


def test_iter_jsonl_slice_offset_and_max(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text(
        "\n".join(json.dumps({"i": i}) for i in range(5)) + "\n",
        encoding="utf-8",
    )
    got = list(iter_jsonl_slice(p, offset=2, max_rows=2))
    assert [r["i"] for r in got] == [2, 3]


def test_count_nonempty_lines(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text("{}\n\n{}\n", encoding="utf-8")
    assert count_nonempty_lines(p) == 2
