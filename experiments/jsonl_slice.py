"""JSONL の行スライス（大規模ファイル向け・全件をメモリに載せない）。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_jsonl_slice(
    path: Path,
    offset: int,
    max_rows: int | None,
) -> Iterator[dict[str, Any]]:
    """先頭 ``offset`` 行をスキップし、最大 ``max_rows`` 行を yield。"""
    skipped = 0
    taken = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if skipped < offset:
                skipped += 1
                continue
            if max_rows is not None and taken >= max_rows:
                return
            yield json.loads(s)
            taken += 1


def count_nonempty_lines(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n
