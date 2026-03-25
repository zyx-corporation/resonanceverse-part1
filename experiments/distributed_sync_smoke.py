"""
Phase 4 補助: 2 プロセス間の IPC 往復（擬似同期）の壁時計を測るスモーク。

単一ノード上で multiprocessing のオーバーヘッドのオーダーを把握するための最小実装である。
本番の共鳴場分割・Jetson クラスタとは別物だが、Phase 4 ロードマップの「観測の置き場所」のプレースホルダになる。

実行（spawn 環境ではモジュール実行が必須）::

    python -m experiments.distributed_sync_smoke [--rounds 200] [--out path.json]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _echo_worker(conn: Any) -> None:
    while True:
        msg = conn.recv()
        if msg is None:
            break
        conn.send(msg)


def _run_ping_pong(
    *,
    rounds: int,
    warmup: int,
) -> dict[str, Any]:
    parent_conn, child_conn = mp.Pipe()
    proc = mp.Process(target=_echo_worker, args=(child_conn,))
    proc.start()
    times_s: list[float] = []
    try:
        for _ in range(warmup):
            parent_conn.send(b"p")
            parent_conn.recv()
        for _ in range(rounds):
            t0 = time.perf_counter()
            parent_conn.send(b"p")
            parent_conn.recv()
            times_s.append(time.perf_counter() - t0)
    finally:
        parent_conn.send(None)
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()

    ms = np.array(times_s, dtype=np.float64) * 1000.0
    return {
        "schema_version": "distributed_sync_smoke.v1",
        "variant": "multiprocessing_pipe_pingpong",
        "rounds": rounds,
        "warmup": warmup,
        "latency_ms_p50": float(np.percentile(ms, 50)),
        "latency_ms_p95": float(np.percentile(ms, 95)),
        "latency_ms_mean": float(np.mean(ms)),
        "disclaimer": (
            "単一マシン上の Pipe 往復。分散共鳴場の実レイテンシではない。"
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 4: 2-process IPC sync smoke")
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    payload = _run_ping_pong(rounds=args.rounds, warmup=args.warmup)
    print("distributed_sync_smoke_ok", json.dumps(payload, ensure_ascii=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
