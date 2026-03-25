"""
Phase 4 補助: 2 プロセス間の IPC 往復（擬似同期）の壁時計を測るスモーク。

単一ノード上で multiprocessing のオーバーヘッドのオーダーを把握するための最小実装である。
本番の共鳴場分割・Jetson クラスタとは別物だが、Phase 4 ロードマップの「観測の置き場所」のプレースホルダになる。

実行（spawn 環境ではモジュール実行が必須）::

    python -m experiments.distributed_sync_smoke [--rounds 200] [--out path.json]
    python -m experiments.distributed_sync_smoke --variant tensor [--tensor-dim 64] [--out path.json]

`--variant tensor` は **NumPy ベクトル**（実数場の最小統計量のスタブ）を Pipe で pickle 往復し、分散同期のオーダー感を追加する。
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


def _tensor_echo_worker(conn: Any) -> None:
    """子: 受け取ったベクトル（numpy）に小さな変換を加えて返す（擬似場更新）。"""
    while True:
        msg = conn.recv()
        if msg is None:
            break
        t = np.asarray(msg, dtype=np.float64)
        conn.send(t * 0.5 + 0.01)


def _run_tensor_ping_pong(
    *,
    rounds: int,
    warmup: int,
    tensor_dim: int,
) -> dict[str, Any]:
    parent_conn, child_conn = mp.Pipe()
    proc = mp.Process(target=_tensor_echo_worker, args=(child_conn,))
    proc.start()
    times_s: list[float] = []
    try:
        for _ in range(warmup):
            x = np.random.randn(1, tensor_dim).astype(np.float64)
            parent_conn.send(x)
            parent_conn.recv()
        for _ in range(rounds):
            x = np.random.randn(1, tensor_dim).astype(np.float64)
            t0 = time.perf_counter()
            parent_conn.send(x)
            parent_conn.recv()
            times_s.append(time.perf_counter() - t0)
    finally:
        parent_conn.send(None)
        proc.join(timeout=10.0)
        if proc.is_alive():
            proc.terminate()

    ms = np.array(times_s, dtype=np.float64) * 1000.0
    return {
        "schema_version": "distributed_sync_smoke.v1",
        "variant": "multiprocessing_pipe_tensor_pingpong",
        "tensor_dim": tensor_dim,
        "payload": "numpy_ndarray_float64",
        "rounds": rounds,
        "warmup": warmup,
        "latency_ms_p50": float(np.percentile(ms, 50)),
        "latency_ms_p95": float(np.percentile(ms, 95)),
        "latency_ms_mean": float(np.mean(ms)),
        "disclaimer": (
            "単一マシン上の Pipe 往復（numpy ベクトル pickle）。分散共鳴場の実レイテンシではない。"
        ),
    }


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
    p.add_argument(
        "--variant",
        choices=("bytes", "tensor"),
        default="bytes",
        help="bytes=従来の Pipe バイト往復、tensor=torch.Tensor の往復",
    )
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--tensor-dim", type=int, default=64, help="--variant tensor のときのベクトル長")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.variant == "tensor":
        payload = _run_tensor_ping_pong(
            rounds=args.rounds,
            warmup=args.warmup,
            tensor_dim=args.tensor_dim,
        )
    else:
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
