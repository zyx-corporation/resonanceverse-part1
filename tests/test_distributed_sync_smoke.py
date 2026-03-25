"""distributed_sync_smoke のスモーク（短ラウンド）。"""

import json
import subprocess
import sys
from pathlib import Path


def test_distributed_sync_smoke_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, "-m", "experiments.distributed_sync_smoke", "--rounds", "5", "--warmup", "1"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    line = r.stdout.strip().splitlines()[-1]
    assert line.startswith("distributed_sync_smoke_ok ")
    payload = json.loads(line.split(" ", 1)[1])
    assert payload.get("schema_version") == "distributed_sync_smoke.v1"
    assert "latency_ms_p50" in payload
