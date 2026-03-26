"""
v7 Phase III 向け: 「あわい」測度 Ω の合成軌跡（人手アノテ無し）。

定義4.1 に基づき、簡略化した Ω を時系列 w_ij, w_ji に対して計算し分布を返す。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def omega_awai(
    w_ij: np.ndarray,
    w_ji_delayed: np.ndarray,
    dw_ij: np.ndarray,
    dw_ji: np.ndarray,
) -> np.ndarray:
    """ベクトル各時刻で Ω を計算（v7 §4.3 の簡略版）。"""
    # ||ẇ_ij||
    v1 = np.linalg.norm(dw_ij, axis=-1)
    # 1 - cos sim
    num = np.sum(w_ij * w_ji_delayed, axis=-1)
    den = np.linalg.norm(w_ij, axis=-1) * np.linalg.norm(w_ji_delayed, axis=-1) + 1e-9
    cos_t = np.clip(num / den, -1.0, 1.0)
    v2 = 1.0 - cos_t
    v3 = _sigmoid(np.linalg.norm(w_ij - w_ji_delayed, axis=-1))
    return v1 * v2 * v3


def run_demo(*, seed: int, T: int, d: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    # 滑らかな疑似軌跡
    t = np.arange(T, dtype=np.float64)
    w_ij = np.stack(
        [np.sin(0.1 * t + rng.uniform(0, 2)) for _ in range(d)], axis=-1
    )
    w_ji_del = np.stack(
        [np.cos(0.12 * t + rng.uniform(0, 2)) for _ in range(d)], axis=-1
    )
    dw_ij = np.diff(np.vstack([w_ij[:1], w_ij]), axis=0)
    dw_ji = np.diff(np.vstack([w_ji_del[:1], w_ji_del]), axis=0)
    o = omega_awai(w_ij, w_ji_del, dw_ij, dw_ji)
    return {
        "schema_version": "v7_phase3a.v1",
        "mode": "synthetic_trajectory",
        "T": T,
        "d": d,
        "seed": seed,
        "awai_mean": float(o.mean()),
        "awai_std": float(o.std()),
        "awai_p95": float(np.percentile(o, 95)),
        "note": "人手の間合い評価との相関は別途 Phase III-A。",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="v7: あわい Ω 合成メトリクス")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--d", type=int, default=6)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    payload = run_demo(seed=args.seed, T=args.T, d=args.d)
    js = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js, encoding="utf-8")
    print("v7_phase3a_ok", json.dumps({"awai_mean": payload["awai_mean"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
