"""
Phase 3 実測計画 P0〜P2: two_tier_sweep + hbm_budget_probe（＋任意 squad）を同一メタで束ねる。

- P0: 比較 JSON と HBM JSON、環境メタ（phase3_claim_bundle.v1）
- P2: --with-squad で squad_span（--demo 時は合成のみ）
- P3 は Router の stride モード（decode_benchmark / two_tier_sweep の --router-step-stride）

オフライン: ``--demo --cpu``（CI 可）。実モデルは ``--model gpt2`` 等（初回ネットあり）。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from core.inference_device import select_inference_device

from experiments.hbm_budget_probe import run_probe
from experiments.two_tier_sweep import run_two_tier_sweep


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": None,
    }
    try:
        import transformers

        out["transformers"] = transformers.__version__
    except Exception:
        pass
    return out


def _cuda_meta() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    d0 = torch.device("cuda:0")
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(d0),
    }


def _mps_meta() -> dict[str, Any]:
    if not hasattr(torch.backends, "mps"):
        return {"available": False}
    return {"available": bool(torch.backends.mps.is_available())}


def build_meta(
    *,
    seed: int,
    model: str,
    demo: bool,
    max_new_tokens: int,
    repeats: int,
    seq_len: int,
    batch_size: int,
    router_step_stride: int | None,
    with_squad: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "phase3_claim_meta.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "node": platform.node(),
        },
        "versions": _versions(),
        "cuda": _cuda_meta(),
        "mps": _mps_meta(),
        "experiment": {
            "seed": seed,
            "model": model,
            "demo": demo,
            "max_new_tokens": max_new_tokens,
            "repeats": repeats,
            "hbm_seq_len": seq_len,
            "hbm_batch_size": batch_size,
            "router_step_stride": router_step_stride,
            "with_squad": with_squad,
        },
    }


def run_claim_bundle(
    *,
    seed: int,
    cpu: bool,
    demo: bool,
    model: str,
    max_new_tokens: int,
    warmup: int,
    repeats: int,
    seq_len: int,
    batch_size: int,
    router_step_stride: int | None,
    with_squad: bool,
    squad_demo: bool,
    squad_max_steps: int,
    squad_max_train: int,
    squad_max_eval: int,
) -> dict[str, Any]:
    meta = build_meta(
        seed=seed,
        model=model,
        demo=demo,
        max_new_tokens=max_new_tokens,
        repeats=repeats,
        seq_len=seq_len,
        batch_size=batch_size,
        router_step_stride=router_step_stride,
        with_squad=with_squad,
    )

    sweep = run_two_tier_sweep(
        seed=seed,
        cpu=cpu,
        demo=demo,
        model=model,
        max_new_tokens=max_new_tokens,
        warmup=warmup,
        repeats=repeats,
        router_step_stride=router_step_stride,
    )

    device = select_inference_device(force_cpu=cpu)
    hbm = run_probe(
        demo=demo,
        model_name=model,
        device=device,
        seq_len=seq_len,
        batch_size=batch_size,
        seed=seed,
    )

    squad_payload: dict[str, Any] | None = None
    if with_squad:
        from experiments.squad_span import run_demo, run_squad_hf

        if squad_demo:
            squad_payload = run_demo(device, squad_max_steps, seed)
        else:
            try:
                squad_payload = run_squad_hf(
                    device,
                    "distilbert-base-uncased",
                    squad_max_steps,
                    squad_max_train,
                    squad_max_eval,
                    256,
                    seed,
                )
            except Exception as e:
                squad_payload = {"squad_span_error": str(e), "skipped": True}

    bundle: dict[str, Any] = {
        "schema_version": "phase3_claim_bundle.v1",
        "meta": meta,
        "two_tier_sweep": sweep,
        "hbm_budget": hbm,
    }
    if squad_payload is not None:
        bundle["squad_span"] = squad_payload
    return bundle


def main() -> None:
    p = argparse.ArgumentParser(description="Phase3 P0–P2: claim bundle (sweep + HBM + opt squad)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--demo", action="store_true", help="transformers 不要経路（two_tier / HBM デモ）")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--router-step-stride", type=int, default=None, metavar="N")
    p.add_argument("--with-squad", action="store_true", help="squad_span を同梱")
    p.add_argument(
        "--squad-demo",
        action="store_true",
        help="squad_span を合成ミニモデルのみ（--with-squad 時。既定は実 HF 経路を試す）",
    )
    p.add_argument("--squad-max-steps", type=int, default=50)
    p.add_argument("--squad-max-train-samples", type=int, default=256)
    p.add_argument("--squad-max-eval-samples", type=int, default=128)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    out_dir = args.out_dir or (_ROOT / "experiments" / "logs" / "phase3_claim")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        bundle = run_claim_bundle(
            seed=args.seed,
            cpu=args.cpu,
            demo=args.demo,
            model=args.model,
            max_new_tokens=args.max_new_tokens,
            warmup=args.warmup,
            repeats=args.repeats,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            router_step_stride=args.router_step_stride,
            with_squad=args.with_squad,
            squad_demo=args.squad_demo or args.demo,
            squad_max_steps=args.squad_max_steps,
            squad_max_train=args.squad_max_train_samples,
            squad_max_eval=args.squad_max_eval_samples,
        )
    except Exception as e:
        print("phase3_claim_run_skip:", e)
        sys.exit(0)

    meta_only = bundle["meta"]
    (out_dir / "phase3_claim_meta.json").write_text(
        json.dumps(meta_only, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "phase3_claim_bundle.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("phase3_claim_run_ok", json.dumps({"out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
