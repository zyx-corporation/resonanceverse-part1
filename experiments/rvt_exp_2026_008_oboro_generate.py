"""
RVT-EXP-2026-008 L3（最小）: 単語ごとの logit から簡易「固着」指標を取り、温度を切り替える貪欲デコード。

計画書の OboroMonitor 全体の代替ではなく、**再現可能な実験用フック**として
``top-1 確率質量`` が閾値超過のステップ連続時に温度を上げる。

**スコープ（スタンドアロン）**: 本モジュールは **HF 因果 LM を直接ロードしてログを JSON 化する CLI**
にとどめる。**審判・Streamlit・統合チャットの generate 経路とは接続しない**（後段でフックを
足す場合も、当ファイルは実験・バンドル用途の参照実装として残す）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class OboroLiteConfig:
    """最小（ライト）プロファイル用パラメータ。"""

    fixation_threshold: float = 0.92
    stuck_steps: int = 3
    temp_base: float = 1.0
    temp_bump: float = 0.15
    temp_max: float = 1.5


@dataclass
class OboroFullConfig(OboroLiteConfig):
    """
    計画書準拠に近い**観測トレース**を追加する本番プロファイル。

    デコード核はライトと同一（貪欲＋固着→温度）。full ではエントロピー・
    top1-top2 マージン等をログに載せ、再現・図表用 JSON を厚くする。
    """

    # 追加: 分布の尖り（低エントロピー）も「固着」扱いに足し算できる
    entropy_stall_threshold: float = 0.85
    use_entropy_stall: bool = True


def softmax_from_logits(logits_1d: np.ndarray) -> np.ndarray:
    x = logits_1d.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(np.clip(x, -50, 50))
    p = e / (np.sum(e) + 1e-12)
    return p


def entropy_from_logits(logits_1d: np.ndarray) -> float:
    p = softmax_from_logits(logits_1d)
    return float(-np.sum(p * np.log(p + 1e-12)))


def top2_margin_from_logits(logits_1d: np.ndarray) -> float:
    x = np.sort(logits_1d.astype(np.float64))
    if x.size < 2:
        return 0.0
    return float(x[-1] - x[-2])


def fixation_mass_from_logits(logits_1d: np.ndarray) -> float:
    """正規化前 logit 行から最大 softmax 質量。"""
    x = logits_1d.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(np.clip(x, -50, 50))
    p = e / (np.sum(e) + 1e-12)
    return float(np.max(p))


def greedy_decode_with_oboro_lite(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompt: str,
    max_new_tokens: int,
    cfg: OboroLiteConfig | None = None,
    profile: str = "lite",
) -> dict[str, Any]:
    """
    貪欲 1 トークンずつ ``forward``（``past_key_values``）。

    ``profile`` が ``\"full\"`` のとき ``OboroFullConfig`` を推奨し、
    追加トレースを ``rvt_exp_008_oboro.v2`` で返す。
    """
    import torch

    if profile == "full":
        cfg = cfg if isinstance(cfg, OboroFullConfig) else OboroFullConfig()
    else:
        cfg = cfg if cfg is not None else OboroLiteConfig()
        if isinstance(cfg, OboroFullConfig):
            cfg = OboroLiteConfig(
                fixation_threshold=cfg.fixation_threshold,
                stuck_steps=cfg.stuck_steps,
                temp_base=cfg.temp_base,
                temp_bump=cfg.temp_bump,
                temp_max=cfg.temp_max,
            )
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    out_ids: list[int] = []
    fixation_trace: list[float] = []
    temp_trace: list[float] = []
    entropy_trace: list[float] = []
    top2_margin_trace: list[float] = []
    stall_reason_trace: list[str] = []
    stuck = 0
    temperature = float(cfg.temp_base)
    past: Any = None

    model.eval()
    with torch.no_grad():
        for _step in range(int(max_new_tokens)):
            if past is None:
                out = model(
                    input_ids=ids,
                    attention_mask=attn_mask,
                    use_cache=True,
                )
            else:
                out = model(
                    input_ids=ids[:, -1:],
                    attention_mask=attn_mask,
                    past_key_values=past,
                    use_cache=True,
                )
            logits = out.logits[:, -1, :].float()
            logits_np = logits[0].cpu().numpy()
            fm = fixation_mass_from_logits(logits_np)
            fixation_trace.append(fm)
            ent = entropy_from_logits(logits_np)
            tm = top2_margin_from_logits(logits_np)
            entropy_trace.append(ent)
            top2_margin_trace.append(tm)
            past = out.past_key_values

            entr_stall = (
                profile == "full"
                and isinstance(cfg, OboroFullConfig)
                and cfg.use_entropy_stall
                and ent <= cfg.entropy_stall_threshold
            )
            if fm >= cfg.fixation_threshold or entr_stall:
                stuck += 1
                if entr_stall and fm < cfg.fixation_threshold:
                    stall_reason_trace.append("entropy")
                else:
                    stall_reason_trace.append("mass")
            else:
                stuck = 0
                stall_reason_trace.append("none")
            if stuck >= cfg.stuck_steps:
                temperature = min(cfg.temp_max, temperature + cfg.temp_bump)
                stuck = 0
            temp_trace.append(temperature)

            adj = logits / temperature if temperature != 1.0 else logits
            next_id = int(torch.argmax(adj, dim=-1).item())
            out_ids.append(next_id)
            next_col = torch.tensor(
                [[next_id]],
                device=device,
                dtype=ids.dtype,
            )
            ids = torch.cat([ids, next_col], dim=1)
            if attn_mask is not None:
                one = torch.ones(
                    (1, 1),
                    device=device,
                    dtype=attn_mask.dtype,
                )
                attn_mask = torch.cat([attn_mask, one], dim=1)
            eid = tokenizer.eos_token_id
            if eid is not None and next_id == int(eid):
                break

    prefix = enc["input_ids"][0].tolist()
    full = tokenizer.decode(prefix + out_ids, skip_special_tokens=True)
    conf: dict[str, Any] = {
        "fixation_threshold": cfg.fixation_threshold,
        "stuck_steps": cfg.stuck_steps,
        "temp_base": cfg.temp_base,
        "temp_bump": cfg.temp_bump,
        "temp_max": cfg.temp_max,
        "profile": profile,
    }
    if profile == "full" and isinstance(cfg, OboroFullConfig):
        conf["entropy_stall_threshold"] = cfg.entropy_stall_threshold
        conf["use_entropy_stall"] = cfg.use_entropy_stall
    base_out: dict[str, Any] = {
        "text": full,
        "new_token_ids": out_ids,
        "logit_fixation": fixation_trace,
        "temperature_applied": temp_trace,
        "config": conf,
        "standalone_scope_ja": (
            "実験用スタンドアロン CLI 出力。審判・チャット generate とは未接続。"
        ),
    }
    if profile == "full":
        base_out["schema_version"] = "rvt_exp_008_oboro.v2"
        base_out["logit_entropy"] = entropy_trace
        base_out["logit_top2_margin"] = top2_margin_trace
        base_out["stall_reason"] = stall_reason_trace
    else:
        base_out["schema_version"] = "rvt_exp_008_oboro_lite.v1"
    return base_out


def build_oboro_demo_payload(*, profile: str) -> dict[str, Any]:
    """
    HF なしでスキーマ検証・CI 向けの固定トレース（実 forward ではない）。
    """
    conf: dict[str, Any] = {
        "fixation_threshold": 0.92,
        "stuck_steps": 3,
        "temp_base": 1.0,
        "temp_bump": 0.15,
        "temp_max": 1.5,
        "profile": profile,
    }
    fixation = [0.88, 0.93, 0.94, 0.95, 0.50]
    temps = [1.0, 1.0, 1.0, 1.15, 1.15]
    out: dict[str, Any] = {
        "text": "[demo] oboro standalone trace (no HF)",
        "new_token_ids": [101, 102, 103, 104, 105],
        "logit_fixation": fixation,
        "temperature_applied": temps,
        "config": conf,
        "standalone_scope_ja": (
            "build_oboro_demo_payload: 合成デモ。run_oboro_cli --demo と同等の用途。"
        ),
        "demo_synthetic": True,
    }
    if profile == "full":
        conf["entropy_stall_threshold"] = 0.85
        conf["use_entropy_stall"] = True
        out["schema_version"] = "rvt_exp_008_oboro.v2"
        out["logit_entropy"] = [2.1, 0.7, 0.65, 0.6, 2.0]
        out["logit_top2_margin"] = [0.5, 2.0, 2.1, 2.2, 0.3]
        out["stall_reason"] = ["none", "mass", "mass", "entropy", "none"]
    else:
        out["schema_version"] = "rvt_exp_008_oboro_lite.v1"
    return out


def run_oboro_cli(argv: list[str] | None = None) -> int:
    """
    argparse 経由で HF 因果 LM を 1 回ロードし貪欲デコード。
    初回は **Hub 取得**が必要な場合あり。
    """
    import argparse
    import json

    p = argparse.ArgumentParser(
        description=(
            "RVT Oboro 貪欲デコード（HF）— スタンドアロン実験用。--demo で HF なし"
        ),
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="HF なしで固定 JSON を出す（スキーマ検証・CI 向け）",
    )
    p.add_argument("--model", default="gpt2")
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    _d = OboroLiteConfig()
    p.add_argument(
        "--fixation-threshold",
        type=float,
        default=_d.fixation_threshold,
    )
    p.add_argument(
        "--stuck-steps",
        type=int,
        default=_d.stuck_steps,
    )
    p.add_argument("--temp-base", type=float, default=_d.temp_base)
    p.add_argument("--temp-bump", type=float, default=_d.temp_bump)
    p.add_argument("--temp-max", type=float, default=_d.temp_max)
    p.add_argument(
        "--profile",
        choices=("lite", "full"),
        default="lite",
        help="full: エントロピー・top2 マージン・stall 理由を v2 スキーマで出力",
    )
    args = p.parse_args(argv)

    if args.demo:
        payload = build_oboro_demo_payload(profile=str(args.profile))
        js = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(js + "\n", encoding="utf-8")
        print(js)
        return 0

    if args.max_new_tokens < 1:
        err = {"error": "max-new-tokens must be >= 1"}
        print(json.dumps(err, ensure_ascii=False))
        return 2

    from core.reproducibility import set_experiment_seed
    from experiments.rvt_exp_2026_008_mrmp_row import _load_hf

    set_experiment_seed(int(args.seed))
    model, tokenizer, device = _load_hf(args.model, cpu=args.cpu)
    if args.profile == "full":
        cfg = OboroFullConfig(
            fixation_threshold=float(args.fixation_threshold),
            stuck_steps=int(args.stuck_steps),
            temp_base=float(args.temp_base),
            temp_bump=float(args.temp_bump),
            temp_max=float(args.temp_max),
        )
    else:
        cfg = OboroLiteConfig(
            fixation_threshold=float(args.fixation_threshold),
            stuck_steps=int(args.stuck_steps),
            temp_base=float(args.temp_base),
            temp_bump=float(args.temp_bump),
            temp_max=float(args.temp_max),
        )
    payload = greedy_decode_with_oboro_lite(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=str(args.prompt),
        max_new_tokens=int(args.max_new_tokens),
        cfg=cfg,
        profile=str(args.profile),
    )
    js = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js + "\n", encoding="utf-8")
    print(js)
    return 0


def main() -> None:
    raise SystemExit(run_oboro_cli())


if __name__ == "__main__":
    main()
