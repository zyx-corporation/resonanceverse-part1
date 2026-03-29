"""
Phase I-A・人手なし経路: 代理変数（トークン数）と最終層 ||S_asym||_F の相関。

6 軸の人手アノテの代替ではない。パイプラインと「長さ—注意非対称」の粗い関係のログ用。
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

# 固定短文（創作・1 行ずつ）。外部コーパス不要。
_BUILTIN_TEXTS: tuple[str, ...] = (
    "A: 金曜に寄せられない？ B: 木曜までに上を通す必要があって。",
    "A: そのリンク昨日のまま？ B: 更新した。見えない？",
    "A: 納期厳しめです。御社の線を教えてください。 B: 具体的にどの程度でしょう。",
    "A: 口うるさかったかしら。 B: うるさい。ありがとう。",
    "A: リスクを説明した上で決めます。 B: 怖いですけど聞きたいです。",
    "A: 本当にそれ言う？ B: 言うよ。本気だから。",
    "A: Hello. B: Hi there.",
    "A: 検討する。 B: わかりました、待ちます。",
    "The quarterly report needs revision before the board meeting.",
    "I cannot agree to that deadline without checking upstream.",
    "Thank you for waiting; the results are preliminary.",
    "Could you clarify whether the earlier link still applies?",
    "We should align on scope before discussing the timeline.",
    "Short reply: OK.",
    "A: 了解。 B: では明日。",
    "Negotiation stalled; both sides need internal approval.",
    "The patient asked for a plain-language summary of risks.",
    "Family dinner was quiet; nobody mentioned the letter.",
    "Merge conflict in branch feature-x; please review.",
    "A: 遅れてすみません。 B: 大丈夫、こちらも詰まってた。",
)


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3 or y.size < 3:
        return None
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    return None if r != r else r


def run_autoproxy(
    *,
    texts: list[str],
    demo: bool,
    model_name: str,
    cpu: bool,
    seed: int,
) -> dict[str, Any]:
    from experiments.v7_phase1a_phi_correlation import (
        extract_hf_attention_layer_stats,
        frobenius_s_asym_demo_from_text,
    )

    try:
        from transformers import AutoTokenizer
    except ImportError:
        AutoTokenizer = None  # type: ignore[misc,assignment]

    tok = None
    if not demo and AutoTokenizer is not None:
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    proxies: list[float] = []
    fros: list[float] = []
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if tok is not None:
            ntok = len(tok.encode(t, add_special_tokens=True))
        else:
            ntok = len(t)
        proxies.append(float(ntok))
        if demo:
            fros.append(frobenius_s_asym_demo_from_text(t))
        else:
            stats, err, _ = extract_hf_attention_layer_stats(
                text=t,
                model_name=model_name,
                cpu=cpu,
                seed=seed,
            )
            if err is not None:
                return {**err, "autoproxy_partial": True}
            assert stats is not None
            fros.append(float(stats[-1]["frobenius_S_asym"]))

    px = np.array(proxies, dtype=np.float64)
    fy = np.array(fros, dtype=np.float64)
    return {
        "schema_version": "v7_phase1a_autoproxy.v1",
        "mode": "demo_fro" if demo else "hf_last_layer_fro",
        "model": model_name if not demo else None,
        "proxy_name": "num_tokens_gpt2" if tok is not None else "char_length",
        "disclaimer": "人手の 6 軸ではない。長さ代理と S_asym の相関のみ。",
        "n_rows": len(fros),
        "pearson_proxy_vs_frobenius_S_asym": _pearson(px, fy),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="v7 Phase I-A: 人手なし・代理変数相関")
    p.add_argument("--demo", action="store_true", help="合成 Fro、文字長代理")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    payload = run_autoproxy(
        texts=list(_BUILTIN_TEXTS),
        demo=args.demo,
        model_name=args.model,
        cpu=args.cpu,
        seed=args.seed,
    )
    js = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js, encoding="utf-8")
    print(
        "v7_phase1a_autoproxy_ok",
        json.dumps({"n_rows": payload.get("n_rows")}, ensure_ascii=False),
    )


if __name__ == "__main__":
    main()
