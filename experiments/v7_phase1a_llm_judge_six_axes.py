"""
v7 Phase I-A: 6 軸（方向付き）を JSONL 行に付与する「LLM 審判」パイプライン。

- --demo: API なし。text と id から決定論的な 0–1 疑似スコア（CI・オフライン）。
- --provider openai: OPENAI_API_KEY が必要。注意モデル（gpt2 等）とは別系統を推奨。

入力 JSONL は少なくとも ``text`` を含む。MRMP 整形行（``speaker_src``, ``speaker_tgt``）を推奨。

出力: 入力行に trust_ab … history_ba（12 列）と llm_judge_meta を付与した JSONL。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.v7_phase1a_pilot_jsonl import PILOT_KEYS  # noqa: E402

PROMPT_TEMPLATE_ID = "v7_llm_judge_prompt_v1"


def _prompt_sha256(system: str, user_template: str) -> str:
    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    h.update(user_template.encode("utf-8"))
    return h.hexdigest()


SYSTEM_PROMPT_JA = """あなたは対話における人間関係の分析者である。与えられた対話断片について、
指定の 6 軸について「話者Aから話者Bへ」「話者Bから話者Aへ」のスコアを 0 以上 1 以下の実数で返す。
出力は JSON オブジェクトのみ。キーは次の 12 個とし、値は数値であること。
trust_ab, trust_ba, authority_ab, authority_ba, proximity_ab, proximity_ba,
intent_ab, intent_ba, affect_ab, affect_ba, history_ab, history_ba

各軸の意味（簡潔）:
- trust: 信頼
- authority: 権威・主導権の非対称
- proximity: 心理的距離の近さ（近いほど高）
- intent: 意図・目的の伝達の非対称
- affect: 感情の伝わり
- history: 過去の共有・文脈の蓄積の参照

話者A・Bのラベルは入力で与えられる。"""


def _user_prompt(text: str, speaker_a: str, speaker_b: str) -> str:
    return f"""話者A（src）: {speaker_a}
話者B（tgt）: {speaker_b}

対話断片（直近のウィンドウ）:
---
{text}
---

上記について 12 キーの JSON のみを返す。"""


def _speakers_from_row(row: dict[str, Any]) -> tuple[str, str]:
    src = row.get("speaker_src")
    tgt = row.get("speaker_tgt")
    if isinstance(src, str) and src.strip():
        a = src.strip()
    else:
        a = "（直前話者なし）"
    if isinstance(tgt, str) and tgt.strip():
        b = tgt.strip()
    else:
        b = "（不明）"
    return a, b


def demo_scores_for_row(row_id: str, text: str) -> dict[str, float]:
    """決定論的 0–1。人手・LLM ではない。"""
    seed = (abs(hash((row_id, text))) % (2**32)) or 1
    rng = np.random.default_rng(seed)
    return {k: float(rng.uniform(0.0, 1.0)) for k in PILOT_KEYS}


def openai_scores_for_row(
    *,
    text: str,
    speaker_a: str,
    speaker_b: str,
    model: str,
    api_key: str,
    temperature: float,
    seed: int,
) -> dict[str, float]:
    user = _user_prompt(text, speaker_a, speaker_b)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_JA},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "seed": seed,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI HTTP {e.code}: {err_body[:500]}"
        ) from e
    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    out: dict[str, float] = {}
    for k in PILOT_KEYS:
        v = parsed.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(max(0.0, min(1.0, float(v))))
        else:
            raise ValueError(f"missing or invalid key: {k}")
    return out


def run_judge(
    *,
    rows: list[dict[str, Any]],
    demo: bool,
    provider: str,
    openai_model: str,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    from experiments.local_env import load_repo_dotenv

    load_repo_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    out_rows: list[dict[str, Any]] = []
    sha = _prompt_sha256(
        SYSTEM_PROMPT_JA,
        _user_prompt("{text}", "{a}", "{b}"),
    )

    for row in rows:
        text = str(row.get("text", "")).strip()
        rid = str(row.get("id", ""))
        if not text:
            continue
        sp_a, sp_b = _speakers_from_row(row)

        if demo:
            scores = demo_scores_for_row(rid or text[:80], text)
            mode = "demo_deterministic_hash"
            model_used = None
        elif provider == "openai":
            if not api_key:
                return {
                    "schema_version": "v7_llm_judge_six_axes.v1",
                    "error": "OPENAI_API_KEY missing",
                    "skipped": True,
                }
            scores = openai_scores_for_row(
                text=text,
                speaker_a=sp_a,
                speaker_b=sp_b,
                model=openai_model,
                api_key=api_key,
                temperature=temperature,
                seed=seed,
            )
            mode = "openai_chat"
            model_used = openai_model
        else:
            return {
                "schema_version": "v7_llm_judge_six_axes.v1",
                "error": f"unknown provider: {provider}",
                "skipped": True,
            }

        merged = {**row, **scores}
        merged["llm_judge_meta"] = {
            "mode": mode,
            "prompt_template_id": PROMPT_TEMPLATE_ID,
            "prompt_sha256": sha,
            "model": model_used,
            "provider": provider if not demo else "none",
            "role_ja": (
                "疑似6軸。demo は決定論乱数。人手金標準ではない。"
            ),
        }
        out_rows.append(merged)

    return {
        "schema_version": "v7_llm_judge_six_axes.v1",
        "n_out_rows": len(out_rows),
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "prompt_sha256": sha,
        "rows": out_rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="v7 6軸 LLM審判（demo / OpenAI）")
    p.add_argument(
        "--jsonl",
        type=Path,
        default=_ROOT / "experiments" / "data" / "v7_mrmp_sample.jsonl",
    )
    p.add_argument("--out-jsonl", type=Path, default=None)
    p.add_argument("--out-summary", type=Path, default=None)
    p.add_argument("--demo", action="store_true", help="API なしの疑似スコア")
    p.add_argument("--provider", default="openai", choices=("openai",))
    p.add_argument("--openai-model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()

    lines = [
        json.loads(x)
        for x in args.jsonl.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if args.max_rows is not None:
        lines = lines[: max(0, args.max_rows)]

    payload = run_judge(
        rows=lines,
        demo=args.demo,
        provider=args.provider,
        openai_model=args.openai_model,
        temperature=args.temperature,
        seed=args.seed,
    )
    if payload.get("error"):
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

    rows = payload["rows"]
    if args.out_jsonl:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_jsonl.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": payload["schema_version"],
        "n_rows": len(rows),
        "prompt_template_id": payload["prompt_template_id"],
        "prompt_sha256": payload["prompt_sha256"],
        "demo": args.demo,
    }
    if args.out_summary:
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(
        "v7_llm_judge_ok",
        json.dumps(
            {"n_rows": len(rows), "demo": args.demo},
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
