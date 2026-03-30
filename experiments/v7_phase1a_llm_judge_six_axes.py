"""
v7 Phase I-A: 6 軸（方向付き）を JSONL 行に付与する「LLM 審判」パイプライン。

- --demo: API なし。text と id から決定論的な 0–1 疑似スコア（CI・オフライン）。
- --provider openai: OPENAI_API_KEY が必要。注意モデル（gpt2 等）とは別系統を推奨。

大規模向け: ``--offset`` / ``--max-rows`` で入力スライス、``--resume`` で出力行数から
オフセットを合わせて追記（中断再開）。429/5xx は指数バックオフでリトライ。

入力 JSONL は少なくとも ``text`` を含む。MRMP 整形行（``speaker_src``, ``speaker_tgt``）を推奨。

出力: 入力行に trust_ab … history_ba（12 列）と llm_judge_meta を付与した JSONL。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.jsonl_slice import (  # noqa: E402
    count_nonempty_lines,
    iter_jsonl_slice,
)
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


def _openai_scores_once(
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
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
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


def openai_scores_for_row(
    *,
    text: str,
    speaker_a: str,
    speaker_b: str,
    model: str,
    api_key: str,
    temperature: float,
    seed: int,
    max_retries: int = 8,
    base_sleep_s: float = 1.0,
) -> dict[str, float]:
    last_err: BaseException | None = None
    for attempt in range(max_retries):
        try:
            return _openai_scores_once(
                text=text,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                model=model,
                api_key=api_key,
                temperature=temperature,
                seed=seed,
            )
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                wait = min(120.0, base_sleep_s * (2**attempt))
                time.sleep(wait)
                last_err = e
                continue
            raise RuntimeError(
                f"OpenAI HTTP {e.code}: {err_body[:500]}"
            ) from e
        except (json.JSONDecodeError, ValueError, KeyError, OSError) as e:
            if attempt < max_retries - 1:
                wait = min(90.0, base_sleep_s * (2**attempt))
                time.sleep(wait)
                last_err = e
                continue
            raise
    raise RuntimeError(f"OpenAI failed after {max_retries} tries") from last_err


def append_llm_judge_to_row(
    row: dict[str, Any],
    *,
    demo: bool,
    provider: str,
    openai_model: str,
    temperature: float,
    seed: int,
    max_retries: int,
    base_sleep_s: float,
) -> dict[str, Any] | None:
    """1 行分。text が空なら None。環境は呼び出し側で load_repo_dotenv 済み想定。"""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    sha = _prompt_sha256(
        SYSTEM_PROMPT_JA,
        _user_prompt("{text}", "{a}", "{b}"),
    )

    text = str(row.get("text", "")).strip()
    rid = str(row.get("id", ""))
    if not text:
        return None
    sp_a, sp_b = _speakers_from_row(row)

    if demo:
        scores = demo_scores_for_row(rid or text[:80], text)
        mode = "demo_deterministic_hash"
        model_used = None
    elif provider == "openai":
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        scores = openai_scores_for_row(
            text=text,
            speaker_a=sp_a,
            speaker_b=sp_b,
            model=openai_model,
            api_key=api_key,
            temperature=temperature,
            seed=seed,
            max_retries=max_retries,
            base_sleep_s=base_sleep_s,
        )
        mode = "openai_chat"
        model_used = openai_model
    else:
        raise ValueError(f"unknown provider: {provider}")

    merged = {**row, **scores}
    merged["llm_judge_meta"] = {
        "mode": mode,
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "prompt_sha256": sha,
        "model": model_used,
        "provider": provider if not demo else "none",
        "role_ja": "疑似6軸。demo は決定論乱数。人手金標準ではない。",
    }
    return merged


def run_judge(
    *,
    rows: list[dict[str, Any]],
    demo: bool,
    provider: str,
    openai_model: str,
    temperature: float,
    seed: int,
    max_retries: int = 8,
    base_sleep_s: float = 1.0,
) -> dict[str, Any]:
    from experiments.local_env import load_repo_dotenv

    load_repo_dotenv()
    out_rows: list[dict[str, Any]] = []
    sha = _prompt_sha256(
        SYSTEM_PROMPT_JA,
        _user_prompt("{text}", "{a}", "{b}"),
    )
    for row in rows:
        try:
            m = append_llm_judge_to_row(
                row,
                demo=demo,
                provider=provider,
                openai_model=openai_model,
                temperature=temperature,
                seed=seed,
                max_retries=max_retries,
                base_sleep_s=base_sleep_s,
            )
        except RuntimeError as e:
            if "OPENAI_API_KEY" in str(e):
                return {
                    "schema_version": "v7_llm_judge_six_axes.v1",
                    "error": str(e),
                    "skipped": True,
                }
            raise
        if m is not None:
            out_rows.append(m)

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
    p.add_argument(
        "--offset",
        type=int,
        default=0,
        help="入力 JSONL の先頭からスキップする行数（--resume 時は上書き）",
    )
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument(
        "--resume",
        action="store_true",
        help="--out-jsonl の既存行数を offset にし、追記モード",
    )
    p.add_argument(
        "--sleep-after-request",
        type=float,
        default=0.0,
        help="各リクエスト成功後の追加待機秒（レート緩和）",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="429/5xx/JSON 失敗時の最大リトライ回数",
    )
    p.add_argument(
        "--base-sleep",
        type=float,
        default=1.0,
        help="リトライ待機の初期秒（指数バックオフ）",
    )
    args = p.parse_args()

    from experiments.local_env import load_repo_dotenv

    load_repo_dotenv()

    jsonl_path = args.jsonl.resolve()
    if not jsonl_path.is_file():
        print(
            json.dumps({"error": "jsonl_not_found", "path": str(jsonl_path)}),
            file=sys.stderr,
        )
        sys.exit(1)

    offset = max(0, args.offset)
    if args.resume:
        if not args.out_jsonl:
            print(
                json.dumps({"error": "resume_requires_out_jsonl"}),
                file=sys.stderr,
            )
            sys.exit(1)
        outp = args.out_jsonl.resolve()
        if outp.is_file():
            offset = count_nonempty_lines(outp)

    sha = _prompt_sha256(
        SYSTEM_PROMPT_JA,
        _user_prompt("{text}", "{a}", "{b}"),
    )

    n_written = 0
    out_handle = None
    try:
        if args.out_jsonl:
            args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if (args.resume and args.out_jsonl.resolve().is_file()) else "w"
            out_handle = args.out_jsonl.open(mode, encoding="utf-8")

        for row in iter_jsonl_slice(
            jsonl_path,
            offset=offset,
            max_rows=args.max_rows,
        ):
            merged = append_llm_judge_to_row(
                row,
                demo=args.demo,
                provider=args.provider,
                openai_model=args.openai_model,
                temperature=args.temperature,
                seed=args.seed,
                max_retries=args.max_retries,
                base_sleep_s=args.base_sleep,
            )
            if merged is None:
                continue
            if out_handle:
                out_handle.write(json.dumps(merged, ensure_ascii=False) + "\n")
                out_handle.flush()
                n_written += 1
            if args.sleep_after_request > 0 and not args.demo:
                time.sleep(args.sleep_after_request)
    except RuntimeError as e:
        if "OPENAI_API_KEY" in str(e):
            print(
                json.dumps(
                    {
                        "schema_version": "v7_llm_judge_six_axes.v1",
                        "error": str(e),
                        "skipped": True,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    finally:
        if out_handle is not None:
            out_handle.close()

    summary = {
        "schema_version": "v7_llm_judge_six_axes.v1",
        "n_rows_written": n_written,
        "input_offset": offset,
        "max_rows": args.max_rows,
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "prompt_sha256": sha,
        "demo": args.demo,
        "resume": args.resume,
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
            {
                "n_rows_written": n_written,
                "input_offset": offset,
                "demo": args.demo,
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
