"""
v7 Phase I-A: 6 軸（方向付き）を JSONL 行に付与する「LLM 審判」パイプライン。

- --demo: API なし。text と id から決定論的な 0–1 疑似スコア（CI・オフライン）。
- --provider openai: OPENAI_API_KEY が必要。注意モデル（gpt2 等）とは別系統を推奨。
- --provider hf_local: Hugging Face の因果 LM をローカル推論（例: Swallow-7B-Instruct）。MPS/CUDA/CPU は ``core.inference_device``。

大規模向け: ``--offset`` / ``--max-rows`` で入力スライス、``--resume`` で出力行数から
オフセットを合わせて追記（中断再開）。429/5xx は指数バックオフでリトライ。

入力 JSONL は少なくとも ``text`` を含む。MRMP 整形行（``speaker_src``, ``speaker_tgt``）を推奨。

出力: 入力行に trust_ab … history_ba（12 列）と llm_judge_meta を付与した JSONL。

プロンプト本文: ``experiments/prompts/v7_llm_judge_prompt_v1.json``（``prompt_template_id``・SHA256 は出力メタに記録）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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

_DEFAULT_PROMPT_JSON = _ROOT / "experiments" / "prompts" / "v7_llm_judge_prompt_v1.json"


def _load_llm_judge_prompt_bundle(path: Path | None = None) -> tuple[str, str, str]:
    p = path or _DEFAULT_PROMPT_JSON
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema_version") != "v7_llm_judge_prompt_bundle.v1":
        raise ValueError("v7_llm_judge_prompt_bundle: schema_version mismatch")
    tid = str(data["prompt_template_id"])
    sys_p = str(data["system_prompt_ja"])
    ut = str(data["user_prompt_template"])
    return tid, sys_p, ut


PROMPT_TEMPLATE_ID, SYSTEM_PROMPT_JA, _USER_PROMPT_TEMPLATE_STR = _load_llm_judge_prompt_bundle()

DEFAULT_HF_JUDGE_MODEL = "tokyotech-llm/Swallow-7b-instruct-hf"


def _strip_markdown_json_fence(text: str) -> str:
    m = re.search(
        r"```(?:json)?\s*\r?\n(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return text.strip()


def _balanced_json_object_slice(s: str, start: int) -> str | None:
    """``start`` は ``{`` の位置。ダブルクォート文字列内の ``{`` ``}`` を無視して外側のオブジェクトを切り出す。"""
    depth = 0
    i = start
    in_string = False
    escape = False
    n = len(s)
    while i < n:
        c = s[i]
        if escape:
            escape = False
            i += 1
            continue
        if in_string:
            if c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
        i += 1
    return None


def _parse_judge_scores_regex(s: str) -> dict[str, Any] | None:
    """モデルが JSON 外にゴミを付けた場合の最終手段（12 キーを個別に探索）。"""
    out: dict[str, Any] = {}
    num = r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?"
    for k in PILOT_KEYS:
        pat = rf'"{re.escape(k)}"\s*:\s*({num})\b'
        m = re.search(pat, s)
        if not m:
            return None
        out[k] = float(m.group(1))
    return out


def parse_llm_judge_json_response(text: str) -> dict[str, Any]:
    """生成文から JSON オブジェクトを抽出（前後の説明文・フェンス・文字列内 ``}`` に対応）。"""
    raw = text
    s = _strip_markdown_json_fence(text)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    i = s.find("{")
    if i < 0:
        blob = _parse_judge_scores_regex(raw)
        if blob is not None:
            return blob
        raise ValueError("no_json_object_in_model_output")
    slice_s = _balanced_json_object_slice(s, i)
    if slice_s is not None:
        try:
            obj = json.loads(slice_s)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    blob = _parse_judge_scores_regex(raw)
    if blob is not None:
        return blob
    raise ValueError("unparseable_json_in_model_output")


def _load_hf_judge(
    model_id: str,
    *,
    force_cpu: bool,
    revision: str | None = None,
) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from core.inference_device import select_inference_device

    device = select_inference_device(force_cpu=force_cpu)
    extra: dict[str, Any] = {}
    if revision:
        extra["revision"] = revision
    tok = AutoTokenizer.from_pretrained(model_id, **extra)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        **extra,
    ).to(device)
    model.eval()
    return model, tok, device


def hf_local_scores_for_row(
    model: Any,
    tokenizer: Any,
    device: Any,
    *,
    text: str,
    speaker_a: str,
    speaker_b: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> dict[str, float]:
    import torch

    from core.reproducibility import set_experiment_seed

    set_experiment_seed(seed)
    user = _user_prompt(text, speaker_a, speaker_b)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_JA},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(
        tokenizer, "chat_template", None
    ):
        batch = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(batch, "input_ids"):
            input_ids = batch["input_ids"]
        else:
            input_ids = batch
    else:
        merged = SYSTEM_PROMPT_JA + "\n\n" + user + "\n\nJSON のみ:"
        enc = tokenizer(merged, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"]
    input_ids = input_ids.to(device)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    try:
        from transformers import GenerationConfig
    except ImportError:
        GenerationConfig = None  # type: ignore[misc,assignment]
    mnt = max(32, int(max_new_tokens))
    if GenerationConfig is not None:
        if temperature <= 1e-6:
            gen_cfg = GenerationConfig(
                max_new_tokens=mnt,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        else:
            gen_cfg = GenerationConfig(
                max_new_tokens=mnt,
                do_sample=True,
                temperature=float(temperature),
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                generation_config=gen_cfg,
            )
    else:
        gen_kw: dict[str, Any] = {
            "max_new_tokens": mnt,
            "pad_token_id": pad_id,
        }
        if eos_id is not None:
            gen_kw["eos_token_id"] = eos_id
        if temperature <= 1e-6:
            gen_kw["do_sample"] = False
        else:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = float(temperature)
        with torch.no_grad():
            out_ids = model.generate(input_ids, **gen_kw)
    new_tokens = out_ids[0, input_ids.shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    parsed = parse_llm_judge_json_response(raw)
    out: dict[str, float] = {}
    for k in PILOT_KEYS:
        v = parsed.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(max(0.0, min(1.0, float(v))))
        else:
            raise ValueError(f"missing or invalid key in hf output: {k}")
    return out


def _prompt_sha256(system: str, user_template: str) -> str:
    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    h.update(user_template.encode("utf-8"))
    return h.hexdigest()


def _canonical_user_slice_for_hash() -> str:
    return _USER_PROMPT_TEMPLATE_STR.format(
        text="{text}", speaker_a="{a}", speaker_b="{b}"
    )


def _user_prompt(text: str, speaker_a: str, speaker_b: str) -> str:
    return _USER_PROMPT_TEMPLATE_STR.format(
        text=text, speaker_a=speaker_a, speaker_b=speaker_b
    )


def judge_prompt_fingerprint_sha256() -> str:
    """審判プロンプト（system + canonical user スライス）の SHA256。事前登録・テスト用。"""
    return _prompt_sha256(SYSTEM_PROMPT_JA, _canonical_user_slice_for_hash())


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


def _openai_api_error_code(err_body: str) -> str | None:
    try:
        j = json.loads(err_body)
        err = j.get("error")
        if isinstance(err, dict):
            c = err.get("code")
            if c is not None:
                return str(c)
            t = err.get("type")
            if t is not None:
                return str(t)
    except (json.JSONDecodeError, TypeError):
        pass
    return None


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
            api_code = _openai_api_error_code(err_body)
            if api_code == "insufficient_quota":
                raise RuntimeError(
                    "OpenAI insufficient_quota: 課金・利用上限に達しています。"
                    " https://platform.openai.com でプラン・残高・利用制限を確認してください。"
                    f" 応答抜粋: {err_body[:400]}"
                ) from e
            retriable = e.code in (429, 500, 502, 503, 504)
            if retriable and attempt < max_retries - 1:
                wait = min(120.0, base_sleep_s * (2**attempt))
                if e.code == 429 and getattr(e, "headers", None):
                    ra = e.headers.get("Retry-After")
                    if ra is not None:
                        try:
                            wait = max(wait, float(ra))
                        except ValueError:
                            pass
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
    hf_bundle: tuple[Any, Any, Any] | None = None,
    hf_model_id: str = DEFAULT_HF_JUDGE_MODEL,
    hf_max_new_tokens: int = 256,
    hf_revision: str | None = None,
) -> dict[str, Any] | None:
    """1 行分。text が空なら None。環境は呼び出し側で load_repo_dotenv 済み想定。"""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    sha = _prompt_sha256(SYSTEM_PROMPT_JA, _canonical_user_slice_for_hash())

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
    elif provider == "hf_local":
        if hf_bundle is None:
            raise ValueError("hf_bundle required for hf_local")
        model, tokenizer, device = hf_bundle
        last_err: BaseException | None = None
        scores = None
        for attempt in range(max_retries):
            try:
                scores = hf_local_scores_for_row(
                    model,
                    tokenizer,
                    device,
                    text=text,
                    speaker_a=sp_a,
                    speaker_b=sp_b,
                    max_new_tokens=hf_max_new_tokens,
                    temperature=temperature,
                    seed=seed + attempt * 17,
                )
                break
            except (ValueError, json.JSONDecodeError, RuntimeError, OSError) as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(min(90.0, base_sleep_s * (2**attempt)))
                    continue
                raise RuntimeError(f"hf_local judge failed after {max_retries} tries") from e
        if scores is None:
            raise RuntimeError(f"hf_local judge failed: {last_err!r}")
        mode = "hf_causal_generate"
        model_used = hf_model_id
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
    if provider == "hf_local":
        merged["llm_judge_meta"]["hf_revision"] = hf_revision
        merged["llm_judge_meta"]["hf_max_new_tokens"] = hf_max_new_tokens
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
    hf_model_id: str = DEFAULT_HF_JUDGE_MODEL,
    hf_max_new_tokens: int = 256,
    hf_revision: str | None = None,
    hf_cpu: bool = False,
) -> dict[str, Any]:
    from experiments.local_env import load_repo_dotenv

    load_repo_dotenv()
    out_rows: list[dict[str, Any]] = []
    sha = _prompt_sha256(SYSTEM_PROMPT_JA, _canonical_user_slice_for_hash())
    hf_bundle = None
    if not demo and provider == "hf_local":
        hf_bundle = _load_hf_judge(
            hf_model_id,
            force_cpu=hf_cpu,
            revision=hf_revision,
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
                hf_bundle=hf_bundle,
                hf_model_id=hf_model_id,
                hf_max_new_tokens=hf_max_new_tokens,
                hf_revision=hf_revision,
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
    p = argparse.ArgumentParser(description="v7 6軸 LLM審判（demo / OpenAI / HF ローカル）")
    p.add_argument(
        "--jsonl",
        type=Path,
        default=_ROOT / "experiments" / "data" / "v7_mrmp_sample.jsonl",
    )
    p.add_argument("--out-jsonl", type=Path, default=None)
    p.add_argument("--out-summary", type=Path, default=None)
    p.add_argument("--demo", action="store_true", help="API なしの疑似スコア")
    p.add_argument(
        "--provider",
        default="openai",
        choices=("openai", "hf_local"),
        help="openai または hf_local（Swallow 等・ローカル GPU/MPS）",
    )
    p.add_argument("--openai-model", default="gpt-4o-mini")
    p.add_argument(
        "--hf-model",
        default=DEFAULT_HF_JUDGE_MODEL,
        help="hf_local 時の Hugging Face モデル ID",
    )
    p.add_argument("--hf-max-new-tokens", type=int, default=256)
    p.add_argument(
        "--hf-revision",
        type=str,
        default=None,
        help="HF の revision（コミット SHA 等）を固定する場合",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="hf_local 時に MPS/CUDA を使わず CPU（メモリ多い Mac 向け）",
    )
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

    sha = _prompt_sha256(SYSTEM_PROMPT_JA, _canonical_user_slice_for_hash())

    hf_bundle = None
    if not args.demo and args.provider == "hf_local":
        hf_bundle = _load_hf_judge(
            args.hf_model,
            force_cpu=args.cpu,
            revision=args.hf_revision,
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
                hf_bundle=hf_bundle,
                hf_model_id=args.hf_model,
                hf_max_new_tokens=args.hf_max_new_tokens,
                hf_revision=args.hf_revision,
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
        "provider": args.provider,
        "hf_model": args.hf_model if args.provider == "hf_local" else None,
        "hf_revision": args.hf_revision,
        "cpu_forced": bool(args.cpu),
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
