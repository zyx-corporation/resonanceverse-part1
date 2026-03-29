"""
MRMP（公式 clone 済み）を v7 実証実験用に整形する。

出力（既定: experiments/logs/mrmp_prepared/）:
  - manifest.json — 件数・スキーマ・ペア規則（v7_corpus_MRMP.md の P1=直前話者）
  - windows.jsonl — 各発話について直近 W ターンを話者付きで連結した text（HF 前向き用）
  - dialogue_eval.jsonl — 対話単位・話者別の MRMP 評価（1–5 を 0–1 に正規化）

生 MRMP は experiments/logs/mrmp_repo/ に無い場合は fetch_mrmp_corpus.py を先に実行する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SCHEMA_VERSION = "v7_mrmp_prepared.v1"
PAIR_RULE = "P1_prev_speaker"

MRMP_EVAL_KEYS = (
    "informativeness",
    "comprehension",
    "familiarity",
    "interest",
    "proactiveness",
    "satisfaction",
)


def _default_corpus_root() -> Path:
    return (
        _ROOT
        / "experiments"
        / "logs"
        / "mrmp_repo"
        / "multi_relational_multi_party_chat_corpus"
    )


def _iter_dialogue_files(dialogues_dir: Path) -> list[Path]:
    if not dialogues_dir.is_dir():
        return []
    out: list[Path] = []
    for sub in sorted(dialogues_dir.iterdir()):
        if sub.is_dir():
            out.extend(sorted(sub.glob("*.json")))
    return out


def _load_dialogue(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _eval_map(dialogue: dict[str, Any]) -> dict[str, dict[str, int]]:
    m: dict[str, dict[str, int]] = {}
    for block in dialogue.get("evaluations") or []:
        sid = str(block.get("interlocutor_id", ""))
        if sid:
            m[sid] = {
                k: int(block[k]) for k in MRMP_EVAL_KEYS if k in block
            }
    return m


def _window_lines(
    utterances: list[dict[str, Any]], end_idx: int, window: int
) -> str:
    start = max(0, end_idx - window + 1)
    lines: list[str] = []
    for j in range(start, end_idx + 1):
        u = utterances[j]
        sp = str(u.get("interlocutor_id", ""))
        tx = str(u.get("text", "")).strip()
        lines.append(f"{sp}: {tx}")
    return "\n".join(lines)


def prepare_mrmp(
    *,
    corpus_root: Path,
    out_dir: Path,
    window: int,
    max_dialogues: int | None,
) -> dict[str, Any]:
    dialogues_dir = corpus_root / "dialogues"
    files = _iter_dialogue_files(dialogues_dir)
    if max_dialogues is not None:
        files = files[: max_dialogues]

    if not files:
        return {
            "schema_version": SCHEMA_VERSION,
            "error": "no_dialogue_json",
            "hint": "Run: python experiments/fetch_mrmp_corpus.py",
            "dialogues_dir": str(dialogues_dir),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    win_path = out_dir / "windows.jsonl"
    dev_path = out_dir / "dialogue_eval.jsonl"

    n_utt = 0
    n_dialogues = 0

    with win_path.open("w", encoding="utf-8") as fw, dev_path.open(
        "w", encoding="utf-8"
    ) as fd:
        for fp in files:
            d = _load_dialogue(fp)
            did = str(d.get("dialogue_id", fp.stem))
            dtype = str(d.get("dialogue_type", ""))
            utterances = list(d.get("utterances") or [])
            if not utterances:
                continue
            n_dialogues += 1
            ev_global = _eval_map(d)

            for row_eval in d.get("evaluations") or []:
                sid = str(row_eval.get("interlocutor_id", ""))
                if not sid:
                    continue
                flat: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "dialogue_id": did,
                    "dialogue_type": dtype,
                    "interlocutor_id": sid,
                }
                for k in MRMP_EVAL_KEYS:
                    if k in row_eval:
                        flat[f"mrmp_{k}_01"] = (int(row_eval[k]) - 1) / 4.0
                fd.write(json.dumps(flat, ensure_ascii=False) + "\n")

            for i, u in enumerate(utterances):
                sp = str(u.get("interlocutor_id", ""))
                uid = int(u.get("utterance_id", i))
                mentions = u.get("mention_to") or []
                prev = utterances[i - 1] if i > 0 else None
                speaker_src = str(prev["interlocutor_id"]) if prev else None

                ev_sp = ev_global.get(sp, {})
                row: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "id": f"{did}_u{uid:05d}",
                    "dialogue_id": did,
                    "dialogue_type": dtype,
                    "utterance_id": uid,
                    "pair_rule": PAIR_RULE,
                    "speaker": sp,
                    "speaker_src": speaker_src,
                    "speaker_tgt": sp,
                    "mention_to": (
                        list(mentions) if isinstance(mentions, list) else []
                    ),
                    "window_turns": window,
                    "text": _window_lines(utterances, i, window),
                }
                for k in MRMP_EVAL_KEYS:
                    if k in ev_sp:
                        row[f"mrmp_{k}_01"] = (int(ev_sp[k]) - 1) / 4.0

                fw.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_utt += 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pair_rule": PAIR_RULE,
        "window_turns": window,
        "corpus_root": str(corpus_root.resolve()),
        "n_dialogues_written": n_dialogues,
        "n_utterance_rows": n_utt,
        "outputs": {
            "windows_jsonl": str(win_path.resolve()),
            "dialogue_eval_jsonl": str(dev_path.resolve()),
        },
        "note_ja": (
            "mrmp_*_01 は MRMP 対話スコアを 0–1 に正規化したもの"
            "（v7 の 6 軸ではない）。"
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="MRMP → v7 実証用 JSONL")
    p.add_argument(
        "--corpus-root",
        type=Path,
        default=_default_corpus_root(),
        help="multi_relational_multi_party_chat_corpus ディレクトリ",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "experiments" / "logs" / "mrmp_prepared",
    )
    p.add_argument("--window", type=int, default=4, help="直近 W 発話を text に連結")
    p.add_argument(
        "--max-dialogues",
        type=int,
        default=None,
        help="先頭 N 対話だけ（デバッグ用）",
    )
    args = p.parse_args()

    root = args.corpus_root.resolve()
    if not (root / "dialogues").is_dir():
        print(
            json.dumps(
                {
                    "error": "corpus_not_found",
                    "path": str(root),
                    "hint": "python experiments/fetch_mrmp_corpus.py",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    m = prepare_mrmp(
        corpus_root=root,
        out_dir=args.out_dir.resolve(),
        window=max(1, args.window),
        max_dialogues=args.max_dialogues,
    )
    if m.get("error"):
        print(json.dumps(m, ensure_ascii=False))
        sys.exit(1)
    print(
        "v7_mrmp_prepare_ok",
        json.dumps(
            {"n_utterance_rows": m.get("n_utterance_rows")},
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
