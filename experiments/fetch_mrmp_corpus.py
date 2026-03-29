"""
MRMP（Multi-Relational Multi-Party Chat Corpus）の全文を公式 GitHub から取得する。

Hugging Face の `datasets.load_dataset` は、環境によってローディングスクリプト非対応で
失敗することがあるため、**リポジトリの JSON 一式**を浅い git clone で取得する（推奨）。

保存先は既定で `experiments/logs/mrmp_repo/`（`.gitignore` 対象・git に含めない）。

  python experiments/fetch_mrmp_corpus.py
  python experiments/fetch_mrmp_corpus.py --force
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

GIT_URL = "https://github.com/nu-dialogue/multi-relational-multi-party-chat-corpus.git"
CORPUS_SUBDIR = "multi_relational_multi_party_chat_corpus"


def _count_dialogue_json(dialogues_dir: Path) -> int:
    if not dialogues_dir.is_dir():
        return 0
    return sum(1 for p in dialogues_dir.rglob("*.json"))


def main() -> None:
    p = argparse.ArgumentParser(description="MRMP 全文取得（公式 GitHub の浅い clone）")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "experiments" / "logs" / "mrmp_repo",
        help="クローン先ディレクトリ",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="既存の out-dir を削除して取り直す",
    )
    args = p.parse_args()

    out = args.out_dir.resolve()
    corpus_root = out / CORPUS_SUBDIR
    dialogues_dir = corpus_root / "dialogues"

    if out.exists() and args.force:
        shutil.rmtree(out)

    if out.exists() and (out / ".git").is_dir() and corpus_root.is_dir():
        print("fetch_mrmp_corpus: already present, skipping clone:", out)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            shutil.rmtree(out)
        subprocess.run(
            ["git", "clone", "--depth", "1", GIT_URL, str(out)],
            check=True,
        )

    rev = ""
    try:
        r = subprocess.run(
            ["git", "-C", str(out), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        rev = r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    ver_path = out / "VERSION"
    version = ver_path.read_text(encoding="utf-8").strip() if ver_path.is_file() else ""

    n_dialogues = _count_dialogue_json(dialogues_dir)
    meta = {
        "source": "github_shallow_clone",
        "git_url": GIT_URL,
        "git_rev": rev or None,
        "corpus_version_file": version or None,
        "path": str(corpus_root),
        "dialogue_json_files": n_dialogues,
        "interlocutors_json": str(corpus_root / "interlocutors.json"),
        "license_note": "CC BY-SA 4.0 — 再配布時はライセンスに従うこと",
    }
    meta_path = out / "fetch_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        "fetch_mrmp_corpus_ok",
        json.dumps(
            {"out_dir": str(out), "dialogue_json_files": n_dialogues, "meta": str(meta_path)},
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
