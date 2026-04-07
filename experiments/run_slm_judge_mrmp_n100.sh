#!/usr/bin/env bash
# MRMP 整形済み windows.jsonl の先頭 100 行で、7B×2 + 3B×1 審判と pair_agreement 3 本を実行する。
#
# 前提: python experiments/v7_mrmp_prepare.py 済みで
#       experiments/logs/mrmp_prepared/windows.jsonl が存在し、非空行が 100 以上あること。
#
# 環境変数: JSONL / OFFSET / OUT_DIR / EXTRA（python への追加引数、例: EXTRA="--demo"）
#
# 例:
#   bash experiments/run_slm_judge_mrmp_n100.sh
#   EXTRA="--demo" bash experiments/run_slm_judge_mrmp_n100.sh   # HF なしスモーク（要 100 行以上）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

JSONL="${JSONL:-experiments/logs/mrmp_prepared/windows.jsonl}"
OFFSET="${OFFSET:-0}"
N="${N:-100}"
OUT_DIR="${OUT_DIR:-experiments/logs/slm_judge_triple_mrmp_n100}"

exec python3 experiments/v7_slm_judge_triple_run.py \
  --jsonl "$JSONL" \
  --n "$N" \
  --offset "$OFFSET" \
  --out-dir "$OUT_DIR" \
  --temperature 0 \
  ${EXTRA:-}
