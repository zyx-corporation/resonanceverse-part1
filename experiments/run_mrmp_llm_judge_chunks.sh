#!/usr/bin/env bash
# MRMP windows.jsonl に対し、6 軸 LLM 審判（v7_phase1a_llm_judge_six_axes.py）を
# CHUNK 行 × N_CHUNKS 回で連続実行し、同一 OUT に追記する。
#
# 前提: OPENAI_API_KEY（リポジトリ直下 .env）。--demo なら API 不要。
#
# 環境変数で上書き可能:
#   SRC   既定: experiments/logs/mrmp_prepared/windows.jsonl
#   OUT   既定: experiments/logs/v7_judge_10k.jsonl
#   CHUNK 既定: 1000
#   N_CHUNKS 既定: 10  （合計 CHUNK×N_CHUNKS 行）
#   SLEEP_AFTER_REQUEST 既定: 0.05
#   LOG   既定: OUT の拡張子手前 + _run.log
#
# 例:
#   bash experiments/run_mrmp_llm_judge_chunks.sh
#   CHUNK=500 N_CHUNKS=4 bash experiments/run_mrmp_llm_judge_chunks.sh
#   bash experiments/run_mrmp_llm_judge_chunks.sh --demo

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SRC="${SRC:-experiments/logs/mrmp_prepared/windows.jsonl}"
OUT="${OUT:-experiments/logs/v7_judge_10k.jsonl}"
CHUNK="${CHUNK:-1000}"
N_CHUNKS="${N_CHUNKS:-10}"
SLEEP="${SLEEP_AFTER_REQUEST:-0.05}"
LOG="${LOG:-${OUT%.jsonl}_run.log}"

DEMO=()
if [[ "${1:-}" == "--demo" ]]; then
  DEMO=(--demo)
  shift
fi

if [[ $# -gt 0 ]]; then
  echo "usage: $0 [--demo]" >&2
  echo "  環境変数 SRC OUT CHUNK N_CHUNKS SLEEP_AFTER_REQUEST LOG を参照" >&2
  exit 1
fi

if [[ ! -f "$SRC" ]]; then
  echo "SRC not found: $SRC" >&2
  echo "先に: python experiments/v7_mrmp_prepare.py" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
mkdir -p "$(dirname "$LOG")"

PY=(python experiments/v7_phase1a_llm_judge_six_axes.py)

{
  echo "=== run_mrmp_llm_judge_chunks start $(date -Iseconds) ==="
  echo "SRC=$SRC"
  echo "OUT=$OUT"
  echo "CHUNK=$CHUNK N_CHUNKS=$N_CHUNKS SLEEP_AFTER_REQUEST=$SLEEP"
  echo "LOG=$LOG"
  echo

  "${PY[@]}" "${DEMO[@]}" --jsonl "$SRC" --offset 0 --max-rows "$CHUNK" \
    --out-jsonl "$OUT" \
    --out-summary "${OUT%.jsonl}_summary_c01.json" \
    --sleep-after-request "$SLEEP"

  for ((i = 2; i <= N_CHUNKS; i++)); do
    printf -v TAG '%02d' "$i"
    echo "=== chunk $i/$N_CHUNKS $(date -Iseconds) lines=$(wc -l <"$OUT") ==="
    "${PY[@]}" "${DEMO[@]}" --jsonl "$SRC" --resume --max-rows "$CHUNK" \
      --out-jsonl "$OUT" \
      --out-summary "${OUT%.jsonl}_summary_c${TAG}.json" \
      --sleep-after-request "$SLEEP"
  done

  echo "=== run_mrmp_llm_judge_chunks done $(date -Iseconds) ==="
  wc -l "$OUT"
} 2>&1 | tee "$LOG"
