#!/usr/bin/env bash
# Swallow-7B-Instruct で windows.jsonl 先頭 n 行を hf_local 審判し、/usr/bin/time -p で壁時計を記録する。
#
# 環境変数: N（既定 500）, OFFSET（既定 0）, JSONL, OUT_PREFIX
#
# 例:
#   bash experiments/run_judge_swallow_n500_walltime.sh
#   N=200 JSONL=... bash experiments/run_judge_swallow_n500_walltime.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

N="${N:-500}"
OFFSET="${OFFSET:-0}"
JSONL="${JSONL:-experiments/logs/mrmp_prepared/windows.jsonl}"
OUT_PREFIX="${OUT_PREFIX:-experiments/logs/judge_swallow_n${N}_walltime}"
LOG="${OUT_PREFIX}_time.log"

mkdir -p "$(dirname "$OUT_PREFIX")"

{
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "n=$N offset=$OFFSET jsonl=$JSONL"
} | tee -a "$LOG"

/usr/bin/time -p python3 experiments/v7_phase1a_llm_judge_six_axes.py \
  --jsonl "$JSONL" \
  --offset "$OFFSET" \
  --max-rows "$N" \
  --provider hf_local \
  --hf-model tokyotech-llm/Swallow-7b-instruct-hf \
  --temperature 0 \
  --out-jsonl "${OUT_PREFIX}.jsonl" \
  --out-summary "${OUT_PREFIX}_summary.json" \
  2>&1 | tee -a "$LOG"

echo "done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
